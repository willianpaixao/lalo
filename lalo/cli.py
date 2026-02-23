"""
Command-line interface for Lalo.
"""

import logging
import os
import signal
import threading
import time
import warnings
from datetime import UTC
from pathlib import Path
from typing import Any

# Configure PyTorch memory allocator BEFORE importing torch
# This helps avoid "reserved but unallocated" memory issues
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import click

# Suppress transformers informational messages and progress bars
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

# Suppress Hugging Face Hub download progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from lalo import __version__
from lalo.audio_manager import AudioManager, StreamingAudioWriter
from lalo.checkpoint import CheckpointManager
from lalo.config import (
    CHECKPOINT_ENABLED,
    CHECKPOINT_STALE_DAYS,
    DEFAULT_FORMAT,
    DEFAULT_LANGUAGE,
    DEFAULT_SPEAKER,
    MODEL_NAME,
    SPEAKER_INFO,
    SUPPORTED_FORMATS,
    SUPPORTED_SPEAKERS,
)
from lalo.epub_parser import parse_epub
from lalo.exceptions import (
    AudioError,
    CheckpointCorruptedError,
    CheckpointMismatchError,
    EPUBError,
    GPUNotAvailableError,
    InvalidChapterSelectionError,
    LaloError,
    TTSError,
    UnsupportedSpeakerError,
)
from lalo.tts_engine import TTSEngine
from lalo.utils import (
    compute_file_hash,
    format_duration,
    parse_chapter_selection,
)

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="lalo")
def main():
    """
    Lalo - EPUB to Audiobook Converter using Qwen3-TTS

    Convert EPUB files to high-quality audiobooks with multi-language support.
    """


@main.command()
@click.argument("epub_file", type=click.Path(exists=True))
@click.option(
    "--speaker",
    "-s",
    default=DEFAULT_SPEAKER,
    help=f'Speaker voice to use (default: {DEFAULT_SPEAKER}). Use "lalo speakers --list" to see all available speakers.',
)
@click.option(
    "--language",
    "-l",
    default=DEFAULT_LANGUAGE,
    help=f'Language for TTS (default: {DEFAULT_LANGUAGE} for auto-detection). Use "lalo languages --list" to see supported languages.',
)
@click.option(
    "--chapters",
    "-c",
    default="all",
    help='Chapters to convert (e.g., "all", "1-5", "1,3,7"). Default: all',
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help=f"Output file path (default: <book_title>.{DEFAULT_FORMAT})",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(SUPPORTED_FORMATS, case_sensitive=False),
    default=DEFAULT_FORMAT,
    help=f"Output audio format (default: {DEFAULT_FORMAT})",
)
@click.option(
    "--instruct",
    "-i",
    default=None,
    help='Instruction for voice control (e.g., "Speak slowly and clearly")',
)
@click.option(
    "--streaming",
    is_flag=True,
    default=False,
    help="Use streaming mode to save memory (saves each chapter incrementally)",
)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    help="Enable parallel chapter processing for faster conversion (default: auto-detect GPUs)",
)
@click.option(
    "--max-parallel",
    type=int,
    default=None,
    help="Maximum parallel chapters (default: auto-detect based on available GPUs)",
)
@click.option(
    "--no-resume",
    is_flag=True,
    default=False,
    help="Ignore any existing checkpoint and start a fresh conversion",
)
def convert(  # pyright: ignore[reportGeneralTypeIssues]
    epub_file: str,
    speaker: str,
    language: str,
    chapters: str,
    output: str | None,
    format: str,
    instruct: str | None,
    streaming: bool,
    parallel: bool,
    max_parallel: int | None,
    no_resume: bool,
):
    """
    Convert an EPUB file to audiobook.

    Examples:

        lalo convert mybook.epub

        lalo convert mybook.epub --speaker Ryan --chapters 1-5

        lalo convert mybook.epub --language Japanese --speaker Ono_Anna

        lalo convert mybook.epub --output ~/audiobooks/mybook.mp3 --format mp3

        lalo convert mybook.epub --streaming  # Use streaming mode for large books
    """
    # Signal handling for graceful shutdown
    interrupted = {"flag": False, "force_exit": False, "reason": "user"}

    def signal_handler(_signum, _frame):
        """Handle Ctrl+C gracefully."""
        if interrupted["force_exit"]:
            # Second Ctrl+C - force exit immediately
            console.print("\n[red]Force exiting...[/red]")
            raise click.Abort()

        # First Ctrl+C - set flag for graceful shutdown
        interrupted["flag"] = True
        interrupted["force_exit"] = True
        console.print("\n[yellow]⚠ Interrupt received. Saving progress...[/yellow]")
        console.print("[yellow]Press Ctrl+C again to force exit[/yellow]")

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Validate speaker
        if speaker not in SUPPORTED_SPEAKERS:
            console.print(f"[red]Error:[/red] Speaker '{speaker}' not supported.")
            console.print("Use 'lalo speakers --list' to see available speakers.")
            raise click.Abort()

        # Parse EPUB
        console.print(f"\n[cyan]Parsing EPUB:[/cyan] {epub_file}")
        book = parse_epub(epub_file)

        console.print(f"[green]✓[/green] Found: {book.title} by {book.author}")
        console.print(f"[green]✓[/green] Total chapters: {len(book.chapters)}")

        # Parse chapter selection
        try:
            selected_indices = parse_chapter_selection(chapters, len(book.chapters))
        except InvalidChapterSelectionError as e:
            console.print(f"[red]Invalid chapter selection:[/red] {e}")
            console.print(
                "[yellow]Hint:[/yellow] Use formats like 'all', '1-5', '1,3,7', or '1-3,5,7-10'"
            )
            console.print(f"[yellow]Available:[/yellow] Chapters 1-{len(book.chapters)}")
            raise click.Abort() from e

        selected_chapters = [book.chapters[i] for i in selected_indices]
        console.print(f"[green]✓[/green] Selected {len(selected_chapters)} chapter(s)")

        # Determine output path
        if output is None:
            # Use input filename (without extension) + output format
            input_path = Path(epub_file)
            output = f"{input_path.stem}.{format}"

        output_path = Path(output)

        # Checkpoint / Resume
        checkpoint_mgr = CheckpointManager(epub_file, output_path)
        checkpoint_data = None
        resumed = False
        epub_hash = compute_file_hash(epub_file)

        sr = 24000  # Default sample rate for Qwen3-TTS

        if CHECKPOINT_ENABLED and not no_resume:
            try:
                checkpoint_data = checkpoint_mgr.load(epub_hash)
            except CheckpointCorruptedError as e:
                console.print(f"[yellow]Warning:[/yellow] {e}")
                console.print("[yellow]Starting fresh conversion[/yellow]")
                checkpoint_data = None

            if checkpoint_data is not None:
                try:
                    checkpoint_mgr.validate(
                        checkpoint_data,
                        epub_hash=epub_hash,
                        speaker=speaker,
                        language=language,
                        instruct=instruct,
                        format=format,
                    )
                except CheckpointMismatchError as e:
                    console.print(f"[yellow]Warning:[/yellow] {e}")
                    console.print("[yellow]Starting fresh conversion[/yellow]")
                    checkpoint_mgr.cleanup(epub_hash)
                    checkpoint_data = None

            if checkpoint_data is not None:
                # Verify cached audio files are intact
                valid_files = checkpoint_mgr.verify_audio_files(checkpoint_data)
                remaining = checkpoint_mgr.get_remaining_chapters(checkpoint_data)

                if not remaining:
                    console.print("[green]✓[/green] All chapters already completed from checkpoint")
                    console.print("[cyan]Skipping to finalization...[/cyan]")
                else:
                    console.print(
                        f"[cyan]Resuming:[/cyan] {len(checkpoint_data.completed_chapters)} "
                        f"chapter(s) already completed, {len(remaining)} remaining"
                    )

                # Filter to only remaining chapters
                remaining_set = set(remaining)
                selected_indices = [
                    i for i in selected_indices if book.chapters[i].number in remaining_set
                ]
                selected_chapters = [book.chapters[i] for i in selected_indices]
                resumed = True

        # Create fresh checkpoint if none exists
        if CHECKPOINT_ENABLED and checkpoint_data is None:
            checkpoint_data = checkpoint_mgr.create_checkpoint(
                epub_hash=epub_hash,
                format=format,
                speaker=speaker,
                language=language,
                instruct=instruct,
                streaming=streaming,
                parallel=parallel,
                selected_chapters=[
                    book.chapters[i].number
                    for i in parse_chapter_selection(chapters, len(book.chapters))
                ],
                model_name=MODEL_NAME,
                sample_rate=sr,
            )

        # When checkpoint is active, force streaming mode so intermediate
        # WAV files are written to the persistent cache directory.
        audio_cache_dir: str | None = None
        if CHECKPOINT_ENABLED and checkpoint_data is not None:
            audio_cache_dir = checkpoint_data.chapter_audio_dir
            if not streaming:
                streaming = True  # Force streaming for checkpoint durability

        # Skip TTS if all chapters are already done (resumed + nothing remaining)
        if resumed and not selected_chapters:
            # All chapters were already completed — jump straight to finalization
            tts_engine: TTSEngine | None = None
            audio_manager = AudioManager()
            conversion_start = time.monotonic()

            resume_writer = StreamingAudioWriter(
                output_path=str(output_path),
                format=format,
                sample_rate=sr,
                cache_dir=audio_cache_dir,
            )
            # Reload completed chapter audio
            valid_files = checkpoint_mgr.verify_audio_files(checkpoint_data)  # type: ignore[arg-type]
            resume_writer.load_existing_chapters(
                valid_files,
                checkpoint_data.completed_titles,  # type: ignore[union-attr]
            )
            chapters_completed = len(valid_files)

            # Jump to finalization (handled by the existing streaming finalize block)
            console.print(f"\n[cyan]Finalizing {format.upper()} export...[/cyan]")
            book_metadata = {"title": book.title, "author": book.author}
            output_file = resume_writer.finalize(book_metadata)
            duration = resume_writer.total_duration
            resume_writer.cleanup()

            # Clean up checkpoint on success
            checkpoint_mgr.cleanup(epub_hash)

            total_chars = sum(len(ch.content) for ch in book.chapters)
            total_words = sum(len(ch.content.split()) for ch in book.chapters)

            console.print("\n[green]✓ Conversion complete![/green]")
            console.print(f"[cyan]Output:[/cyan] {output_file}")
            console.print(f"[cyan]Duration:[/cyan] {format_duration(duration)}")
            console.print(f"[cyan]Chapters:[/cyan] {chapters_completed}")
            console.print(f"[cyan]Input:[/cyan] {total_words:,} words, {total_chars:,} characters")
            elapsed = time.monotonic() - conversion_start
            console.print(f"[cyan]Elapsed:[/cyan] {format_duration(elapsed)}")
            return

        # Initialize TTS engine (may be unloaded later for parallel processing)
        console.print("\n[cyan]Initializing TTS Engine...[/cyan]")
        try:
            tts_engine = TTSEngine()
            console.print("[green]✓[/green] TTS Engine loaded successfully")
        except GPUNotAvailableError as e:
            console.print(f"[red]GPU Error:[/red] {e}")
            console.print(
                "[yellow]Hint:[/yellow] Ensure you have a CUDA-compatible GPU and PyTorch with CUDA support installed"
            )
            raise click.Abort() from e
        except TTSError as e:
            console.print(f"[red]TTS Engine Error:[/red] {e}")
            console.print(
                "[yellow]Hint:[/yellow] Check your internet connection and try again. The model will be downloaded on first use."
            )
            raise click.Abort() from e

        # Initialize audio manager
        audio_manager = AudioManager()

        # Start timing the conversion
        conversion_start = time.monotonic()

        # Process chapters with progress bar
        if streaming:
            console.print("\n[cyan]Converting chapters to audio in streaming mode...[/cyan]")
        else:
            console.print("\n[cyan]Converting chapters to audio...[/cyan]")

        audio_segments = []

        # Initialize streaming writer if in streaming mode
        streaming_writer: StreamingAudioWriter | None = None
        if streaming:
            streaming_writer = StreamingAudioWriter(
                output_path=str(output_path),
                format=format,
                sample_rate=sr,
                cache_dir=audio_cache_dir,
            )
            # If resuming, reload completed chapters into the writer
            if resumed and checkpoint_data is not None:
                valid_files = checkpoint_mgr.verify_audio_files(checkpoint_data)
                streaming_writer.load_existing_chapters(
                    valid_files,
                    checkpoint_data.completed_titles,
                )

        # Track progress for graceful shutdown
        chapters_completed = (
            len(checkpoint_data.completed_chapters)
            if resumed and checkpoint_data is not None
            else 0
        )

        # Try parallel processing if enabled
        use_parallel = parallel and len(selected_chapters) >= 2
        parallel_processor = None

        if use_parallel:
            try:
                from lalo.config import PARALLEL_PROCESSING_ENABLED
                from lalo.parallel_processor import ParallelChapterProcessor

                if PARALLEL_PROCESSING_ENABLED:
                    parallel_processor = ParallelChapterProcessor(max_parallel=max_parallel)
                    use_parallel = parallel_processor.should_use_parallel(len(selected_chapters))
                else:
                    use_parallel = False
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Parallel processing unavailable: {e}")
                console.print("[yellow]Falling back to sequential processing[/yellow]")
                use_parallel = False

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Parallel processing branch
            if use_parallel and parallel_processor:
                console.print(
                    f"[cyan]Using parallel processing with "
                    f"{parallel_processor.actual_parallel} worker(s)[/cyan]"
                )

                # Unload sequential TTS engine before parallel workers load
                # to free GPU memory (otherwise OOM when workers try to load models)
                if tts_engine is not None:
                    del tts_engine
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    tts_engine = None

                # Thread lock for progress callback synchronization
                progress_lock = threading.Lock()

                # Pre-create progress tasks for chapters that will be processed
                # (limited to max_workers to avoid showing too many at once)
                from rich.progress import TaskID

                chapter_tasks: dict[int, dict[str, Any]] = {}

                # Create initial progress bars for up to max_workers chapters
                initial_display = min(
                    len(selected_chapters), parallel_processor.actual_parallel + 1
                )
                for idx in range(initial_display):
                    chapter = selected_chapters[idx]
                    # Estimate total chunks for this chapter (static method, no model loading)
                    chunks = TTSEngine._chunk_text(chapter.content)
                    task: TaskID = progress.add_task(  # pyright: ignore[reportInvalidTypeForm]
                        f"[cyan]Ch {chapter.number}/{len(selected_chapters)}: {chapter.title[:35]}...",
                        total=len(chunks),
                    )
                    chapter_tasks[idx] = {"task": task, "last_chunk": 0}

                # Progress callback for parallel processing
                def parallel_progress_callback(
                    chapter_idx: int, current_chunk: int, total_chunks: int
                ):
                    with progress_lock:
                        # Create progress task if not already created
                        if chapter_idx not in chapter_tasks:
                            chapter = selected_chapters[chapter_idx]
                            chunks = TTSEngine._chunk_text(chapter.content)
                            task: TaskID = progress.add_task(  # pyright: ignore[reportInvalidTypeForm]
                                f"[cyan]Ch {chapter.number}/{len(selected_chapters)}: {chapter.title[:35]}...",
                                total=len(chunks),
                            )
                            chapter_tasks[chapter_idx] = {"task": task, "last_chunk": 0}

                        # Update progress
                        task_info = chapter_tasks[chapter_idx]
                        chunks_to_advance = current_chunk - task_info["last_chunk"]
                        if chunks_to_advance > 0:
                            progress.update(task_info["task"], advance=chunks_to_advance)
                            task_info["last_chunk"] = current_chunk

                # Process all chapters in parallel
                try:
                    # Determine language for each chapter
                    chapter_languages = []
                    for chapter in selected_chapters:
                        if language.lower() == "auto":
                            chapter_language = chapter.language or "English"
                        else:
                            chapter_language = language
                        chapter_languages.append(chapter_language)

                    # Call parallel processor
                    results = parallel_processor.process_chapters(
                        chapters=selected_chapters,
                        language=chapter_languages[0],  # TODO: support per-chapter languages
                        speaker=speaker,
                        instruct=instruct,
                        progress_callback=parallel_progress_callback,
                    )

                    if results is None:
                        # Parallel processing not used, fall back to sequential
                        console.print(
                            "[yellow]Parallel processing not used, using sequential[/yellow]"
                        )
                        use_parallel = False
                    else:
                        # Process results
                        for idx, (audio, sr) in enumerate(results):
                            if interrupted["flag"]:
                                console.print(
                                    "\n[yellow]⚠ Interrupted during parallel processing[/yellow]"
                                )
                                break

                            chapter = selected_chapters[idx]

                            # In streaming mode, write immediately; otherwise accumulate
                            if streaming_writer:
                                chapter_title = f"Chapter {chapter.number}: {chapter.title}"
                                streaming_writer.write_chapter(audio, chapter_title, chapter.number)
                            else:
                                audio_segments.append(audio)

                            # Mark chapter as complete
                            chapters_completed += 1
                            if checkpoint_data is not None:
                                chapter_title_ckpt = f"Chapter {chapter.number}: {chapter.title}"
                                checkpoint_mgr.mark_chapter_completed(
                                    checkpoint_data,
                                    chapter.number,
                                    chapter_title_ckpt,
                                )

                            # Mark progress task as complete
                            if idx in chapter_tasks:
                                task_id = chapter_tasks[idx]["task"]
                                # Get the task to find total
                                task_obj = next(
                                    (t for t in progress.tasks if t.id == task_id), None
                                )
                                if task_obj is not None:
                                    progress.update(
                                        task_id,
                                        completed=task_obj.total or 100,
                                    )

                except Exception as e:
                    # Check if it's a CUDA OOM error - save progress if we have completed chapters
                    is_oom_error = "CUDA out of memory" in str(e) or "OutOfMemoryError" in str(
                        type(e).__name__
                    )

                    # Check if exception has partial results attached
                    partial_results = getattr(e, "partial_results", None)

                    if is_oom_error and partial_results:
                        # Process partial results that completed before OOM
                        console.print(
                            "\n[red]CUDA Out of Memory Error:[/red] GPU ran out of memory during processing"
                        )
                        console.print(
                            f"[yellow]⚠ Saving {len(partial_results)} completed chapter(s) before exit...[/yellow]"
                        )

                        # Process the completed chapters
                        for idx, (audio, sr) in enumerate(partial_results):
                            chapter = selected_chapters[idx]
                            if streaming_writer:
                                chapter_title = f"Chapter {chapter.number}: {chapter.title}"
                                streaming_writer.write_chapter(audio, chapter_title, chapter.number)
                            else:
                                audio_segments.append(audio)
                            chapters_completed += 1
                            if checkpoint_data is not None:
                                chapter_title_ckpt = f"Chapter {chapter.number}: {chapter.title}"
                                checkpoint_mgr.mark_chapter_completed(
                                    checkpoint_data,
                                    chapter.number,
                                    chapter_title_ckpt,
                                )

                        # Trigger save by setting interrupted flag and record OOM as the reason
                        interrupted["flag"] = True
                        interrupted["reason"] = "oom"
                        # Continue to the interruption handler below which will save progress
                    else:
                        console.print(
                            f"\n[yellow]Warning:[/yellow] Parallel processing failed: {e}"
                        )
                        console.print("[yellow]Falling back to sequential processing[/yellow]")
                        use_parallel = False
                        # Reset for sequential processing
                        audio_segments = []
                        chapters_completed = 0

            # Sequential processing branch (original code or fallback)
            if not use_parallel:
                # Re-load TTS engine if it was unloaded for parallel processing
                if tts_engine is None:
                    console.print("\n[cyan]Loading TTS Engine for sequential processing...[/cyan]")
                    tts_engine = TTSEngine()

                for chapter_idx, chapter in enumerate(selected_chapters):
                    # Check for interruption before processing next chapter
                    if interrupted["flag"]:
                        console.print(
                            f"\n[yellow]⚠ Interrupted during chapter {chapter.number}[/yellow]"
                        )
                        break

                    # Determine language for this chapter
                    chapter_language = language
                    if language.lower() == "auto":
                        chapter_language = chapter.language or "English"

                    # Count chunks for this chapter to set proper total
                    chunks = tts_engine._chunk_text(chapter.content)
                    total_chunks = len(chunks)

                    # Create task for this chapter with actual chunk count as total
                    task = progress.add_task(
                        f"[cyan]Ch {chapter.number}/{len(selected_chapters)}: {chapter.title[:35]}...",
                        total=total_chunks,
                    )

                    # Progress callback for chunk-level updates
                    last_chunk = [0]  # Use list to modify in nested function

                    def update_progress(current_chunk: int, total_chunks: int):
                        # Calculate how many chunks to advance
                        chunks_to_advance = current_chunk - last_chunk[0]
                        if chunks_to_advance > 0:
                            progress.update(task, advance=chunks_to_advance)
                            last_chunk[0] = current_chunk

                    # Generate audio for chapter
                    try:
                        audio, sr = tts_engine.generate(
                            text=chapter.content,
                            language=chapter_language,
                            speaker=speaker,
                            instruct=instruct,
                            progress_callback=update_progress,
                        )

                        # In streaming mode, write immediately; otherwise accumulate
                        if streaming_writer:
                            chapter_title = f"Chapter {chapter.number}: {chapter.title}"
                            streaming_writer.write_chapter(audio, chapter_title, chapter.number)
                            # Don't keep in memory in streaming mode
                        else:
                            audio_segments.append(audio)

                        # Mark chapter as complete
                        chapters_completed += 1
                        if checkpoint_data is not None:
                            chapter_title_ckpt = f"Chapter {chapter.number}: {chapter.title}"
                            checkpoint_mgr.mark_chapter_completed(
                                checkpoint_data,
                                chapter.number,
                                chapter_title_ckpt,
                            )

                    except UnsupportedSpeakerError as e:
                        console.print(f"\n[red]Speaker Error:[/red] {e}")
                        console.print(
                            "[yellow]Hint:[/yellow] Use 'lalo speakers --list' to see available speakers"
                        )
                        raise click.Abort() from e
                    except TTSError as e:
                        # Check if it's a CUDA OOM error
                        is_oom_error = "CUDA out of memory" in str(e) or "OutOfMemoryError" in str(
                            type(e).__name__
                        )

                        if is_oom_error and chapters_completed > 0:
                            console.print(
                                f"\n[red]CUDA Out of Memory Error:[/red] GPU ran out of memory during chapter {chapter.number}"
                            )
                            console.print(
                                f"[yellow]⚠ Saving {chapters_completed} completed chapter(s) before exit...[/yellow]"
                            )
                            # Trigger save by setting interrupted flag and record OOM as the reason
                            interrupted["flag"] = True
                            interrupted["reason"] = "oom"
                            # Break to save progress
                            break
                        else:
                            console.print(
                                f"\n[red]TTS Error in chapter {chapter.number}:[/red] {e}"
                            )
                            console.print(f"[yellow]Chapter:[/yellow] {chapter.title}")
                            raise click.Abort() from e
                    except RuntimeError as e:
                        # Handle CUDA OOM raised directly as RuntimeError / torch.cuda.OutOfMemoryError
                        is_oom_error = "CUDA out of memory" in str(e) or "OutOfMemoryError" in str(
                            type(e).__name__
                        )

                        if is_oom_error and chapters_completed > 0:
                            console.print(
                                f"\n[red]CUDA Out of Memory Error:[/red] GPU ran out of memory during chapter {chapter.number}"
                            )
                            console.print(
                                f"[yellow]⚠ Saving {chapters_completed} completed chapter(s) before exit...[/yellow]"
                            )
                            # Trigger save by setting interrupted flag and record OOM as the reason
                            interrupted["flag"] = True
                            interrupted["reason"] = "oom"
                            # Break to save progress
                            break
                        else:
                            # Re-raise non-OOM runtime errors to be handled by outer logic
                            raise
                    except AudioError as e:
                        console.print(f"\n[red]Audio Error in chapter {chapter.number}:[/red] {e}")
                        console.print(f"[yellow]Chapter:[/yellow] {chapter.title}")
                        raise click.Abort() from e

                    # Mark chapter as complete in progress bar (in case callback didn't reach 100%)
                    progress.update(task, completed=total_chunks)

        # Check if we were interrupted
        if interrupted["flag"]:
            # Handle graceful shutdown - save what we have
            if chapters_completed == 0:
                console.print("\n[yellow]⚠ No chapters completed. Nothing to save.[/yellow]")
                if streaming_writer:
                    streaming_writer.cleanup()
                raise click.Abort()

            console.print(
                f"\n[yellow]⚠ Interrupted! Saved {chapters_completed} of {len(selected_chapters)} chapter(s)[/yellow]"
            )

            # Calculate resume chapter range
            resume_chapters = None
            if chapters_completed < len(selected_chapters):
                first_remaining = selected_indices[chapters_completed]
                last_remaining = selected_indices[-1]

                # Format resume command
                if first_remaining == last_remaining:
                    resume_chapters = str(first_remaining + 1)  # Convert to 1-indexed
                else:
                    resume_chapters = f"{first_remaining + 1}-{last_remaining + 1}"

            # Save partial progress based on mode
            if streaming_writer:
                # Streaming mode: finalize what we have
                console.print(f"\n[cyan]Finalizing partial {format.upper()} export...[/cyan]")

                book_metadata = {
                    "title": book.title,
                    "author": book.author,
                }

                output_file = streaming_writer.finalize(book_metadata)
                duration = streaming_writer.total_duration
                streaming_writer.cleanup()

                console.print(f"[green]✓[/green] Partial duration: {format_duration(duration)}")
            else:
                # Non-streaming mode: export completed segments
                if format == "m4b":
                    console.print("\n[cyan]Creating partial M4B audiobook...[/cyan]")

                    # Get chapter titles for completed chapters only
                    completed_chapters = selected_chapters[:chapters_completed]
                    chapter_titles = [
                        f"Chapter {ch.number}: {ch.title}" for ch in completed_chapters
                    ]

                    book_metadata = {
                        "title": book.title,
                        "author": book.author,
                    }

                    output_file = audio_manager.export_m4b(
                        audio_segments=audio_segments,
                        chapter_titles=chapter_titles,
                        output_path=str(output_path),
                        sample_rate=sr,
                        book_metadata=book_metadata,
                    )

                    total_samples = sum(len(seg) for seg in audio_segments)
                    duration = total_samples / sr
                    console.print(f"[green]✓[/green] Partial duration: {format_duration(duration)}")
                else:
                    # WAV/MP3: Concatenate completed segments
                    console.print("\n[cyan]Concatenating completed audio...[/cyan]")
                    full_audio = audio_manager.concatenate(audio_segments)

                    duration = audio_manager.get_duration(full_audio, sr)
                    console.print(f"[green]✓[/green] Partial duration: {format_duration(duration)}")

                    console.print(f"\n[cyan]Exporting to {format.upper()}...[/cyan]")
                    output_file = audio_manager.export(
                        full_audio,
                        str(output_path),
                        format=format,
                        sample_rate=sr,
                    )

            # Show summary with context-specific message
            if interrupted.get("reason") == "oom":
                console.print("\n[yellow]⚠ Conversion stopped due to GPU out of memory[/yellow]")
            else:
                console.print("\n[yellow]⚠ Conversion interrupted by user[/yellow]")

            console.print(f"[cyan]Partial output:[/cyan] {output_file}")
            console.print(
                f"[cyan]Chapters saved:[/cyan] {chapters_completed} of {len(selected_chapters)}"
            )
            console.print(f"[cyan]Duration:[/cyan] {format_duration(duration)}")

            # Show checkpoint-based resume hint
            if CHECKPOINT_ENABLED and checkpoint_data is not None:
                console.print("\n[cyan]Checkpoint saved.[/cyan] Resume with:")
                console.print(f"  lalo convert {epub_file}")
            elif resume_chapters:
                # Fallback to manual resume hint if checkpoint is disabled
                console.print(
                    f"\n[cyan]ℹ To resume:[/cyan] lalo convert {epub_file} --chapters {resume_chapters} --output {Path(output_file).stem}_continued.{format}"
                )

            raise click.Abort()

        console.print("[green]✓[/green] All chapters processed successfully")

        # Clean up checkpoint on success (before finalization so we don't
        # leave stale checkpoints if finalization succeeds)
        if CHECKPOINT_ENABLED and checkpoint_data is not None:
            checkpoint_mgr.cleanup(epub_hash)

        # Handle streaming mode finalization
        if streaming_writer:
            console.print(f"\n[cyan]Finalizing {format.upper()} export...[/cyan]")

            # Prepare book metadata
            book_metadata = {
                "title": book.title,
                "author": book.author,
            }

            output_file = streaming_writer.finalize(book_metadata)
            duration = streaming_writer.total_duration
            streaming_writer.cleanup()

            console.print(f"[green]✓[/green] Total duration: {format_duration(duration)}")

        # Export based on format (non-streaming mode)
        elif format == "m4b":
            # M4B: Export with chapter markers (no concatenation needed)
            console.print("\n[cyan]Creating M4B audiobook with chapter markers...[/cyan]")

            # Prepare chapter titles
            chapter_titles = [f"Chapter {ch.number}: {ch.title}" for ch in selected_chapters]

            # Prepare book metadata
            book_metadata = {
                "title": book.title,
                "author": book.author,
            }

            # Export M4B with chapters
            output_file = audio_manager.export_m4b(
                audio_segments=audio_segments,
                chapter_titles=chapter_titles,
                output_path=str(output_path),
                sample_rate=sr,
                book_metadata=book_metadata,
            )

            # Calculate duration from segments
            total_samples = sum(len(seg) for seg in audio_segments)
            duration = total_samples / sr
            console.print(f"[green]✓[/green] Total duration: {format_duration(duration)}")

        else:
            # WAV/MP3: Concatenate then export
            console.print("\n[cyan]Concatenating audio...[/cyan]")
            full_audio = audio_manager.concatenate(audio_segments)

            # Get duration
            duration = audio_manager.get_duration(full_audio, sr)
            console.print(f"[green]✓[/green] Total duration: {format_duration(duration)}")

            # Export to file
            console.print(f"\n[cyan]Exporting to {format.upper()}...[/cyan]")
            output_file = audio_manager.export(
                full_audio,
                str(output_path),
                format=format,
                sample_rate=sr,
            )

        # Calculate input statistics
        total_chars = sum(len(ch.content) for ch in selected_chapters)
        total_words = sum(len(ch.content.split()) for ch in selected_chapters)

        # Success message
        console.print("\n[green]✓ Conversion complete![/green]")
        console.print(f"[cyan]Output:[/cyan] {output_file}")
        console.print(f"[cyan]Duration:[/cyan] {format_duration(duration)}")
        console.print(f"[cyan]Chapters:[/cyan] {len(selected_chapters)}")
        console.print(f"[cyan]Input:[/cyan] {total_words:,} words, {total_chars:,} characters")
        elapsed = time.monotonic() - conversion_start
        console.print(f"[cyan]Elapsed:[/cyan] {format_duration(elapsed)}")

    except click.Abort:
        raise
    except EPUBError as e:
        console.print(f"\n[red]EPUB Error:[/red] {e}")
        console.print(f"[yellow]File:[/yellow] {epub_file}")
        console.print("[yellow]Hint:[/yellow] Ensure the file is a valid EPUB format")
        raise click.Abort() from e
    except AudioError as e:
        console.print(f"\n[red]Audio Export Error:[/red] {e}")
        console.print(f"[yellow]Output:[/yellow] {output}")
        console.print("[yellow]Hint:[/yellow] Check disk space and file permissions")
        raise click.Abort() from e
    except LaloError as e:
        # Catch any other custom Lalo exceptions
        console.print(f"\n[red]Error:[/red] {e}")
        raise click.Abort() from e
    except RuntimeError as e:
        # Catch CUDA OOM and other runtime errors
        is_oom_error = "CUDA out of memory" in str(e) or "out of memory" in str(e).lower()

        if is_oom_error:
            console.print(f"\n[red]CUDA Out of Memory Error:[/red] {e}")
            console.print(
                "\n[yellow]Recommendations:[/yellow]\n"
                "  1. Use --streaming mode to save memory: lalo convert <file> --streaming\n"
                "  2. Process fewer chapters at once: --chapters 1-10\n"
                "  3. Disable parallel processing: --no-parallel\n"
                "  4. Reduce batch size in config.py: TTS_BATCH_SIZE = 1"
            )
        else:
            console.print(f"\n[red]Runtime Error:[/red] {e}")
        raise click.Abort() from e
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}")
        console.print(
            "[yellow]This is a bug![/yellow] Please report it at https://github.com/willianpaixao/lalo/issues"
        )
        raise click.Abort() from e


@main.command()
@click.option("--list", "list_speakers", is_flag=True, help="List all available speakers")
def speakers(list_speakers: bool):
    """
    Show information about available speakers.

    Examples:

        lalo speakers --list
    """
    if not list_speakers:
        console.print("Use --list to see available speakers")
        return

    table = Table(title="Available Speakers")
    table.add_column("Speaker", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Native Language", style="yellow")

    for speaker_name, info in SPEAKER_INFO.items():
        table.add_row(speaker_name, info["description"], info["native_language"])

    console.print(table)


@main.command()
@click.option("--list", "list_languages", is_flag=True, help="List all supported languages")
def languages(list_languages: bool):
    """
    Show information about supported languages.

    Examples:

        lalo languages --list
    """
    if not list_languages:
        console.print("Use --list to see supported languages")
        return

    from lalo.config import QWEN_SUPPORTED_LANGUAGES

    table = Table(title="Supported Languages")
    table.add_column("Language", style="cyan")

    for lang in QWEN_SUPPORTED_LANGUAGES:
        table.add_row(lang)

    console.print(table)


@main.command()
@click.argument("epub_file", type=click.Path(exists=True))
def inspect(epub_file: str):
    """
    Inspect an EPUB file and list all chapters.

    Shows chapter number and title for each chapter.

    Examples:

        lalo inspect mybook.epub
    """
    try:
        # Parse EPUB
        console.print(f"\n[cyan]Inspecting EPUB:[/cyan] {epub_file}")
        book = parse_epub(epub_file)

        console.print(f"[green]✓[/green] {book.title} by {book.author}")
        console.print(f"[green]✓[/green] Total chapters: {len(book.chapters)}\n")

        # Create table for chapters without frame
        table = Table(show_header=True, show_edge=False, show_lines=False, box=None)
        table.add_column("Chapter", style="cyan", no_wrap=True)
        table.add_column("Title", style="white")

        for chapter in book.chapters:
            table.add_row(
                str(chapter.number),
                chapter.title,
            )

        console.print(table)

    except click.Abort:
        raise
    except EPUBError as e:
        console.print(f"\n[red]EPUB Error:[/red] {e}")
        console.print(f"[yellow]File:[/yellow] {epub_file}")
        console.print("[yellow]Hint:[/yellow] Ensure the file is a valid EPUB format")
        raise click.Abort() from e
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}")
        console.print(
            "[yellow]This is a bug![/yellow] Please report it at https://github.com/willianpaixao/lalo/issues"
        )
        raise click.Abort() from e


@main.group()
def cache():
    """Manage the checkpoint cache."""


@cache.command("list")
def cache_list():
    """
    List all cached checkpoints.

    Shows checkpoint details including EPUB name, progress, age,
    and disk usage.

    Examples:

        lalo cache list
    """
    import json as _json
    from datetime import datetime as _dt

    cache_base = CheckpointManager._resolve_cache_dir(None)

    if not cache_base.exists():
        console.print("[yellow]No checkpoint cache found.[/yellow]")
        return

    checkpoints_found = 0
    total_size = 0

    table = Table(title="Cached Checkpoints")
    table.add_column("EPUB", style="cyan", no_wrap=True, max_width=40)
    table.add_column("Progress", style="white")
    table.add_column("Format", style="yellow")
    table.add_column("Age", style="white")
    table.add_column("Size", style="white")

    for entry in sorted(cache_base.iterdir()):
        if not entry.is_dir():
            continue
        ckpt_file = entry / "checkpoint.json"
        if not ckpt_file.exists():
            continue

        try:
            raw = _json.loads(ckpt_file.read_text())
        except (OSError, _json.JSONDecodeError):
            continue

        epub_name = Path(raw.get("epub_file", "unknown")).name
        completed = len(raw.get("completed_chapters", []))
        total = raw.get("total_chapters", "?")
        fmt = raw.get("format", "?")
        updated = raw.get("updated_at", "")

        # Calculate age
        try:
            updated_dt = _dt.fromisoformat(updated)
            age_delta = _dt.now(UTC) - updated_dt
            if age_delta.days > 0:
                age_str = f"{age_delta.days}d ago"
            else:
                hours = age_delta.seconds // 3600
                age_str = f"{hours}h ago" if hours > 0 else "<1h ago"
        except (ValueError, TypeError):
            age_str = "unknown"

        # Calculate directory size
        dir_size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        total_size += dir_size
        if dir_size >= 1024 * 1024 * 1024:
            size_str = f"{dir_size / (1024 * 1024 * 1024):.1f} GB"
        elif dir_size >= 1024 * 1024:
            size_str = f"{dir_size / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{dir_size / 1024:.0f} KB"

        table.add_row(epub_name, f"{completed}/{total}", fmt, age_str, size_str)
        checkpoints_found += 1

    if checkpoints_found == 0:
        console.print("[yellow]No checkpoints found.[/yellow]")
        return

    console.print(table)

    # Total size summary
    if total_size >= 1024 * 1024 * 1024:
        total_str = f"{total_size / (1024 * 1024 * 1024):.1f} GB"
    elif total_size >= 1024 * 1024:
        total_str = f"{total_size / (1024 * 1024):.1f} MB"
    else:
        total_str = f"{total_size / 1024:.0f} KB"
    console.print(f"\n[cyan]Total cache size:[/cyan] {total_str}")
    console.print(f"[cyan]Cache location:[/cyan] {cache_base}")


@cache.command("clean")
@click.option(
    "--days",
    type=int,
    default=CHECKPOINT_STALE_DAYS,
    help=f"Remove checkpoints older than N days (default: {CHECKPOINT_STALE_DAYS})",
)
@click.option(
    "--all",
    "clean_all",
    is_flag=True,
    default=False,
    help="Remove all checkpoints regardless of age",
)
def cache_clean(days: int, clean_all: bool):
    """
    Remove stale or all cached checkpoints.

    Examples:

        lalo cache clean              # Remove checkpoints older than 30 days

        lalo cache clean --days 7     # Remove checkpoints older than 7 days

        lalo cache clean --all        # Remove all checkpoints
    """
    import json as _json
    import shutil
    from datetime import datetime as _dt, timedelta

    cache_base = CheckpointManager._resolve_cache_dir(None)

    if not cache_base.exists():
        console.print("[yellow]No checkpoint cache found.[/yellow]")
        return

    cutoff = _dt.now(UTC) - timedelta(days=days)
    removed = 0
    freed_bytes = 0

    for entry in sorted(cache_base.iterdir()):
        if not entry.is_dir():
            continue
        ckpt_file = entry / "checkpoint.json"
        if not ckpt_file.exists():
            continue

        should_remove = clean_all

        if not should_remove:
            try:
                raw = _json.loads(ckpt_file.read_text())
                updated = raw.get("updated_at", "")
                updated_dt = _dt.fromisoformat(updated)
                if updated_dt < cutoff:
                    should_remove = True
            except (OSError, _json.JSONDecodeError, ValueError, TypeError):
                # Cannot parse — consider it stale
                should_remove = True

        if should_remove:
            dir_size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            freed_bytes += dir_size
            epub_name = "unknown"
            try:
                raw = _json.loads(ckpt_file.read_text())
                epub_name = Path(raw.get("epub_file", "unknown")).name
            except Exception:
                pass

            shutil.rmtree(entry)
            console.print(f"[red]Removed:[/red] {epub_name} ({entry.name})")
            removed += 1

    if removed == 0:
        if clean_all:
            console.print("[yellow]No checkpoints to remove.[/yellow]")
        else:
            console.print(f"[yellow]No checkpoints older than {days} days.[/yellow]")
    else:
        if freed_bytes >= 1024 * 1024 * 1024:
            freed_str = f"{freed_bytes / (1024 * 1024 * 1024):.1f} GB"
        elif freed_bytes >= 1024 * 1024:
            freed_str = f"{freed_bytes / (1024 * 1024):.1f} MB"
        else:
            freed_str = f"{freed_bytes / 1024:.0f} KB"
        console.print(f"\n[green]✓[/green] Removed {removed} checkpoint(s), freed {freed_str}")


if __name__ == "__main__":
    main()
