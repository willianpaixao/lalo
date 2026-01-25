"""
Command-line interface for Lalo.
"""

import logging
import os
import signal
import threading
import warnings
from pathlib import Path
from typing import Any

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
from lalo.config import (
    DEFAULT_FORMAT,
    DEFAULT_LANGUAGE,
    DEFAULT_SPEAKER,
    SPEAKER_INFO,
    SUPPORTED_FORMATS,
    SUPPORTED_SPEAKERS,
)
from lalo.epub_parser import parse_epub
from lalo.exceptions import (
    AudioError,
    EPUBError,
    GPUNotAvailableError,
    InvalidChapterSelectionError,
    LaloError,
    TTSError,
    UnsupportedSpeakerError,
)
from lalo.tts_engine import TTSEngine
from lalo.utils import (
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
def convert(
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
    interrupted = {"flag": False, "force_exit": False}

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

        # Initialize TTS engine (may be unloaded later for parallel processing)
        console.print("\n[cyan]Initializing TTS Engine...[/cyan]")
        tts_engine: TTSEngine | None
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

        # Process chapters with progress bar
        if streaming:
            console.print("\n[cyan]Converting chapters to audio in streaming mode...[/cyan]")
        else:
            console.print("\n[cyan]Converting chapters to audio...[/cyan]")

        audio_segments = []
        sr = 24000  # Default sample rate for Qwen3-TTS

        # Initialize streaming writer if in streaming mode
        streaming_writer = None
        if streaming:
            streaming_writer = StreamingAudioWriter(
                output_path=str(output_path),
                format=format,
                sample_rate=sr,
            )

        # Track progress for graceful shutdown
        chapters_completed = 0

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
                    task: TaskID = progress.add_task(
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
                            task: TaskID = progress.add_task(
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
                    console.print(f"\n[yellow]Warning:[/yellow] Parallel processing failed: {e}")
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

                    except UnsupportedSpeakerError as e:
                        console.print(f"\n[red]Speaker Error:[/red] {e}")
                        console.print(
                            "[yellow]Hint:[/yellow] Use 'lalo speakers --list' to see available speakers"
                        )
                        raise click.Abort() from e
                    except TTSError as e:
                        console.print(f"\n[red]TTS Error in chapter {chapter.number}:[/red] {e}")
                        console.print(f"[yellow]Chapter:[/yellow] {chapter.title}")
                        raise click.Abort() from e
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

            # Show summary
            console.print("\n[yellow]⚠ Conversion interrupted by user[/yellow]")
            console.print(f"[cyan]Partial output:[/cyan] {output_file}")
            console.print(
                f"[cyan]Chapters saved:[/cyan] {chapters_completed} of {len(selected_chapters)}"
            )
            console.print(f"[cyan]Duration:[/cyan] {format_duration(duration)}")

            # Show resume command if there are remaining chapters
            if resume_chapters:
                console.print(
                    f"\n[cyan]ℹ To resume:[/cyan] lalo convert {epub_file} --chapters {resume_chapters} --output {Path(output_file).stem}_continued.{format}"
                )

            raise click.Abort()

        console.print("[green]✓[/green] All chapters processed successfully")

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


if __name__ == "__main__":
    main()
