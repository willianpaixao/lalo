"""
Audio processing and export functionality.
"""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from lalo.config import AUDIO_SAMPLE_RATE, M4B_BITRATE, MP3_BITRATE, SUPPORTED_FORMATS
from lalo.exceptions import (
    AudioExportError,
    EmptyAudioError,
    UnsupportedAudioFormatError,
)


class AudioManager:
    """Handles audio concatenation and export to various formats."""

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE):
        """
        Initialize AudioManager.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def concatenate(self, audio_segments: list[np.ndarray]) -> np.ndarray:
        """
        Concatenate multiple audio segments into a single array.

        Args:
            audio_segments: List of numpy arrays containing audio data

        Returns:
            Concatenated audio as numpy array

        Raises:
            EmptyAudioError: If audio_segments is empty
        """
        if not audio_segments:
            raise EmptyAudioError()

        if len(audio_segments) == 1:
            return audio_segments[0]

        return np.concatenate(audio_segments)

    def export_wav(
        self,
        audio: np.ndarray,
        output_path: str | Path,
        sample_rate: int | None = None,
    ) -> Path:
        """
        Export audio to WAV file.

        Args:
            audio: Audio data as numpy array
            output_path: Path to output WAV file
            sample_rate: Sample rate (uses instance default if None)

        Returns:
            Path object for the created file
        """
        sr = sample_rate if sample_rate is not None else self.sample_rate
        output_path = Path(output_path)

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write WAV file
        sf.write(str(output_path), audio, sr)

        return output_path

    def export_mp3(
        self,
        audio: np.ndarray,
        output_path: str | Path,
        sample_rate: int | None = None,
        bitrate: str = MP3_BITRATE,
    ) -> Path:
        """
        Export audio to MP3 file.

        Args:
            audio: Audio data as numpy array
            output_path: Path to output MP3 file
            sample_rate: Sample rate (uses instance default if None)
            bitrate: MP3 bitrate (e.g., "192k")

        Returns:
            Path object for the created file
        """
        sr = sample_rate if sample_rate is not None else self.sample_rate
        output_path = Path(output_path)

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to WAV first (in-memory)
        import io

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sr, format="WAV")
        wav_buffer.seek(0)

        # Load WAV with pydub
        audio_segment = AudioSegment.from_wav(wav_buffer)

        # Export to MP3
        audio_segment.export(
            str(output_path),
            format="mp3",
            bitrate=bitrate,
        )

        return output_path

    def export_m4b(
        self,
        audio_segments: list[np.ndarray],
        chapter_titles: list[str],
        output_path: str | Path,
        sample_rate: int | None = None,
        bitrate: str = M4B_BITRATE,
        book_metadata: dict | None = None,
    ) -> Path:
        """
        Export audio to M4B format with chapter markers.

        Args:
            audio_segments: List of audio arrays (one per chapter)
            chapter_titles: List of chapter names
            output_path: Path to output M4B file
            sample_rate: Sample rate (uses instance default if None)
            bitrate: AAC bitrate (e.g., "192k")
            book_metadata: Optional dict with 'title', 'author'

        Returns:
            Path object for the created M4B file
        """

        sr = sample_rate if sample_rate is not None else self.sample_rate
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Track chapter timestamps and build combined audio
        chapter_markers = []
        current_time_ms = 0
        combined_audio = AudioSegment.empty()

        for i, (audio, title) in enumerate(zip(audio_segments, chapter_titles)):
            # Record chapter start time (in milliseconds)
            chapter_markers.append({"title": title, "start_time": current_time_ms})

            # Convert numpy array to AudioSegment
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            segment = AudioSegment(data=audio_bytes, sample_width=2, frame_rate=sr, channels=1)

            # Add to combined audio
            combined_audio += segment
            current_time_ms += len(segment)

        # Create chapter metadata file for ffmpeg
        import tempfile

        metadata_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        metadata_file.write(";FFMETADATA1\n")
        metadata_file.write(
            f"title={book_metadata.get('title', 'Audiobook') if book_metadata else 'Audiobook'}\n"
        )
        metadata_file.write(f"artist={book_metadata.get('author', '') if book_metadata else ''}\n")
        metadata_file.write("genre=Audiobook\n")
        metadata_file.write("\n")

        # Add chapters to metadata
        for marker in chapter_markers:
            metadata_file.write("[CHAPTER]\n")
            metadata_file.write("TIMEBASE=1/1000\n")
            metadata_file.write(f"START={marker['start_time']}\n")
            # Calculate end time (next chapter start or end of audio)
            idx = chapter_markers.index(marker)
            if idx < len(chapter_markers) - 1:
                end_time = chapter_markers[idx + 1]["start_time"]
            else:
                end_time = current_time_ms
            metadata_file.write(f"END={end_time}\n")
            metadata_file.write(f"title={marker['title']}\n")
            metadata_file.write("\n")

        metadata_file.close()

        # Export to temporary M4A without chapters first
        temp_m4a_no_chapters = output_path.with_suffix(".temp.m4a")
        combined_audio.export(
            str(temp_m4a_no_chapters), format="ipod", bitrate=bitrate, parameters=["-vn"]
        )

        # Use ffmpeg to add chapters and create final M4B
        final_path = output_path.with_suffix(".m4b")
        import subprocess

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i",
            str(temp_m4a_no_chapters),  # Input audio
            "-i",
            metadata_file.name,  # Metadata with chapters
            "-map_metadata",
            "1",  # Use metadata from second input
            "-c",
            "copy",  # Copy audio without re-encoding
            "-movflags",
            "+faststart",  # Optimize for streaming
            str(final_path),
        ]

        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

        # Cleanup temp files
        temp_m4a_no_chapters.unlink()
        Path(metadata_file.name).unlink()

        return final_path

    def export(
        self,
        audio: np.ndarray,
        output_path: str | Path,
        format: str = "mp3",
        sample_rate: int | None = None,
        **kwargs,
    ) -> Path:
        """
        Export audio to file with automatic format detection.

        Note: For M4B export, use export_m4b() directly as it requires
        separate chapter segments and metadata.

        Args:
            audio: Audio data as numpy array
            output_path: Path to output file
            format: Audio format ('wav', 'mp3')
            sample_rate: Sample rate (uses instance default if None)
            **kwargs: Additional format-specific arguments

        Returns:
            Path object for the created file

        Raises:
            UnsupportedAudioFormatError: If format is not supported
            AudioExportError: If export fails
        """
        format = format.lower()

        if format == "wav":
            return self.export_wav(audio, output_path, sample_rate)
        if format == "mp3":
            bitrate = kwargs.get("bitrate", MP3_BITRATE)
            return self.export_mp3(audio, output_path, sample_rate, bitrate)
        if format == "m4b":
            raise UnsupportedAudioFormatError(
                format,
                ["wav", "mp3"],  # M4B requires export_m4b() method
            )
        raise UnsupportedAudioFormatError(format, SUPPORTED_FORMATS)

    def get_duration(self, audio: np.ndarray, sample_rate: int | None = None) -> float:
        """
        Get duration of audio in seconds.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate (uses instance default if None)

        Returns:
            Duration in seconds
        """
        sr = sample_rate if sample_rate is not None else self.sample_rate
        return len(audio) / sr


class StreamingAudioWriter:
    """
    Context manager for streaming audio export.

    Writes audio chapters incrementally to disk, avoiding memory accumulation.
    Supports resumption if process is interrupted.
    """

    def __init__(
        self,
        output_path: str | Path,
        format: str = "mp3",
        sample_rate: int = AUDIO_SAMPLE_RATE,
        bitrate: str = MP3_BITRATE,
    ):
        """
        Initialize streaming audio writer.

        Args:
            output_path: Path to final output file
            format: Audio format ('wav', 'mp3', 'm4b')
            sample_rate: Audio sample rate
            bitrate: Bitrate for compressed formats
        """
        self.output_path = Path(output_path)
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.bitrate = bitrate

        # Validate format
        if self.format not in SUPPORTED_FORMATS:
            raise UnsupportedAudioFormatError(self.format, SUPPORTED_FORMATS)

        # Create temp directory for chapter files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="lalo_"))
        self.chapter_files: list[Path] = []
        self.chapter_titles: list[str] = []
        self.total_duration = 0.0

    def write_chapter(
        self,
        audio: np.ndarray,
        chapter_title: str,
        chapter_number: int,
    ) -> None:
        """
        Write a single chapter to temporary storage.

        Args:
            audio: Audio data as numpy array
            chapter_title: Title of the chapter
            chapter_number: Chapter number (for ordering)
        """
        # Create temporary WAV file for this chapter
        temp_file = self.temp_dir / f"chapter_{chapter_number:04d}.wav"

        # Write to WAV (lossless intermediate format)
        sf.write(str(temp_file), audio, self.sample_rate)

        # Track the file
        self.chapter_files.append(temp_file)
        self.chapter_titles.append(chapter_title)

        # Update duration
        duration = len(audio) / self.sample_rate
        self.total_duration += duration

    def finalize(self, book_metadata: dict | None = None) -> Path:
        """
        Combine all chapters and export to final format.

        Args:
            book_metadata: Optional metadata (title, author)

        Returns:
            Path to final output file
        """
        if not self.chapter_files:
            raise EmptyAudioError()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.format == "wav":
            # For WAV, concatenate all chapter files
            combined_audio = AudioSegment.empty()
            for chapter_file in self.chapter_files:
                segment = AudioSegment.from_wav(str(chapter_file))
                combined_audio += segment

            combined_audio.export(str(self.output_path), format="wav")

        elif self.format == "mp3":
            # For MP3, concatenate and export with bitrate
            combined_audio = AudioSegment.empty()
            for chapter_file in self.chapter_files:
                segment = AudioSegment.from_wav(str(chapter_file))
                combined_audio += segment

            combined_audio.export(
                str(self.output_path),
                format="mp3",
                bitrate=self.bitrate,
            )

        elif self.format == "m4b":
            # For M4B, use chapter markers
            self._export_m4b_with_chapters(book_metadata)

        return self.output_path

    def _export_m4b_with_chapters(self, book_metadata: dict | None = None) -> None:
        """Export to M4B with chapter markers."""
        # Load all chapters and track timestamps
        combined_audio = AudioSegment.empty()
        chapter_markers = []
        current_time_ms = 0

        for i, (chapter_file, title) in enumerate(zip(self.chapter_files, self.chapter_titles)):
            # Record chapter start
            chapter_markers.append(
                {
                    "title": title,
                    "start_time": current_time_ms,
                }
            )

            # Load and add segment
            segment = AudioSegment.from_wav(str(chapter_file))
            combined_audio += segment
            current_time_ms += len(segment)

        # Create metadata file
        metadata_file = self.temp_dir / "metadata.txt"
        with open(metadata_file, "w") as f:
            f.write(";FFMETADATA1\n")
            f.write(
                f"title={book_metadata.get('title', 'Audiobook') if book_metadata else 'Audiobook'}\n"
            )
            f.write(f"artist={book_metadata.get('author', '') if book_metadata else ''}\n")
            f.write("genre=Audiobook\n\n")

            # Add chapters
            for i, marker in enumerate(chapter_markers):
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={marker['start_time']}\n")

                # Calculate end time
                if i < len(chapter_markers) - 1:
                    end_time = chapter_markers[i + 1]["start_time"]
                else:
                    end_time = current_time_ms

                f.write(f"END={end_time}\n")
                f.write(f"title={marker['title']}\n\n")

        # Export to temporary M4A without chapters
        temp_m4a = self.temp_dir / "temp.m4a"
        combined_audio.export(
            str(temp_m4a),
            format="ipod",
            bitrate=self.bitrate,
            parameters=["-vn"],
        )

        # Use ffmpeg to add chapter metadata
        import subprocess

        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            str(temp_m4a),
            "-i",
            str(metadata_file),
            "-map_metadata",
            "1",
            "-codec",
            "copy",
            "-y",
            str(self.output_path),
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise AudioExportError("m4b", str(self.output_path), Exception(error_msg)) from e

    def cleanup(self) -> None:
        """Remove temporary files."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temp files."""
        self.cleanup()
        return False
