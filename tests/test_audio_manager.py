"""
Tests for audio_manager module.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lalo.audio_manager import AudioManager, StreamingAudioWriter
from lalo.exceptions import (
    AudioExportError,
    EmptyAudioError,
    FFmpegNotFoundError,
    UnsupportedAudioFormatError,
)


class TestAudioManager:
    """Tests for AudioManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AudioManager(sample_rate=24000)
        # Create sample audio data
        self.sample_audio = np.random.randn(24000).astype(np.float32)  # 1 second

    def test_concatenate_single_segment(self):
        """Test concatenating a single audio segment."""
        result = self.manager.concatenate([self.sample_audio])
        assert np.array_equal(result, self.sample_audio)

    def test_concatenate_multiple_segments(self):
        """Test concatenating multiple audio segments."""
        segments = [self.sample_audio, self.sample_audio, self.sample_audio]
        result = self.manager.concatenate(segments)
        expected_length = len(self.sample_audio) * 3
        assert len(result) == expected_length

    def test_concatenate_empty_raises_error(self):
        """Test that concatenating empty list raises error."""
        with pytest.raises(EmptyAudioError, match="Cannot process empty audio"):
            self.manager.concatenate([])

    def test_get_duration(self):
        """Test duration calculation."""
        duration = self.manager.get_duration(self.sample_audio)
        assert abs(duration - 1.0) < 0.01  # Should be ~1 second

    def test_export_unsupported_format_raises_error(self):
        """Test that exporting to unsupported format raises error."""
        with pytest.raises(UnsupportedAudioFormatError, match="Unsupported audio format"):
            self.manager.export(self.sample_audio, "test.ogg", format="ogg")

    def test_export_wav(self):
        """Test exporting to WAV format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"

            result = self.manager.export_wav(self.sample_audio, output_path)

            assert result.exists()
            assert result.suffix == ".wav"

    def test_export_wav_creates_directory(self):
        """Test that export_wav creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "test.wav"

            result = self.manager.export_wav(self.sample_audio, output_path)

            assert result.exists()
            assert result.parent.exists()

    def test_export_mp3(self):
        """Test exporting to MP3 format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp3"

            result = self.manager.export_mp3(self.sample_audio, output_path)

            assert result.exists()
            assert result.suffix == ".mp3"

    def test_export_mp3_custom_bitrate(self):
        """Test exporting MP3 with custom bitrate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp3"

            result = self.manager.export_mp3(self.sample_audio, output_path, bitrate="128k")

            assert result.exists()

    def test_export_generic_wav(self):
        """Test generic export method with WAV format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"

            result = self.manager.export(self.sample_audio, output_path, format="wav")

            assert result.exists()

    def test_export_generic_mp3(self):
        """Test generic export method with MP3 format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp3"

            result = self.manager.export(self.sample_audio, output_path, format="mp3")

            assert result.exists()

    def test_export_m4b_raises_error_without_chapters(self):
        """Test that M4B export through generic export raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.m4b"

            with pytest.raises(UnsupportedAudioFormatError):
                self.manager.export(self.sample_audio, output_path, format="m4b")

    @patch("subprocess.run")
    def test_export_m4b_with_chapters(self, mock_subprocess):
        """Test exporting to M4B format with chapter markers."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.m4b"

            segments = [self.sample_audio, self.sample_audio]
            titles = ["Chapter 1", "Chapter 2"]
            metadata = {"title": "Test Book", "author": "Test Author"}

            result = self.manager.export_m4b(
                segments,
                titles,
                output_path,
                book_metadata=metadata,
            )

            assert result.suffix == ".m4b"
            # ffmpeg should have been called
            assert mock_subprocess.called

    @patch("subprocess.run")
    def test_export_m4b_chapter_timestamps(self, mock_subprocess):
        """Test that M4B export creates correct chapter timestamps."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.m4b"

            # Create segments of known duration
            segment1 = np.random.randn(24000).astype(np.float32)  # 1 second
            segment2 = np.random.randn(48000).astype(np.float32)  # 2 seconds

            segments = [segment1, segment2]
            titles = ["Chapter 1", "Chapter 2"]

            self.manager.export_m4b(segments, titles, output_path)

            # Should have created temp metadata file
            assert mock_subprocess.called

    def test_get_duration_custom_sample_rate(self):
        """Test duration calculation with custom sample rate."""
        duration = self.manager.get_duration(self.sample_audio, sample_rate=48000)
        assert abs(duration - 0.5) < 0.01  # Should be ~0.5 seconds at 48kHz

    @patch("subprocess.run")
    def test_export_m4b_ffmpeg_not_found(self, mock_subprocess):
        """Test that missing ffmpeg raises FFmpegNotFoundError."""
        # Simulate ffmpeg not installed
        mock_subprocess.side_effect = FileNotFoundError("ffmpeg not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.m4b"

            segments = [self.sample_audio, self.sample_audio]
            titles = ["Chapter 1", "Chapter 2"]

            with pytest.raises(FFmpegNotFoundError, match="ffmpeg is required"):
                self.manager.export_m4b(segments, titles, output_path)

    @patch("subprocess.run")
    def test_export_m4b_ffmpeg_execution_error(self, mock_subprocess):
        """Test that ffmpeg execution errors are properly wrapped."""
        import subprocess

        # Simulate ffmpeg execution failure (not FileNotFoundError)
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"Invalid codec"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.m4b"

            segments = [self.sample_audio, self.sample_audio]
            titles = ["Chapter 1", "Chapter 2"]

            with pytest.raises(AudioExportError, match="Invalid codec"):
                self.manager.export_m4b(segments, titles, output_path)


class TestStreamingAudioWriter:
    """Tests for StreamingAudioWriter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_audio = np.random.randn(24000).astype(np.float32)

    def test_streaming_writer_initialization(self):
        """Test StreamingAudioWriter initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp3"

            writer = StreamingAudioWriter(output_path, format="mp3")

            assert writer.format == "mp3"
            assert writer.temp_dir.exists()

            writer.cleanup()

    def test_streaming_writer_unsupported_format(self):
        """Test that unsupported format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.ogg"

            with pytest.raises(UnsupportedAudioFormatError):
                StreamingAudioWriter(output_path, format="ogg")

    def test_write_chapter(self):
        """Test writing a chapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp3"

            writer = StreamingAudioWriter(output_path, format="mp3")
            writer.write_chapter(self.sample_audio, "Chapter 1", 1)

            assert len(writer.chapter_files) == 1
            assert writer.chapter_files[0].exists()
            assert len(writer.chapter_titles) == 1

            writer.cleanup()

    def test_write_multiple_chapters(self):
        """Test writing multiple chapters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp3"

            writer = StreamingAudioWriter(output_path, format="mp3")
            writer.write_chapter(self.sample_audio, "Chapter 1", 1)
            writer.write_chapter(self.sample_audio, "Chapter 2", 2)
            writer.write_chapter(self.sample_audio, "Chapter 3", 3)

            assert len(writer.chapter_files) == 3
            assert len(writer.chapter_titles) == 3

            writer.cleanup()

    def test_finalize_wav(self):
        """Test finalizing to WAV format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"

            writer = StreamingAudioWriter(output_path, format="wav")
            writer.write_chapter(self.sample_audio, "Chapter 1", 1)

            result = writer.finalize()

            assert result.exists()
            assert result.suffix == ".wav"

            writer.cleanup()

    def test_finalize_mp3(self):
        """Test finalizing to MP3 format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp3"

            writer = StreamingAudioWriter(output_path, format="mp3")
            writer.write_chapter(self.sample_audio, "Chapter 1", 1)

            result = writer.finalize()

            assert result.exists()
            assert result.suffix == ".mp3"

            writer.cleanup()

    @patch("subprocess.run")
    def test_finalize_m4b(self, mock_subprocess):
        """Test finalizing to M4B format."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.m4b"

            writer = StreamingAudioWriter(output_path, format="m4b")
            writer.write_chapter(self.sample_audio, "Chapter 1", 1)
            writer.write_chapter(self.sample_audio, "Chapter 2", 2)

            metadata = {"title": "Test Book", "author": "Test Author"}
            writer.finalize(book_metadata=metadata)

            assert mock_subprocess.called

            writer.cleanup()

    def test_finalize_empty_raises_error(self):
        """Test that finalizing with no chapters raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp3"

            writer = StreamingAudioWriter(output_path, format="mp3")

            with pytest.raises(EmptyAudioError):
                writer.finalize()

            writer.cleanup()

    def test_context_manager(self):
        """Test using StreamingAudioWriter as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp3"

            with StreamingAudioWriter(output_path, format="mp3") as writer:
                temp_dir = writer.temp_dir
                writer.write_chapter(self.sample_audio, "Chapter 1", 1)
                writer.finalize()

            # Temp directory should be cleaned up
            assert not temp_dir.exists()

    def test_duration_tracking(self):
        """Test that total duration is tracked correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp3"

            writer = StreamingAudioWriter(output_path, format="mp3", sample_rate=24000)
            writer.write_chapter(self.sample_audio, "Chapter 1", 1)  # 1 second
            writer.write_chapter(self.sample_audio, "Chapter 2", 2)  # 1 second

            assert abs(writer.total_duration - 2.0) < 0.1

            writer.cleanup()

    @patch("subprocess.run")
    def test_m4b_ffmpeg_error_handling(self, mock_subprocess):
        """Test that ffmpeg errors are properly handled."""
        # Simulate ffmpeg failure with CalledProcessError
        import subprocess

        mock_subprocess.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"ffmpeg failed"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.m4b"

            writer = StreamingAudioWriter(output_path, format="m4b")
            writer.write_chapter(self.sample_audio, "Chapter 1", 1)

            with pytest.raises(AudioExportError):
                writer.finalize()

            writer.cleanup()

    @patch("subprocess.run")
    def test_m4b_ffmpeg_not_found_streaming(self, mock_subprocess):
        """Test that missing ffmpeg raises FFmpegNotFoundError in streaming mode."""
        # Simulate ffmpeg not installed
        mock_subprocess.side_effect = FileNotFoundError("ffmpeg not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.m4b"

            writer = StreamingAudioWriter(output_path, format="m4b")
            writer.write_chapter(self.sample_audio, "Chapter 1", 1)

            with pytest.raises(FFmpegNotFoundError, match="ffmpeg is required"):
                writer.finalize()

            writer.cleanup()
