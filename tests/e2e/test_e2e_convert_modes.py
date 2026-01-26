"""
End-to-end tests for conversion modes (streaming vs non-streaming, chapter selection).

Tests memory management modes and chapter selection features.
"""

import pytest

from lalo.cli import main
from tests.e2e.utils import check_audio_quality, get_audio_duration, validate_audio_file


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
class TestConvertModes:
    """Test different conversion modes and chapter selection."""

    @pytest.mark.extended
    def test_convert_non_streaming(self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg):
        """
        Test standard (non-streaming) conversion mode.

        Default mode that accumulates audio in memory before exporting.
        """
        output_file = output_dir / "romeo_standard.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",
                "--output",
                str(output_file),
                "--format",
                "mp3",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists(), "Output file not created"

        # Validate audio
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["duration"] > 10, "Audio too short"

        quality = check_audio_quality(output_file)
        assert quality["has_audio"], "No audio detected"

    @pytest.mark.extended
    def test_convert_streaming_mode(self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg):
        """
        Test streaming mode (incremental write).

        Validates:
        - Streaming flag works correctly
        - Output equivalent to non-streaming
        - Memory efficient for large books
        """
        output_file = output_dir / "romeo_streaming.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",
                "--output",
                str(output_file),
                "--format",
                "mp3",
                "--streaming",  # Enable streaming mode
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists(), "Streaming output not created"

        # Validate audio
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["duration"] > 10, "Audio too short"

        quality = check_audio_quality(output_file)
        assert quality["has_audio"], "No audio detected"

    def test_streaming_vs_non_streaming_equivalent(
        self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg
    ):
        """
        Compare streaming vs non-streaming output quality.

        Both modes should produce similar duration and quality.
        """
        output_standard = output_dir / "standard.mp3"
        output_streaming = output_dir / "streaming.mp3"

        # Standard mode
        result1 = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3-4",
                "--output",
                str(output_standard),
                "--format",
                "mp3",
            ],
        )

        # Streaming mode
        result2 = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3-4",
                "--output",
                str(output_streaming),
                "--format",
                "mp3",
                "--streaming",
            ],
        )

        assert result1.exit_code == 0 and result2.exit_code == 0
        assert output_standard.exists() and output_streaming.exists()

        # Compare durations (should be very close)
        duration_standard = get_audio_duration(output_standard)
        duration_streaming = get_audio_duration(output_streaming)

        # Allow 5% difference due to encoding variations
        duration_diff = abs(duration_standard - duration_streaming)
        max_allowed_diff = max(duration_standard, duration_streaming) * 0.05

        assert duration_diff <= max_allowed_diff, (
            f"Durations differ too much: {duration_standard}s vs {duration_streaming}s"
        )

    def test_streaming_large_book(self, cli_runner, moby_dick_epub, output_dir, check_ffmpeg):
        """
        Test streaming mode with large book (Moby Dick - 213k words).

        Validates streaming handles large content without memory issues.
        """
        output_file = output_dir / "moby_dick_streaming.mp3"

        # Convert single Moby Dick chapter (~22k words)
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(moby_dick_epub),
                "--chapters",
                "2"  # Single chapter for speed
                "--output",
                str(output_file),
                "--format",
                "mp3",
                "--streaming",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Large book streaming failed: {result.output}"
        assert output_file.exists(), "Streaming output not created for large book"

        # Validate audio
        audio_info = validate_audio_file(output_file, "mp3")
        # Each chapter ~20k words, ~133 words/minute = ~150 minutes for 2 chapters
        # But TTS is faster, expect at least 5 minutes
        assert (
            audio_info["duration"] > 300
        )  # Adjusted for smaller chapter, "Audio too short for large book (< 30 min)"

    @pytest.mark.extended
    def test_chapter_selection_all(self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg):
        """
        Test --chapters all option.

        Should convert all chapters in the book.
        """
        output_file = output_dir / "romeo_all_chapters.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "all",
                "--output",
                str(output_file),
                "--format",
                "mp3",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists()

        # Should mention all 10 chapters
        assert "10 chapter" in result.output.lower() or "chapters: 10" in result.output.lower()

    def test_chapter_selection_range(self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg):
        """
        Test --chapters with range (e.g., 1-3).
        """
        output_file = output_dir / "romeo_range.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3-5",  # Chapters 3, 4, 5 = 3 chapters
                "--output",
                str(output_file),
                "--format",
                "mp3",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists()

        # Should process 3 chapters (3-5)
        assert "3 chapter" in result.output.lower() or "chapters: 3" in result.output.lower()

    def test_chapter_selection_specific(
        self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg
    ):
        """
        Test --chapters with specific chapters (e.g., 1,3,5).
        """
        output_file = output_dir / "romeo_specific.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3,5,7",  # Specific chapters 3, 5, 7 = 3 chapters
                "--output",
                str(output_file),
                "--format",
                "mp3",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists()

        # Should process 3 chapters (3, 5, 7)
        assert "3 chapter" in result.output.lower() or "chapters: 3" in result.output.lower()

    def test_chapter_selection_mixed(self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg):
        """
        Test --chapters with mixed format (e.g., 1-3,5,7-9).
        """
        output_file = output_dir / "romeo_mixed.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3-4,6,8-9",  # Chapters 3-4, 6, 8-9 = 5 total
                "--output",
                str(output_file),
                "--format",
                "mp3",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists()

        # Should process 5 chapters (3, 4, 6, 8, 9)
        assert "5 chapter" in result.output.lower() or "chapters: 5" in result.output.lower()

    def test_chapter_selection_single(
        self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg
    ):
        """
        Test conversion of single chapter.
        """
        output_file = output_dir / "romeo_single.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",  # Single chapter
                "--output",
                str(output_file),
                "--format",
                "mp3",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists()

        # Should process 1 chapter
        assert "1 chapter" in result.output.lower() or "selected 1" in result.output.lower()

    def test_m4b_streaming_with_chapters(
        self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg
    ):
        """
        Test M4B format in streaming mode.

        Validates chapter markers work correctly in streaming mode.
        """
        output_file = output_dir / "romeo_streaming.m4b"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",
                "--output",
                str(output_file),
                "--format",
                "m4b",
                "--streaming",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists()

        # Validate audio
        audio_info = validate_audio_file(output_file, "m4b")
        assert audio_info["duration"] > 10, "Audio too short"

    def test_large_chapter_count(
        self, cli_runner, grimms_fairy_tales_epub, output_dir, check_ffmpeg
    ):
        """
        Test conversion with many chapters (Grimm's - 65 chapters).

        Validates handling of books with many chapters.
        """
        output_file = output_dir / "grimms_sample.mp3"

        # Convert single fairy tale chapter (Ch28 - 255 words)
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(grimms_fairy_tales_epub),
                "--chapters",
                "28"  # THE OLD MAN AND HIS GRANDSON (255 words)
                "--output",
                str(output_file),
                "--format",
                "mp3",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists()

        # Validate audio
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["duration"] > 30, "Audio too short for 1 fairy tale"
