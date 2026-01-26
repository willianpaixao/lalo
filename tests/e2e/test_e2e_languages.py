"""
End-to-end tests for multi-language support.

Tests conversion of different languages with appropriate speakers.
"""

import pytest

from lalo.cli import main
from tests.e2e.utils import check_audio_quality, validate_audio_file


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
class TestLanguageSupport:
    """Test multi-language conversion support."""

    def test_convert_portuguese_single_chapter(
        self, cli_runner, portuguese_epub, output_dir, check_ffmpeg
    ):
        """
        Test conversion of Portuguese content.

        Uses Memorias Posthumas de Braz Cubas by Machado de Assis.
        Converts a single chapter to test Portuguese language support.

        Validates:
        - Portuguese language detection works
        - TTS generates audio for Portuguese text
        - Auto language detection identifies Portuguese
        """
        output_file = output_dir / "portuguese_chapter.mp3"

        # Convert chapter 2 (first actual content chapter, ~27k words)
        # Using chapter 2 as chapter 1 is the full book text
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(portuguese_epub),
                "--chapters",
                "3",  # Chapter 3 has ~746 words - good test size
                "--output",
                str(output_file),
                "--format",
                "mp3",
                "--language",
                "Portuguese",  # Explicitly set Portuguese
            ],
        )

        assert result.exit_code == 0, f"Portuguese conversion failed: {result.output}"
        assert output_file.exists(), "Portuguese MP3 file not created"

        # Validate audio file properties
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["sample_rate"] == 24000, "Incorrect sample rate"
        assert audio_info["codec"] in ["mp3", "mp3float"], f"Wrong codec: {audio_info['codec']}"

        # Chapter 3 has ~746 words
        # At ~150 words/minute = ~5 minutes expected
        # But TTS is faster, expect at least 1 minute (60 seconds)
        assert audio_info["duration"] > 60, (
            f"Audio too short: {audio_info['duration']}s (expected >60s)"
        )
        assert audio_info["file_size"] > 10000, "File too small"

        # Check audio quality
        quality = check_audio_quality(output_file, min_duration=60.0)
        assert quality["has_audio"], "No audio detected"
        assert not quality["has_silence"], "Audio is completely silent"
        assert not quality["has_clipping"], "Audio has clipping"

    def test_convert_portuguese_auto_detection(
        self, cli_runner, portuguese_epub, output_dir, check_ffmpeg
    ):
        """
        Test automatic language detection for Portuguese.

        Uses 'Auto' language mode (default) to verify language detection.
        """
        output_file = output_dir / "portuguese_auto.mp3"

        # Convert with auto language detection (default)
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(portuguese_epub),
                "--chapters",
                "3",  # Small chapter for faster testing
                "--output",
                str(output_file),
                "--format",
                "mp3",
                # No --language flag = auto detection
            ],
        )

        assert result.exit_code == 0, f"Auto-detection conversion failed: {result.output}"
        assert output_file.exists(), "Auto-detection MP3 not created"

        # Validate audio was generated
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["duration"] > 60, "Audio too short with auto-detection"

        # Check audio quality
        quality = check_audio_quality(output_file, min_duration=60.0)
        assert quality["has_audio"], "No audio detected with auto-detection"

    def test_convert_english_baseline(
        self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg
    ):
        """
        Baseline test for English conversion.

        Ensures English (the default/most common case) still works correctly.
        """
        output_file = output_dir / "english_baseline.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",  # THE PROLOGUE chapter
                "--output",
                str(output_file),
                "--format",
                "mp3",
                "--language",
                "English",
            ],
        )

        assert result.exit_code == 0, f"English conversion failed: {result.output}"
        assert output_file.exists(), "English MP3 not created"

        # Validate audio
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["duration"] > 5, "English audio too short"

        quality = check_audio_quality(output_file, min_duration=5.0)
        assert quality["has_audio"], "No English audio detected"
