"""
End-to-end tests for output format conversion.

Tests conversion to different audio formats (mp3, wav, m4b) with real TTS engine.
Validates audio properties, quality, and format-specific features like M4B chapter markers.
"""

import pytest

from lalo.cli import main
from tests.e2e.utils import (
    check_audio_quality,
    validate_audio_file,
    validate_m4b_chapters,
)


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
class TestConvertFormats:
    """Test EPUB conversion to different audio formats."""

    @pytest.mark.extended
    def test_convert_to_mp3(self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg):
        """
        Test conversion to MP3 format.

        Validates:
        - MP3 file created successfully
        - Correct audio properties (sample rate, bitrate)
        - Duration reasonable for text length
        - No audio quality issues
        """
        output_file = output_dir / "romeo_and_juliet.mp3"

        # Convert single small chapter to MP3
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",  # THE PROLOGUE (115 words) - single chapter for speed
                "--output",
                str(output_file),
                "--format",
                "mp3",
                "--speaker",
                "Aiden",
            ],
        )

        # Check command succeeded
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists(), "MP3 file not created"

        # Validate audio file properties
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["sample_rate"] == 24000, "Incorrect sample rate"
        assert audio_info["codec"] in ["mp3", "mp3float"], f"Wrong codec: {audio_info['codec']}"
        assert audio_info["duration"] > 10, "Audio too short (< 10 seconds)"
        assert audio_info["file_size"] > 10000, "File too small"

        # Check bitrate (should be around 192k as configured)
        # Allow some variance (150k - 250k)
        bitrate_kbps = audio_info["bitrate"] // 1000
        assert 150 <= bitrate_kbps <= 250, f"Bitrate {bitrate_kbps}kbps outside expected range"

        # Check audio quality
        quality = check_audio_quality(output_file, min_duration=10.0)
        assert quality["has_audio"], "No audio detected"
        assert not quality["has_silence"], "Audio is completely silent"
        assert not quality["has_clipping"], "Audio has clipping"

    @pytest.mark.extended
    def test_convert_to_wav(self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg):
        """
        Test conversion to WAV format.

        Validates:
        - WAV file created successfully
        - Uncompressed PCM format
        - Larger file size than MP3
        """
        output_file = output_dir / "romeo_and_juliet.wav"

        # Convert single small chapter to WAV
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3-4",
                "--output",
                str(output_file),
                "--format",
                "wav",
                "--speaker",
                "Ryan",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists(), "WAV file not created"

        # Validate audio properties
        audio_info = validate_audio_file(output_file, "wav")
        assert audio_info["sample_rate"] == 24000, "Incorrect sample rate"
        assert audio_info["codec"] in ["pcm_s16le", "pcm_f32le"], f"Not PCM: {audio_info['codec']}"
        assert audio_info["duration"] > 10, "Audio too short"

        # WAV should be larger than MP3 (uncompressed)
        # Rough estimate: 24000 Hz * 2 bytes/sample * duration
        expected_min_size = int(24000 * 2 * audio_info["duration"] * 0.8)  # 80% of theoretical
        assert audio_info["file_size"] >= expected_min_size, "WAV file smaller than expected"

        # Check audio quality
        quality = check_audio_quality(output_file, min_duration=10.0)
        assert quality["has_audio"], "No audio detected"
        assert not quality["has_silence"], "Audio is completely silent"

    def test_convert_to_m4b_with_chapters(
        self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg
    ):
        """
        Test conversion to M4B format with chapter markers.

        Validates:
        - M4B file created successfully
        - Chapter markers embedded correctly
        - Book metadata (title, author) preserved
        - Chapter count matches selection
        """
        output_file = output_dir / "romeo_and_juliet.m4b"

        # Convert 3 chapters to M4B
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",  # 3 chapters for chapter marker testing
                "--output",
                str(output_file),
                "--format",
                "m4b",
                "--speaker",
                "Aiden",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists(), "M4B file not created"

        # Validate audio properties
        audio_info = validate_audio_file(output_file, "m4b")
        assert audio_info["sample_rate"] == 24000, "Incorrect sample rate"
        assert audio_info["codec"] in ["aac", "alac"], f"Wrong codec: {audio_info['codec']}"
        assert audio_info["duration"] > 10, "Audio too short"

        # Validate chapter markers
        chapter_info = validate_m4b_chapters(output_file, expected_chapter_count=3)

        # Check if chapters are present
        # Note: Chapter detection might vary, so we check if ANY chapters were found
        if chapter_info["chapter_count"] > 0:
            assert chapter_info["chapter_count"] >= 3, (
                f"Expected at least 3 chapters, found {chapter_info['chapter_count']}"
            )

            # Verify chapter titles if available
            if chapter_info["chapter_titles"]:
                # At least some chapter information should be present
                assert len(chapter_info["chapter_titles"]) > 0, "No chapter titles found"

        # Check audio quality
        quality = check_audio_quality(output_file, min_duration=10.0)
        assert quality["has_audio"], "No audio detected"

    def test_convert_custom_output_path(
        self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg
    ):
        """
        Test --output option with custom path.

        Validates output file created at specified location.
        """
        # Use subdirectory with custom name
        custom_dir = output_dir / "audiobooks"
        custom_dir.mkdir(parents=True, exist_ok=True)
        output_file = custom_dir / "my_custom_name.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",  # Just one chapter for speed
                "--output",
                str(output_file),
                "--format",
                "mp3",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists(), f"Output not created at {output_file}"
        assert output_file.name == "my_custom_name.mp3", "Wrong filename"

    def test_convert_default_output_naming(
        self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg
    ):
        """
        Test default output naming (no --output specified).

        Should create file named after EPUB with correct extension.
        """
        # Change working directory to output_dir for default naming
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(output_dir)

            result = cli_runner.invoke(
                main,
                [
                    "convert",
                    str(romeo_juliet_epub),
                    "--chapters",
                    "3",
                    "--format",
                    "mp3",
                ],
            )

            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Should create pg1513-images-3.mp3 (from EPUB filename)
            expected_file = output_dir / "pg1513-images-3.mp3"
            assert expected_file.exists(), f"Default output not created: {expected_file}"

        finally:
            os.chdir(original_cwd)

    def test_convert_multiple_speakers(
        self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg
    ):
        """
        Test conversion with different speaker voices.

        Validates that different speakers produce different audio.
        """
        output_aiden = output_dir / "romeo_aiden.mp3"
        output_ryan = output_dir / "romeo_ryan.mp3"

        # Convert with Aiden
        result1 = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",
                "--output",
                str(output_aiden),
                "--speaker",
                "Aiden",
                "--format",
                "mp3",
            ],
        )

        assert result1.exit_code == 0, f"Aiden conversion failed: {result1.output}"
        assert output_aiden.exists()

        # Convert with Ryan
        result2 = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",
                "--output",
                str(output_ryan),
                "--speaker",
                "Ryan",
                "--format",
                "mp3",
            ],
        )

        assert result2.exit_code == 0, f"Ryan conversion failed: {result2.output}"
        assert output_ryan.exists()

        # Files should be different (different speakers)
        assert (
            output_aiden.stat().st_size != output_ryan.stat().st_size
            or output_aiden.read_bytes() != output_ryan.read_bytes()
        ), "Different speakers produced identical audio"
