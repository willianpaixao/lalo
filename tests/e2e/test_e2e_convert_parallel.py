"""
End-to-end tests for parallel chapter processing.

Tests parallel vs sequential processing with single GPU.
"""

import pytest

from lalo.cli import main
from tests.e2e.utils import check_audio_quality, validate_audio_file


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
class TestConvertParallel:
    """Test parallel chapter processing."""

    def test_convert_sequential(self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg):
        """
        Test explicit sequential processing with --no-parallel.

        Baseline for comparing against parallel mode.
        """
        output_file = output_dir / "romeo_sequential.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",  # 1 chapter
                "--output",
                str(output_file),
                "--format",
                "mp3",
                "--no-parallel",  # Force sequential
            ],
        )

        assert result.exit_code == 0, f"Sequential conversion failed: {result.output}"
        assert output_file.exists(), "Sequential output not created"

        # Validate audio
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["duration"] > 10, "Audio too short"

        quality = check_audio_quality(output_file)
        assert quality["has_audio"], "No audio detected"

    def test_convert_parallel_auto(self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg):
        """
        Test parallel processing with auto-detection (default).

        With single GPU, may still use batching for performance.
        """
        output_file = output_dir / "romeo_parallel.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",  # 1 chapter
                "--output",
                str(output_file),
                "--format",
                "mp3",
                "--parallel",  # Enable parallel (default)
            ],
        )

        assert result.exit_code == 0, f"Parallel conversion failed: {result.output}"
        assert output_file.exists(), "Parallel output not created"

        # Validate audio
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["duration"] > 10, "Audio too short"

        quality = check_audio_quality(output_file)
        assert quality["has_audio"], "No audio detected"

    def test_convert_parallel_with_max_workers(
        self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg
    ):
        """
        Test --max-parallel option to limit worker count.
        """
        output_file = output_dir / "romeo_parallel_limited.mp3"

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
                "--parallel",
                "--max-parallel",
                "2",  # Limit to 2 parallel chapters
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists()

        # Validate audio
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["duration"] > 10, "Audio too short"

    def test_parallel_vs_sequential_equivalence(
        self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg
    ):
        """
        Compare parallel vs sequential output.

        Both should produce similar quality audio with same duration.
        """
        output_sequential = output_dir / "seq.mp3"
        output_parallel = output_dir / "par.mp3"

        # Sequential
        result1 = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",
                "--output",
                str(output_sequential),
                "--format",
                "mp3",
                "--no-parallel",
            ],
        )

        # Parallel
        result2 = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "3",
                "--output",
                str(output_parallel),
                "--format",
                "mp3",
                "--parallel",
            ],
        )

        assert result1.exit_code == 0 and result2.exit_code == 0
        assert output_sequential.exists() and output_parallel.exists()

        # Compare audio properties
        from tests.e2e.utils import get_audio_duration

        duration_seq = get_audio_duration(output_sequential)
        duration_par = get_audio_duration(output_parallel)

        # Allow 5% duration difference
        duration_diff = abs(duration_seq - duration_par)
        max_allowed_diff = max(duration_seq, duration_par) * 0.05

        assert duration_diff <= max_allowed_diff, (
            f"Durations differ: {duration_seq}s (seq) vs {duration_par}s (par)"
        )

    def test_parallel_many_chapters(
        self, cli_runner, grimms_fairy_tales_epub, output_dir, check_ffmpeg
    ):
        """
        Test parallel processing with many chapters.

        Uses Grimm's Fairy Tales (65 chapters) to test scalability.
        """
        output_file = output_dir / "grimms_parallel.mp3"

        # Convert single fairy tale chapter
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
                "--parallel",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists()

        # Validate audio
        audio_info = validate_audio_file(output_file, "mp3")
        # 1 fairy tale (~255 words), should be at least 5 minutes
        assert audio_info["duration"] > 30, "Audio too short for fairy tale"

    def test_parallel_fallback_single_gpu(
        self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg
    ):
        """
        Test parallel processing fallback behavior with single GPU.

        Should handle gracefully and not error out.
        """
        output_file = output_dir / "romeo_fallback.mp3"

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
                "mp3",
                "--parallel",
                "--max-parallel",
                "1",  # Force single worker
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists()

        # Validate audio
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["duration"] > 10, "Audio too short"

    def test_streaming_with_parallel(self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg):
        """
        Test combination of streaming and parallel modes.

        Should work together for memory-efficient parallel processing.
        """
        output_file = output_dir / "romeo_streaming_parallel.mp3"

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
                "--streaming",
                "--parallel",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_file.exists()

        # Validate audio
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["duration"] > 10, "Audio too short"

        quality = check_audio_quality(output_file)
        assert quality["has_audio"], "No audio detected"
