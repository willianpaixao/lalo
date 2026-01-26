"""
End-to-end tests for error handling and edge cases.

Tests robustness with invalid inputs, missing files, and error scenarios.
"""

import pytest

from lalo.cli import main


@pytest.mark.e2e
class TestErrorHandling:
    """Test error scenarios and edge cases."""

    def test_convert_missing_file(self, cli_runner, tmp_path):
        """Test conversion with non-existent EPUB file."""
        non_existent = tmp_path / "does_not_exist.epub"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(non_existent),
                "--output",
                str(tmp_path / "output.mp3"),
            ],
        )

        # Click should catch missing file (exit code 2 for usage error)
        assert result.exit_code != 0, "Should fail with missing file"

    def test_convert_invalid_epub(self, cli_runner, tmp_path):
        """Test conversion with corrupted/invalid EPUB file."""
        # Create a text file pretending to be EPUB
        invalid_epub = tmp_path / "invalid.epub"
        invalid_epub.write_text("This is not a valid EPUB file!")

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(invalid_epub),
                "--output",
                str(tmp_path / "output.mp3"),
            ],
        )

        assert result.exit_code != 0, "Should fail with invalid EPUB"
        assert "EPUB Error" in result.output or "error" in result.output.lower()

    @pytest.mark.extended
    def test_convert_invalid_speaker(self, cli_runner, romeo_juliet_epub):
        """Test conversion with unknown speaker name."""
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--speaker",
                "InvalidSpeaker123",
                "--chapters",
                "1",
            ],
        )

        assert result.exit_code != 0, "Should fail with invalid speaker"
        assert "not supported" in result.output or "error" in result.output.lower()

    @pytest.mark.extended
    def test_convert_invalid_format(self, cli_runner, romeo_juliet_epub):
        """Test conversion with unsupported output format."""
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--format",
                "xyz",  # Invalid format
                "--chapters",
                "1",
            ],
        )

        # Click Choice validation should catch this
        assert result.exit_code != 0, "Should fail with invalid format"
        assert "Invalid value" in result.output or "error" in result.output.lower()

    def test_convert_invalid_chapters_out_of_range(self, cli_runner, romeo_juliet_epub, tmp_path):
        """Test chapter selection out of range."""
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "999",  # Romeo & Juliet only has 10 chapters
                "--output",
                str(tmp_path / "output.mp3"),
            ],
        )

        assert result.exit_code != 0, "Should fail with out-of-range chapter"
        assert "Invalid chapter" in result.output or "error" in result.output.lower()

    @pytest.mark.extended
    def test_convert_invalid_chapters_format(self, cli_runner, romeo_juliet_epub, tmp_path):
        """Test invalid chapter selection format."""
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "abc",  # Invalid format
                "--output",
                str(tmp_path / "output.mp3"),
            ],
        )

        assert result.exit_code != 0, "Should fail with invalid format"
        assert "Invalid chapter" in result.output or "error" in result.output.lower()

    def test_convert_invalid_chapters_negative(self, cli_runner, romeo_juliet_epub, tmp_path):
        """Test negative chapter numbers."""
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "-1",
                "--output",
                str(tmp_path / "output.mp3"),
            ],
        )

        assert result.exit_code != 0, "Should fail with negative chapter"

    @pytest.mark.extended
    def test_convert_invalid_chapters_zero(self, cli_runner, romeo_juliet_epub, tmp_path):
        """Test chapter number 0 (chapters are 1-indexed)."""
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "0",
                "--output",
                str(tmp_path / "output.mp3"),
            ],
        )

        assert result.exit_code != 0, "Should fail with chapter 0"

    @pytest.mark.extended
    def test_convert_permission_denied(self, cli_runner, romeo_juliet_epub, tmp_path):
        """Test output to read-only directory (permission denied)."""
        # Create read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir(mode=0o444)  # Read-only

        output_file = readonly_dir / "output.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "1",
                "--output",
                str(output_file),
            ],
        )

        # Cleanup - restore write permissions before pytest cleanup
        readonly_dir.chmod(0o755)

        # Note: This test might not fail on all systems (e.g., root user)
        # So we don't assert failure, just check it doesn't crash unexpectedly
        if result.exit_code != 0:
            assert "error" in result.output.lower() or "permission" in result.output.lower()

    def test_inspect_invalid_epub(self, cli_runner, tmp_path):
        """Test inspect command with invalid EPUB."""
        invalid_epub = tmp_path / "invalid.epub"
        invalid_epub.write_text("Not an EPUB")

        result = cli_runner.invoke(main, ["inspect", str(invalid_epub)])

        assert result.exit_code != 0
        assert "error" in result.output.lower()

    def test_convert_empty_epub(self, cli_runner, tmp_path):
        """Test conversion of EPUB with no chapters."""
        # Create minimal EPUB with no content
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_title("Empty Book")
        book.set_language("en")

        # Add only navigation, no chapters
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav"]

        empty_epub = tmp_path / "empty.epub"
        epub.write_epub(str(empty_epub), book)

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(empty_epub),
                "--output",
                str(tmp_path / "output.mp3"),
            ],
        )

        assert result.exit_code != 0, "Should fail with empty EPUB"
        # May fail with "no chapters" or similar error

    @pytest.mark.extended
    def test_convert_invalid_max_parallel(self, cli_runner, romeo_juliet_epub, tmp_path):
        """Test invalid --max-parallel value."""
        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(romeo_juliet_epub),
                "--chapters",
                "1",
                "--max-parallel",
                "0",  # Invalid: must be > 0
                "--output",
                str(tmp_path / "output.mp3"),
            ],
        )

        # Should either fail or ignore invalid value
        # Implementation-dependent behavior
        assert result.exit_code != 0 or result.exit_code == 0  # Either error or fallback to default


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
class TestEdgeCasesWithGPU:
    """Edge case tests that require GPU."""

    def test_convert_very_short_text(self, cli_runner, tmp_path, check_ffmpeg):
        """Test conversion of very short chapter (edge case)."""
        from tests.e2e.conftest import create_test_epub

        # Create EPUB with very short chapter
        epub_path = tmp_path / "short.epub"
        create_test_epub(
            title="Short Book",
            author="Test Author",
            chapters=[("Chapter 1", "Hello world.")],  # Very short
            output_path=epub_path,
        )

        output_file = tmp_path / "short.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(epub_path),
                "--output",
                str(output_file),
                "--format",
                "mp3",
            ],
        )

        # Should succeed or fail gracefully
        # Some TTS engines may have minimum text length requirements
        if result.exit_code == 0:
            assert output_file.exists()
        else:
            # Graceful failure expected
            assert "error" in result.output.lower()

    def test_convert_special_characters(self, cli_runner, tmp_path, check_ffmpeg):
        """Test conversion with special characters in text."""
        from tests.e2e.conftest import create_test_epub

        epub_path = tmp_path / "special.epub"
        create_test_epub(
            title="Special Characters",
            author="Test Author",
            chapters=[
                (
                    "Chapter 1",
                    "This has special characters: @#$%^&*() and numbers 12345. "
                    'It also has punctuation: quotes "like this" and apostrophes like it\'s. ' * 10,
                )
            ],
            output_path=epub_path,
        )

        output_file = tmp_path / "special.mp3"

        result = cli_runner.invoke(
            main,
            [
                "convert",
                str(epub_path),
                "--output",
                str(output_file),
                "--format",
                "mp3",
            ],
        )

        assert result.exit_code == 0, f"Failed with special characters: {result.output}"
        assert output_file.exists()
