"""
End-to-end tests for basic CLI commands.

Tests fundamental CLI interface without heavy conversion:
- Version display
- Help commands
- Speakers list
- Languages list
- EPUB inspection
"""

import pytest

from lalo import __version__
from lalo.cli import main


@pytest.mark.e2e
class TestCLIBasicCommands:
    """Test basic CLI commands that don't require GPU."""

    def test_cli_version(self, cli_runner):
        """Test lalo --version displays correct version."""
        result = cli_runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert __version__ in result.output
        assert "lalo" in result.output.lower()

    def test_cli_help(self, cli_runner):
        """Test lalo --help shows usage information."""
        result = cli_runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "EPUB to Audiobook Converter" in result.output
        assert "convert" in result.output
        assert "inspect" in result.output
        assert "speakers" in result.output
        assert "languages" in result.output

    def test_convert_help(self, cli_runner):
        """Test lalo convert --help shows convert command options."""
        result = cli_runner.invoke(main, ["convert", "--help"])

        assert result.exit_code == 0
        assert "--speaker" in result.output
        assert "--language" in result.output
        assert "--chapters" in result.output
        assert "--output" in result.output
        assert "--format" in result.output
        assert "--streaming" in result.output
        assert "--parallel" in result.output

    def test_speakers_list(self, cli_runner):
        """Test lalo speakers --list shows all available speakers."""
        result = cli_runner.invoke(main, ["speakers", "--list"])

        assert result.exit_code == 0

        # Verify all 9 speakers are shown
        expected_speakers = [
            "Vivian",
            "Serena",
            "Uncle_Fu",
            "Dylan",
            "Eric",
            "Ryan",
            "Aiden",
            "Ono_Anna",
            "Sohee",
        ]

        for speaker in expected_speakers:
            assert speaker in result.output

        # Verify table headers
        assert "Speaker" in result.output
        assert "Description" in result.output
        assert "Native Language" in result.output

    def test_languages_list(self, cli_runner):
        """Test lalo languages --list shows all supported languages."""
        result = cli_runner.invoke(main, ["languages", "--list"])

        assert result.exit_code == 0

        # Verify all 10 supported languages
        expected_languages = [
            "Chinese",
            "English",
            "Japanese",
            "Korean",
            "German",
            "French",
            "Russian",
            "Portuguese",
            "Spanish",
            "Italian",
        ]

        for language in expected_languages:
            assert language in result.output

    @pytest.mark.extended
    def test_inspect_epub(self, cli_runner, romeo_juliet_epub):
        """Test lalo inspect command shows chapter information."""
        result = cli_runner.invoke(main, ["inspect", str(romeo_juliet_epub)])

        assert result.exit_code == 0

        # Verify book metadata
        assert "Romeo and Juliet" in result.output
        assert "William Shakespeare" in result.output
        assert "Total chapters: 10" in result.output

        # Verify chapter table headers
        assert "Chapter" in result.output
        assert "Title" in result.output

    def test_inspect_invalid_file(self, cli_runner, tmp_path):
        """Test inspect with non-existent file shows error."""
        non_existent = tmp_path / "does_not_exist.epub"

        result = cli_runner.invoke(main, ["inspect", str(non_existent)])

        # Click should catch missing file before our code
        assert result.exit_code != 0

    def test_inspect_non_epub_file(self, cli_runner, tmp_path):
        """Test inspect with non-EPUB file shows appropriate error."""
        # Create a text file
        text_file = tmp_path / "not_an_epub.txt"
        text_file.write_text("This is not an EPUB file")

        result = cli_runner.invoke(main, ["inspect", str(text_file)])

        assert result.exit_code != 0
        assert "EPUB Error" in result.output or "error" in result.output.lower()

    def test_speakers_without_list_flag(self, cli_runner):
        """Test speakers command without --list flag shows hint."""
        result = cli_runner.invoke(main, ["speakers"])

        assert result.exit_code == 0
        assert "--list" in result.output

    def test_languages_without_list_flag(self, cli_runner):
        """Test languages command without --list flag shows hint."""
        result = cli_runner.invoke(main, ["languages"])

        assert result.exit_code == 0
        assert "--list" in result.output
