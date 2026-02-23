"""
Tests for utils module.
"""

import tempfile
from pathlib import Path

import pytest

from lalo.exceptions import InvalidChapterSelectionError, InvalidFilePathError
from lalo.utils import (
    compute_file_hash,
    detect_language,
    format_duration,
    parse_chapter_selection,
    sanitize_filename,
    validate_file_exists,
)


class TestLanguageDetection:
    """Tests for language detection functionality."""

    def test_detect_english(self):
        text = "This is a sample English text."
        assert detect_language(text) == "English"

    def test_detect_chinese(self):
        text = "这是一个中文文本示例。"
        assert detect_language(text) == "Chinese"

    def test_empty_text_defaults_to_english(self):
        assert detect_language("") == "English"
        assert detect_language("   ") == "English"

    def test_unsupported_language_defaults_to_english(self, monkeypatch):
        """Test that unsupported languages default to English."""
        # Mock langdetect to return an unsupported language code
        # "fi" (Finnish) is not in LANGUAGE_MAP, so should fallback to English
        monkeypatch.setattr("lalo.utils.langdetect.detect", lambda text: "fi")

        text = "Some text"
        result = detect_language(text)
        assert result == "English"

    def test_detection_error_defaults_to_english(self):
        """Test that detection errors default to English."""
        # Very short text might cause detection to fail
        result = detect_language("a")
        assert result == "English"


class TestChapterSelection:
    """Tests for chapter selection parsing."""

    def test_all_chapters(self):
        result = parse_chapter_selection("all", 10)
        assert result == list(range(10))

    def test_single_chapter(self):
        result = parse_chapter_selection("5", 10)
        assert result == [4]  # 0-based index

    def test_chapter_range(self):
        result = parse_chapter_selection("3-7", 10)
        assert result == [2, 3, 4, 5, 6]  # 0-based indices

    def test_multiple_chapters(self):
        result = parse_chapter_selection("1,3,5", 10)
        assert result == [0, 2, 4]

    def test_mixed_selection(self):
        result = parse_chapter_selection("1,3-5,8", 10)
        assert result == [0, 2, 3, 4, 7]

    def test_out_of_bounds_raises_error(self):
        with pytest.raises(InvalidChapterSelectionError, match="out of bounds"):
            parse_chapter_selection("15", 10)

    def test_invalid_range_raises_error(self):
        with pytest.raises(InvalidChapterSelectionError, match="start must be <= end"):
            parse_chapter_selection("7-3", 10)

    def test_invalid_format_raises_error(self):
        with pytest.raises(InvalidChapterSelectionError, match="Invalid chapter number"):
            parse_chapter_selection("abc", 10)

    def test_range_start_out_of_bounds(self):
        """Test that range with start out of bounds raises error."""
        with pytest.raises(InvalidChapterSelectionError, match="out of bounds"):
            parse_chapter_selection("0-5", 10)

    def test_range_end_out_of_bounds(self):
        """Test that range with end out of bounds raises error."""
        with pytest.raises(InvalidChapterSelectionError, match="out of bounds"):
            parse_chapter_selection("5-15", 10)

    def test_invalid_range_format(self):
        """Test that invalid range format raises error."""
        with pytest.raises(InvalidChapterSelectionError, match="Invalid range format"):
            parse_chapter_selection("1-abc", 10)

    def test_negative_chapter_number(self):
        """Test that negative chapter number raises error."""
        # Negative numbers are parsed as ranges and raise format error
        with pytest.raises(InvalidChapterSelectionError):
            parse_chapter_selection("-1", 10)


class TestDurationFormatting:
    """Tests for duration formatting."""

    def test_format_seconds_only(self):
        assert format_duration(45) == "0:45"

    def test_format_minutes_and_seconds(self):
        assert format_duration(325) == "5:25"

    def test_format_with_hours(self):
        assert format_duration(3665) == "1:01:05"

    def test_zero_duration(self):
        assert format_duration(0) == "0:00"


class TestFilenameSanitization:
    """Tests for filename sanitization."""

    def test_remove_invalid_characters(self):
        result = sanitize_filename("Book: Title? Version 2.0!")
        assert ":" not in result
        assert "?" not in result

    def test_preserve_valid_characters(self):
        result = sanitize_filename("My_Book-Title_2024")
        assert result == "My_Book-Title_2024"

    def test_strip_spaces_and_dots(self):
        result = sanitize_filename("  .Book Title.  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")
        assert not result.startswith(".")

    def test_empty_filename_returns_untitled(self):
        result = sanitize_filename("")
        assert result == "untitled"

    def test_long_filename_truncated(self):
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) <= 255


class TestValidateFileExists:
    """Tests for file validation."""

    def test_valid_file_exists(self):
        """Test that valid file passes validation."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = Path(f.name)

        try:
            result = validate_file_exists(str(temp_file))
            assert result == temp_file
            assert result.exists()
        finally:
            temp_file.unlink()

    def test_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises error."""
        with pytest.raises(InvalidFilePathError, match="File not found"):
            validate_file_exists("/nonexistent/path/file.txt")

    def test_directory_raises_error(self):
        """Test that directory path raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(InvalidFilePathError, match="Path is not a file"):
                validate_file_exists(tmpdir)


class TestComputeFileHash:
    """Tests for file hashing utility."""

    def test_hash_deterministic(self):
        """Same file should always produce the same hash."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world")
            temp_path = Path(f.name)

        try:
            hash1 = compute_file_hash(str(temp_path))
            hash2 = compute_file_hash(str(temp_path))
            assert hash1 == hash2
        finally:
            temp_path.unlink()

    def test_hash_changes_with_content(self):
        """Different content should produce different hashes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"content A")
            path_a = Path(f.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"content B")
            path_b = Path(f.name)

        try:
            hash_a = compute_file_hash(str(path_a))
            hash_b = compute_file_hash(str(path_b))
            assert hash_a != hash_b
        finally:
            path_a.unlink()
            path_b.unlink()

    def test_hash_format(self):
        """Hash should be in 'algorithm:hexdigest' format."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test data")
            temp_path = Path(f.name)

        try:
            result = compute_file_hash(str(temp_path))
            assert result.startswith("sha256:")
            # SHA-256 hex digest is 64 characters
            hex_part = result.split(":", 1)[1]
            assert len(hex_part) == 64
            # Should only contain hex characters
            int(hex_part, 16)
        finally:
            temp_path.unlink()

    def test_hash_accepts_path_object(self):
        """Should accept Path objects as well as strings."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"path object test")
            temp_path = Path(f.name)

        try:
            result = compute_file_hash(temp_path)
            assert result.startswith("sha256:")
        finally:
            temp_path.unlink()

    def test_hash_nonexistent_file_raises(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="Cannot hash file"):
            compute_file_hash("/nonexistent/file.epub")
