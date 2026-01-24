"""
Tests for utils module.
"""

import pytest

from lalo.exceptions import InvalidChapterSelectionError
from lalo.utils import (
    detect_language,
    format_duration,
    parse_chapter_selection,
    sanitize_filename,
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
