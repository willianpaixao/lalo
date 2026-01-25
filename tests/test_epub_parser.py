"""
Tests for epub_parser module.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from ebooklib import epub

from lalo.epub_parser import (
    Book,
    Chapter,
    clean_html,
    extract_chapters,
    get_chapter_summary,
    parse_epub,
)
from lalo.exceptions import EPUBInvalidError, EPUBNoChaptersError, EPUBNotFoundError, EPUBParseError


class TestCleanHTML:
    """Tests for HTML cleaning functionality."""

    def test_clean_simple_paragraph(self):
        """Test cleaning simple paragraph HTML."""
        html = "<p>This is a simple paragraph.</p>"
        result = clean_html(html)
        assert result == "This is a simple paragraph."

    def test_clean_multiple_paragraphs(self):
        """Test cleaning multiple paragraphs with proper XML/HTML structure."""
        # Use proper XML structure with wrapper element (like real EPUB content)
        html = "<body><p>First paragraph.</p><p>Second paragraph.</p></body>"
        result = clean_html(html)
        assert "First paragraph." in result
        assert "Second paragraph." in result
        # Should have both paragraphs separated by double newline
        parts = result.split("\n\n")
        assert len(parts) == 2

    def test_remove_script_and_style(self):
        """Test that script and style tags are removed."""
        html = """
        <div>
            <script>alert('bad');</script>
            <style>.class { color: red; }</style>
            <p>Good content</p>
        </div>
        """
        result = clean_html(html)
        assert "alert" not in result
        assert "color: red" not in result
        assert "Good content" in result

    def test_clean_nested_tags(self):
        """Test cleaning deeply nested HTML."""
        html = "<div><div><p><strong>Nested</strong> content</p></div></div>"
        result = clean_html(html)
        assert "Nested content" in result

    def test_headings_preserved(self):
        """Test that headings are preserved."""
        # Use proper XML structure
        html = "<body><h1>Chapter Title</h1><p>Content here.</p></body>"
        result = clean_html(html)
        assert "Chapter Title" in result
        assert "Content here." in result
        # Should have both parts separated
        parts = result.split("\n\n")
        assert len(parts) == 2

    def test_list_items_preserved(self):
        """Test that list items are extracted."""
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        result = clean_html(html)
        assert "Item 1" in result
        assert "Item 2" in result

    def test_deduplication(self):
        """Test that duplicate text is removed."""
        # Duplicate paragraphs should be deduplicated to avoid redundancy
        html = "<div><p>Unique text</p><p>Unique text</p></div>"
        result = clean_html(html)
        # With deduplication, duplicate content appears only once
        assert result.count("Unique text") == 1
        assert result == "Unique text"

    def test_empty_html(self):
        """Test handling of empty HTML."""
        html = "<div></div>"
        result = clean_html(html)
        assert result == ""

    def test_html_with_only_whitespace(self):
        """Test HTML with only whitespace."""
        html = "<p>   </p><p>\n\t</p>"
        result = clean_html(html)
        assert result.strip() == ""

    def test_fallback_to_div(self):
        """Test fallback to div extraction when no structured content."""
        html = "<div>Fallback content in div</div>"
        result = clean_html(html)
        assert "Fallback content in div" in result

    def test_malformed_html(self):
        """Test handling of malformed HTML."""
        html = "<p>Unclosed paragraph<div>Mixed tags</p></div>"
        result = clean_html(html)
        # BeautifulSoup should handle this gracefully
        assert "Unclosed paragraph" in result or "Mixed tags" in result

    def test_unicode_content(self):
        """Test handling of Unicode content."""
        # Use proper XML structure
        html = "<body><p>日本語のテキスト</p><p>Émojis: 😀🎉</p></body>"
        result = clean_html(html)
        assert "日本語のテキスト" in result
        assert "😀🎉" in result or "Émojis:" in result  # Emojis should be preserved
        # Should have both paragraphs
        parts = result.split("\n\n")
        assert len(parts) == 2


class TestExtractChapters:
    """Tests for chapter extraction from EPUB."""

    def test_extract_single_chapter(self):
        """Test extracting a single chapter."""
        mock_book = Mock(spec=epub.EpubBook)

        # Create mock chapter item with enough content (>100 chars after cleaning)
        # Wrap in body tag like real EPUB content for proper XML parsing
        mock_item = Mock()
        content = b"<body><h1>Chapter 1</h1><p>Chapter content here. This is a longer paragraph with enough text to pass the minimum length requirement for chapter extraction. We need at least 100 characters.</p></body>"
        mock_item.get_content.return_value = content
        mock_item.get_name.return_value = "chapter1.html"

        mock_book.get_items_of_type.return_value = [mock_item]

        chapters = extract_chapters(mock_book)

        assert len(chapters) == 1
        assert chapters[0].number == 1
        assert "Chapter 1" in chapters[0].title
        assert "Chapter content" in chapters[0].content

    def test_extract_multiple_chapters(self):
        """Test extracting multiple chapters."""
        mock_book = Mock(spec=epub.EpubBook)

        mock_items = []
        for i in range(3):
            mock_item = Mock()
            # Add enough content to pass the 100 char minimum, wrapped in body
            content = f"<body><h1>Chapter {i + 1}</h1><p>This is the content for chapter {i + 1}. It needs to be long enough to pass the minimum length requirement of 100 characters after HTML cleaning.</p></body>"
            mock_item.get_content.return_value = content.encode()
            mock_item.get_name.return_value = f"chapter{i + 1}.html"
            mock_items.append(mock_item)

        mock_book.get_items_of_type.return_value = mock_items

        chapters = extract_chapters(mock_book)

        assert len(chapters) == 3
        assert chapters[0].number == 1
        assert chapters[1].number == 2
        assert chapters[2].number == 3

    def test_skip_short_chapters(self):
        """Test that very short content is skipped (likely not real chapters)."""
        mock_book = Mock(spec=epub.EpubBook)

        # One real chapter, one too short
        mock_item1 = Mock()
        mock_item1.get_content.return_value = (
            b"<body><p>Too short</p></body>"  # Less than 100 chars
        )
        mock_item1.get_name.return_value = "toc.html"

        mock_item2 = Mock()
        # Real chapter with enough content, wrapped in body
        content = b"<body><h1>Real Chapter</h1><p>This is enough content to be considered a real chapter. We need at least 100 characters of actual text content after HTML is cleaned.</p></body>"
        mock_item2.get_content.return_value = content
        mock_item2.get_name.return_value = "chapter1.html"

        mock_book.get_items_of_type.return_value = [mock_item1, mock_item2]

        chapters = extract_chapters(mock_book)

        assert len(chapters) == 1
        assert "Real Chapter" in chapters[0].title

    def test_title_from_heading(self):
        """Test that chapter title is extracted from heading tag."""
        mock_book = Mock(spec=epub.EpubBook)

        mock_item = Mock()
        content = b"<body><h2>Extracted Title</h2><p>This is the chapter content with enough text to meet the minimum length requirement of 100 characters after HTML cleaning is applied.</p></body>"
        mock_item.get_content.return_value = content
        mock_item.get_name.return_value = "ugly_filename.xhtml"

        mock_book.get_items_of_type.return_value = [mock_item]

        chapters = extract_chapters(mock_book)

        assert len(chapters) == 1
        assert chapters[0].title == "Extracted Title"

    def test_title_fallback_to_filename(self):
        """Test fallback to filename when no heading found."""
        mock_book = Mock(spec=epub.EpubBook)

        mock_item = Mock()
        content = b"<p>No heading in this chapter, but we have enough content to meet the minimum 100 character requirement after HTML cleaning is done by BeautifulSoup.</p>"
        mock_item.get_content.return_value = content
        mock_item.get_name.return_value = "chapter_one.html"

        mock_book.get_items_of_type.return_value = [mock_item]

        chapters = extract_chapters(mock_book)

        # Should use stem of filename
        assert chapters[0].title == "chapter_one"

    def test_title_truncation(self):
        """Test that very long titles are truncated."""
        mock_book = Mock(spec=epub.EpubBook)

        long_title = "A" * 150
        mock_item = Mock()
        mock_item.get_content.return_value = (
            f"<h1>{long_title}</h1><p>Content.</p>".encode() + b"a" * 100
        )
        mock_item.get_name.return_value = "chapter.html"

        mock_book.get_items_of_type.return_value = [mock_item]

        chapters = extract_chapters(mock_book)

        assert len(chapters[0].title) <= 103  # 100 chars + "..."
        assert chapters[0].title.endswith("...")

    def test_language_detection(self):
        """Test that language is detected for each chapter."""
        mock_book = Mock(spec=epub.EpubBook)

        mock_item = Mock()
        content = b"<p>This is English content with enough text to meet the minimum requirement. We need at least 100 characters of actual content for the chapter to be included.</p>"
        mock_item.get_content.return_value = content
        mock_item.get_name.return_value = "chapter.html"

        mock_book.get_items_of_type.return_value = [mock_item]

        chapters = extract_chapters(mock_book)

        assert len(chapters) == 1
        assert chapters[0].language is not None
        # Language detection should work for English
        assert chapters[0].language == "English"

    def test_empty_book(self):
        """Test handling of book with no items."""
        mock_book = Mock(spec=epub.EpubBook)
        mock_book.get_items_of_type.return_value = []

        chapters = extract_chapters(mock_book)

        assert chapters == []


class TestParseEPUB:
    """Tests for EPUB parsing."""

    def test_parse_nonexistent_file(self):
        """Test that parsing nonexistent file raises error."""
        with pytest.raises(EPUBNotFoundError, match="EPUB file not found"):
            parse_epub("/nonexistent/file.epub")

    def test_parse_non_epub_file(self):
        """Test that parsing non-EPUB file raises error."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_file = Path(f.name)

        try:
            with pytest.raises(EPUBInvalidError, match="extension is not .epub"):
                parse_epub(str(temp_file))
        finally:
            temp_file.unlink()

    @patch("lalo.epub_parser.epub.read_epub")
    def test_parse_valid_epub(self, mock_read_epub):
        """Test parsing a valid EPUB file."""
        # Create temporary EPUB file
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
            temp_file = Path(f.name)

        try:
            # Mock EPUB book
            mock_book = Mock(spec=epub.EpubBook)
            mock_book.get_metadata.side_effect = lambda ns, name: {
                ("DC", "title"): [("Test Book", {})],
                ("DC", "creator"): [("Test Author", {})],
                ("DC", "language"): [("en", {})],
            }.get((ns, name), [])

            # Mock chapter with sufficient content, wrapped in body
            mock_item = Mock()
            content = b"<body><h1>Chapter 1</h1><p>This is the content for the first chapter. It needs to be long enough to pass the 100 character minimum requirement after HTML cleaning.</p></body>"
            mock_item.get_content.return_value = content
            mock_item.get_name.return_value = "chapter1.html"
            mock_book.get_items_of_type.return_value = [mock_item]

            mock_read_epub.return_value = mock_book

            book = parse_epub(str(temp_file))

            assert book.title == "Test Book"
            assert book.author == "Test Author"
            assert book.language == "en"
            assert len(book.chapters) == 1
        finally:
            temp_file.unlink()

    @patch("lalo.epub_parser.epub.read_epub")
    def test_parse_epub_missing_metadata(self, mock_read_epub):
        """Test parsing EPUB with missing metadata."""
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
            temp_file = Path(f.name)

        try:
            mock_book = Mock(spec=epub.EpubBook)
            # Return empty lists for missing metadata
            mock_book.get_metadata.return_value = []

            mock_item = Mock()
            content = b"<p>This is sufficient content for a chapter. We need to ensure it meets the minimum 100 character requirement after HTML cleaning is performed.</p>"
            mock_item.get_content.return_value = content
            mock_item.get_name.return_value = "chapter.html"
            mock_book.get_items_of_type.return_value = [mock_item]

            mock_read_epub.return_value = mock_book

            book = parse_epub(str(temp_file))

            # Should use defaults
            assert book.title == temp_file.stem
            assert book.author == "Unknown Author"
            assert book.language is None
        finally:
            temp_file.unlink()

    @patch("lalo.epub_parser.epub.read_epub")
    def test_parse_epub_no_chapters(self, mock_read_epub):
        """Test that EPUB with no chapters raises error."""
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
            temp_file = Path(f.name)

        try:
            mock_book = Mock(spec=epub.EpubBook)
            mock_book.get_metadata.return_value = []
            mock_book.get_items_of_type.return_value = []

            mock_read_epub.return_value = mock_book

            with pytest.raises(EPUBNoChaptersError, match="No readable chapters found"):
                parse_epub(str(temp_file))
        finally:
            temp_file.unlink()

    @patch("lalo.epub_parser.epub.read_epub")
    def test_parse_epub_corrupted_file(self, mock_read_epub):
        """Test that corrupted EPUB raises parse error."""
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
            temp_file = Path(f.name)

        try:
            mock_read_epub.side_effect = Exception("Corrupted ZIP file")

            with pytest.raises(EPUBParseError, match="Failed to parse EPUB"):
                parse_epub(str(temp_file))
        finally:
            temp_file.unlink()


class TestChapterAndBook:
    """Tests for Chapter and Book dataclasses."""

    def test_chapter_str(self):
        """Test Chapter string representation."""
        chapter = Chapter(number=1, title="Introduction", content="Some content")
        assert str(chapter) == "Chapter 1: Introduction"

    def test_book_str(self):
        """Test Book string representation."""
        chapters = [
            Chapter(number=1, title="Ch1", content="Content1"),
            Chapter(number=2, title="Ch2", content="Content2"),
        ]
        book = Book(title="Test Book", author="Author", language="en", chapters=chapters)
        result = str(book)

        assert "Test Book" in result
        assert "Author" in result
        assert "2 chapters" in result


class TestGetChapterSummary:
    """Tests for chapter summary formatting."""

    def test_chapter_summary_format(self):
        """Test that chapter summary is properly formatted."""
        chapters = [
            Chapter(number=1, title="First Chapter", content="Word " * 100),
            Chapter(number=2, title="Second Chapter", content="Word " * 200),
        ]
        book = Book(title="Test Book", author="Test Author", language="en", chapters=chapters)

        summary = get_chapter_summary(book)

        assert "Test Book" in summary
        assert "Test Author" in summary
        assert "Total chapters: 2" in summary
        assert "1. First Chapter" in summary
        assert "2. Second Chapter" in summary
        assert "(100 words)" in summary
        assert "(200 words)" in summary

    def test_chapter_summary_with_long_title(self):
        """Test chapter summary with very long title."""
        long_title = "A" * 200
        chapters = [Chapter(number=1, title=long_title, content="Content")]
        book = Book(title="Book", author="Author", language="en", chapters=chapters)

        summary = get_chapter_summary(book)

        # Should still be formatted properly
        assert "1." in summary
        assert long_title[:50] in summary  # At least part of it
