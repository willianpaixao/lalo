"""
Integration tests for EPUB parsing to chapter extraction.

Tests the flow: EPUB file → Parser → Chapters
"""

from lalo.epub_parser import parse_epub, get_chapter_summary
from lalo.utils import parse_chapter_selection, sanitize_filename


class TestEPUBToChaptersIntegration:
    """Test EPUB parsing produces valid chapters for downstream components."""

    def test_epub_to_chapters_integration(self, simple_test_epub):
        """Parse real EPUB and verify chapters are valid for TTS."""
        # Parse actual EPUB file
        book = parse_epub(str(simple_test_epub))

        # Verify structure is valid
        assert len(book.chapters) == 3
        assert book.title == "Test Book"
        assert book.author == "Test Author"

        # Verify all chapters have content
        assert all(chapter.content for chapter in book.chapters)

        # Verify content meets minimum length for processing
        assert all(len(chapter.content) >= 100 for chapter in book.chapters)

        # Verify language detection worked
        assert all(chapter.language is not None for chapter in book.chapters)

        # Verify text is clean (no HTML tags)
        for chapter in book.chapters:
            assert "<" not in chapter.content
            assert ">" not in chapter.content
            assert "&lt;" not in chapter.content

    def test_chapter_selection_with_real_epub(self, simple_test_epub):
        """Test chapter selection on real EPUB data."""
        book = parse_epub(str(simple_test_epub))

        # Test various selections
        all_chapters = parse_chapter_selection("all", len(book.chapters))
        range_chapters = parse_chapter_selection("1-2", len(book.chapters))
        single_chapter = parse_chapter_selection("2", len(book.chapters))
        mixed_chapters = parse_chapter_selection("1,3", len(book.chapters))

        # Verify correct chapters selected
        assert len(all_chapters) == len(book.chapters)
        assert all_chapters == [0, 1, 2]

        assert len(range_chapters) == 2
        assert range_chapters == [0, 1]

        assert len(single_chapter) == 1
        assert single_chapter == [1]

        assert len(mixed_chapters) == 2
        assert mixed_chapters == [0, 2]

        # Verify selected chapters are valid
        selected = [book.chapters[i] for i in mixed_chapters]
        assert len(selected) == 2
        assert selected[0].number == 1
        assert selected[1].number == 3

    def test_metadata_extraction_integration(self, epub_no_metadata):
        """Test metadata extraction with fallbacks."""
        book = parse_epub(str(epub_no_metadata))

        # Verify fallback values used
        assert book.author == "Unknown Author"
        assert book.title  # Should use filename

        # Verify filename is sanitized and usable
        filename = sanitize_filename(book.title)
        assert len(filename) > 0
        assert "/" not in filename
        assert "\\" not in filename
        assert ":" not in filename

    def test_unicode_content_integration(self, unicode_test_epub):
        """Test EPUB with Unicode content."""
        book = parse_epub(str(unicode_test_epub))

        # Verify all chapters parsed
        assert len(book.chapters) == 3

        # Verify Unicode content preserved
        chapter_contents = [ch.content for ch in book.chapters]
        assert any("English" in content for content in chapter_contents)
        assert any("日本語" in content for content in chapter_contents)
        assert any("中文" in content for content in chapter_contents)

        # Verify language detection worked
        languages = [ch.language for ch in book.chapters]
        assert "English" in languages
        # Note: Japanese/Chinese detection may vary

    def test_chapter_summary_integration(self, simple_test_epub):
        """Test chapter summary generation from real EPUB."""
        book = parse_epub(str(simple_test_epub))

        summary = get_chapter_summary(book)

        # Verify summary contains key information
        assert book.title in summary
        assert book.author in summary
        assert "Total chapters: 3" in summary

        # Verify all chapter titles appear
        for chapter in book.chapters:
            assert chapter.title in summary
