"""
Shared fixtures for integration tests.
"""

import numpy as np
import pytest
from ebooklib import epub


class MockTTSEngine:
    """Realistic TTS mock that returns valid audio arrays."""

    def __init__(self):
        self.sample_rate = 24000
        self.calls = []

    def generate(self, text, language, speaker, instruct=None, progress_callback=None):
        """Generate realistic audio based on text length."""
        self.calls.append(
            {
                "text": text[:50],  # Store snippet
                "language": language,
                "speaker": speaker,
            }
        )

        # Realistic audio: ~100 samples per character
        num_samples = max(1000, len(text) * 100)
        audio = np.random.randn(num_samples).astype(np.float32) * 0.1

        # Simulate chunking for progress
        if progress_callback:
            chunks = self._chunk_text(text)
            for i, _ in enumerate(chunks):
                progress_callback(i + 1, len(chunks))

        return audio, self.sample_rate

    def _chunk_text(self, text, max_chars=2000):
        """Simple chunking."""
        if len(text) <= max_chars:
            return [text]
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    def get_supported_speakers(self):
        return ["Ryan", "Emily", "Grace", "Ono_Anna"]

    def get_supported_languages(self):
        return ["English", "Chinese", "Japanese", "Korean", "German", "French"]


@pytest.fixture
def mock_tts_engine():
    """Provide mock TTS engine for integration tests."""
    return MockTTSEngine()


def create_test_epub(title, author, chapters, output_path):
    """
    Create minimal valid EPUB for testing.

    Args:
        title: Book title
        author: Book author
        chapters: List of (chapter_title, content) tuples
        output_path: Path to save EPUB
    """
    book = epub.EpubBook()
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)

    epub_chapters = []
    for i, (chapter_title, content) in enumerate(chapters, 1):
        chapter = epub.EpubHtml(title=chapter_title, file_name=f"chap_{i:02d}.xhtml", lang="en")
        # Proper EPUB structure with body tag
        chapter.content = f"""
        <html>
        <head><title>{chapter_title}</title></head>
        <body>
            <h1>{chapter_title}</h1>
            <p>{content}</p>
        </body>
        </html>
        """
        book.add_item(chapter)
        epub_chapters.append(chapter)

    # Add navigation
    book.toc = list(epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Define spine
    book.spine = ["nav"] + epub_chapters

    # Write EPUB
    epub.write_epub(str(output_path), book)
    return output_path


@pytest.fixture
def simple_test_epub(tmp_path):
    """Create a simple 3-chapter test EPUB."""
    epub_path = tmp_path / "test_simple.epub"
    chapters = [
        (
            "Introduction",
            "This is the introduction chapter. It provides an overview of the book and sets the stage for what is to come. "
            * 5,
        ),
        (
            "Main Content",
            "This is the main content chapter with substantial text to ensure it passes the minimum length requirements. "
            * 5,
        ),
        (
            "Conclusion",
            "This is the conclusion chapter that wraps up all the main points and provides final thoughts on the subject matter. "
            * 5,
        ),
    ]
    return create_test_epub("Test Book", "Test Author", chapters, epub_path)


@pytest.fixture
def unicode_test_epub(tmp_path):
    """Create EPUB with Unicode content."""
    epub_path = tmp_path / "test_unicode.epub"
    chapters = [
        ("English Chapter", "This is English text content. " * 10),
        ("日本語章", "これは日本語のテキストコンテンツです。" * 15),
        ("中文章节", "这是中文文本内容。" * 15),
    ]
    return create_test_epub("Unicode Test Book", "Test Author", chapters, epub_path)


@pytest.fixture
def epub_no_metadata(tmp_path):
    """Create EPUB without metadata."""
    epub_path = tmp_path / "test_no_metadata.epub"

    book = epub.EpubBook()
    # Intentionally don't set title, author, language

    chapter = epub.EpubHtml(title="Chapter", file_name="chap_01.xhtml", lang="en")
    chapter.content = "<html><body><h1>Chapter</h1><p>" + ("Content. " * 20) + "</p></body></html>"

    book.add_item(chapter)
    book.toc = [chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]

    epub.write_epub(str(epub_path), book)
    return epub_path


@pytest.fixture
def cli_runner():
    """Provide Click CLI test runner."""
    from click.testing import CliRunner

    return CliRunner()
