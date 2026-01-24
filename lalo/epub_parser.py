"""
EPUB parsing and content extraction with structure preservation.
"""

from dataclasses import dataclass
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from lalo.exceptions import (
    EPUBInvalidError,
    EPUBNoChaptersError,
    EPUBNotFoundError,
    EPUBParseError,
)
from lalo.utils import detect_language


@dataclass
class Chapter:
    """Represents a chapter from an EPUB book."""

    number: int
    title: str
    content: str  # Plain text with preserved paragraph structure
    language: str | None = None

    def __str__(self) -> str:
        return f"Chapter {self.number}: {self.title}"


@dataclass
class Book:
    """Represents an EPUB book with metadata and chapters."""

    title: str
    author: str
    language: str | None
    chapters: list[Chapter]

    def __str__(self) -> str:
        return f"{self.title} by {self.author} ({len(self.chapters)} chapters)"


def clean_html(html_content: str) -> str:
    """
    Clean HTML content and extract text while preserving structure.

    Args:
        html_content: Raw HTML content

    Returns:
        Cleaned text with paragraph breaks preserved
    """
    soup = BeautifulSoup(html_content, "lxml-xml")

    # Remove script and style elements
    for script in soup(["script", "style", "meta", "link"]):
        script.decompose()

    # Get text with preserved structure
    # Strategy: Prefer structured elements (h1-h6, p, li) over containers (div)

    # First pass: collect structured content (headings, paragraphs, list items)
    structured_parts = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        text = element.get_text(separator=" ", strip=True)
        if text and len(text) > 0:
            structured_parts.append(text)

    # If we found structured content, use it (with deduplication)
    if structured_parts:
        seen = set()
        text_parts = []
        for text in structured_parts:
            if text not in seen:
                text_parts.append(text)
                seen.add(text)
    else:
        # Fallback: use divs if no structured content found
        text_parts = []
        seen = set()
        for element in soup.find_all(["div"]):
            text = element.get_text(separator=" ", strip=True)
            if text and text not in seen:
                text_parts.append(text)
                seen.add(text)

    # Last resort: just get all text
    if not text_parts:
        text = soup.get_text(separator=" ", strip=True)
        if text:
            text_parts.append(text)

    # Join with double newline to preserve paragraph breaks
    return "\n\n".join(text_parts)


def extract_chapters(book: epub.EpubBook) -> list[Chapter]:
    """
    Extract chapters from an EPUB book.

    Args:
        book: Parsed EPUB book object

    Returns:
        List of Chapter objects with content and metadata
    """
    chapters = []
    chapter_num = 1

    # Get all items in the book
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    for item in items:
        # Get HTML content
        html_content = item.get_content().decode("utf-8", errors="ignore")

        # Clean HTML and extract text
        text_content = clean_html(html_content)

        # Skip if content is too short (likely not a real chapter)
        if len(text_content.strip()) < 100:
            continue

        # Try to get chapter title from the item
        title = item.get_name()

        # Try to extract a better title from the HTML
        soup = BeautifulSoup(html_content, "lxml-xml")
        heading = soup.find(["h1", "h2", "h3"])
        if heading:
            heading_text = heading.get_text(strip=True)
            if heading_text:
                title = heading_text

        # Clean up the title
        if title:
            # Remove file extension if present
            title = Path(title).stem
            # Limit title length
            if len(title) > 100:
                title = title[:100] + "..."
        else:
            title = f"Chapter {chapter_num}"

        # Auto-detect language for this chapter
        language = detect_language(text_content[:1000])  # Use first 1000 chars

        chapter = Chapter(number=chapter_num, title=title, content=text_content, language=language)

        chapters.append(chapter)
        chapter_num += 1

    return chapters


def parse_epub(file_path: str) -> Book:
    """
    Parse an EPUB file and extract all content with structure.

    Args:
        file_path: Path to the EPUB file

    Returns:
        Book object with metadata and chapters

    Raises:
        EPUBNotFoundError: If file doesn't exist
        EPUBInvalidError: If file is not a valid EPUB
        EPUBNoChaptersError: If no readable chapters found
        EPUBParseError: If parsing fails
    """
    path = Path(file_path)

    if not path.exists():
        raise EPUBNotFoundError(file_path)

    if not path.suffix.lower() == ".epub":
        raise EPUBInvalidError(file_path, "File extension is not .epub")

    try:
        # Read EPUB file
        book = epub.read_epub(str(path))

        # Extract metadata
        title = book.get_metadata("DC", "title")
        title = title[0][0] if title else path.stem

        author = book.get_metadata("DC", "creator")
        author = author[0][0] if author else "Unknown Author"

        language = book.get_metadata("DC", "language")
        language = language[0][0] if language else None

        # Extract chapters
        chapters = extract_chapters(book)

        if not chapters:
            raise EPUBNoChaptersError(file_path)

        return Book(title=title, author=author, language=language, chapters=chapters)

    except (EPUBNotFoundError, EPUBInvalidError, EPUBNoChaptersError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Wrap any other exceptions
        raise EPUBParseError(file_path, e) from e


def get_chapter_summary(book: Book) -> str:
    """
    Get a formatted summary of all chapters in the book.

    Args:
        book: Book object

    Returns:
        Formatted string listing all chapters
    """
    lines = [f"\n{book.title} by {book.author}"]
    lines.append(f"Total chapters: {len(book.chapters)}\n")

    for chapter in book.chapters:
        word_count = len(chapter.content.split())
        lines.append(f"  {chapter.number:2d}. {chapter.title} ({word_count} words)")

    return "\n".join(lines)
