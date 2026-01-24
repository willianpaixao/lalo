"""
Utility functions for Lalo.
"""

import re
from pathlib import Path

import langdetect

from lalo.config import LANGUAGE_MAP, QWEN_SUPPORTED_LANGUAGES
from lalo.exceptions import InvalidChapterSelectionError, InvalidFilePathError


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text: Text to detect language from

    Returns:
        Qwen3-TTS compatible language name (e.g., "English", "Chinese")
        Falls back to "English" if detection fails or language not supported
    """
    if not text or not text.strip():
        return "English"

    try:
        # Use langdetect to identify language code
        detected_code = langdetect.detect(text)

        # Map to Qwen3-TTS language
        qwen_language = LANGUAGE_MAP.get(detected_code)

        if qwen_language and qwen_language in QWEN_SUPPORTED_LANGUAGES:
            return qwen_language
        # Default to English for unsupported languages
        return "English"

    except Exception:
        # If detection fails, default to English
        return "English"


def parse_chapter_selection(selection: str, total_chapters: int) -> list[int]:
    """
    Parse chapter selection string into list of chapter indices.

    Supports formats:
    - "all" -> all chapters
    - "1,3,5" -> specific chapters
    - "1-5" -> range of chapters
    - "1,3-5,7" -> combination

    Args:
        selection: Chapter selection string
        total_chapters: Total number of chapters available

    Returns:
        List of chapter indices (0-based)

    Raises:
        ValueError: If selection format is invalid or indices out of range
    """
    if selection.lower() == "all":
        return list(range(total_chapters))

    selected: set[int] = set()

    # Split by comma
    parts = selection.split(",")

    for part in parts:
        part = part.strip()

        # Check if it's a range (e.g., "1-5")
        if "-" in part:
            try:
                start, end = part.split("-")
                start_idx = int(start.strip()) - 1  # Convert to 0-based
                end_idx = int(end.strip()) - 1  # Convert to 0-based

                if start_idx < 0 or end_idx >= total_chapters:
                    raise InvalidChapterSelectionError(
                        selection, f"Range {part} is out of bounds (1-{total_chapters})"
                    )

                if start_idx > end_idx:
                    raise InvalidChapterSelectionError(
                        selection, f"Range {part}: start must be <= end"
                    )

                selected.update(range(start_idx, end_idx + 1))

            except ValueError as e:
                if "invalid literal" in str(e):
                    raise InvalidChapterSelectionError(
                        selection, f"Invalid range format: {part}. Expected format like '1-5'"
                    ) from e
                raise
        else:
            # Single chapter number
            try:
                chapter_idx = int(part) - 1  # Convert to 0-based

                if chapter_idx < 0 or chapter_idx >= total_chapters:
                    raise InvalidChapterSelectionError(
                        selection, f"Chapter {part} is out of bounds (1-{total_chapters})"
                    )

                selected.add(chapter_idx)

            except ValueError as e:
                if "invalid literal" in str(e):
                    raise InvalidChapterSelectionError(
                        selection, f"Invalid chapter number: {part}. Expected integer"
                    ) from e
                raise

    return sorted(list(selected))


def validate_file_exists(file_path: str) -> Path:
    """
    Validate that a file exists.

    Args:
        file_path: Path to file

    Returns:
        Path object for the file

    Raises:
        InvalidFilePathError: If file doesn't exist or path is not a file
    """
    path = Path(file_path)

    if not path.exists():
        raise InvalidFilePathError(file_path, "File not found")

    if not path.is_file():
        raise InvalidFilePathError(file_path, "Path is not a file")

    return path


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1:23:45" or "5:30")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove invalid characters for filenames
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Remove leading/trailing spaces and dots
    filename = filename.strip(". ")

    # Limit length to 255 characters (common filesystem limit)
    if len(filename) > 255:
        filename = filename[:255]

    return filename or "untitled"
