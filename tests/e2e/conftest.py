"""
Shared fixtures for end-to-end tests.
"""

from pathlib import Path

import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU (skip in CI)")
    config.addinivalue_line("markers", "slow: mark test as slow (>30 seconds)")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end integration test")
    config.addinivalue_line(
        "markers", "extended: mark test as requiring --extended flag (disabled by default)"
    )


def pytest_addoption(parser):
    """Add custom CLI options."""
    parser.addoption(
        "--keep-outputs",
        action="store_true",
        default=False,
        help="Preserve test outputs in tests/output/ directory",
    )
    parser.addoption(
        "--extended",
        action="store_true",
        default=False,
        help="Run extended tests including Romeo & Juliet (disabled by default)",
    )


@pytest.fixture(scope="session")
def epub_fixtures_dir():
    """Return path to EPUB fixtures directory."""
    return Path(__file__).parent / "fixtures" / "epubs"


@pytest.fixture(scope="session")
def romeo_juliet_epub(epub_fixtures_dir):
    """
    Romeo and Juliet EPUB (small/medium).
    - 10 chapters
    - ~27k words
    - English
    - Good for: format tests, chapter selection
    """
    epub_path = epub_fixtures_dir / "pg1513-images-3.epub"
    if not epub_path.exists():
        pytest.skip(f"EPUB not found: {epub_path}")
    return epub_path


@pytest.fixture(scope="session")
def grimms_fairy_tales_epub(epub_fixtures_dir):
    """
    Grimms' Fairy Tales EPUB (large).
    - 65 chapters
    - ~100k words
    - English
    - Good for: parallel processing, chapter selection stress tests
    """
    epub_path = epub_fixtures_dir / "pg2591-images-3.epub"
    if not epub_path.exists():
        pytest.skip(f"EPUB not found: {epub_path}")
    return epub_path


@pytest.fixture(scope="session")
def moby_dick_epub(epub_fixtures_dir):
    """
    Moby Dick EPUB (very large).
    - 12 chapters
    - ~213k words
    - English
    - Good for: streaming mode, memory stress tests
    """
    epub_path = epub_fixtures_dir / "pg2701-images-3.epub"
    if not epub_path.exists():
        pytest.skip(f"EPUB not found: {epub_path}")
    return epub_path


@pytest.fixture(scope="session")
def portuguese_epub(epub_fixtures_dir):
    """
    Memórias Póstumas de Brás Cubas EPUB (Portuguese).
    - 5 chapters
    - ~63k words
    - Portuguese language
    - Good for: multi-language testing, Portuguese TTS
    """
    epub_path = epub_fixtures_dir / "pg54829-images-3.epub"
    if not epub_path.exists():
        pytest.skip(f"EPUB not found: {epub_path}")
    return epub_path


@pytest.fixture
def output_dir(tmp_path, request):
    """
    Output directory with optional preservation.

    By default uses temporary directory (auto-cleanup).
    With --keep-outputs flag, saves to tests/output/<test_name>/
    """
    if request.config.getoption("--keep-outputs", default=False):
        output_path = Path("tests/output") / request.node.name
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    return tmp_path


@pytest.fixture(autouse=True)
def check_gpu_marker(request):
    """Skip tests marked with @pytest.mark.gpu if no GPU available."""
    if request.node.get_closest_marker("gpu"):
        if not torch.cuda.is_available():
            pytest.skip("GPU required for this test (CUDA not available)")


@pytest.fixture(autouse=True)
def check_extended_marker(request):
    """Skip tests marked with @pytest.mark.extended unless --extended is used."""
    if request.node.get_closest_marker("extended"):
        # Check if --extended flag is set
        if not request.config.getoption("--extended", default=False):
            pytest.skip("Extended tests disabled by default (use --extended to enable)")


@pytest.fixture(scope="session")
def check_ffmpeg():
    """Check if ffmpeg/ffprobe are available."""
    from tests.e2e.utils import check_ffmpeg_available

    if not check_ffmpeg_available():
        pytest.skip("ffmpeg and ffprobe are required for E2E tests")


def create_test_epub(title, author, chapters, output_path):
    """
    Create minimal valid EPUB for testing.

    Args:
        title: Book title
        author: Book author
        chapters: List of (chapter_title, content) tuples
        output_path: Path to save EPUB
    """
    from ebooklib import epub

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
def cli_runner():
    """Provide Click CLI test runner."""
    from click.testing import CliRunner

    return CliRunner()
