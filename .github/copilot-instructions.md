# Copilot Instructions for Lalo

This file provides context and guidelines for GitHub Copilot when working on the Lalo EPUB to Audiobook Converter project.

## Project Overview

Lalo is a Python 3.12+ CLI tool that converts EPUB files to high-quality audiobooks using Qwen3-TTS. It supports:
- 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
- 9 high-quality voice profiles with natural speech characteristics
- GPU-accelerated batch processing with CUDA
- Multiple export formats (WAV, MP3, M4B with chapter markers)
- Auto language detection and flexible chapter selection

## Code Style & Standards

### Type Safety
- **Required:** Full type annotations on all functions (params and returns)
- Use modern syntax: `list[str]`, `dict[str, int]`, `str | None` (not `Optional[str]`)
- Type checkers: mypy (strict) and pyright enabled

```python
# ✅ Good
def parse_chapters(selection: str, total: int) -> list[int]:
    """Parse chapter selection string."""
    pass

# ❌ Bad - missing types
def parse_chapters(selection, total):
    pass
```

### Formatting (Ruff)
- **Line length:** 100 characters max
- **Quotes:** Double quotes for strings
- **Indentation:** 4 spaces (no tabs)
- Run `ruff format` before committing

### Import Order
1. Standard library (alphabetically sorted)
2. Third-party packages (alphabetically sorted)
3. Local application imports (alphabetically sorted)

```python
# Standard library
import logging
from pathlib import Path
from typing import Any

# Third-party
import click
import torch
from rich.console import Console

# Local
from lalo.config import DEFAULT_SPEAKER
from lalo.exceptions import EPUBError, TTSError
```

### Naming Conventions
- **Functions/variables:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private/internal:** Prefix with `_`

## Error Handling

### Custom Exception Hierarchy
Always use domain-specific exceptions from `lalo/exceptions.py`:

```
LaloError (base)
├── EPUBError (EPUBNotFoundError, EPUBInvalidError, EPUBParseError, EPUBNoChaptersError)
├── TTSError (GPUNotAvailableError, TTSModelLoadError, UnsupportedLanguageError, UnsupportedSpeakerError)
├── AudioError (AudioExportError, UnsupportedAudioFormatError, EmptyAudioError, FFmpegNotFoundError)
└── ValidationError (InvalidChapterSelectionError, InvalidFilePathError)
```

**Best practices:**
- Never use bare `Exception` - use specific exception classes
- Include context in exception messages
- Chain exceptions with `from e` to preserve stack traces
- Handle user-facing errors in CLI layer (`cli.py`)

```python
# ✅ Good - specific exception with context
try:
    book = parse_epub(file_path)
except EPUBParseError as e:
    console.print(f"[red]Failed to parse EPUB:[/red] {e}")
    raise click.Abort() from e

# ❌ Bad - bare exception
except Exception:
    raise
```

## Project Structure

```
lalo/
├── __init__.py          # Package initialization, version
├── cli.py               # Click-based CLI interface (user-facing commands)
├── config.py            # Configuration constants (speakers, languages, defaults)
├── exceptions.py        # Custom exception classes
├── epub_parser.py       # EPUB parsing logic (ebooklib + BeautifulSoup)
├── tts_engine.py        # TTS engine wrapper (Qwen3-TTS integration)
├── audio_manager.py     # Audio processing and export (soundfile, pydub)
├── parallel_processor.py # Multi-GPU parallel processing
└── utils.py             # Utility functions

tests/
├── conftest.py          # Pytest fixtures
├── test_*.py            # Unit tests (mirror source structure)
└── integration/         # Integration tests
```

## Key Development Commands

### Testing
```bash
# Run all tests with coverage
pytest tests/ -v --cov=lalo --cov-report=term-missing

# Run specific test file
pytest tests/test_epub_parser.py -v

# Run specific test function
pytest tests/test_epub_parser.py::TestCleanHTML::test_clean_simple_paragraph -v
```

### Linting & Type Checking
```bash
# Lint with auto-fix
ruff check --fix

# Format code
ruff format

# Type check
mypy lalo/ --show-error-codes --pretty
pyright lalo/
```

### Installation
```bash
# Development mode with dependencies
pip install -e ".[dev]"

# System dependency (required for audio export)
sudo apt-get install ffmpeg
```

## Common Patterns

### Progress Callbacks
```python
def process_with_progress(
    items: list[Any],
    progress_callback: Callable[[int, int], None] | None = None
) -> None:
    """Process items with optional progress reporting."""
    total = len(items)
    for idx, item in enumerate(items):
        # Process item
        if progress_callback:
            progress_callback(idx + 1, total)
```

### Rich Console Output
```python
from rich.console import Console

console = Console()
console.print("[cyan]Info message[/cyan]")
console.print("[green]✓[/green] Success")
console.print("[yellow]⚠ Warning[/yellow]")
console.print("[red]Error:[/red] Something failed")
```

### Resource Cleanup
```python
# Always clean up GPU resources
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use context managers for files
with StreamingAudioWriter(output_path) as writer:
    writer.write_chapter(audio, title, number)
```

## Configuration

All configuration constants live in `lalo/config.py`:
- `DEFAULT_SPEAKER = "Aiden"` - Default voice profile
- `DEFAULT_LANGUAGE = "Auto"` - Auto-detect language per chapter
- `SUPPORTED_SPEAKERS` - List of 9 voice profiles
- `SUPPORTED_LANGUAGES` - List of 10 supported languages
- `TTS_CHUNK_SIZE = 2000` - Optimal chunk size for Qwen3-TTS
- `MP3_BITRATE = "192k"` - Default MP3 export quality

**Never hardcode values** - always reference config constants.

## Testing Requirements

- All new features **must** include tests
- Maintain or improve code coverage (target >90%)
- Test both success and error paths
- Use descriptive test names: `test_<what>_<condition>_<expected>`

```python
class TestEPUBParser:
    """Tests for EPUB parsing functionality."""

    def test_parse_valid_epub_returns_book(self):
        """Test parsing a valid EPUB returns Book object."""
        pass

    def test_parse_missing_file_raises_not_found(self):
        """Test parsing non-existent file raises EPUBNotFoundError."""
        pass
```

## Performance Considerations

- **GPU batching:** Process multiple text chunks in single batch (configured via `TTS_BATCH_SIZE`)
- **Chunk size:** 2000 characters optimal for Qwen3-TTS model
- **Streaming mode:** Use for books with 50+ chapters to avoid memory issues
- **Parallel processing:** Auto-detects multiple GPUs for concurrent chapter conversion

## Git Workflow

- **Main branch:** `main` (stable releases)
- **Development:** Feature branches (`feature/<name>`)
- **Commits:** Use conventional commit style
- **CI/CD:** All PRs must pass lint, type-check, tests, and build

## Key Files to Reference

- `pyproject.toml` - Dependencies, tool configuration, project metadata
- `lalo/config.py` - All configuration constants
- `lalo/exceptions.py` - Exception hierarchy
- `AGENTS.md` - Comprehensive agent guidelines
- `CONTRIBUTING.md` - Contribution guidelines

## When Suggesting Code

1. **Always** include type annotations
2. **Always** use domain-specific exceptions (from `lalo/exceptions.py`)
3. **Prefer** Rich console for user-facing output
4. **Follow** existing patterns in the codebase
5. **Keep** lines under 100 characters
6. **Use** double quotes for strings
7. **Include** docstrings for public APIs
8. **Test** your suggestions fit the existing architecture

## Example: Adding a New Feature

If adding a new audio export format:

1. **Update config** (`lalo/config.py`): Add format constant
2. **Update exceptions** (`lalo/exceptions.py`): Add specific error if needed
3. **Implement logic** (`lalo/audio_manager.py`): Add export method with types
4. **Update CLI** (`lalo/cli.py`): Add format to Click choices
5. **Add tests** (`tests/test_audio_manager.py`): Test success and error cases
6. **Update docs** (`README.md`): Document the new format

---

**For more detailed guidelines, see:**
- `AGENTS.md` - Comprehensive development guidelines
- `CONTRIBUTING.md` - Contribution workflow
- `README.md` - User documentation and examples
