# Agent Guidelines for Lalo

This document provides coding standards and development guidelines for AI agents working on the Lalo EPUB to Audiobook Converter project.

## Project Overview

Lalo is a Python 3.12+ CLI tool that converts EPUB files to audiobooks using Qwen3-TTS, supporting 10 languages and 9 voice profiles. The codebase emphasizes type safety, comprehensive error handling, and GPU-accelerated batch processing.

## Build, Lint, and Test Commands

### Installation
```bash
# Development installation with all dependencies
pip install -e ".[dev]"

# Install system dependencies (required for audio export)
sudo apt-get install ffmpeg  # Ubuntu/Debian
```

### Linting and Formatting
```bash
# Run ruff linter with auto-fix
ruff check --fix

# Check formatting (without applying)
ruff format --check --diff

# Apply formatting
ruff format

# Run pre-commit hooks on all files
pre-commit run --all-files
```

### Type Checking
```bash
# Run mypy type checker
mypy lalo/ --show-error-codes --pretty

# Run pyright type checker
pyright lalo/
```

### Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=lalo --cov-report=term-missing

# Run specific test file
pytest tests/test_epub_parser.py -v

# Run specific test function
pytest tests/test_epub_parser.py::TestCleanHTML::test_clean_simple_paragraph -v

# Run specific test class
pytest tests/test_epub_parser.py::TestCleanHTML -v

# Run tests matching a pattern
pytest tests/ -k "epub" -v

# Run integration tests only
pytest tests/integration/ -v

# Run tests without coverage (faster)
pytest tests/ -v

# Generate HTML coverage report
pytest tests/ --cov=lalo --cov-report=html
# View at htmlcov/index.html
```

### Building
```bash
# Build distribution packages
python -m build

# Check package quality
twine check dist/*
```

## Code Style Guidelines

### Import Organization
Imports must be organized in the following order, with blank lines between groups:

1. **Standard library imports** (alphabetically sorted)
2. **Third-party imports** (alphabetically sorted)
3. **Local application imports** (alphabetically sorted)

```python
# Standard library
import logging
import os
import signal
from pathlib import Path
from typing import Any

# Third-party
import click
import numpy as np
import torch
from rich.console import Console

# Local
from lalo.config import DEFAULT_SPEAKER, SUPPORTED_SPEAKERS
from lalo.exceptions import AudioError, EPUBError, TTSError
from lalo.utils import format_duration
```

**Special cases:**
- Environment configuration imports may appear before third-party imports if they modify behavior
- Suppress warnings/logging before imports that generate them

### Formatting

**Tooling:** Ruff formatter (compatible with Black)
- **Line length:** 100 characters maximum
- **Quotes:** Double quotes for strings (`"string"`)
- **Indentation:** 4 spaces (no tabs)
- **Trailing commas:** Respect magic trailing comma

### Type Annotations

**Required:**
- All function signatures (parameters and return types)
- Class attributes when type is not obvious
- Use modern syntax: `list[str]`, `dict[str, int]`, `str | None` (not `Optional[str]`)

```python
# Good
def parse_chapter_selection(selection: str, total_chapters: int) -> list[int]:
    """Parse chapter selection string."""
    pass

# Good - Union with None
def get_speaker(name: str | None = None) -> str:
    pass

# Bad - missing types
def parse_chapters(selection, total):
    pass
```

**Type hints configuration:**
- `mypy` checks with `warn_return_any = true`
- `pyright` enabled for stricter checking
- Suppress third-party import warnings: `disable_error_code = ["import-untyped"]`

### Naming Conventions

- **Functions/variables:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private/internal:** Prefix with `_`
- **Type variables:** `TitleCase` (e.g., `AudioSegment`)

```python
# Constants
DEFAULT_SPEAKER = "Aiden"
TTS_CHUNK_SIZE = 2000

# Classes
class EPUBParser:
    pass

# Functions and variables
def format_duration(seconds: float) -> str:
    pass

# Private methods
def _chunk_text(text: str) -> list[str]:
    pass
```

### Docstrings

Use triple-quoted docstrings for all public modules, classes, and functions:

```python
"""
Module-level docstring explaining purpose.
"""

def convert_to_audio(text: str, speaker: str) -> np.ndarray:
    """
    Convert text to audio using TTS.

    Args:
        text: Input text to convert
        speaker: Speaker voice name

    Returns:
        Audio samples as numpy array

    Raises:
        TTSError: If conversion fails
    """
    pass
```

### Error Handling

**Custom exception hierarchy:**
```
LaloError (base)
├── EPUBError
│   ├── EPUBNotFoundError
│   ├── EPUBInvalidError
│   ├── EPUBParseError
│   └── EPUBNoChaptersError
├── TTSError
│   ├── GPUNotAvailableError
│   ├── TTSModelLoadError
│   ├── UnsupportedLanguageError
│   └── UnsupportedSpeakerError
├── AudioError
│   ├── AudioExportError
│   ├── UnsupportedAudioFormatError
│   ├── EmptyAudioError
│   └── FFmpegNotFoundError
└── ValidationError
    ├── InvalidChapterSelectionError
    └── InvalidFilePathError
```

**Best practices:**
- Always use domain-specific exceptions (never bare `Exception`)
- Include context in exception messages
- Chain exceptions with `from e` to preserve stack traces
- Handle exceptions at appropriate levels (user-facing in CLI, propagate in libraries)

```python
# Good - specific exception with context
try:
    book = parse_epub(file_path)
except EPUBParseError as e:
    console.print(f"[red]Failed to parse EPUB:[/red] {e}")
    raise click.Abort() from e

# Bad - bare exception
except Exception:
    raise
```

## Code Organization

### Module Structure
```
lalo/
├── __init__.py          # Package initialization, version
├── cli.py               # Click-based CLI interface
├── config.py            # Configuration constants
├── exceptions.py        # Custom exception classes
├── epub_parser.py       # EPUB parsing logic
├── tts_engine.py        # TTS engine wrapper
├── audio_manager.py     # Audio processing and export
├── parallel_processor.py # Multi-GPU parallel processing
└── utils.py             # Utility functions

tests/
├── __init__.py
├── conftest.py          # Pytest fixtures
├── test_*.py            # Unit tests (mirror source structure)
└── integration/         # Integration tests
    ├── __init__.py
    ├── conftest.py
    └── test_*.py
```

### Configuration Management
- All configuration in `lalo/config.py`
- Environment variables for runtime configuration
- No hardcoded values in logic files

## Testing Standards

**Framework:** pytest with pytest-cov

**Requirements:**
- All new features must include tests
- Maintain or improve code coverage
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

**Fixtures:** Use `conftest.py` for shared fixtures

**Allowed in tests (per ruff config):**
- `assert` statements (S101)
- Magic values (PLR2004)
- Unused arguments (ARG)

## Git Workflow

**Branch strategy:**
- `main`: Stable releases
- `develop`: Integration branch
- Feature branches: `feature/<name>`

**Commit messages:**
- Use conventional commits style
- Be concise but descriptive
- Focus on "why" over "what"

**CI/CD:**
- All PRs must pass: lint, type-check, tests, build
- Coverage reports uploaded to Codecov
- Releases automated via GitHub Actions

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

### Rich Console Output
```python
from rich.console import Console

console = Console()
console.print("[cyan]Info message[/cyan]")
console.print("[green]✓[/green] Success")
console.print("[yellow]⚠ Warning[/yellow]")
console.print("[red]Error:[/red] Something failed")
```

## Performance Considerations

- **GPU batching:** Process multiple text chunks in single batch (configured via `TTS_BATCH_SIZE`)
- **Chunk size:** 2000 characters optimal for Qwen3-TTS
- **Streaming mode:** Use for books with 50+ chapters to avoid memory issues
- **Parallel processing:** Auto-detects GPUs for multi-chapter parallel conversion

## Key Files to Review

Before making changes, review:
- `pyproject.toml` - Dependencies and tool configuration
- `lalo/config.py` - All configuration constants
- `lalo/exceptions.py` - Exception hierarchy
- `.github/workflows/ci.yml` - CI pipeline

## Additional Notes

- **Python version:** Minimum 3.12 (uses modern type hints)
- **GPU requirement:** CUDA-compatible GPU required for inference
- **Pre-commit hooks:** Ruff linting and mypy enabled
- **Ignored rules:** E402 (module imports), T201 (print in CLI), PLR0913 (many args), PLR2004 (magic values)
