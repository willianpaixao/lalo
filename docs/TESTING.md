# Testing Guide

This guide covers how to run and write tests for Lalo, including both unit tests and end-to-end tests.

## Table of Contents

- [Quick Start](#quick-start)
- [Running Unit Tests](#running-unit-tests)
- [Running End-to-End Tests](#running-end-to-end-tests)

## Quick Start

### Prerequisites

```bash
# 1. Install development dependencies
pip install -e ".[dev]"

# 2. Install ffmpeg (required for E2E tests)
sudo apt-get install ffmpeg

# 3. Verify GPU (required for E2E tests)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Run All Tests

```bash
# Run all tests (unit + integration, no E2E)
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=lalo --cov-report=term-missing

# Run all tests including E2E (requires GPU, ~5-10 min default, ~15-20 min with --extended)
pytest tests/e2e/ -v -m gpu

# Run all E2E tests including extended tests
pytest tests/e2e/ --extended -v -m gpu
```

## Running Unit Tests

Unit tests validate individual components in isolation using mocks.

### Run All Unit Tests

```bash
# Run all unit tests (fast, no GPU)
pytest tests/ -v --ignore=tests/e2e

# With coverage report
pytest tests/ -v --cov=lalo --cov-report=html --ignore=tests/e2e
# View coverage: open htmlcov/index.html
```

### Run Specific Test Files

```bash
# Test EPUB parser
pytest tests/test_epub_parser.py -v

# Test audio manager
pytest tests/test_audio_manager.py -v

# Test TTS engine
pytest tests/test_tts_engine.py -v

# Test exception handling
pytest tests/test_exceptions.py -v

# Test utilities
pytest tests/test_utils.py -v
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/test_epub_parser.py::TestCleanHTML -v

# Run specific test function
pytest tests/test_epub_parser.py::TestCleanHTML::test_clean_simple_paragraph -v

# Run tests matching a pattern
pytest tests/ -k "epub" -v
```

### Coverage Reports

```bash
# Terminal coverage report
pytest tests/ --cov=lalo --cov-report=term-missing

# HTML coverage report (detailed)
pytest tests/ --cov=lalo --cov-report=html
open htmlcov/index.html

# XML coverage report (for CI)
pytest tests/ --cov=lalo --cov-report=xml
```

## Running End-to-End Tests

End-to-end tests validate the complete conversion pipeline using real TTS engine and real EPUB files.

### Prerequisites for E2E Tests

```bash
# 1. Ensure GPU is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 2. Ensure ffmpeg is installed
ffmpeg -version
ffprobe -version

# 3. EPUB test files should be present
ls tests/e2e/fixtures/epubs/
# Should show: pg1513-images-3.epub, pg2591-images-3.epub, pg2701-images-3.epub, pg54829-images-3.epub
```

### Quick Smoke Test (No GPU)

```bash
# Test CLI commands without GPU (7 seconds)
pytest tests/e2e/test_e2e_cli_basic.py -v
```

### Run All E2E Tests

```bash
# Default E2E suite (requires GPU, ~5-10 minutes)
pytest tests/e2e/ -v -m gpu

# Extended E2E suite (requires GPU, ~15-20 minutes)
pytest tests/e2e/ --extended -v -m gpu

# Keep output audio files for inspection
pytest tests/e2e/ -v -m gpu --keep-outputs
# Outputs saved to: tests/output/<test_name>/

# Run only extended tests
pytest tests/e2e/ --extended -v -m "gpu and extended"
```

### Run E2E Tests by Category

```bash
# CLI commands (10 tests, no GPU, ~1 min)
pytest tests/e2e/test_e2e_cli_basic.py -v

# Format conversion (7 tests, GPU, ~1-2 min)
pytest tests/e2e/test_e2e_convert_formats.py -v -m gpu

# Streaming and chapter selection (12 tests, GPU, ~2-3 min)
pytest tests/e2e/test_e2e_convert_modes.py -v -m gpu

# Parallel processing (8 tests, GPU, ~2-3 min)
pytest tests/e2e/test_e2e_convert_parallel.py -v -m gpu

# Error handling (11 tests, mixed, ~1-2 min)
pytest tests/e2e/test_e2e_error_handling.py -v

# Multi-language support (3 tests, GPU, ~1 min)
pytest tests/e2e/test_e2e_languages.py -v -m gpu
```

### Run Specific E2E Tests

```bash
# Test MP3 conversion
pytest tests/e2e/test_e2e_convert_formats.py::TestConvertFormats::test_convert_to_mp3 -v -m gpu

# Test streaming mode
pytest tests/e2e/test_e2e_convert_modes.py::TestConvertModes::test_convert_streaming_mode -v -m gpu

# Test chapter selection
pytest tests/e2e/ -k "chapter_selection" -v -m gpu

# Test parallel processing
pytest tests/e2e/test_e2e_convert_parallel.py::TestConvertParallel::test_convert_parallel_auto -v -m gpu
```

### Inspect Test Outputs

```bash
# Run tests and keep outputs
pytest tests/e2e/ -v -m gpu --keep-outputs
ls tests/output/
```

## Test Organization

### Pytest Markers

Tests are organized using pytest markers:

```python
@pytest.mark.gpu          # Requires GPU (auto-skips if no CUDA)
@pytest.mark.slow         # Takes >30 seconds
@pytest.mark.e2e          # End-to-end integration test
@pytest.mark.extended     # Requires --extended flag (disabled by default for faster runs)
```

Run tests by marker:

```bash
# GPU tests only
pytest tests/ -m gpu -v

# Slow tests only
pytest tests/ -m "gpu and slow" -v

# E2E tests only
pytest tests/ -m e2e -v

# Exclude GPU tests (for CI without GPU)
pytest tests/ -m "not gpu" -v

# Extended tests only
pytest tests/e2e/ --extended -m "gpu and extended" -v
```

### Test Fixtures

Common fixtures available across tests:

#### Unit Test Fixtures (tests/conftest.py)
- Standard pytest fixtures
- Mock objects for TTS engine, audio manager

#### Integration Test Fixtures (tests/integration/conftest.py)
- `cli_runner` - Click CLI test runner
- `mock_tts_engine` - Realistic TTS mock
- `simple_test_epub` - Generated 3-chapter EPUB
- `unicode_test_epub` - EPUB with Japanese/Chinese content
- `epub_no_metadata` - EPUB without metadata

#### E2E Test Fixtures (tests/e2e/conftest.py)
- `romeo_juliet_epub` - Romeo & Juliet (10 chapters, 27k words) **[Disabled by default, use --extended]**
- `grimms_fairy_tales_epub` - Grimm's Fairy Tales (65 chapters, 100k words)
- `moby_dick_epub` - Moby Dick (12 chapters, 213k words)
- `portuguese_epub` - Memórias Póstumas (5 chapters, 63k words, Portuguese)
- `output_dir` - Output directory with optional preservation
- `check_ffmpeg` - Ensure ffmpeg available (auto-skip if missing)

### Local Pre-Commit Testing

Before committing code:

```bash
# 1. Run linting
ruff check --fix
ruff format

# 2. Run type checking
mypy lalo/ --show-error-codes --pretty

# 3. Run unit tests
pytest tests/ -v --ignore=tests/e2e

# 4. (Optional) Quick E2E smoke test
pytest tests/e2e/test_e2e_cli_basic.py -v
```

### Pre-Release Testing

Before releasing a new version:

```bash
# 1. Run full test suite including E2E (extended)
pytest tests/e2e/ --extended -v -m gpu --keep-outputs

# 2. Manually verify audio outputs
ls tests/output/
# Listen to generated files

# 3. Check coverage
pytest tests/ --cov=lalo --cov-report=html
open htmlcov/index.html

# 4. Build and check distribution
python -m build
twine check dist/*
```

## Troubleshooting

### Common Issues

#### GPU Not Available

**Problem**: E2E tests skipped with "GPU required"

**Solution**:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### ffmpeg Not Found

**Problem**: E2E tests skipped with "ffmpeg required"

**Solution**:
```bash
# Install ffmpeg
sudo apt-get install ffmpeg

# Verify installation
ffmpeg -version
ffprobe -version
```

#### Out of Memory (OOM)

**Problem**: GPU runs out of memory during E2E tests

**Solution**:
```bash
# Reduce parallel workers
pytest tests/e2e -v -m gpu --max-parallel 1

# Use smaller EPUB
pytest tests/e2e/test_e2e_convert_formats.py -v -m gpu

# Close other GPU applications
```

#### Tests Taking Too Long

**Problem**: E2E tests take too long

**Solution**:
```bash
# Run default suite (~5-10 min)
pytest tests/e2e/ -v -m gpu

# Run specific tests instead of full suite
pytest tests/e2e/test_e2e_cli_basic.py -v

# Skip extended tests (disabled by default, but explicitly exclude)
pytest tests/e2e/ -v -m "gpu and not extended"

# Test fewer chapters in EPUBs (if writing new tests)
# Tests already optimized to use smallest chapters
```

#### Import Errors

**Problem**: `ModuleNotFoundError` when running tests

**Solution**:
```bash
# Install package in editable mode
pip install -e ".[dev]"

# Verify installation
python -c "import lalo; print(lalo.__version__)"
```

#### Coverage Not Working

**Problem**: Coverage report shows 0% or errors

**Solution**:
```bash
# Install pytest-cov
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=lalo --cov-report=term-missing
```

### Debug Failed Tests

```bash
# Run with verbose output
pytest tests/test_my_feature.py -vv

# Show print statements
pytest tests/test_my_feature.py -v -s

# Show full traceback
pytest tests/test_my_feature.py -v --tb=long

# Drop into debugger on failure
pytest tests/test_my_feature.py -v --pdb

# Keep test outputs for inspection
pytest tests/e2e/ -v -m gpu --keep-outputs
ls tests/output/
```
