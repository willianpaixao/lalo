# End-to-End Testing Suite for Lalo

This directory contains comprehensive end-to-end (E2E) tests for the Lalo EPUB to Audiobook converter.

## Overview

The E2E test suite validates the complete conversion pipeline using:
- **Real TTS engine** (requires GPU)
- **Real EPUB files** (Project Gutenberg books)
- **All output formats** (MP3, WAV, M4B)
- **All major features** (streaming, parallel processing, chapter selection)

## Test Organization

### Test Files

| File | Description | Test Count | Default Time | Extended Time |
|------|-------------|------------|--------------|---------------|
| `test_e2e_cli_basic.py` | CLI commands (version, help, speakers, languages, inspect) | 10 | ~7s | ~7s |
| `test_e2e_convert_formats.py` | Output formats (mp3, wav, m4b), chapter markers | 7 | ~1-2min | ~3-4min |
| `test_e2e_convert_modes.py` | Streaming vs non-streaming, chapter selection | 12 | ~2-3min | ~4-6min |
| `test_e2e_convert_parallel.py` | Parallel vs sequential processing | 8 | ~2-3min | ~4-6min |
| `test_e2e_error_handling.py` | Error scenarios and edge cases | 11 | ~1-2min | ~1-2min |
| `test_e2e_languages.py` | Multi-language support (Portuguese) | 3 | ~1min | ~1min |

**Total**: 51 tests
- **Default mode** (~35 tests): ~5-10 minutes (extended tests skipped)
- **Extended mode** (51 tests): ~15-20 minutes (all tests including Romeo & Juliet)

### Test EPUBs

Located in `fixtures/epubs/`:

- **pg1513-images-3.epub** - Romeo and Juliet (10 chapters, ~27k words) **[Extended tests only]**
- **pg2591-images-3.epub** - Grimm's Fairy Tales (65 chapters, ~100k words)
- **pg2701-images-3.epub** - Moby Dick (12 chapters, ~213k words)
- **pg54829-images-3.epub** - Memórias Póstumas de Brás Cubas (5 chapters, ~63k words, Portuguese)

## Running Tests

### Prerequisites

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Install system dependencies
sudo apt-get install ffmpeg

# 3. Ensure GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Running All E2E Tests

```bash
# Run default E2E tests (requires GPU, ~5-10 minutes, extended tests skipped)
pytest tests/e2e/ -v -m gpu

# Run extended E2E tests (requires GPU, ~15-20 minutes, all tests including Romeo)
pytest tests/e2e/ --extended -v -m gpu

# Run with coverage
pytest tests/e2e/ -v -m gpu --cov=lalo

# Keep test outputs for inspection
pytest tests/e2e/ -v -m gpu --keep-outputs
# Outputs saved to: tests/output/<test_name>/
```

### Running Specific Test Files

```bash
# Basic CLI tests (no GPU required)
pytest tests/e2e/test_e2e_cli_basic.py -v

# Format conversion tests
pytest tests/e2e/test_e2e_convert_formats.py -v -m gpu

# Streaming mode tests
pytest tests/e2e/test_e2e_convert_modes.py -v -m gpu

# Parallel processing tests
pytest tests/e2e/test_e2e_convert_parallel.py -v -m gpu

# Error handling tests
pytest tests/e2e/test_e2e_error_handling.py -v
```

### Running Specific Tests

```bash
# Run single test
pytest tests/e2e/test_e2e_convert_formats.py::TestConvertFormats::test_convert_to_mp3 -v -m gpu

# Run tests matching pattern
pytest tests/e2e/ -k "streaming" -v -m gpu

# Run slow tests only
pytest tests/e2e/ -m "gpu and slow" -v

# Run only extended tests
pytest tests/e2e/ --extended -m "gpu and extended" -v

# Exclude extended tests (same as default)
pytest tests/e2e/ -m "gpu and not extended" -v
```

### Test Markers

- `@pytest.mark.gpu` - Requires GPU (skipped if CUDA unavailable)
- `@pytest.mark.slow` - Takes >30 seconds
- `@pytest.mark.e2e` - End-to-end integration test
- `@pytest.mark.extended` - Requires --extended flag (disabled by default for faster runs)

## Test Coverage

### Features Tested

✅ **CLI Commands**
- Version display
- Help text
- Speaker listing
- Language listing
- EPUB inspection

✅ **Output Formats**
- MP3 conversion with correct bitrate
- WAV conversion (uncompressed)
- M4B conversion with chapter markers
- Chapter metadata validation

✅ **Conversion Modes**
- Standard (non-streaming) mode
- Streaming mode (memory efficient)
- Streaming vs non-streaming equivalence
- Large book handling (Moby Dick)

✅ **Chapter Selection**
- All chapters (`--chapters all`)
- Range selection (`--chapters 1-5`)
- Specific chapters (`--chapters 1,3,5`)
- Mixed selection (`--chapters 1-3,5,7-9`)
- Single chapter

✅ **Parallel Processing**
- Sequential mode (`--no-parallel`)
- Parallel auto-detection
- Custom worker limit (`--max-parallel`)
- Parallel vs sequential equivalence
- Many chapters (65 chapters)

✅ **Error Handling**
- Missing EPUB file
- Invalid EPUB file
- Invalid speaker
- Invalid format
- Invalid chapter selection
- Permission denied
- Empty EPUB
- Edge cases (short text, special characters)

✅ **Multi-Language Support**
- Portuguese language conversion
- Portuguese-specific speakers
- Language validation

✅ **Audio Quality**
- File creation validation
- Sample rate verification (24kHz)
- Codec validation
- Duration checks
- Silence detection
- Clipping detection
- Bitrate validation

## Validation Strategy

### Audio Validation

Tests use `ffprobe` and `soundfile` to validate:

1. **File Properties** (`validate_audio_file`)
   - Duration
   - Sample rate (24kHz)
   - Codec (mp3, pcm, aac)
   - Bitrate (MP3: ~192kbps)
   - File size

2. **Audio Quality** (`check_audio_quality`)
   - Non-empty audio
   - No complete silence
   - No clipping
   - No long silence periods

3. **M4B Chapter Markers** (`validate_m4b_chapters`)
   - Chapter count matches expected
   - Chapter titles present
   - Chapter timestamps valid
   - Metadata embedded (title, author)

### Expected Behavior

| Test Scenario | Expected Result |
|---------------|-----------------|
| Valid EPUB conversion | Exit code 0, audio file created, valid properties |
| Invalid speaker | Exit code != 0, error message shown |
| Missing file | Exit code != 0, caught by Click |
| Streaming mode | Same quality as non-streaming, lower memory |
| Parallel mode | Same quality as sequential, potentially faster |
| M4B chapters | Chapter markers embedded, accessible via ffprobe |

## CI/CD Integration

### Automatic E2E Test Exclusion in CI

E2E tests are **automatically skipped in CI** using two mechanisms:

1. **Directory-level exclusion**: `--ignore=tests/e2e`
2. **Marker-based skipping**: `-m "not gpu"` skips `@pytest.mark.gpu` tests

This ensures CI pipeline:
- ✅ Runs fast (no GPU tests)
- ✅ Doesn't fail due to missing GPU
- ✅ Still tests all unit and integration tests
- ✅ Provides coverage reports

### GitHub Actions Configuration

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run tests (exclude E2E/GPU tests)
        run: |
          # Exclude E2E tests and skip GPU-marked tests
          pytest tests/ -v --cov=lalo \
            --ignore=tests/e2e \
            -m "not gpu"
```

### Local Development Workflow

```bash
# Before committing
pytest tests/e2e/test_e2e_cli_basic.py -v  # Quick smoke test (no GPU)

# Before releasing
pytest tests/e2e/test_e2e_*.py -v -m gpu --keep-outputs  # Full E2E suite

# Check outputs
ls tests/output/  # Inspect generated audio files
```

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Tests will be skipped if no GPU
pytest tests/e2e/test_e2e_convert_formats.py -v
# Output: SKIPPED [1] ... GPU required for this test (CUDA not available)
```

### ffmpeg Not Found

```bash
# Install ffmpeg
sudo apt-get install ffmpeg

# Verify
ffmpeg -version
ffprobe -version
```

### Tests Fail with OOM

If GPU runs out of memory:

1. Reduce parallel workers: `--max-parallel 1`
2. Use smaller EPUB (Romeo & Juliet instead of Moby Dick)
3. Close other GPU applications

### Keep Test Outputs

```bash
# Save outputs for debugging
pytest tests/e2e/test_e2e_*.py -v -m gpu --keep-outputs

# Outputs in tests/output/<test_name>/
ls tests/output/
```

## Adding New Tests

### Test Structure

```python
@pytest.mark.e2e
@pytest.mark.gpu  # If requires GPU
@pytest.mark.slow  # If takes >30s
class TestMyFeature:
    def test_my_scenario(self, cli_runner, romeo_juliet_epub, output_dir, check_ffmpeg):
        """Test description."""
        output_file = output_dir / "output.mp3"

        result = cli_runner.invoke(main, [
            "convert",
            str(romeo_juliet_epub),
            "--chapters", "3",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Validate audio
        from tests.e2e.utils import validate_audio_file
        audio_info = validate_audio_file(output_file, "mp3")
        assert audio_info["duration"] > 0
```

### Fixtures Available

- `cli_runner` - Click CLI test runner
- `romeo_juliet_epub` - Small/medium EPUB (10 chapters)
- `grimms_fairy_tales_epub` - Large EPUB (65 chapters)
- `moby_dick_epub` - Very large EPUB (213k words)
- `output_dir` - Temporary output directory (auto-cleanup)
- `check_ffmpeg` - Ensure ffmpeg available (auto-skip if missing)

### Utility Functions

```python
from tests.e2e.utils import (
    validate_audio_file,      # Get audio properties
    check_audio_quality,       # Check for silence/clipping
    validate_m4b_chapters,     # Validate chapter markers
    get_audio_duration,        # Get duration only
    estimate_expected_duration, # Estimate from text
    parse_cli_output,          # Parse CLI output
)
```

## Performance Benchmarks

Expected test execution times (single GPU RTX 3090):

| Test | EPUB | Chapters | Format | Mode | Time |
|------|------|----------|--------|------|------|
| Basic MP3 | Romeo | 2 | MP3 | Standard | ~1-2 min |
| WAV conversion | Romeo | 2 | WAV | Standard | ~1-2 min |
| M4B chapters | Romeo | 3 | M4B | Standard | ~2-3 min |
| Streaming mode | Romeo | 3 | MP3 | Streaming | ~2-3 min |
| Large book | Moby Dick | 2 | MP3 | Streaming | ~5-10 min |
| Many chapters | Grimms | 10 | MP3 | Parallel | ~3-5 min |
| Full suite (default) | All | Various | All | All | ~5-10 min |
| Full suite (--extended) | All | Various | All | All | ~15-20 min |

## Known Issues

1. **Chapter marker detection**: M4B chapter marker validation may vary depending on ffmpeg version and mutagen library. Tests check for presence but may not catch all metadata.

2. **Duration variance**: TTS output duration can vary slightly between runs (~2-5%) due to model randomness. Tests allow 5% tolerance.

3. **Parallel processing**: With single GPU, parallel mode may fall back to sequential with batching. Tests validate correctness, not necessarily performance improvement.

## Future Enhancements

- [ ] Add performance benchmarking tests
- [ ] Test multi-GPU parallel processing
- [ ] Add memory usage validation
- [ ] Test resume/interruption scenarios
- [ ] Add stress tests (100+ chapters)
- [ ] Test custom voice instructions
- [ ] Add audio quality metrics (MOS, SNR)
- [ ] Test non-English languages with appropriate speakers
