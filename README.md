# Lalo - EPUB to Audiobook Converter

Convert EPUB ebooks to high-quality audiobooks using [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), an advanced text-to-speech model.

## Features

- **Multi-Language Support**: Converts EPUB files in 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
- **Premium Voice Selection**: Choose from 9 high-quality voice profiles covering various genders, ages, and languages
- **Chapter Selection**: Convert specific chapters or the entire book
- **Auto Language Detection**: Automatically detects the language of each chapter
- **Voice Control**: Optional natural language instructions for voice modulation
- **Multiple Export Formats**: Export to WAV (lossless), MP3 (compressed), or M4B (audiobook with chapter markers)
- **Chapter Markers**: M4B format includes navigable chapter bookmarks
- **Checkpoint & Resume**: Automatically saves progress after each chapter; interrupted conversions resume where they left off by simply re-running the same command

## Requirements

- **Hardware**: NVIDIA GPU with CUDA support (required)
- **Python**: 3.12 or higher
- **CUDA**: Compatible CUDA drivers installed

## Installation

### 1. Install ffmpeg (required for MP3 and M4B export)

#### Ubuntu/Debian
```bash
sudo apt-get install ffmpeg
```

### 2. Install Lalo

#### Option A: Install from PyPI (Recommended)

Using pip:
```bash
pip install lalo-tts
```

Using uv (faster):
```bash
uv pip install lalo-tts
```

#### Option B: Install from Source (Development)

Using uv (recommended):
```bash
# Clone the repository
git clone https://github.com/willianpaixao/lalo.git
cd lalo

# Sync dependencies and install in development mode (including dev tools)
uv sync --extra dev
```

Using pip:
```bash
# Clone the repository
git clone https://github.com/willianpaixao/lalo.git
cd lalo

# Install in development mode
pip install -e .
```

### 3. Install FlashAttention2 (recommended for faster inference)

```bash
pip install flash-attn --no-build-isolation
```

## Quick Start

### Basic Usage

Convert an entire EPUB to audiobook with default settings:

```bash
lalo convert mybook.epub
```

This will:
- Auto-detect the language
- Use the default speaker (Aiden - American English male)
- Convert all chapters
- Output to `mybook.mp3` (input filename with .mp3 extension)

### Advanced Examples

**Convert specific chapters:**
```bash
lalo convert mybook.epub --chapters 1-5
```

**Choose a different speaker:**
```bash
lalo convert mybook.epub --speaker Ryan
```

**Specify language explicitly:**
```bash
lalo convert mybook.epub --language Japanese --speaker Ono_Anna
```

**Custom output path and format:**
```bash
lalo convert mybook.epub --output ~/audiobooks/mybook.wav --format wav
```

**M4B Audiobook format with chapter markers:**
```bash
lalo convert mybook.epub --format m4b --output mybook.m4b
```
- Creates audiobook with navigable chapter markers
- Compatible with iTunes, Apple Books, audiobook players, VLC
- Each chapter becomes a bookmark in the player
- Includes book metadata (title, author)

**Add voice instructions:**
```bash
lalo convert mybook.epub --speaker Aiden --instruct "Speak slowly and clearly"
```

**Voice control with instructions:**
```bash
lalo convert mybook.epub --speaker Aiden --instruct "Speak slowly and clearly"
```
- Control speaking style, pace, emotion, and tone
- Uses natural language instructions
- See [Voice Instructions](#voice-instructions) section for more examples

**Streaming mode (for very long books):**
```bash
lalo convert longbook.epub --streaming
```
- Writes chapters incrementally to disk
- Recommended for books with 50+ chapters
- Crash-resilient - completed chapters are preserved

**Combine multiple options:**
```bash
lalo convert mybook.epub \
  --speaker Vivian \
  --language Chinese \
  --chapters 1,3,5-10 \
  --output ~/audiobooks/mybook.mp3 \
  --format mp3
```

## Available Speakers

List all available speakers:

```bash
lalo speakers --list
```

### Speaker Profiles

| Speaker | Description | Native Language |
|---------|-------------|-----------------|
| **Vivian** | Bright, slightly edgy young female voice | Chinese |
| **Serena** | Warm, gentle young female voice | Chinese |
| **Uncle_Fu** | Seasoned male voice with a low, mellow timbre | Chinese |
| **Dylan** | Youthful Beijing male voice with a clear, natural timbre | Chinese (Beijing Dialect) |
| **Eric** | Lively Chengdu male voice with a slightly husky brightness | Chinese (Sichuan Dialect) |
| **Ryan** | Dynamic male voice with strong rhythmic drive | English |
| **Aiden** | Sunny American male voice with a clear midrange | English (default) |
| **Ono_Anna** | Playful Japanese female voice with a light, nimble timbre | Japanese |
| **Sohee** | Warm Korean female voice with rich emotion | Korean |

**Tip**: Use each speaker's native language for best quality, though all speakers can speak any supported language.

## Supported Languages

View all supported languages:

```bash
lalo languages --list
```

Supported languages:
- Chinese
- English
- Japanese
- Korean
- German
- French
- Russian
- Portuguese
- Spanish
- Italian

## Voice Instructions

The `--instruct` flag allows you to control the voice characteristics using natural language instructions. This gives you fine-grained control over how the audiobook sounds.

### Basic Usage

```bash
lalo convert book.epub --speaker Aiden --instruct "Your instruction here"
```

### Instruction Categories

#### 1. Speaking Pace and Rhythm

Control how fast or slow the speaker reads:

```bash
# Slow and deliberate
lalo convert book.epub --instruct "Speak slowly and clearly"

# Fast-paced narration
lalo convert book.epub --instruct "Speak quickly with energy"

# Moderate, natural pace
lalo convert book.epub --instruct "Speak at a comfortable, conversational pace"

# Rhythmic and measured
lalo convert book.epub --instruct "Speak with a steady, rhythmic cadence"
```

#### 2. Emotion and Tone

Add emotional qualities to the narration:

```bash
# Happy and upbeat
lalo convert book.epub --instruct "Speak with joy and enthusiasm"

# Serious and somber
lalo convert book.epub --instruct "Speak in a serious, contemplative tone"

# Mysterious and intriguing
lalo convert book.epub --instruct "Speak mysteriously with a hint of suspense"

# Warm and friendly
lalo convert book.epub --instruct "Speak warmly as if talking to a close friend"

# Dramatic and intense
lalo convert book.epub --instruct "Speak dramatically with strong emotion"
```

#### 3. Voice Characteristics

Modify vocal qualities:

```bash
# Soft and gentle
lalo convert book.epub --instruct "Speak softly with a gentle voice"

# Clear and articulate
lalo convert book.epub --instruct "Speak clearly with precise articulation"

# Deep and resonant
lalo convert book.epub --instruct "Speak with a deep, resonant voice"

# Bright and energetic
lalo convert book.epub --instruct "Speak with a bright, energetic tone"
```

#### 4. Genre-Specific Styles

Tailor the voice to match the book's genre:

```bash
# Fiction/Novel
lalo convert novel.epub --instruct "Narrate expressively like an audiobook narrator"

# Non-fiction/Educational
lalo convert textbook.epub --instruct "Speak clearly and professionally, like a teacher"

# Mystery/Thriller
lalo convert thriller.epub --instruct "Speak with tension and suspense"

# Children's Book
lalo convert kids-book.epub --instruct "Speak playfully and animated"

# Poetry
lalo convert poems.epub --instruct "Read poetically with feeling and rhythm"

# Technical Manual
lalo convert manual.epub --instruct "Speak neutrally and clearly without emotion"
```

#### 5. Combined Instructions

Mix multiple characteristics for nuanced control:

```bash
# Fantasy epic
lalo convert fantasy.epub --instruct "Speak slowly and dramatically with a sense of wonder"

# Business book
lalo convert business.epub --instruct "Speak professionally and confidently at a moderate pace"

# Horror novel
lalo convert horror.epub --instruct "Speak quietly with tension, building suspense"

# Romance novel
lalo convert romance.epub --instruct "Speak warmly and emotionally with passion"

# Comedy
lalo convert comedy.epub --instruct "Speak playfully with good humor and lightness"
```

### Tips for Effective Instructions

1. **Be Specific**: Detailed instructions yield better results
   - ✅ Good: "Speak slowly with clear articulation and warmth"
   - ❌ Vague: "Speak nicely"

2. **Keep It Natural**: Use conversational language
   - ✅ Good: "Speak as if telling a story to friends"
   - ❌ Technical: "Increase prosodic variation by 20%"

3. **Match the Content**: Choose instructions that fit the book
   - Textbook → Clear and professional
   - Novel → Expressive and engaging
   - Poetry → Rhythmic and emotional

4. **Experiment**: Try different instructions to find what works best
   - Test a single chapter first with `--chapters 1`
   - Compare different instruction styles

5. **Language Matters**: Instructions work best in the speaker's native language
   - English speaker (Aiden, Ryan) -> English instructions
   - Chinese speaker (Vivian, Serena) -> Chinese instructions may work better

### Examples by Use Case

#### Academic Textbook
```bash
lalo convert textbook.epub \
  --speaker Aiden \
  --instruct "Speak clearly and professionally, emphasizing key concepts" \
  --format m4b
```

#### Fantasy Novel
```bash
lalo convert fantasy-novel.epub \
  --speaker Ryan \
  --instruct "Narrate dramatically with a sense of epic adventure" \
  --format m4b
```

#### Self-Help Book
```bash
lalo convert self-help.epub \
  --speaker Serena \
  --instruct "Speak warmly and encouragingly, as if coaching a friend" \
  --format mp3
```

#### Mystery Thriller
```bash
lalo convert mystery.epub \
  --speaker Uncle_Fu \
  --instruct "Speak with tension and suspense, maintaining mystery" \
  --format m4b
```

#### Children's Story
```bash
lalo convert kids-story.epub \
  --speaker Ono_Anna \
  --instruct "Speak playfully and expressively with lots of energy" \
  --format mp3
```

### Language-Specific Instructions

You can provide instructions in the target language for better results:

```bash
# Chinese book with Chinese instructions
lalo convert chinese-book.epub \
  --speaker Vivian \
  --language Chinese \
  --instruct "用温柔的语气慢慢地说"

# Japanese book with Japanese instructions
lalo convert japanese-book.epub \
  --speaker Ono_Anna \
  --language Japanese \
  --instruct "明るく元気に話してください"

# Korean book with Korean instructions
lalo convert korean-book.epub \
  --speaker Sohee \
  --language Korean \
  --instruct "따뜻하고 감정적으로 말해주세요"
```

### Notes

- Instructions are **optional** - if omitted, the speaker uses their default natural voice
- Instructions work best with the **CustomVoice** models (default)
- Results may vary based on the complexity of the instruction
- Some instructions may work better with certain speakers

## Chapter Selection Format

The `--chapters` option supports flexible selection:

- `all` - Convert all chapters (default)
- `1,3,5` - Specific chapters
- `1-5` - Range of chapters
- `1,3-5,7-10` - Combination of specific chapters and ranges

Examples:
```bash
lalo convert book.epub --chapters 1-3          # First three chapters
lalo convert book.epub --chapters 1,5,10       # Chapters 1, 5, and 10
lalo convert book.epub --chapters 1-5,10-15    # Chapters 1-5 and 10-15
```

## Checkpoint & Resume

Lalo automatically saves progress after each chapter so interrupted conversions
can be resumed by re-running the same command. No extra flags are needed.

### How It Works

1. When a conversion starts, Lalo creates a checkpoint file and caches each
   completed chapter as a WAV file under `~/.cache/lalo/` (or `$XDG_CACHE_HOME/lalo/`).
2. If the process is interrupted (Ctrl+C, GPU out-of-memory, crash), the
   checkpoint persists on disk.
3. Re-running the same `lalo convert` command auto-detects the checkpoint,
   validates the EPUB hasn't changed, skips completed chapters, and picks up
   where it left off.
4. On successful completion the checkpoint and cached audio are deleted
   automatically.

### Example

```bash
# Start a conversion — gets interrupted after 5 of 20 chapters
lalo convert moby-dick.epub --format m4b
# ^C
# ⚠ Interrupt received. Saving progress...
# Checkpoint saved. Resume with:
#   lalo convert moby-dick.epub

# Resume — just re-run the same command
lalo convert moby-dick.epub --format m4b
# Resuming: 5 chapter(s) already completed, 15 remaining
# ...
# ✓ Conversion complete!
```

### Skipping Resume

To ignore an existing checkpoint and start from scratch:

```bash
lalo convert mybook.epub --no-resume
```

### Managing the Cache

List all cached checkpoints:

```bash
lalo cache list
```

Remove stale checkpoints (default: older than 30 days):

```bash
lalo cache clean
```

Remove checkpoints older than 7 days:

```bash
lalo cache clean --days 7
```

Remove all checkpoints:

```bash
lalo cache clean --all
```

## CLI Commands

### `lalo convert`

Convert an EPUB file to audiobook.

```
Usage: lalo convert [OPTIONS] EPUB_FILE

Options:
  -s, --speaker TEXT            Speaker voice (default: Aiden)
  -l, --language TEXT           Language for TTS (default: Auto)
  -c, --chapters TEXT           Chapters to convert (default: all)
  -o, --output PATH             Output file path
  -f, --format [wav|mp3|m4b]    Output audio format (default: mp3)
  -i, --instruct TEXT           Voice control instruction
  --streaming                   Enable streaming mode (incremental saving)
  --parallel / --no-parallel    Enable parallel chapter processing (default: auto-detect GPUs)
  --max-parallel INTEGER        Maximum parallel chapters (default: auto-detect)
  --no-resume                   Ignore existing checkpoint and start fresh
  --help                        Show this message and exit
```

### `lalo inspect`

Inspect an EPUB file and list all chapters.

```
Usage: lalo inspect [OPTIONS] EPUB_FILE

Options:
  --help  Show this message and exit
```

Example:
```bash
lalo inspect mybook.epub
```

This displays the book's metadata (title and author) and a list of all chapters with their numbers and titles. Useful for previewing a book's structure before conversion or for selecting specific chapters to convert.

### `lalo speakers`

Show information about available speakers.

```
Usage: lalo speakers [OPTIONS]

Options:
  --list  List all available speakers
  --help  Show this message and exit
```

### `lalo languages`

Show information about supported languages.

```
Usage: lalo languages [OPTIONS]

Options:
  --list  List all supported languages
  --help  Show this message and exit
```

### `lalo cache list`

List all cached checkpoints with progress, age, and disk usage.

```
Usage: lalo cache list [OPTIONS]

Options:
  --help  Show this message and exit
```

### `lalo cache clean`

Remove stale or all cached checkpoints.

```
Usage: lalo cache clean [OPTIONS]

Options:
  --days INTEGER  Remove checkpoints older than N days (default: 30)
  --all           Remove all checkpoints regardless of age
  --help          Show this message and exit
```

## Merging M4B Files

If you converted a book in parts (e.g., chapters 1-10, 11-20, 21-30), you can merge them into a single M4B file while preserving all chapter markers:

```bash
./scripts/merge_m4b.sh complete_book.m4b part1.m4b part2.m4b part3.m4b
```

**Features**:
- Preserves all chapter markers with correct timestamps
- Uses first file's metadata as source of truth
- Warns about metadata mismatches
- Fast concatenation without re-encoding
- Works with any M4B files (not just Lalo-generated)

**Requirements**:
- `jq` (for JSON processing): `sudo apt-get install jq` or `brew install jq`
- `bc` (for precise duration arithmetic): `sudo apt-get install bc` or `brew install bc`
- `realpath` (for resolving absolute file paths; usually part of `coreutils`): `sudo apt-get install coreutils` or `brew install coreutils`

**Preview before merging**:
```bash
./scripts/merge_m4b.sh --dry-run output.m4b part1.m4b part2.m4b
```

**Options**:
- `-v, --verbose` - Show detailed ffmpeg output
- `-n, --dry-run` - Preview without actually merging
- `-f, --force` - Overwrite output file if it exists
- `-k, --keep-temp` - Keep temporary files for debugging
- `-h, --help` - Show help message

## Configuration

Default settings can be customized in `lalo/config.py`:

- `MODEL_NAME`: Qwen3-TTS model to use
- `DEFAULT_SPEAKER`: Default speaker (Aiden)
- `DEFAULT_LANGUAGE`: Default language detection mode (Auto)
- `DEFAULT_FORMAT`: Default output format (mp3)
- `MP3_BITRATE`: MP3 bitrate (192k)
- `CHECKPOINT_ENABLED`: Enable automatic checkpoint/resume (default: True)
- `CHECKPOINT_CACHE_DIR`: Cache directory override (default: `~/.cache/lalo` or `$XDG_CACHE_HOME/lalo`)
- `CHECKPOINT_STALE_DAYS`: Days before `lalo cache clean` considers a checkpoint stale (default: 30)

## How It Works

1. **EPUB Parsing**: Extracts book metadata and chapter content while preserving structure
2. **Language Detection**: Auto-detects language per chapter using `langdetect`
3. **Checkpoint Init**: Computes a SHA-256 hash of the EPUB and creates (or resumes from) a checkpoint in `~/.cache/lalo/`
4. **Text Processing**: Chunks long chapters into manageable segments (optimized to 2000 chars)
5. **TTS Generation**: Converts text to speech using Qwen3-TTS-12Hz-1.7B-CustomVoice with GPU batch processing; the checkpoint is updated after each chapter
6. **Audio Export**:
   - Intermediate chapter WAV files are stored in the cache directory for crash resilience
   - On completion, chapters are concatenated and exported to the target format (WAV, MP3, or M4B)
   - The checkpoint and cached audio are cleaned up automatically

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

### Development Setup

Using uv (recommended):
```bash
# Clone repository
git clone https://github.com/willianpaixao/lalo.git
cd lalo

# Sync all dependencies (including dev extra)
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run linting
uv run ruff check --fix
uv run ruff format

# Run type checking
uv run mypy lalo/
```

Using pip:
```bash
# Clone repository
git clone https://github.com/willianpaixao/lalo.git
cd lalo

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check --fix
ruff format

# Run type checking
mypy lalo/
```

See [Testing Guide](docs/TESTING.md) for comprehensive testing instructions.
