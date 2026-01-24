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

```bash
pip install lalo-tts
```

#### Option B: Install from Source (Development)

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
- Output to `<book_title>.mp3`

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

## CLI Commands

### `lalo convert`

Convert an EPUB file to audiobook.

```
Usage: lalo convert [OPTIONS] EPUB_FILE

Options:
  -s, --speaker TEXT        Speaker voice (default: Aiden)
  -l, --language TEXT       Language for TTS (default: Auto)
  -c, --chapters TEXT       Chapters to convert (default: all)
  -o, --output PATH         Output file path
  -f, --format [wav|mp3|m4b] Output audio format (default: mp3)
  -i, --instruct TEXT       Voice control instruction
  --streaming               Enable streaming mode (incremental saving)
  --help                    Show this message and exit
```

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

## Configuration

Default settings can be customized in `lalo/config.py`:

- `MODEL_NAME`: Qwen3-TTS model to use
- `DEFAULT_SPEAKER`: Default speaker (Aiden)
- `DEFAULT_LANGUAGE`: Default language detection mode (Auto)
- `DEFAULT_FORMAT`: Default output format (mp3)
- `MP3_BITRATE`: MP3 bitrate (192k)

## How It Works

1. **EPUB Parsing**: Extracts book metadata and chapter content while preserving structure
2. **Language Detection**: Auto-detects language per chapter using `langdetect`
3. **Text Processing**: Chunks long chapters into manageable segments (optimized to 2000 chars)
4. **TTS Generation**: Converts text to speech using Qwen3-TTS-12Hz-1.7B-CustomVoice with GPU batch processing
5. **Audio Export**:
   - **Regular mode**: Concatenates all chapters in memory, then exports
   - **Streaming mode**: Writes each chapter to disk immediately, then combines at end

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.
