"""
Configuration constants and default settings for Lalo.
"""

from typing import Any

# Model Configuration
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_DTYPE = "bfloat16"
MODEL_DEVICE = "cuda:0"
USE_FLASH_ATTENTION = True

# Default Settings
DEFAULT_SPEAKER = "Aiden"
DEFAULT_LANGUAGE = "Auto"
DEFAULT_FORMAT = "mp3"
DEFAULT_OUTPUT = "audiobook.mp3"

# Audio Settings
AUDIO_SAMPLE_RATE = 24000
MP3_BITRATE = "192k"
M4B_BITRATE = "192k"

# Supported Formats
SUPPORTED_FORMATS = ["wav", "mp3", "m4b"]

# Qwen3-TTS Supported Languages
QWEN_SUPPORTED_LANGUAGES = [
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# Language Detection to Qwen3 Language Mapping
LANGUAGE_MAP: dict[str, str] = {
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

# Speaker Information
SPEAKER_INFO: dict[str, dict[str, str]] = {
    "Vivian": {
        "description": "Bright, slightly edgy young female voice",
        "native_language": "Chinese",
    },
    "Serena": {
        "description": "Warm, gentle young female voice",
        "native_language": "Chinese",
    },
    "Uncle_Fu": {
        "description": "Seasoned male voice with a low, mellow timbre",
        "native_language": "Chinese",
    },
    "Dylan": {
        "description": "Youthful Beijing male voice with a clear, natural timbre",
        "native_language": "Chinese (Beijing Dialect)",
    },
    "Eric": {
        "description": "Lively Chengdu male voice with a slightly husky brightness",
        "native_language": "Chinese (Sichuan Dialect)",
    },
    "Ryan": {
        "description": "Dynamic male voice with strong rhythmic drive",
        "native_language": "English",
    },
    "Aiden": {
        "description": "Sunny American male voice with a clear midrange",
        "native_language": "English",
    },
    "Ono_Anna": {
        "description": "Playful Japanese female voice with a light, nimble timbre",
        "native_language": "Japanese",
    },
    "Sohee": {
        "description": "Warm Korean female voice with rich emotion",
        "native_language": "Korean",
    },
}

# Get list of supported speakers
SUPPORTED_SPEAKERS: list[str] = list(SPEAKER_INFO.keys())

# TTS Generation Settings
MAX_NEW_TOKENS: int | None = None  # Use model default (recommended)
GENERATION_CONFIG: dict[str, Any] = {
    # Empty by default - model uses optimal settings
}

# Text Chunking Settings
# Optimal chunk size for Qwen3-TTS based on model architecture
# - Qwen3 context: 2000-8000 tokens (~4 chars/token)
# - Safe range: 1500-4000 characters
TTS_CHUNK_SIZE: int = 2000  # characters per chunk

# GPU Batch Processing Settings
# Process multiple chunks simultaneously for better GPU utilization
# - Batch size 8: Good for 24GB+ VRAM
# - Batch size 4: Good for 16-24GB VRAM
# - Batch size 2: Conservative for 8-12GB VRAM (default for parallel processing)
# - Batch size 1: Safest for <8GB VRAM or when using parallel chapter processing
TTS_BATCH_SIZE: int = 2  # chunks processed in parallel

# Parallel Chapter Processing Settings
# Process multiple chapters simultaneously across GPUs
PARALLEL_PROCESSING_ENABLED: bool = True  # Enable auto-detection and parallel processing
MAX_PARALLEL_CHAPTERS: int | None = None  # Auto-detect based on available GPUs (None = auto)
PARALLEL_DEVICES: list[str] | None = None  # Auto-detect GPUs (None = auto, or ["cuda:0", "cuda:1"])
MODEL_VRAM_MB: int = 4000  # Estimated VRAM per model instance in MB
PARALLEL_SAFETY_FACTOR: float = 0.8  # Use 80% of available VRAM (conservative)
PARALLEL_MIN_CHAPTERS: int = 2  # Minimum chapters to use parallel processing

# Checkpoint / Resume Settings
# When enabled, Lalo saves progress after each chapter and auto-resumes on re-run.
# Use --no-resume to force a fresh start.
CHECKPOINT_ENABLED: bool = True
CHECKPOINT_CACHE_DIR: str | None = None  # None = ~/.cache/lalo (or $XDG_CACHE_HOME/lalo)
CHECKPOINT_STALE_DAYS: int = 30  # 'lalo cache clean' removes checkpoints older than this
