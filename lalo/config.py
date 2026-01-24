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
# - Default 2000: 4x fewer chunks than previous 500, ~2-3x speedup
TTS_CHUNK_SIZE: int = 2000  # characters per chunk

# GPU Batch Processing Settings
# Process multiple chunks simultaneously for better GPU utilization
# - Batch size 4: Good for 8-16GB VRAM (RTX 3080/3090)
# - Batch size 8: Good for 24GB+ VRAM (RTX 3090/4090)
# - Batch size 2: Conservative for 6-8GB VRAM
# Expected speedup: 1.5-2x on top of chunk optimization
TTS_BATCH_SIZE: int = 4  # chunks processed in parallel
