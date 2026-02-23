"""
Lalo - EPUB to Audiobook Converter using Qwen3-TTS

A command-line tool for converting EPUB files to audiobooks using
Qwen3-TTS text-to-speech models.
"""

__version__ = "0.1.0"
__author__ = "Willian Paixao"
__email__ = "willian@paixao.net"

from lalo.config import (
    DEFAULT_LANGUAGE,
    DEFAULT_SPEAKER,
    MODEL_NAME,
    SUPPORTED_FORMATS,
)
from lalo.text_normalization import (
    URLReplacementStrategy,
    normalize_text_for_speech,
    normalize_urls,
)

__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_SPEAKER",
    "MODEL_NAME",
    "SUPPORTED_FORMATS",
    "URLReplacementStrategy",
    "__author__",
    "__email__",
    "__version__",
    "normalize_text_for_speech",
    "normalize_urls",
]
