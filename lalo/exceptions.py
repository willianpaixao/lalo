"""
Custom exceptions for Lalo EPUB to Audiobook Converter.

This module defines domain-specific exceptions to provide better error handling
and more informative error messages.
"""


class LaloError(Exception):
    """Base exception for all Lalo errors."""

    pass


# EPUB-related exceptions
class EPUBError(LaloError):
    """Base exception for EPUB-related errors."""

    pass


class EPUBNotFoundError(EPUBError):
    """Raised when an EPUB file is not found."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        super().__init__(f"EPUB file not found: {file_path}")


class EPUBInvalidError(EPUBError):
    """Raised when a file is not a valid EPUB."""

    def __init__(self, file_path: str, reason: str | None = None):
        self.file_path = file_path
        self.reason = reason
        message = f"Invalid EPUB file: {file_path}"
        if reason:
            message += f" - {reason}"
        super().__init__(message)


class EPUBParseError(EPUBError):
    """Raised when EPUB parsing fails."""

    def __init__(self, file_path: str, original_error: Exception | None = None):
        self.file_path = file_path
        self.original_error = original_error
        message = f"Failed to parse EPUB: {file_path}"
        if original_error:
            message += f" - {original_error}"
        super().__init__(message)


class EPUBNoChaptersError(EPUBError):
    """Raised when no readable chapters are found in EPUB."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        super().__init__(f"No readable chapters found in EPUB: {file_path}")


# TTS-related exceptions
class TTSError(LaloError):
    """Base exception for TTS-related errors."""

    pass


class GPUNotAvailableError(TTSError):
    """Raised when CUDA GPU is required but not available."""

    def __init__(self):
        super().__init__(
            "CUDA GPU is required but not available.\n"
            "Please ensure you have:\n"
            "  1. NVIDIA GPU with CUDA support\n"
            "  2. PyTorch with CUDA installed\n"
            "  3. Proper CUDA drivers installed\n"
            "\nTo install PyTorch with CUDA, visit: https://pytorch.org/get-started/locally/"
        )


class TTSModelLoadError(TTSError):
    """Raised when TTS model fails to load."""

    def __init__(self, model_name: str, original_error: Exception | None = None):
        self.model_name = model_name
        self.original_error = original_error
        message = f"Failed to load TTS model: {model_name}"
        if original_error:
            message += f" - {original_error}"
        super().__init__(message)


class UnsupportedLanguageError(TTSError):
    """Raised when language is not supported."""

    def __init__(self, language: str, supported_languages: list[str]):
        self.language = language
        self.supported_languages = supported_languages
        super().__init__(
            f"Language '{language}' is not supported. "
            f"Supported languages: {', '.join(supported_languages)}"
        )


class UnsupportedSpeakerError(TTSError):
    """Raised when speaker is not supported."""

    def __init__(self, speaker: str, supported_speakers: list[str]):
        self.speaker = speaker
        self.supported_speakers = supported_speakers
        super().__init__(
            f"Speaker '{speaker}' is not supported. "
            f"Supported speakers: {', '.join(supported_speakers)}"
        )


# Audio-related exceptions
class AudioError(LaloError):
    """Base exception for audio-related errors."""

    pass


class AudioExportError(AudioError):
    """Raised when audio export fails."""

    def __init__(self, format: str, output_path: str, original_error: Exception | None = None):
        self.format = format
        self.output_path = output_path
        self.original_error = original_error
        message = f"Failed to export audio to {format}: {output_path}"
        if original_error:
            message += f" - {original_error}"
        super().__init__(message)


class UnsupportedAudioFormatError(AudioError):
    """Raised when audio format is not supported."""

    def __init__(self, format: str, supported_formats: list[str]):
        self.format = format
        self.supported_formats = supported_formats
        super().__init__(
            f"Unsupported audio format: {format}. Supported formats: {', '.join(supported_formats)}"
        )


class EmptyAudioError(AudioError):
    """Raised when attempting to process empty audio."""

    def __init__(self):
        super().__init__("Cannot process empty audio segments")


class FFmpegNotFoundError(AudioError):
    """Raised when ffmpeg is required but not found."""

    def __init__(self):
        super().__init__(
            "ffmpeg is required for this operation but was not found.\n"
            "Please install ffmpeg:\n"
            "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html"
        )


# Validation exceptions
class ValidationError(LaloError):
    """Base exception for validation errors."""

    pass


class InvalidChapterSelectionError(ValidationError):
    """Raised when chapter selection is invalid."""

    def __init__(self, selection: str, reason: str):
        self.selection = selection
        self.reason = reason
        super().__init__(f"Invalid chapter selection '{selection}': {reason}")


class InvalidFilePathError(ValidationError):
    """Raised when file path is invalid."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Invalid file path '{file_path}': {reason}")
