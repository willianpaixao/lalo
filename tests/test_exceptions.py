"""
Tests for exceptions module.
"""

from lalo.exceptions import (
    AudioError,
    AudioExportError,
    EmptyAudioError,
    EPUBError,
    EPUBInvalidError,
    EPUBNoChaptersError,
    EPUBNotFoundError,
    EPUBParseError,
    FFmpegNotFoundError,
    GPUNotAvailableError,
    InvalidChapterSelectionError,
    InvalidFilePathError,
    LaloError,
    TTSError,
    TTSModelLoadError,
    UnsupportedAudioFormatError,
    UnsupportedLanguageError,
    UnsupportedSpeakerError,
)


class TestBaseExceptions:
    """Tests for base exception classes."""

    def test_lalo_error_is_base(self):
        """Test that LaloError is base for all custom exceptions."""
        error = LaloError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_epub_error_inheritance(self):
        """Test that EPUBError inherits from LaloError."""
        error = EPUBError("EPUB error")
        assert isinstance(error, LaloError)
        assert isinstance(error, Exception)

    def test_audio_error_inheritance(self):
        """Test that AudioError inherits from LaloError."""
        error = AudioError("Audio error")
        assert isinstance(error, LaloError)

    def test_tts_error_inheritance(self):
        """Test that TTSError inherits from LaloError."""
        error = TTSError("TTS error")
        assert isinstance(error, LaloError)


class TestEPUBExceptions:
    """Tests for EPUB-related exceptions."""

    def test_epub_not_found_error(self):
        """Test EPUBNotFoundError message."""
        error = EPUBNotFoundError("/path/to/book.epub")
        message = str(error)

        assert "not found" in message.lower()
        assert "/path/to/book.epub" in message

    def test_epub_invalid_error(self):
        """Test EPUBInvalidError message."""
        error = EPUBInvalidError("/path/to/file.txt", "Not an EPUB file")
        message = str(error)

        assert "invalid" in message.lower()
        assert "/path/to/file.txt" in message
        assert "Not an EPUB file" in message

    def test_epub_no_chapters_error(self):
        """Test EPUBNoChaptersError message."""
        error = EPUBNoChaptersError("/path/to/book.epub")
        message = str(error)

        assert "no readable chapters" in message.lower()
        assert "/path/to/book.epub" in message

    def test_epub_parse_error(self):
        """Test EPUBParseError message."""
        original_error = ValueError("Corrupted ZIP")
        error = EPUBParseError("/path/to/book.epub", original_error)
        message = str(error)

        assert "failed to parse" in message.lower()
        assert "/path/to/book.epub" in message
        assert "Corrupted ZIP" in message


class TestAudioExceptions:
    """Tests for audio-related exceptions."""

    def test_empty_audio_error(self):
        """Test EmptyAudioError message."""
        error = EmptyAudioError()
        message = str(error)

        assert "empty audio" in message.lower()

    def test_unsupported_audio_format_error(self):
        """Test UnsupportedAudioFormatError message."""
        error = UnsupportedAudioFormatError("ogg", ["mp3", "wav", "m4b"])
        message = str(error)

        assert "unsupported" in message.lower()
        assert "ogg" in message
        assert "mp3" in message
        assert "wav" in message

    def test_audio_export_error(self):
        """Test AudioExportError message."""
        original_error = IOError("Disk full")
        error = AudioExportError("mp3", "/path/to/output.mp3", original_error)
        message = str(error)

        assert "export" in message.lower()
        assert "mp3" in message
        assert "/path/to/output.mp3" in message
        assert "Disk full" in message


class TestTTSExceptions:
    """Tests for TTS-related exceptions."""

    def test_gpu_not_available_error(self):
        """Test GPUNotAvailableError message."""
        error = GPUNotAvailableError()
        message = str(error)

        assert "cuda" in message.lower() or "gpu" in message.lower()
        assert "not available" in message.lower()

    def test_unsupported_language_error(self):
        """Test UnsupportedLanguageError message."""
        error = UnsupportedLanguageError("Klingon", ["English", "Chinese", "Japanese"])
        message = str(error)

        assert "not supported" in message.lower()
        assert "language" in message.lower()
        assert "Klingon" in message
        assert "English" in message

    def test_unsupported_speaker_error(self):
        """Test UnsupportedSpeakerError message."""
        error = UnsupportedSpeakerError("UnknownSpeaker", ["Ryan", "Emily", "Grace"])
        message = str(error)

        assert "not supported" in message.lower()
        assert "speaker" in message.lower()
        assert "UnknownSpeaker" in message
        assert "Ryan" in message

    def test_tts_model_load_error(self):
        """Test TTSModelLoadError message."""
        original_error = RuntimeError("Network error")
        error = TTSModelLoadError("my-model", original_error)
        message = str(error)

        assert "failed to load" in message.lower()
        assert "my-model" in message
        assert "Network error" in message
        assert error.model_name == "my-model"
        assert error.original_error == original_error

    def test_tts_model_load_error_without_original(self):
        """Test TTSModelLoadError without original error."""
        error = TTSModelLoadError("my-model", None)
        message = str(error)

        assert "failed to load" in message.lower()
        assert "my-model" in message
        assert error.model_name == "my-model"
        assert error.original_error is None

    def test_ffmpeg_not_found_error(self):
        """Test FFmpegNotFoundError message."""
        error = FFmpegNotFoundError()
        message = str(error)

        assert "ffmpeg" in message.lower()
        assert "required" in message.lower()
        # Should include installation instructions
        assert "install" in message.lower()
        # Should mention common package managers
        assert "apt-get" in message or "brew" in message or "download" in message.lower()


class TestValidationExceptions:
    """Tests for validation-related exceptions."""

    def test_invalid_chapter_selection_error(self):
        """Test InvalidChapterSelectionError message."""
        error = InvalidChapterSelectionError("1-50", "Range exceeds available chapters")
        message = str(error)

        assert "invalid chapter selection" in message.lower()
        assert "1-50" in message
        assert "Range exceeds available chapters" in message

    def test_invalid_file_path_error(self):
        """Test InvalidFilePathError message."""
        error = InvalidFilePathError("/invalid/path.txt", "File not found")
        message = str(error)

        assert "invalid file path" in message.lower()
        assert "/invalid/path.txt" in message
        assert "File not found" in message


class TestExceptionDetails:
    """Tests for exception details and attributes."""

    def test_epub_invalid_has_details(self):
        """Test that EPUBInvalidError stores file path and reason."""
        error = EPUBInvalidError("/test.epub", "Bad format")

        # The exception should contain the information
        assert hasattr(error, "args")
        message = str(error)
        assert "/test.epub" in message
        assert "Bad format" in message

    def test_audio_export_error_has_details(self):
        """Test that AudioExportError stores format and path."""
        original = IOError("Write failed")
        error = AudioExportError("mp3", "/output.mp3", original)

        message = str(error)
        assert "mp3" in message
        assert "/output.mp3" in message

    def test_unsupported_format_lists_alternatives(self):
        """Test that UnsupportedAudioFormatError lists supported formats."""
        error = UnsupportedAudioFormatError("flac", ["mp3", "wav", "m4b"])
        message = str(error)

        # Should mention all supported formats
        assert "mp3" in message
        assert "wav" in message
        assert "m4b" in message
