"""
Tests for tts_engine module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from lalo.exceptions import (
    GPUNotAvailableError,
    TTSError,
    TTSModelLoadError,
    UnsupportedLanguageError,
    UnsupportedSpeakerError,
)
from lalo.tts_engine import TTSEngine


class TestTextChunking:
    """Tests for text chunking algorithm."""

    def setup_method(self):
        """Set up test fixtures."""
        # We'll test chunking without initializing the full engine
        # by creating a mock engine and calling _chunk_text directly
        with patch("lalo.tts_engine.torch.cuda.is_available", return_value=True):
            with patch("lalo.tts_engine.TTSEngine._load_model"):
                self.engine = TTSEngine()

    def test_short_text_no_chunking(self):
        """Test that short text is not chunked."""
        text = "This is a short text."
        chunks = self.engine._chunk_text(text, max_chars=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_max_chars_no_chunking(self):
        """Test text exactly at max_chars is not chunked."""
        text = "a" * 500
        chunks = self.engine._chunk_text(text, max_chars=500)
        assert len(chunks) == 1

    def test_split_by_paragraphs(self):
        """Test that text is split by paragraph boundaries."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = self.engine._chunk_text(text, max_chars=30)

        assert len(chunks) > 1
        # Each paragraph should be in a separate chunk or grouped appropriately

    def test_split_by_sentences(self):
        """Test that long paragraphs are split by sentences."""
        # Create a long paragraph that needs sentence splitting
        text = "First sentence. " * 100  # Will exceed max_chars
        chunks = self.engine._chunk_text(text, max_chars=200)

        assert len(chunks) > 1
        # Each chunk should end at sentence boundary
        for chunk in chunks[:-1]:  # All but last
            # Should end with punctuation and space or just punctuation
            assert chunk.rstrip().endswith(".")

    def test_very_long_sentence_handling(self):
        """Test handling of sentences longer than max_chars."""
        # A single very long sentence with no punctuation
        text = "word " * 500  # 2500+ chars, no sentence breaks
        chunks = self.engine._chunk_text(text, max_chars=200)

        # Without punctuation/paragraph breaks, keeps text together
        # This preserves sentence integrity
        assert len(chunks) >= 1

    def test_preserve_paragraph_structure(self):
        """Test that paragraph breaks are preserved in chunks."""
        para1 = "Short paragraph one."
        para2 = "Short paragraph two."
        text = f"{para1}\n\n{para2}"

        chunks = self.engine._chunk_text(text, max_chars=500)

        # Should keep paragraphs together
        assert len(chunks) == 1
        assert "\n\n" in chunks[0]

    def test_empty_text(self):
        """Test handling of empty text."""
        chunks = self.engine._chunk_text("", max_chars=500)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_whitespace_only_text(self):
        """Test handling of whitespace-only text."""
        chunks = self.engine._chunk_text("   \n\n   ", max_chars=500)
        assert len(chunks) == 1

    def test_no_punctuation_text(self):
        """Test text with no punctuation stays as single chunk."""
        text = "word " * 100  # No periods or other punctuation (500 chars total)
        chunks = self.engine._chunk_text(text, max_chars=200)

        # Without sentence boundaries, text is kept as a single chunk
        # even if it exceeds max_chars (to avoid breaking mid-word)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()  # Chunks are stripped of whitespace

    def test_mixed_punctuation(self):
        """Test text with mixed punctuation marks."""
        text = "Question? Exclamation! Statement. " * 50
        chunks = self.engine._chunk_text(text, max_chars=200)

        assert len(chunks) > 1
        # Chunks should respect sentence boundaries

    def test_unicode_text_chunking(self):
        """Test chunking of Unicode text."""
        text = "日本語の文章です。" * 100  # Japanese text
        chunks = self.engine._chunk_text(text, max_chars=200)

        assert len(chunks) >= 1
        # Should handle Unicode properly

    def test_chunk_size_customization(self):
        """Test that custom chunk size is respected with proper splits."""
        # Use text with natural split points (sentences)
        text = "This is sentence one. " * 100  # Multiple sentences

        chunks_500 = self.engine._chunk_text(text, max_chars=500)
        chunks_200 = self.engine._chunk_text(text, max_chars=200)

        # Smaller max_chars should produce more chunks
        assert len(chunks_200) >= len(chunks_500)

    def test_default_chunk_size(self):
        """Test that default chunk size is 2000."""
        # Use text with natural split points
        text = "This is a sentence. " * 300  # Will exceed 2000 chars
        chunks = self.engine._chunk_text(text)  # Use default (2000)

        # Should produce chunks with sentences
        assert len(chunks) >= 2


class TestTTSEngineInitialization:
    """Tests for TTS Engine initialization and validation."""

    @patch("lalo.tts_engine.torch.cuda.is_available", return_value=False)
    @patch("lalo.tts_engine.Qwen3TTSModel.from_pretrained")
    def test_gpu_not_available_raises_error(self, mock_model, mock_cuda):
        """Test that initialization fails without GPU."""
        with pytest.raises(GPUNotAvailableError, match="CUDA GPU"):
            TTSEngine()

    @patch("lalo.tts_engine.torch.cuda.is_available", return_value=True)
    @patch("lalo.tts_engine.Qwen3TTSModel.from_pretrained")
    def test_successful_initialization(self, mock_model, mock_cuda):
        """Test successful initialization with GPU."""
        mock_model.return_value = Mock()

        engine = TTSEngine()

        assert engine.device == "cuda:0"
        assert engine.dtype == "bfloat16"
        assert mock_model.called

    @patch("lalo.tts_engine.torch.cuda.is_available", return_value=True)
    @patch("lalo.tts_engine.Qwen3TTSModel.from_pretrained")
    def test_custom_device_and_dtype(self, mock_model, mock_cuda):
        """Test initialization with custom device and dtype."""
        mock_model.return_value = Mock()

        engine = TTSEngine(device="cuda:1", dtype="float16")

        assert engine.device == "cuda:1"
        assert engine.dtype == "float16"

    @patch("lalo.tts_engine.torch.cuda.is_available", return_value=True)
    @patch("lalo.tts_engine.Qwen3TTSModel.from_pretrained")
    def test_flash_attention_enabled_with_bfloat16(self, mock_model, mock_cuda):
        """Test that flash attention is used with compatible dtype."""
        mock_model.return_value = Mock()

        TTSEngine(use_flash_attention=True, dtype="bfloat16")

        # Check that flash attention was passed to model
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs.get("attn_implementation") == "flash_attention_2"

    @patch("lalo.tts_engine.torch.cuda.is_available", return_value=True)
    @patch("lalo.tts_engine.Qwen3TTSModel.from_pretrained")
    def test_flash_attention_disabled_with_float32(self, mock_model, mock_cuda):
        """Test that flash attention is not used with float32."""
        mock_model.return_value = Mock()

        TTSEngine(use_flash_attention=True, dtype="float32")

        # Flash attention should not be enabled for float32
        call_kwargs = mock_model.call_args[1]
        assert "attn_implementation" not in call_kwargs

    @patch("lalo.tts_engine.torch.cuda.is_available", return_value=True)
    @patch("lalo.tts_engine.Qwen3TTSModel.from_pretrained")
    def test_model_load_error_raises_custom_exception(self, mock_model, mock_cuda):
        """Test that model loading errors are wrapped in TTSModelLoadError."""
        # Simulate model loading failure
        original_error = RuntimeError("Failed to download model")
        mock_model.side_effect = original_error

        with pytest.raises(TTSModelLoadError) as exc_info:
            TTSEngine()

        # Verify error details
        assert "Qwen3-TTS" in str(exc_info.value)
        assert exc_info.value.model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        assert exc_info.value.original_error == original_error

    @patch("lalo.tts_engine.torch.cuda.is_available", return_value=True)
    @patch("lalo.tts_engine.Qwen3TTSModel.from_pretrained")
    def test_model_load_error_with_network_failure(self, mock_model, mock_cuda):
        """Test model loading error with network failure."""
        mock_model.side_effect = ConnectionError("Network unreachable")

        with pytest.raises(TTSModelLoadError, match="Failed to load TTS model"):
            TTSEngine()


class TestTTSGeneration:
    """Tests for TTS audio generation."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("lalo.tts_engine.torch.cuda.is_available", return_value=True):
            with patch("lalo.tts_engine.Qwen3TTSModel.from_pretrained") as mock_model:
                self.mock_model_instance = Mock()
                mock_model.return_value = self.mock_model_instance
                self.engine = TTSEngine()

    def test_unsupported_language_raises_error(self):
        """Test that unsupported language raises error."""
        with pytest.raises(UnsupportedLanguageError, match="not supported"):
            self.engine.generate(
                text="Test text",
                language="Klingon",
                speaker="Ryan",
            )

    def test_unsupported_speaker_raises_error(self):
        """Test that unsupported speaker raises error."""
        with pytest.raises(UnsupportedSpeakerError, match="not supported"):
            self.engine.generate(
                text="Test text",
                language="English",
                speaker="UnknownSpeaker",
            )

    def test_empty_text_handling(self):
        """Test that empty text is handled gracefully."""
        # Empty text creates a single empty chunk ['']
        # The TTS engine will process it (though it's not a real use case)
        sample_audio = np.random.randn(100).astype(np.float32)  # Small audio
        self.mock_model_instance.generate_custom_voice.return_value = ([sample_audio], 24000)

        # Should not crash, just return minimal audio
        audio, sr = self.engine.generate(
            text="",
            language="English",
            speaker="Ryan",
        )

        assert isinstance(audio, np.ndarray)
        assert sr == 24000

    def test_generate_single_chunk(self):
        """Test generating audio from single chunk."""
        # Mock the model's generate method
        sample_audio = np.random.randn(24000).astype(np.float32)
        self.mock_model_instance.generate_custom_voice.return_value = ([sample_audio], 24000)

        audio, sr = self.engine.generate(
            text="Short text",
            language="English",
            speaker="Ryan",
        )

        assert isinstance(audio, np.ndarray)
        assert sr == 24000
        assert len(audio) > 0

    def test_generate_multiple_chunks(self):
        """Test generating audio from multiple chunks."""
        # Create text that will definitely be chunked (use paragraphs)
        long_text = (
            "This is paragraph one. " * 50 + "\n\n" + "This is paragraph two. " * 50
        )  # Will be split by paragraphs

        # Mock returns different audio for each chunk
        chunk1 = np.random.randn(24000).astype(np.float32)
        chunk2 = np.random.randn(24000).astype(np.float32)

        call_count = [0]

        def mock_generate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ([chunk1], 24000)
            else:
                return ([chunk2], 24000)

        self.mock_model_instance.generate_custom_voice.side_effect = mock_generate

        audio, sr = self.engine.generate(
            text=long_text,
            language="English",
            speaker="Ryan",
        )

        # Should have generated audio
        assert len(audio) > 0
        assert sr == 24000

    def test_progress_callback_called(self):
        """Test that progress callback is called during generation."""
        # Short text to have predictable chunks
        text = "Short sentence. " * 10

        sample_audio = np.random.randn(24000).astype(np.float32)
        self.mock_model_instance.generate_custom_voice.return_value = ([sample_audio], 24000)

        callback_calls = []

        def progress_callback(current, total):
            callback_calls.append((current, total))

        self.engine.generate(
            text=text,
            language="English",
            speaker="Ryan",
            progress_callback=progress_callback,
        )

        # Callback should have been called
        assert len(callback_calls) > 0
        # Last call should indicate completion
        assert callback_calls[-1][0] == callback_calls[-1][1]

    def test_batch_processing_multiple_chunks(self):
        """Test batch processing of multiple chunks."""
        # Create text that will produce multiple chunks
        text = "Sentence. " * 300

        # Mock batch processing
        batch_audio1 = np.random.randn(24000).astype(np.float32)
        batch_audio2 = np.random.randn(24000).astype(np.float32)

        self.mock_model_instance.generate_custom_voice.return_value = (
            [[batch_audio1], [batch_audio2]],  # Batch of 2
            24000,
        )

        audio, sr = self.engine.generate(
            text=text,
            language="English",
            speaker="Ryan",
        )

        assert len(audio) > 0
        assert sr == 24000

    def test_instruct_parameter_passed(self):
        """Test that instruct parameter is passed to model."""
        sample_audio = np.random.randn(24000).astype(np.float32)
        self.mock_model_instance.generate_custom_voice.return_value = ([sample_audio], 24000)

        self.engine.generate(
            text="Test",
            language="English",
            speaker="Ryan",
            instruct="Speak slowly",
        )

        # Check that instruct was passed
        call_kwargs = self.mock_model_instance.generate_custom_voice.call_args[1]
        assert "instruct" in call_kwargs or call_kwargs.get("instruct") is not None

    def test_invalid_audio_segment_raises_error(self):
        """Test that invalid audio segment raises error."""
        # Create text with multiple paragraphs to force chunking
        text = "Paragraph one. " * 100 + "\n\n" + "Paragraph two. " * 100

        # Mock returning non-numpy array in batch mode
        self.mock_model_instance.generate_custom_voice.return_value = (
            [["invalid"], ["invalid"]],  # Not numpy arrays
            24000,
        )

        with pytest.raises(TTSError, match="expected numpy array"):
            self.engine.generate(
                text=text,
                language="English",
                speaker="Ryan",
            )

    def test_zero_dimensional_array_raises_error(self):
        """Test that zero-dimensional array raises error."""
        text = "Paragraph one. " * 100 + "\n\n" + "Paragraph two. " * 100
        chunk1 = np.random.randn(24000).astype(np.float32)
        chunk2 = np.array(5.0)  # Zero-dimensional

        self.mock_model_instance.generate_custom_voice.return_value = ([[chunk1], [chunk2]], 24000)

        with pytest.raises(TTSError, match="zero-dimensional"):
            self.engine.generate(
                text=text,
                language="English",
                speaker="Ryan",
            )

    def test_multidimensional_array_flattened(self):
        """Test that multi-dimensional arrays are flattened."""
        text = "Paragraph one. " * 100 + "\n\n" + "Paragraph two. " * 100
        chunk1 = np.random.randn(24000).astype(np.float32)
        chunk2 = np.random.randn(2, 12000).astype(np.float32)  # 2D array

        # Return as batch (list of lists)
        self.mock_model_instance.generate_custom_voice.return_value = ([[chunk1], [chunk2]], 24000)

        audio, sr = self.engine.generate(
            text=text,
            language="English",
            speaker="Ryan",
        )

        # Should succeed by flattening multi-dimensional arrays
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1


class TestTTSEngineHelpers:
    """Tests for TTS Engine helper methods."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("lalo.tts_engine.torch.cuda.is_available", return_value=True):
            with patch("lalo.tts_engine.Qwen3TTSModel.from_pretrained"):
                self.engine = TTSEngine()

    def test_get_supported_speakers(self):
        """Test getting list of supported speakers."""
        speakers = self.engine.get_supported_speakers()
        assert isinstance(speakers, list)
        assert len(speakers) > 0
        assert "Ryan" in speakers

    def test_get_supported_languages(self):
        """Test getting list of supported languages."""
        languages = self.engine.get_supported_languages()
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "English" in languages

    def test_get_speaker_info_valid(self):
        """Test getting info for valid speaker."""
        info = self.engine.get_speaker_info("Ryan")
        assert isinstance(info, dict)
        assert "description" in info
        assert "native_language" in info

    def test_get_speaker_info_invalid(self):
        """Test getting info for invalid speaker raises error."""
        with pytest.raises(ValueError, match="Speaker .* not found"):
            self.engine.get_speaker_info("NonexistentSpeaker")
