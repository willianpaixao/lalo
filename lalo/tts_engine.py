"""
TTS Engine wrapper for Qwen3-TTS integration.
"""

import logging
import re
from collections.abc import Callable

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

# Suppress transformers generation warnings
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

from lalo.config import (
    AUDIO_SAMPLE_RATE,
    MODEL_DEVICE,
    MODEL_DTYPE,
    MODEL_NAME,
    QWEN_SUPPORTED_LANGUAGES,
    SPEAKER_INFO,
    SUPPORTED_SPEAKERS,
    TTS_BATCH_SIZE,
    TTS_CHUNK_SIZE,
    USE_FLASH_ATTENTION,
)
from lalo.exceptions import (
    GPUNotAvailableError,
    TTSError,
    UnsupportedLanguageError,
    UnsupportedSpeakerError,
)


class TTSEngine:
    """Wrapper for Qwen3-TTS model with GPU validation and text processing."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = MODEL_DEVICE,
        dtype: str = MODEL_DTYPE,
        use_flash_attention: bool = USE_FLASH_ATTENTION,
    ):
        """
        Initialize TTS Engine.

        Args:
            model_name: Hugging Face model ID or local path
            device: Device to run model on (e.g., "cuda:0")
            dtype: Data type for model (e.g., "bfloat16")
            use_flash_attention: Whether to use FlashAttention2

        Raises:
            RuntimeError: If CUDA GPU is not available
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

        # Validate GPU availability
        self._validate_gpu()

        # Load model
        self.model = self._load_model()

    def _validate_gpu(self) -> None:
        """
        Validate that CUDA GPU is available.

        Raises:
            GPUNotAvailableError: If CUDA GPU is not available
        """
        if not torch.cuda.is_available():
            raise GPUNotAvailableError()

    def _load_model(self) -> Qwen3TTSModel:
        """
        Load the Qwen3-TTS model.

        Returns:
            Loaded Qwen3TTSModel instance
        """
        # Convert dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        # Prepare model loading arguments
        model_kwargs = {
            "device_map": self.device,
            "dtype": torch_dtype,
        }

        # Add flash attention if requested and dtype is compatible
        if self.use_flash_attention and self.dtype in ["bfloat16", "float16"]:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load model
        model = Qwen3TTSModel.from_pretrained(self.model_name, **model_kwargs)

        return model

    def _chunk_text(self, text: str, max_chars: int = TTS_CHUNK_SIZE) -> list[str]:
        """
        Split text into chunks for processing.

        Splits on sentence boundaries when possible to maintain natural speech.

        Default chunk size of 2000 characters is optimized for Qwen3-TTS-12Hz:
        - Aligns with official evaluation setup (max_new_tokens=2048)
        - Produces ~170 seconds of audio per chunk
        - 4x fewer chunks than previous 500-char default
        - Expected 2-3x speedup in processing

        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk (default: 2000)

        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]

        chunks = []

        # Split by paragraphs first
        paragraphs = text.split("\n\n")

        current_chunk = ""

        for para in paragraphs:
            # If paragraph is too long, split by sentences
            if len(para) > max_chars:
                # Split by sentence endings
                sentences = re.split(r"([.!?]+\s+)", para)

                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]  # Add punctuation back

                    if len(current_chunk) + len(sentence) > max_chars:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += sentence
            # Add paragraph to current chunk if it fits
            elif len(current_chunk) + len(para) + 2 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            elif current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def generate(
        self,
        text: str,
        language: str,
        speaker: str,
        instruct: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from text using Qwen3-TTS.

        Args:
            text: Text to convert to speech
            language: Language for TTS (from QWEN_SUPPORTED_LANGUAGES)
            speaker: Speaker name (from SUPPORTED_SPEAKERS)
            instruct: Optional instruction for voice control
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            UnsupportedLanguageError: If language is not supported
            UnsupportedSpeakerError: If speaker is not supported
        """
        # Validate inputs
        if language not in QWEN_SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(language, QWEN_SUPPORTED_LANGUAGES)

        if speaker not in SUPPORTED_SPEAKERS:
            raise UnsupportedSpeakerError(speaker, SUPPORTED_SPEAKERS)

        # Chunk text for processing
        chunks = self._chunk_text(text)
        total_chunks = len(chunks)

        # Validate we have text to process
        if not chunks:
            raise TTSError("Cannot generate audio from empty text")

        # Generate audio with GPU batching for better utilization
        audio_segments = []
        sr: int = AUDIO_SAMPLE_RATE  # Default sample rate, will be updated by model
        batch_size = TTS_BATCH_SIZE

        # Process chunks in batches
        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            current_batch_size = len(batch_chunks)

            # Check if we can batch process (multiple chunks)
            if current_batch_size > 1:
                # Batch processing: send multiple chunks at once
                batch_languages = [language] * current_batch_size
                batch_speakers = [speaker] * current_batch_size
                batch_instructs = [instruct if instruct else ""] * current_batch_size

                # Generate audio for entire batch (type ignore for qwen_tts library stubs)
                batch_wavs, sr = self.model.generate_custom_voice(
                    text=batch_chunks,
                    language=batch_languages,
                    speaker=batch_speakers,
                    instruct=batch_instructs,
                )  # type: ignore[arg-type]

                # Extract audio from each result in the batch
                # batch_wavs is a list of lists, where each inner list contains the audio array
                for i, wavs in enumerate(batch_wavs):
                    # Each 'wavs' is a list containing the audio array, take first element
                    audio_segments.append(wavs[0] if isinstance(wavs, list) else wavs)
                    if progress_callback:
                        progress_callback(batch_start + i + 1, total_chunks)
            else:
                # Single chunk: process normally
                gen_kwargs = {
                    "text": batch_chunks[0],
                    "language": language,
                    "speaker": speaker,
                }

                # Only add instruct if provided and non-empty
                if instruct:
                    gen_kwargs["instruct"] = instruct

                # Generate audio (type ignore for qwen_tts library stubs)
                wavs, sr = self.model.generate_custom_voice(**gen_kwargs)  # type: ignore[arg-type]

                # wavs is a list, take the first element
                audio_segments.append(wavs[0])

                # Update progress
                if progress_callback:
                    progress_callback(batch_start + 1, total_chunks)

        # Concatenate all audio segments
        if len(audio_segments) == 1:
            full_audio = audio_segments[0]
        else:
            valid_segments = []
            for i, seg in enumerate(audio_segments):
                if not isinstance(seg, np.ndarray):
                    raise TTSError(
                        f"Invalid audio segment {i}: expected numpy array, got {type(seg)}"
                    )
                if seg.ndim == 0:
                    raise TTSError(
                        f"Invalid audio segment {i}: zero-dimensional array cannot be concatenated"
                    )
                if seg.ndim > 1:
                    # Flatten multi-dimensional arrays
                    seg = seg.flatten()
                valid_segments.append(seg)

            full_audio = np.concatenate(valid_segments)

        return full_audio, sr

    def get_supported_speakers(self) -> list[str]:
        """Get list of supported speakers."""
        return SUPPORTED_SPEAKERS

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        return QWEN_SUPPORTED_LANGUAGES

    def get_speaker_info(self, speaker: str) -> dict:
        """
        Get information about a specific speaker.

        Args:
            speaker: Speaker name

        Returns:
            Dictionary with speaker information

        Raises:
            ValueError: If speaker not found
        """
        if speaker not in SPEAKER_INFO:
            raise ValueError(f"Speaker '{speaker}' not found")

        return SPEAKER_INFO[speaker]
