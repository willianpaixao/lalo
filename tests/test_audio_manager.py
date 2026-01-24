"""
Tests for audio_manager module.
"""

import numpy as np
import pytest

from lalo.audio_manager import AudioManager
from lalo.exceptions import EmptyAudioError, UnsupportedAudioFormatError


class TestAudioManager:
    """Tests for AudioManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AudioManager(sample_rate=24000)
        # Create sample audio data
        self.sample_audio = np.random.randn(24000).astype(np.float32)  # 1 second

    def test_concatenate_single_segment(self):
        """Test concatenating a single audio segment."""
        result = self.manager.concatenate([self.sample_audio])
        assert np.array_equal(result, self.sample_audio)

    def test_concatenate_multiple_segments(self):
        """Test concatenating multiple audio segments."""
        segments = [self.sample_audio, self.sample_audio, self.sample_audio]
        result = self.manager.concatenate(segments)
        expected_length = len(self.sample_audio) * 3
        assert len(result) == expected_length

    def test_concatenate_empty_raises_error(self):
        """Test that concatenating empty list raises error."""
        with pytest.raises(EmptyAudioError, match="Cannot process empty audio"):
            self.manager.concatenate([])

    def test_get_duration(self):
        """Test duration calculation."""
        duration = self.manager.get_duration(self.sample_audio)
        assert abs(duration - 1.0) < 0.01  # Should be ~1 second

    def test_export_unsupported_format_raises_error(self):
        """Test that exporting to unsupported format raises error."""
        with pytest.raises(UnsupportedAudioFormatError, match="Unsupported audio format"):
            self.manager.export(self.sample_audio, "test.ogg", format="ogg")
