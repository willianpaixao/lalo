"""
Tests for parallel chapter processing functionality.
"""

from unittest import mock

import numpy as np
import pytest
import torch

from lalo.parallel_processor import ChapterWorker, GPUManager, ParallelChapterProcessor


class TestGPUManager:
    """Tests for GPU detection and management."""

    def test_detect_gpus_no_cuda(self):
        """Test GPU detection when CUDA is unavailable."""
        with mock.patch("torch.cuda.is_available", return_value=False):
            gpus = GPUManager.detect_gpus()
            assert gpus == []

    def test_detect_gpus_single_gpu(self):
        """Test GPU detection with single GPU."""
        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.device_count", return_value=1):
                gpus = GPUManager.detect_gpus()
                assert gpus == ["cuda:0"]

    def test_detect_gpus_multiple_gpus(self):
        """Test GPU detection with multiple GPUs."""
        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.device_count", return_value=4):
                gpus = GPUManager.detect_gpus()
                assert gpus == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    def test_get_vram_available_no_gpu(self):
        """Test VRAM detection when device doesn't exist."""
        vram = GPUManager.get_vram_available("cuda:99")
        assert vram == 0  # Should return 0 on error

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_get_vram_available_real_gpu(self):
        """Test VRAM detection on real GPU (if available)."""
        vram = GPUManager.get_vram_available("cuda:0")
        assert vram > 0  # Should have some VRAM available

    def test_can_fit_model_insufficient_vram(self):
        """Test model fit check with insufficient VRAM."""
        with mock.patch.object(GPUManager, "get_vram_available", return_value=1000):  # Only 1GB
            can_fit = GPUManager.can_fit_model("cuda:0", model_vram_mb=4000)  # Need 4GB
            assert not can_fit

    def test_can_fit_model_sufficient_vram(self):
        """Test model fit check with sufficient VRAM."""
        with mock.patch.object(
            GPUManager, "get_vram_available", return_value=8000
        ):  # 8GB available
            can_fit = GPUManager.can_fit_model("cuda:0", model_vram_mb=4000)  # Need 4GB
            assert can_fit


class TestParallelChapterProcessor:
    """Tests for parallel chapter processor."""

    def test_initialization_no_gpus(self):
        """Test initialization when no GPUs available."""
        with mock.patch.object(GPUManager, "detect_gpus", return_value=[]):
            processor = ParallelChapterProcessor()
            assert processor.actual_parallel == 0
            assert processor.devices == []

    def test_initialization_single_gpu(self):
        """Test initialization with single GPU."""
        with mock.patch.object(GPUManager, "detect_gpus", return_value=["cuda:0"]):
            with mock.patch.object(GPUManager, "can_fit_model", return_value=True):
                processor = ParallelChapterProcessor()
                assert processor.actual_parallel == 1
                assert processor.devices == ["cuda:0"]

    def test_initialization_multiple_gpus(self):
        """Test initialization with multiple GPUs."""
        with mock.patch.object(GPUManager, "detect_gpus", return_value=["cuda:0", "cuda:1"]):
            with mock.patch.object(GPUManager, "can_fit_model", return_value=True):
                processor = ParallelChapterProcessor()
                assert processor.actual_parallel == 2
                assert processor.devices == ["cuda:0", "cuda:1"]

    def test_initialization_max_parallel_limit(self):
        """Test that max_parallel limits worker count."""
        with mock.patch.object(
            GPUManager, "detect_gpus", return_value=["cuda:0", "cuda:1", "cuda:2"]
        ):
            with mock.patch.object(GPUManager, "can_fit_model", return_value=True):
                processor = ParallelChapterProcessor(max_parallel=2)
                assert processor.actual_parallel == 2  # Limited by max_parallel

    def test_initialization_insufficient_vram(self):
        """Test initialization when GPUs don't have enough VRAM."""
        with mock.patch.object(GPUManager, "detect_gpus", return_value=["cuda:0"]):
            with mock.patch.object(
                GPUManager, "can_fit_model", return_value=False
            ):  # Not enough VRAM
                processor = ParallelChapterProcessor()
                assert processor.actual_parallel == 0  # Should disable parallel

    def test_should_use_parallel_insufficient_chapters(self):
        """Test that parallel is skipped for too few chapters."""
        with mock.patch.object(GPUManager, "detect_gpus", return_value=["cuda:0"]):
            with mock.patch.object(GPUManager, "can_fit_model", return_value=True):
                processor = ParallelChapterProcessor(min_chapters=3)
                assert not processor.should_use_parallel(2)  # Only 2 chapters
                assert processor.should_use_parallel(3)  # 3 chapters OK
                assert processor.should_use_parallel(10)  # More chapters OK

    def test_should_use_parallel_no_workers(self):
        """Test that parallel is skipped when no workers available."""
        with mock.patch.object(GPUManager, "detect_gpus", return_value=[]):
            processor = ParallelChapterProcessor()
            assert not processor.should_use_parallel(10)  # Even with many chapters

    def test_process_chapters_fallback_to_sequential(self):
        """Test that processor returns None when parallel not suitable."""
        with mock.patch.object(GPUManager, "detect_gpus", return_value=[]):
            processor = ParallelChapterProcessor()

            # Mock chapter
            class MockChapter:
                content = "Test content"
                number = 1
                title = "Test"

            chapters = [MockChapter()]
            result = processor.process_chapters(
                chapters=chapters,
                language="English",
                speaker="Aiden",
            )
            assert result is None  # Should return None to signal fallback


class TestChapterWorker:
    """Tests for chapter worker."""

    def test_worker_initialization(self):
        """Test worker initialization."""
        worker = ChapterWorker(device="cuda:0", worker_id=0)
        assert worker.device == "cuda:0"
        assert worker.worker_id == 0
        assert worker.tts_engine is None  # Lazy loading

    def test_worker_cleanup(self):
        """Test worker cleanup."""
        worker = ChapterWorker(device="cuda:0", worker_id=0)
        worker.tts_engine = mock.MagicMock()  # Simulate loaded model

        with mock.patch("torch.cuda.empty_cache") as mock_cache:
            worker.cleanup()
            assert worker.tts_engine is None
            mock_cache.assert_called_once()


class TestPreAllocatedConcatenation:
    """Tests for pre-allocated array concatenation."""

    def test_audio_manager_concatenate_single_segment(self):
        """Test concatenation with single segment returns same array content."""
        from lalo.audio_manager import AudioManager

        manager = AudioManager()
        segment = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = manager.concatenate([segment])
        np.testing.assert_array_equal(result, segment)

    def test_audio_manager_concatenate_multiple_segments(self):
        """Test concatenation with multiple segments."""
        from lalo.audio_manager import AudioManager

        manager = AudioManager()
        seg1 = np.array([1.0, 2.0], dtype=np.float32)
        seg2 = np.array([3.0, 4.0], dtype=np.float32)
        seg3 = np.array([5.0, 6.0], dtype=np.float32)

        result = manager.concatenate([seg1, seg2, seg3])
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_audio_manager_concatenate_empty_raises_error(self):
        """Test concatenation with empty list raises error."""
        from lalo.audio_manager import AudioManager
        from lalo.exceptions import EmptyAudioError

        manager = AudioManager()
        with pytest.raises(EmptyAudioError):
            manager.concatenate([])

    def test_audio_manager_concatenate_preserves_dtype(self):
        """Test that concatenation preserves data type."""
        from lalo.audio_manager import AudioManager

        manager = AudioManager()
        seg1 = np.array([1.0, 2.0], dtype=np.float64)
        seg2 = np.array([3.0, 4.0], dtype=np.float64)

        result = manager.concatenate([seg1, seg2])
        assert result.dtype == np.float64
