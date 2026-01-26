"""
Parallel chapter processing for improved performance.

Automatically detects available GPUs and processes multiple chapters
simultaneously for significant speedup
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch

from lalo.config import (
    MAX_PARALLEL_CHAPTERS,
    MODEL_VRAM_MB,
    PARALLEL_DEVICES,
    PARALLEL_MIN_CHAPTERS,
    PARALLEL_SAFETY_FACTOR,
)
from lalo.exceptions import TTSError

logger = logging.getLogger(__name__)


class GPUManager:
    """Detect and manage available GPU devices."""

    @staticmethod
    def detect_gpus() -> list[str]:
        """
        Detect all available CUDA devices.

        Returns:
            List of device names (e.g., ["cuda:0", "cuda:1"])
            Empty list if no CUDA available
        """
        if not torch.cuda.is_available():
            logger.info("No CUDA GPUs detected")
            return []

        gpu_count = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(gpu_count)]
        logger.info(f"Detected {gpu_count} GPU(s): {devices}")
        return devices

    @staticmethod
    def get_vram_available(device: str) -> int:
        """
        Get available VRAM in MB for a device.

        Args:
            device: Device name (e.g., "cuda:0")

        Returns:
            Available VRAM in MB
        """
        try:
            device_idx = int(device.split(":")[-1])
            # Get total and allocated VRAM
            total_vram = torch.cuda.get_device_properties(device_idx).total_memory
            allocated_vram = torch.cuda.memory_allocated(device_idx)
            reserved_vram = torch.cuda.memory_reserved(device_idx)

            # Available = total - max(allocated, reserved)
            used_vram = max(allocated_vram, reserved_vram)
            available_vram_mb = (total_vram - used_vram) / (1024 * 1024)

            logger.debug(
                f"{device}: {available_vram_mb:.0f} MB available "
                f"({total_vram / (1024 * 1024):.0f} MB total)"
            )
            return int(available_vram_mb)
        except Exception as e:
            logger.warning(f"Could not get VRAM for {device}: {e}")
            return 0

    @staticmethod
    def can_fit_model(device: str, model_vram_mb: int = MODEL_VRAM_MB) -> bool:
        """
        Check if device has enough VRAM for model.

        Args:
            device: Device name (e.g., "cuda:0")
            model_vram_mb: Required VRAM in MB (default from config)

        Returns:
            True if device has enough VRAM
        """
        available = GPUManager.get_vram_available(device)
        can_fit = available >= model_vram_mb
        logger.debug(
            f"{device}: {'Can' if can_fit else 'Cannot'} fit model "
            f"({available} MB available, {model_vram_mb} MB required)"
        )
        return can_fit


class ChapterWorker:
    """
    Worker that processes a chapter on a specific GPU.

    Uses lazy model loading to avoid loading models unnecessarily.
    """

    def __init__(self, device: str, worker_id: int):
        """
        Initialize worker.

        Args:
            device: CUDA device (e.g., "cuda:0")
            worker_id: Unique worker identifier
        """
        from lalo.tts_engine import TTSEngine as _TTSEngine

        self.device = device
        self.worker_id = worker_id
        self.tts_engine: _TTSEngine | None = None  # Lazy loaded
        self.lock = threading.Lock()
        logger.debug(f"Worker {worker_id} initialized on {device}")

    def _load_model(self) -> None:
        """Load TTS model on this worker's device (lazy loading)."""
        if self.tts_engine is not None:
            return

        with self.lock:
            # Double-check after acquiring lock
            if self.tts_engine is not None:
                return

            logger.info(f"Worker {self.worker_id}: Loading TTS model on {self.device}")
            from lalo.tts_engine import TTSEngine

            try:
                self.tts_engine = TTSEngine(device=self.device)
                logger.info(f"Worker {self.worker_id}: Model loaded successfully")
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Failed to load model: {e}")
                raise

    def process_chapter(
        self,
        chapter: Any,
        language: str,
        speaker: str,
        instruct: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Process a single chapter on this worker's GPU.

        Args:
            chapter: Chapter object with content
            language: Language for TTS
            speaker: Speaker name
            instruct: Optional voice instructions
            progress_callback: Optional progress callback

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Lazy load model
        self._load_model()

        logger.debug(
            f"Worker {self.worker_id}: Processing chapter {chapter.number} on {self.device}"
        )

        # Ensure model is loaded
        if self.tts_engine is None:
            raise RuntimeError(f"Worker {self.worker_id}: TTS engine not loaded")

        try:
            audio, sr = self.tts_engine.generate(
                text=chapter.content,
                language=language,
                speaker=speaker,
                instruct=instruct,
                progress_callback=progress_callback,
            )
            logger.debug(
                f"Worker {self.worker_id}: Chapter {chapter.number} complete ({len(audio)} samples)"
            )
            return audio, sr
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error processing chapter {chapter.number}: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up worker resources."""
        if self.tts_engine is not None:
            logger.debug(f"Worker {self.worker_id}: Cleaning up")
            # Model cleanup (if needed)
            self.tts_engine = None
            # Clear CUDA cache
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()


class ParallelChapterProcessor:
    """
    Process multiple chapters in parallel across available GPUs.

    Automatically detects GPUs and manages worker pool for optimal performance.
    Falls back gracefully to sequential processing if parallel unavailable.
    """

    def __init__(
        self,
        max_parallel: int | None = MAX_PARALLEL_CHAPTERS,
        devices: list[str] | None = PARALLEL_DEVICES,
        safety_factor: float = PARALLEL_SAFETY_FACTOR,
        min_chapters: int = PARALLEL_MIN_CHAPTERS,
    ):
        """
        Initialize parallel processor.

        Args:
            max_parallel: Maximum parallel chapters (None = auto-detect)
            devices: List of devices to use (None = auto-detect)
            safety_factor: VRAM safety factor (0.0-1.0)
            min_chapters: Minimum chapters to use parallel processing
        """
        # Validate safety_factor
        if not 0.0 <= safety_factor <= 1.0:
            raise ValueError(f"safety_factor must be between 0.0 and 1.0, got {safety_factor}")

        self.max_parallel = max_parallel
        self.safety_factor = safety_factor
        self.min_chapters = min_chapters
        self.workers: list[ChapterWorker] = []
        self.devices: list[str]  # Always a list after initialization

        # Auto-detect GPUs if not specified
        if devices is None:
            self.devices = GPUManager.detect_gpus()
        else:
            self.devices = devices

        # Determine actual parallelism
        self._determine_parallelism()

        logger.info(
            f"ParallelChapterProcessor initialized: "
            f"{self.actual_parallel} workers on {len(self.devices)} GPU(s)"
        )

    def _determine_parallelism(self) -> None:
        """Determine actual number of parallel workers based on GPUs and VRAM."""
        if not self.devices:
            # No GPUs - fallback to sequential
            self.actual_parallel = 0
            logger.warning("No GPUs available - parallel processing disabled")
            return

        # Filter devices by VRAM availability
        # Apply safety factor: we want available_vram * safety_factor >= MODEL_VRAM_MB
        # This means: available_vram >= MODEL_VRAM_MB / safety_factor
        if self.safety_factor > 0.0:
            required_vram = int(MODEL_VRAM_MB / self.safety_factor)
        else:
            # safety_factor == 0.0: no safety margin
            required_vram = MODEL_VRAM_MB
        usable_devices = [
            dev for dev in self.devices if GPUManager.can_fit_model(dev, required_vram)
        ]

        if not usable_devices:
            # No devices with enough VRAM
            self.actual_parallel = 0
            logger.warning(
                f"No GPUs with sufficient VRAM ({required_vram} MB required) - "
                "parallel processing disabled"
            )
            return

        # Determine number of workers
        if self.max_parallel is None:
            # Auto-detect: one worker per usable GPU
            self.actual_parallel = len(usable_devices)
        else:
            # User specified - limit to available GPUs
            self.actual_parallel = min(self.max_parallel, len(usable_devices))

        # Update self.devices to only contain usable devices
        self.devices = usable_devices[: self.actual_parallel]

        logger.info(
            f"Parallel processing enabled: {self.actual_parallel} workers "
            f"on devices: {self.devices}"
        )

    def _create_workers(self, num_workers: int) -> list[ChapterWorker]:
        """Create worker pool."""
        workers = []
        for i in range(num_workers):
            # Round-robin device assignment
            device = self.devices[i % len(self.devices)]
            worker = ChapterWorker(device=device, worker_id=i)
            workers.append(worker)
        return workers

    def should_use_parallel(self, num_chapters: int) -> bool:
        """
        Determine if parallel processing should be used.

        Args:
            num_chapters: Number of chapters to process

        Returns:
            True if parallel processing should be used
        """
        # Need at least min_chapters and actual parallelism > 0
        use_parallel = self.actual_parallel > 0 and num_chapters >= self.min_chapters

        if use_parallel:
            logger.info(
                f"Using parallel processing for {num_chapters} chapters "
                f"with {self.actual_parallel} workers"
            )
        else:
            logger.info(
                f"Using sequential processing for {num_chapters} chapters "
                f"(parallel workers: {self.actual_parallel}, "
                f"min chapters: {self.min_chapters})"
            )

        return use_parallel

    def process_chapters(
        self,
        chapters: list[Any],
        language: str,
        speaker: str,
        instruct: str | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> list[tuple[np.ndarray, int]] | None:
        """
        Process chapters in parallel.

        Args:
            chapters: List of chapter objects
            language: Language for TTS
            speaker: Speaker name
            instruct: Optional voice instructions
            progress_callback: Optional callback(chapter_idx, current_chunk, total_chunks)

        Returns:
            List of (audio_array, sample_rate) tuples in same order as chapters
        """
        if not self.should_use_parallel(len(chapters)):
            # Fallback to sequential processing
            logger.info("Falling back to sequential processing")
            return None  # Signal to use sequential

        # Create workers
        self.workers = self._create_workers(self.actual_parallel)

        try:
            results: list[tuple[np.ndarray, int] | None] = [None] * len(chapters)  # Preserve order

            with ThreadPoolExecutor(max_workers=self.actual_parallel) as executor:
                # Submit all chapters
                future_to_idx = {}
                for idx, chapter in enumerate(chapters):
                    # Get worker for this chapter (round-robin)
                    worker = self.workers[idx % len(self.workers)]

                    # Create progress callback for this chapter
                    chapter_progress_callback = None
                    if progress_callback:
                        # Wrap to include chapter index
                        def make_callback(chapter_idx):
                            def callback(current, total):
                                progress_callback(chapter_idx, current, total)

                            return callback

                        chapter_progress_callback = make_callback(idx)

                    # Submit chapter to worker
                    future = executor.submit(
                        worker.process_chapter,
                        chapter,
                        language,
                        speaker,
                        instruct,
                        chapter_progress_callback,
                    )
                    future_to_idx[future] = idx

                # Collect results
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        audio, sr = future.result()
                        results[idx] = (audio, sr)
                        logger.debug(f"Chapter {idx} completed")
                    except Exception as e:
                        logger.error(f"Chapter {idx} failed: {e}")
                        raise TTSError(f"Parallel processing failed for chapter {idx}: {e}")

            # All results should be filled, cast to expected type
            return results  # type: ignore[return-value]

        finally:
            # Cleanup workers
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up worker resources."""
        logger.debug("Cleaning up parallel processor")
        for worker in self.workers:
            try:
                worker.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up worker {worker.worker_id}: {e}")
        self.workers = []
