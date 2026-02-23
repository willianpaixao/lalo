"""
Checkpoint and resume support for long-running conversions.

Saves progress after each chapter so interrupted conversions can be
automatically resumed by re-running the same command.  Checkpoint data
and intermediate audio files are stored under the XDG cache directory
(``~/.cache/lalo`` by default).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from lalo.config import CHECKPOINT_CACHE_DIR
from lalo.exceptions import CheckpointCorruptedError, CheckpointMismatchError

logger = logging.getLogger(__name__)

# Current checkpoint schema version.  Bump when the serialised format changes
# in a backwards-incompatible way.
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class CheckpointData:
    """All state needed to resume an interrupted conversion."""

    version: int
    epub_file: str
    epub_hash: str
    output_path: str
    format: str
    speaker: str
    language: str
    instruct: str | None
    streaming: bool
    parallel: bool
    total_chapters: int
    selected_chapters: list[int]  # 1-indexed chapter numbers
    completed_chapters: list[int]  # 1-indexed chapter numbers
    chapter_audio_dir: str
    model_name: str
    sample_rate: int
    started_at: str  # ISO 8601
    updated_at: str  # ISO 8601
    # Titles for completed chapters, kept in sync with completed_chapters
    completed_titles: list[str] = field(default_factory=list)


class CheckpointManager:
    """Manage checkpoint files and cached audio for a single conversion.

    Usage::

        mgr = CheckpointManager(epub_file, output_path)

        # Check for a previous run
        data = mgr.load()
        if data is not None:
            mgr.validate(data, epub_hash=..., speaker=..., ...)
            remaining = mgr.get_remaining_chapters(data)
            ...

        # During conversion, after each chapter
        mgr.mark_chapter_completed(data, chapter_number, chapter_title)

        # On success
        mgr.cleanup()
    """

    def __init__(
        self,
        epub_file: str | Path,
        output_path: str | Path,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.epub_file = Path(epub_file).resolve()
        self.output_path = Path(output_path).resolve()
        self._cache_base = self._resolve_cache_dir(cache_dir)
        self._epub_hash: str | None = None  # Lazily computed

    # Cache directory helpers

    @staticmethod
    def _resolve_cache_dir(override: str | Path | None) -> Path:
        """Return the base cache directory, respecting XDG and overrides."""
        if override is not None:
            return Path(override).expanduser()
        if CHECKPOINT_CACHE_DIR is not None:
            return Path(CHECKPOINT_CACHE_DIR).expanduser()
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            return Path(xdg) / "lalo"
        return Path.home() / ".cache" / "lalo"

    def _epub_hash_prefix(self, epub_hash: str) -> str:
        """Extract a short prefix from the full hash for directory naming."""
        # "sha256:abcdef..." -> first 16 hex chars
        _, hexdigest = epub_hash.split(":", 1)
        return hexdigest[:16]

    def get_checkpoint_dir(self, epub_hash: str) -> Path:
        """Return the per-book cache directory."""
        return self._cache_base / self._epub_hash_prefix(epub_hash)

    def get_checkpoint_path(self, epub_hash: str) -> Path:
        """Return the path to the checkpoint JSON file."""
        return self.get_checkpoint_dir(epub_hash) / "checkpoint.json"

    def get_audio_cache_dir(self, epub_hash: str) -> Path:
        """Return the directory where intermediate chapter WAV files are stored."""
        return self.get_checkpoint_dir(epub_hash) / "audio"

    # Persistence

    def save(self, data: CheckpointData) -> None:
        """Atomically write checkpoint data to disk.

        Uses write-to-tmp + ``os.replace`` to avoid partial writes.
        """
        data.updated_at = datetime.now(UTC).isoformat()
        path = self.get_checkpoint_path(data.epub_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(data), indent=2))
        os.replace(tmp, path)
        logger.debug("Checkpoint saved: %s", path)

    def load(self, epub_hash: str) -> CheckpointData | None:
        """Load an existing checkpoint, or return ``None``.

        Raises:
            CheckpointCorruptedError: If the file exists but cannot be parsed
                or has an incompatible schema version.
        """
        path = self.get_checkpoint_path(epub_hash)
        if not path.exists():
            return None

        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise CheckpointCorruptedError(str(path), f"Cannot read JSON: {exc}") from exc

        # Schema version check
        version = raw.get("version")
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCorruptedError(
                str(path),
                f"Unsupported schema version {version} (expected {CHECKPOINT_SCHEMA_VERSION})",
            )

        # Backwards-compat: older checkpoints may lack completed_titles
        if "completed_titles" not in raw:
            raw["completed_titles"] = []

        try:
            return CheckpointData(**raw)
        except TypeError as exc:
            raise CheckpointCorruptedError(str(path), f"Invalid checkpoint data: {exc}") from exc

    # Validation

    def validate(
        self,
        data: CheckpointData,
        epub_hash: str,
        speaker: str,
        language: str,
        instruct: str | None,
        format: str,
    ) -> None:
        """Validate that a checkpoint matches the current conversion settings.

        The EPUB hash is checked strictly (raises on mismatch).  Other
        settings produce warnings so the user can decide.

        Raises:
            CheckpointMismatchError: If the EPUB hash does not match.
        """
        if data.epub_hash != epub_hash:
            raise CheckpointMismatchError("epub_hash", data.epub_hash, epub_hash)

        # Non-fatal mismatches — warn but allow resume
        for field_name, ckpt_val, cur_val in [
            ("speaker", data.speaker, speaker),
            ("language", data.language, language),
            ("instruct", data.instruct or "", instruct or ""),
            ("format", data.format, format),
        ]:
            if ckpt_val != cur_val:
                logger.warning(
                    "Checkpoint setting '%s' differs: checkpoint='%s', current='%s'. "
                    "Using current value.",
                    field_name,
                    ckpt_val,
                    cur_val,
                )

    # Chapter tracking

    def mark_chapter_completed(
        self,
        data: CheckpointData,
        chapter_number: int,
        chapter_title: str,
    ) -> None:
        """Record a chapter as completed and persist the checkpoint."""
        if chapter_number not in data.completed_chapters:
            data.completed_chapters.append(chapter_number)
            data.completed_titles.append(chapter_title)
        self.save(data)

    @staticmethod
    def get_remaining_chapters(data: CheckpointData) -> list[int]:
        """Return 1-indexed chapter numbers not yet completed (in order)."""
        completed = set(data.completed_chapters)
        return [ch for ch in data.selected_chapters if ch not in completed]

    def verify_audio_files(self, data: CheckpointData) -> list[Path]:
        """Check that WAV files for all completed chapters exist on disk.

        If a file is missing the corresponding chapter is removed from
        ``completed_chapters`` (and ``completed_titles``) so it will be
        re-processed on resume.

        Returns:
            List of valid WAV file paths in chapter order.
        """
        audio_dir = Path(data.chapter_audio_dir)
        valid_files: list[Path] = []
        valid_chapters: list[int] = []
        valid_titles: list[str] = []

        for idx, ch_num in enumerate(data.completed_chapters):
            wav_path = audio_dir / f"chapter_{ch_num:04d}.wav"
            if wav_path.exists():
                valid_files.append(wav_path)
                valid_chapters.append(ch_num)
                if idx < len(data.completed_titles):
                    valid_titles.append(data.completed_titles[idx])
            else:
                logger.warning(
                    "Audio file missing for chapter %d: %s — will re-process",
                    ch_num,
                    wav_path,
                )

        if len(valid_chapters) != len(data.completed_chapters):
            data.completed_chapters = valid_chapters
            data.completed_titles = valid_titles
            self.save(data)

        return valid_files

    # Cleanup

    def cleanup(self, epub_hash: str) -> None:
        """Delete the checkpoint file and all cached audio."""
        ckpt_dir = self.get_checkpoint_dir(epub_hash)
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            logger.info("Checkpoint cleaned up: %s", ckpt_dir)

    # Factory helper

    def create_checkpoint(
        self,
        epub_hash: str,
        format: str,
        speaker: str,
        language: str,
        instruct: str | None,
        streaming: bool,
        parallel: bool,
        selected_chapters: list[int],
        model_name: str,
        sample_rate: int,
    ) -> CheckpointData:
        """Create a fresh ``CheckpointData`` and persist it.

        The audio cache directory is created on disk as a side-effect.
        """
        audio_dir = self.get_audio_cache_dir(epub_hash)
        audio_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(UTC).isoformat()
        data = CheckpointData(
            version=CHECKPOINT_SCHEMA_VERSION,
            epub_file=str(self.epub_file),
            epub_hash=epub_hash,
            output_path=str(self.output_path),
            format=format,
            speaker=speaker,
            language=language,
            instruct=instruct,
            streaming=streaming,
            parallel=parallel,
            total_chapters=len(selected_chapters),
            selected_chapters=selected_chapters,
            completed_chapters=[],
            chapter_audio_dir=str(audio_dir),
            model_name=model_name,
            sample_rate=sample_rate,
            started_at=now,
            updated_at=now,
            completed_titles=[],
        )
        self.save(data)
        return data
