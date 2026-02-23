"""
Tests for checkpoint module.
"""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lalo.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointData,
    CheckpointManager,
)
from lalo.exceptions import (
    CheckpointCorruptedError,
    CheckpointError,
    CheckpointMismatchError,
    LaloError,
)


class TestCheckpointExceptions:
    """Tests for checkpoint exception hierarchy and messages."""

    def test_checkpoint_error_inherits_from_lalo_error(self):
        """CheckpointError should be a subclass of LaloError."""
        assert issubclass(CheckpointError, LaloError)

    def test_corrupted_error_inherits_from_checkpoint_error(self):
        """CheckpointCorruptedError should be a subclass of CheckpointError."""
        assert issubclass(CheckpointCorruptedError, CheckpointError)

    def test_mismatch_error_inherits_from_checkpoint_error(self):
        """CheckpointMismatchError should be a subclass of CheckpointError."""
        assert issubclass(CheckpointMismatchError, CheckpointError)

    def test_corrupted_error_message(self):
        """CheckpointCorruptedError should include path and reason."""
        error = CheckpointCorruptedError("/tmp/ckpt.json", "bad JSON")
        assert "/tmp/ckpt.json" in str(error)
        assert "bad JSON" in str(error)
        assert error.checkpoint_path == "/tmp/ckpt.json"
        assert error.reason == "bad JSON"

    def test_mismatch_error_message(self):
        """CheckpointMismatchError should include field, expected, and actual."""
        error = CheckpointMismatchError("epub_hash", "abc", "xyz")
        assert "epub_hash" in str(error)
        assert "abc" in str(error)
        assert "xyz" in str(error)
        assert error.field == "epub_hash"
        assert error.expected == "abc"
        assert error.actual == "xyz"


class TestCheckpointData:
    """Tests for the CheckpointData dataclass."""

    def test_create_minimal(self):
        """Should create a CheckpointData with all required fields."""
        data = CheckpointData(
            version=1,
            epub_file="/tmp/book.epub",
            epub_hash="sha256:abcdef1234567890",
            output_path="/tmp/book.mp3",
            format="mp3",
            speaker="Aiden",
            language="Auto",
            instruct=None,
            streaming=True,
            parallel=False,
            total_chapters=10,
            selected_chapters=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            completed_chapters=[],
            chapter_audio_dir="/tmp/audio",
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            sample_rate=24000,
            started_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
        )
        assert data.version == 1
        assert data.completed_chapters == []
        assert data.completed_titles == []

    def test_completed_titles_default(self):
        """completed_titles should default to empty list."""
        data = CheckpointData(
            version=1,
            epub_file="x",
            epub_hash="sha256:abc",
            output_path="x.mp3",
            format="mp3",
            speaker="Aiden",
            language="Auto",
            instruct=None,
            streaming=True,
            parallel=False,
            total_chapters=1,
            selected_chapters=[1],
            completed_chapters=[],
            chapter_audio_dir="/tmp",
            model_name="model",
            sample_rate=24000,
            started_at="",
            updated_at="",
        )
        assert data.completed_titles == []


class TestCheckpointManager:
    """Tests for CheckpointManager persistence, validation, and lifecycle."""

    def setup_method(self):
        """Set up test fixtures."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

        # Create a fake EPUB file for hashing
        self.epub_path = self.tmpdir / "test_book.epub"
        self.epub_path.write_bytes(b"fake epub content for testing")

        self.output_path = self.tmpdir / "output.mp3"
        self.cache_dir = self.tmpdir / "cache"

        self.mgr = CheckpointManager(
            epub_file=str(self.epub_path),
            output_path=str(self.output_path),
            cache_dir=str(self.cache_dir),
        )

        # Compute hash once for reuse
        from lalo.utils import compute_file_hash

        self.epub_hash = compute_file_hash(str(self.epub_path))

    def teardown_method(self):
        """Clean up temp directory."""
        self._tmpdir.cleanup()

    def _make_checkpoint_data(self, **overrides) -> CheckpointData:
        """Helper to create a CheckpointData with sensible defaults."""
        now = datetime.now(UTC).isoformat()
        defaults = dict(
            version=CHECKPOINT_SCHEMA_VERSION,
            epub_file=str(self.epub_path),
            epub_hash=self.epub_hash,
            output_path=str(self.output_path),
            format="mp3",
            speaker="Aiden",
            language="Auto",
            instruct=None,
            streaming=True,
            parallel=False,
            total_chapters=5,
            selected_chapters=[1, 2, 3, 4, 5],
            completed_chapters=[],
            chapter_audio_dir=str(self.mgr.get_audio_cache_dir(self.epub_hash)),
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            sample_rate=24000,
            started_at=now,
            updated_at=now,
            completed_titles=[],
        )
        defaults.update(overrides)
        return CheckpointData(**defaults)

    # Persistence

    def test_save_and_load_roundtrip(self):
        """Save then load should return equivalent data."""
        data = self._make_checkpoint_data()
        self.mgr.save(data)

        loaded = self.mgr.load(self.epub_hash)
        assert loaded is not None
        assert loaded.epub_file == data.epub_file
        assert loaded.epub_hash == data.epub_hash
        assert loaded.format == data.format
        assert loaded.speaker == data.speaker
        assert loaded.selected_chapters == data.selected_chapters
        assert loaded.completed_chapters == data.completed_chapters

    def test_load_returns_none_when_no_checkpoint(self):
        """load() should return None when no checkpoint file exists."""
        result = self.mgr.load(self.epub_hash)
        assert result is None

    def test_load_raises_on_corrupted_json(self):
        """load() should raise CheckpointCorruptedError for invalid JSON."""
        ckpt_path = self.mgr.get_checkpoint_path(self.epub_hash)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text("not valid json {{{")

        with pytest.raises(CheckpointCorruptedError, match="Cannot read JSON"):
            self.mgr.load(self.epub_hash)

    def test_load_raises_on_wrong_schema_version(self):
        """load() should raise CheckpointCorruptedError for wrong schema version."""
        data = self._make_checkpoint_data()
        self.mgr.save(data)

        # Tamper with version
        ckpt_path = self.mgr.get_checkpoint_path(self.epub_hash)
        raw = json.loads(ckpt_path.read_text())
        raw["version"] = 999
        ckpt_path.write_text(json.dumps(raw))

        with pytest.raises(CheckpointCorruptedError, match="Unsupported schema version"):
            self.mgr.load(self.epub_hash)

    def test_save_is_atomic(self):
        """Save should not leave partial checkpoint files on disk."""
        data = self._make_checkpoint_data()
        self.mgr.save(data)

        ckpt_path = self.mgr.get_checkpoint_path(self.epub_hash)
        assert ckpt_path.exists()

        # No .tmp file should remain
        tmp_path = ckpt_path.with_suffix(".tmp")
        assert not tmp_path.exists()

    def test_save_updates_updated_at(self):
        """save() should update the updated_at timestamp."""
        data = self._make_checkpoint_data()
        original_updated_at = data.updated_at
        self.mgr.save(data)

        loaded = self.mgr.load(self.epub_hash)
        assert loaded is not None
        # updated_at should have been refreshed by save()
        assert loaded.updated_at >= original_updated_at

    # Validation

    def test_validate_passes_with_matching_hash(self):
        """validate() should pass when EPUB hash matches."""
        data = self._make_checkpoint_data()
        # Should not raise
        self.mgr.validate(
            data,
            epub_hash=self.epub_hash,
            speaker="Aiden",
            language="Auto",
            instruct=None,
            format="mp3",
        )

    def test_validate_raises_on_hash_mismatch(self):
        """validate() should raise CheckpointMismatchError for wrong hash."""
        data = self._make_checkpoint_data()
        with pytest.raises(CheckpointMismatchError, match="epub_hash"):
            self.mgr.validate(
                data,
                epub_hash="sha256:0000000000000000",
                speaker="Aiden",
                language="Auto",
                instruct=None,
                format="mp3",
            )

    def test_validate_warns_on_setting_mismatch(self, caplog):
        """validate() should log warnings for non-hash setting differences."""
        data = self._make_checkpoint_data()
        import logging

        with caplog.at_level(logging.WARNING, logger="lalo.checkpoint"):
            self.mgr.validate(
                data,
                epub_hash=self.epub_hash,
                speaker="Ryan",  # Different from checkpoint's "Aiden"
                language="English",  # Different from checkpoint's "Auto"
                instruct="Speak softly",  # Different from checkpoint's None
                format="wav",  # Different from checkpoint's "mp3"
            )

        assert "speaker" in caplog.text
        assert "language" in caplog.text
        assert "instruct" in caplog.text
        assert "format" in caplog.text

    # Chapter tracking

    def test_mark_chapter_completed(self):
        """mark_chapter_completed() should add chapter and persist."""
        data = self._make_checkpoint_data()
        self.mgr.save(data)

        self.mgr.mark_chapter_completed(data, 1, "Chapter 1: Intro")
        assert 1 in data.completed_chapters
        assert "Chapter 1: Intro" in data.completed_titles

        # Verify it was persisted
        loaded = self.mgr.load(self.epub_hash)
        assert loaded is not None
        assert 1 in loaded.completed_chapters

    def test_mark_chapter_completed_idempotent(self):
        """Marking the same chapter twice should not duplicate it."""
        data = self._make_checkpoint_data()
        self.mgr.save(data)

        self.mgr.mark_chapter_completed(data, 3, "Chapter 3: Test")
        self.mgr.mark_chapter_completed(data, 3, "Chapter 3: Test")
        assert data.completed_chapters.count(3) == 1

    def test_get_remaining_chapters(self):
        """get_remaining_chapters() should return uncompleted chapters in order."""
        data = self._make_checkpoint_data(
            selected_chapters=[1, 2, 3, 4, 5],
            completed_chapters=[1, 3],
        )
        remaining = CheckpointManager.get_remaining_chapters(data)
        assert remaining == [2, 4, 5]

    def test_get_remaining_chapters_all_completed(self):
        """get_remaining_chapters() should return empty list when all done."""
        data = self._make_checkpoint_data(
            selected_chapters=[1, 2, 3],
            completed_chapters=[1, 2, 3],
        )
        remaining = CheckpointManager.get_remaining_chapters(data)
        assert remaining == []

    # Audio file verification

    def test_verify_audio_files_with_all_present(self):
        """verify_audio_files() should return all files when all exist."""
        data = self._make_checkpoint_data(
            completed_chapters=[1, 2],
            completed_titles=["Ch 1", "Ch 2"],
        )
        audio_dir = Path(data.chapter_audio_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Create fake WAV files
        sample = np.zeros(24000, dtype=np.float32)
        for ch in [1, 2]:
            sf.write(str(audio_dir / f"chapter_{ch:04d}.wav"), sample, 24000)

        valid = self.mgr.verify_audio_files(data)
        assert len(valid) == 2
        assert data.completed_chapters == [1, 2]

    def test_verify_audio_files_removes_missing(self):
        """verify_audio_files() should remove chapters with missing audio."""
        data = self._make_checkpoint_data(
            completed_chapters=[1, 2, 3],
            completed_titles=["Ch 1", "Ch 2", "Ch 3"],
        )
        audio_dir = Path(data.chapter_audio_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Only create WAV for chapter 1 and 3 — chapter 2 is missing
        sample = np.zeros(24000, dtype=np.float32)
        sf.write(str(audio_dir / "chapter_0001.wav"), sample, 24000)
        sf.write(str(audio_dir / "chapter_0003.wav"), sample, 24000)

        self.mgr.save(data)  # Persist so verify can re-save

        valid = self.mgr.verify_audio_files(data)
        assert len(valid) == 2
        assert data.completed_chapters == [1, 3]
        assert data.completed_titles == ["Ch 1", "Ch 3"]

    # Cleanup

    def test_cleanup_deletes_checkpoint_and_audio(self):
        """cleanup() should remove the entire checkpoint directory."""
        data = self._make_checkpoint_data()
        self.mgr.save(data)

        # Create audio dir with a file
        audio_dir = Path(data.chapter_audio_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "chapter_0001.wav").write_bytes(b"fake audio")

        ckpt_dir = self.mgr.get_checkpoint_dir(self.epub_hash)
        assert ckpt_dir.exists()

        self.mgr.cleanup(self.epub_hash)
        assert not ckpt_dir.exists()

    def test_cleanup_noop_when_no_checkpoint(self):
        """cleanup() should not raise when no checkpoint exists."""
        # Should not raise
        self.mgr.cleanup(self.epub_hash)

    # Cache directory resolution

    def test_cache_dir_uses_provided_override(self):
        """Should use the explicitly provided cache_dir."""
        custom = self.tmpdir / "custom_cache"
        mgr = CheckpointManager(self.epub_path, self.output_path, cache_dir=str(custom))
        ckpt_dir = mgr.get_checkpoint_dir(self.epub_hash)
        assert str(ckpt_dir).startswith(str(custom))

    def test_cache_dir_respects_xdg(self, monkeypatch):
        """Should use XDG_CACHE_HOME when set."""
        xdg_cache = self.tmpdir / "xdg_cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache))
        monkeypatch.setattr("lalo.checkpoint.CHECKPOINT_CACHE_DIR", None)

        mgr = CheckpointManager(self.epub_path, self.output_path)
        ckpt_dir = mgr.get_checkpoint_dir(self.epub_hash)
        assert str(ckpt_dir).startswith(str(xdg_cache / "lalo"))

    def test_cache_dir_default_fallback(self, monkeypatch):
        """Should fall back to ~/.cache/lalo when no XDG or override."""
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr("lalo.checkpoint.CHECKPOINT_CACHE_DIR", None)

        mgr = CheckpointManager(self.epub_path, self.output_path)
        ckpt_dir = mgr.get_checkpoint_dir(self.epub_hash)
        assert ".cache/lalo" in str(ckpt_dir)

    # Factory helper

    def test_create_checkpoint(self):
        """create_checkpoint() should persist a fresh checkpoint."""
        data = self.mgr.create_checkpoint(
            epub_hash=self.epub_hash,
            format="mp3",
            speaker="Aiden",
            language="Auto",
            instruct=None,
            streaming=True,
            parallel=False,
            selected_chapters=[1, 2, 3],
            model_name="test-model",
            sample_rate=24000,
        )
        assert data.version == CHECKPOINT_SCHEMA_VERSION
        assert data.completed_chapters == []
        assert len(data.selected_chapters) == 3

        # Should be on disk
        loaded = self.mgr.load(self.epub_hash)
        assert loaded is not None
        assert loaded.epub_hash == self.epub_hash

        # Audio dir should exist
        audio_dir = Path(data.chapter_audio_dir)
        assert audio_dir.exists()

    def test_load_backwards_compat_no_completed_titles(self):
        """load() should handle checkpoints without completed_titles field."""
        data = self._make_checkpoint_data()
        self.mgr.save(data)

        # Remove completed_titles from the JSON
        ckpt_path = self.mgr.get_checkpoint_path(self.epub_hash)
        raw = json.loads(ckpt_path.read_text())
        del raw["completed_titles"]
        ckpt_path.write_text(json.dumps(raw))

        loaded = self.mgr.load(self.epub_hash)
        assert loaded is not None
        assert loaded.completed_titles == []
