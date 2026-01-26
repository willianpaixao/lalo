"""
Utility functions for end-to-end tests.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import soundfile as sf
from mutagen.mp4 import MP4


class AudioFileInfo(TypedDict):
    """Audio file information."""

    duration: float
    sample_rate: int
    channels: int
    file_size: int
    codec: str
    bitrate: int
    format: str


def validate_audio_file(file_path: Path, expected_format: str) -> AudioFileInfo:
    """
    Validate audio file properties using ffprobe.

    Args:
        file_path: Path to audio file
        expected_format: Expected format (mp3, wav, m4b)

    Returns:
        dict with keys: duration, sample_rate, channels, file_size, codec
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Use ffprobe to get detailed audio information
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse ffprobe output: {e}") from e

    # Extract audio stream info
    audio_stream = None
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    if not audio_stream:
        raise ValueError(f"No audio stream found in {file_path}")

    # Extract format info
    format_info = probe_data.get("format", {})

    return {
        "duration": float(format_info.get("duration", 0)),
        "sample_rate": int(audio_stream.get("sample_rate", 0)),
        "channels": int(audio_stream.get("channels", 0)),
        "file_size": int(format_info.get("size", 0)),
        "codec": str(audio_stream.get("codec_name", "unknown")),
        "bitrate": int(format_info.get("bit_rate", 0)),
        "format": str(expected_format),
    }


class AudioQualityInfo(TypedDict):
    """Audio quality information."""

    has_audio: bool
    has_silence: bool
    has_clipping: bool
    has_long_silence: bool
    duration: float
    rms: float
    max_amplitude: float
    sample_rate: int


def check_audio_quality(file_path: Path, min_duration: float = 0.5) -> AudioQualityInfo:
    """
    Check audio quality for common issues using soundfile and ffmpeg.

    Args:
        file_path: Path to audio file
        min_duration: Minimum expected duration in seconds

    Returns:
        dict with boolean flags: has_audio, has_silence, has_clipping
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Read audio file with soundfile
    try:
        data, sr = sf.read(str(file_path))
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file: {e}") from e

    # Convert to numpy array if needed
    if isinstance(data, list):
        data = np.array(data)

    # Handle stereo vs mono
    if data.ndim > 1:
        # Convert stereo to mono for analysis
        data = np.mean(data, axis=1)

    duration = len(data) / sr

    # Check for silence (RMS below threshold)
    rms = np.sqrt(np.mean(data**2))
    has_silence = rms < 0.001

    # Check for clipping (samples at max value)
    max_val = np.max(np.abs(data))
    has_clipping = max_val > 0.99

    # Detect long silence periods (>2 seconds of near-silence)
    # Sliding window approach
    window_size = int(2 * sr)  # 2 seconds
    has_long_silence = False
    if len(data) > window_size:
        for i in range(0, len(data) - window_size, window_size // 2):
            window = data[i : i + window_size]
            window_rms = np.sqrt(np.mean(window**2))
            if window_rms < 0.001:
                has_long_silence = True
                break

    return {
        "has_audio": duration >= min_duration,
        "has_silence": has_silence,
        "has_clipping": has_clipping,
        "has_long_silence": has_long_silence,
        "duration": duration,
        "rms": float(rms),
        "max_amplitude": float(max_val),
        "sample_rate": sr,
    }


def validate_m4b_chapters(file_path: Path, expected_chapter_count: int) -> dict[str, Any]:
    """
    Validate M4B file has correct chapter markers using mutagen.

    Args:
        file_path: Path to M4B file
        expected_chapter_count: Expected number of chapters

    Returns:
        dict with validation results
    """
    if not file_path.exists():
        raise FileNotFoundError(f"M4B file not found: {file_path}")

    try:
        audio = MP4(str(file_path))
    except Exception as e:
        raise RuntimeError(f"Failed to read M4B file: {e}") from e

    # Extract chapters from M4B
    chapters = []
    chapter_titles = []

    # M4B chapters are stored in the chpl atom (chapter list)
    if hasattr(audio, "tags") and audio.tags:
        # Try to find chapter information
        # Mutagen stores chapters in a specific way for M4B
        for key, value in audio.tags.items():
            if "chpl" in str(key).lower() or "chapter" in str(key).lower():
                chapters.append(value)

    # Alternative: Use ffprobe to get chapter information
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_chapters",
        str(file_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        ffprobe_chapters = probe_data.get("chapters", [])

        for chapter in ffprobe_chapters:
            chapter_info = {
                "id": chapter.get("id"),
                "start_time": float(chapter.get("start_time", 0)),
                "end_time": float(chapter.get("end_time", 0)),
                "title": chapter.get("tags", {}).get("title", "Unknown"),
            }
            chapter_titles.append(chapter_info["title"])
            chapters.append(chapter_info)

    except (subprocess.CalledProcessError, json.JSONDecodeError):
        # If ffprobe fails, fall back to mutagen-only detection
        pass

    # Get metadata
    metadata: dict[str, str] = {}
    if audio.tags:
        title_tag = audio.tags.get("\xa9nam")
        author_tag = audio.tags.get("\xa9ART")
        album_tag = audio.tags.get("\xa9alb")

        metadata = {
            "title": title_tag[0] if title_tag and isinstance(title_tag, list) else "Unknown",
            "author": author_tag[0] if author_tag and isinstance(author_tag, list) else "Unknown",
            "album": album_tag[0] if album_tag and isinstance(album_tag, list) else "Unknown",
        }

    return {
        "chapter_count": len(chapters),
        "expected_count": expected_chapter_count,
        "chapters_match": len(chapters) == expected_chapter_count,
        "chapter_titles": chapter_titles,
        "chapters": chapters,
        "metadata": metadata,
    }


def get_audio_duration(file_path: Path) -> float:
    """
    Get audio file duration using ffprobe.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(file_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        return float(probe_data.get("format", {}).get("duration", 0))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to get audio duration: {e}") from e


def estimate_expected_duration(
    text_content: str, words_per_minute: int = 150, buffer_factor: float = 1.2
) -> float:
    """
    Estimate expected audio duration from text content.

    Args:
        text_content: Text content to convert
        words_per_minute: Average speaking rate (default: 150 WPM)
        buffer_factor: Multiplier for safety margin (default: 1.2 = 20% buffer)

    Returns:
        Estimated duration in seconds
    """
    word_count = len(text_content.split())
    duration_minutes = word_count / words_per_minute
    duration_seconds = duration_minutes * 60 * buffer_factor
    return duration_seconds


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg and ffprobe are available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def parse_cli_output(output: str) -> dict[str, str | int | None]:
    """
    Parse CLI output for common patterns.

    Args:
        output: CLI output text

    Returns:
        dict with extracted information
    """
    info: dict[str, str | int | None] = {
        "title": None,
        "author": None,
        "chapters": None,
        "duration": None,
        "output_file": None,
    }

    # Extract title
    title_match = re.search(r"Found:\s*(.+?)\s+by\s+(.+)", output)
    if title_match:
        info["title"] = str(title_match.group(1).strip())
        info["author"] = str(title_match.group(2).strip())

    # Extract chapter count
    chapters_match = re.search(r"Total chapters:\s*(\d+)", output)
    if chapters_match:
        info["chapters"] = int(chapters_match.group(1))

    # Extract duration
    duration_match = re.search(r"Duration:\s*(.+)", output)
    if duration_match:
        info["duration"] = str(duration_match.group(1).strip())

    # Extract output file
    output_match = re.search(r"Output:\s*(.+)", output)
    if output_match:
        info["output_file"] = str(output_match.group(1).strip())

    return info
