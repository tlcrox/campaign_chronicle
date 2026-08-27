#!/usr/bin/env python3
"""
sessions.py - session discovery and audio/video source detection.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Extensions that CARRY audio we can transcribe — video containers included,
# since a session with no Audacity project is transcribed straight from the
# video's embedded track.
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm', '.aac', '.mp4', '.mkv'}
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v', '.flv', '.wmv'}

# Audio that stands ALONE, which is a different question from the above and the
# one session classification asks. A video's audio is embedded and belongs to the
# video workflow; classifying a video-only session as "standalone audio" routes
# it to the audio tool, where it never gets diarization-based speaker mapping.
AUDIO_ONLY_EXTENSIONS = AUDIO_EXTENSIONS - VIDEO_EXTENSIONS


class NoSessionsError(Exception):
    """Raised when no session directories were provided to process."""


def find_sessions(session_dirs: List[Path] = None) -> List[Path]:
    """
    Resolve the explicit list of session folders to process.

    Args:
        session_dirs: Explicit session directories to process.

    Returns:
        Sorted list of existing session directories.

    Raises:
        NoSessionsError: if ``session_dirs`` is empty/None, or if every entry was
            skipped (fail loud rather than silently process nothing).

    This is the ONLY place session directories are checked for existence.
    Config reads configuration; orchestrate resolves paths; validation happens
    here, once, and reports what it dropped rather than quietly returning a
    shorter list.
    """
    if not session_dirs:
        raise NoSessionsError(
            "No session directories to process. Pass --session-dirs, or set "
            "orchestration.session_dirs in config.yaml."
        )

    sessions, skipped = [], []
    logger.info(f"Using {len(session_dirs)} explicit session director(ies)")
    for d in session_dirs:
        if d.is_dir():
            sessions.append(d)
            logger.info(f"  - {d.name}")
        else:
            skipped.append(d)
            reason = "not a directory" if d.exists() else "does not exist"
            logger.warning(f"  x Session directory skipped ({reason}): {d}")

    if skipped:
        logger.warning(
            f"  {len(skipped)} of {len(session_dirs)} session director(ies) skipped"
        )
    if not sessions:
        raise NoSessionsError(
            f"None of the {len(session_dirs)} requested session director(ies) exist. "
            f"Checked: {', '.join(str(d) for d in session_dirs[:5])}"
            + (" ..." if len(session_dirs) > 5 else "")
        )
    return sorted(sessions)


def media_files(directory: Path, extensions: set, recursive: bool = False) -> List[Path]:
    """Files under ``directory`` whose extension is in ``extensions``, sorted.

    One walk, filtering on ``.suffix``, rather than a glob per extension: the
    walk is paid once instead of ten times, which matters most for the recursive
    search of an Audacity project folder.

    Two things fall out of comparing suffixes rather than pattern-matching them.
    The comparison is lowercased, so ``.MP4`` is found on Linux as well as on
    Windows — where the filesystem, not the code, was providing the
    case-insensitivity the comments claim. And ``is_file()`` excludes a
    *directory* named ``something.mp4``, which a bare glob happily returned.

    Sorted because filesystem order is not defined; no caller depends on the
    order today, and a stable one is cheaper to reason about than an arbitrary one.
    """
    walk = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(p for p in walk if p.is_file() and p.suffix.lower() in extensions)


def detect_audio_source(session_dir: Path) -> Tuple[str, List[Path]]:
    """
    Detect audio source type and return (source_type, audio_files).

    Source types — exactly the three the callers distinguish:
    - 'audacity': Audacity project with exported audio
    - 'standalone': separate audio files, no video
    - 'video': everything else, transcribed from the video's embedded track.
      An Audacity project whose exports are missing lands here too: there is no
      separate audio to prefer, which is the only thing the callers ask.

    Returns:
        Tuple of (source_type, list_of_audio_files)
    """
    # Look for Audacity projects
    aup_files = list(session_dir.glob("*.aup3")) + list(session_dir.glob("*.aup"))

    if aup_files:
        # Audacity project found - look for exported audio in subdirectories ONLY
        # Do NOT look in root directory (which contains video files)
        # Preference: Audacity exports (high quality) > video embedded audio
        audio_files = []

        # Look in expanded Audacity directories (craig/*, Issue_*/*, etc.)
        # These contain the exported audio from Audacity, sometimes several
        # levels down, hence the recursive walk.
        for expanded_dir in session_dir.glob("*"):
            if expanded_dir.is_dir():
                audio_files.extend(
                    media_files(expanded_dir, AUDIO_EXTENSIONS, recursive=True))

        if audio_files:
            return "audacity", audio_files
        else:
            # No exported audio found in subdirectories
            # Will fall back to video embedded audio (always available)
            logger.warning(f"  ⊘ Audacity project found but no exported audio files. Will use video audio.")
            return "video", []

    # Look for standalone audio files — see AUDIO_ONLY_EXTENSIONS for why this is
    # not simply AUDIO_EXTENSIONS.
    audio_files = media_files(session_dir, AUDIO_ONLY_EXTENSIONS)

    if audio_files:
        return "standalone", audio_files

    # Only video present -> audio is embedded in the video.
    return "video", []

def find_video_files(session_dir: Path) -> List[Path]:
    """Video files directly in the session directory, sorted by name."""
    return media_files(session_dir, VIDEO_EXTENSIONS)


def probe_duration(video_path: Path) -> Optional[float]:
    """Return a video's duration in seconds via ffprobe, or None if unavailable.

    Shared by the scene and transcript merges so both offset multi-video
    timelines by identical per-video durations. Graceful: if ffprobe isn't
    installed or the probe fails, returns None so the caller can fall back.
    """
    if not shutil.which("ffprobe"):
        return None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=60,
            # Decode explicitly; a non-ASCII byte in ffprobe's output must not
            # crash the capture thread under the Windows cp1252 locale default.
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return None
