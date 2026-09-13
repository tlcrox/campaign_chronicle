#!/usr/bin/env python3
"""
text.py - reading files a human may have saved in the wrong encoding.

Python's ``open()`` and ``read_text()`` use the locale encoding on Windows,
usually cp1252, so every read in this codebase pins ``encoding="utf-8"``. That
makes the two platforms behave the same, but it is not enough for files people
hand-edit: Notepad and similar still save cp1252 by default, and an em-dash or
smart quote in such a file is not valid UTF-8. The resulting UnicodeDecodeError
is a ValueError, so an ``except OSError`` around the read will not catch it and
the run dies with a traceback.

``read_text_lenient`` is for those files — config.yaml, speaker_config.json,
the recognition hint lists. Tool-written files (offset.txt, roi_history.json,
transcripts) stay pinned to UTF-8 with no fallback: a bad byte there is
corruption, not a save-as mistake.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_text_lenient(path) -> str:
    """Read a text file as UTF-8, falling back to cp1252 with a warning."""
    path = Path(path)
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning(
            f"{path} is not valid UTF-8; falling back to cp1252. "
            f"Re-save it as UTF-8 to silence this.")
        return data.decode("cp1252", errors="replace")
