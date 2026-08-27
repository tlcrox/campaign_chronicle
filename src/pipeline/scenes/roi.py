#!/usr/bin/env python3
"""
roi.py — the reusable ROI ("Region of Interest") module for the WhisperX
scene-detection pipeline.

It provides two distinct things, both usable by any other Python process:

* ``RoiFile`` — an object model for the multi-ROI *file* (``roi_history.json``):
  find, read, parse, and query the hierarchical, time-varying, per-video ROIs.
* ``resolve_single_roi`` — precedence logic for the *single* fixed ROI value
  used by single-ROI scene detection (CLI arg > ``config.scenes.roi`` > full
  frame). This does NOT read the ROI file; the two concerns are separate.

``RoiFile`` isolates *all* manipulation of the ROI file (find, read, parse) so
callers never re-implement JSON parsing, format detection, or timestamp math.

Canonical ROI file format (the ONLY supported format)
=====================================================
An ROI file is a JSON document whose **top-level keys are video filenames**.
Each video maps to an optional ``_metadata`` block (e.g. ``fps``) plus one or
more entries keyed by an ``HH:MM:SS`` timestamp; each entry holds the crop
rectangle that becomes valid at that moment:

    {
      "2026-04-20 15-39-46.mkv": {
        "_metadata": {"fps": 60.0},
        "00:00:00": {"frame": 0,     "roi": "487 188 1368 811"},
        "00:05:16": {"frame": 18991, "roi": "705 91 827 127"}
      },
      "2026-04-20 18-47-12.mkv": {
        "_metadata": {"fps": 60.0},
        "00:00:00": {"frame": 0, "roi": "1464 68 1564 119"}
      }
    }

Coordinates are ``"x1 y1 x2 y2"`` (top-left and bottom-right, in pixels).

This layout matches the real files in ``TestWhisper/*/roi_history.json``. A
"flat" layout (timestamp keys at the top level, with no video key) is **not**
supported and is rejected with :class:`RoiParseError`.

Quick start
===========
    from pipeline.scenes.roi import RoiFile, resolve_single_roi

    roi = RoiFile.load("session/roi_history.json")
    for video in roi.videos:
        for seg in roi.segments(video):
            print(seg.start, seg.end, seg.roi)

    # Look up the ROI active at a given moment (video name required):
    roi.roi_at(316, video="2026-04-20 15-39-46.mkv")   # -> "705 91 827 127"

    # Resolve against config.yaml without knowing the path:
    roi = RoiFile.from_config()               # uses scenes.roi_file + source_dir

    # Single fixed ROI value (unrelated to the ROI file):
    resolve_single_roi(None, cfg)             # -> config.scenes.roi, or "" (full frame)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
import pipeline.common.timecode

PathLike = Union[str, Path]

# Sentinel used for the end time of the final segment: "process to end of video".
END_OF_VIDEO: int = -1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RoiError(Exception):
    """Base class for all ROI file errors."""


class RoiFileNotFoundError(RoiError):
    """The ROI file could not be located on disk."""


class RoiParseError(RoiError):
    """The ROI file exists but its contents are invalid."""


class RoiVideoNotFoundError(RoiError):
    """A requested video key is not present in the ROI file."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoiEntry:
    """A single timestamped ROI entry as stored in the file."""

    timestamp: str          # "HH:MM:SS"
    start: int              # seconds (derived from timestamp)
    roi: str                # "x1 y1 x2 y2"
    frame: Optional[int] = None

    @property
    def coords(self) -> Tuple[int, int, int, int]:
        """ROI as an ``(x1, y1, x2, y2)`` integer tuple."""
        parts = self.roi.split()
        if len(parts) != 4:
            raise RoiParseError(f"ROI must have 4 values, got {self.roi!r}")
        try:
            return tuple(int(p) for p in parts)  # type: ignore[return-value]
        except ValueError as exc:
            raise RoiParseError(f"ROI values must be integers: {self.roi!r}") from exc


@dataclass(frozen=True)
class RoiSegment:
    """A half-open time span ``[start, end)`` during which ``roi`` applies.

    ``end`` is :data:`END_OF_VIDEO` (-1) for the final segment, meaning
    "process through the end of the video".
    """

    start: int              # seconds (inclusive)
    end: int                # seconds (exclusive), or END_OF_VIDEO
    roi: str                # "x1 y1 x2 y2"
    description: str
    frame: Optional[int] = None

    @property
    def is_final(self) -> bool:
        return self.end == END_OF_VIDEO

    def contains(self, seconds: float) -> bool:
        if seconds < self.start:
            return False
        if self.is_final:
            return True
        return seconds < self.end

    def as_pipe(self) -> str:
        """Render as ``start|end|roi|description`` (bash-friendly)."""
        return f"{self.start}|{self.end}|{self.roi}|{self.description}"


def _looks_like_timestamp(key: str) -> bool:
    """True if ``key`` is an ``HH:MM:SS`` style timestamp."""
    return isinstance(key, str) and key.count(":") == 2


# ---------------------------------------------------------------------------
# The class
# ---------------------------------------------------------------------------
class RoiFile:
    """Object model for an ROI configuration file (hierarchical / video-keyed).

    Construct directly from a path with :meth:`load`, from already-parsed data
    with :meth:`from_dict`, by searching directories with :meth:`find`, or from
    the project ``config.yaml`` with :meth:`from_config`.
    """

    METADATA_KEY = "_metadata"

    def __init__(self, data: dict, path: Optional[Path] = None):
        if not isinstance(data, dict):
            raise RoiParseError("ROI data must be a JSON object at the top level")
        self._validate_layout(data)
        self._data: dict = data
        self.path: Optional[Path] = Path(path) if path is not None else None

    # -- construction --------------------------------------------------------
    @classmethod
    def load(cls, path: PathLike) -> "RoiFile":
        """Read and parse an ROI file from ``path``."""
        p = Path(path)
        if not p.exists():
            raise RoiFileNotFoundError(f"ROI file not found: {p}")
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise RoiParseError(f"Invalid JSON in {p}: {exc}") from exc
        except OSError as exc:
            raise RoiParseError(f"Failed to read {p}: {exc}") from exc
        return cls(data, path=p)

    @classmethod
    def from_dict(cls, data: dict, path: Optional[PathLike] = None) -> "RoiFile":
        """Build directly from an already-parsed dict (useful for tests)."""
        return cls(data, path=Path(path) if path else None)

    @classmethod
    def find(cls, name: str, search_dirs: List[PathLike]) -> "RoiFile":
        """Locate ``name`` and load it.

        If ``name`` is an absolute path it is used directly; otherwise each
        directory in ``search_dirs`` is checked in order for ``dir/name``.
        Raises :class:`RoiFileNotFoundError` if nothing matches.
        """
        candidate = Path(name)
        if candidate.is_absolute():
            return cls.load(candidate)

        tried: List[str] = []
        for d in search_dirs:
            p = Path(d) / name
            tried.append(str(p))
            if p.exists():
                return cls.load(p)
        raise RoiFileNotFoundError(
            f"ROI file {name!r} not found. Searched: {', '.join(tried) or '(no dirs)'}"
        )

    @classmethod
    def resolve_path(cls, name: str, search_dirs: List[PathLike]) -> Optional[Path]:
        """Return the resolved path for ``name`` without loading it, or None."""
        candidate = Path(name)
        if candidate.is_absolute():
            return candidate if candidate.exists() else None
        for d in search_dirs:
            p = Path(d) / name
            if p.exists():
                return p
        return None

    @classmethod
    def from_config(cls, config=None) -> Optional["RoiFile"]:
        """Build from the project ``config.yaml`` (``scenes.roi_file``).

        Resolves the configured filename against ``config.source_dir``. Returns
        ``None`` when no ``roi_file`` is configured. Imports ``config`` lazily so
        this module has no hard dependency on it.
        """
        if config is None:
            from pipeline.config import get_config  # local import: optional dependency
            config = get_config()

        roi_file_name = config.get("scenes", "roi_file")
        if not roi_file_name:
            return None
        return cls.find(roi_file_name, [config.source_dir])

    # -- layout validation ---------------------------------------------------
    @classmethod
    def _validate_layout(cls, data: dict) -> None:
        """Ensure ``data`` is the canonical hierarchical (video-keyed) layout.

        The legacy "flat" layout (top-level ``HH:MM:SS`` keys) is rejected: a
        top-level timestamp key is the unambiguous signature of a flat file.
        """
        video_keys = [k for k in data if k != cls.METADATA_KEY]
        if not video_keys:
            raise RoiParseError("ROI file has no video entries")
        for key in video_keys:
            if _looks_like_timestamp(key):
                raise RoiParseError(
                    f"Top-level timestamp key {key!r} found: the flat ROI format "
                    "is not supported. Wrap timestamp entries under a "
                    "video-filename key (hierarchical format)."
                )
            if not isinstance(data[key], dict):
                raise RoiParseError(
                    f"Video entry {key!r} must be a JSON object of timestamp entries"
                )

    # -- format introspection ------------------------------------------------
    @property
    def videos(self) -> List[str]:
        """Sorted list of video keys."""
        return sorted(k for k in self._data if k != self.METADATA_KEY)

    # -- per-video data access ----------------------------------------------
    def _block_for(self, video: str) -> dict:
        """Return the dict holding timestamp entries for ``video``."""
        if video is None:
            raise RoiVideoNotFoundError(
                "A video name is required. Available: " + ", ".join(self.videos)
            )
        if video not in self._data:
            raise RoiVideoNotFoundError(
                f"Video {video!r} not found. Available: {self.videos}"
            )
        return self._data[video]

    def metadata(self, video: str) -> dict:
        """Return the ``_metadata`` block (fps, etc.) for ``video`` or ``{}``."""
        block = self._block_for(video)
        meta = block.get(self.METADATA_KEY, {})
        return meta if isinstance(meta, dict) else {}

    def fps(self, video: str) -> Optional[float]:
        """Frames-per-second from ``video``'s metadata, if present."""
        fps = self.metadata(video).get("fps")
        return float(fps) if fps is not None else None

    def entries(self, video: str) -> List[RoiEntry]:
        """Return all ROI entries for ``video``, sorted by time.

        Entries lacking a usable ``roi`` are skipped (a warning-free skip; use
        :meth:`validate` for strict checking).
        """
        block = self._block_for(video)
        out: List[RoiEntry] = []
        for key, value in block.items():
            if key == self.METADATA_KEY or str(key).startswith("_"):
                continue
            if not _looks_like_timestamp(key):
                continue
            if not isinstance(value, dict) or not value.get("roi"):
                continue
            out.append(
                RoiEntry(
                    timestamp=key,
                    # timestamp_to_seconds() returns float (it's the shared,
                    # sub-second-capable parser); RoiEntry.start/RoiSegment.start
                    # /end are declared int (ROI files are whole-second HH:MM:SS
                    # only), so cast here once rather than downstream at each
                    # printed/rendered call site.
                    start=int(pipeline.common.timecode.timestamp_to_seconds(key)),
                    roi=value["roi"],
                    frame=value.get("frame"),
                )
            )
        out.sort(key=lambda e: e.start)
        return out

    def segments(self, video: str) -> List[RoiSegment]:
        """Convert ``video``'s entries into contiguous half-open segments.

        Each segment runs until the next entry's start; the last segment ends
        at :data:`END_OF_VIDEO` (-1).
        """
        entries = self.entries(video)
        if not entries:
            raise RoiParseError(f"No valid ROI entries found for video {video!r}")
        segs: List[RoiSegment] = []
        for i, entry in enumerate(entries):
            end = entries[i + 1].start if i + 1 < len(entries) else END_OF_VIDEO
            segs.append(
                RoiSegment(
                    start=entry.start,
                    end=end,
                    roi=entry.roi,
                    description=f"ROI at {entry.timestamp}",
                    frame=entry.frame,
                )
            )
        return segs

    def roi_at(self, seconds: float, video: str) -> Optional[str]:
        """Return the ROI string active at ``seconds`` for ``video``.

        Returns ``None`` if ``seconds`` precedes the first entry.
        """
        for seg in self.segments(video):
            if seg.contains(seconds):
                return seg.roi
        return None

    # -- iteration & validation ---------------------------------------------
    def iter_all_segments(self) -> Iterator[Tuple[str, RoiSegment]]:
        """Yield ``(video, segment)`` pairs across every video in the file."""
        for video in self.videos:
            for seg in self.segments(video):
                yield video, seg

    def validate(self) -> List[str]:
        """Return a list of human-readable problems; empty means the file is clean."""
        problems: List[str] = []
        targets = self.videos
        if not targets:
            problems.append("ROI file has no video entries.")
        for video in targets:
            block = self._block_for(video)
            ts_keys = [k for k in block if _looks_like_timestamp(k)]
            label = f"video {video!r}"
            if not ts_keys:
                problems.append(f"No timestamp entries found in {label}.")
            for key in ts_keys:
                value = block[key]
                if not isinstance(value, dict) or not value.get("roi"):
                    problems.append(f"{label}: entry {key} missing 'roi'.")
                    continue
                try:
                    RoiEntry(key, 0, value["roi"]).coords
                except RoiParseError as exc:
                    problems.append(f"{label}: entry {key}: {exc}")
        return problems

    def to_dict(self) -> dict:
        """Return the raw parsed data."""
        return self._data

    def __repr__(self) -> str:
        where = f" path={self.path}" if self.path else ""
        return f"<RoiFile videos={len(self.videos)}{where}>"


# ---------------------------------------------------------------------------
# Single fixed ROI resolution (distinct from the multi-ROI file above)
# ---------------------------------------------------------------------------
def resolve_single_roi(cli_roi: Optional[str], config) -> str:
    """Resolve the effective single ROI crop string for scene detection.

    This is the single-ROI concern only; it never reads the ROI *file*
    (``RoiFile``). Resolution order:

      1. ``cli_roi`` when not ``None`` wins (an explicit ``""`` forces
         full-frame analysis).
      2. Otherwise fall back to ``config.scene_roi`` IF IT IS SET.
      3. Otherwise return ``""`` (full frame).

    Returns the ROI string ("" means full frame).
    """
    if cli_roi is not None:
        # Caller was explicit (including an explicit "" to force full frame).
        return cli_roi.strip()

    config_roi = config.scene_roi
    if config_roi and str(config_roi).strip():
        return str(config_roi).strip()

    return ""


# ---------------------------------------------------------------------------
# CLI — reproduces the `start|end|roi|description` output of the old
# extract_roi_segments.py so it can serve as a drop-in replacement later.
# ---------------------------------------------------------------------------
def _main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect an ROI file and emit segments as start|end|roi|description.",
    )
    parser.add_argument("roi_file", help="Path to the ROI JSON file")
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Video filename (required: ROI files are video-keyed)",
    )
    parser.add_argument(
        "--list-videos", action="store_true", help="List video keys and exit"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate the file and exit"
    )
    args = parser.parse_args(argv)

    try:
        roi = RoiFile.load(args.roi_file)
    except RoiError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.list_videos:
        for v in roi.videos:
            print(v)
        return 0

    if args.validate:
        problems = roi.validate()
        if problems:
            for p in problems:
                print(f"INVALID: {p}", file=sys.stderr)
            return 1
        print("OK")
        return 0

    try:
        segments = roi.segments(args.video)
    except RoiError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for seg in segments:
        print(seg.as_pipe())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
