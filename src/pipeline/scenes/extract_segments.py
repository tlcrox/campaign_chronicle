#!/usr/bin/env python3
"""
extract_segments.py - turn an ROI file into scene-detection time segments.

This is the single place the ROI file is read during multi-ROI scene detection.
It is a thin wrapper over ``RoiFile`` (the canonical ROI model), emitting one
segment per line as ``start|end|roi|description`` for the shell driver
(detect_scenes_multi.sh) to consume.

Replaces the old segments/extract_roi_segments.py (which re-implemented ROI
parsing). Output is byte-identical to that script for valid inputs.

Usage:
    python3 -m pipeline.scenes.extract_segments <roi_file.json> [video_name]

``video_name`` is required for hierarchical ROI files (filename keys).
"""

from __future__ import annotations

import sys
from typing import List, Optional

from pipeline.scenes.roi import RoiFile, RoiError


def extract(roi_path: str, video: Optional[str] = None) -> List[str]:
    """Return segment lines (``start|end|roi|description``) for ``video``."""
    roi = RoiFile.load(roi_path)
    return [seg.as_pipe() for seg in roi.segments(video)]


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "Usage: python3 -m pipeline.scenes.extract_segments "
            "<roi_file.json> [video_name]",
            file=sys.stderr,
        )
        print(
            "       video_name is required for hierarchical ROI files",
            file=sys.stderr,
        )
        return 1

    roi_path = argv[0]
    video = argv[1] if len(argv) > 1 else None

    try:
        lines = extract(roi_path, video)
    except RoiError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not lines:
        print("ERROR: No segments extracted from ROI file", file=sys.stderr)
        return 1

    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
