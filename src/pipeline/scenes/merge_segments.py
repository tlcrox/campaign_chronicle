#!/usr/bin/env python3
"""
merge_segments.py - reassemble per-segment scene CSVs into one CSV for a video.

Multi-ROI scene detection splits a video into time segments, runs scenedetect on
each (producing scene times relative to that segment), and writes the results
into ``<temp_dir>/segment_<i>/``. This step stitches them back together.

Each segment's absolute start time is carried forward by detect_scenes_multi.sh, which
writes ``segment_<i>/offset.txt`` (seconds) at split time. We add that offset to
the segment-relative scene times, renumber scenes sequentially across segments,
and emit absolute timestamps.

Usage:
    python3 -m pipeline.scenes.merge_segments <temp_dir> <output_csv>
    for example
    python -m pipeline.scenes.merge_segments "../output/<video>" "<path>/<source>/<video>/<video>.csv"
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from pipeline.common.timecode import seconds_to_timestamp
from pipeline.common.scenes import format_scene_name

# Columns read from each per-segment PySceneDetect CSV.
START_COL = "Start Time (seconds)"
END_COL = "End Time (seconds)"
LENGTH_FRAMES_COL = "Length (frames)"
START_FRAME_COL = "Start Frame"
END_FRAME_COL = "End Frame"

# Output columns. Absolute Start Timecode + per-scene frame length; all other
# PySceneDetect columns are intentionally dropped.
SCENE_COL = "Scene Number"
START_TC_COL = "Start Timecode"
DURATION_COL = "Duration (seconds)"
OUTPUT_FIELDS = [SCENE_COL, START_TC_COL, START_COL, END_COL, DURATION_COL, LENGTH_FRAMES_COL]


class MergeError(Exception):
    """Raised when segments cannot be merged."""


def _segment_index(p: Path) -> int:
    """Sort key: integer after 'segment_' (e.g. segment_10 -> 10)."""
    try:
        return int(p.name.split("segment_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _find_segment_dirs(temp_dir: Path) -> List[Path]:
    return sorted(
        (d for d in temp_dir.glob("segment_*") if d.is_dir()),
        key=_segment_index,
    )


def _read_offset(segment_dir: Path) -> float:
    """Absolute start (seconds) for a segment, from offset.txt (0.0 if absent)."""
    offset_file = segment_dir / "offset.txt"
    if not offset_file.exists():
        print(f"WARNING: no offset.txt in {segment_dir.name}, assuming 0.0", file=sys.stderr)
        return 0.0
    try:
        return float(offset_file.read_text().strip())
    except ValueError:
        print(f"WARNING: bad offset.txt in {segment_dir.name}, assuming 0.0", file=sys.stderr)
        return 0.0


def _find_scene_csv(segment_dir: Path) -> Optional[Path]:
    matches = sorted(segment_dir.glob("*-Scenes.csv")) or sorted(segment_dir.glob("*Scenes.csv"))
    return matches[0] if matches else None


def _read_header_rows(csv_path: Path) -> Tuple[List[str], List[dict]]:
    """Return (fieldnames, rows). Scenedetect may emit a preamble line before the
    real header, so locate the line that contains the scene-time columns."""
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        lines = fh.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if START_COL in line and END_COL in line:
            header_idx = i
            break
    if header_idx is None:
        raise MergeError(f"{csv_path}: could not find a header with '{START_COL}'")

    reader = csv.DictReader(lines[header_idx:])
    return reader.fieldnames or [], list(reader)


def _scene_frame_length(row: dict):
    """Per-scene length in frames. Prefer the 'Length (frames)' column; fall back
    to End Frame - Start Frame. Returns int, or '' if unavailable. Frame length is
    a count, so it needs no absolute-time offset."""
    val = row.get(LENGTH_FRAMES_COL)
    if val not in (None, ""):
        try:
            return int(float(val))
        except ValueError:
            pass
    try:
        return int(float(row[END_FRAME_COL])) - int(float(row[START_FRAME_COL]))
    except (KeyError, ValueError, TypeError):
        return ""


# PySceneDetect's raw per-segment image: "<segment-stem>-Scene-NNN-MM.ext".
_SEG_IMAGE_RE = re.compile(r".*-Scene-(\d+)-(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def _segment_scene_images(segment_dir: Path) -> dict:
    """Map a segment's original scene number -> its raw image path.

    One image per scene is assumed (num_images == 1, the config default); the
    first image seen for a scene wins.
    """
    out: dict = {}
    for img in sorted(segment_dir.iterdir()):
        if not img.is_file():
            continue
        m = _SEG_IMAGE_RE.match(img.name)
        if m:
            out.setdefault(int(m.group(1)), img)
    return out


def _orig_scene_number(row: dict):
    """The per-segment scene number from a scenedetect CSV row (or None)."""
    try:
        return int(float(row.get(SCENE_COL, "")))
    except (TypeError, ValueError):
        return None


def merge_segments(temp_dir, output_csv, images_out=None, video_index: int = 1) -> int:
    """Reassemble one video's ``segment_*`` outputs into a cohesive, time-ordered
    scene list — and, if ``images_out`` is given, the matching scene images.

    Segment-relative scene times are shifted by each segment's ``offset.txt`` and
    scenes are renumbered sequentially across segments. When ``images_out`` is
    provided, each scene's image is copied there renamed to the canonical
    ``Scene-{video_index:02d}-{scene:03d}.ext`` using the **same** running scene
    number as the CSV — so CSV row N always pairs with ``Scene-VV-NNN`` (the
    CSV↔image alignment is guaranteed by construction, in one place).

    Returns the number of merged scenes. Raises MergeError if nothing merged.
    """
    temp_dir = Path(temp_dir)
    output_csv = Path(output_csv)
    images_out = Path(images_out) if images_out is not None else None

    segment_dirs = _find_segment_dirs(temp_dir)
    if not segment_dirs:
        raise MergeError(f"No segment_* directories found in {temp_dir}")

    merged: List[dict] = []
    image_ops: List[Tuple[Path, str]] = []
    scene_number = 1

    for segment_dir in segment_dirs:
        offset = _read_offset(segment_dir)
        csv_path = _find_scene_csv(segment_dir)
        if csv_path is None:
            print(f"WARNING: no scene CSV in {segment_dir.name}, skipping", file=sys.stderr)
            continue

        _, rows = _read_header_rows(csv_path)
        images = _segment_scene_images(segment_dir) if images_out is not None else {}
        for row in rows:
            try:
                start = float(row[START_COL]) + offset
                end = float(row[END_COL]) + offset
            except (KeyError, ValueError) as exc:
                print(f"WARNING: bad row in {csv_path.name}: {exc}", file=sys.stderr)
                continue
            merged.append({
                SCENE_COL: scene_number,
                START_TC_COL: seconds_to_timestamp(start, millis=True),  # absolute
                START_COL: start,
                END_COL: end,
                DURATION_COL: round(end - start, 3),
                LENGTH_FRAMES_COL: _scene_frame_length(row),
            })
            if images_out is not None:
                orig = _orig_scene_number(row)
                src = images.get(orig) if orig is not None else None
                if src is not None:
                    ext = src.suffix.lstrip(".").lower() or "jpg"
                    image_ops.append((src, format_scene_name(video_index, scene_number, ext)))
                else:
                    print(f"WARNING: no image for scene {orig} in {segment_dir.name}", file=sys.stderr)
            scene_number += 1

    if not merged:
        raise MergeError("No scenes found in any segment")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # newline="" is required by the csv module (it manages its own terminator);
    # lineterminator then has to be set explicitly, because csv's default is
    # "\r\n" on every platform — newline="" only stops a SECOND translation.
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged)

    if images_out is not None:
        images_out.mkdir(parents=True, exist_ok=True)
        for src, dest in image_ops:
            try:
                shutil.copy2(src, images_out / dest)
            except OSError as exc:
                print(f"WARNING: could not copy {src.name} -> {dest}: {exc}", file=sys.stderr)

    print(f"Merged {len(merged)} scenes from {len(segment_dirs)} segment(s)")
    return len(merged)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("Usage: python3 -m pipeline.scenes.merge_segments <temp_dir> <output_csv>", file=sys.stderr)
        return 1
    try:
        merge_segments(argv[0], argv[1])
    except MergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
