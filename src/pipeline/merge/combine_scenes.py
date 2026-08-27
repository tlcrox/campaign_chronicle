#!/usr/bin/env python3
"""
combine_scenes.py - combine a session's per-video scene outputs into one
storyboard input.

The scene half of the old combine_week.py (the transcript half moved to
combine_transcripts.py). Stitches every video's scene CSV and scene images onto
one continuous session timeline: scenes are renumbered/offset and images are
copied into a unified folder under a 2-part (video, scene) key.

pandas is required for CSV merging and guarded at import.

Importable API:
    merge_scene_csvs(csv_files, cfg) -> (df, scene_remap, title_row)
    merge_image_folders(image_dirs, csv_files, output_dir, dry_run)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List

from pipeline.common.scenes import (
    format_scene_name,
    iter_scene_images,
    parse_raw_scene_name,
    parse_scene_name,
)
from pipeline.common.timecode import seconds_to_timestamp

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError as e:
    HAS_PANDAS = False
    pd = None
    import sys
    print(f"WARNING: pandas import failed: {e}", file=sys.stderr)

logger = logging.getLogger(__name__)


def merge_scene_csvs(csv_files: List[Path], cfg, video_durations: dict = None):
    """
    Merge multiple scene detection CSVs onto one continuous session timeline.

    Expects CSVs with only column headers (line 1) and data (lines 2+).
    PySceneDetect's timing metadata line is stripped by detect_scenes.sh.

    Scenes are renumbered sequentially, and each subsequent video's absolute
    time/frame columns are shifted by the cumulative length of the preceding
    videos so the timeline is continuous across videos.

    Args:
        csv_files: per-video scene CSVs, in play order (creation-time order).
        cfg: config object (column names).
        video_durations: optional {csv_file_path: seconds}. When a video's true
            duration is known (e.g. via ffprobe) it is used to shift the NEXT
            video; otherwise the end of the last detected scene is used as a
            proxy (exact unless the video has trailing content with no scene).

    Returns: (merged_dataframe, scene_remapping dict {csv_file_path: offset}, title_row)
    """
    video_durations = video_durations or {}
    if not HAS_PANDAS:
        logger.error("pandas required for CSV merging. Install with: pip install pandas --break-system-packages")
        return None, {}, None

    merged_rows = []
    time_offset = 0.0       # cumulative seconds of the preceding videos
    frame_offset = 0        # cumulative frames of the preceding videos
    scene_remap = {}        # Maps CSV file path to its video index

    scene_num_col = cfg.scene_number_column
    start_time_col = cfg.start_time_column
    end_time_col = cfg.end_time_column
    video_col = cfg.video_column

    # Process in the order given (the caller supplies videos in creation-time
    # order); do NOT re-sort by filename, or the running time offset would be
    # applied in the wrong sequence. The 1-based enumerate index is the video
    # index stamped into the Video column — the 2-part key paired with the
    # per-video Scene Number.
    for video_idx, csv_file in enumerate(csv_files, 1):
        try:
            # Read CSV with default header (line 1 contains column names)
            # PySceneDetect's timing metadata is already removed by detect_scenes.sh
            df = pd.read_csv(csv_file)
        except Exception as e:
            logger.warning(f"Could not load CSV {csv_file}: {e}")
            continue

        logger.debug(f"  CSV columns: {list(df.columns)}")

        if scene_num_col not in df.columns:
            logger.error(f"Scene number column '{scene_num_col}' not found in {csv_file}")
            logger.error(f"  Expected column: '{scene_num_col}'")
            logger.error(f"  Available columns: {list(df.columns)}")
            continue

        # Track this CSV's video index. Scene Number stays per-video; the Video
        # column disambiguates the same scene number across different videos.
        scene_remap[csv_file] = video_idx
        logger.debug(f"  CSV {csv_file.name}: video_idx={video_idx} time_offset={time_offset}")

        # Length of THIS video, used to shift the NEXT one. Prefer a true probed
        # duration; otherwise fall back to the end of the last detected scene.
        proxy_seconds = (
            float(df[end_time_col].max()) if end_time_col in df.columns and len(df) else 0.0
        )
        proxy_frames = (
            int(df["End Frame"].max()) if "End Frame" in df.columns and len(df) else 0
        )
        probed = video_durations.get(csv_file)
        if probed is not None and probed > 0:
            video_seconds = float(probed)
            # Derive frames at this CSV's fps so frame numbers stay consistent
            # with the probed duration.
            fps = (proxy_frames / proxy_seconds) if proxy_seconds > 0 else 0
            video_frames = int(round(video_seconds * fps)) if fps else proxy_frames
        else:
            video_seconds = proxy_seconds
            video_frames = proxy_frames

        # Offset each row onto the continuous session timeline. Scene number +
        # scene_offset; absolute time/frame columns + running offsets; timecodes
        # recomputed from the shifted seconds. Length (*) columns are per-scene
        # deltas and are left untouched.
        for _, row in df.iterrows():
            row_copy = row.copy()
            # Scene Number stays as the per-video number; Video identifies which
            # video it belongs to (the 2-part key the storyboard pairs on).
            row_copy[scene_num_col] = int(row_copy[scene_num_col])
            row_copy[video_col] = video_idx
            if start_time_col in df.columns:
                row_copy[start_time_col] = float(row_copy[start_time_col]) + time_offset
            if end_time_col in df.columns:
                row_copy[end_time_col] = float(row_copy[end_time_col]) + time_offset
            if "Start Frame" in df.columns:
                row_copy["Start Frame"] = int(row_copy["Start Frame"]) + frame_offset
            if "End Frame" in df.columns:
                row_copy["End Frame"] = int(row_copy["End Frame"]) + frame_offset
            if "Start Timecode" in df.columns and start_time_col in df.columns:
                row_copy["Start Timecode"] = seconds_to_timestamp(row_copy[start_time_col], millis=True)
            if "End Timecode" in df.columns and end_time_col in df.columns:
                row_copy["End Timecode"] = seconds_to_timestamp(row_copy[end_time_col], millis=True)
            merged_rows.append(row_copy)

        time_offset += video_seconds
        frame_offset += video_frames

    if not merged_rows:
        logger.warning(f"No valid rows to merge from {len(csv_files)} CSV file(s)")
        return None, scene_remap, None

    merged_df = pd.DataFrame(merged_rows)
    return merged_df, scene_remap, None


def merge_image_folders(image_dirs: List[Path], csv_files: List[Path], output_dir: Path,  dry_run: bool = False) -> None:
    """
    Copy and rename images from multiple source folders to unified folder.

    Each source directory is one video; the video index is assigned by directory
    order (01, 02, 03…). The scene number is taken from the input filename —
    which may already be canonical ``Scene-{video:02d}-{scene:03d}.<ext>`` or a
    raw ``<prefix>-Scene-NNN-MM.<ext>`` — and re-emitted through the single source
    of truth, ``pipeline.common.scenes.format_scene_name``. The source file's own
    extension is preserved, so a png folder stays png; this path also carries the
    hand-captured images, which is why it matches on the supported set rather than
    on the session's configured format.

    The emitted name is ``Scene-{video:02d}-{scene:03d}`` where scene is the
    PER-VIDEO scene number (taken from the source filename) and video is the
    merge-order index. Together they form the 2-part key the storyboard pairs on
    (matched against the merged CSV's Video + Scene Number columns), which still
    survives the user deleting images from the auto-generated set. ``image_dirs``
    must be in the SAME order as ``csv_files`` (both come from the caller in video
    creation-time order); this function does not re-sort them.

    Args:
        image_dirs: image source directories, one per video, in play order
        csv_files: corresponding CSV files (same order as image_dirs)
        output_dir: Output directory for merged images
    """
    if dry_run:
        logger.info(f"    [DRY RUN] Would merge images from {len(image_dirs)} folder(s) to {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # image_dirs is used in the given order (NOT re-sorted); the 1-based index is
    # the video index, matching the Video column merge_scene_csvs stamps.
    for video_idx, img_dir in enumerate(image_dirs, 1):
        if not img_dir.exists():
            logger.warning(f"Image directory not found: {img_dir}")
            continue

        logger.debug(f"  Processing images from {img_dir.name}: video_index={video_idx}")

        image_count = 0
        for img_file in iter_scene_images(img_dir, "*Scene-*"):
            # Keep the PER-VIDEO scene number from the source filename and only
            # reassign the video index (merge order). Accept a canonical
            # "Scene-NN-NNN.<ext>" or a raw PySceneDetect "<prefix>-Scene-NNN-MM.<ext>".
            parsed = parse_scene_name(img_file.name)
            if parsed is not None:
                scene_num = parsed[1]
            else:
                raw = parse_raw_scene_name(img_file.name)
                if raw is None:
                    logger.debug(f"Skipping non-standard image filename: {img_file.name}")
                    continue
                scene_num = raw[0]

            # Copy keeps the source extension: renaming a .png to .jpg would put
            # the wrong bytes behind the name.
            ext = img_file.suffix.lstrip(".").lower()
            new_name = format_scene_name(video_idx, scene_num, ext)
            try:
                shutil.copy2(img_file, output_dir / new_name)
                image_count += 1
            except Exception as e:
                logger.error(f"Failed to copy {img_file.name} to {new_name}: {e}")
