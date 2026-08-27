#!/usr/bin/env python3
"""
Merge multiple scene image folders and CSVs from a multi-video session.

Combines scenes from multiple videos into a single merged set with standardized naming:
- Scene-{video:02d}-{scene:03d}.jpg (video index based on creation order)
- Merged CSV with scenes in chronological order

Video ordering is determined by file creation time (used as "start time" of recording).

Usage (CLI):
    python merge_scenes.py --session-dir Week_77
    python merge_scenes.py --session-dir Week_77 --output Week_77/merged_scenes
    python merge_scenes.py --session-dir Week_77 --config config.yaml --dry-run
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

from pipeline.config import get_config, resolve_tool_config
from pipeline.common.mounts import output_dir_for, SCENES_OUTPUT_SUBDIR, COMBINED_OUTPUT_SUBDIR
from pipeline.common.sessions import media_files, probe_duration
from pipeline.merge.combine_scenes import merge_image_folders, merge_scene_csvs, HAS_PANDAS

from pipeline.common.logs import setup_logging

# Configured in main(), never here: importing a module must not reconfigure
# logging for whatever process happened to import it.
logger = logging.getLogger(__name__)

# Video file extensions to look for
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v', '.flv', '.wmv'}


def find_videos_by_creation_time(session_dir: Path) -> List[Tuple[Path, float]]:
    """
    Find all video files and sort by creation time.

    Returns:
        List of (video_path, creation_time) tuples sorted by creation time
    """
    videos = []

    for video_file in media_files(session_dir, VIDEO_EXTENSIONS):
        # Use creation time (ctime on Unix, birthtime on macOS)
        # Falls back to modification time if creation time unavailable
        stat = video_file.stat()
        ctime = stat.st_birthtime if hasattr(stat, 'st_birthtime') else stat.st_mtime
        videos.append((video_file, ctime))

    # Sort by creation time
    videos.sort(key=lambda v: v[1])

    return videos


def find_scene_dirs_and_csvs(
    session_dir: Path,
    config
) -> List[Tuple[Path, Path]]:
    """
    Find scene output directories and their corresponding CSV files.

    Looks for manual_scenes and 'scenes_output' (WhisperX).

    Returns:
        List of (scene_dir, csv_file) tuples
    """
    scenes_output = output_dir_for(session_dir, SCENES_OUTPUT_SUBDIR, config)
    screens_folder = session_dir / config.get("scenes", "manual_source","manual_source")

    scene_pairs = []

    # Prefer user-captured "manual" scenes (from CaptureScenes, written to the
    # configured scenes.manual_source folder as one per-video subdir containing a
    # scenes.manual_csv_name CSV) over the auto-detected scenes_output.
    if screens_folder.exists():
        manual_csv_name = config.get("scenes", "manual_csv_name", "Scenes.csv")
        csv_files = list(screens_folder.glob(f"*/{manual_csv_name}"))
        if not csv_files:
            # The manual folder is present (so orchestrate skipped PySceneDetect),
            # but there are no scene CSVs inside it — the storyboard would end up
            # with no scenes. Warn loudly rather than fail silently. Expected
            # layout: one per-video subfolder under the manual folder, each with a
            # <manual_csv_name> file.
            logger.warning(
                f"  ⚠ Manual scenes folder '{screens_folder.name}/' exists but "
                f"contains no '*/{manual_csv_name}'. No manual scenes will be used "
                f"(expected one per-video subfolder, each with a {manual_csv_name})."
            )
        for csv_file in csv_files:
            scene_dir = csv_file.parent
            logger.debug(f"  manual scene dir: {scene_dir} (csv {csv_file.name})")
            if scene_dir.exists():
                scene_pairs.append((scene_dir, csv_file))
                
    # Fall back to auto-detected scenes_output if no manual scenes were found.
    if not scene_pairs and scenes_output.exists():
        csv_files = list(scenes_output.glob("*/*-Scenes.csv"))
        for csv_file in csv_files:
            scene_dir = csv_file.parent
            logger.debug(f"  detected scene dir: {scene_dir} (csv {csv_file.name})")
            if scene_dir.exists():
                scene_pairs.append((scene_dir, csv_file))

    return scene_pairs


def order_scenes_by_video_time(
    session_dir: Path,
    scene_pairs: List[Tuple[Path, Path]]
) -> List[Tuple[Path, Path, int]]:
    """
    Order scene directories/CSVs based on video file creation times.

    Args:
        session_dir: Session directory
        scene_pairs: List of (scene_dir, csv_file) tuples

    Returns:
        List of (scene_dir, csv_file, video_index) tuples ordered by video creation time
    """
    # Find all videos with creation times
    videos = find_videos_by_creation_time(session_dir)

    if not videos:
        logger.warning(f"  ⊘ No video files found in {session_dir}")
        return [(s, c, i+1) for i, (s, c) in enumerate(scene_pairs)]

    logger.info(f"  Found {len(videos)} video file(s), ordered by creation time:")
    for i, (video_path, ctime) in enumerate(videos, 1):
        from datetime import datetime
        creation_time = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"    {i}. {video_path.name} (created {creation_time})")

    # Match scene dirs to videos by filename
    # Try to find a scene dir/CSV that matches each video
    ordered_pairs = []
    for video_idx, (video_path, ctime) in enumerate(videos, 1):
        video_stem = video_path.stem

        # Look for matching scene pair
        matched = None
        for scene_dir, csv_file in scene_pairs:
            # Match if CSV/dir name contains video stem or is close enough
            if video_stem.lower() in csv_file.stem.lower() or \
               video_stem.lower() in scene_dir.name.lower():
                matched = (scene_dir, csv_file, video_idx)
                break

        if matched:
            ordered_pairs.append(matched)
            logger.debug(f"    Matched {video_path.name} → {matched[1].stem}")
        else:
            logger.warning(f"    ⊘ No scene CSV found for {video_path.name}")

    # If we didn't match all scene pairs by filename, append remaining ones
    matched_csvs = {pair[1] for pair in ordered_pairs}
    next_idx = len(videos) + 1
    for scene_dir, csv_file in scene_pairs:
        if csv_file not in matched_csvs:
            ordered_pairs.append((scene_dir, csv_file, next_idx))
            logger.warning(f"    Appending unmatched: {csv_file.stem} (index {next_idx})")
            next_idx += 1

    return ordered_pairs


def run(
    session_dir: Path,
    output_dir: Path = None,
    config=None,
    dry_run: bool = False
) -> bool:
    """
    Merge scene images and CSVs from multiple videos.

    Videos are ordered by file creation time. Scene images are renamed to
    Scene-{video:02d}-{scene:03d}.jpg format. CSVs are merged with scenes
    in chronological order.

    Args:
        session_dir: Path to session directory
        output_dir: Output directory (defaults to session_dir/combined_output)
        config: Config object
        dry_run: If True, show what would happen without making any changes

    Returns:
        True if successful
    """
    if config is None:
        config = get_config()

    if not HAS_PANDAS:
        logger.error("  ✗ pandas is required for CSV merging. Install with: pip install pandas")
        return False

    session_dir = Path(session_dir)
    session_name = session_dir.name

    logger.info(f"Merging scenes for {session_name}...")

    # Find scene directories and CSVs
    scene_pairs = find_scene_dirs_and_csvs(session_dir, config)

    if not scene_pairs:
        # No scenes is a valid outcome for an audio-only / no-scenes session, not
        # an error. Return success (no-op) so the merge chain proceeds to the
        # transcript-only storyboard instead of failing here.
        logger.info(f"  ⊘ No scene directories/CSVs found — nothing to merge (transcript-only session)")
        return True

    logger.info(f"  Found {len(scene_pairs)} scene source(s)")

    # Order by video creation time
    logger.info(f"  → Ordering scenes by video creation time...")
    ordered_pairs = order_scenes_by_video_time(session_dir, scene_pairs)

    scene_dirs = [pair[0] for pair in ordered_pairs]
    csv_files = [pair[1] for pair in ordered_pairs]

    # Merge image folders
    logger.info(f"  → Merging {len(scene_dirs)} scene folder(s)...")

    if output_dir is None:
        output_dir = output_dir_for(session_dir, COMBINED_OUTPUT_SUBDIR, config)
    else:
        output_dir = Path(output_dir)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Merge images with renaming. Scene numbers stay per-video (the 2-part key
    # with the Video column in the merged CSV).
    try:
        merge_image_folders(
            scene_dirs,
            csv_files,
            output_dir,
            dry_run=dry_run
        )
        logger.info(f"  ✓ Scene images merged to {output_dir.name}/")
    except Exception as e:
        logger.error(f"  ✗ Failed to merge images: {e}")
        return False

    # Probe each video's true duration (ffprobe) so the per-video time offset in
    # the merged CSV is exact, not just the last-scene-end proxy. Matched to the
    # CSVs by video stem; None (probe unavailable) => proxy fallback per CSV.
    videos = find_videos_by_creation_time(session_dir)
    duration_by_stem = {v.stem: probe_duration(v) for v, _ in videos}
    video_durations = {}
    for csv_file in csv_files:
        for stem, dur in duration_by_stem.items():
            if dur is not None and stem.lower() in csv_file.stem.lower():
                video_durations[csv_file] = dur
                break

    # Merge CSVs
    logger.info(f"  → Merging scene CSVs...")
    try:
        merged_df, remap, title_row = merge_scene_csvs(csv_files, config, video_durations)

        merged_csv_path = output_dir / f"{session_name}_Scenes.csv"
        if not dry_run:
            # lineterminator: pandas defaults to os.linesep (CRLF on Windows),
            # which fights the repo's eol=lf policy and re-dirties goldens.
            merged_df.to_csv(merged_csv_path, index=False, lineterminator="\n")

        logger.info(f"  ✓ Scene CSV merged: {merged_csv_path.name}")
        logger.info(f"    Total scenes: {len(merged_df)}")
    except Exception as e:
        logger.error(f"  ✗ Failed to merge CSVs: {e}")
        return False

    logger.info(f"✓ Scene merge complete for {session_name}")
    return True


def main():
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Merge scene images and CSVs from multiple videos by creation time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge scenes (ordered by video creation time)
  python merge_scenes.py --session-dir Week_77

  # Specify output directory
  python merge_scenes.py --session-dir Week_77 --output Week_77/combined_scenes

  # Use custom config
  python merge_scenes.py --session-dir Week_77 --config config.yaml

  # Dry-run
  python merge_scenes.py --session-dir Week_77 --dry-run

Video Ordering:
  Videos are automatically ordered by file creation time (start time of recording).
  Scene images are renamed to Scene-{video:02d}-{scene:03d}.jpg based on this order.
  CSV files are merged with scenes in chronological order.
        """
    )

    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Path to session directory (e.g., Week_77)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: {session_dir}/combined_output)"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making any changes"
    )

    args = parser.parse_args()

    # Config: explicit --config wins, else walk up from --session-dir.
    resolve_tool_config(args.config, args.session_dir)

    success = run(
        args.session_dir,
        output_dir=args.output,
        dry_run=args.dry_run
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
