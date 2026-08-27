#!/usr/bin/env python3
"""
Detect scenes in video using multiple ROIs (Regions of Interest).

This tool handles:
- Reading ROI configuration from roi-file or ROI.json
-- Bind-mounting the session directory READ-ONLY as the container's input
- Running PySceneDetect with multiple ROI regions

Can be run standalone or imported and called from orchestrate.py.

Usage (CLI):
    python detect_scenes_multi_roi.py --session-dir Week_21 --roi-file roi.json
    python detect_scenes_multi_roi.py --session-dir Week_21 --roi-file roi.json --dry-run
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from pipeline.config import get_config, resolve_tool_config
from pipeline.common.mounts import (
    clear_mount, session_mounts, UnsafeOutputDir, SCENES_OUTPUT_SUBDIR,
)
from pipeline.common.docker import run_docker_command, compose_run
from pipeline.common.sessions import find_video_files
from pipeline.scenes.roi import RoiFile, RoiError
from pipeline.scenes.merge_segments import merge_segments, MergeError

from pipeline.common.logs import setup_logging

# Configured in main(), never here: importing a module must not reconfigure
# logging for whatever process happened to import it.
logger = logging.getLogger(__name__)

def set_env(cfg):
    """Set up environment variables from config for Docker."""
    env_vars = {}

    # Unconditional: config validation guarantees all four are present and
    # usable, and detect_scenes_multi.sh refuses to start without them.
    env_vars["SCENE_THRESHOLD"] = str(cfg.scene_threshold)
    env_vars["SCENE_MIN_LEN"] = cfg.scene_min_length
    env_vars["SCENE_NUM_IMAGES"] = str(cfg.scene_num_images)
    env_vars["SCENE_IMAGE_FORMAT"] = cfg.scene_image_format

    # VIDEO_INDEX is not used in multi-ROI (processes all videos at once)
    # but included for consistency with single-ROI
    env_vars["VIDEO_INDEX"] = "01"


    return env_vars


def run(
    session_dir: Path,
    roi_file: Path = None,
    config=None,
    dry_run: bool = False
) -> bool:
    """
    Detect scenes in video using multiple ROIs (Regions of Interest).

    Analyzes each ROI region independently for scene changes. Requires a JSON
    configuration file with ROI definitions in flat or hierarchical format.

    Args:
        session_dir: Path to session directory (e.g., Week_21)
        roi_file: Path to ROI configuration file (JSON format with timestamps)
                 If not provided, looks for roi_history.json in session_dir
        config: Config object (defaults to get_config())
        dry_run: If True, show what would happen without making any changes

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = get_config()
    repo_root = config.repo_root

    session_dir = Path(session_dir)
    logger.info(f"Detecting scenes (multi-ROI) in {session_dir.name}...")

    # Determine ROI file location
    if roi_file is None:
        roi_file = session_dir / "roi_history.json"
    else:
        roi_file = Path(roi_file)

    # Verify ROI file exists
    if not roi_file.exists():
        logger.error(f"  ✗ ROI file not found: {roi_file}")
        return False

    logger.info(f"  Using ROI config: {roi_file.name}")

    # Load and validate ROI configuration using RoiFile class
    try:
        roi_config = RoiFile.load(roi_file)

        # Validate the ROI file
        problems = roi_config.validate()
        if problems:
            logger.error(f"  ✗ ROI file validation failed:")
            for problem in problems:
                logger.error(f"    - {problem}")
            return False

        # Log information about the ROI configuration (always video-keyed)
        logger.info(f"  Loaded ROI config for {len(roi_config.videos)} video(s):")
        for video in roi_config.videos:
            segments = roi_config.segments(video)
            logger.info(f"    - {video}: {len(segments)} segment(s)")
            for seg in segments:
                end_str = "end of video" if seg.is_final else f"{seg.end}s"
                logger.info(f"      [{seg.start}s - {end_str}] roi={seg.roi}")

    except RoiError as e:
        logger.error(f"  ✗ Failed to load ROI file: {e}")
        return False
    except Exception as e:
        logger.error(f"  ✗ Unexpected error loading ROI file: {e}")
        return False

    # Find video files
    video_files = find_video_files(session_dir)
    if not video_files:
        logger.warning(f"  ⊘ No video files found in {session_dir}")
        return False

    logger.info(f"  Found {len(video_files)} video file(s)")

    # Per-session bind mounts: session dir READ-ONLY input, scenes_output writable.
    # (The ROI file lives in the session dir, so the container reads it from the
    # read-only input mount — nothing is staged.)
    try:
        volumes, dir_env, scenes_out = session_mounts(
            session_dir,
            SCENES_OUTPUT_SUBDIR,
            "VIDEO_DIR", "SCENES_DIR",
            output_base=getattr(config, "output_base_dir", None),
            session_key=(config.session_key(session_dir)
                         if getattr(config, "output_base_dir", None) else None),
        )
    except UnsafeOutputDir as e:
        logger.error(f"  ✗ {e}")
        return False

    if not dry_run:
        scenes_out.mkdir(parents=True, exist_ok=True)
        clear_mount(scenes_out, dry_run=dry_run)  # derived output only (guarded)

    # Run scene detection with multi-ROI (config-based env + the ROI file name).
    num_videos = len(roi_config.videos)
    logger.info(f"  → Running PySceneDetect (analyzing {num_videos} video(s) with time-based ROI regions)...")
    env_vars = set_env(config)
    env_vars["ROI_FILE"] = roi_file.name
    env_vars.update(dir_env)

    logger.info(f"  threshold={env_vars.get('SCENE_THRESHOLD', 'default')}  min_scene_len={env_vars.get('SCENE_MIN_LEN', 'default')}  num_images={env_vars.get('SCENE_NUM_IMAGES', 'default')}  format={env_vars.get('SCENE_IMAGE_FORMAT', 'default')}")
    logger.info(f"  input (ro):  {session_dir.resolve()}")
    logger.info(f"  output:      {scenes_out}")

    if not run_docker_command(
        compose_run("scenes_multi", repo_root),
        dry_run,
        env_vars=env_vars,
        cwd=repo_root,
        volumes=volumes,
        log_dir=scenes_out,
    ):
        logger.error(f"  ✗ Scene detection failed")
        return False

    # Reassemble each video's per-segment output host-side: the container left
    # raw scenedetect results under <video>/_segments/segment_i/. One pass per
    # video renumbers scenes in time order and emits the canonical CSV + images
    # together (pipeline/scenes/merge_segments.py). Video index = sorted order.
    if not dry_run:
        for video_index, video_dir in enumerate(
            sorted(d for d in scenes_out.iterdir() if d.is_dir()), start=1
        ):
            seg_dir = video_dir / "_segments"
            if not seg_dir.is_dir():
                continue
            out_csv = video_dir / f"{video_dir.name}-Scenes.csv"
            try:
                n = merge_segments(seg_dir, out_csv, images_out=video_dir, video_index=video_index)
                logger.info(f"  ✓ {video_dir.name}: reassembled {n} scene(s) as video {video_index:02d}")
            except MergeError as e:
                logger.error(f"  ✗ {video_dir.name}: reassembly failed: {e}")
                return False
            shutil.rmtree(seg_dir, ignore_errors=True)

    logger.info(f"✓ Scene detection complete - output in {scenes_out}")
    return True


def main():
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Detect scenes in video using multiple ROIs (Regions of Interest)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_scenes_multi_roi.py --session-dir Week_21 --roi-file roi_history.json
  python detect_scenes_multi_roi.py --session-dir Week_21 --roi-file ./rois/Week21.json
  python detect_scenes_multi_roi.py --session-dir Week_21 --roi-file roi_history.json --dry-run

ROI File Format (JSON - Flat or Hierarchical):
  Flat (single video):
    {
      "00:00:00": {"frame": 0, "roi": "529 249 1288 755"},
      "00:05:00": {"frame": 7500, "roi": "400 200 1400 800"}
    }
  
  Hierarchical (multiple videos):
    {
      "video1.mkv": {
        "00:00:00": {"frame": 0, "roi": "529 249 1288 755"},
        "00:05:00": {"frame": 7500, "roi": "400 200 1400 800"}
      },
      "video2.mkv": {
        ...
      }
    }
        """
    )

    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Path to session directory (e.g., Week_21)"
    )

    parser.add_argument(
        "--roi-file",
        type=Path,
        default=None,
        help="Path to ROI configuration file (JSON format with timestamps). If not provided, looks for roi_history.json in session-dir."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (defaults to project root config.yaml)"
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
        roi_file=args.roi_file,
        dry_run=args.dry_run
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
