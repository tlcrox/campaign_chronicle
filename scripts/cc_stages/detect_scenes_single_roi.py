#!/usr/bin/env python3
"""
Detect scenes in video using a single ROI (Region of Interest).

This tool handles:
- Bind-mounting the session directory READ-ONLY as the container's input
  (source video is never copied, modified, or deleted)
- Running PySceneDetect with a single fixed ROI region
- Extracting scene change images + scene metadata CSV directly into
  <session>/scenes_output (bind-mounted writable; no copy-out / mirror step)

Uses per-session bind mounts (via `docker compose run -v`) instead of the shared
./video / ./output staging area, so runs are isolated and the originating source
material is protected by the read-only input mount.

ROI is PySceneDetect's crop rectangle, given as corners "x1 y1 x2 y2"
(top-left and bottom-right, in pixels), space-separated, e.g. "0 0 1920 1080".

Can be run standalone or imported and called from orchestrate.py.

Usage (CLI):
    python detect_scenes_single_roi.py --session-dir Week_77 --roi "100 100 1720 880"
    python detect_scenes_single_roi.py --session-dir Week_77 --config config.yaml --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import get_config, resolve_tool_config
from pipeline.common.mounts import (
    clear_mount, session_mounts, UnsafeOutputDir, SCENES_OUTPUT_SUBDIR,
)
from pipeline.common.scenes import rename_scene_images
from pipeline.common.docker import run_docker_command, compose_run
from pipeline.common.sessions import find_video_files
from pipeline.scenes.roi import resolve_single_roi

from pipeline.common.logs import setup_logging

# Configured in main(), never here: importing a module must not reconfigure
# logging for whatever process happened to import it.
logger = logging.getLogger(__name__)

def set_env(roi, cfg):
    env_vars = {}

    # Always set SCENE_ROI (empty string if not provided)
    env_vars["SCENE_ROI"] = roi if roi else ""
    if roi:
        logger.info(f"    Analyzing region: {roi}")
    else:
        logger.info(f"    Analyzing full frame")

    # Unconditional: config validation guarantees all four are present and
    # usable, and detect_scenes.sh refuses to start without them. Skipping one
    # here would produce a refusal, not a fallback.
    env_vars["SCENE_THRESHOLD"] = str(cfg.scene_threshold)
    env_vars["SCENE_MIN_LEN"] = cfg.scene_min_length
    env_vars["SCENE_NUM_IMAGES"] = str(cfg.scene_num_images)
    env_vars["SCENE_IMAGE_FORMAT"] = cfg.scene_image_format

    # Set video index (01 for single-video, will be overridden in multi-video)
    env_vars["VIDEO_INDEX"] = "01"


    return env_vars


def run(session_dir: Path, roi: str = None, config=None, dry_run: bool = False) -> bool:
    """
    Detect scenes in video using a single ROI region.

    ROI format: "x1 y1 x2 y2" (space-separated, e.g., "0 0 1920 1080")

    ROI resolution:
      - If ``roi`` is provided (not None), it is used as-is. An explicit empty
        string forces full-frame analysis.
      - If ``roi`` is None (e.g. not passed on the CLI), the ROI is pulled from
        the config file (config.scene_roi) IF IT IS SET.
      - If neither is set, scene detection falls back to the full frame.

    Args:
        session_dir: Path to session directory (e.g., Week_77)
        roi: ROI region as "x1 y1 x2 y2" string, or None to use config
        config: Config object (defaults to get_config())
        dry_run: If True, show what would happen without making any changes

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = get_config()
    repo_root = config.repo_root

    roi = resolve_single_roi(roi, config)

    session_dir = Path(session_dir)
    roi_desc = f"ROI: {roi}" if roi else "full frame (no ROI)"
    logger.info(f"Detecting scenes ({roi_desc}) in {session_dir.name}...")

    # Verify the session has video (used only for the early-exit + log; the
    # container reads the session dir directly — no staging copy).
    video_files = find_video_files(session_dir)
    if not video_files:
        logger.warning(f"  ⊘ No video files found in {session_dir}")
        return False

    logger.info(f"  Found {len(video_files)} video file(s)")

    # Per-session bind mounts: session dir READ-ONLY input, scenes_output writable.
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

    # Run scene detection: point the entrypoint at the mounted session dirs.
    logger.info(f"  → Running PySceneDetect ({roi_desc})...")
    env_vars = set_env(roi, config)
    env_vars.update(dir_env)

    logger.info(f"  threshold= {env_vars.get('SCENE_THRESHOLD', 'default')}  min_scene_len= {env_vars.get('SCENE_MIN_LEN', 'default')}  num_images= {env_vars.get('SCENE_NUM_IMAGES', 'default')} format= {env_vars.get('SCENE_IMAGE_FORMAT', 'default')}  roi= {env_vars.get('SCENE_ROI', 'full frame')}")
    logger.info(f"  input (ro):  {session_dir.resolve()}")
    logger.info(f"  output:      {scenes_out}")

    if not run_docker_command(
        compose_run("scenes", repo_root),
        dry_run,
        env_vars=env_vars,
        cwd=repo_root,
        volumes=volumes,
        log_dir=scenes_out,
    ):
        logger.error(f"  ✗ Scene detection failed")
        return False

    # PySceneDetect writes raw "<video>-Scene-NNN-MM.ext" names; canonicalise
    # them host-side to Scene-{video:02d}-{scene:03d}.ext (single source of truth).
    if not dry_run:
        renamed = rename_scene_images(scenes_out)
        logger.info(f"  ✓ Canonicalised {renamed} scene image name(s)")

    logger.info(f"  ✓ Scene detection complete — output in {scenes_out}")
    return True


def main():
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Detect scenes in video using a single ROI (Region of Interest)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze full frame (no ROI restriction)
  python detect_scenes_single_roi.py --session-dir Week_77

  # Analyze specific region (corners: top-left x1=100 y1=100, bottom-right x2=1720 y2=880)
  python detect_scenes_single_roi.py --session-dir Week_77 --roi "100 100 1720 880"

  # Use with custom config
  python detect_scenes_single_roi.py --session-dir Week_77 --roi "100 100 1720 880" --config custom_config.yaml

  # Dry-run to see what would happen
  python detect_scenes_single_roi.py --session-dir Week_77 --roi "100 100 1720 880" --dry-run

ROI Format (PySceneDetect crop):
  "x1 y1 x2 y2" - all space-separated integers (top-left and bottom-right corners, pixels)
  Example: "0 0 1920 1080" (crop from corner 0,0 to corner 1920,1080 = full HD frame)
        """
    )

    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Path to session directory (e.g., Week_77)"
    )

    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help='ROI region as "x1 y1 x2 y2" (space-separated). '
             'Omit to use the ROI from config.yaml if set, else full frame. '
             'Pass an empty string ("") to force full-frame analysis.'
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

    success = run(args.session_dir, roi=args.roi, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
