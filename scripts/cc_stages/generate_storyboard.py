#!/usr/bin/env python3
"""
Generate a storyboard Word document from a merged transcript, optionally with
scene images.

Combines a merged transcript (JSON or TXT) with scene images and a metadata CSV
to create a formatted Word document with proper timing, speaker names, and
embedded scene images. The CSV and images are OPTIONAL: a session with no scenes
(e.g. Craig audio-only) produces a transcript-only document.

Usage (CLI):
    python generate_storyboard.py --session-dir Week_77
    python generate_storyboard.py --session-dir Week_77 --output Week_77/storyboard.docx
    python generate_storyboard.py --transcript transcript.json --output output.docx          # transcript-only
    python generate_storyboard.py --transcript transcript.json --scenes scenes/ --csv scenes.csv --output output.docx
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import get_config, resolve_tool_config
from pipeline.common.mounts import output_dir_for, COMBINED_OUTPUT_SUBDIR, OUTPUT_ROOT
from pipeline.common.scenes import iter_scene_images
from pipeline.merge.storyboard import generate_storyboard

from pipeline.common.logs import setup_logging

# Configured in main(), never here: importing a module must not reconfigure
# logging for whatever process happened to import it.
logger = logging.getLogger(__name__)


def run(
    session_dir: Path = None,
    transcript_path: Path = None,
    scenes_dir: Path = None,
    csv_path: Path = None,
    output_path: Path = None,
    config=None,
    dry_run: bool = False
) -> bool:
    """
    Generate a storyboard Word document.

    Can be called in two ways:

    1. With session_dir: Finds merged transcript, CSV, and images automatically
       from combined_output/ directory

    2. With explicit paths: Uses provided paths for transcript, scenes, CSV, and output

    Args:
        session_dir: Path to session directory (e.g., Week_77)
        transcript_path: Path to transcript JSON/TXT file (if not using session_dir)
        scenes_dir: Path to scenes image folder (if not using session_dir)
        csv_path: Path to scenes CSV file (if not using session_dir)
        output_path: Output Word document path
        config: Config object
        dry_run: If True, show what would happen without making any changes

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = get_config()

    # Mode 1: Session directory mode
    if session_dir:
        session_dir = Path(session_dir)
        session_name = session_dir.name

        logger.info(f"Generating storyboard for {session_name}...")

        # Merged scenes go to combined_output/
        combined_dir = output_dir_for(session_dir, COMBINED_OUTPUT_SUBDIR, config)

        # Find combined_output
        if not combined_dir.exists():
            logger.error(f"  ✗ Combined output directory not found: {combined_dir}")
            logger.info(f"    Run merge_scenes and merge_transcripts first")
            return False
        
        # Find transcript in transcriptions
        transcript_files = list(combined_dir.glob("*_transcript_combined.*"))
        if not transcript_files:
            logger.error(f"  ✗ No merged transcript found in {combined_dir}")
            logger.info(f"    Run merge_transcripts first")
            return False
        transcript_path = transcript_files[0]

        # Scenes are optional — a no-scenes session (e.g. Craig audio-only) has
        # no *_Scenes.csv. Fall back to a transcript-only document.
        csv_files = list(combined_dir.glob("*_Scenes.csv"))
        csv_path = csv_files[0] if csv_files else None
        scenes_dir = combined_dir

        # Default output path
        if output_path is None:
            # Inside OUTPUT_ROOT, not loose in the session dir: the storyboard is
            # generated output, and the whole point of the output root is that
            # everything the tool produces lands in one deletable folder. It sits
            # at the root of cc_output/ rather than under combined_output/
            # because it is the final deliverable, not an intermediate.
            output_path = (output_dir_for(session_dir, OUTPUT_ROOT, config)
                           / f"{session_name}_storyboard.docx")
            output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"  Transcript: {transcript_path.name} (from {combined_dir}/)")
        if csv_path is not None:
            logger.info(f"  Scenes CSV: {csv_path.name} (from {combined_dir}/)")
            logger.info(f"  Scenes images: {scenes_dir.name}/")
        else:
            logger.info(f"  ⊘ No merged scenes CSV in {combined_dir}/ — transcript-only document")
        logger.info(f"  Output: {output_path.name}")

    # Mode 2: Explicit paths mode. Only the transcript is required; --csv and
    # --scenes are optional (omit both for a transcript-only document).
    else:
        if not transcript_path:
            logger.error(f"  ✗ Must provide either --session-dir OR --transcript")
            return False

        transcript_path = Path(transcript_path)
        scenes_dir = Path(scenes_dir) if scenes_dir else None
        csv_path = Path(csv_path) if csv_path else None

        if not transcript_path.exists():
            logger.error(f"  ✗ Transcript not found: {transcript_path}")
            return False
        if csv_path is not None and not csv_path.exists():
            logger.error(f"  ✗ CSV not found: {csv_path}")
            return False
        if scenes_dir is not None and not scenes_dir.exists():
            logger.error(f"  ✗ Scenes directory not found: {scenes_dir}")
            return False

        if output_path is None:
            output_path = Path(transcript_path.parent) / f"{transcript_path.stem}_storyboard.docx"

        logger.info(f"Generating storyboard...")
        logger.info(f"  Transcript: {transcript_path.name}")
        if csv_path is not None:
            logger.info(f"  Scenes CSV: {csv_path.name}")
        if scenes_dir is not None:
            logger.info(f"  Scenes dir: {scenes_dir.name}")
        logger.info(f"  Output: {output_path.name}")

    # Validate inputs. Only the transcript is required; the scene CSV and image
    # folder are optional (a no-scenes session yields a transcript-only doc).
    if not transcript_path.exists():
        logger.error(f"  ✗ Transcript file not found: {transcript_path}")
        return False

    have_scenes = bool(
        csv_path and Path(csv_path).exists()
        and scenes_dir and Path(scenes_dir).exists()
    )
    scene_images = []
    if have_scenes:
        scene_images = iter_scene_images(scenes_dir, "Scene-*")
        if not scene_images:
            logger.warning(f"  ⊘ No scene images in {scenes_dir}; transcript-only document")
    else:
        csv_path = None  # no CSV (or no image dir) -> transcript-only
        logger.info(f"  ⊘ No scene CSV/images; transcript-only document")

    # Generate storyboard
    logger.info(f"  → Generating storyboard document...")

    if dry_run:
        logger.info(f"  [DRY RUN] Would generate: {output_path}")
        logger.info(f"    With {len(scene_images)} scene image(s)")
        return True

    try:
        generate_storyboard(
            str(csv_path) if csv_path else None,
            str(transcript_path),
            str(scenes_dir) if scenes_dir else None,
            str(output_path),
        )
        logger.info(f"  ✓ Storyboard document created: {output_path.name}")
    except Exception as e:
        logger.error(f"  ✗ Storyboard generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Generate a storyboard Word document from transcript and scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage modes:

1. FROM SESSION DIRECTORY:
   python generate_storyboard.py --session-dir Week_77
   (Looks for merged files in Week_77/combined_output/)

2. FROM EXPLICIT PATHS:
   python generate_storyboard.py --transcript transcript.json --csv scenes.csv --scenes scenes/ --output output.docx

Examples:
  # Auto-find merged files
  python generate_storyboard.py --session-dir Week_77

  # Specify custom output location
  python generate_storyboard.py --session-dir Week_77 --output my_storyboard.docx

  # Use explicit paths
  python generate_storyboard.py --transcript Week_77/combined_output/merged.json --csv scenes.csv --scenes scenes/ --output storyboard.docx

  # Dry-run
  python generate_storyboard.py --session-dir Week_77 --dry-run
        """
    )

    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Session directory (e.g., Week_77). Finds merged files automatically in combined_output/"
    )

    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Transcript file path (JSON or TXT). Required if not using --session-dir"
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Scenes metadata CSV file. Optional — omit for a transcript-only document"
    )

    parser.add_argument(
        "--scenes",
        type=Path,
        default=None,
        help="Scenes images directory. Optional — omit for a transcript-only document"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Word document path (default: {session_name}_storyboard.docx or based on transcript name)"
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
    
    logger.info(f"storyboard args {args}")

    # Validate arguments
    if args.session_dir:
        if args.transcript or args.csv or args.scenes:
            logger.error("Cannot mix --session-dir with explicit path arguments (--transcript, --csv, --scenes)")
            sys.exit(1)
    else:
        if not args.transcript:
            logger.error("Must provide either --session-dir OR --transcript (with optional --csv / --scenes)")
            parser.print_help()
            sys.exit(1)

    # Config: explicit --config wins, else walk up from --session-dir.
    resolve_tool_config(args.config, args.session_dir)

    success = run(
        session_dir=args.session_dir,
        transcript_path=args.transcript,
        scenes_dir=args.scenes,
        csv_path=args.csv,
        output_path=args.output,
        dry_run=args.dry_run
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
