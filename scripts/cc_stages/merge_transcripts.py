#!/usr/bin/env python3
"""
Merge multiple transcript JSON files into a single combined transcript.

This tool handles both workflows:
- With Audacity Project: Finds speaker name from filename using the
  filename_mapping in speaker_config.json, interleaves segments by timestamp
- Video with Diarization: Uses already-mapped character names from per-video files,
  merges by timestamp

Detects workflow automatically based on transcript file naming patterns.

Usage (CLI):
    python merge_transcripts.py --session-dir Week_74
    python merge_transcripts.py --session-dir Week_77 --config config.yaml
    python merge_transcripts.py --session-dir Week_74 --output Week_74/merged.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline.config import get_config, resolve_tool_config
from pipeline.common.mounts import output_dir_for, SESSION_OUTPUT_SUBDIR, COMBINED_OUTPUT_SUBDIR
from pipeline.common.sessions import find_video_files, probe_duration
from pipeline.merge.combine_transcripts import (
    merge_transcripts,
    load_merge_speaker_data,
)

from pipeline.common.logs import setup_logging

# Configured in main(), never here: importing a module must not reconfigure
# logging for whatever process happened to import it.
logger = logging.getLogger(__name__)


def detect_workflow(transcript_files: list, filename_mapping: dict) -> str:
    """
    Detect which workflow these transcripts are from based on filename patterns.

    With Audacity Project: a filename contains one of the user's filename_mapping tokens
                (e.g., "1-nedoking.json").
    Video with Diarization: no filename_mapping token matches (e.g., "video.json").

    With an empty filename_mapping (none configured), everything is Video with Diarization.

    Returns: "A" or "B"
    """
    logger.info(f"mapping using {filename_mapping}")
    for f in transcript_files:
        filename_lower = f.name.lower()
        for token in (filename_mapping or {}):
            if str(token).lower() in filename_lower:
                logger.info(f"  Detected With Audacity Project: Found speaker token '{token}' in {f.name}")
                return "A"

    logger.info(f"  Detected Video with Diarization: No speaker tokens found in filenames")
    return "B"


def _transcript_last_end(transcript_file: Path) -> float:
    """Fallback per-video duration: the last spoken 'end' time in the transcript."""
    try:
        with open(transcript_file, encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError, OSError):
        return 0.0
    ends = [float(s.get("end", 0.0)) for s in data.get("segments", [])]
    return max(ends) if ends else 0.0


def diarization_video_offsets(session_dir: Path, transcript_files: list) -> dict:
    """Per-video start offsets on the concatenated-session timeline.

    Each Video with Diarization per-video transcript is 0-based; to line up with the offset
    scene times, shift each video's segments by the summed duration of the
    videos before it. Videos are ordered by creation time (matching the scene
    merge) and measured with ffprobe (falling back to the transcript's own last
    spoken time when ffprobe is unavailable). Returns {} for a single video.
    """
    if len(transcript_files) < 2:
        return {}
    tf_by_stem = {tf.stem: tf for tf in transcript_files}
    matched = [v for v in find_video_files(session_dir) if v.stem in tf_by_stem]
    if len(matched) < 2:
        return {}
    matched.sort(key=lambda v: v.stat().st_ctime)
    offsets, running = {}, 0.0
    for v in matched:
        tf = tf_by_stem[v.stem]
        offsets[tf] = running
        dur = probe_duration(v)
        if dur is None:
            dur = _transcript_last_end(tf)
        running += dur or 0.0
    return offsets


def build_merge_sources(session_dir: Path, transcript_files: list, filename_mapping: dict) -> list:
    """Front-end source adapter: turn per-source transcript files into
    ``merge_transcripts`` sources, assigning each its timeline offset.

    Speakers are already mapped in place by the map_speakers stage, so no
    per-source speaker override is needed here — each segment keeps its own name.
    Offsets follow the source layout: parallel per-player tracks (Audacity /
    Workflow A) share the session timeline (offset 0); serial per-video tracks
    (diarization / Workflow B) are shifted by the cumulative duration of the
    videos before them.
    """
    if detect_workflow(transcript_files, filename_mapping) == "A":
        offsets = {}  # parallel tracks share one timeline
    else:
        offsets = diarization_video_offsets(session_dir, transcript_files)
        if offsets:
            logger.info(f"  Offsetting {len(offsets)} per-video transcript(s) onto the session timeline")

    sources = []
    for transcript_file in sorted(transcript_files):
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"  Could not load transcript {transcript_file}: {e}")
            continue
        if 'segments' not in data:
            logger.warning(f"  No segments in {transcript_file.name}")
            continue
        sources.append({
            "segments": data['segments'],
            "offset": float(offsets.get(transcript_file, 0.0)),
            "speaker": None,
            "name": transcript_file.name,
        })
    return sources


def run(
    session_dir: Path,
    output_file: Path = None,
    config=None,
    dry_run: bool = False
) -> bool:
    """
    Merge all transcript JSON files in session's output directory.

    Args:
        session_dir: Path to session directory (e.g., Week_74)
        output_file: Optional output file path. If not provided, creates
                    {session_dir}/combined_output/{session_name}_transcript_combined.json
        config: Config object (defaults to get_config())
        dry_run: If True, show what would happen without making any changes

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = get_config()
        
    logger.info(f"looking up config {config.speaker_config_file}")

    session_dir = Path(session_dir)
    session_name = session_dir.name

    logger.info(f"Merging transcripts for {session_name}...")

    # Find session output directory
    output_dir = output_dir_for(session_dir, SESSION_OUTPUT_SUBDIR, config)

    if not output_dir.exists():
        logger.error(f"  ✗ Output directory not found: {output_dir}")
        return False

    # Find all JSON transcript files
    transcript_files = sorted(output_dir.glob("*.json"))

    # Filter out already-merged files
    transcript_files = [
        f for f in transcript_files
        if "combined" not in f.name and "_merged" not in f.name
    ]

    if not transcript_files:
        logger.warning(f"  ⊘ No transcript files found in {output_dir}")
        return False

    logger.info(f"  Found {len(transcript_files)} transcript file(s):")
    for f in transcript_files:
        logger.info(f"    - {f.name}")

    # Load the filename->name map from speaker_config.json (empty when
    # unconfigured). It only selects the offset strategy (parallel vs serial) now;
    # speaker resolution and its fail-loud reporting are owned by the map_speakers
    # stage, which has already run and stamped names on these files.
    filename_mapping, _ = load_merge_speaker_data(config)

    # Build sources (per-file segments + timeline offset) and merge on one timeline.
    logger.info(f"  → Merging...")
    sources = build_merge_sources(session_dir, transcript_files, filename_mapping)
    merged_data = merge_transcripts(sources)

    if not merged_data.get("segments"):
        logger.warning(f"  ⊘ Merge produced no segments")
        return False

    logger.info(f"  ✓ Merged {len(merged_data['segments'])} segments")

    # Determine output file
    if output_file is None:
        # Output to combined_output/ to match merge_scenes.py and generate_storyboard.py
        combined_dir = output_dir_for(session_dir, COMBINED_OUTPUT_SUBDIR, config)
        output_file = combined_dir / f"{session_name}_transcript_combined.json"
    else:
        output_file = Path(output_file)

    # Write merged transcript
    if dry_run:
        logger.info(f"  [DRY RUN] Would write {len(merged_data['segments'])} segments to {output_file}")
        return True

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(merged_data, f, indent=2)
        logger.info(f"  ✓ Merged transcript saved: {output_file.name}")
    except Exception as e:
        logger.error(f"  ✗ Failed to write output: {e}")
        return False

    logger.info(f"✓ Merge complete for {session_name}")
    return True


def main():
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Merge multiple transcript JSON files into a single combined transcript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all transcripts in Week_74/transcriptions/
  python merge_transcripts.py --session-dir Week_74

  # Merge and save to custom location
  python merge_transcripts.py --session-dir Week_74 --output Week_74/merged.json

  # Use custom config
  python merge_transcripts.py --session-dir Week_74 --config custom_config.yaml

  # Dry-run to see what would happen
  python merge_transcripts.py --session-dir Week_74 --dry-run

Workflow Detection:
  - With Audacity Project: Per-speaker files (names contain filename_mapping tokens like "nedoking", "thirty")
    → Merges by extracting speaker from filename and interleaving by timestamp
  - Video with Diarization: Per-video files (no speaker tokens in names)
    → Merges using already-mapped character names, interleaved by timestamp
        """
    )

    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Path to session directory (e.g., Week_74)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: {session_dir}/combined_output/{session_name}_transcript_combined.json)"
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
        output_file=args.output,
        dry_run=args.dry_run
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
