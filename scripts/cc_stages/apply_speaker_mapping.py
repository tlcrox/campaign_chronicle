#!/usr/bin/env python3
"""
Apply speaker_config.json mappings to a WhisperX transcript (standalone tool).

Thin CLI around pipeline.transcribe.apply_speaker_mapping — maps generic
SPEAKER_XX diarization IDs to character names using speaker_config.json
(`global_mapping` plus per-session `session_mappings`). This is Workflow B, and
it updates both segment-level and word-level speaker fields.

This is the standalone-tool counterpart to the in-pipeline function that
orchestrate.py and the transcribe stage already call, so the speaker-mapping step
has an independent tool too.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline.config import get_config, resolve_tool_config
from pipeline.transcribe.map_speakers import (
    apply_speaker_mapping,
    require_diarization_for_mapping,
)

from pipeline.common.logs import setup_logging

# Configured in main(), never here: importing a module must not reconfigure
# logging for whatever process happened to import it.
logger = logging.getLogger(__name__)


def apply_speaker_mapping_tool(
    transcript_file: Path,
    session_dir: Path,
    output_file: Path = None,
    speaker_config_file: Path = None,
    config=None,
    dry_run: bool = False,
) -> bool:
    """
    Load a transcript, apply speaker_config.json mappings, and save it.

    Args:
        transcript_file: Path to the WhisperX JSON transcript.
        session_dir: Session directory. Drives session-specific mappings, and
            is where config resolution starts, so it is required rather than
            guessed: the inference this replaced took the transcript's
            grandparent, which is ``cc_output`` on the current layout, not the
            session.
        output_file: Where to write the mapped transcript. Defaults to
            overwriting ``transcript_file``.
        speaker_config_file: Explicit speaker_config.json path. If None, the
            config default (``config.speaker_config_file``) is used.
        config: Config object (defaults to get_config()).
        dry_run: If True, show what would happen without making any changes.

    Returns:
        True on success, False otherwise.
    """
    if config is None:
        config = get_config()

    # Fail loud if mapping is configured but diarization is off (no SPEAKER_XX
    # to rename). Honors an explicit --speaker-config override.
    require_diarization_for_mapping(config, speaker_config_file)

    transcript_file = Path(transcript_file)
    if not transcript_file.exists():
        logger.error(f"  ✗ Transcript not found: {transcript_file}")
        return False

        # …/<session>/transcriptions/<name>.json  ->  <session>
    session_dir = Path(session_dir)

    out = Path(output_file) if output_file else transcript_file

    logger.info(
        f"Applying speaker mapping to {transcript_file.name} "
        f"(session: {session_dir.name})..."
    )

    if dry_run:
        logger.info(f"  [DRY RUN] Would map speakers and write {out}")
        return True

    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"  ✗ Could not load transcript: {e}")
        return False

    mapped = apply_speaker_mapping(
        transcript, session_dir, config, speaker_config_file,
        transcript_name=transcript_file.stem,
    )

    try:
        with open(out, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(mapped, f, indent=2)
    except IOError as e:
        logger.error(f"  ✗ Could not write {out}: {e}")
        return False

    logger.info(f"  ✓ Wrote mapped transcript: {out}")
    return True


def main():
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Apply speaker_config.json mappings to a WhisperX transcript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python apply_speaker_mapping.py --transcript "Week 13/transcriptions/Week13_a.json"
  python apply_speaker_mapping.py --transcript t.json --session-dir "Week 13" --output mapped.json
  python apply_speaker_mapping.py --transcript t.json --speaker-config speaker_config.json --dry-run
        """
    )

    parser.add_argument(
        "--transcript",
        type=Path,
        required=True,
        help="Path to the WhisperX JSON transcript to map"
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Session directory. Drives session-specific mappings, and is where "
             "config resolution starts."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (defaults to overwriting the input transcript)"
    )
    parser.add_argument(
        "--speaker-config",
        type=Path,
        default=None,
        help="Path to speaker_config.json (defaults to config's speaker_config_file)"
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

    success = apply_speaker_mapping_tool(
        args.transcript,
        session_dir=args.session_dir,
        output_file=args.output,
        speaker_config_file=args.speaker_config,
        dry_run=args.dry_run,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
