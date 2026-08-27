#!/usr/bin/env python3
"""
Map speakers on a session's transcripts (pipeline stage tool).

Runs the speaker-ID stage: replace ``SPEAKER_XX`` diarization IDs with character
names from ``speaker_config.json``, applied in place to every
``<session>/transcriptions/*.json``. This is the discrete step between
transcription and merge, so it works whether the pipeline is driven by
``orchestrate.py`` OR the tools are run piecemeal (``run_tests.py``, or by hand
to redo a step orchestrate failed on). This is the video/diarization workflow
(Workflow B); the Audacity/Workflow-A filename resolver still runs inside the
merge until it, too, moves here.

Usage (CLI):
    python map_speakers.py --session-dir Week_77
    python map_speakers.py --session-dir Week_77 --config config.yaml
    python map_speakers.py --session-dir Week_77 --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import get_config, resolve_tool_config
from pipeline.transcribe.map_speakers import map_speakers

from pipeline.common.logs import setup_logging

# Configured in main(), never here: importing a module must not reconfigure
# logging for whatever process happened to import it.
logger = logging.getLogger(__name__)


def run(session_dir, config=None, dry_run: bool = False,
        resolver: str = "auto") -> bool:
    """Stage entry point. The work lives in pipeline.transcribe.map_speakers;
    this is the seam orchestrate and the CLI both call, so every stage is
    reached the same way regardless of which layer implements it.
    """
    return map_speakers(session_dir, config or get_config(),
                        dry_run=dry_run, resolver=resolver)


def main():
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Map SPEAKER_XX diarization IDs to character names on a session's transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python map_speakers.py --session-dir Week_77
  python map_speakers.py --session-dir Week_77 --config custom_config.yaml
  python map_speakers.py --session-dir Week_77 --dry-run
        """
    )

    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Path to session directory (maps transcriptions/*.json in place)"
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

    success = run(args.session_dir, config=get_config(), dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
