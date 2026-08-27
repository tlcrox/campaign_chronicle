#!/usr/bin/env python3
"""
Clean a session's transcripts (pipeline stage tool).

Runs the clean-transcription stage: two ordered passes applied in place to every
``<session>/transcriptions/*.json`` — a confidence/hallucination pass (opt-in via
``whisper.clean.enabled``) and a filler-phrase pass. This is the discrete step
between map_speakers and merge, so it works whether the pipeline is driven by
``orchestrate.py`` OR the tools are run piecemeal (``run_tests.py``, or by hand
to redo a step orchestrate failed on).

Usage (CLI):
    python clean_transcription.py --session-dir Week_77
    python clean_transcription.py --session-dir Week_77 --config config.yaml
    python clean_transcription.py --session-dir Week_77 --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import get_config, resolve_tool_config
from pipeline.transcribe.clean_transcription import clean_transcript

from pipeline.common.logs import setup_logging

# Configured in main(), never here: importing a module must not reconfigure
# logging for whatever process happened to import it.
logger = logging.getLogger(__name__)


def run(session_dir, config=None, dry_run: bool = False) -> bool:
    """Stage entry point. The work lives in
    pipeline.transcribe.clean_transcription; this is the seam orchestrate and
    the CLI both call.
    """
    return clean_transcript(session_dir, config or get_config(), dry_run=dry_run)


def main():
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Clean a session's transcripts (confidence + filler passes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean_transcription.py --session-dir Week_77
  python clean_transcription.py --session-dir Week_77 --config custom_config.yaml
  python clean_transcription.py --session-dir Week_77 --dry-run
        """
    )

    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Path to session directory (cleans transcriptions/*.json in place)"
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
