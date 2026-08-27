#!/usr/bin/env python3
"""
Run orchestrate.py against one or more test session directories.

Thin wrapper: it forwards --source-dir / --session-dirs straight to
orchestrate.py (the same interface orchestrate itself uses), plus --dry-run.
There is no scenario layer — pass the session directories you want to
process, exactly like everything else in the suite.

Examples:
    python run_orchestrate_tests.py --session-dirs MultiROI
    python run_orchestrate_tests.py --session-dirs SingleVideo MultiVideo "Weeks/Week 13"
    python run_orchestrate_tests.py --session-dirs MultiROI --dry-run
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from pipeline.common.mounts import OUTPUT_ROOT
from pipeline.config import find_repo_root

# Shared golden-comparison logic (same module run_tests.py uses). Sibling import
# when run as a script; package import under -m tests.integration.*.
try:
    from output_compare import compare_output_tree, wipe_output_tree
except ImportError:  # pragma: no cover - only the -m invocation path
    from tests.integration.output_compare import compare_output_tree, wipe_output_tree

from pipeline.common.logs import setup_logging

# Configured in main(), never here: importing a module must not reconfigure
# logging for whatever process happened to import it.
logger = logging.getLogger(__name__)

# Repo root, walking up to the directory containing pyproject.toml — the one
# implementation, in pipeline.config. required=True because a driver with no
# repo above it cannot do anything useful.
REPO_ROOT = find_repo_root(__file__, required=True)
ORCHESTRATE = REPO_ROOT / "scripts" / "orchestrate.py"

# Integration-test layout: tests/test_source/ holds the INPUTS (and the
# governing config.yaml); tests/expected/ holds the GOLDEN outcomes.
# config/config.yaml is the MANUAL-run config and is deliberately not used here,
# so a developer's own source_dir can't influence the suite.
FIXTURES = REPO_ROOT / "tests" / "test_source"
EXPECTED = REPO_ROOT / "tests" / "expected"
FIXTURE_CONFIG = FIXTURES / "config.yaml"
DEFAULT_SOURCE_DIR = FIXTURES

DEFAULT_TEST_DIRS = [
    "SingleVideo",      # Basic: 1 video, embedded audio
    "MultiVideo",       # Multiple videos, embedded audio (tests VIDEO_INDEX)
    "SingleCraig",      # Single video + Audacity audio
    "MultiCraig",       # Multiple videos + shared Audacity audio + per-video ROI
    "MultiROI",         # Multiple videos + time-based ROI regions
    "SingleROI",        # Single video default ROI from config.
    "ScenesCraig",      # Reference: already has scene output
    "Weeks/Week 13",    # Video only
    "Weeks/Week 14",    # Audio only
]

def run_one_session(source_dir: Path, session_dir: str, dry_run: bool = False,
                    timeout: int = 600) -> bool:
    """Run orchestrate.py against a single session, with its own timeout."""
    cmd = [
        sys.executable,
        str(ORCHESTRATE),
        "--source-dir", str(source_dir),
        "--session-dirs", session_dir,
    ]
    if dry_run:
        cmd.append("--dry-run")

    logger.info(f"Running: {' '.join(cmd)}")

    # Pin the child to the fixture config unless the caller overrode it, so a
    # developer's config/config.yaml can never influence a test run.
    env = dict(os.environ)
    env.setdefault("WHISPERX_CONFIG", str(FIXTURE_CONFIG))

    try:
        # Stream orchestrate's output live so the run is visible as it happens.
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {session_dir} timed out ({timeout}s)")
        return False
    except Exception as e:
        logger.error(f"✗ {session_dir}: error running orchestrate: {e}")
        return False

    if result.returncode != 0:
        logger.error(f"✗ {session_dir} failed (return code {result.returncode})")
        return False

    logger.info(f"✓ {session_dir} succeeded")
    return True


def compare_session(source_dir: Path, session_dir: str) -> bool:
    """Compare a session's produced output against its golden.

    orchestrate writes output BESIDE the source, so the produced tree is
    <source_dir>/<session>/cc_output; the golden is tests/expected/<session>/cc_output.
    (run_tests.py redirects output elsewhere but compares the identical subtree —
    same comparator, different roots.)
    """
    produced_cc = Path(source_dir) / session_dir / OUTPUT_ROOT
    expected_cc = EXPECTED / session_dir / OUTPUT_ROOT
    comparison = compare_output_tree(produced_cc, expected_cc)
    for line in comparison.summary_lines():
        (logger.info if comparison.ok else logger.error)(line)
    return comparison.ok


def run_orchestrate(source_dir: Path, session_dirs: list, dry_run: bool = False,
                    timeout: int = 600, compare: bool = True, clean: bool = True) -> bool:
    """Run orchestrate once per session, each with its own timeout budget.

    The timeout is per session, not shared across the run: a shared budget kills
    whichever session happens to be in flight when it expires, regardless of its
    own health, and reports the wrong one as failed.

    A session passes only if orchestrate succeeded AND (unless disabled, or a dry
    run) its output matches the golden in tests/expected/. Each session's output
    is wiped BEFORE it runs (unless disabled), so the comparison only ever sees
    this run's output — here the target is <source>/<session>/cc_output, beside
    the input media, which is exactly why wipe_output_tree refuses any dir not
    named cc_output.
    """
    results = {}
    for sd in session_dirs:
        if clean and not dry_run:
            wipe_output_tree(Path(source_dir) / sd / OUTPUT_ROOT, OUTPUT_ROOT)
        ok = run_one_session(source_dir, sd, dry_run=dry_run, timeout=timeout)
        if ok and compare and not dry_run:
            ok = compare_session(source_dir, sd) and ok
        results[sd] = ok

    passed = [sd for sd, ok in results.items() if ok]
    failed = [sd for sd, ok in results.items() if not ok]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Orchestrate tests: {len(passed)}/{len(results)} session(s) passed")
    for sd in failed:
        logger.error(f"  ✗ {sd}")
    logger.info(f"{'=' * 60}")
    return not failed


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Run orchestrate.py against test session directories"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Root directory containing the session folders (default: {DEFAULT_SOURCE_DIR})"
    )
    parser.add_argument(
        "--session-dirs",
        nargs="+",
        default=DEFAULT_TEST_DIRS,
        help="One or more session directories to process, relative to --source-dir "
             '(e.g. MultiROI SingleVideo "Weeks/Week 13")'
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making any changes"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-session timeout in seconds (default: 600). Each session gets "
             "its own budget, not a single shared one."
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip comparing produced output against tests/expected/ goldens."
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not wipe each session's cc_output before it runs "
             "(by default output is cleaned first so the run starts fresh)."
    )

    args = parser.parse_args()

    success = run_orchestrate(
        args.source_dir, args.session_dirs,
        dry_run=args.dry_run, timeout=args.timeout,
        compare=not args.no_compare, clean=not args.no_clean,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
