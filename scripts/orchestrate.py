#!/usr/bin/env python3
"""
WhisperX Orchestrator

Processes an explicit list of session folders (from --session-dirs or
config.yaml orchestration.session_dirs): configures WhisperX mounts, runs
transcription/scene detection, and tracks progress.

Usage:
    python orchestrate.py --help
    python orchestrate.py --session-dirs "<path>/Session_1" "<path>/Session_2" --parallel 2
    python orchestrate.py --dry-run   # uses config.yaml session_dirs
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.config import get_config

from pipeline.common.scenes import find_candidate_manual_scenes
from pipeline.common.sessions import (
    find_sessions,
    NoSessionsError,
    detect_audio_source,
    find_video_files,
)
# Stages are composed as functions, not subprocesses. Mounts, docker invocation
# and output handling belong inside each stage, which is why orchestrate imports
# no mount/docker/env helpers of its own.
# Stages are imported as MODULES, each exposing run(session_dir, ..., config,
# dry_run). The module name is the stage's identity and `run` is the verb, so
# there is one convention rather than the three this replaced (a bare name, a
# _tool suffix where the plain name collided with the pipeline function being
# wrapped, and a _stage suffix where the work lived in pipeline/). Which layer
# implements a stage is now an implementation detail of its module.
from cc_stages import (
    transcribe,
    detect_scenes_single_roi,
    detect_scenes_multi_roi,
    map_speakers,
    clean_transcription,
    merge_transcripts,
    merge_scenes,
    generate_storyboard,
)

def _apply_config_env_from_argv() -> None:
    """Honor a ``--config PATH`` CLI arg before the config singleton is built.

    get_config() caches a process-wide singleton, so WHISPERX_CONFIG has to be
    set before the first call — any later change is silently ignored once the
    singleton exists. A throwaway parser (add_help=False) scans argv without
    stealing --help, which main()'s real parser still handles.

    Called from main(), never at import: this reads sys.argv, and a module that
    does that on import acts on whatever command line the importing program
    happened to have. It used to run here at module scope, so importing
    orchestrate under pytest scanned pytest's argv — and `--config` naming a
    missing file called sys.exit() during test collection.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre.add_argument("--source-dir", type=Path, default=None)
    args, _ = pre.parse_known_args()
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            sys.exit(f"--config: file not found: {cfg_path}")
        os.environ["WHISPERX_CONFIG"] = str(cfg_path)
    elif args.source_dir and not os.environ.get("WHISPERX_CONFIG"):
        # No explicit --config: assume config.yaml lives at the source root.
        cfg_path = Path(args.source_dir) / "config.yaml"
        if cfg_path.exists():
            os.environ["WHISPERX_CONFIG"] = str(cfg_path)


from pipeline.common.logs import setup_logging

# Configured in main(), never here: importing a module must not reconfigure
# logging for whatever process happened to import it.
logger = logging.getLogger(__name__)


def resolve_session_dirs(session_dirs, source_dir: Path) -> List[Path]:
    """Resolve session directories against the effective source dir.

    Relative entries are joined to ``source_dir``; absolute entries pass
    through. No filesystem check happens here — existence is validated in
    exactly one place, ``find_sessions()``, which reports what it skipped.

    Both the normal and --merge-only paths call this, so the two branches
    cannot drift apart.
    """
    return [d if d.is_absolute() else source_dir / d for d in (session_dirs or [])]


def run_merge_tools(session_dir: Path, config, dry_run: bool = False) -> bool:
    """Compose the three merge tool functions (transcripts, scenes, storyboard)
    for a session. Returns True if all succeed.

    ``config`` is passed rather than reached for: every stage below already
    accepts one, and closing over a module-global made this composable only
    against the process-wide singleton.
    """
    logger.info(f"    Merging transcripts...")
    if not merge_transcripts.run(session_dir, config=config, dry_run=dry_run):
        logger.warning(f"    ⊘ Transcript merge failed")
        return False

    logger.info(f"    Merging scenes...")
    if not merge_scenes.run(session_dir, config=config, dry_run=dry_run):
        logger.warning(f"    ⊘ Scene merge failed")
        return False

    logger.info(f"    Generating storyboard...")
    if not generate_storyboard.run(session_dir=session_dir, config=config, dry_run=dry_run):
        logger.warning(f"    ⊘ Storyboard failed")
        return False

    return True




def process_session(
    session_dir: Path,
    config,
    dry_run: bool = False,
    no_scenes: bool = False,
    no_transcription: bool = False,
    roi_file: str = ""
) -> Tuple[bool, str, List[str]]:
    """Process a single session.

    Returns (success, status, failures):
      - success: False if any step (transcription, scene detection, merge)
        failed; True otherwise.
      - status: "no_media" (nothing to do), "completed" (ran, no failures), or
        "failed" (ran, at least one step failed).
      - failures: one human-readable message per failed step. Empty on success.
    """
    session_name = session_dir.name
    failures: List[str] = []

    logger.info(f"Processing: {session_name}")

    # Detect what media the session has. The tools mount the session directory
    # read-only and write their outputs back into it — no staging/mirroring here.
    audio_source, audio_files = detect_audio_source(session_dir)
    video_files = find_video_files(session_dir)
    has_video = bool(video_files)
    has_separate_audio = audio_source in ("standalone", "audacity")
    logger.info(f"  Audio source: {audio_source}")

    if not has_separate_audio and not has_video:
        logger.info(f"  ⊘ No audio or video files found, skipping")
        return True, "no_media", failures

    # 1. Transcription. One stage handles every source type — it bind-mounts the
    #    session read-only and writes transcriptions/ itself. Speaker-ID is a
    #    separate stage run afterwards, so all sources are identified the same way.
    if not no_transcription:
        logger.info(f"Starting Audio Detection")
        if audio_source in ("audacity", "standalone") or has_video:
            ok = transcribe.run(session_dir, config=config, dry_run=dry_run)
            if ok:
                ok = map_speakers.run(session_dir, config=config, dry_run=dry_run)
        else:
            ok = True
        if not ok:
            logger.error(f"  ✗ Transcription failed")
            failures.append("transcription failed")
        else:
            # Confidence (opt-in) + filler removal on the per-source
            # transcripts, before merge. Uniform across all workflows.
            clean_transcription.run(session_dir, config=config, dry_run=dry_run)

    # 2. Scene detection (video only) — UNLESS the session provides manual
    #    scenes. If a scenes.manual_source folder (e.g. screens/) exists in the
    #    session, the user has captured scenes by hand, so skip PySceneDetect and
    #    let the merge step consume those instead (see merge_scenes.py).
    #    A roi_file selects multi-ROI; the tool reads the ROI file straight from
    #    the read-only session mount.
    manual_source = config.scene_manual_source
    has_manual_scenes = bool(manual_source) and (session_dir / manual_source).exists()

    # Before falling through to auto-detection, check the session isn't holding
    # hand-captured scenes under a DIFFERENT folder name. Auto-detecting in that
    # case silently discards the user's work and produces a storyboard built
    # from the wrong scenes — output that looks correct and isn't. Refuse rather
    # than guess; the remedy is one config line either side.
    if not no_scenes and has_video and not has_manual_scenes:
        candidates = find_candidate_manual_scenes(
            session_dir,
            config.get("scenes", "manual_csv_name", "Scenes.csv"),
            exclude=manual_source,
        )
        if candidates:
            names = ", ".join(f"{c.name}/" for c in candidates)
            logger.error(
                f"  ✗ Found hand-captured scenes in {names} but scenes.manual_source "
                f"is '{manual_source}'. Refusing to run PySceneDetect, which would "
                f"discard them and build the storyboard from auto-detected scenes.\n"
                f"    Fix either side:\n"
                f"      - set scenes.manual_source: {candidates[0].name}  in config.yaml, or\n"
                f"      - rename the folder to '{manual_source}' "
                f"(CaptureScreens: output.scenes_dir_name)"
            )
            return False, "no_media", "Found unused hand-captured scenes"

    if not no_scenes and has_video and has_manual_scenes:
        logger.info(f"  → Manual scenes present ({manual_source}/), skipping PySceneDetect")
    elif not no_scenes and has_video:
        logger.info(f"Starting Scene Detection")
        # Multi-ROI only when the ROI file actually exists in this session.
        # Otherwise (no roi_file configured, or it's missing here) fall back to
        # single-ROI using scenes.roi from config.yaml — a missing multi-ROI file
        # should degrade gracefully, not fail the session.
        roi_path = (session_dir / roi_file) if roi_file else None
        if roi_path and roi_path.exists():
            ok = detect_scenes_multi_roi.run(
                session_dir, roi_file=roi_path, config=config, dry_run=dry_run
            )
        else:
            if roi_file:
                logger.info(
                    f"  ROI file '{roi_file}' not found in {session_name}; "
                    f"falling back to single-ROI from config"
                )
            ok = detect_scenes_single_roi.run(session_dir, config=config, dry_run=dry_run)
        if not ok:
            logger.error(f"  ✗ Scene detection failed")
            failures.append("scene detection failed")

    # Merge transcripts + (optional) scenes into a storyboard document. This is
    # transcript-driven, NOT video-gated: an audio-only / no-scenes session still
    # produces a transcript-only document. Scene merge is skipped gracefully when
    # there are no scenes (see run_merge_tools / merge_scenes).
    if not no_transcription and config.auto_merge:
        logger.info(f"  → Merging transcripts and scenes...")
        try:
            if run_merge_tools(session_dir, config, dry_run):
                logger.info(f"  ✓ Merge complete")
            else:
                logger.warning(f"  ⊘ Merge failed")
                failures.append("merge/storyboard failed")
        except Exception as e:
            logger.error(f"  ✗ Merge/storyboard failed: {e}")
            failures.append(f"merge/storyboard raised an exception: {e}")
    elif not no_transcription and not config.auto_merge:
        logger.info(f"  → Skipping merge (auto_merge disabled - delete images, then run --merge-only)")

    if failures:
        logger.error(
            f"  ✗ {session_name} finished with {len(failures)} failure(s): "
            f"{'; '.join(failures)}"
        )
        return False, "failed", failures

    logger.info(f"  ✓ Completed")
    return True, "completed", failures


def _run_session(session_dir: Path, config, dry_run: bool, no_scenes: bool,
                 no_transcription: bool, roi_file: str) -> Tuple[str, List[str]]:
    """Process one session, turning an unhandled exception into a failed status.

    Both the parallel and the sequential path call this, so the exception policy
    is written once instead of restated per branch. Takes its arguments
    explicitly rather than an argparse namespace, so it can be exercised without
    building one.

    Returns ``(status, failures)``. process_session's first return value says
    whether everything succeeded, which is the same question as ``status ==
    "completed"``; the counting has always used the status, so it is dropped here.
    """
    try:
        _, status, failures = process_session(
            session_dir, config, dry_run, no_scenes, no_transcription, roi_file)
        return status, failures
    except Exception as e:
        logger.error(f"Session processing failed: {e}")
        return "failed", [f"unhandled exception: {e}"]


def _tally(session_dir: Path, status: str, failures: List[str],
           counts: Dict[str, int], session_failures: Dict[str, List[str]]) -> None:
    """Record one session's outcome. Anything not completed or failed is skipped."""
    if status == "completed":
        counts["completed"] += 1
    elif status == "failed":
        counts["failed"] += 1
        session_failures[session_dir.name] = failures
    else:
        counts["skipped"] += 1


def main():
    # Both of these used to run at import. Reading argv and the filesystem is
    # an entry point's job: doing it at module scope meant importing orchestrate
    # — which one unit test does, for a single pure function — scanned the
    # importing program's command line and built the process-wide config
    # singleton before any caller had chosen one.
    _apply_config_env_from_argv()
    try:
        cfg = get_config()
    except (ValueError, FileNotFoundError) as e:
        # _validate() writes these for a person to read; a traceback buries the
        # message under a stack that says nothing about which knob is wrong.
        sys.exit(str(e))

    parser = argparse.ArgumentParser(
        description="Orchestrate WhisperX processing for Champions/GGG Issues sessions"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=cfg.source_dir,
        help=f"Root directory containing session folders (default: {cfg.source_dir})"
    )
    parser.add_argument(
        "--session-dirs",
        nargs="+",
        type=Path,
        # NOT cfg.session_dirs: that resolves against the *config's* source_dir,
        # which is decided before --source-dir is parsed. Default to the raw
        # configured values and resolve them against the effective source dir in
        # resolve_session_dirs() below, so --source-dir actually takes effect.
        default=cfg.session_dirs_config,
        help="Explicit list of session directories to process. "
             "Defaults to orchestration.session_dirs from config.yaml when set. "
             "Example: --session-dirs MultiVideo SingleVideo MultiROI"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=cfg.parallel_workers,
        help=f"Number of concurrent sessions (default: {cfg.parallel_workers})"
    )
    parser.add_argument(
        "--no-scenes",
        action="store_true",
        help="Don't produce scene detection output"
    )
    parser.add_argument(
        "--roi-file",
        type=str,
        default=os.environ.get("ROI_FILE", ""),
        help="Filename of ROI config JSON (e.g., 'roi_history.json'). File will be looked for in each session directory. Can also set ROI_FILE environment variable."
    )
    parser.add_argument(
        "--no-transcription",
        action="store_true",
        help="Don't run transcription (only scene detection)"
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip processing, only run merge/combine step (requires existing outputs)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (defaults to project root config.yaml). "
             "Applied to WHISPERX_CONFIG before config is loaded."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making any changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # --config was already applied to WHISPERX_CONFIG at import time (see
    # _apply_config_env_from_argv), so the config singleton picked it up; it is
    # re-declared here only so it appears in --help and passes arg validation.

    # First thing after the args are known: --verbose picks the level, so this
    # has to wait for parse_args rather than run at the top of main().
    setup_logging(verbose=args.verbose)

    # If no --roi-file (and no ROI_FILE env) was given, fall back to
    # scenes.roi_file from config.yaml, so a configured multi-ROI file is honored
    # without having to repeat it on the command line. Empty => single-ROI.
    effective_roi_file = args.roi_file or (cfg.scene_roi_file or "")

    # Handle --merge-only mode
    if args.merge_only:
        logger.info("=" * 50)
        logger.info("WhisperX Merge-Only Mode")
        logger.info("=" * 50)
        logger.info(f"Source:        {args.source_dir}")

        resolved_session_dirs = resolve_session_dirs(args.session_dirs, args.source_dir)

        logger.info(f"Session Dirs:  {len(resolved_session_dirs)} explicit director(ies)")
        logger.info(f"Dry Run:        {args.dry_run}")
        logger.info("=" * 50)

        # Resolve the explicit sessions to merge
        try:
            sessions = find_sessions(resolved_session_dirs)
        except NoSessionsError as e:
            logger.error(str(e))
            return 1

        if not sessions:
            logger.error("No session directories found to merge")
            return 1

        # Merge each explicit session directly.
        merged_count = 0
        for session in sessions:
            try:
                logger.info(f"  → Merging {session.name}...")
                if run_merge_tools(session, cfg, args.dry_run):
                    logger.info(f"  ✓ {session.name} merged")
                    merged_count += 1
                else:
                    logger.warning(f"  ⊘ {session.name} merge failed")
            except Exception as e:
                logger.error(f"  ✗ Merging {session.name} failed: {e}")

        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Merge Complete: {merged_count} session(s) merged")
        logger.info("=" * 50)
        return 0

    # Normal processing mode
    logger.info("=" * 50)
    logger.info("WhisperX Orchestrator")
    logger.info("=" * 50)
    logger.info(f"Source:        {args.source_dir}")

    resolved_session_dirs = resolve_session_dirs(args.session_dirs, args.source_dir)

    logger.info(f"Session Dirs:  {len(resolved_session_dirs)} explicit director(ies)")
    logger.info(f"Parallel Jobs: {args.parallel}")
    logger.info(f"Dry Run:        {args.dry_run}")
    logger.info(f"Transcription:  {not args.no_transcription}")
    logger.info(f"Scene Detection: {not args.no_scenes}")
    if effective_roi_file:
        via = "flag/env" if args.roi_file else "config.yaml"
        roi_mode = f"Multi-ROI ({effective_roi_file}, via {via})"
    else:
        roi_mode = "Single-ROI"
    logger.info(f"Scene Mode:     {roi_mode}")
    logger.info(f"Auto Merge:     {cfg.auto_merge}")
    logger.info("=" * 50)

    # Resolve the explicit sessions to process
    try:
        sessions = find_sessions(resolved_session_dirs)
    except NoSessionsError as e:
        logger.error(str(e))
        return 1

    if not sessions:
        logger.error("No sessions found (all listed directories missing?)")
        return 1

    logger.info(f"Found {len(sessions)} session(s) to process\n")

    # Process sessions. A dict rather than three ints so _tally can record into
    # it — the counting is shared by both paths below.
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    # session name -> list of failure messages, for the end-of-run report.
    session_failures: Dict[str, List[str]] = {}

    if args.parallel > 1:
        # Parallel processing. _run_session swallows the exception, so
        # future.result() cannot raise and this branch needs no try of its own.
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    _run_session, session, cfg, args.dry_run, args.no_scenes,
                    args.no_transcription, effective_roi_file,
                ): session for session in sessions
            }
            for future in as_completed(futures):
                status, failures = future.result()
                _tally(futures[future], status, failures, counts, session_failures)
    else:
        # Sequential processing — the default, and deliberately on this thread:
        # Ctrl-C and tracebacks behave the way anyone debugging expects.
        for session in sessions:
            status, failures = _run_session(
                session, cfg, args.dry_run, args.no_scenes,
                args.no_transcription, effective_roi_file)
            _tally(session, status, failures, counts, session_failures)
            logger.info("")

    logger.info("")
    logger.info("=" * 50)
    logger.info(f"Processing Complete: {counts['completed']} completed, "
                f"{counts['skipped']} skipped, {counts['failed']} failed")
    if session_failures:
        logger.info("Failures:")
        for name, msgs in session_failures.items():
            for msg in msgs:
                logger.info(f"  - {name}: {msg}")
    logger.info("=" * 50)
    return 1 if counts["failed"] > 0 else 0


if __name__ == "__main__":

    sys.exit(main())
