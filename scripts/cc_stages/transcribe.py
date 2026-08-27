#!/usr/bin/env python3
"""
Transcribe a session's media with WhisperX — one stage for all source types.

Discovery is the only thing that varies; everything after it (mounts, docker
invocation, logging, error handling) is shared. Keep it that way — add a branch
to discover_source(), not a second copy of the run body.

WHAT DIFFERS BY SOURCE
    video       find_video_files(); requires diarization, because speaker
                identity comes from WhisperX's SPEAKER_XX labels
    standalone  loose audio files beside the session
    audacity    per-speaker exports; AUDIO_DIR is narrowed to the exports
                subfolder so the session's video is not transcribed as well

The source type is auto-detected, so callers need not dispatch. There is no
override flag: every caller trusts detect_audio_source, and a session it reads
wrongly is a bug to fix there rather than to work around per-run.

Usage (CLI):
    python transcribe.py --session-dir "Week 14"
    python transcribe.py --session-dir "Week 14" --dry-run
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pipeline.config import get_config, resolve_tool_config
from pipeline.common.mounts import (
    clear_mount, session_mounts, UnsafeOutputDir, SESSION_OUTPUT_SUBDIR,
)
from pipeline.common.docker import run_docker_command, compose_run
from pipeline.common.sessions import detect_audio_source, find_video_files
from pipeline.transcribe.map_speakers import require_diarization_for_mapping
from pipeline.transcribe.docker_env import whisper_env

from pipeline.common.logs import setup_logging

# Configured in main(), never here: importing a module must not reconfigure
# logging for whatever process happened to import it.
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSource:
    """What a session offers WhisperX, and how this source type differs."""

    kind: str                                  # video | standalone | audacity
    files: List[Path] = field(default_factory=list)
    label: str = "file"                        # for logging
    # Sub-path of the session to point AUDIO_DIR at, instead of the session root.
    # Only Audacity uses this: its per-speaker exports live in a subfolder, and
    # transcribing the session root would pick up the video as well.
    in_subdir: Optional[Path] = None
    # Video speakers come from diarization (SPEAKER_XX), so it must be enabled.
    needs_diarization: bool = False


def _audacity_export_subdir(session_dir: Path, audio_files: List[Path]) -> Optional[Path]:
    """Common parent of the per-speaker exports, relative to the session.

    Returns None when the exports sit at the session root — there is no subfolder
    to narrow to, and transcription would also pick up the video.
    """
    export_dir = Path(os.path.commonpath([str(f.parent) for f in audio_files]))
    try:
        rel = export_dir.relative_to(session_dir)
    except ValueError:
        return None
    if rel == Path("."):
        logger.warning(
            "  ⊘ Per-speaker exports resolve to the session root, which also holds "
            "the video; transcription cannot exclude the video. Put the exports in "
            "a subdirectory (e.g. the .aup folder)."
        )
        return None
    return rel


def discover_source(session_dir: Path) -> Optional[TranscriptionSource]:
    """Work out what this session is, or None when there is nothing to transcribe."""
    session_dir = Path(session_dir)
    audio_source, audio_files = detect_audio_source(session_dir)

    if audio_source == "audacity":
        if not audio_files:
            logger.warning(f"  ⊘ No audio files found in Audacity project")
            return None
        return TranscriptionSource(
            kind="audacity",
            files=audio_files,
            label="per-speaker audio file",
            in_subdir=_audacity_export_subdir(session_dir, audio_files),
        )

    if audio_source == "standalone":
        if not audio_files:
            logger.warning(f"  ⊘ No standalone audio files found")
            return None
        return TranscriptionSource(
            kind="standalone", files=audio_files, label="standalone audio file")

    video_files = find_video_files(session_dir)
    if not video_files:
        logger.warning(f"  ⊘ No video files found in {session_dir}")
        return None
    return TranscriptionSource(
        kind="video", files=video_files, label="video file", needs_diarization=True)


def run(session_dir: Path, config=None, dry_run: bool = False) -> bool:
    """Transcribe a session, whatever kind of media it holds.

    Speaker mapping is NOT done here — it is its own stage
    (pipeline.transcribe.map_speakers), run afterwards, so every source type is
    identified the same way.
    """
    if config is None:
        config = get_config()
    session_dir = Path(session_dir)

    src = discover_source(session_dir)
    if src is None:
        return False

    logger.info(f"Transcribing {src.kind} in {session_dir.name}...")
    logger.info(f"  Found {len(src.files)} {src.label}(s)")

    if src.needs_diarization:
        require_diarization_for_mapping(config)

    try:
        volumes, dir_env, out_dir = session_mounts(
            session_dir,
            SESSION_OUTPUT_SUBDIR,
            "AUDIO_DIR", "OUTPUT_DIR",
            output_base=getattr(config, "output_base_dir", None),
            session_key=(config.session_key(session_dir)
                         if getattr(config, "output_base_dir", None) else None),
        )
    except UnsafeOutputDir as e:
        logger.error(f"  ✗ {e}")
        return False

    if src.in_subdir is not None:
        dir_env["AUDIO_DIR"] = (
            dir_env["AUDIO_DIR"].rstrip("/") + "/" + src.in_subdir.as_posix())
        logger.info(
            f"  exports dir: {src.in_subdir.as_posix()}  →  AUDIO_DIR={dir_env['AUDIO_DIR']}")

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        clear_mount(out_dir, dry_run=dry_run)  # derived output only (guarded)

    logger.info(f"  → Running WhisperX transcription...")
    env_vars = whisper_env(config)
    env_vars.update(dir_env)
    logger.info(f"  input (ro):  {session_dir.resolve()}")
    logger.info(f"  output:      {out_dir}")

    repo_root = config.repo_root
    if not run_docker_command(compose_run("whisperx", repo_root), dry_run,
                              env_vars=env_vars, cwd=repo_root, volumes=volumes,
                              log_dir=out_dir):
        logger.error(f"  ✗ Transcription failed")
        return False

    logger.info(f"  ✓ Transcription completed")
    logger.info(f"✓ Transcription complete for {session_dir.name}")
    return True


def main():
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Transcribe a session's media with WhisperX (source auto-detected)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py --session-dir "Week 14"
  python transcribe.py --session-dir "Week 14" --config custom_config.yaml
  python transcribe.py --session-dir "Week 14" --dry-run
        """
    )
    parser.add_argument("--session-dir", type=Path, required=True,
                        help="Path to session directory (e.g. Week_77, /path/to/Session_1)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.yaml (defaults to <repo>/config/config.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without making any changes")

    args = parser.parse_args()

    # Config: explicit --config wins, else walk up from --session-dir.
    resolve_tool_config(args.config, args.session_dir)

    success = run(args.session_dir, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
