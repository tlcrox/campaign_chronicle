#!/usr/bin/env python3
"""
mounts.py - Docker mount points and mount directory helpers.

Defines hard-coded mount paths that match docker-compose.yml.
These should NOT be configurable — they must always match the Docker setup.

Also provides helpers for copying, clearing, and mirroring files to/from mounts.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# Hard-coded Docker mount points (must match docker-compose.yml)
# ============================================================================

# Host-side mount points (relative to project root where docker-compose.yml is)
# These are the directories that get bind-mounted into the Docker container
AUDIO_MOUNT_HOST = Path("./audio")
VIDEO_MOUNT_HOST = Path("./video")
OUTPUT_MOUNT_HOST = Path("./output")
SCRIPTS_MOUNT_HOST = Path("./scripts")
VOICES_MOUNT_HOST = Path("../voices")  # Speaker voice embeddings/voiceprints

# Container-side mount points (used by scripts running inside Docker)
# These are the paths visible to processes running in the container
AUDIO_MOUNT_CONTAINER = Path("/audio")
VIDEO_MOUNT_CONTAINER = Path("/video")
OUTPUT_MOUNT_CONTAINER = Path("/output")
SCRIPTS_MOUNT_CONTAINER = Path("/usr/local/bin/scripts")
VOICES_MOUNT_CONTAINER = Path("/voices")  # Not currently mounted in docker-compose

# Default usage: host-side paths (used by orchestrate.py and tools calling Docker)
AUDIO_MOUNT = AUDIO_MOUNT_HOST
VIDEO_MOUNT = VIDEO_MOUNT_HOST
OUTPUT_MOUNT = OUTPUT_MOUNT_HOST
SCRIPTS_MOUNT = SCRIPTS_MOUNT_HOST
VOICES_MOUNT = VOICES_MOUNT_HOST

# ============================================================================
# Per-session output subdir names (host-side, one level under each session dir)
# ============================================================================
# Single source of truth: the transcribe/detect tools WRITE to these subdirs and
# the merge tools READ from them. Hard-coded (not configurable) — this is an
# internal contract between the tools, in the same spirit as the mount points
# above. Both sides import these constants, so a name change moves writer and
# reader together.
#
# EVERYTHING GENERATED LIVES UNDER ONE ROOT:
#
#     Week 13/
#       video.mp4            <- the user's media
#       cc_output/
#         transcriptions/
#         scenes_output/
#         combined_output/
#
# Keep it that way. One root means the source folder stays readable, "delete
# everything this tool made" is a single rmtree, and a test run resets with no
# risk of touching input media.
OUTPUT_ROOT = "cc_output"                   # all generated output, one folder

SESSION_OUTPUT_SUBDIR = f"{OUTPUT_ROOT}/transcriptions"    # WhisperX transcripts
SCENES_OUTPUT_SUBDIR = f"{OUTPUT_ROOT}/scenes_output"      # PySceneDetect CSVs + images
COMBINED_OUTPUT_SUBDIR = f"{OUTPUT_ROOT}/combined_output"  # merged + storyboard

# ============================================================================
# Mount directory helpers
# ============================================================================



def clear_mount(mount_dir: Path, dry_run: bool = False):
    """Clear mount directory"""
    if not mount_dir.exists():
        return

    if dry_run:
        logger.info(f"    [DRY RUN] Would clear {mount_dir}")
        return

    try:
        for item in mount_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    except Exception as e:
        logger.error(f"Failed to clear {mount_dir}: {e}")


class UnsafeOutputDir(ValueError):
    """Raised when the derived output dir isn't safely inside its output base."""


def session_output_base(session_dir: Path, output_base=None, session_key=None) -> Path:
    """Where this session's generated output lives.

    Default (``output_base=None``): the session directory itself, so output lands
    at ``<session>/cc_output/...`` — alongside the input, which is what a user
    processing their own recordings usually wants.

    With ``output_base`` set: ``<output_base>/<session_key>/`` — the source tree
    is left completely untouched. ``session_key`` MUST be the session's path
    relative to ``source_dir``, not its leaf name: two sources can each hold a
    ``Week 13``, and keying on the leaf would silently merge them into one output
    folder. (The same collapse of ``Weeks/Week 13`` to ``Week 13`` has already
    caused a bug once, in run_tests.py.)

    Either way the OUTPUT_ROOT level is preserved, so both layouts produce an
    identical ``cc_output/...`` tree and one directory comparison works for both.
    """
    if not output_base:
        return Path(session_dir)
    key = Path(session_key) if session_key else Path(session_dir).name
    # Reject any ANCHORED key, not just is_absolute(): on Windows
    # Path("/etc").is_absolute() is False (no drive letter), yet
    # base / "/etc" still escapes to the drive root ("C:\etc"). `.anchor` is
    # non-empty for absolute AND drive-relative-rooted paths on both OSes, while
    # staying empty for the relative keys (e.g. "Weeks/Week 13") this expects.
    if Path(key).anchor:
        raise UnsafeOutputDir(f"session_key must be relative, got {key!r}")
    return Path(output_base) / key


def output_dir_for(session_dir: Path, output_subdir: str, config=None) -> Path:
    """Resolve a session's output dir for a given subdir, honouring the config.

    Uses ``getattr`` rather than attribute access so any object that quacks like
    a config still works — several tests (and any external caller) pass small
    stubs that predate ``output_base_dir``. A stub without it simply gets the
    default "beside the source" behaviour instead of an AttributeError.
    """
    base = getattr(config, "output_base_dir", None) if config is not None else None
    key = None
    if base is not None and hasattr(config, "session_key"):
        key = config.session_key(session_dir)
    return session_output_base(session_dir, base, key) / output_subdir


def session_mounts(
    session_dir: Path,
    output_subdir: str,
    in_env_key: str,
    out_env_key: str,
    in_mount: str = "/session_input",
    out_mount: str = "/session_output",
    output_base=None,
    session_key=None,
):
    """Build per-session bind mounts for a Docker run.

    Mounts the session directory READ-ONLY as ``in_mount`` (so the container can
    never modify or delete the originating source), and ``<session>/<output_subdir>``
    writable as ``out_mount``. Returns the ``docker compose run -v`` volume
    strings plus the env vars that point the container's input/output dir
    variables at those mounts.

    Args:
        session_dir: The session directory (its own contents are the input).
        output_subdir: Name of the derived-output subdir under the session
            (e.g. ``"scenes_output"`` or ``"transcriptions"``).
        in_env_key / out_env_key: Container env var names for the input/output
            dirs (e.g. ``"VIDEO_DIR"``/``"SCENES_DIR"`` or ``"AUDIO_DIR"``/``"OUTPUT_DIR"``).
        in_mount / out_mount: Container-side mount paths.

    Returns:
        ``(volumes, dir_env, out_dir)`` where
          * ``volumes`` = ``["<session>:<in_mount>:ro", "<out_dir>:<out_mount>"]``
          * ``dir_env`` = ``{in_env_key: in_mount, out_env_key: out_mount}``
          * ``out_dir`` = resolved ``Path`` of ``<session>/<output_subdir>``

    Raises:
        UnsafeOutputDir: if ``output_subdir`` would resolve to the output base
            itself, an ancestor, or anywhere outside it (e.g. ``".."`` or an
            absolute path). Guards against clearing/mounting source.

    The guard is anchored on the OUTPUT BASE, not the session: with
    ``output_base`` set the two differ, but the invariant is the same one that
    matters — ``clear_mount`` must never be handed a directory that could
    contain source material.
    """
    session_abs = Path(session_dir).resolve()
    out_base = session_output_base(session_dir, output_base, session_key)
    out_dir = (out_base / output_subdir).resolve()
    out_base_abs = out_base.resolve()

    if out_dir == out_base_abs or out_base_abs not in out_dir.parents:
        raise UnsafeOutputDir(
            f"Unsafe output dir {out_dir} for base {out_base_abs} "
            f"(output_subdir={output_subdir!r} must stay inside the output base)"
        )

    volumes = [
        f"{session_abs}:{in_mount}:ro",   # SOURCE — read-only, never mutated/deleted
        f"{out_dir}:{out_mount}",         # derived output only
    ]
    dir_env = {in_env_key: in_mount, out_env_key: out_mount}
    return volumes, dir_env, out_dir



