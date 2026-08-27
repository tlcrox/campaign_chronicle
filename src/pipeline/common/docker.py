#!/usr/bin/env python3
"""
docker.py - thin wrapper around `docker compose run` with env injection.

Commands are argument LISTS, never shell strings, and run without ``shell=True``.
No shell parses them, so a path or hint value containing a quote, a ``%VAR%`` or
a ``$VAR`` reaches the container as written instead of being re-interpreted on
the way past. Logs render the list back with :func:`printable_command`, which is
the argv that actually ran rather than a string that a shell might still rewrite.

The working directory is an explicit ``cwd`` argument, never a module global, so
callers with different roots can share this. Default ``cwd=None`` runs in the
current directory, matching subprocess defaults.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Compose narrates container lifecycle on stderr before the container's own
# output ("Container <name> Creating" / "Created"). Its presence is the one
# reliable signal that the container actually STARTED — `docker compose run`
# returns exit 1 for a daemon failure, a container failure and a missing binary
# alike, so the exit code cannot tell them apart.
#
# Load-bearing: do not add `--progress quiet` (or a TTY-suppressing flag) to the
# compose invocation in compose_run, or these lines disappear and every failure
# starts looking like a pre-start one again.
_CONTAINER_STARTED = re.compile(
    r"^\s*Container\s+\S+\s+(Creating|Created|Starting|Started)", re.M | re.I)

# Substrings that indicate a *transient* Docker daemon/engine problem (worth a
# retry) rather than a real command failure (bad input, transcription error).
# Seen e.g. when Docker Desktop's engine 500s on a project-network lookup during
# rapid back-to-back `docker compose run` invocations.
#
# ": eof" and not "eof": Docker writes the truncated-stream case as
# "...: EOF", while the messages that merely CONTAIN those three letters are
# almost all genuine failures — "unexpected EOF while reading input" from a
# truncated video, "EOFError", "tar: Unexpected EOF in archive". A bare "eof"
# retried every one of them.
_TRANSIENT_DOCKER_MARKERS = (
    "500 internal server error",
    "internal server error for api route",
    "the server supports the requested api version",
    "error during connect",
    "cannot connect to the docker daemon",
    "connection refused",
    ": eof",
)


def _is_transient_docker_error(stderr: str) -> bool:
    """True only for a failure that happened BEFORE the container started.

    Retrying exists for one thing: the daemon failing to set the run up. Once
    the container is running, whatever it reports is the run's own verdict —
    ffmpeg refusing a truncated file, a model that will not load, a config the
    entrypoint rejects — and repeating it just spends another
    DOCKER_RUN_TIMEOUT reaching the same answer. At two hours an attempt, a
    corrupt input used to cost six.

    The marker list only applies once the container is ruled out, so a marker
    that is too broad can no longer swallow a container-side failure.
    """
    s = stderr or ""
    if _CONTAINER_STARTED.search(s):
        return False
    return any(marker in s.lower() for marker in _TRANSIENT_DOCKER_MARKERS)


# Wall clock for ONE `docker compose run`, in seconds. Two hours, sized for the
# worst realistic single call: a multi-file session transcribed with large-v3 and
# diarization, plus a first-run model download into the whisperx-models volume.
#
# Not a budget for a session or a run — orchestrate invokes the containers many
# times, and each retry gets its own full allowance. Anything wrapping this call
# should import the value rather than restate it, so a driver can never kill a
# container the layer beneath it is still waiting on.
DOCKER_RUN_TIMEOUT = 7200

# On failure the terminal gets the tail and the full streams go to a file. A
# failed transcription can hold thousands of lines, and the path that dumps them
# is the one taken when something has already gone wrong — the useful part is
# nearly always the last thing said before the exit.
FAILURE_TAIL_LINES = 10
DOCKER_LOG_NAME = "docker.log"


# Which compose file defines which service. There is no root docker-compose.yml,
# so every invocation must pass -f. Paths are relative to the repo root (the cwd
# these commands run in).
COMPOSE_FILES = {
    "whisperx": "compose/whisperx/docker-compose.yml",
    "scenes": "compose/pyscenedetect/docker-compose.yml",
    "scenes_multi": "compose/pyscenedetect/docker-compose.yml",
}


class UnknownService(KeyError):
    """Raised when a service has no registered compose file."""


def printable_command(command: List[str]) -> str:
    """Render an argv list the way you would retype it, for logs and dry runs.

    Only ever for display. The list is what runs; this is what gets logged, so a
    failure line can be pasted into a terminal and reproduce the failure exactly.
    """
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def compose_run(service: str, repo_root: Optional[Union[str, Path]] = None) -> List[str]:
    """Build the ``docker compose ... run --rm <service>`` argv for a service.

    Callers pass the result to :func:`run_docker_command` with ``cwd`` set to the
    repo root, which inserts any ``-e``/``-v`` flags after the ``--rm`` element.

    Note compose resolves relative paths inside a compose file against that
    FILE's directory, not the cwd — which is why the compose files use ../../
    to reach back to the repo root.

    ``repo_root`` is used to locate ``.env``. Compose loads ``.env`` from the
    *project directory*, which defaults to the directory holding the compose
    file — here ``compose/<service>/``, not the repo root. Without an explicit
    ``--env-file`` the root ``.env`` is invisible and ``${HF_TOKEN:-}`` resolves
    to empty, so diarization fails with an unset token even though ``.env`` is
    right there. HF_TOKEN is a secret and deliberately NOT passed as ``-e`` (see
    pipeline.transcribe.docker_env), so this is its only route into the
    container.

    ``--project-directory`` would also find ``.env``, but it re-bases every
    relative path inside the compose file and would break the ../../ mounts.
    """
    try:
        compose_file = COMPOSE_FILES[service]
    except KeyError:
        raise UnknownService(
            f"No compose file registered for service {service!r}. "
            f"Known: {', '.join(sorted(COMPOSE_FILES))}"
        ) from None

    command = ["docker", "compose"]
    if repo_root is not None:
        env_path = Path(repo_root) / ".env"
        if env_path.is_file():
            command += ["--env-file", str(env_path)]
    command += ["-f", compose_file, "run", "--rm", service]
    return command


def _log_stream(text: Optional[str], log, label: Optional[str] = None) -> None:
    """Log a captured stream line by line, indented. Silent when it is empty."""
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    if not lines:
        return
    if label:
        log(f"  {label}:")
    for line in lines:
        log(f"    {line}")


def _log_tail(text: Optional[str], label: str, log,
              limit: int = FAILURE_TAIL_LINES) -> None:
    """Log the last ``limit`` non-blank lines, saying how many were left out."""
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    if not lines:
        return
    shown = lines[-limit:]
    elided = len(lines) - len(shown)
    suffix = f" ({elided} earlier line(s) in the log)" if elided else ""
    log(f"  last {len(shown)} line(s) of {label}{suffix}:")
    for line in shown:
        log(f"    {line}")


def _write_failure_log(log_dir, printable: str, result) -> Optional[Path]:
    """Append the whole run — command and both streams — to <log_dir>/docker.log.

    Appends rather than truncates. Normally there is one entry: the stage clears
    this directory before it runs, and only the final attempt of a retry writes.
    Append is for the cases that break that assumption — a caller that skips the
    clear, or a stage invoked twice in one process — where truncating would throw
    away the earlier failure. Each entry carries a timestamp, the exit code and
    the command, so they stay tellable apart.

    Never raises: this runs while reporting a failure, and a problem writing the
    log must not replace the failure being reported.
    """
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / DOCKER_LOG_NAME
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"\n{'=' * 78}\n")
            fh.write(f"{stamp}  exit {result.returncode}\n{printable}\n")
            fh.write(f"{'=' * 78}\n")
            for label in ("stdout", "stderr"):
                fh.write(f"\n--- {label} ---\n")
                fh.write(getattr(result, label, None) or "")
                fh.write("\n")
        return path
    except OSError as e:
        logger.warning(f"Could not write {DOCKER_LOG_NAME} to {log_dir}: {e}")
        return None


def run_docker_command(
    command: List[str],
    dry_run: bool = False,
    env_vars: Optional[Dict[str, str]] = None,
    cwd: Optional[Union[str, Path]] = None,
    retries: int = 2,
    backoff: float = 3.0,
    volumes: Optional[List[str]] = None,
    log_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """Run a docker compose command with optional env vars and bind mounts.

``command`` is an argv list, typically from :func:`compose_run`.

    ``env_vars`` are injected as ``-e KEY=VALUE`` (overriding the compose
    ``environment:``). ``volumes`` are injected as ``-v HOST:CONTAINER[:ro]``
    (each entry a ready-formatted string; host paths must be absolute). Both are
    inserted right after the ``--rm`` element. Values are passed as their own argv
    entries and never quoted by hand: no shell sees them, so a quote, an ``&`` or
    a ``%VAR%`` inside a path or a hint string arrives intact.

    ``log_dir`` is where a failed run's full output is written (as
    ``docker.log``); the terminal gets only the tail. Callers pass the output
    directory of the stage making the call — ``cc_output/transcriptions`` or
    ``cc_output/scenes_output`` — so the log sits with the output it explains,
    and two stages failing in one session do not interleave in one file. That
    directory is also cleared at the start of each run of that stage, so the log
    describes the current run rather than accumulating across them.

    Omit it and the tail is still all that reaches the terminal — the output is
    simply not kept.

    Retries only on *transient* Docker daemon errors (see
    ``_TRANSIENT_DOCKER_MARKERS``) up to ``retries`` times with a linear
    ``backoff`` (seconds × attempt). Real command failures (non-zero exit with
    no transient marker) fail immediately, as before.
    """
    # Build command with -e / -v flags if provided
    flags = []
    if env_vars:
        for key, value in env_vars.items():
            flags += ["-e", f"{key}={value}"]
    if volumes:
        for volume in volumes:
            flags += ["-v", volume]

    full_command = list(command)
    if flags:
        # Insert flags right after '--rm'. A command without it cannot carry
        # them, and silently dropping every env var (which a string replace on a
        # missing needle would do) is far worse than refusing to run.
        try:
            at = full_command.index("--rm") + 1
        except ValueError:
            raise ValueError(
                "run_docker_command needs a '--rm' element to insert -e/-v after; "
                f"got {full_command!r}"
            ) from None
        full_command[at:at] = flags

    printable = printable_command(full_command)

    if dry_run:
        extra = ""
        if volumes:
            extra = "  [mounts: " + "; ".join(volumes) + "]"
        logger.info(f"    [DRY RUN] Would run: {printable}{extra}")
        return True

    attempt = 0
    while True:
        try:
            result = subprocess.run(
                full_command,
                cwd=cwd,
                capture_output=True,
                text=True,
                # Docker emits UTF-8 status glyphs (✔, the ⠿ spinner) before the
                # container's own output. Decode as UTF-8 explicitly and replace
                # anything undecodable, so the pipe reader can't die with a
                # UnicodeDecodeError under the Windows cp1252 locale default.
                encoding="utf-8",
                errors="replace",
                timeout=DOCKER_RUN_TIMEOUT,
            )
        except FileNotFoundError as e:
            # Without a shell there is no "'docker' is not recognized" on stderr
            # to capture, so the command has to be named here or the message is
            # just an errno with no subject.
            logger.error(f"Command not found: {printable}")
            logger.error(f"    {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to run command: {printable}")
            logger.error(f"    {e}")
            return False

        if result.returncode == 0:
            _log_stream(result.stdout, logger.info)
            return True

        # Non-zero exit: retry only if it looks like a transient daemon hiccup.
        if _is_transient_docker_error(result.stderr) and attempt < retries:
            attempt += 1
            wait = backoff * attempt
            logger.warning(
                f"    Transient Docker error (attempt {attempt}/{retries}); "
                f"retrying in {wait:.0f}s..."
            )
            logger.debug(f"    stderr: {(result.stderr or '').strip()[:500]}")
            time.sleep(wait)
            continue

        # BOTH streams, but to a file. `docker compose run` puts its own progress
        # on stderr and the CONTAINER's output on stdout, so a stage that fails
        # inside the container explains itself on stdout — which was once
        # discarded entirely, and then dumped whole. The terminal gets the tail,
        # because the error is nearly always the last thing said; the middle is
        # what the log is for.
        logger.error(f"Command failed (exit {result.returncode}): {printable}")
        log_path = _write_failure_log(log_dir, printable, result) if log_dir else None
        if (result.stderr or "").strip():
            _log_tail(result.stderr, "stderr", logger.error)
        else:
            _log_tail(result.stdout, "stdout", logger.error)
        if log_path:
            logger.error(f"  full output: {log_path}")
        return False
    
