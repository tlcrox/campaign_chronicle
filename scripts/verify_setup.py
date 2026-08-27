#!/usr/bin/env python3
"""
Verify the environment can actually run the pipeline.

Replaces verify_setup.sh, which was a WSL script rather than a portable one:
it hardcoded `/mnt/d/Champions/GGG Issues` (Git Bash maps D: to /d/, so that
path only exists under WSL), required `python3` (Windows installs `python.exe`),
and took `dirname` of itself as the repo root — correct only while it sat at the
repo root, silently wrong once it moved into scripts/.

This version imports what it checks rather than restating it: the repo root, the
compose-file registry and the config all come from the code they describe, so
they cannot drift out of sync the way the shell version did.

Usage:
    uv run python scripts/verify_setup.py
    uv run python scripts/verify_setup.py --source-dir "D:/Champions/Season 2"
    uv run python scripts/verify_setup.py --quiet     # only problems

Exit status is 0 when nothing is broken (warnings are fine), 1 otherwise — so it
is usable as a pre-flight check in a script.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ANSI colour, disabled when the terminal can't take it. Windows 10+ terminals
# and PowerShell handle these; older consoles and redirected output do not.
_COLOR = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
if os.name == "nt" and _COLOR:  # enable VT processing on legacy consoles
    os.system("")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


OK, WARN, BAD = "  [ok]  ", "  [warn]", "  [FAIL]"


class Report:
    """Collects results so the summary can distinguish fatal from cosmetic."""

    def __init__(self, quiet: bool = False):
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.quiet = quiet

    def ok(self, msg: str, detail: str = "") -> None:
        if not self.quiet:
            print(_c("0;32", OK) + f" {msg}")
            if detail:
                print(f"           {detail}")

    def warn(self, msg: str, detail: str = "") -> None:
        self.warnings.append(msg)
        print(_c("1;33", WARN) + f" {msg}")
        if detail:
            print(f"           {detail}")

    def fail(self, msg: str, detail: str = "") -> None:
        self.failures.append(msg)
        print(_c("0;31", BAD) + f" {msg}")
        if detail:
            print(f"           {detail}")


def _run(cmd: list[str], timeout: int = 60):
    """Run a command, returning (ok, combined output). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode == 0, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return False, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return False, f"{cmd[0]}: timed out after {timeout}s"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"{cmd[0]}: {e}"


def repo_root() -> Path:
    """Walk up to the directory holding pyproject.toml.

    Deliberately not a parents[N] index — a hardcoded depth breaks silently if
    this file moves.

    A deliberate copy of ``pipeline.config.find_repo_root``, and the only one
    left. This script's most useful check is whether ``pipeline`` imports at
    all (see check_project_installed), so it cannot import from the package to
    find the root: on a broken install it would die with an ImportError instead
    of reporting the problem it exists to report. The tool that verifies the
    install cannot depend on the install.
    """
    for d in Path(__file__).resolve().parents:
        if (d / "pyproject.toml").is_file():
            return d
    raise SystemExit("Cannot locate the repo root (no pyproject.toml above this file)")


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_python(r: Report) -> None:
    v = sys.version_info
    if v < (3, 11):
        r.fail(f"Python {v.major}.{v.minor} is too old",
               "pyproject.toml requires >=3.11. Run via `uv run python ...`")
    else:
        r.ok(f"Python {v.major}.{v.minor}.{v.micro}", sys.executable)


def check_project_installed(r: Report, root: Path) -> bool:
    """The single most useful check: is the project importable?"""
    try:
        import pipeline  # noqa: F401
        import cc_stages  # noqa: F401
    except ImportError as e:
        r.fail(f"Project not importable: {e}",
               "Run `uv sync` (installs pipeline + cc_stages into .venv)")
        return False
    import pipeline as _p
    installed_from = Path(_p.__file__).resolve().parents[1]
    if installed_from != (root / "src").resolve():
        r.warn("`pipeline` resolves outside this repo", f"from {installed_from}")
    else:
        r.ok("Project importable (pipeline, cc_stages)")
    return True


def check_uv(r: Report) -> None:
    if shutil.which("uv") is None:
        r.warn("uv not on PATH",
               "Optional, but the documented workflow. winget install --id=astral-sh.uv -e")
        return
    ok, out = _run(["uv", "--version"])
    r.ok(out.strip() if ok else "uv found")


def check_docker(r: Report) -> bool:
    if shutil.which("docker") is None:
        r.fail("Docker not found on PATH", "Required to run transcription or scene detection")
        return False
    ok, out = _run(["docker", "--version"])
    if not ok:
        r.fail("`docker --version` failed", out.strip()[:200])
        return False
    r.ok(out.strip())

    ok, out = _run(["docker", "compose", "version"])
    if not ok:
        r.fail("Docker Compose v2 not available", "`docker compose version` failed")
        return False
    r.ok(out.strip().splitlines()[0])

    ok, out = _run(["docker", "info"], timeout=30)
    if not ok:
        r.fail("Docker daemon not responding", "Is Docker Desktop running?")
        return False
    r.ok("Docker daemon responding")
    return True


def check_compose_files(r: Report, root: Path, docker_ok: bool) -> None:
    """Validate the compose files the code will actually invoke."""
    try:
        from pipeline.common.docker import COMPOSE_FILES
    except ImportError:
        r.fail("Cannot import COMPOSE_FILES", "Project not installed?")
        return

    for service, rel in sorted(COMPOSE_FILES.items()):
        path = root / rel
        if not path.is_file():
            r.fail(f"compose file missing for '{service}'", str(path))
            continue
        if not docker_ok:
            r.warn(f"'{service}': {rel} present, not validated (no Docker)")
            continue
        ok, out = _run(["docker", "compose", "-f", str(path), "config"], timeout=60)
        if ok:
            r.ok(f"'{service}' compose config valid", rel)
        else:
            first = next((l for l in out.splitlines() if l.strip()), "")
            r.fail(f"'{service}' compose config INVALID", first[:200])


def check_images(r: Report, docker_ok: bool) -> None:
    if not docker_ok:
        return
    ok, out = _run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
    if not ok:
        r.warn("Could not list Docker images")
        return
    have = out.split()
    for image in ("whisperx:local", "pyscenedetect:local"):
        if image in have:
            r.ok(f"image {image} built")
        else:
            r.warn(f"image {image} not built yet",
                   f"docker compose -f compose/{image.split(':')[0].replace('whisperx','whisperx').replace('pyscenedetect','pyscenedetect')}/docker-compose.yml build")


def check_env_file(r: Report, root: Path) -> None:
    """.env is the ONLY route HF_TOKEN takes into the container."""
    env = root / ".env"
    if not env.is_file():
        r.warn(".env not found",
               "Copy .env.example to .env. Required only when diarization is on.")
        return

    token = ""
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("HF_TOKEN="):
            token = line.split("=", 1)[1].strip()

    try:
        from pipeline.config import get_config
        diarize = bool(get_config().get("whisper", "diarize"))
    except Exception:
        diarize = None

    if token:
        r.ok(".env present with HF_TOKEN", f"{token[:6]}… ({len(token)} chars)")
    elif diarize:
        r.fail(".env has no HF_TOKEN but whisper.diarize is true",
               "Diarization will fail. Token: https://huggingface.co/settings/tokens")
    else:
        r.warn(".env present but HF_TOKEN is empty",
               "Fine while whisper.diarize is false")


def check_speaker_config(r: Report, cfg, src: Path) -> None:
    """speaker_config.json — user data, resolved relative to source_dir.

    Worth checking because every failure mode here is SILENT. When the file is
    missing, map_speakers logs a warning and skips, so the run completes with raw
    SPEAKER_XX labels in the storyboard. When it is present but the mapping key
    for this workflow is empty, the same thing happens with no warning at all.
    """
    # Resolved against the EFFECTIVE source dir, so --source-dir applies here
    # too. cfg.speaker_config_file always uses the config's own source_dir.
    try:
        name = cfg.get("speakers", "config_file", "speaker_config.json")
    except Exception as e:
        r.warn(f"speaker_config path unresolvable: {e}")
        return
    path = Path(src) / name

    if not path.is_file():
        r.warn(f"speaker_config.json not found: {path}",
               "Speaker mapping is skipped; transcripts keep raw SPEAKER_XX labels")
        return

    import json
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        r.fail(f"speaker_config.json is not valid JSON: {e}", str(path))
        return

    # Workflow A (Audacity) keys off filenames; Workflow B (diarization) off
    # SPEAKER_XX. A file carrying neither maps nothing.
    filename_map = {k: v for k, v in (data.get("filename_mapping") or {}).items()
                    if k != "comment"}
    global_map = {k: v for k, v in (data.get("global_mapping") or {}).items()
                  if k != "comment"}
    session_map = data.get("session_mappings") or {}
    fillers = data.get("filler_phrases") or []

    parts = []
    if filename_map:
        parts.append(f"{len(filename_map)} filename mapping(s)")
    if global_map:
        parts.append(f"{len(global_map)} global SPEAKER_XX mapping(s)")
    if session_map:
        parts.append(f"{len(session_map)} per-session override(s)")
    if fillers:
        parts.append(f"{len(fillers)} filler phrase(s)")

    if not (filename_map or global_map or session_map):
        r.warn("speaker_config.json has no speaker mappings",
               "Needs filename_mapping (Audacity) or global_mapping/session_mappings "
               "(diarization). Without one, speakers stay unmapped.")
    else:
        r.ok("speaker_config.json valid", ", ".join(parts))


def _check_hint_file(r: Report, cfg, src: Path, key: str, default: str,
                     flag: str, note: str) -> None:
    """Report on one of the two recognition-hint files.

    Both are silently optional: a missing file just means the flag is omitted,
    which looks identical to a misspelled filename, or to the two files being
    crossed. Naming the file AND the flag it feeds turns a silent no-op into an
    informed choice.
    """
    try:
        name = cfg.get("whisper", key, default)
    except Exception:
        return
    if not name:
        r.warn(f"whisper.{key} unset", f"{flag} will not be passed")
        return
    path = Path(src) / name
    if not path.is_file():
        r.warn(f"{name} not found ({flag})",
               f"{flag} omitted. {note}")
        return
    text = " ".join(path.read_text(encoding="utf-8", errors="replace").split())
    if not text:
        r.warn(f"{name} is empty ({flag})", f"{flag} omitted")
        return
    preview = text[:56] + ("…" if len(text) > 56 else "")
    r.ok(f"{name} -> {flag} ({len(text.split())} word(s))", preview)


def check_hints(r: Report, cfg, src: Path) -> None:
    _check_hint_file(
        r, cfg, src, "hotwords_file", "whisperx_hotwords.txt", "--hotwords",
        "Applies for the whole run; proper nouns and jargon belong here.")
    _check_hint_file(
        r, cfg, src, "initial_prompt_file", "whisperx_initial_prompt.txt",
        "--initial_prompt",
        "Seeds the first window only; spelling and style conventions.")


def check_config(r: Report, root: Path, source_override: str = None) -> None:
    try:
        from pipeline.config import Config, get_config
    except ImportError:
        return

    expected = root / "config" / "config.yaml"
    if not expected.is_file() and not os.getenv("WHISPERX_CONFIG"):
        r.fail(f"config.yaml not found at {expected}",
               "Copy config/config_sample.yaml to config/config.yaml")
        return
    try:
        cfg = get_config()
    except Exception as e:
        r.fail(f"config.yaml failed to load: {e}")
        return

    r.ok(f"config loaded", str(Config._resolve_config_path()))

    src = Path(source_override) if source_override else cfg.source_dir
    if src is None:
        r.warn("orchestration.source_dir not set", "Pass --source-dir when running")
    elif src.is_dir():
        r.ok(f"source_dir reachable", str(src))
        sessions = cfg.session_dirs_config
        if not sessions:
            r.warn("no session_dirs configured", "Pass --session-dirs when running")
        else:
            missing = [d for d in sessions if not (src / d).is_dir()]
            if missing:
                r.warn(f"{len(missing)} of {len(sessions)} configured session(s) missing",
                       ", ".join(str(m) for m in missing[:4]))
            else:
                r.ok(f"all {len(sessions)} configured session(s) present")
    else:
        r.fail(f"source_dir not reachable: {src}",
               "Fix orchestration.source_dir, or pass --source-dir")

    base = cfg.output_base_dir
    r.ok(f"output goes {'to ' + str(base) if base else 'beside the input (cc_output/)'}")

    # Both of these live NEXT TO THE RECORDINGS, not in the repo, and both are
    # skipped silently when absent — so a pre-flight check is the only place a
    # typo'd filename shows up before the output is already wrong.
    if src is not None:
        check_speaker_config(r, cfg, src)
        check_hints(r, cfg, src)


def check_layout(r: Report, root: Path) -> None:
    """Files the pipeline actually needs, at the paths it actually uses."""
    required = [
        "scripts/orchestrate.py",
        "scripts/cc_stages/transcribe.py",
        "docker/whisperx/entrypoint.sh",
        "docker/pyscenedetect/detect_scenes.sh",
        "docker/pyscenedetect/detect_scenes_multi.sh",
    ]
    missing = [p for p in required if not (root / p).is_file()]
    if missing:
        for m in missing:
            r.fail(f"missing: {m}")
    else:
        r.ok(f"all {len(required)} required files present")

    # The container executes these directly off the bind-mounted working tree,
    # so a CRLF shebang fails with `/bin/bash^M: bad interpreter`.
    crlf = [p for p in required if p.endswith(".sh")
            and (root / p).is_file() and b"\r\n" in (root / p).read_bytes()[:4096]]
    if crlf:
        for p in crlf:
            r.fail(f"CRLF line endings in {p}",
                   "Will fail in-container. Check .gitattributes, then re-checkout.")
    else:
        r.ok("container shell scripts are LF")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", default=None,
                    help="Check this source directory instead of config's")
    ap.add_argument("--quiet", action="store_true", help="Only show problems")
    args = ap.parse_args()

    root = repo_root()
    r = Report(quiet=args.quiet)

    print(_c("0;34", "=== Setup verification ===") + f"  {root}\n")

    print("Python & project")
    check_python(r)
    check_uv(r)
    installed = check_project_installed(r, root)

    print("\nLayout")
    check_layout(r, root)

    print("\nDocker")
    docker_ok = check_docker(r)
    if installed:
        check_compose_files(r, root, docker_ok)
        check_images(r, docker_ok)

    if installed:
        print("\nConfiguration")
        check_config(r, root, args.source_dir)
        check_env_file(r, root)

    print()
    if r.failures:
        print(_c("0;31", f"{len(r.failures)} problem(s) must be fixed:"))
        for f in r.failures:
            print(f"  - {f}")
        return 1

    if r.warnings:
        print(_c("1;33", f"Usable, with {len(r.warnings)} warning(s)."))
    else:
        print(_c("0;32", "All checks passed."))

    print("\nNext steps:")
    print("  1. Dry run:   uv run python scripts/orchestrate.py --dry-run")
    print('  2. One session: uv run python scripts/orchestrate.py --session-dirs "Week 14"')
    print("  3. See docs/ORCHESTRATION.md for the full command surface")
    return 0


if __name__ == "__main__":
    sys.exit(main())
