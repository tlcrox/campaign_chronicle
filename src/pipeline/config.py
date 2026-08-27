#!/usr/bin/env python3
"""
Unified configuration loader for WhisperX pipeline.
Loads config.yaml with environment variable overrides.
Resolves paths relative to source directory and whisperx root.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# pipeline.common.scenes is a leaf module (stdlib imports only) and must stay
# that way: it may not import this one, or config loading becomes a cycle.
from pipeline.common.scenes import normalize_image_format

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Install with: pip install pyyaml")
    sys.exit(1)

logger = logging.getLogger(__name__)


# ============================================================================
# The complete default configuration.
#
# config.yaml is the master source for everything except HF_TOKEN (which is a
# secret and lives in .env). A file is merged OVER this at load, so the loaded
# config always has every section and key the codebase reads — a config may say
# as little as it likes, and nothing downstream has to guess.
#
# This is the ONLY place a default value is written. Properties below read
# through get() without restating one; two copies of a default is how they come
# to disagree.
#
# Keys whose default is None are "unset, and that is meaningful" — an absent
# source_dir means "the config file's own directory", an absent roi means full
# frame. They are listed anyway so the shape of a config is discoverable here.
#
# whisper.clean carries only `enabled`: the three thresholds belong to
# pipeline.transcribe.remove_hallucination.DEFAULT_THRESHOLDS, which merges the
# config over them. Restating them here would be a second copy.
# ============================================================================
DEFAULTS = {
    "whisper": {
        "model": "large-v3",
        "language": "en",
        "device": "cuda",
        "compute_type": "float16",
        "batch_size": 16,
        "output_format": "json",
        "diarize": False,
        "hf_token": None,
        "hotwords_file": "whisperx_hotwords.txt",
        "initial_prompt_file": "whisperx_initial_prompt.txt",
        "clean": {"enabled": False},
    },
    "scenes": {
        "threshold": 5.0,
        "min_length": "1s",
        "num_images": 1,
        "image_format": "jpg",
        "roi": None,
        "roi_file": None,
        "manual_source": "screens",
        "manual_csv_name": "Scenes.csv",
    },
    "orchestration": {
        "source_dir": None,
        "session_dirs": None,
        "parallel_workers": 1,
    },
    "output": {
        "base_dir": None,
    },
    "merge": {
        "auto_merge": True,
        "csv": {
            "scene_number_column": "Scene Number",
            "start_time_column": "Start Time (seconds)",
            "end_time_column": "End Time (seconds)",
            "video_column": "Video",
        },
        "images": {"width": 6.0},
        "document": {"title": "Transcript"},
    },
    "speakers": {
        "config_file": "speaker_config.json",
        "pyannote": {"metrics_enabled": False, "api_key": None},
    },
}

# Values the container tooling will reject, checked here instead so a run fails
# at load with a sentence rather than an hour in with a stack trace.
WHISPER_DEVICES = ("cuda", "cpu")
WHISPER_COMPUTE_TYPES = ("float16", "float32", "bfloat16", "int8",
                         "int8_float16", "int8_float32")
# CTranslate2 has no efficient float16 path on CPU and refuses at model load.
CPU_COMPUTE_TYPES = ("int8", "float32")
WHISPER_OUTPUT_FORMATS = ("json", "txt", "srt", "vtt", "tsv", "aud", "all")


def _deep_merge(defaults: dict, loaded, path: str = ""):
    """Merge ``loaded`` over ``defaults``, recursing into nested sections.

    A section the file omits, or writes as an empty YAML mapping (which
    safe_load returns as None), takes the defaults wholesale. A file value of
    the wrong shape — a scalar where a section belongs — raises rather than
    being merged into nonsense.
    """
    if loaded is None:
        return {k: (dict(v) if isinstance(v, dict) else v)
                for k, v in defaults.items()}
    if not isinstance(loaded, dict):
        where = path or "the config"
        raise ValueError(
            f"{where} should be a section (a mapping), not {type(loaded).__name__}"
        )

    merged = {}
    for key, default_value in defaults.items():
        here = f"{path}.{key}" if path else key
        if key not in loaded:
            merged[key] = dict(default_value) if isinstance(default_value, dict) else default_value
        elif isinstance(default_value, dict):
            merged[key] = _deep_merge(default_value, loaded[key], here)
        else:
            merged[key] = loaded[key]

    # Keys the file adds that DEFAULTS does not know about are kept as-is:
    # remove_hallucination's thresholds arrive this way, and an unknown key is
    # not this loader's business to reject.
    for key, value in loaded.items():
        if key not in merged:
            merged[key] = value
    return merged



def _read_text(path) -> str:
    """Read a text file as UTF-8, falling back to cp1252.

    These files are hand-edited on Windows, where Notepad and similar still save
    cp1252 by default. An em-dash or smart quote in such a file is not valid
    UTF-8, and the resulting UnicodeDecodeError is a ValueError — so an
    `except OSError` around the read will not catch it and the run dies.
    """
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning(
            f"{path} is not valid UTF-8; falling back to cp1252. "
            f"Re-save it as UTF-8 to silence this.")
        return data.decode("cp1252", errors="replace")


def find_repo_root(start=None, required: bool = False):
    """The directory holding ``pyproject.toml``, walking up from ``start``.

    ``start`` defaults to this module; callers elsewhere should pass their own
    ``__file__``, so they ask about the repo above themselves rather than above
    the installed package.

    Returns None when there is none above it, unless ``required``. Absence is
    not always a failure: inside the containers only ``src/`` is mounted (at
    /usr/local/bin/pylib), so there is no repo above this file — and nothing
    there runs ``docker compose``, which is the only thing the answer is for. A
    caller that genuinely cannot proceed without one passes ``required=True``
    and gets an exception naming where it looked.

    A walk rather than ``parents[N]``: a hardcoded depth is silently wrong the
    day the file moves, and this one has already moved once (scripts/ ->
    src/pipeline/).
    """
    start = Path(start).resolve() if start else Path(__file__).resolve()
    for directory in start.parents:
        if (directory / "pyproject.toml").is_file():
            return directory
    if required:
        raise RuntimeError(f"repo root not found: no pyproject.toml above {start}")
    return None


class Config:
    """Configuration manager for WhisperX pipeline."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config from YAML file.

        Args:
            config_path: Path to config.yaml. If None, it is resolved (in order):
                1. the WHISPERX_CONFIG environment variable,
                2. <repo root>/config/config.yaml.
            There is no upward search and no cwd fallback: the repo layout is
            fixed. Inside the container the layout does not exist, so
            docker-compose sets WHISPERX_CONFIG to the mounted config path.
        """
        if config_path is None:
            config_path = self._resolve_config_path()
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        # Complete the document before anything reads it. safe_load returns None
        # for an empty file and for a section header with nothing under it, so
        # this is also what keeps the env overrides below from indexing None.
        try:
            self._config = _deep_merge(DEFAULTS, loaded)
        except ValueError as e:
            raise ValueError(f"{config_path}: {e}") from None

        # Two distinct roots — do not conflate them:
        #  - _repo_root: the CODE directory (holds pyproject.toml and compose/,
        #    the cwd for `docker compose`). Found by walking up from this module,
        #    so it is correct no matter where the config file lives — lets an
        #    override config sit with the source data without breaking Docker.
        #    None inside the containers, where there is no repo above src/.
        #  - _config_dir: the config file's own directory, used to resolve paths
        #    declared *in* the config (e.g. a relative source_dir).
        self._repo_root = find_repo_root()
        self._config_dir = config_path.parent
        # Filled by _apply_env_overrides, reported by _validate: a value is
        # verified wherever it came from, and an override that cannot be parsed
        # is a misconfiguration exactly like a bad one in the file.
        self._env_problems = []
        self._apply_env_overrides()
        self._validate(config_path)

    def _validate(self, config_path: Path) -> None:
        """Verify the completed config can actually run, at load, naming the file.

        Every problem is collected and reported together: fixing a config one
        error per run is its own kind of misery. Checks are cheap and purely
        about the config's own values — this runs on every entry point,
        including inside the containers — so nothing here touches the disk or
        the network. Environmental checks (does source_dir exist, is Docker up)
        belong to scripts/verify_setup.py.
        """
        problems = list(self._env_problems)

        def _positive_number(section, key, kind=float, minimum=None):
            value = self.get(section, key)
            try:
                number = kind(value)
            except (TypeError, ValueError):
                problems.append(
                    f"{section}.{key} must be a number, got {value!r}")
                return
            floor = minimum if minimum is not None else 0
            if number <= 0 or (minimum is not None and number < minimum):
                limit = f"at least {minimum}" if minimum is not None else "greater than 0"
                problems.append(f"{section}.{key} must be {limit}, got {value!r}")

        # scenes.image_format — the one knob with its own explanatory message.
        try:
            normalize_image_format(self.get("scenes", "image_format"))
        except ValueError as e:
            problems.append(str(e))

        device = str(self.get("whisper", "device") or "").lower()
        compute_type = str(self.get("whisper", "compute_type") or "").lower()
        if device not in WHISPER_DEVICES:
            problems.append(
                f"whisper.device must be one of: {', '.join(WHISPER_DEVICES)}; "
                f"got {self.get('whisper', 'device')!r}")
        if compute_type not in WHISPER_COMPUTE_TYPES:
            problems.append(
                f"whisper.compute_type must be one of: "
                f"{', '.join(WHISPER_COMPUTE_TYPES)}; "
                f"got {self.get('whisper', 'compute_type')!r}")
        elif device == "cpu" and compute_type not in CPU_COMPUTE_TYPES:
            problems.append(
                f"whisper.compute_type {compute_type!r} has no CPU implementation "
                f"in CTranslate2 and is rejected at model load. With device: cpu "
                f"use one of: {', '.join(CPU_COMPUTE_TYPES)}.")

        output_format = str(self.get("whisper", "output_format") or "").lower()
        if output_format not in WHISPER_OUTPUT_FORMATS:
            problems.append(
                f"whisper.output_format must be one of: "
                f"{', '.join(WHISPER_OUTPUT_FORMATS)}; got {output_format!r}")

        _positive_number("whisper", "batch_size", int, minimum=1)
        _positive_number("scenes", "threshold", float)
        _positive_number("scenes", "num_images", int, minimum=1)
        _positive_number("orchestration", "parallel_workers", int, minimum=1)

        if not str(self.get("scenes", "min_length") or "").strip():
            problems.append(
                "scenes.min_length is empty; use a PySceneDetect duration "
                "such as '1s', a frame count, or 'HH:MM:SS'")

        roi = self.get("scenes", "roi")
        if roi:
            parts = str(roi).split()
            if len(parts) != 4 or not all(p.lstrip("-").isdigit() for p in parts):
                problems.append(
                    f"scenes.roi must be four integers 'x y width height', got {roi!r}")

        width = (self.get("merge", "images") or {}).get("width")
        try:
            if float(width) <= 0:
                problems.append(f"merge.images.width must be greater than 0, got {width!r}")
        except (TypeError, ValueError):
            problems.append(f"merge.images.width must be a number, got {width!r}")

        if problems:
            listed = "\n".join(f"  - {p}" for p in problems)
            raise ValueError(
                f"{config_path} cannot run as configured:\n{listed}")

        # Not fatal: HF_TOKEN is the one setting that does NOT live here. It
        # comes from .env via compose --env-file, which this process cannot see.
        # Only meaningful where there IS a repo: inside the containers the
        # token arrives through compose, and a warning about a .env that could
        # never be there is noise.
        if (self.get("whisper", "diarize")
                and not os.getenv("HF_TOKEN")
                and self._repo_root is not None
                and not (self._repo_root / ".env").is_file()):
            logger.warning(
                "whisper.diarize is on, but HF_TOKEN is not in the environment "
                "and there is no .env at %s. It is the one setting that does not "
                "live in this file: compose passes it from .env, so diarization "
                "will fail with an unset token.", self._repo_root)

    @staticmethod
    def _resolve_config_path() -> Path:
        """Locate config.yaml: WHISPERX_CONFIG env, else <repo>/config/config.yaml.

        No searching for the *config*: it has exactly one home under the repo
        root, and a missing one fails immediately naming the exact path it
        wanted. The repo root itself is found by walking up to pyproject.toml.

        WHISPERX_CONFIG keeps top precedence because three callers depend on it:
        the containers (which set it to /usr/local/bin/config.yaml, where the repo
        layout does not exist), tests/run_tests.py (fixture config), and
        --config / resolve_tool_config() (per-run overrides).
        """
        env_path = os.getenv("WHISPERX_CONFIG")
        if env_path:
            return Path(env_path)

        repo_root = find_repo_root()
        if repo_root is None:
            raise FileNotFoundError(
                "Cannot locate config.yaml: no pyproject.toml above "
                f"{Path(__file__).resolve()}, so there is no repo root to look "
                "under. Set WHISPERX_CONFIG to the config file's path.")
        return repo_root / "config" / "config.yaml"

    def _env_number(self, section: str, key: str, raw: str, kind) -> None:
        """Apply a numeric env override, recording the value if it will not parse.

        The alternative — and what this replaced — is `except ValueError: pass`,
        which leaves WHISPER_BATCH_SIZE=sixteen silently not applied and the run
        continuing on a default the caller did not ask for.
        """
        try:
            self._config[section][key] = kind(raw)
        except (TypeError, ValueError):
            self._env_problems.append(
                f"{section}.{key} was overridden with {raw!r}, which is not "
                f"{'an integer' if kind is int else 'a number'}")

    def _apply_env_overrides(self):
        """Apply environment variable overrides to loaded config.

        Every section DEFAULTS declares is present by now, so these writes cannot
        fail on a section the config file happened to omit.
        """
        # Whisper overrides
        if os.getenv("WHISPER_MODEL"):
            self._config["whisper"]["model"] = os.getenv("WHISPER_MODEL")
        if os.getenv("WHISPER_LANGUAGE"):
            self._config["whisper"]["language"] = os.getenv("WHISPER_LANGUAGE")
        if os.getenv("WHISPER_COMPUTE_TYPE"):
            self._config["whisper"]["compute_type"] = os.getenv("WHISPER_COMPUTE_TYPE")
        if os.getenv("WHISPER_DEVICE"):
            self._config["whisper"]["device"] = os.getenv("WHISPER_DEVICE")
        if os.getenv("WHISPER_BATCH_SIZE"):
            self._env_number("whisper", "batch_size",
                             os.getenv("WHISPER_BATCH_SIZE"), int)
        if os.getenv("WHISPER_OUTPUT_FORMAT"):
            self._config["whisper"]["output_format"] = os.getenv(
                "WHISPER_OUTPUT_FORMAT"
            )
        if os.getenv("WHISPER_DIARIZE"):
            diarize_str = os.getenv("WHISPER_DIARIZE").lower()
            self._config["whisper"]["diarize"] = diarize_str in (
                "true",
                "1",
                "yes",
                "on",
            )
        if os.getenv("HF_TOKEN"):
            self._config["whisper"]["hf_token"] = os.getenv("HF_TOKEN")

        # Scene detection overrides
        if os.getenv("SCENE_THRESHOLD"):
            self._env_number("scenes", "threshold",
                             os.getenv("SCENE_THRESHOLD"), float)
        if os.getenv("SCENE_MIN_LEN"):
            self._config["scenes"]["min_length"] = os.getenv("SCENE_MIN_LEN")
        if os.getenv("SCENE_NUM_IMAGES"):
            self._env_number("scenes", "num_images",
                             os.getenv("SCENE_NUM_IMAGES"), int)
        if os.getenv("SCENE_IMAGE_FORMAT"):
            self._config["scenes"]["image_format"] = os.getenv("SCENE_IMAGE_FORMAT")
        if os.getenv("SCENE_ROI"):
            self._config["scenes"]["roi"] = os.getenv("SCENE_ROI")
        if os.getenv("ROI_FILE"):
            self._config["scenes"]["roi_file"] = os.getenv("ROI_FILE")
            
        # Orchestration overrides
        if os.getenv("DEFAULT_SOURCE_DIR"):
            self._config["orchestration"]["source_dir"] = os.getenv(
                "DEFAULT_SOURCE_DIR"
            )

        # Speaker overrides
        if os.getenv("PYANNOTE_METRICS_ENABLED"):
            metrics_str = os.getenv("PYANNOTE_METRICS_ENABLED").lower()
            self._config["speakers"]["pyannote"]["metrics_enabled"] = metrics_str in (
                "true",
                "1",
                "yes",
                "on",
            )
        if os.getenv("PYANNOTE_API_KEY"):
            self._config["speakers"]["pyannote"]["api_key"] = os.getenv(
                "PYANNOTE_API_KEY"
            )

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Every section and key in DEFAULTS is present after load, so callers do
        not pass a default for those — DEFAULTS is where the value is written.
        The ``default`` argument is for keys DEFAULTS does not describe.

        Args:
            section: Top-level config section (whisper, scenes, orchestration, merge, speakers)
            key: Key within the section
            default: Value for a key DEFAULTS does not carry

        Returns:
            Configuration value or default
        """
        try:
            return self._config[section][key]
        except (KeyError, TypeError):
            return default

    # ========================================================================
    # WHISPER CONFIGURATION
    # ========================================================================

    @property
    def whisper_model(self) -> str:
        """Whisper model name."""
        return self.get("whisper", "model")

    @property
    def whisper_language(self) -> str:
        """Whisper language (BCP-47 code)."""
        return self.get("whisper", "language")

    @property
    def whisper_compute_type(self) -> str:
        """Whisper compute type."""
        return self.get("whisper", "compute_type")

    @property
    def whisper_device(self) -> str:
        """Torch device WhisperX runs on ("cuda" or "cpu").

        Defaults to "cuda", which is WhisperX's own default, so an unset key
        behaves exactly as before this was plumbed through.

        NOTE the coupling with ``compute_type``: CTranslate2 (via faster-whisper)
        has no efficient float16 path on CPU, so "cpu" must be paired with
        "int8" or "float32". Pairing it with the "float16" default fails inside
        the container rather than here.
        """
        return self.get("whisper", "device")

    @property
    def whisper_batch_size(self) -> int:
        """Whisper batch size."""
        return self.get("whisper", "batch_size")

    @property
    def whisper_output_format(self) -> str:
        """Whisper output format."""
        return self.get("whisper", "output_format")

    @property
    def whisper_diarize(self) -> bool:
        """Enable speaker diarization."""
        return self.get("whisper", "diarize")

    # ------------------------------------------------------------------
    # Recognition hints: TWO distinct WhisperX features, two separate files.
    #
    #   --hotwords        biases recognition toward specific terms for the WHOLE
    #                     run. Proper nouns and jargon belong here ("Mysteria",
    #                     not "Mysterio"/"mystery").
    #   --initial_prompt  seeds the decoder's FIRST WINDOW only, influencing
    #                     style, spelling and punctuation conventions ("DEX", not
    #                     "decks"). Its effect fades across a long session, so
    #                     terms that must hold throughout belong in hotwords.
    #
    # Both filenames are configurable and resolve against source_dir, so each
    # recording source carries its own vocabulary. A missing file means the flag
    # is simply omitted.
    # ------------------------------------------------------------------

    def _source_relative_text(self, section: str, key: str, default: str) -> str:
        """Read a source-dir-relative hint file down to one shell-safe line.

        Lines whose first non-space character is ``#`` are dropped. These files
        are hand-maintained vocabulary lists, and the shipped samples are
        organised with comment headers ("# --- Names / proper nouns ---"); without
        stripping them, that commentary would be handed to WhisperX as hint terms
        and dilute the real ones.

        Remaining text is collapsed to a single line. Quotes are NOT stripped:
        run_docker_command passes env values as their own argv entries with no
        shell involved, so a quoted term reaches WhisperX as written.
        Missing/unreadable/unset -> "".
        """
        try:
            filename = self.get(section, key, default)
        except OSError:
            return ""
        if not filename:
            return ""
        try:
            path = self.source_dir / filename
        except ValueError:
            # source_dir not configured; nothing to read.
            return ""
        try:
            text = _read_text(path)
        except OSError:
            return ""
        lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        return " ".join(" ".join(lines).split())

    @property
    def whisper_hotwords_file(self) -> Path:
        """Path to the hotwords file, relative to source_dir (--hotwords)."""
        try:
            filename = self.get("whisper", "hotwords_file")
        except OSError:
            return ""
        return self.source_dir / filename

    @property
    def whisper_hotwords(self) -> str:
        """Hotwords string read from ``whisper_hotwords_file``."""
        return self._source_relative_text(
            "whisper", "hotwords_file", "whisperx_hotwords.txt")

    @property
    def whisper_initial_prompt_file(self) -> Path:
        """Path to the initial-prompt file, relative to source_dir."""
        try:
            filename = self.get("whisper", "initial_prompt_file")
        except OSError:
            return ""
        return self.source_dir / filename

    @property
    def whisper_initial_prompt(self) -> str:
        """Initial-prompt text read from ``whisper_initial_prompt_file``."""
        return self._source_relative_text(
            "whisper", "initial_prompt_file", "whisperx_initial_prompt.txt")

    # ========================================================================
    # SCENE DETECTION CONFIGURATION
    # ========================================================================

    @property
    def scene_threshold(self) -> float:
        """Scene detection sensitivity threshold."""
        return self.get("scenes", "threshold")

    @property
    def scene_min_length(self) -> str:
        """Minimum scene length."""
        return self.get("scenes", "min_length")

    @property
    def scene_num_images(self) -> int:
        """Number of images per scene."""
        return self.get("scenes", "num_images")

    @property
    def scene_image_format(self) -> str:
        """Image format for scene detection: "jpg" or "png".

        Normalised ("jpeg" -> "jpg") and already validated at load, so callers
        can pass it straight to the container without re-checking it.
        """
        return normalize_image_format(self.get("scenes", "image_format"))

    @property
    def scene_roi(self) -> Optional[str]:
        """Region of Interest for video cropping (x1 y1 x2 y2)."""
        return self.get("scenes", "roi")

    @property
    def scene_roi_file(self) -> Optional[str]:
        """File that defines Region of Interest for video cropping."""
        return self.get("scenes", "roi_file")

    @property
    def scene_manual_source(self) -> Optional[str]:
        """Directory for manually created screen snapshots."""
        return self.get("scenes", "manual_source")

    @property
    def scene_manual_csv_name(self) -> Optional[str]:
        """Filename for manually captured screen snapshots CSV file."""
        return self.get("scenes", "manual_csv_name")

    # ========================================================================
    # ORCHESTRATION CONFIGURATION
    # ========================================================================

    @property
    def repo_root(self) -> Path:
        """The CODE directory: cwd for ``docker compose``, and where ``.env`` lives.

        Every stage that runs a container needs this — it is what makes the
        ``-f compose/…/docker-compose.yml`` relative path resolve, and what
        ``compose_run`` uses to find the ``.env`` that carries ``HF_TOKEN``. That
        makes it part of the contract, so it has a public name rather than being
        read off the private attribute from outside the class.

        **Not** ``_config_dir``: this is found by walking up from this module, so
        an override config can sit beside the source data without breaking
        Docker, while paths declared *inside* a config resolve against that file
        instead.

        Raises RuntimeError when there is no repo above this file — the case
        inside the containers, where only ``src/`` is mounted. Nothing there runs
        ``docker compose``, so reaching for it there is the bug, and saying so
        beats handing back a directory that merely exists.
        """
        if self._repo_root is None:
            raise RuntimeError(
                "No repo root: no pyproject.toml above "
                f"{Path(__file__).resolve()}. Expected inside the containers, "
                "which mount only src/ and never run `docker compose`.")
        return self._repo_root

    @property
    def source_dir(self) -> Path:
        """Source directory with session folders.

        When ``orchestration.source_dir`` (or the ``DEFAULT_SOURCE_DIR`` env var)
        is set it is honored — absolute, or resolved against the config file's own
        directory. When it is unset, it defaults to the config file's directory:
        config.yaml is expected to live at the source root, alongside
        speaker_config.json, so a source folder is self-describing and portable
        (no absolute path baked into the file).
        """
        source = self.get("orchestration", "source_dir") or os.getenv("DEFAULT_SOURCE_DIR")
        if not source:
            return self._config_dir
        return self._resolve_path(source)

    @property
    def session_dirs_config(self) -> list[Path]:
        """The session directories exactly as configured — no resolution, no I/O.

        Callers that may override the base directory (orchestrate, via
        ``--source-dir``) need the raw values so they can resolve against the
        *effective* source dir rather than the one this config happens to name.
        Accepts a single directory or a list. Returns [] when unset.
        """
        dirs_config = self.get("orchestration", "session_dirs")
        if not dirs_config:
            return []
        if isinstance(dirs_config, str):
            dirs_config = [dirs_config]
        return [Path(d) for d in dirs_config]

    @property
    def session_dirs(self) -> list[Path]:
        """Configured session directories, resolved against ``source_dir``.

        To process a whole hierarchy, iterate it in a wrapper and call
        orchestrate once per folder (see README "Automating the processing of a
        whole hierarchy of recordings").

        Relative entries are resolved against ``source_dir``; absolute entries
        are returned as-is. Returns [] when unset.

        This deliberately performs NO filesystem check. Reading configuration
        and validating the filesystem are separate jobs: existence is checked in
        exactly one place, ``pipeline.common.sessions.find_sessions``, which can
        report what it skipped. When this property also filtered, entries were
        silently dropped here — before argparse had even parsed ``--source-dir``
        — so overriding the source directory could not resurrect them.
        """
        return [
            d if d.is_absolute() else self.source_dir / d
            for d in self.session_dirs_config
        ]

    @property
    def parallel_workers(self) -> int:
        """Number of parallel processing workers."""
        return self.get("orchestration", "parallel_workers")

    @property
    def output_base_dir(self) -> Optional[Path]:
        """Root for generated output, or None to write beside the input.

        None (the default) keeps today's behaviour: ``<session>/cc_output/...``.
        Set ``output.base_dir`` to send everything to a separate tree instead,
        leaving the source untouched — useful when the source is read-only,
        shared, or simply shouldn't accumulate artifacts.

        ``CC_OUTPUT_BASE`` overrides the config file, which is how a single run
        can be redirected without editing config: run_tests.py uses it so its
        results land in tests/test_output/ while run_orchestrate_tests.py, which
        does not set it, keeps writing beside the source. Running both then
        leaves two independent result trees, and neither can satisfy the other's
        checks by accident.

        Relative paths resolve against the config file's directory, consistent
        with source_dir.
        """
        env = os.getenv("CC_OUTPUT_BASE")
        if env:
            # An env var is set by whoever is launching the run, so a relative
            # value means "relative to where I am" — NOT relative to the config
            # file, which the caller may not even know the location of.
            return Path(env).expanduser().resolve()
        base = self.get("output", "base_dir")
        if not base:
            return None
        # A value from the config file resolves against that file's directory,
        # consistent with source_dir.
        return self._resolve_path(base)

    def session_key(self, session_dir) -> Path:
        """This session's identity under an output base: its path relative to
        source_dir, falling back to the leaf name when it sits outside.

        Relative — NOT the leaf name — because two sources can each contain a
        ``Week 13``; keying on the leaf would merge them into one output folder.
        """
        p = Path(session_dir).resolve()
        try:
            return p.relative_to(Path(self.source_dir).resolve())
        except ValueError:
            return Path(p.name)

    # Mount directories, the per-session output subdir names
    # (SESSION_OUTPUT_SUBDIR / SCENES_OUTPUT_SUBDIR / COMBINED_OUTPUT_SUBDIR),
    # and the docker service names ("whisperx" / "scenes" / "scenes_multi") are
    # all hard-coded in pipeline.common.mounts and the tools to match
    # docker-compose.yml. They are an internal contract, deliberately NOT
    # configurable.

    # ========================================================================
    # MERGE / STORYBOARD CONFIGURATION
    # ========================================================================

    @property
    def scene_number_column(self) -> str:
        """Scene number column name in CSV."""
        return self.get("merge", "csv")["scene_number_column"]

    @property
    def start_time_column(self) -> str:
        """Start time column name in CSV."""
        return (
            self.get("merge", "csv")["start_time_column"]
        )

    @property
    def end_time_column(self) -> str:
        """End time column name in CSV."""
        return self.get("merge", "csv")["end_time_column"]

    @property
    def video_column(self) -> str:
        """Video-index column in the merged scene CSV (the second half of the
        (Video, Scene Number) key that pairs an image to its row)."""
        return self.get("merge", "csv")["video_column"]

    @property
    def image_width_inches(self) -> float:
        """Image width in inches for storyboard."""
        return self.get("merge", "images")["width"]

    @property
    def document_title(self) -> str:
        """Title for generated storyboard document."""
        return self.get("merge", "document")["title"]

    @property
    def auto_merge(self) -> bool:
        """Automatically merge and generate storyboard after orchestrate."""
        return self.get("merge", "auto_merge")

    # ========================================================================
    # SOURCE-RELATIVE PATHS (for speaker config, samples, etc.)
    # ========================================================================

    @property
    def speaker_config_file(self) -> Path:
        """Path to speaker_config.json relative to source directory."""
        filename = self.get("speakers", "config_file")
        return self.source_dir / filename

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _resolve_path(self, path: str) -> Path:
        """Resolve a config-declared path: absolute as-is, otherwise relative to
        the config file's own directory (``_config_dir``)."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self._config_dir / p

    #def to_dict(self) -> dict:
    #    """Return raw config dictionary."""
    #    return self._config

    def __repr__(self) -> str:
        """String representation."""
        return f"<Config source_dir={self.source_dir} {len(self.session_dirs)} session_dirs>"


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get current config instance, loading if necessary."""
    global _config
    if _config is None:
        _config = Config()
    return _config


# By convention a source root holds config.yaml and a session sits at most two
# directories under it ("Weeks/Week 13"), so the search stops there. An unbounded
# walk climbs out of the source tree entirely — config.yaml is a common filename,
# and the next one up might be a drive root's, governing a run it knows nothing
# about. Two levels covers every layout this project uses, with the flat case
# (config beside the session) needing only one.
CONFIG_MAX_PARENTS = 2


def find_config_upward(start) -> Optional[Path]:
    """Walk up from ``start`` (a file or directory) to the nearest config.yaml.

    A session's governing config is the config.yaml at (or above) the source root
    the session lives under, so the per-tool CLIs locate it by walking up from
    ``--session-dir``. Returns None when none is found within
    ``CONFIG_MAX_PARENTS`` directories.

    ``start`` is expected to be a session directory. Every caller passes one:
    a path inside the generated ``cc_output/`` tree sits deeper than the
    convention allows, and the fix for that is to pass the session, not to teach
    this function where output lives.
    """
    p = Path(start).resolve()
    if p.is_file():
        p = p.parent

    for d in [p, *p.parents][:CONFIG_MAX_PARENTS + 1]:
        candidate = d / "config.yaml"
        logger.info("Config: searching for config at %s ", candidate)
        if candidate.exists():
            return candidate
    return None


def resolve_tool_config(config_arg=None, start_path=None) -> None:
    """Point WHISPERX_CONFIG at the right config before the singleton is built.

    Precedence: an explicit ``--config`` wins; else, if WHISPERX_CONFIG is not
    already set, walk up from ``start_path`` (typically ``--session-dir``) to the
    nearest config.yaml. If nothing is found the environment is left untouched, so
    ``Config()`` falls back to its own resolution. Must be called in a tool's
    ``main()`` before the first ``get_config()``.
    """
    if config_arg:
        p = Path(config_arg)
        if not p.exists():
            raise FileNotFoundError(f"--config: file not found: {p}")
        os.environ["WHISPERX_CONFIG"] = str(p)
        logger.info("Config: %s (--config)", p)
        return
    if os.environ.get("WHISPERX_CONFIG"):
        logger.info("Config: %s (WHISPERX_CONFIG)", os.environ["WHISPERX_CONFIG"])
        return
    if start_path:
        found = find_config_upward(start_path)
        if found:
            os.environ["WHISPERX_CONFIG"] = str(found)
            logger.info("Config: %s (found above %s)", found, start_path)
        else:
            logger.info(
                "Config: none within %d directories above %s; using the repo default",
                CONFIG_MAX_PARENTS, start_path)


if __name__ == "__main__":
    # Quick test
    cfg = get_config()
    print("Configuration loaded successfully:")
    print(f"  Whisper Model: {cfg.whisper_model}")
    print(f"  Scene Threshold: {cfg.scene_threshold}")
    print(f"  Source Dir: {cfg.source_dir}")
    print(f"  Speaker Config: {cfg.speaker_config_file}")

