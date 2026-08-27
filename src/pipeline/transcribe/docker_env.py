#!/usr/bin/env python3
"""
docker_env.py — build the WHISPER_* environment for the `whisperx` container
from config.yaml.

This makes config.yaml the single source of truth for transcription: the host
tools pass these as `-e KEY=VALUE` flags to `docker compose run`, which override
whatever the compose `environment:` block would supply. `HF_TOKEN` is
intentionally NOT included here — it is a secret and flows from `.env` through
docker-compose (only needed when diarization is on).
"""

from __future__ import annotations

from typing import Dict


def whisper_env(config) -> Dict[str, str]:
    """Return the WHISPER_* env dict for the whisperx container, from config."""
    return {
        "WHISPER_MODEL": str(config.whisper_model),
        "WHISPER_LANGUAGE": str(config.whisper_language or ""),
        "WHISPER_COMPUTE_TYPE": str(config.whisper_compute_type),
        # Paired with compute_type: CPU has no efficient float16 path, so
        # device=cpu needs compute_type int8/float32. getattr, not attribute
        # access: duck-typed configs predating this key must still work, and the
        # default matches Config.whisper_device / whisperx's own.
        "WHISPER_DEVICE": str(getattr(config, "whisper_device", "cuda")),
        "WHISPER_BATCH_SIZE": str(config.whisper_batch_size),
        "WHISPER_OUTPUT_FORMAT": str(config.whisper_output_format),
        "WHISPER_DIARIZE": "true" if config.whisper_diarize else "false",
        # Two distinct recognition hints, each from its own source-relative
        # file; "" when absent, in which case the entrypoint skips that flag.
        #   HOTWORDS       -> --hotwords       (whole run; proper nouns, jargon)
        #   INITIAL_PROMPT -> --initial_prompt (first window; spelling/style)
        "WHISPER_HOTWORDS": str(config.whisper_hotwords),
        "WHISPER_INITIAL_PROMPT": str(getattr(config, "whisper_initial_prompt", "")),
    }
