#!/usr/bin/env python3
"""
clean_transcription.py — the "clean transcription" stage.

Runs two ordered passes over a session's per-source transcripts, in place,
BETWEEN map_speakers and the merge:

  A. confidence / hallucination — drop low-confidence segments using the
     ``whisper.clean`` thresholds (see pipeline.transcribe.remove_hallucination).
     OPT-IN via ``whisper.clean.enabled`` (default off): until deliberately
     turned on this pass is a no-op, so the golden output is unchanged.

  B. filler — drop filler-phrase segments (``filler_phrases`` from
     speaker_config.json). Filler removal belongs here, not in the merges: doing
     it once upstream leaves both merge paths to simply interleave.

Empty-text segments are left for the merge to drop (its existing
``if not text: continue``), so this stage only removes what the two passes name.

Importable API:
    clean_transcript(session_dir, config, dry_run=False) -> bool
"""

import json
import logging
from pathlib import Path

from pipeline.common.mounts import output_dir_for, SESSION_OUTPUT_SUBDIR
from pipeline.merge.combine_transcripts import is_filler, load_merge_speaker_data
from pipeline.transcribe.remove_hallucination import clean_segments, thresholds_from_config

logger = logging.getLogger(__name__)


def _confidence_enabled(config) -> bool:
    """The confidence pass runs only when ``whisper.clean.enabled`` is truthy.

    Defaults off so adding this stage is a no-op until the thresholds are
    deliberately turned on (config has no ``whisper.clean`` section today).
    """
    try:
        return bool((config.get("whisper", "clean", {}) or {}).get("enabled", False))
    except Exception:
        return False


def clean_transcript(session_dir: Path, config, dry_run: bool = False) -> bool:
    """Run the clean passes on ``<session>/transcriptions/*.json`` (excluding
    combined files), rewriting each in place. Returns True when transcripts were
    found to process.
    """
    session_dir = Path(session_dir)
    out_dir = output_dir_for(session_dir, SESSION_OUTPUT_SUBDIR, config)
    transcript_files = sorted(
        f for f in out_dir.glob("*.json") if "combined" not in f.name
    )
    if not transcript_files:
        logger.warning(f"  ⊘ No transcript files to clean")
        return False

    conf_enabled = _confidence_enabled(config)
    thresholds = thresholds_from_config(config) if conf_enabled else None
    _, filler_phrases = load_merge_speaker_data(config)

    logger.info(
        f"  → Cleaning transcripts "
        f"(confidence pass {'ON' if conf_enabled else 'off'}, "
        f"{len(filler_phrases)} filler phrase(s))..."
    )
    total_before = total_after = 0
    for transcript_file in transcript_files:
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"    Could not load {transcript_file.name}: {e}")
            continue

        segments = data.get('segments', [])
        before = len(segments)

        if conf_enabled:
            segments = clean_segments(segments, thresholds)          # Pass A
        segments = [
            s for s in segments
            if not is_filler(s.get('text', '').strip(), filler_phrases)
        ]                                                            # Pass B

        data['segments'] = segments
        if not dry_run:
            with open(transcript_file, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(data, f, indent=2)

        total_before += before
        total_after += len(segments)
        logger.debug(f"    {transcript_file.name}: {before} -> {len(segments)} segment(s)")

    logger.info(
        f"  ✓ Cleaned transcripts: {total_before} -> {total_after} segment(s) "
        f"({total_before - total_after} dropped)"
    )
    return True
