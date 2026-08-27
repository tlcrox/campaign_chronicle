#!/usr/bin/env python3
"""
combine_transcripts.py - combine a session's per-source WhisperX transcripts into
one timeline-ordered transcript.

The transcript half of the old combine_week.py (the scene half moved to
combine_scenes.py). Stitches every source's segments — Audacity per-player audio
or per-video diarization — onto one session timeline. Speaker mapping and filler
removal are separate upstream stages (map_speakers, clean_transcription); this
module only interleaves.

No pandas dependency here (that's scene-side), so the lightweight transcribe
stages can import this without pulling pandas in.

Importable API:
    is_filler(text, filler_phrases) -> bool
    load_merge_speaker_data(cfg) -> (filename_mapping, filler_phrases)
    speaker_from_filename(transcript_file, filename_mapping) -> str | None
    merge_transcripts(sources) -> dict                    # shared interleave core
    merge_transcripts_audacity(transcript_files) -> dict  # raw per-player files
        (resolves speaker from filename; used before map_speakers has run). The
        post-map_speakers merge path builds sources and calls merge_transcripts()
        directly — see tools/merge_transcripts.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

# SPEAKER_MAP (Craig filename token -> character name) and FILLER_PHRASES are
# user data, not code — they live in the user's speaker_config.json under the
# "filename_mapping" and "filler_phrases" keys, loaded via load_merge_speaker_data().

logger = logging.getLogger(__name__)


def is_filler(text: str, filler_phrases=None) -> bool:
    """True if the whole utterance is just filler / minor acknowledgment.

    ``filler_phrases`` is a user-supplied set (from speaker_config.json); when
    empty or None, nothing is treated as filler — no phrases are baked in.
    """
    if not filler_phrases:
        return False
    normalized = text.strip().lower().rstrip(" .,!?;:")
    return normalized in filler_phrases


def load_merge_speaker_data(cfg):
    """Load user merge data from speaker_config.json (source_dir-relative).

    Returns (filename_mapping, filler_phrases):
      - filename_mapping: With Audacity Project Craig filename token -> character name
        (from the "filename_mapping" key).
      - filler_phrases: lowercased set of throwaway utterances to drop (from the
        "filler_phrases" key).

    Returns ({}, set()) when the file or keys are absent — no speaker names or
    filler phrases are baked into the codebase.
    """
    filename_mapping, filler_phrases = {}, set()
    try:
        path = Path(cfg.speaker_config_file)
    except Exception:
        return filename_mapping, filler_phrases
    if not path.exists():
        return filename_mapping, filler_phrases
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return filename_mapping, filler_phrases
    fm = data.get("filename_mapping") or {}
    filename_mapping = {k: v for k, v in fm.items() if str(k).lower() != "comment"}
    fp = data.get("filler_phrases") or []
    filler_phrases = {str(p).strip().lower() for p in fp if str(p).strip()}
    return filename_mapping, filler_phrases


def speaker_from_filename(transcript_file, filename_mapping):
    """Resolve a per-speaker file's character name from a token in its filename
    (Workflow A / Audacity), falling back to a same-named audio file next to it.
    Returns None if no token matches. Extracted so the map_speakers stage can do
    this resolution instead of the merge.
    """
    filename_mapping = filename_mapping or {}
    filename = transcript_file.name.lower()
    for token, display_name in filename_mapping.items():
        if str(token).lower() in filename:
            return display_name
    # Fallback: a same-named audio file in the same directory.
    for ext in (".wav", ".flac", ".m4a", ".mp3", ".ogg"):
        candidate = transcript_file.parent / transcript_file.name.replace(".json", ext)
        if candidate.exists():
            audio_name = candidate.name.lower()
            for token, display_name in filename_mapping.items():
                if str(token).lower() in audio_name:
                    return display_name
    return None


def _first_named_speaker(segments):
    """The file's speaker if it's already a resolved character name (not a raw
    ``SPEAKER_XX`` id), else None. Lets the merge prefer a name already stamped by
    the map_speakers filename stage rather than re-resolving from the filename."""
    for segment in segments or []:
        spk = segment.get("speaker")
        if not spk:
            continue
        return None if re.match(r"^SPEAKER_\d+$", str(spk)) else spk
    return None


def merge_transcripts(sources: List[Dict]) -> Dict:
    """Unified transcript merge: interleave every source's segments onto one
    timeline, in chronological order. This is the shared core behind both the
    Audacity (Workflow A) and diarization (Workflow B) front ends.

    ``sources`` is a list of per-source descriptors, each a dict of:
      - ``segments``: the source's WhisperX segment list.
      - ``offset``: seconds added to every segment's start/end, shifting a 0-based
        track onto the session timeline. Parallel tracks (Audacity per-player
        audio) share a timeline and use 0; serial tracks (concatenated videos)
        use the cumulative duration of the ones before them.
      - ``speaker``: optional single name stamped on every segment of this source
        (Workflow A — one speaker per file). When None/absent, each segment keeps
        its own already-mapped speaker (Workflow B).
      - ``name``: label used only for logging.

    Empty-text segments are dropped. Segments are NOT coalesced: a coalesced turn
    keeps only its first segment's start, so it could straddle a scene boundary
    and make the storyboard flush post-image speech before the image. The
    storyboard coalesces same-speaker runs within each inter-image window instead
    (see storyboard.py). Returns ``{"segments": [...], "language": "en"}``, or
    ``{"segments": []}`` when nothing merged.
    """
    utterances = []  # List of (start, end, speaker, segment_data)
    per_speaker_segments = {}

    for source in sources:
        offset = float(source.get("offset", 0.0))
        override = source.get("speaker")
        count = 0
        for segment in source.get("segments", []):
            text = segment.get('text', '').strip()
            if not text:
                continue
            start = float(segment.get('start', 0.0)) + offset
            end = float(segment.get('end', 0.0)) + offset
            speaker = override if override else segment.get('speaker', 'Unknown')
            utterances.append((start, end, speaker, segment.copy()))
            per_speaker_segments[speaker] = per_speaker_segments.get(speaker, 0) + 1
            count += 1
        logger.debug(f"    {str(source.get('name', '?')):<20} {count} segment(s)")

    if not utterances:
        return {"segments": []}

    # Sort by timestamp to interleave speakers chronologically.
    # Key: (start_time, end_time, speaker) - ensures correct chronological order.
    utterances.sort(key=lambda u: (u[0], u[1], u[2]))

    merged_segments = []
    for start, end, speaker, segment in utterances:
        segment_copy = segment.copy()
        segment_copy['start'] = start
        segment_copy['end'] = end
        segment_copy['speaker'] = speaker
        merged_segments.append(segment_copy)

    logger.info(f"  ✓ Merged {len(merged_segments)} segments from {len(per_speaker_segments)} speaker(s):")
    for spk, count in sorted(per_speaker_segments.items(), key=lambda kv: -kv[1]):
        logger.info(f"    - {spk:<20} {count} segment(s)")

    return {"segments": merged_segments, "language": "en"}


def merge_transcripts_audacity(transcript_files: List[Path], filename_mapping=None) -> Dict:
    """Workflow A front end (Audacity per-speaker audio): resolve one speaker per
    file, then merge on a shared timeline (offset 0).

    Each input JSON is a single-speaker export. The speaker is taken from a name
    already stamped on the segments (by the map_speakers filename stage) or, for
    callers that haven't run that stage yet, resolved from the filename via
    ``filename_mapping``. Files with no resolvable speaker are reported loudly and
    reflected in the ``skipped_files`` / ``skipped_details`` return keys.
    """
    if not transcript_files:
        return {"segments": []}

    filename_mapping = filename_mapping or {}

    sources = []
    skipped_files = []  # Track files that couldn't be mapped

    logger.info(f"  Merging {len(transcript_files)} transcript file(s)...")

    # Pass 1: Load all transcripts and extract speaker from filename
    for transcript_file in sorted(transcript_files):
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"  Could not load transcript {transcript_file}: {e}")
            skipped_files.append((transcript_file.name, f"Failed to load: {e}"))
            continue

        if 'segments' not in transcript:
            logger.warning(f"  No segments in {transcript_file.name}")
            skipped_files.append((transcript_file.name, "No 'segments' key in JSON"))
            continue

        # Speaker resolution: prefer a name already stamped on the segments (by
        # the map_speakers filename stage); otherwise resolve it from the filename
        # here, for callers that haven't run that stage yet.
        speaker = _first_named_speaker(transcript.get('segments', []))
        if not speaker:
            speaker = speaker_from_filename(transcript_file, filename_mapping)

        if not speaker:
            logger.error(f"  ✗ MISSING MAPPING: Could not identify speaker for {transcript_file.name}")
            logger.error(f"    Available filename tokens: {', '.join(sorted(filename_mapping.keys()))}")
            skipped_files.append((transcript_file.name, "No matching speaker token in filename_mapping"))
            continue

        # One source per file: offset 0 (per-player tracks share a timeline) and
        # the resolved speaker stamped on every segment. Empty-text drop, filler
        # removal (clean_transcription stage), and interleaving all happen in the
        # shared merge_transcripts core.
        sources.append({
            "segments": transcript['segments'],
            "offset": 0.0,
            "speaker": speaker,
            "name": speaker,
        })

    # Report skipped files loudly
    if skipped_files:
        logger.error(f"\n  ╔{'═' * 68}╗")
        logger.error(f"  ║ ⚠️  MISSING SPEAKER MAPPINGS - {len(skipped_files)} file(s) SKIPPED ⚠️")
        logger.error(f"  ╠{'═' * 68}╣")
        for filename, reason in skipped_files:
            logger.error(f"  ║ • {filename:<50} | {reason}")
        logger.error(f"  ║")
        logger.error(f"  ║ ACTION REQUIRED: Add these speakers to filename_mapping in")
        logger.error(f"  ║ speaker_config.json OR fix the filenames to match existing tokens.")
        logger.error(f"  ╚{'═' * 68}╝\n")

    result = merge_transcripts(sources)
    if not result["segments"]:
        logger.error("  ✗ FATAL: No utterances found in any file (all files were skipped or empty)")
        return {"segments": []}

    # The merge tool serializes this whole dict into combined_output, so the
    # skipped-file bookkeeping travels with the merged transcript.
    result["skipped_files"] = len(skipped_files)
    result["skipped_details"] = skipped_files
    return result
