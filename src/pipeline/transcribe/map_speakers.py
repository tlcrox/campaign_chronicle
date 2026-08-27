#!/usr/bin/env python3
"""
Apply speaker_config.json mappings to WhisperX transcripts.

Maps generic SPEAKER_XX IDs to character names using speaker_config.json.
Used in Workflow B (video-only with diarization).

Replaces speaker fields directly in transcript, including word-level speaker info.

Usage:
    from pipeline.transcribe.map_speakers import apply_speaker_mapping
    transcript = apply_speaker_mapping(transcript_dict, session_dir, config)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from pipeline.common.mounts import output_dir_for, SESSION_OUTPUT_SUBDIR

logger = logging.getLogger(__name__)


def require_diarization_for_mapping(config, speaker_config_file: Optional[Path] = None) -> None:
    """Flag a self-contradictory or easy-to-miss speaker-mapping setup.

    Speaker mapping renames the ``SPEAKER_XX`` diarization IDs to character
    names from ``speaker_config.json``. Those IDs only exist when
    ``whisper.diarize`` is true, which gives four combinations:

    - speaker_config.json present, diarize OFF: mapping is configured but can
      never do anything (diarization never produces SPEAKER_XX tokens to
      rename). Almost always a mistake — raise loudly.
    - diarize ON, no speaker_config.json (missing, or unresolvable): a normal
      bootstrapping state — you typically need to see a diarized transcript's
      SPEAKER_XX IDs before you know what to map them to. Does NOT raise, but
      it's easy to not notice that mapping is silently doing nothing, so warn
      loudly instead.
    - diarize OFF, no speaker_config.json: fully consistent — no mapping is
      wanted, and none will happen. Silent.
    - diarize ON, speaker_config.json present: everything lines up. Silent.
    """
    try:
        diarize_on = bool(config.whisper_diarize)
    except Exception:
        diarize_on = False

    if speaker_config_file is None:
        try:
            speaker_config_file = Path(config.speaker_config_file)
        except Exception:
            speaker_config_file = None
    else:
        speaker_config_file = Path(speaker_config_file)

    config_exists = speaker_config_file is not None and speaker_config_file.exists()

    if config_exists and not diarize_on:
        raise ValueError(
            f"{speaker_config_file} exists (speaker mapping is configured) but "
            "whisper.diarize is false. Diarization produces the SPEAKER_XX tokens "
            "that mapping renames, so mapping would silently do nothing. Enable "
            "whisper.diarize in config.yaml, or remove/rename the speaker_config.json."
        )

    if diarize_on and not config_exists:
        where = speaker_config_file if speaker_config_file is not None else "the configured speaker_config_file"
        logger.warning(
            f"DID YOU REALLY MEAN TO? whisper.diarize is true but no speaker_config.json "
            f"was found at {where} — the transcript will keep raw SPEAKER_XX labels; no "
            "speaker-name mapping will happen. Create the file to map speakers, or ignore "
            "this if that's expected (e.g. you haven't picked speaker names yet)."
        )


def apply_speaker_mapping(
    transcript: Dict,
    session_dir: Path,
    config,
    speaker_config_file: Optional[Path] = None,
    transcript_name: Optional[str] = None,
) -> Dict:
    """
    Apply speaker_config.json mappings to WhisperX transcript.

    Replaces SPEAKER_XX IDs with character names from speaker_config.json.
    Updates both segment-level and word-level speaker fields.

    Args:
        transcript: WhisperX JSON transcript dict
        session_dir: Path to session directory (for session-key extraction)
        config: Config object (for default speaker_config path)
        speaker_config_file: Optional override path to speaker_config.json
        transcript_name: This transcript's stem (e.g. "Week13_a"). When given,
            a per-file ``session_mappings`` key "<session_key>/<transcript_name>"
            is tried before the per-session one — required because diarization
            SPEAKER_XX IDs are assigned per file, so a multi-recording session
            can need a different mapping per file. Omit for a session-level lookup.

    Returns:
        Modified transcript dict with character names instead of SPEAKER_XX
    """
    if not speaker_config_file:
        # Use config default or look in project root
        try:
            speaker_config_file = Path(config.speaker_config_file)
        except Exception as e:
            logger.debug(f"Failed to get speaker_config_file from config: {e}. Using default.")
            speaker_config_file = Path("speaker_config.json")

    if not speaker_config_file.exists():
        logger.warning(f"speaker_config.json not found at {speaker_config_file}, skipping speaker mapping")
        return transcript

    # Load speaker config
    try:
        with open(speaker_config_file, 'r', encoding='utf-8') as f:
            speaker_config = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load speaker_config.json: {e}")
        return transcript

    if not transcript or 'segments' not in transcript:
        logger.warning("No segments in transcript")
        return transcript

    # Session identity for the mapping lookup. Diarization assigns SPEAKER_XX IDs
    # PER FILE (pyannote numbers each recording independently; there is no
    # cross-file voiceprint identity unless embeddings are enabled), so a session
    # with several recordings can need a different mapping per file — SPEAKER_01 in
    # one file may be a different person than SPEAKER_01 in the next.
    #
    # Keys use the session's path RELATIVE TO source_dir — the same identity
    # config.session_key drives everywhere else — rendered with forward slashes so
    # the "Weeks/Week 14" style keys in speaker_config.json match on Windows too.
    try:
        session_key = config.session_key(Path(session_dir)).as_posix()
    except Exception:
        session_key = Path(session_dir).name

    # Build mapping: SPEAKER_XX -> character name.
    # Priority: most-specific session/file mapping > global mappings.
    mapping = {}
    if "global_mapping" in speaker_config:
        mapping.update(speaker_config["global_mapping"])

    matched_key = None
    if "session_mappings" in speaker_config:
        session_mappings = speaker_config["session_mappings"]
        # Most specific -> least: per-file, then per-session, then legacy leaf name
        # and absolute path (back-compat with configs written before session_key).
        candidate_keys = []
        if transcript_name:
            candidate_keys.append(f"{session_key}/{transcript_name}")
        candidate_keys.append(session_key)
        for legacy in (Path(session_dir).name, str(session_dir)):
            if legacy not in candidate_keys:
                candidate_keys.append(legacy)
        for key in candidate_keys:
            if key in session_mappings:
                mapping.update(session_mappings[key])
                matched_key = key
                logger.debug(f"Found session-specific mapping for {key!r}")
                break

    logger.info(f"Speaker mapping for {matched_key or session_key}: {mapping}")

    # Apply mappings to segments
    unmapped_speakers = set()
    mapped_count = 0

    for segment in transcript.get('segments', []):
        if 'speaker' in segment:
            old_speaker = segment['speaker']

            # Look up in mapping
            if old_speaker in mapping:
                new_speaker = mapping[old_speaker]
                segment['speaker'] = new_speaker
                mapped_count += 1
                logger.debug(f"Mapped {old_speaker} → {new_speaker}")

                # Also update word-level speaker info
                if 'words' in segment:
                    for word in segment['words']:
                        if word.get('speaker') == old_speaker:
                            word['speaker'] = new_speaker
            else:
                unmapped_speakers.add(old_speaker)

    # Log results
    logger.info(f"Speaker mapping complete: {mapped_count} segments mapped")
    if unmapped_speakers:
        logger.warning(f"Unmapped speaker IDs: {unmapped_speakers}")

    return transcript


def map_speakers(session_dir: Path, config, dry_run: bool = False,
                       resolver: str = "auto") -> bool:
    """Speaker-ID stage: resolve character names on a session's per-source
    transcripts, in place, AFTER transcription and BEFORE merge.

    resolver:
      - ``"diarization"`` (Workflow B): map ``SPEAKER_XX`` -> name via
        speaker_config.json (video, standalone audio).
      - ``"filename"`` (Workflow A): each per-speaker file IS one speaker, named
        by a token in its filename; stamp that name on every segment (Audacity).
      - ``"auto"`` (default): pick ``filename`` when a transcript filename matches
        a ``filename_mapping`` token, else ``diarization`` — the same detection the
        merge uses, so callers don't have to know the workflow.

    Maps every ``<session>/transcriptions/*.json`` (excluding combined files).
    Returns True when at least one transcript was mapped.
    """
    session_dir = Path(session_dir)
    out_dir = output_dir_for(session_dir, SESSION_OUTPUT_SUBDIR, config)
    transcript_files = sorted(
        f for f in out_dir.glob("*.json") if "combined" not in f.name
    )
    if not transcript_files:
        logger.warning(f"  ⊘ No transcript files found")
        return False

    if resolver == "auto":
        resolver = _detect_resolver(transcript_files, config)

    if resolver == "filename":
        return _map_by_filename(transcript_files, config, dry_run)
    return _map_by_diarization(transcript_files, session_dir, config, dry_run)


def _detect_resolver(transcript_files, config) -> str:
    """Workflow A (``filename``) when any transcript filename contains a
    filename_mapping token, else Workflow B (``diarization``)."""
    try:
        from pipeline.merge.combine_transcripts import load_merge_speaker_data
        filename_mapping, _ = load_merge_speaker_data(config)
    except Exception:
        filename_mapping = {}
    for f in transcript_files:
        name = f.name.lower()
        for token in (filename_mapping or {}):
            if str(token).lower() in name:
                return "filename"
    return "diarization"


def _map_by_diarization(transcript_files, session_dir, config, dry_run) -> bool:
    """Workflow B: ``SPEAKER_XX`` -> character name via speaker_config.json."""
    logger.info(f"  → Applying speaker mappings (Workflow B)...")
    try:
        mapped_count = 0
        for transcript_file in transcript_files:
            logger.debug(f"    Mapping speakers in {transcript_file.name}...")
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript = json.load(f)
                transcript = apply_speaker_mapping(
                    transcript, session_dir, config,
                    transcript_name=transcript_file.stem,
                )
                if not dry_run:
                    with open(transcript_file, 'w', encoding='utf-8', newline='\n') as f:
                        json.dump(transcript, f, indent=2)
                mapped_count += 1
            except Exception as e:
                logger.warning(f"    Could not map speakers in {transcript_file.name}: {e}")
        if mapped_count > 0:
            logger.info(f"  ✓ Speaker mappings applied to {mapped_count} transcript file(s)")
            return True
        logger.warning(f"  ⊘ No transcripts were mapped")
        return False
    except Exception as e:
        logger.warning(f"  ⊘ Speaker mapping failed: {e}")
        return False


def _map_by_filename(transcript_files, config, dry_run) -> bool:
    """Workflow A: each per-speaker file IS one speaker, named by a filename token.
    Stamps that name on every segment/word; fails loud on an unresolvable file
    (matching the old merge's skipped-files behavior)."""
    from pipeline.merge.combine_transcripts import load_merge_speaker_data, speaker_from_filename
    filename_mapping, _ = load_merge_speaker_data(config)
    logger.info(f"  → Applying speaker names from filenames (Workflow A)...")
    skipped = []
    mapped_count = 0
    for transcript_file in transcript_files:
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            skipped.append((transcript_file.name, f"Failed to load: {e}"))
            continue
        if 'segments' not in transcript:
            skipped.append((transcript_file.name, "No 'segments' key in JSON"))
            continue
        speaker = speaker_from_filename(transcript_file, filename_mapping)
        if not speaker:
            logger.error(f"  ✗ MISSING MAPPING: no filename token matched {transcript_file.name}")
            logger.error(f"    Available filename tokens: {', '.join(sorted(filename_mapping.keys()))}")
            skipped.append((transcript_file.name, "No matching speaker token in filename_mapping"))
            continue
        for segment in transcript['segments']:
            # Segment-level only — matches the previous Audacity merge, which left
            # word-level speaker untouched. (Byte-identical golden; the word-level
            # inconsistency vs the diarization path is pre-existing, not for 1c.)
            segment['speaker'] = speaker
        if not dry_run:
            with open(transcript_file, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(transcript, f, indent=2)
        mapped_count += 1
        logger.debug(f"    {transcript_file.name} -> {speaker}")

    if skipped:
        logger.error(f"\n  ╔{'═' * 68}╗")
        logger.error(f"  ║ ⚠️  MISSING SPEAKER MAPPINGS - {len(skipped)} file(s) SKIPPED ⚠️")
        logger.error(f"  ╠{'═' * 68}╣")
        for filename, reason in skipped:
            logger.error(f"  ║ • {filename:<50} | {reason}")
        logger.error(f"  ╚{'═' * 68}╝\n")
        return False  # fail loud, matching the old skipped-files behavior

    if mapped_count > 0:
        logger.info(f"  ✓ Speaker names applied to {mapped_count} transcript file(s)")
        return True
    logger.warning(f"  ⊘ No transcripts were mapped")
    return False


def apply_speaker_mapping_to_file(
    transcript_file: Path,
    session_dir: Path,
    config,
    output_file: Optional[Path] = None,
    speaker_config_file: Optional[Path] = None
) -> Path:
    """
    Apply speaker mappings to a transcript file and save result.

    Args:
        transcript_file: Path to input WhisperX JSON
        session_dir: Path to session directory
        config: Config object
        output_file: Optional output file path (defaults to same as input)
        speaker_config_file: Optional override path to speaker_config.json

    Returns:
        Path to output file
    """
    # Load transcript
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Could not load transcript {transcript_file}: {e}")
