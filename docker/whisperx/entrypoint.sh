#!/usr/bin/env bash
# WhisperX container entrypoint.
#
# Two modes:
#   1. No args  -> batch-transcribe every audio file in $AUDIO_DIR (default /audio)
#                  to $OUTPUT_DIR (default /output) using env-var config.
#   2. With args -> pass through directly to `whisperx`, e.g.
#                   docker compose run --rm whisperx --help
set -euo pipefail

MODEL="${WHISPER_MODEL:?config.yaml owns this (whisper.model); pass -e WHISPER_MODEL=... for a bare container run}"
LANGUAGE="${WHISPER_LANGUAGE:-}"   # empty = let whisperx auto-detect
# Paired with COMPUTE_TYPE: CTranslate2 has no efficient float16 path on CPU, so
# DEVICE=cpu needs COMPUTE_TYPE int8 or float32. Default matches whisperx's own.
# Cuda should use COMPUTE_TYPE float16
COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:?config.yaml owns this (whisper.compute_type); pass -e WHISPER_COMPUTE_TYPE=... for a bare container run}"
DEVICE="${WHISPER_DEVICE:?config.yaml owns this (whisper.device); pass -e WHISPER_DEVICE=... for a bare container run}"

BATCH_SIZE="${WHISPER_BATCH_SIZE:?config.yaml owns this (whisper.batch_size); pass -e WHISPER_BATCH_SIZE=... for a bare container run}"
OUTPUT_FORMAT="${WHISPER_OUTPUT_FORMAT:?config.yaml owns this (whisper.output_format); pass -e WHISPER_OUTPUT_FORMAT=... for a bare container run}"
DIARIZE="${WHISPER_DIARIZE:?config.yaml owns this (whisper.diarize); pass -e WHISPER_DIARIZE=... for a bare container run}"
HOTWORDS="${WHISPER_HOTWORDS:-}"           # optional: empty omits --hotwords
INITIAL_PROMPT="${WHISPER_INITIAL_PROMPT:-}"   # optional: empty omits the flag
AUDIO_DIR="${AUDIO_DIR:-/audio}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

# Direct passthrough to whisperx when the user supplies their own args.
if [ "$#" -gt 0 ]; then
    exec whisperx "$@"
fi

# Batch mode. Gather supported audio/video extensions (case-insensitive).
shopt -s nullglob nocaseglob
files=(
    "$AUDIO_DIR"/*.mp3
    "$AUDIO_DIR"/*.wav
    "$AUDIO_DIR"/*.m4a
    "$AUDIO_DIR"/*.flac
    "$AUDIO_DIR"/*.ogg
    "$AUDIO_DIR"/*.opus
    "$AUDIO_DIR"/*.webm
    "$AUDIO_DIR"/*.mp4
    "$AUDIO_DIR"/*.mkv
    "$AUDIO_DIR"/*.aac
)
shopt -u nocaseglob

if [ "${#files[@]}" -eq 0 ]; then
    cat <<EOF
No audio files found in $AUDIO_DIR.

Options:
  * Put audio into the bind-mounted ./audio folder and re-run.
  * Or call whisperx directly, e.g.:
      docker compose run --rm whisperx --help
      docker compose run --rm whisperx /audio/myfile.wav --model medium
EOF
    exit 0
fi

args=(
    --model "$MODEL"
    --compute_type "$COMPUTE_TYPE"
    --device "$DEVICE"
    --batch_size "$BATCH_SIZE"
    --output_dir "$OUTPUT_DIR"
    --output_format "$OUTPUT_FORMAT"
)

# whisperx detects the language itself when --language is omitted, so an empty
# value means exactly that. Passing --language "" instead would not: it is an
# explicit empty argument, not an absent flag.
if [ -n "$LANGUAGE" ]; then
    args+=(--language "$LANGUAGE")
fi

# Hint phrases (rare names/jargon) improve recognition. Opt-in: only added when
# WHISPER_HOTWORDS is non-empty (set from config's hotwords_file on the host).
if [ -n "$HOTWORDS" ]; then
    args+=(--hotwords "$HOTWORDS")
fi
# --initial_prompt seeds the FIRST WINDOW only (spelling/style conventions),
# where --hotwords applies for the whole run. Separate files, separate flags.
if [ -n "$INITIAL_PROMPT" ]; then
    args+=(--initial_prompt "$INITIAL_PROMPT")
fi

# Diarization is opt-in because it needs a Hugging Face token and an accepted
# pyannote model license.
if [ "${DIARIZE,,}" = "true" ] || [ "${DIARIZE,,}" = "1" ] || [ "${DIARIZE,,}" = "yes" ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "ERROR: WHISPER_DIARIZE=true but HF_TOKEN is not set." >&2
        echo "Get a token at https://huggingface.co/settings/tokens and accept the" >&2
        echo "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 terms." >&2
        exit 1
    fi
    args+=(--diarize --hf_token "$HF_TOKEN")
fi

echo "WhisperX batch mode"
echo "  model=$MODEL  language=${LANGUAGE:-(auto-detect)}  compute_type=$COMPUTE_TYPE  device=$DEVICE"
echo "  diarize=$DIARIZE  output_format=$OUTPUT_FORMAT"
[ -n "$HOTWORDS" ] && echo "  hotwords=on (${#HOTWORDS} chars)" || echo "  hotwords=off"
[ -n "$INITIAL_PROMPT" ] && echo "  initial_prompt=on (${#INITIAL_PROMPT} chars)" || echo "  initial_prompt=off"
echo "  ${#files[@]} file(s) to process"
echo

for f in "${files[@]}"; do
    echo "=== Transcribing: $(basename "$f") ==="
    whisperx "${args[@]}" "$f"
    echo
done

echo "Done. Outputs in $OUTPUT_DIR."
