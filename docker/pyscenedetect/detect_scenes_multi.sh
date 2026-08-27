#!/usr/bin/env bash
# PySceneDetect multi-ROI processor for time-based ROI configurations.
#
# Workflow:
#   1. Load ROI config file from config.yaml (scenes.roi_file)
#   2. Extract time_segments with different ROIs
#   3. Split video into segments using ffmpeg (with re-encoding for reliability)
#   4. Run scenedetect on each segment with its configured ROI
#   5. Merge results: combine CSVs, adjust timestamps, consolidate images
#
# Note: Uses re-encoding (-c:v libx264) for segment splitting to ensure
# valid video files at arbitrary timestamps. This is slower than -c copy
# but more reliable. To use faster copy mode, set: FFMPEG_COPY=true
#
# Usage:
#   docker compose run --rm scenes_multi <video_file>
#   Or set VIDEO_DIR and it processes all videos in that directory
set -euo pipefail

# Make the pipeline package importable for `python -m pipeline...` (D2).
export PYTHONPATH="/usr/local/bin/pylib:${PYTHONPATH:-}"

THRESHOLD="${SCENE_THRESHOLD:?config.yaml owns this (scenes.threshold); pass -e SCENE_THRESHOLD=... for a bare container run}"
MIN_SCENE_LEN="${SCENE_MIN_LEN:?config.yaml owns this (scenes.min_length); pass -e SCENE_MIN_LEN=... for a bare container run}"
NUM_IMAGES="${SCENE_NUM_IMAGES:?config.yaml owns this (scenes.num_images); pass -e SCENE_NUM_IMAGES=... for a bare container run}"
IMAGE_FORMAT="${SCENE_IMAGE_FORMAT:?config.yaml owns this (scenes.image_format); pass -e SCENE_IMAGE_FORMAT=... for a bare container run}"
VIDEO_DIR="${VIDEO_DIR:-/video}"
SCENES_DIR="${SCENES_DIR:-/output/scenes}"
ROI_FILE="${ROI_FILE:?config.yaml owns this (scenes.roi_file); pass -e ROI_FILE=... for a bare container run}"
VIDEO_INDEX="${VIDEO_INDEX:-01}"  # Video sequence number for naming (01, 02, 03, etc.)

# Temporary working directory for segment files and debugging
# Using /output/temp so you can inspect intermediate results
TEMP_DIR="${TEMP_DIR:-/output/temp}"
DEBUG="${DEBUG:-false}"            # Verbose tracing of segment extraction
SPLIT_ONLY="${SPLIT_ONLY:-false}"  # If true, only split video and skip scene detection

# Truthiness for this script's own switches. Accepts the spellings a person
# would actually type; anything else is false. The SPLIT_ONLY guard used to be
# `= "false"`, which meant SPLIT_ONLY=0 or SPLIT_ONLY=no read as "not the string
# false" and silently DISABLED scene detection — the opposite of the intent.
is_true() {
    case "${1,,}" in
        true|1|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# Map IMAGE_FORMAT to scenedetect flag
FORMAT_FLAG="-j"  # Default to JPEG
case "$IMAGE_FORMAT" in
  jpg|jpeg) FORMAT_FLAG="-j" ;;
  png) FORMAT_FLAG="-p" ;;
  webp)
    echo "ERROR: SCENE_IMAGE_FORMAT=webp is not supported." >&2
    echo "       PySceneDetect can write WebP, but python-docx cannot embed it," >&2
    echo "       so the storyboard stage would fail after detection had run." >&2
    echo "       Set scenes.image_format to jpg or png." >&2
    exit 2 ;;
  *)
    echo "ERROR: SCENE_IMAGE_FORMAT='$IMAGE_FORMAT' is not a supported format." >&2
    echo "       Set scenes.image_format to jpg or png." >&2
    exit 2 ;;
esac


# Extract time segments from roi_history.json format
# Supports both flat and hierarchical formats
# Format flat: {"HH:MM:SS": {"frame": N, "roi": "x1 y1 x2 y2"}, ...}
# Format hierarchical: {"video.mkv": {"HH:MM:SS": {"frame": N, "roi": "x1 y1 x2 y2"}, ...}, ...}
# Output: segments in format "start|end|roi|description"
extract_segments() {
    local roi_file="$1"
    local video_name="$2"  # Optional: video filename for hierarchical format

    # Verify ROI file exists
    if [ ! -f "$roi_file" ]; then
        echo "ERROR: ROI file not found at: $roi_file" >&2
        return 1
    fi

    if is_true "$DEBUG"; then
        echo "DEBUG: Extracting segments via pipeline.scenes.extract_segments..." >&2
        echo "DEBUG:   ROI file: $roi_file" >&2
        echo "DEBUG:   Video name: ${video_name:-'(none)'}" >&2
    fi

    # ROI parsing lives in the RoiFile class (pipeline.scenes.extract_segments).
    if [ -n "$video_name" ]; then
        python3 -m pipeline.scenes.extract_segments "$roi_file" "$video_name" 2>&1
    else
        python3 -m pipeline.scenes.extract_segments "$roi_file" 2>&1
    fi
}

# Convert seconds to HH:MM:SS format for ffmpeg
seconds_to_timestamp() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d\n" "$hours" "$minutes" "$secs"
}

# Split video into segments using ffmpeg
split_video() {
    local input_file="$1"
    local output_dir="$2"
    local start_sec=$3
    local end_sec=$4
    local segment_num=$5

    mkdir -p "$output_dir"

    local segment_file="$output_dir/segment_${segment_num}.mp4"
    local ffmpeg_log="$output_dir/ffmpeg_segment_${segment_num}.log"
    local start_ts=$(seconds_to_timestamp "$start_sec")

    echo "  split_video: $start_sec - $end_sec "
    # Calculate segment duration
    local duration=$((end_sec - start_sec))

    # Handle final segment (end_sec == -1 means "to end of file")
    if [ "$end_sec" -eq -1 ]; then
        echo "  Splitting: $start_ts - END -> segment_${segment_num}.mp4"
        # Don't use -to parameter for final segment, let it run to end of video
        ffmpeg -hide_banner -loglevel error -i "$input_file" -ss "$start_ts" \
            -c:v libx264 -preset ultrafast -crf 28 -x264opts keyint=30:min-keyint=30 \
            -c:a aac -b:a 96k "$segment_file" -y > "$ffmpeg_log" 2>&1
    else
        echo "  Splitting: $start_ts - duration ${duration}s -> segment_${segment_num}.mp4"
        # Use duration (-t) instead of end time (-to) for more reliable short clips
        echo "ffmpeg -hide_banner -loglevel error -i $input_file -ss $start_ts -t $duration -c:v libx264 -preset ultrafast -crf 28 -x264opts keyint=30:min-keyint=30 -c:a aac -b:a 96k $segment_file -y > $ffmpeg_log 2>&1"
        ffmpeg -hide_banner -loglevel error -i "$input_file" -ss "$start_ts" -t "$duration" \
            -c:v libx264 -preset ultrafast -crf 28 -x264opts keyint=30:min-keyint=30 \
            -c:a aac -b:a 96k "$segment_file" -y > "$ffmpeg_log" 2>&1
    fi

    if [ ! -f "$segment_file" ]; then
        echo "ERROR: ffmpeg failed to create segment file at segment $segment_num" >&2
        echo "ERROR: Expected file: $segment_file" >&2
        if [ -f "$ffmpeg_log" ]; then
            echo "ERROR: ffmpeg output:" >&2
            cat "$ffmpeg_log" >&2
        fi
        return 1
    fi

    if [ ! -s "$segment_file" ]; then
        local size=$(stat -c%s "$segment_file" 2>/dev/null || stat -f%z "$segment_file" 2>/dev/null || echo "0")
        echo "ERROR: segment file is empty (size: $size bytes) at segment $segment_num" >&2
        if [ -f "$ffmpeg_log" ]; then
            echo "ERROR: ffmpeg output:" >&2
            cat "$ffmpeg_log" >&2
        fi
        return 1
    fi

    if is_true "$DEBUG"; then
        local size=$(stat -c%s "$segment_file" 2>/dev/null || stat -f%z "$segment_file" 2>/dev/null)
        echo "DEBUG: Created segment file: $(basename "$segment_file") (${size} bytes)"
    fi
    #rm -f "$ffmpeg_log"
}

# Remove PySceneDetect's timing metadata line from CSV files
# PySceneDetect puts timing info on line 1, headers on line 2, data on line 3+
# We normalize to: headers on line 1, data on line 2+
clean_csv() {
    local output_dir="$1"

    # Find all *-Scenes.csv files in the output directory
    shopt -s nullglob
    for csv_file in "$output_dir"/*-Scenes.csv; do
        if [ -f "$csv_file" ]; then
            # Remove first line (PySceneDetect timing metadata)
            # Create temp file, skip first line, write back
            tail -n +2 "$csv_file" > "${csv_file}.tmp"
            mv "${csv_file}.tmp" "$csv_file"
        fi
    done
    shopt -u nullglob
}

# Run scenedetect on a single segment
run_scenedetect_segment() {
    local segment_file="$1"
    local output_dir="$2"
    local roi="$3"
    local segment_num=$4

    mkdir -p "$output_dir"

    echo "  Processing: segment_${segment_num}.mp4 (ROI: $roi)"

    if [ -z "$roi" ]; then
        # No ROI, process full frame
        scenedetect \
            --input "$segment_file" \
            --output "$output_dir" \
            detect-content \
                --threshold "$THRESHOLD" \
                --min-scene-len "$MIN_SCENE_LEN" \
            list-scenes \
            save-images \
                --num-images "$NUM_IMAGES" \
                $FORMAT_FLAG 2>/dev/null || return 1
    else
        # Apply ROI cropping - convert format to comma-separated
        
        scenedetect \
            --input "$segment_file" \
            --output "$output_dir" \
            --crop $roi \
            detect-content \
                --threshold "$THRESHOLD" \
                --min-scene-len "$MIN_SCENE_LEN" \
            list-scenes \
            save-images \
                --num-images "$NUM_IMAGES" \
                $FORMAT_FLAG 2>/dev/null || return 1
    fi

    # Clean CSV files (remove PySceneDetect's timing metadata line)
    clean_csv "$output_dir"
}

# NOTE: the per-segment CSV+image reassembly (renumber in time order + canonical
# naming) used to live here (merge_scene_csvs / merge_scene_images). It now runs
# host-side in one place after the docker call —
# pipeline/scenes/merge_segments.py, invoked by detect_scenes_multi_roi.py.

# Process a single video with time-based ROI splitting
process_video_multi_roi() {
    local input_video="$1"
    local roi_config_file="$2"

    local stem="$(basename "${input_video%.*}")"
    local video_filename="$(basename "$input_video")"  # Full filename with extension for hierarchical format
    local output_dir="$SCENES_DIR/$stem"
    # Raw per-segment scenedetect output goes under the (mounted) output dir so
    # the host can reassemble it after the docker run.
    local segment_temp_dir="$output_dir/_segments"
    mkdir -p "$output_dir"

    echo "=== Multi-ROI Scene Detection: $video_filename ==="
    echo "ROI Config: $(basename "$roi_config_file")"

    # Extract segments
    echo "Extracting time segments from: $(basename "$roi_config_file")"
    if is_true "$DEBUG"; then
        echo "DEBUG: Looking for video: $video_filename"
    fi

    local segments=()
    while IFS='|' read -r start end roi desc; do
        segments+=("$start|$end|$roi|$desc")
        if is_true "$DEBUG"; then
            echo "DEBUG:   - $desc (${start}s-${end}s) ROI: $roi"
        fi
    done < <(extract_segments "$roi_config_file" "$video_filename")

    if [ ${#segments[@]} -eq 0 ]; then
        echo "ERROR: No time segments found in ROI config" >&2
        if is_true "$DEBUG"; then
            echo "DEBUG: Check roi_history.json format. It should have HH:MM:SS timestamps as keys."
            echo "DEBUG: Example: {\"00:00:00\": {\"frame\": 0, \"roi\": \"x1 y1 x2 y2\"}, ...}"
        fi
        return 1
    fi

    echo "Found ${#segments[@]} time segments"

    # Process each segment
    local seg_idx=0
    for segment_spec in "${segments[@]}"; do
        IFS='|' read -r start_sec end_sec roi desc <<< "$segment_spec"

        echo "Segment $seg_idx: $desc :: (${start_sec}s - ${end_sec}s)"

        # Split video
        split_video "$input_video" "$segment_temp_dir" "$start_sec" "$end_sec" "$seg_idx" || return 1

        # Skip scene detection if SPLIT_ONLY mode enabled (for debugging)
        if ! is_true "$SPLIT_ONLY"; then
            # Run scenedetect
            run_scenedetect_segment \
                "$segment_temp_dir/segment_${seg_idx}.mp4" \
                "$segment_temp_dir/segment_${seg_idx}" \
                "$roi" \
                "$seg_idx" || return 1

            # D6: record this segment's absolute start so merge_segments can
            # convert segment-relative scene times to absolute (no ROI re-read).
            mkdir -p "$segment_temp_dir/segment_${seg_idx}"
            echo "$start_sec" > "$segment_temp_dir/segment_${seg_idx}/offset.txt"
        else
            echo "  SPLIT_ONLY: Skipping scenedetect for segment_${seg_idx}.mp4"
        fi

        ((seg_idx++))
    done

    # Reassembly (renumber the CSV in time order + emit the canonical images) is
    # done HOST-SIDE after the docker run, in one place —
    # pipeline/scenes/merge_segments.py. The container leaves the raw per-segment
    # output under "$segment_temp_dir" for the host to consume and clean up.
    echo "Done (raw segments left for host reassembly): $segment_temp_dir"
    echo
}

# Main entry point
main() {
    # Resolve ROI file path
    # ROI_FILE can be either:
    # 1. A full path (e.g., "/video/roi_history.json" from orchestrate.py)
    # 2. Just a filename (e.g., "roi_history.json" to look for in VIDEO_DIR)
    # 3. Not set (use default resolution logic)

    local roi_file=""
    if [ -n "${ROI_FILE:-}" ]; then
        # ROI_FILE is set
        if [[ "$ROI_FILE" == /* ]]; then
            # Absolute path
            roi_file="$ROI_FILE"
        else
            # Relative filename, look in VIDEO_DIR
            roi_file="$VIDEO_DIR/$ROI_FILE"
        fi
    fi

    if is_true "$DEBUG"; then
        echo "DEBUG: ROI_FILE env var: ${ROI_FILE:-'(not set)'}"
        echo "DEBUG: Resolved roi_file: $roi_file"
    fi

    # If arguments provided, process specific video file
    if [ "$#" -gt 0 ]; then
        for video_file in "$@"; do
            if [ -f "$video_file" ]; then
                process_video_multi_roi "$video_file" "$roi_file" || exit 1
            else
                echo "ERROR: Video file not found: $video_file" >&2
                exit 1
            fi
        done
        exit 0
    fi

    # Otherwise, process all videos in VIDEO_DIR
    shopt -s nullglob nocaseglob
    files=(
        "$VIDEO_DIR"/*.mp4
        "$VIDEO_DIR"/*.mkv
        "$VIDEO_DIR"/*.mov
        "$VIDEO_DIR"/*.webm
        "$VIDEO_DIR"/*.avi
        "$VIDEO_DIR"/*.m4v
        "$VIDEO_DIR"/*.flv
        "$VIDEO_DIR"/*.wmv
    )
    shopt -u nocaseglob

    if [ "${#files[@]}" -eq 0 ]; then
        cat <<EOF
No video files found in $VIDEO_DIR.

Usage:
  docker compose run --rm scenes_multi
  docker compose run --rm scenes_multi /video/myfile.mp4

Requirements:
  - ROI config file must be specified in config.yaml (roi_file)
  - ROI config file must contain "time_segments" array
  - For single ROI processing, use detect_scenes.sh instead
EOF
        exit 0
    fi

    echo "Multi-ROI batch mode"
    echo "  threshold=$THRESHOLD  min_scene_len=$MIN_SCENE_LEN  num_images=$NUM_IMAGES  format=$IMAGE_FORMAT"
    echo "  roi_config=$(basename "$roi_file")"
    echo "  ${#files[@]} video(s) to process"
    echo

    # Process all videos
    for f in "${files[@]}"; do
        process_video_multi_roi "$f" "$roi_file" || exit 1
    done
}

echo "Done. Scene data in $SCENES_DIR."
echo "Scene files use standard naming: Scene-{video:02d}-{scene:03d}.{ext}"

# Call main function
main "$@"
