#!/usr/bin/env bash
# PySceneDetect batch processor - single ROI mode.
#
# Processes videos with a single ROI value from SCENE_ROI environment variable.
# For time-varying ROI (from time_segments), use detect_scenes_multi.sh instead.
#
# Output files use standardized naming: Scene-{VIDEO_INDEX:02d}-{scene:03d}.{ext}
# Where:
#   VIDEO_INDEX = video sequence number (01, 02, 03 for multi-video; defaults to 01)
#   scene       = scene number within that video (001, 002, 003, etc.)
#
# Two modes:
#   1. No args  -> detect scenes in every video in $VIDEO_DIR (default /video),
#                  writing per-video output to $SCENES_DIR (default /output/scenes).
#                  Each video's folder contains:
#                    <basename>-Scenes.csv   - scene list with timestamps
#                    Scene-{VIDEO_INDEX:02d}-001.jpg, Scene-{VIDEO_INDEX:02d}-002.jpg, etc.
#   2. With args -> pass through directly to `scenedetect`, e.g.
#                   docker compose run --rm scenes --help
set -euo pipefail

THRESHOLD="${SCENE_THRESHOLD:?config.yaml owns this (scenes.threshold); pass -e SCENE_THRESHOLD=... for a bare container run}"
MIN_SCENE_LEN="${SCENE_MIN_LEN:?config.yaml owns this (scenes.min_length); pass -e SCENE_MIN_LEN=... for a bare container run}"
NUM_IMAGES="${SCENE_NUM_IMAGES:?config.yaml owns this (scenes.num_images); pass -e SCENE_NUM_IMAGES=... for a bare container run}"
IMAGE_FORMAT="${SCENE_IMAGE_FORMAT:?config.yaml owns this (scenes.image_format); pass -e SCENE_IMAGE_FORMAT=... for a bare container run}"
VIDEO_DIR="${VIDEO_DIR:-/video}"
SCENES_DIR="${SCENES_DIR:-/output/scenes}"
VIDEO_INDEX="${VIDEO_INDEX:-01}"  # Video sequence number for multi-video sessions (default 01 for single)
ROI="${SCENE_ROI-}"   # optional: empty (or unset) means full frame

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

if [ "$#" -gt 0 ]; then
    exec scenedetect "$@"
fi

echo using "Video dir $VIDEO_DIR and Scenes dir $SCENES_DIR"

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

Options:
  * Put a video into the bind-mounted ./video folder and re-run.
  * Or call scenedetect directly, e.g.:
      docker compose run --rm scenes --help
      docker compose run --rm scenes -i /video/session.mp4 -o /output/scenes/session detect-content list-scenes save-images
EOF
    exit 0
fi

mkdir -p "$SCENES_DIR"

echo "PySceneDetect batch mode (single ROI)"
echo "  threshold=$THRESHOLD  min_scene_len=$MIN_SCENE_LEN  num_images=$NUM_IMAGES  format=$IMAGE_FORMAT  roi=$ROI"
echo "  video_index=$VIDEO_INDEX (for multi-video sessions)"
echo "  ${#files[@]} video(s) to process"
echo

# Function to remove PySceneDetect's timing metadata line from CSV
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

for f in "${files[@]}"; do
    stem="$(basename "${f%.*}")"
    out="$SCENES_DIR/$stem"
    mkdir -p "$out"
    echo "=== Scenes: $(basename "$f") -> $out ==="

    # Build scenedetect command with optional ROI
    declare -a cmd=(
        scenedetect
        --input "$f"
        --output "$out"
    )
    
    if [ -n "$ROI" ]; then
        echo "  roi=$ROI"
        cmd+=(
            --crop $ROI
        )
    else
        echo "  roi=(none - full video)"
    fi

    cmd+=(
        detect-content
            --threshold "$THRESHOLD"
            --min-scene-len "$MIN_SCENE_LEN"
        list-scenes
        save-images
            --num-images "$NUM_IMAGES"
            $FORMAT_FLAG
    )

    # [*] joins the array into one string for the human; [@] keeps each element
    # a separate argument for the shell. Executing [*] looks for a program named
    # after the whole command line.
    echo "=== cmd ${cmd[*]} ==="
    "${cmd[@]}"
    echo

    # NOTE: scene images are left with PySceneDetect's raw "<video>-Scene-NNN-MM"
    # names here. They are canonicalised to Scene-{video:02d}-{scene:03d} host-side
    # after the docker run (pipeline/common/scenes.py::rename_scene_images), which
    # is where the video index is correctly assigned across multi-video sessions.

    # Remove PySceneDetect's timing metadata line from CSV (line 1)
    # PySceneDetect outputs: timing_line, headers, data_rows
    # We want: headers, data_rows only
    echo "  → Cleaning CSV (removing timing metadata line)..."
    clean_csv "$out"
    echo "  ✓ CSV cleaned"
    echo
done

echo "Done. Scene data in $SCENES_DIR."
echo "Scene files use standard naming: Scene-{video:02d}-{scene:03d}.{ext}"
