#!/usr/bin/env python3
"""
Scene naming and file conventions.

Defines standard patterns for scene file naming to ensure consistency
across all tools and workflows.
"""

import logging
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def find_candidate_manual_scenes(session_dir, manual_csv_name: str = "Scenes.csv",
                                 exclude: str = None) -> List[Path]:
    """Directories at the session root that look like hand-captured scenes.

    A manual capture folder holds one per-video subdirectory, each containing a
    ``manual_csv_name``. This finds any such folder whose NAME differs from the
    configured ``scenes.manual_source`` — i.e. the user captured scenes but the
    pipeline will not see them.

    Why this exists: the manual-scenes gate keys purely on the configured folder
    name existing. When it doesn't, orchestrate falls through to PySceneDetect,
    silently discards the hand-captured work, and emits a storyboard built from
    auto-detected scenes — output that looks fine and is wrong. That happened
    with CaptureScreens, which writes to ``scenes_output/`` by default while the
    pipeline expects ``screens/``.

    Deliberately narrow: only the session's immediate subdirectories, only ones
    actually containing ``*/<manual_csv_name>``, and never the configured name
    itself. Generated output lives under cc_output/ and so cannot match.
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        return []
    found = []
    for d in sorted(p for p in session_dir.iterdir() if p.is_dir()):
        if exclude and d.name == exclude:
            continue
        if list(d.glob(f"*/{manual_csv_name}")):
            found.append(d)
    return found

# Hard-coded scene naming pattern: Scene-{video:02d}-{scene:03d}.<ext>
# Where:
#   video = video sequence number (01, 02, 03, etc. for multi-video sessions)
#   scene = scene number within that video (001, 002, 003, etc.)
#   ext   = the session's image format, jpg or png
#
# Examples:
#   Scene-01-001.jpg  (video 1, scene 1)
#   Scene-01-042.png  (video 1, scene 42)
#   Scene-02-001.jpg  (video 2, scene 1)

SCENE_NAME_PATTERN = "Scene-{video:02d}-{scene:03d}"

# The two formats a session may be configured to produce. PySceneDetect can also
# write WebP (-w), and it is deliberately NOT offered: python-docx has no WebP
# reader (docx/image/ ships bmp, gif, jpeg, png, tiff), so add_picture() would
# raise UnrecognizedImageError at the storyboard stage — after detection, the
# rename and the merge had all already succeeded. Rejecting it at config load is
# the difference between a clear message and a long run that dies at the end.
SCENE_IMAGE_FORMATS = ("jpg", "png")

# Extensions recognised on disk, for both generated and hand-captured images.
# ".jpeg" is accepted as an input spelling: PySceneDetect's -j always writes
# ".jpg", but a manual capture folder may hold either.
SCENE_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def normalize_image_format(value) -> str:
    """Normalise a configured ``scenes.image_format`` to "jpg" or "png".

    Raises ValueError naming the offending value and the accepted set. WebP gets
    its own message because it is the one value a reader would reasonably expect
    to work — PySceneDetect supports it and the container flag exists.
    """
    text = str(value or "").strip().lower().lstrip(".")
    if text == "jpeg":
        text = "jpg"
    if text in SCENE_IMAGE_FORMATS:
        return text
    accepted = ", ".join(SCENE_IMAGE_FORMATS)
    if not text:
        raise ValueError(
            "scenes.image_format is empty. Remove the key to take the default "
            f"({SCENE_IMAGE_FORMATS[0]}), or set it to one of: {accepted}."
        )
    if text == "webp":
        raise ValueError(
            "scenes.image_format 'webp' is not supported: PySceneDetect can "
            "write it, but python-docx cannot embed it, so the storyboard stage "
            f"would fail after detection had already run. Use one of: {accepted}."
        )
    raise ValueError(
        f"scenes.image_format {value!r} is not a supported image format. "
        f"Use one of: {accepted}."
    )


def iter_scene_images(directory, pattern: str = "*Scene-*") -> List[Path]:
    """Scene images in ``directory``, any supported extension, sorted by name.

    The single place the pipeline decides what counts as a scene image. Callers
    match on the NAME pattern only ("Scene-*" for canonical names, "*Scene-*" to
    include PySceneDetect's raw "<prefix>-Scene-NNN-MM") and let this supply the
    extensions, so a session's format never has to be threaded through to a glob.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.glob(pattern)
         if p.is_file() and p.suffix.lower() in SCENE_IMAGE_EXTENSIONS),
        key=lambda p: p.name,
    )


def format_scene_name(video_index: int, scene_number: int, extension: str = "jpg") -> str:
    """
    Format a scene filename following the standard naming convention.

    Args:
        video_index: 1-based video sequence number (1, 2, 3, etc.)
        scene_number: 1-based scene number within the video (1, 2, 3, etc.)
        extension: File extension (default: jpg)

    Returns:
        Formatted scene filename (e.g., "Scene-01-001.jpg")
    """
    return f"{SCENE_NAME_PATTERN.format(video=video_index, scene=scene_number)}.{extension}"


def parse_scene_name(filename: str) -> tuple[int, int] | None:
    """
    Parse a scene filename to extract video index and scene number.

    Args:
        filename: Scene filename (e.g., "Scene-01-001.jpg")

    Returns:
        Tuple of (video_index, scene_number) or None if parsing fails
    """
    match = re.match(r"Scene-(\d{2})-(\d{3})", filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


# PySceneDetect's raw per-scene image: "<video-stem>-Scene-NNN-MM.ext"
# (a prefix, then the scene number, then the image index within the scene).
# The leading ".+-" requires the prefix, so canonical files (which start with
# "Scene-") never match — keeping the rename idempotent.
_RAW_SCENE_RE = re.compile(r"^.+-Scene-(\d+)-(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_raw_scene_name(filename: str):
    """``(scene_number, extension)`` from PySceneDetect's raw output name.

    Matches ``<prefix>-Scene-NNN-MM.ext``; returns None for a canonical name (no
    prefix) or a non-scene file. The extension comes back lowercased and without
    the dot, ready for format_scene_name.
    """
    m = _RAW_SCENE_RE.match(filename)
    if not m:
        return None
    return int(m.group(1)), m.group(3).lower()


def rename_scene_images(scenes_dir, num_images: int = 1) -> int:
    """Rename PySceneDetect's raw scene images to the canonical name, host-side.

    PySceneDetect writes ``<video-stem>-Scene-NNN-MM.ext`` into one subdirectory
    per video under ``scenes_dir``. This walks those subdirs (sorted), assigns
    the video index by order — so multi-video sessions get 01, 02, 03… (this is
    where VIDEO_INDEX is correctly established, not the container) — and renames
    each image to ``Scene-{video:02d}-{scene:03d}.ext`` using the scene number
    parsed from the raw name.

    Idempotent: already-canonical files (no ``<prefix>-`` before ``Scene-``) are
    skipped. Assumes one image per scene (``num_images == 1``, the config
    default); with more, a canonical name has no image slot, so collisions are
    logged and skipped rather than overwriting.

    Returns the number of files renamed.
    """
    scenes_dir = Path(scenes_dir)
    if not scenes_dir.exists():
        return 0

    renamed = 0
    subdirs = sorted(d for d in scenes_dir.iterdir() if d.is_dir())
    for video_index, subdir in enumerate(subdirs, start=1):
        for img in sorted(subdir.iterdir()):
            if not img.is_file():
                continue
            parsed = parse_raw_scene_name(img.name)
            if parsed is None:
                continue  # canonical already, or not a scene image
            scene, ext = parsed
            target = subdir / format_scene_name(video_index, scene, ext)
            if target == img:
                continue
            if target.exists():
                logger.warning(
                    f"Scene image target already exists, skipping: {target.name} "
                    f"(num_images>1 not representable in canonical naming?)"
                )
                continue
            img.rename(target)
            renamed += 1
    return renamed
