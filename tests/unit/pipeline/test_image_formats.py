#!/usr/bin/env python3
"""The session image format (jpg | png) carried through every non-Docker stage.

PySceneDetect can write jpg, png and webp; the pipeline supports only the first
two. WebP is rejected at config load because python-docx cannot embed it, so a
webp run would otherwise detect, rename and merge successfully and then fail at
the storyboard — the last stage of a long run.

Everything below the container boundary is exercised twice, once per supported
format. The container scripts themselves (detect_scenes.sh, detect_scenes_multi.sh)
are out of scope here: they are shell, and their mapping is a case statement.
"""

import base64
import csv
import io
import tempfile
import unittest
from pathlib import Path

from docx import Document

from pipeline.common.scenes import (
    SCENE_IMAGE_EXTENSIONS,
    SCENE_IMAGE_FORMATS,
    format_scene_name,
    iter_scene_images,
    normalize_image_format,
    parse_raw_scene_name,
    rename_scene_images,
)
from pipeline.config import Config, get_config
from pipeline.merge.combine_scenes import merge_image_folders
from pipeline.merge.storyboard import generate_storyboard
from pipeline.scenes.merge_segments import merge_segments

# Smallest valid 1x1 images. Real bytes, not b"x": python-docx parses the header
# when embedding, so a placeholder would pass the globbing tests and hide the
# only failure mode that matters for the storyboard.
PIXELS = {
    "png": base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQ"
        "DwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    ),
    "jpg": base64.b64decode(
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
        "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAB"
        "AAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAA"
        "AAD/2gAIAQEAAD8AKp//2Q=="
    ),
}

FORMATS = ("jpg", "png")

CSV_HEADER = (
    "Scene Number,Start Frame,Start Timecode,Start Time (seconds),"
    "End Frame,End Timecode,End Time (seconds),"
    "Length (frames),Length (timecode),Length (seconds)\n"
)


def write_image(path: Path, ext: str) -> Path:
    """Write a real 1x1 image of the given format at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PIXELS["jpg" if ext in ("jpg", "jpeg") else ext])
    return path


class NormalizeImageFormat(unittest.TestCase):
    """The one place a configured format string is interpreted."""

    def test_accepts_both_supported_formats(self):
        for fmt in FORMATS:
            self.assertEqual(normalize_image_format(fmt), fmt)
            self.assertIn(fmt, SCENE_IMAGE_FORMATS)

    def test_normalises_spelling(self):
        self.assertEqual(normalize_image_format("JPG"), "jpg")
        self.assertEqual(normalize_image_format("jpeg"), "jpg")
        self.assertEqual(normalize_image_format("JPEG"), "jpg")
        self.assertEqual(normalize_image_format(".png"), "png")
        self.assertEqual(normalize_image_format("  PNG  "), "png")

    def test_webp_is_rejected_and_says_why(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_image_format("webp")
        message = str(ctx.exception)
        self.assertIn("webp", message)
        self.assertIn("python-docx", message)
        self.assertIn("jpg, png", message)

    def test_other_values_are_rejected(self):
        for bad in ("gif", "bmp", "tiff", "", None, "jpgg"):
            with self.assertRaises(ValueError, msg=f"accepted {bad!r}"):
                normalize_image_format(bad)


class ConfigImageFormat(unittest.TestCase):
    """A bad format fails at config load, not five stages later."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _config(self, image_format_line: str) -> Path:
        path = self.root / "config.yaml"
        # Only the section under test: every other section is defaulted at load.
        io.open(path, "w", encoding="utf-8", newline="\n").write(
            "scenes:\n" + image_format_line
        )
        return path

    def test_each_supported_format_loads(self):
        for fmt in FORMATS:
            cfg = Config(self._config(f"  image_format: {fmt}\n"))
            self.assertEqual(cfg.scene_image_format, fmt)

    def test_jpeg_is_normalised_for_callers(self):
        cfg = Config(self._config("  image_format: jpeg\n"))
        self.assertEqual(cfg.scene_image_format, "jpg")

    def test_missing_key_defaults_to_jpg(self):
        cfg = Config(self._config("  num_images: 1\n"))
        self.assertEqual(cfg.scene_image_format, "jpg")

    def test_webp_fails_at_load_naming_the_file(self):
        path = self._config("  image_format: webp\n")
        with self.assertRaises(ValueError) as ctx:
            Config(path)
        message = str(ctx.exception)
        self.assertIn(str(path), message)
        self.assertIn("python-docx", message)

    def test_unsupported_value_fails_at_load(self):
        with self.assertRaises(ValueError):
            Config(self._config("  image_format: gif\n"))


class IterSceneImages(unittest.TestCase):
    """The single definition of "a scene image on disk"."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_every_supported_extension(self):
        for ext in ("jpg", "jpeg", "png"):
            write_image(self.root / f"Scene-01-00{len(ext)}.{ext}", ext)
        found = {p.suffix.lower() for p in iter_scene_images(self.root, "Scene-*")}
        self.assertEqual(found, set(SCENE_IMAGE_EXTENSIONS))

    def test_ignores_webp_and_non_images(self):
        write_image(self.root / "Scene-01-001.png", "png")
        (self.root / "Scene-01-002.webp").write_bytes(b"x")
        (self.root / "Scene-01-003.txt").write_bytes(b"x")
        (self.root / "Scene-01-004.jpg").mkdir()  # a directory, not a file
        names = [p.name for p in iter_scene_images(self.root, "Scene-*")]
        self.assertEqual(names, ["Scene-01-001.png"])

    def test_pattern_separates_canonical_from_raw(self):
        write_image(self.root / "Scene-01-001.png", "png")
        write_image(self.root / "clip-Scene-002-01.png", "png")
        self.assertEqual([p.name for p in iter_scene_images(self.root, "Scene-*")],
                         ["Scene-01-001.png"])
        self.assertEqual(len(iter_scene_images(self.root, "*Scene-*")), 2)

    def test_sorted_by_name(self):
        for n in (3, 1, 2):
            write_image(self.root / f"Scene-01-00{n}.jpg", "jpg")
        self.assertEqual([p.name for p in iter_scene_images(self.root, "Scene-*")],
                         ["Scene-01-001.jpg", "Scene-01-002.jpg", "Scene-01-003.jpg"])

    def test_missing_directory_is_empty_not_an_error(self):
        self.assertEqual(iter_scene_images(self.root / "nope"), [])


class RawSceneNames(unittest.TestCase):
    """PySceneDetect's raw output name, parsed once for every caller."""

    def test_parses_supported_formats(self):
        for ext in ("jpg", "jpeg", "png"):
            self.assertEqual(parse_raw_scene_name(f"clip-Scene-007-01.{ext}"),
                             (7, ext))

    def test_case_insensitive(self):
        self.assertEqual(parse_raw_scene_name("clip-Scene-007-01.PNG"), (7, "png"))

    def test_rejects_canonical_and_unsupported(self):
        self.assertIsNone(parse_raw_scene_name("Scene-01-007.jpg"))
        self.assertIsNone(parse_raw_scene_name("clip-Scene-007-01.webp"))
        self.assertIsNone(parse_raw_scene_name("holiday.jpg"))


class RenameSceneImages(unittest.TestCase):
    """The host-side canonical rename keeps each file's own extension."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_renames_each_format_in_place(self):
        for ext in FORMATS:
            scenes = self.root / ext
            write_image(scenes / "v1" / f"clip-Scene-001-01.{ext}", ext)
            write_image(scenes / "v2" / f"other-Scene-004-01.{ext}", ext)
            self.assertEqual(rename_scene_images(scenes), 2)
            self.assertTrue((scenes / "v1" / f"Scene-01-001.{ext}").exists())
            self.assertTrue((scenes / "v2" / f"Scene-02-004.{ext}").exists())

    def test_mixed_formats_keep_their_own_extension(self):
        scenes = self.root / "mixed"
        write_image(scenes / "v1" / "clip-Scene-001-01.jpg", "jpg")
        write_image(scenes / "v1" / "clip-Scene-002-01.png", "png")
        self.assertEqual(rename_scene_images(scenes), 2)
        self.assertTrue((scenes / "v1" / "Scene-01-001.jpg").exists())
        self.assertTrue((scenes / "v1" / "Scene-01-002.png").exists())


class MergeImageFolders(unittest.TestCase):
    """The merge that also carries hand-captured images."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_canonical_images_merge_in_each_format(self):
        for ext in FORMATS:
            base = self.root / f"canonical_{ext}"
            dirs = []
            for video in (1, 2):
                d = base / f"v{video}"
                write_image(d / format_scene_name(video, 1, ext), ext)
                dirs.append(d)
            out = base / "combined"
            merge_image_folders(dirs, [], out)
            self.assertEqual(
                [p.name for p in iter_scene_images(out, "Scene-*")],
                [f"Scene-01-001.{ext}", f"Scene-02-001.{ext}"],
            )

    def test_raw_names_are_canonicalised_in_each_format(self):
        for ext in FORMATS:
            base = self.root / f"raw_{ext}"
            d = base / "v1"
            write_image(d / f"clip-Scene-003-01.{ext}", ext)
            out = base / "combined"
            merge_image_folders([d], [], out)
            self.assertTrue((out / f"Scene-01-003.{ext}").exists())

    def test_extension_is_preserved_not_rewritten(self):
        """A png must not arrive as Scene-01-001.jpg holding png bytes."""
        d = self.root / "png_source"
        write_image(d / "Scene-01-001.png", "png")
        out = self.root / "combined"
        merge_image_folders([d], [], out)
        self.assertTrue((out / "Scene-01-001.png").exists())
        self.assertFalse((out / "Scene-01-001.jpg").exists())
        self.assertEqual((out / "Scene-01-001.png").read_bytes(), PIXELS["png"])

    def test_mixed_folders_each_keep_their_format(self):
        d1 = self.root / "mixed" / "v1"
        d2 = self.root / "mixed" / "v2"
        write_image(d1 / "Scene-01-001.jpg", "jpg")
        write_image(d2 / "Scene-01-001.png", "png")
        out = self.root / "mixed" / "combined"
        merge_image_folders([d1, d2], [], out)
        self.assertEqual([p.name for p in iter_scene_images(out, "Scene-*")],
                         ["Scene-01-001.jpg", "Scene-02-001.png"])


class MergeSegmentsImageFormats(unittest.TestCase):
    """The multi-ROI merge renames segment images to canonical names."""

    def _segment(self, tmp: Path, idx: int, scenes, ext: str) -> Path:
        d = tmp / f"segment_{idx}"
        d.mkdir(parents=True)
        (d / "offset.txt").write_text("0")
        lines = ["Timecode List:,00:00:00.000\n", CSV_HEADER]
        for i, (start, end, frames) in enumerate(scenes, start=1):
            lines.append(
                f"{i},0,00:00:00.000,{start},{frames},00:00:00.000,{end},"
                f"{frames},00:00:00.000,{end - start}\n"
            )
        (d / f"segment_{idx}-Scenes.csv").write_text("".join(lines))
        for i in range(1, len(scenes) + 1):
            write_image(d / f"segment_{idx}-Scene-{i:03d}-01.{ext}", ext)
        return d

    def test_images_renamed_in_each_format(self):
        for ext in FORMATS:
            with tempfile.TemporaryDirectory() as d:
                tmp = Path(d)
                self._segment(tmp, 0, [(0.0, 5.0, 150), (5.0, 10.0, 150)], ext)
                video = tmp / "video"
                merge_segments(tmp, video / "v-Scenes.csv",
                               images_out=video, video_index=1)
                self.assertEqual(
                    [p.name for p in iter_scene_images(video, "Scene-*")],
                    [f"Scene-01-001.{ext}", f"Scene-01-002.{ext}"],
                )


class StoryboardEmbedsBothFormats(unittest.TestCase):
    """The stage that actually opens the image bytes — where webp would die."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = get_config()

    def tearDown(self):
        self.tmp.cleanup()

    def _session(self, ext: str):
        """A one-video, two-scene session: CSV, transcript and images."""
        base = self.root / ext
        images = base / "images"
        for scene in (1, 2):
            write_image(images / format_scene_name(1, scene, ext), ext)

        csv_path = base / "Scenes.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with io.open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow([self.cfg.video_column,
                             self.cfg.scene_number_column,
                             self.cfg.start_time_column])
            writer.writerow([1, 1, 1.0])
            writer.writerow([1, 2, 20.0])

        transcript = base / "transcript.txt"
        io.open(transcript, "w", encoding="utf-8", newline="\n").write(
            "[0:05.00] Alice: before the second scene\n"
            "[0:30.00] Bob: after it\n"
        )
        return csv_path, transcript, images, base / "out.docx"

    def test_images_are_embedded_in_each_format(self):
        for ext in FORMATS:
            csv_path, transcript, images, out = self._session(ext)
            generate_storyboard(str(csv_path), str(transcript), str(images), str(out))
            self.assertTrue(out.exists(), f"{ext}: no document written")
            self.assertEqual(len(Document(str(out)).inline_shapes), 2,
                             f"{ext}: both scene images should be embedded")


if __name__ == "__main__":
    unittest.main()
