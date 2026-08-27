#!/usr/bin/env python3
"""
Tests for pipeline.common.scenes — the canonical scene-image naming helpers
(Scene-{video:02d}-{scene:03d}.jpg) and the host-side rename that canonicalises
PySceneDetect's raw output after the docker run.

Run from scripts/:
    python3 -m unittest pipeline.common.test_scenes -v
"""

import tempfile
import unittest
from pathlib import Path


from pipeline.common.scenes import (  # noqa
    find_candidate_manual_scenes,  # noqa: E402
    format_scene_name,
    parse_scene_name,
    rename_scene_images,
)


class FormatParse(unittest.TestCase):
    def test_format(self):
        self.assertEqual(format_scene_name(1, 1), "Scene-01-001.jpg")
        self.assertEqual(format_scene_name(2, 42, "png"), "Scene-02-042.png")
        self.assertEqual(format_scene_name(12, 345), "Scene-12-345.jpg")

    def test_parse_canonical(self):
        self.assertEqual(parse_scene_name("Scene-01-001.jpg"), (1, 1))
        self.assertEqual(parse_scene_name("Scene-02-042.png"), (2, 42))

    def test_parse_rejects_raw_and_junk(self):
        # Raw PySceneDetect / prefixed names are NOT canonical.
        self.assertIsNone(parse_scene_name("video-Scene-001-01.jpg"))
        self.assertIsNone(parse_scene_name("Scene-1-1.jpg"))     # wrong widths
        self.assertIsNone(parse_scene_name("frame.jpg"))

    def test_round_trip(self):
        self.assertEqual(parse_scene_name(format_scene_name(3, 7)), (3, 7))


class RenameSceneImages(unittest.TestCase):
    def _raw(self, subdir, stem, scene, img=1, ext="jpg"):
        (subdir / f"{stem}-Scene-{scene:03d}-{img:02d}.{ext}").write_bytes(b"x")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.scenes = Path(self.tmp.name)
        # Two per-video subdirs (sorted -> video 01, 02), raw PySceneDetect names.
        self.v1 = self.scenes / "2026-a"
        self.v2 = self.scenes / "2026-b"
        self.v1.mkdir()
        self.v2.mkdir()
        self._raw(self.v1, "2026-a", 1)
        self._raw(self.v1, "2026-a", 2)
        self._raw(self.v2, "2026-b", 1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_renames_to_canonical_with_video_index(self):
        n = rename_scene_images(self.scenes)
        self.assertEqual(n, 3)
        self.assertEqual(sorted(p.name for p in self.v1.glob("*.jpg")),
                         ["Scene-01-001.jpg", "Scene-01-002.jpg"])
        self.assertEqual([p.name for p in self.v2.glob("*.jpg")],
                         ["Scene-02-001.jpg"])

    def test_idempotent(self):
        rename_scene_images(self.scenes)
        # Second pass: everything is already canonical -> nothing renamed.
        self.assertEqual(rename_scene_images(self.scenes), 0)

    def test_leaves_canonical_untouched(self):
        (self.v1 / "Scene-01-009.jpg").write_bytes(b"x")   # already canonical
        rename_scene_images(self.scenes)
        self.assertTrue((self.v1 / "Scene-01-009.jpg").exists())

    def test_preserves_extension(self):
        self._raw(self.v2, "2026-b", 2, ext="png")
        rename_scene_images(self.scenes)
        self.assertTrue((self.v2 / "Scene-02-002.png").exists())

    def test_missing_dir_returns_zero(self):
        self.assertEqual(rename_scene_images(self.scenes / "nope"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class CandidateManualScenes(unittest.TestCase):
    """Guard against silently auto-detecting over hand-captured scenes.

    orchestrate's manual-scenes gate keys on the configured folder NAME existing.
    When a user captures scenes into a differently-named folder (CaptureScreens
    writes scenes_output/ by default; the pipeline expects screens/), the gate
    misses them, PySceneDetect runs on the video, and the storyboard is built
    from the wrong scenes — output that looks right and is not.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session = Path(self.tmp.name) / "Week 14"
        self.session.mkdir()
        self.addCleanup(self.tmp.cleanup)

    def _capture(self, folder, csv="Scenes.csv"):
        d = self.session / folder / "Week14"
        d.mkdir(parents=True)
        (d / csv).write_text("Scene Number\n1\n")
        return d

    def test_detects_a_misnamed_capture_folder(self):
        self._capture("scenes_output")
        found = find_candidate_manual_scenes(self.session, "Scenes.csv", exclude="screens")
        self.assertEqual([f.name for f in found], ["scenes_output"])

    def test_no_false_alarm_when_correctly_named(self):
        self._capture("screens")
        self.assertEqual(
            find_candidate_manual_scenes(self.session, "Scenes.csv", exclude="screens"), [])

    def test_generated_output_never_matches(self):
        """cc_output/ holds auto-detected results named <stem>-Scenes.csv."""
        d = self.session / "cc_output" / "scenes_output" / "Week14"
        d.mkdir(parents=True)
        (d / "Week14-Scenes.csv").write_text("x")
        self.assertEqual(
            find_candidate_manual_scenes(self.session, "Scenes.csv", exclude="screens"), [])

    def test_empty_session_is_fine(self):
        self.assertEqual(find_candidate_manual_scenes(self.session, "Scenes.csv"), [])

    def test_missing_session_dir_is_fine(self):
        self.assertEqual(
            find_candidate_manual_scenes(self.session / "nope", "Scenes.csv"), [])

    def test_honours_a_custom_csv_name(self):
        self._capture("captures", csv="MyScenes.csv")
        self.assertEqual(
            [f.name for f in find_candidate_manual_scenes(self.session, "MyScenes.csv")],
            ["captures"])
