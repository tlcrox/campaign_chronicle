#!/usr/bin/env python3
"""
Unit tests for roi.RoiFile.

The ROI file format is hierarchical / video-keyed ONLY (matching the real
TestWhisper/*/roi_history.json samples). The legacy "flat" layout is rejected.

Run any of these (all work):
    python3 test_roi.py                          # from scripts/pipeline/scenes/
    python3 -m unittest pipeline.scenes.test_roi # from scripts/
"""

import json
import tempfile
import unittest
from pathlib import Path


from pipeline.scenes.roi import (
    END_OF_VIDEO,
    RoiEntry,
    RoiFile,
    RoiFileNotFoundError,
    RoiParseError,
    RoiSegment,
    RoiVideoNotFoundError,
)

# Real-shaped hierarchical fixture (mirrors MultiROI/roi_history.json).
HIER = {
    "2026-04-20 15-39-46.mkv": {
        "_metadata": {"fps": 60.0},
        "00:00:00": {"frame": 0, "roi": "487 188 1368 811"},
        "00:05:16": {"frame": 18991, "roi": "705 91 827 127"},
        "00:05:17": {"frame": 19075, "roi": "485 187 1367 812"},
    },
    "2026-04-20 18-47-12.mkv": {
        "_metadata": {"fps": 60.0},
        "00:00:00": {"frame": 0, "roi": "1464 68 1564 119"},
    },
}

VIDEO1 = "2026-04-20 15-39-46.mkv"
VIDEO2 = "2026-04-20 18-47-12.mkv"

# Top-level timestamp keys = the rejected flat layout.
FLAT = {
    "00:00:00": {"frame": 0, "roi": "529 249 1288 755"},
    "00:05:00": {"frame": 7500, "roi": "400 200 1400 800"},
}

# The shipped example lives in the repo's config/ dir.
# parents[4] = tests/unit/pipeline/scenes -> repo root.
EXAMPLE_FILE = (
    Path(__file__).resolve().parents[4] / "config" / "roi_config.example.json"
)


class HierarchicalFormat(unittest.TestCase):
    def setUp(self):
        self.roi = RoiFile.from_dict(HIER)

    def test_videos_sorted(self):
        self.assertEqual(self.roi.videos, [VIDEO1, VIDEO2])

    def test_metadata_and_fps(self):
        self.assertEqual(self.roi.fps(VIDEO1), 60.0)
        self.assertEqual(self.roi.fps(VIDEO2), 60.0)
        self.assertEqual(self.roi.metadata(VIDEO1), {"fps": 60.0})

    def test_metadata_excluded_from_entries(self):
        entries = self.roi.entries(VIDEO1)
        self.assertEqual(len(entries), 3)
        self.assertTrue(all(e.timestamp != "_metadata" for e in entries))

    def test_entries(self):
        entries = self.roi.entries(VIDEO1)
        self.assertEqual(entries[0], RoiEntry("00:00:00", 0, "487 188 1368 811", 0))
        self.assertEqual([e.start for e in entries], [0, 316, 317])

    def test_segments_per_video(self):
        segs = self.roi.segments(VIDEO1)
        self.assertEqual([s.start for s in segs], [0, 316, 317])
        self.assertEqual([s.end for s in segs], [316, 317, END_OF_VIDEO])
        self.assertTrue(segs[-1].is_final)

    def test_single_entry_video_is_final(self):
        segs = self.roi.segments(VIDEO2)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0], RoiSegment(0, END_OF_VIDEO, "1464 68 1564 119", "ROI at 00:00:00", 0))

    def test_roi_at(self):
        self.assertEqual(self.roi.roi_at(0, VIDEO1), "487 188 1368 811")
        self.assertEqual(self.roi.roi_at(315, VIDEO1), "487 188 1368 811")
        self.assertEqual(self.roi.roi_at(316, VIDEO1), "705 91 827 127")
        self.assertEqual(self.roi.roi_at(317, VIDEO1), "485 187 1367 812")
        self.assertEqual(self.roi.roi_at(99999, VIDEO1), "485 187 1367 812")

    def test_pipe_render(self):
        self.assertEqual(
            self.roi.segments(VIDEO1)[0].as_pipe(),
            "0|316|487 188 1368 811|ROI at 00:00:00",
        )

    def test_missing_video_raises(self):
        with self.assertRaises(RoiVideoNotFoundError):
            self.roi.segments("nope.mkv")

    def test_video_required(self):
        with self.assertRaises(RoiVideoNotFoundError):
            self.roi.segments(None)

    def test_iter_all_segments(self):
        pairs = list(self.roi.iter_all_segments())
        self.assertEqual(len(pairs), 4)          # 3 in video1 + 1 in video2
        self.assertEqual(pairs[0][0], VIDEO1)
        self.assertEqual(pairs[-1][0], VIDEO2)


class FlatFormatRejected(unittest.TestCase):
    """Only the nested layout is accepted; flat top-level timestamp keys raise."""

    def test_flat_dict_raises(self):
        with self.assertRaises(RoiParseError):
            RoiFile.from_dict(FLAT)

    def test_single_flat_entry_raises(self):
        with self.assertRaises(RoiParseError):
            RoiFile.from_dict({"00:00:00": {"frame": 0, "roi": "1 2 3 4"}})

    def test_flat_file_on_disk_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "roi_history.json"
            p.write_text(json.dumps(FLAT))
            with self.assertRaises(RoiParseError):
                RoiFile.load(p)

    def test_only_metadata_raises(self):
        with self.assertRaises(RoiParseError):
            RoiFile.from_dict({"_metadata": {"fps": 60.0}})

    def test_empty_dict_raises(self):
        with self.assertRaises(RoiParseError):
            RoiFile.from_dict({})

    def test_video_value_not_object_raises(self):
        with self.assertRaises(RoiParseError):
            RoiFile.from_dict({"video.mkv": "not a dict"})


class Coords(unittest.TestCase):
    def test_coords_tuple(self):
        e = RoiEntry("00:00:00", 0, "529 249 1288 755")
        self.assertEqual(e.coords, (529, 249, 1288, 755))

    def test_bad_coords(self):
        with self.assertRaises(RoiParseError):
            RoiEntry("00:00:00", 0, "1 2 3").coords
        with self.assertRaises(RoiParseError):
            RoiEntry("00:00:00", 0, "a b c d").coords


class LoadingAndErrors(unittest.TestCase):
    def test_load_missing_file(self):
        with self.assertRaises(RoiFileNotFoundError):
            RoiFile.load("/no/such/file.json")

    def test_load_bad_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write("{ not valid json ")
            name = fh.name
        try:
            with self.assertRaises(RoiParseError):
                RoiFile.load(name)
        finally:
            Path(name).unlink()

    def test_find_in_search_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "roi_history.json"
            p.write_text(json.dumps(HIER))
            roi = RoiFile.find("roi_history.json", [Path("/nope"), Path(d)])
            self.assertEqual(roi.videos, [VIDEO1, VIDEO2])
            self.assertEqual(RoiFile.resolve_path("roi_history.json", [Path(d)]), p)

    def test_find_not_found(self):
        with self.assertRaises(RoiFileNotFoundError):
            RoiFile.find("missing.json", [Path("/nope")])
        self.assertIsNone(RoiFile.resolve_path("missing.json", [Path("/nope")]))

    def test_top_level_not_object(self):
        with self.assertRaises(RoiParseError):
            RoiFile.from_dict([1, 2, 3])

    def test_empty_segments_raises(self):
        # Constructs fine (video block present) but has no timestamp entries.
        roi = RoiFile.from_dict({"video.mkv": {"_metadata": {"fps": 60}}})
        with self.assertRaises(RoiParseError):
            roi.segments("video.mkv")


class Validation(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(RoiFile.from_dict(HIER).validate(), [])

    def test_detects_missing_roi(self):
        bad = {"video.mkv": {"00:00:00": {"frame": 0}}}
        problems = RoiFile.from_dict(bad).validate()
        self.assertTrue(any("missing 'roi'" in p for p in problems))

    def test_detects_bad_coords(self):
        bad = {"video.mkv": {"00:00:00": {"roi": "1 2 3"}}}
        problems = RoiFile.from_dict(bad).validate()
        self.assertTrue(any("4 values" in p for p in problems))


class LoadFromDisk(unittest.TestCase):
    """Exercises the real file I/O path (load -> parse -> segments) with a
    temp file, so it is deterministic and needs no external session data."""

    def test_load_and_segments(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "roi_history.json"
            p.write_text(json.dumps(HIER))
            roi = RoiFile.load(p)
            self.assertEqual(roi.videos, [VIDEO1, VIDEO2])
            self.assertEqual(roi.validate(), [])
            for video in roi.videos:
                segs = roi.segments(video)
                self.assertGreater(len(segs), 0)
                self.assertEqual(segs[0].start, 0)
                self.assertTrue(segs[-1].is_final)


class ExampleFileStaysValid(unittest.TestCase):
    """The shipped example must remain a valid hierarchical file."""

    def test_example_loads_and_validates(self):
        self.assertTrue(EXAMPLE_FILE.exists(), f"missing {EXAMPLE_FILE}")
        roi = RoiFile.load(EXAMPLE_FILE)
        self.assertGreater(len(roi.videos), 0)
        self.assertEqual(roi.validate(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
