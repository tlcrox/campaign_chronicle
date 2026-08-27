#!/usr/bin/env python3
"""Tests for combine_scenes.py scene-merge functions."""

import tempfile
import unittest
from pathlib import Path



class _SceneCfg:
    """Minimal config stand-in exposing the CSV column names the scene merge reads."""
    scene_number_column = "Scene Number"
    start_time_column = "Start Time (seconds)"
    end_time_column = "End Time (seconds)"
    video_column = "Video"


class TestSceneMerge2PartKey(unittest.TestCase):
    """Scene merge keys on (Video, per-video Scene Number), not a global number."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _csv(self, sub, rows):
        d = self.root / sub
        d.mkdir(parents=True, exist_ok=True)
        p = d / "x-Scenes.csv"
        lines = ["Scene Number,Start Time (seconds),End Time (seconds)"]
        lines += [f"{n},{s},{e}" for n, s, e in rows]
        p.write_text("\n".join(lines) + "\n")
        return p

    def test_csv_keeps_per_video_scene_number_and_adds_video_column(self):
        from pipeline.merge.combine_scenes import merge_scene_csvs, HAS_PANDAS
        if not HAS_PANDAS:
            self.skipTest("pandas not available")
        c1 = self._csv("v1", [(1, 0.0, 4.0), (2, 5.0, 9.0)])
        c2 = self._csv("v2", [(1, 0.0, 3.0), (2, 4.0, 7.0)])
        mdf, _, _ = merge_scene_csvs([c1, c2], _SceneCfg(), {c1: 10.0, c2: 8.0})
        self.assertEqual(list(mdf["Video"]), [1, 1, 2, 2])
        self.assertEqual(list(mdf["Scene Number"]), [1, 2, 1, 2])          # per-video, NOT 1..4
        self.assertEqual(list(mdf["Start Time (seconds)"]), [0.0, 5.0, 10.0, 14.0])  # session timeline

    def test_image_names_restart_per_video(self):
        from pipeline.merge.combine_scenes import merge_image_folders
        from pipeline.common.scenes import format_scene_name
        v1, v2 = self.root / "v1", self.root / "v2"
        for d, vi in [(v1, 1), (v2, 2)]:
            d.mkdir(parents=True, exist_ok=True)
            for s in (1, 2):
                (d / format_scene_name(vi, s)).write_bytes(b"")
        out = self.root / "out"
        merge_image_folders([v1, v2], [], out)
        names = sorted(p.name for p in out.glob("Scene-*.jpg"))
        self.assertEqual(
            names,
            ["Scene-01-001.jpg", "Scene-01-002.jpg", "Scene-02-001.jpg", "Scene-02-002.jpg"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
