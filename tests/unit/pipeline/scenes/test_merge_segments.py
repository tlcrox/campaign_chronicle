#!/usr/bin/env python3
"""
Tests for pipeline.scenes.merge_segments and extract_segments.

Run from scripts/:
    python3 -m unittest pipeline.scenes.test_merge_segments -v
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path


from pipeline.scenes.merge_segments import merge_segments, MergeError, OUTPUT_FIELDS  # noqa: E402
from pipeline.scenes import extract_segments  # noqa: E402

# A scenedetect-style CSV (with a preamble line before the real header).
# Real PySceneDetect list-scenes CSV layout (a "Timecode List:" preamble line is
# also written before this header; the reader skips it).
CSV_HEADER = (
    "Scene Number,Start Frame,Start Timecode,Start Time (seconds),"
    "End Frame,End Timecode,End Time (seconds),"
    "Length (frames),Length (timecode),Length (seconds)\n"
)


def write_segment(temp_dir: Path, idx: int, offset, scenes):
    """Create temp_dir/segment_<idx>/ with offset.txt and a scenes CSV.

    scenes: list of (start_rel_sec, end_rel_sec, length_frames).
    """
    d = temp_dir / f"segment_{idx}"
    d.mkdir(parents=True)
    if offset is not None:
        (d / "offset.txt").write_text(str(offset))
    lines = ["Timecode List:,00:00:00.000\n", CSV_HEADER]
    for i, (s, e, lf) in enumerate(scenes, start=1):
        # Start Frame=0, End Frame=lf so the End-Start fallback also yields lf.
        lines.append(f"{i},0,00:00:00.000,{s},{lf},00:00:00.000,{e},{lf},00:00:00.000,{e - s}\n")
    (d / f"segment_{idx}-Scenes.csv").write_text("".join(lines))
    return d


def read_rows(path: Path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class MergeSegments(unittest.TestCase):
    def test_offsets_renumber_timecode_frames(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            write_segment(tmp, 0, 0, [(0.0, 5.0, 150), (5.0, 10.0, 150)])
            write_segment(tmp, 1, 250, [(0.0, 3.0, 90)])
            out = tmp / "merged.csv"
            n = merge_segments(tmp, out)

            self.assertEqual(n, 3)
            rows = read_rows(out)
            self.assertEqual([r["Scene Number"] for r in rows], ["1", "2", "3"])
            self.assertEqual([r["Start Time (seconds)"] for r in rows], ["0.0", "5.0", "250.0"])
            self.assertEqual([r["End Time (seconds)"] for r in rows], ["5.0", "10.0", "253.0"])
            self.assertEqual(rows[2]["Duration (seconds)"], "3.0")
            # absolute Start Timecode (offset applied)
            self.assertEqual([r["Start Timecode"] for r in rows],
                             ["00:00:00.000", "00:00:05.000", "00:04:10.000"])
            # per-scene frame length, unchanged by offset
            self.assertEqual([r["Length (frames)"] for r in rows], ["150", "150", "90"])

    def test_segment_order_numeric(self):
        # segment_10 must sort after segment_2, not lexically before it.
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            write_segment(tmp, 2, 100, [(0.0, 1.0, 30)])
            write_segment(tmp, 10, 900, [(0.0, 1.0, 30)])
            out = tmp / "merged.csv"
            merge_segments(tmp, out)
            rows = read_rows(out)
            self.assertEqual([r["Start Time (seconds)"] for r in rows], ["100.0", "900.0"])

    def test_frame_length_fallback_from_frames(self):
        # No 'Length (frames)' value -> fall back to End Frame - Start Frame.
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            seg = tmp / "segment_0"
            seg.mkdir()
            (seg / "offset.txt").write_text("0")
            (seg / "segment_0-Scenes.csv").write_text(
                CSV_HEADER + "1,10,00:00:00.000,0.0,250,00:00:00.000,5.0,,,5.0\n"
            )
            out = tmp / "merged.csv"
            merge_segments(tmp, out)
            self.assertEqual(read_rows(out)[0]["Length (frames)"], "240")  # 250 - 10

    def test_missing_offset_defaults_zero(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            write_segment(tmp, 0, None, [(2.0, 4.0, 60)])  # no offset.txt
            out = tmp / "merged.csv"
            merge_segments(tmp, out)
            rows = read_rows(out)
            self.assertEqual(rows[0]["Start Time (seconds)"], "2.0")

    def test_missing_csv_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            write_segment(tmp, 0, 0, [(0.0, 1.0, 30)])
            (tmp / "segment_1").mkdir()  # empty: no CSV
            out = tmp / "merged.csv"
            n = merge_segments(tmp, out)
            self.assertEqual(n, 1)

    def test_no_segments_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(MergeError):
                merge_segments(Path(d), Path(d) / "out.csv")

    def test_output_fields(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            write_segment(tmp, 0, 0, [(0.0, 1.0, 30)])
            out = tmp / "merged.csv"
            merge_segments(tmp, out)
            with open(out, newline="") as fh:
                header = fh.readline().strip().split(",")
            self.assertEqual(header, OUTPUT_FIELDS)


class MergeSegmentsImages(unittest.TestCase):
    """merge_segments also emits the per-scene images, renamed to canonical
    Scene-{video:02d}-{scene:03d} using the SAME scene number as the CSV."""

    def _seg(self, tmp, idx, offset, scenes):
        d = write_segment(tmp, idx, offset, scenes)
        # one raw PySceneDetect image per scene: segment_<idx>-Scene-<n>-01.jpg
        for i in range(1, len(scenes) + 1):
            (d / f"segment_{idx}-Scene-{i:03d}-01.jpg").write_bytes(b"x")
        return d

    def test_images_aligned_to_csv_scene_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._seg(tmp, 0, 0, [(0.0, 5.0, 150), (5.0, 10.0, 150)])
            self._seg(tmp, 1, 100, [(0.0, 4.0, 120)])
            video = tmp / "video"
            n = merge_segments(tmp, video / "video-Scenes.csv", images_out=video, video_index=1)
            self.assertEqual(n, 3)
            # CSV scene N <-> Scene-01-00N.jpg (renumbered across segments)
            self.assertEqual(sorted(p.name for p in video.glob("*.jpg")),
                             ["Scene-01-001.jpg", "Scene-01-002.jpg", "Scene-01-003.jpg"])
            rows = read_rows(video / "video-Scenes.csv")
            self.assertEqual([r["Scene Number"] for r in rows], ["1", "2", "3"])

    def test_video_index_in_name(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._seg(tmp, 0, 0, [(0.0, 1.0, 30)])
            video = tmp / "video"
            merge_segments(tmp, video / "v.csv", images_out=video, video_index=3)
            self.assertTrue((video / "Scene-03-001.jpg").exists())

    def test_no_images_out_is_csv_only(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._seg(tmp, 0, 0, [(0.0, 1.0, 30)])
            out = tmp / "merged.csv"
            merge_segments(tmp, out)  # images_out=None: no image copies
            self.assertTrue(out.exists())
            self.assertEqual(list(out.parent.glob("*.jpg")), [])


class ExtractSegments(unittest.TestCase):
    HIER = {
        "video1.mkv": {
            "_metadata": {"fps": 60.0},
            "00:00:00": {"frame": 0, "roi": "479 185 1374 817"},
            "00:04:10": {"frame": 15000, "roi": "909 493 1127 743"},
        }
    }

    def test_extract_pipe_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "roi.json"
            p.write_text(json.dumps(self.HIER))
            lines = extract_segments.extract(str(p), "video1.mkv")
            self.assertEqual(
                lines,
                [
                    "0|250|479 185 1374 817|ROI at 00:00:00",
                    "250|-1|909 493 1127 743|ROI at 00:04:10",
                ],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
