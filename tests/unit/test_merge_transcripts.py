#!/usr/bin/env python3
"""Unit tests for the merge_transcripts stage's source adapter (build_merge_sources)."""

import json
import tempfile
import unittest
from pathlib import Path


from cc_stages.merge_transcripts import build_merge_sources


class BuildMergeSources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, segments):
        p = self.root / name
        p.write_text(json.dumps({"segments": segments}))
        return p

    def test_audacity_sources_offset_zero_no_speaker_override(self):
        """Workflow A (filename token matches): parallel tracks share the timeline
        (offset 0) and carry no speaker override — names came from map_speakers."""
        f1 = self._write("1-nedoking.json", [{"start": 0.0, "end": 1.0, "speaker": "NedoKing", "text": "Hi"}])
        f2 = self._write("2-thirty.json", [{"start": 1.0, "end": 2.0, "speaker": "Thirty", "text": "Yo"}])
        sources = build_merge_sources(self.root, [f1, f2], {"nedoking": "NedoKing", "thirty": "Thirty"})
        self.assertEqual([s["offset"] for s in sources], [0.0, 0.0])
        self.assertEqual([s["speaker"] for s in sources], [None, None])
        self.assertEqual([s["name"] for s in sources], ["1-nedoking.json", "2-thirty.json"])
        self.assertEqual(sources[0]["segments"][0]["text"], "Hi")

    def test_single_video_no_offset(self):
        """Workflow B with a single transcript: no offsets (nothing to shift)."""
        f = self._write("video.json", [{"start": 0.0, "end": 1.0, "speaker": "Alice", "text": "Hello"}])
        sources = build_merge_sources(self.root, [f], filename_mapping={})
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["offset"], 0.0)
        self.assertIsNone(sources[0]["speaker"])

    def test_bad_file_skipped(self):
        """Unloadable or segment-less files are skipped, not fatal."""
        good = self._write("video.json", [{"start": 0.0, "end": 1.0, "speaker": "Alice", "text": "Hi"}])
        bad = self.root / "broken.json"
        bad.write_text("{not json")
        nose = self._write("nosegs.json", {})  # overwrites with no 'segments'
        (self.root / "nosegs.json").write_text(json.dumps({"language": "en"}))
        sources = build_merge_sources(self.root, [good, bad, self.root / "nosegs.json"], {})
        self.assertEqual([s["name"] for s in sources], ["video.json"])


if __name__ == "__main__":
    unittest.main()
