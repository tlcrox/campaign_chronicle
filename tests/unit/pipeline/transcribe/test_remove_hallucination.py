#!/usr/bin/env python3
"""
Tests for pipeline.transcribe.clean.

Run from scripts/:
    python3 -m unittest pipeline.transcribe.test_clean -v
"""

import json
import tempfile
import unittest
from pathlib import Path


from pipeline.transcribe.remove_hallucination import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    clean_dir,
    clean_segments,
    thresholds_from_config,
)

# A keeper plus three that each violate exactly one threshold, plus an empty one.
SEGMENTS = [
    {"no_speech_prob": 0.1, "avg_logprob": -0.5, "compression_ratio": 1.5, "text": "real speech"},
    {"no_speech_prob": 0.9, "avg_logprob": -0.5, "compression_ratio": 1.5, "text": "silence"},
    {"no_speech_prob": 0.1, "avg_logprob": -2.0, "compression_ratio": 1.5, "text": "low conf"},
    {"no_speech_prob": 0.1, "avg_logprob": -0.5, "compression_ratio": 9.0, "text": "looped loop"},
    {"no_speech_prob": 0.1, "avg_logprob": -0.5, "compression_ratio": 1.5, "text": "   "},
]


class FakeConfig:
    def __init__(self, clean): self._clean = clean
    def get(self, section, key, default=None):
        if section == "whisper" and key == "clean":
            return self._clean
        return default


class CleanSegments(unittest.TestCase):
    def test_keeps_only_valid(self):
        kept = clean_segments(SEGMENTS, DEFAULT_THRESHOLDS)
        self.assertEqual([s["text"] for s in kept], ["real speech"])

    def test_default_when_no_thresholds(self):
        kept = clean_segments(SEGMENTS)  # uses DEFAULT_THRESHOLDS
        self.assertEqual(len(kept), 1)

    def test_threshold_override_loosens(self):
        loose = {"no_speech_max": 1.0, "logprob_min": -10.0, "compression_max": 100.0}
        kept = clean_segments(SEGMENTS, loose)
        self.assertEqual(len(kept), 4)  # all non-empty-text segments survive


class ThresholdsFromConfig(unittest.TestCase):
    def test_reads_config_values(self):
        cfg = FakeConfig({"no_speech_max": 0.3, "logprob_min": -0.2, "compression_max": 2.0})
        t = thresholds_from_config(cfg)
        self.assertEqual(t, {"no_speech_max": 0.3, "logprob_min": -0.2, "compression_max": 2.0})

    def test_partial_config_merges_defaults(self):
        cfg = FakeConfig({"no_speech_max": 0.4})
        t = thresholds_from_config(cfg)
        self.assertEqual(t["no_speech_max"], 0.4)
        self.assertEqual(t["logprob_min"], DEFAULT_THRESHOLDS["logprob_min"])

    def test_empty_config_uses_defaults(self):
        self.assertEqual(thresholds_from_config(FakeConfig({})), DEFAULT_THRESHOLDS)


class CleanDir(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            ind, outd = Path(d) / "in", Path(d) / "out"
            ind.mkdir()
            (ind / "a.json").write_text(json.dumps({"segments": SEGMENTS, "text": "x"}))
            totals = clean_dir(ind, outd, DEFAULT_THRESHOLDS)
            self.assertEqual(totals, {"in": 5, "out": 1, "files": 1})
            out = json.loads((outd / "a.json").read_text())
            self.assertEqual(len(out["segments"]), 1)
            self.assertEqual(out["text"], "real speech")

    def test_missing_dir_raises(self):
        with self.assertRaises(NotADirectoryError):
            clean_dir("/no/such/dir", "/tmp/out")


if __name__ == "__main__":
    unittest.main(verbosity=2)
