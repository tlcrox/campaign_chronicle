#!/usr/bin/env python3
"""Unit tests for the clean_transcription stage."""

import json
import tempfile
import unittest
from pathlib import Path


from pipeline.common.mounts import SESSION_OUTPUT_SUBDIR
from pipeline.transcribe.clean_transcription import clean_transcript


class _CleanCfg:
    """Config stub exposing just what the clean stage reads:
    ``speaker_config_file`` (for filler phrases) and ``get(section, key, default)``
    (for the whisper.clean confidence thresholds/enable flag)."""

    def __init__(self, speaker_config_file, clean=None):
        self.speaker_config_file = str(speaker_config_file)
        self._clean = clean or {}

    def get(self, section, key, default=None):
        if section == "whisper" and key == "clean":
            return self._clean
        return default


class CleanTranscriptStage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session = Path(self.tmp.name)
        self.out = self.session / SESSION_OUTPUT_SUBDIR
        self.out.mkdir(parents=True)
        self.scf = self.session / "speaker_config.json"
        self.scf.write_text(json.dumps({"filler_phrases": ["um", "uh", "yeah"]}))

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, segments):
        p = self.out / name
        p.write_text(json.dumps({"segments": segments}))
        return p

    def _segs(self, name):
        return json.loads((self.out / name).read_text())["segments"]

    def test_filler_dropped(self):
        """Filler-phrase segments are removed; real speech and empty text remain."""
        self._write("alice.json", [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "uh"},
            {"start": 2.0, "end": 3.0, "text": "World"},
        ])
        self.assertTrue(clean_transcript(self.session, _CleanCfg(self.scf)))
        self.assertEqual([s["text"] for s in self._segs("alice.json")], ["Hello", "World"])

    def test_empty_text_left_for_merge(self):
        """Empty-text segments are NOT dropped here — the merge drops those."""
        self._write("a.json", [
            {"start": 0.0, "end": 1.0, "text": "Hi"},
            {"start": 1.0, "end": 2.0, "text": "  "},
        ])
        clean_transcript(self.session, _CleanCfg(self.scf))
        self.assertEqual(len(self._segs("a.json")), 2)

    def test_confidence_off_is_noop(self):
        """With whisper.clean.enabled absent, low-confidence segments are kept."""
        self._write("a.json", [
            {"start": 0.0, "end": 1.0, "text": "keep me",
             "no_speech_prob": 0.99, "avg_logprob": -5.0, "compression_ratio": 9.9},
        ])
        clean_transcript(self.session, _CleanCfg(self.scf))
        self.assertEqual(len(self._segs("a.json")), 1)

    def test_confidence_on_drops_low_confidence(self):
        """With the confidence pass enabled, low-confidence segments are dropped."""
        self._write("a.json", [
            {"start": 0.0, "end": 1.0, "text": "good",
             "no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 1.5},
            {"start": 1.0, "end": 2.0, "text": "noise",
             "no_speech_prob": 0.99, "avg_logprob": -5.0, "compression_ratio": 9.9},
        ])
        cfg = _CleanCfg(self.scf, clean={"enabled": True})
        clean_transcript(self.session, cfg)
        self.assertEqual([s["text"] for s in self._segs("a.json")], ["good"])

    def test_combined_files_excluded(self):
        """Combined files are left untouched."""
        self._write("session_transcript_combined.json", [
            {"start": 0.0, "end": 1.0, "text": "uh"},
        ])
        clean_transcript(self.session, _CleanCfg(self.scf))
        self.assertEqual(len(self._segs("session_transcript_combined.json")), 1)

    def test_dry_run_leaves_files_untouched(self):
        self._write("a.json", [
            {"start": 0.0, "end": 1.0, "text": "Hi"},
            {"start": 1.0, "end": 2.0, "text": "uh"},
        ])
        clean_transcript(self.session, _CleanCfg(self.scf), dry_run=True)
        self.assertEqual(len(self._segs("a.json")), 2)

    def test_no_transcripts_returns_false(self):
        self.assertFalse(clean_transcript(self.session, _CleanCfg(self.scf)))


if __name__ == "__main__":
    unittest.main()
