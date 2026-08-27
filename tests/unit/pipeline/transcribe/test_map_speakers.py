#!/usr/bin/env python3
"""
Tests for pipeline.transcribe.map_speakers.apply_speaker_mapping — specifically
the session_mappings LOOKUP LADDER.

Diarization assigns SPEAKER_XX IDs per FILE, so a session with several recordings
can need a different mapping per file. apply_speaker_mapping therefore resolves
session_mappings most-specific-first:

    <session_key>/<transcript_name>   per-file
    <session_key>                     per-session
    <leaf name> / <abs path>          legacy back-compat
    global_mapping                    fallback

session_key is the session's path relative to source_dir, rendered with forward
slashes (.as_posix()) so the "Weeks/Week 13" style keys match on Windows too.

These exercise the library function directly; the stage wrapper is covered by
tests/unit/test_apply_speaker_mapping.py.

Run from scripts/:
    python3 -m unittest pipeline.test_map_speakers -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.transcribe.map_speakers import apply_speaker_mapping  # noqa: E402


def _transcript():
    """Two mappable speakers + one that is in no mapping (must be left as-is)."""
    return {
        "segments": [
            {"speaker": "SPEAKER_00", "text": "a",
             "words": [{"word": "a", "speaker": "SPEAKER_00"}]},
            {"speaker": "SPEAKER_01", "text": "b",
             "words": [{"word": "b", "speaker": "SPEAKER_01"}]},
            {"speaker": "SPEAKER_99", "text": "c",
             "words": [{"word": "c", "speaker": "SPEAKER_99"}]},
        ]
    }


SPEAKER_CONFIG = {
    "global_mapping": {"SPEAKER_00": "GZero", "SPEAKER_01": "GOne"},
    "session_mappings": {
        "Weeks/Week 13": {"SPEAKER_01": "SessionOne"},
        "Weeks/Week 13/Week13_a": {"SPEAKER_00": "AZero", "SPEAKER_01": "AOne"},
        "Weeks/Week 13/Week13_b": {"SPEAKER_01": "BOne"},
        "Week 13": {"SPEAKER_01": "LeafOne"},   # legacy leaf-name key
    },
}


class _KeyedConfig:
    """Config stub that resolves session identity relative to source_dir, exactly
    like the real Config.session_key (path relative to source, else leaf name)."""

    def __init__(self, source_dir):
        self._source = Path(source_dir).resolve()

    def session_key(self, session_dir):
        p = Path(session_dir).resolve()
        try:
            return p.relative_to(self._source)
        except ValueError:
            return Path(p.name)


class _NoSessionKeyConfig:
    """Legacy stub with no session_key -> forces the leaf-name fallback branch."""


def _segment_speakers(res):
    return [s["speaker"] for s in res["segments"]]


def _word_speakers(res):
    return [s["words"][0]["speaker"] for s in res["segments"]]


class SessionMappingLookupLadder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name)
        self.cfg_path = self.src / "speaker_config.json"
        self.cfg_path.write_text(json.dumps(SPEAKER_CONFIG), encoding="utf-8")
        self.cfg = _KeyedConfig(self.src)
        self.session13 = self.src / "Weeks" / "Week 13"
        self.session99 = self.src / "Weeks" / "Week 99"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, session_dir, transcript_name, config=None):
        return apply_speaker_mapping(
            _transcript(), session_dir, config or self.cfg,
            speaker_config_file=self.cfg_path, transcript_name=transcript_name,
        )

    def test_per_file_key_wins_over_session_and_global(self):
        # "Weeks/Week 13/Week13_a" overrides BOTH SPEAKER_00 and SPEAKER_01,
        # beating the per-session ("SessionOne") and global entries.
        res = self._run(self.session13, "Week13_a")
        self.assertEqual(_segment_speakers(res), ["AZero", "AOne", "SPEAKER_99"])
        # word-level tracks the segment mapping.
        self.assertEqual(_word_speakers(res), ["AZero", "AOne", "SPEAKER_99"])

    def test_falls_back_to_session_key_when_no_per_file(self):
        # "Week13_c" has no per-file entry -> per-session "Weeks/Week 13" applies
        # (maps only SPEAKER_01); SPEAKER_00 falls through to global.
        res = self._run(self.session13, "Week13_c")
        self.assertEqual(_segment_speakers(res), ["GZero", "SessionOne", "SPEAKER_99"])

    def test_two_files_in_one_session_map_independently(self):
        # THE requirement: same session, different files, different mappings.
        a = self._run(self.session13, "Week13_a")
        b = self._run(self.session13, "Week13_b")
        self.assertEqual(_segment_speakers(a)[1], "AOne")
        self.assertEqual(_segment_speakers(b)[1], "BOne")

    def test_global_when_session_not_in_mappings(self):
        res = self._run(self.session99, "whatever")
        self.assertEqual(_segment_speakers(res), ["GZero", "GOne", "SPEAKER_99"])

    def test_forward_slash_keys_match_regardless_of_os_separator(self):
        # The config keys use "/". session_key is rendered with .as_posix(), so a
        # nested "Weeks/Week 13/..." key matches even on Windows, where str(Path)
        # would be "Weeks\\Week 13\\..." and miss. A successful match proves it.
        res = self._run(self.session13, "Week13_b")
        self.assertEqual(_segment_speakers(res)[1], "BOne")

    def test_unmapped_speaker_is_left_untouched(self):
        res = self._run(self.session13, "Week13_a")
        self.assertEqual(res["segments"][2]["speaker"], "SPEAKER_99")
        self.assertEqual(res["segments"][2]["words"][0]["speaker"], "SPEAKER_99")

    def test_legacy_leaf_name_fallback_when_config_lacks_session_key(self):
        # A config without session_key() -> the lookup falls back to the leaf name
        # ("Week 13"), so pre-session_key configs keep working.
        res = apply_speaker_mapping(
            _transcript(), self.session13, _NoSessionKeyConfig(),
            speaker_config_file=self.cfg_path, transcript_name="Week13_a",
        )
        self.assertEqual(_segment_speakers(res)[1], "LeafOne")


if __name__ == "__main__":
    unittest.main()
