#!/usr/bin/env python3
"""Tests for combine_transcripts.py merge functions."""

import json
import tempfile
import unittest
from pathlib import Path


from pipeline.merge.combine_transcripts import (
    merge_transcripts,
    merge_transcripts_audacity,
    is_filler,
)


class TestMergeTranscriptsCore(unittest.TestCase):
    """Test the shared merge_transcripts(sources) core directly."""

    def test_parallel_sources_offset_zero_interleave(self):
        """Parallel tracks (offset 0, overlapping times) interleave by timestamp,
        each source's speaker override stamped on its segments."""
        sources = [
            {"name": "Alice", "offset": 0.0, "speaker": "Alice", "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello"},
                {"start": 3.0, "end": 4.0, "text": "Bye"},
            ]},
            {"name": "Bob", "offset": 0.0, "speaker": "Bob", "segments": [
                {"start": 1.0, "end": 2.0, "text": "Hi"},
                {"start": 2.0, "end": 3.0, "text": "Yo"},
            ]},
        ]
        result = merge_transcripts(sources)
        self.assertEqual([(s["speaker"], s["text"], s["start"]) for s in result["segments"]],
                         [("Alice", "Hello", 0.0), ("Bob", "Hi", 1.0),
                          ("Bob", "Yo", 2.0), ("Alice", "Bye", 3.0)])

    def test_serial_sources_offset_shifts_times(self):
        """Serial tracks: a source's offset shifts its segment start/end onto the
        session timeline; per-segment speakers are kept when no override is given."""
        sources = [
            {"name": "v1", "offset": 0.0, "speaker": None, "segments": [
                {"start": 0.0, "end": 1.0, "speaker": "Alice", "text": "Part 1"},
            ]},
            {"name": "v2", "offset": 10.0, "speaker": None, "segments": [
                {"start": 0.0, "end": 1.0, "speaker": "Bob", "text": "Part 2"},
            ]},
        ]
        result = merge_transcripts(sources)
        self.assertEqual([(s["speaker"], s["start"], s["end"]) for s in result["segments"]],
                         [("Alice", 0.0, 1.0), ("Bob", 10.0, 11.0)])

    def test_empty_text_dropped_and_no_sources_returns_empty(self):
        self.assertEqual(merge_transcripts([]), {"segments": []})
        one = merge_transcripts([{"name": "x", "offset": 0.0, "speaker": "X", "segments": [
            {"start": 0.0, "end": 1.0, "text": "  "},
            {"start": 1.0, "end": 2.0, "text": "real"},
        ]}])
        self.assertEqual([s["text"] for s in one["segments"]], ["real"])


class TestMergeTranscriptsAudacity(unittest.TestCase):
    """Test Workflow A: Audacity per-speaker audio files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_transcript(self, filename, segments):
        path = self.root / filename
        path.write_text(json.dumps({"segments": segments}))
        return path

    def test_merge_single_speaker(self):
        """Single speaker transcript merges correctly."""
        files = [self._write_transcript("speaker1.json", [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "World"},
        ])]
        mapping = {"speaker1": "Alice"}
        result = merge_transcripts_audacity(files, mapping)
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["segments"][0]["speaker"], "Alice")

    def test_merge_multiple_speakers_interleaved(self):
        """Multiple speaker transcripts interleave by timestamp."""
        files = [
            self._write_transcript("alice.json", [
                {"start": 0.0, "end": 1.0, "text": "Hello"},
                {"start": 3.0, "end": 4.0, "text": "How are you?"},
            ]),
            self._write_transcript("bob.json", [
                {"start": 1.0, "end": 2.0, "text": "Hi there"},
                {"start": 2.0, "end": 3.0, "text": "I'm good"},
            ])
        ]
        mapping = {"alice": "Alice", "bob": "Bob"}
        result = merge_transcripts_audacity(files, mapping)
        self.assertEqual(len(result["segments"]), 4)
        self.assertEqual(result["segments"][0]["speaker"], "Alice")
        self.assertEqual(result["segments"][1]["speaker"], "Bob")

    def test_filler_not_dropped_by_merge(self):
        """Filler removal is not the merge's job — it happens upstream in the
        clean_transcription stage, so the merge passes filler through and only
        drops empty-text segments. (See test_clean_transcription for the drop.)"""
        files = [self._write_transcript("alice.json", [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "uh"},
            {"start": 2.0, "end": 3.0, "text": "World"},
        ])]
        mapping = {"alice": "Alice"}
        result = merge_transcripts_audacity(files, mapping)
        self.assertEqual([s["text"] for s in result["segments"]], ["Hello", "uh", "World"])


class TestFillerDetection(unittest.TestCase):
    """Test filler phrase detection."""

    def test_exact_match(self):
        fillers = {"um", "uh", "er"}
        self.assertTrue(is_filler("um", fillers))

    def test_case_insensitive(self):
        fillers = {"um", "uh"}
        self.assertTrue(is_filler("UM", fillers))

    def test_non_filler_not_matched(self):
        fillers = {"um", "uh"}
        self.assertFalse(is_filler("hello", fillers))


if __name__ == "__main__":
    unittest.main(verbosity=2)
