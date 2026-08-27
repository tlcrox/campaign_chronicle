#!/usr/bin/env python3
"""Comprehensive tests for storyboard.py - transcript parsing and document generation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


from pipeline.merge.storyboard import (
    parse_txt_transcript,
    parse_json_transcript,
    detect_transcript_format,
    _flush_speaker_entries,
    _add_dialogue_paragraph,
)


class TestTranscriptParsing(unittest.TestCase):
    """Test transcript format detection and parsing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_detect_transcript_format_json(self):
        """Detect JSON format from .json extension."""
        json_file = self.root / "transcript.json"
        json_file.touch()
        self.assertTrue(detect_transcript_format(json_file))

    def test_detect_transcript_format_txt(self):
        """Detect TXT format from .txt extension."""
        txt_file = self.root / "transcript.txt"
        txt_file.touch()
        self.assertFalse(detect_transcript_format(txt_file))

    def test_detect_transcript_format_case_insensitive(self):
        """Format detection is case-insensitive."""
        upper_json = self.root / "transcript.JSON"
        upper_json.touch()
        self.assertTrue(detect_transcript_format(upper_json))

    def test_parse_txt_transcript_basic(self):
        """Parse basic TXT transcript with speaker and dialogue."""
        txt_file = self.root / "transcript.txt"
        txt_file.write_text(
            "[0:10.50] Alice: Hello\n"
            "[0:15.00] Bob: Hi there\n"
        )
        entries = parse_txt_transcript(txt_file)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["speaker"], "Alice")
        self.assertEqual(entries[0]["text"], "Hello")
        self.assertAlmostEqual(entries[0]["start"], 10.5)
        self.assertEqual(entries[1]["speaker"], "Bob")

    def test_parse_txt_transcript_with_hours(self):
        """Parse TXT format with HH:MM:SS.ss timestamps."""
        txt_file = self.root / "transcript.txt"
        txt_file.write_text(
            "[1:05:30.00] Alice: Long conversation\n"
            "[1:05:35.50] Bob: Response\n"
        )
        entries = parse_txt_transcript(txt_file)
        self.assertEqual(len(entries), 2)
        # 1 hour + 5 minutes + 30 seconds = 3930 seconds
        self.assertAlmostEqual(entries[0]["start"], 3930.0)
        self.assertAlmostEqual(entries[1]["start"], 3935.5)

    def test_parse_txt_transcript_with_colons_in_dialogue(self):
        """Parse dialogue that contains colons."""
        txt_file = self.root / "transcript.txt"
        txt_file.write_text("[0:10.00] Alice: Time is 12:30 PM\n")
        entries = parse_txt_transcript(txt_file)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], "Time is 12:30 PM")

    def test_parse_txt_transcript_skips_invalid_lines(self):
        """Skip lines that don't match timestamp format."""
        txt_file = self.root / "transcript.txt"
        txt_file.write_text(
            "[0:10.00] Alice: Valid\n"
            "This is invalid\n"
            "[0:15.00] Bob: Also valid\n"
        )
        entries = parse_txt_transcript(txt_file)
        self.assertEqual(len(entries), 2)

    def test_parse_json_transcript_basic(self):
        """Parse basic JSON transcript with segments."""
        json_file = self.root / "transcript.json"
        data = {
            "segments": [
                {"start": 0.0, "end": 2.5, "speaker": "Alice", "text": "Hello", "confidence": 1.0},
                {"start": 2.5, "end": 5.0, "speaker": "Bob", "text": "Hi", "confidence": 0.95},
            ]
        }
        json_file.write_text(json.dumps(data))
        entries = parse_json_transcript(json_file)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["speaker"], "Alice")
        self.assertEqual(entries[0]["confidence"], 1.0)
        self.assertEqual(entries[1]["confidence"], 0.95)

    def test_parse_json_transcript_with_missing_speaker(self):
        """Handle MISSING speaker markers by using last known speaker."""
        json_file = self.root / "transcript.json"
        data = {
            "segments": [
                {"start": 0.0, "end": 2.0, "speaker": "Alice", "text": "Hello", "confidence": 1.0},
                {"start": 2.0, "end": 4.0, "speaker": "MISSING", "text": "Unclear", "confidence": 0.5},
                {"start": 4.0, "end": 6.0, "speaker": "Bob", "text": "Hi", "confidence": 1.0},
                {"start": 6.0, "end": 8.0, "speaker": None, "text": "Also unclear", "confidence": 0.3},
            ]
        }
        json_file.write_text(json.dumps(data))
        entries = parse_json_transcript(json_file)
        self.assertEqual(len(entries), 4)
        self.assertEqual(entries[1]["speaker"], "Alice")  # Carried from previous
        self.assertEqual(entries[3]["speaker"], "Bob")      # Carried from previous

    def test_parse_json_transcript_strips_whitespace(self):
        """Strip whitespace from text."""
        json_file = self.root / "transcript.json"
        data = {
            "segments": [
                {"start": 0.0, "end": 2.0, "speaker": "Alice", "text": "  Hello  ", "confidence": 1.0},
            ]
        }
        json_file.write_text(json.dumps(data))
        entries = parse_json_transcript(json_file)
        self.assertEqual(entries[0]["text"], "Hello")

    def test_parse_json_transcript_invalid_file(self):
        """Handle invalid JSON gracefully."""
        json_file = self.root / "transcript.json"
        json_file.write_text("not valid json {")
        entries = parse_json_transcript(json_file)
        self.assertEqual(len(entries), 0)

    def test_parse_json_transcript_missing_segments_key(self):
        """Handle JSON without 'segments' key."""
        json_file = self.root / "transcript.json"
        data = {"no_segments": []}
        json_file.write_text(json.dumps(data))
        entries = parse_json_transcript(json_file)
        self.assertEqual(len(entries), 0)

class TestSpeakerFlushLogic(unittest.TestCase):
    """Test the speaker flush logic that accumulates dialogue."""

    def setUp(self):
        self.mock_doc = MagicMock()

    def test_flush_speaker_entries_single_speaker(self):
        """Flush entries from a single speaker."""
        entries = [
            {"speaker": "Alice", "text": "Hello", "confidence": 1.0},
            {"speaker": "Alice", "text": "How are you?", "confidence": 1.0},
        ]
        entry_idx, added = _flush_speaker_entries(entries, 0, self.mock_doc, False)
        self.assertEqual(entry_idx, 2)
        self.assertEqual(added, 1)  # One paragraph for Alice's dialogue

    def test_flush_speaker_entries_multiple_speakers(self):
        """Separate entries by speaker."""
        entries = [
            {"speaker": "Alice", "text": "Hello", "confidence": 1.0},
            {"speaker": "Bob", "text": "Hi", "confidence": 1.0},
            {"speaker": "Alice", "text": "How are you?", "confidence": 1.0},
        ]
        entry_idx, added = _flush_speaker_entries(entries, 0, self.mock_doc, False)
        self.assertEqual(entry_idx, 3)
        self.assertEqual(added, 3)  # Three speakers/transitions

    def test_flush_speaker_entries_with_stop_condition(self):
        """Stop flushing when stop condition is met."""
        entries = [
            {"speaker": "Alice", "text": "First", "confidence": 1.0, "start": 0.0},
            {"speaker": "Alice", "text": "Second", "confidence": 1.0, "start": 2.0},
            {"speaker": "Bob", "text": "Third", "confidence": 1.0, "start": 5.0},
        ]
        entry_idx, added = _flush_speaker_entries(
            entries, 0, self.mock_doc, False, 2.5  # stop before the entry at 5.0
        )
        self.assertEqual(entry_idx, 2)  # Should stop before the third entry

    def test_flush_speaker_entries_stop_at_zero(self):
        """0.0 is a real stop time, so the check cannot be a truthiness test."""
        entries = [
            {"speaker": "Alice", "text": "First", "confidence": 1.0, "start": 0.0},
            {"speaker": "Bob", "text": "Second", "confidence": 1.0, "start": 2.0},
        ]
        entry_idx, added = _flush_speaker_entries(entries, 0, self.mock_doc, False, 0.0)
        self.assertEqual(entry_idx, 0)   # stops immediately; nothing flushed
        self.assertEqual(added, 0)

    def test_flush_speaker_entries_none_flushes_everything(self):
        entries = [
            {"speaker": "Alice", "text": "First", "confidence": 1.0, "start": 0.0},
            {"speaker": "Bob", "text": "Second", "confidence": 1.0, "start": 2.0},
        ]
        entry_idx, added = _flush_speaker_entries(entries, 0, self.mock_doc, False, None)
        self.assertEqual(entry_idx, 2)

    def test_flush_speaker_entries_empty_entries(self):
        """Handle empty entry list."""
        entries = []
        entry_idx, added = _flush_speaker_entries(entries, 0, self.mock_doc, False)
        self.assertEqual(entry_idx, 0)
        self.assertEqual(added, 0)

    def test_flush_speaker_entries_starting_offset(self):
        """Start flushing from a non-zero offset."""
        entries = [
            {"speaker": "Alice", "text": "First", "confidence": 1.0},
            {"speaker": "Bob", "text": "Second", "confidence": 1.0},
        ]
        entry_idx, added = _flush_speaker_entries(entries, 1, self.mock_doc, False)
        self.assertEqual(entry_idx, 2)
        self.assertEqual(added, 1)  # Only Bob's entry

    def test_add_dialogue_paragraph_with_speaker(self):
        """Add paragraph with speaker bold and dialogue."""
        mock_doc = MagicMock()
        mock_paragraph = MagicMock()
        mock_doc.add_paragraph.return_value = mock_paragraph

        dialogue_parts = [{"text": "Hello there", "confidence": 1.0}]
        _add_dialogue_paragraph(mock_doc, "Alice", dialogue_parts, False)

        mock_doc.add_paragraph.assert_called_once()
        # Check that speaker run was created with bold
        calls = mock_paragraph.add_run.call_args_list
        self.assertTrue(any("Alice:" in str(call) for call in calls))

    def test_add_dialogue_paragraph_multiple_parts(self):
        """Add paragraph with multiple dialogue parts."""
        mock_doc = MagicMock()
        mock_paragraph = MagicMock()
        mock_doc.add_paragraph.return_value = mock_paragraph

        dialogue_parts = [
            {"text": "Part 1", "confidence": 1.0},
            {"text": "Part 2", "confidence": 1.0},
        ]
        _add_dialogue_paragraph(mock_doc, "Bob", dialogue_parts, False)

        mock_doc.add_paragraph.assert_called_once()
        self.assertGreaterEqual(len(mock_paragraph.add_run.call_args_list), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
