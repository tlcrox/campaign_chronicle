#!/usr/bin/env python3
"""
Tests for pipeline.common.timecode.

Run from scripts/:
    python3 -m unittest pipeline.common.test_timecode -v
"""

import unittest


from pipeline.common.timecode import (
    TimecodeError,
    seconds_to_timestamp,
    timestamp_to_seconds,
    to_ffmpeg_timestamp,
)


class ParseTimestamp(unittest.TestCase):
    def test_hms(self):
        self.assertEqual(timestamp_to_seconds("01:02:03"), 3723.0)
        self.assertEqual(timestamp_to_seconds("00:00:00"), 0.0)

    def test_fractional(self):
        self.assertAlmostEqual(timestamp_to_seconds("00:00:12.34"), 12.34)
        self.assertAlmostEqual(timestamp_to_seconds("01:02:03.5"), 3723.5)

    def test_mm_ss(self):
        self.assertEqual(timestamp_to_seconds("05:00"), 300.0)
        self.assertAlmostEqual(timestamp_to_seconds("12.89"), 12.89)

    def test_bracketed(self):
        self.assertAlmostEqual(timestamp_to_seconds("[00:00:12.34]"), 12.34)
        self.assertAlmostEqual(timestamp_to_seconds(" [01:00:00] "), 3600.0)

    def test_number_passthrough(self):
        self.assertEqual(timestamp_to_seconds(42), 42.0)
        self.assertEqual(timestamp_to_seconds(3.5), 3.5)

    def test_errors(self):
        for bad in ("", "  ", "aa:bb", "1:2:3:4", None, True):
            with self.assertRaises(TimecodeError):
                timestamp_to_seconds(bad)


class FormatSeconds(unittest.TestCase):
    def test_hms(self):
        self.assertEqual(seconds_to_timestamp(3723), "01:02:03")
        self.assertEqual(seconds_to_timestamp(0), "00:00:00")
        self.assertEqual(seconds_to_timestamp(59.9), "00:00:59")  # truncates without millis

    def test_millis(self):
        self.assertEqual(seconds_to_timestamp(12.34, millis=True), "00:00:12.340")
        self.assertEqual(to_ffmpeg_timestamp(3723.5), "01:02:03.500")

    def test_negative(self):
        self.assertEqual(seconds_to_timestamp(-5), "-00:00:05")

    def test_round_trip(self):
        for s in (0, 1, 59, 60, 3599, 3600, 3723):
            self.assertEqual(timestamp_to_seconds(seconds_to_timestamp(s)), float(s))


if __name__ == "__main__":
    unittest.main(verbosity=2)
