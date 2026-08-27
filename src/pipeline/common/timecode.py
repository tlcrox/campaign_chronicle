#!/usr/bin/env python3
"""
timecode.py - the single home for timestamp <-> seconds conversion.

Every caller parses timestamps through here, so format support cannot drift.
Accepted:

  * "HH:MM:SS"        (integer seconds, e.g. ROI files)
  * "HH:MM:SS.sss"    (fractional seconds)
  * "MM:SS" / "MM:SS.sss"
  * "[HH:MM:SS.ss]"   (bracketed, e.g. old .txt transcripts)
  * a number          (already seconds; returned as float)

It is intentionally dependency-free so any process can import it.
"""

from __future__ import annotations

from typing import Union

Number = Union[int, float]


class TimecodeError(ValueError):
    """Raised when a timestamp string cannot be parsed."""


def timestamp_to_seconds(ts: Union[str, Number]) -> float:
    """Parse a timestamp into a float number of seconds.

    Accepts bracketed and unbracketed "H:M:S(.s)" or "M:S(.s)" strings, a bare
    "S(.s)" string, or a number (returned as float).
    """
    if isinstance(ts, bool):  # guard: bool is an int subclass
        raise TimecodeError(f"invalid timestamp: {ts!r}")
    if isinstance(ts, (int, float)):
        return float(ts)
    if ts is None:
        raise TimecodeError("timestamp is None")

    s = ts.strip().strip("[]").strip()
    if not s:
        raise TimecodeError("empty timestamp")

    try:
        parts = [float(p) for p in s.split(":")]
    except ValueError as exc:
        raise TimecodeError(f"non-numeric timestamp: {ts!r}") from exc

    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0.0, parts[0], parts[1]
    elif len(parts) == 1:
        hours, minutes, seconds = 0.0, 0.0, parts[0]
    else:
        raise TimecodeError(f"invalid timestamp (too many ':' parts): {ts!r}")

    return hours * 3600.0 + minutes * 60.0 + seconds


def seconds_to_timestamp(seconds: Number, *, millis: bool = False, decimals: int = 3) -> str:
    """Format seconds as ``HH:MM:SS`` (default) or ``HH:MM:SS.sss`` (millis=True).

    Negative inputs are rendered with a leading ``-``.
    """
    seconds = float(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if millis:
        width = 3 + decimals  # two int digits + '.' + decimals
        return f"{sign}{hours:02d}:{minutes:02d}:{secs:0{width}.{decimals}f}"
    return f"{sign}{hours:02d}:{minutes:02d}:{int(secs):02d}"


def to_ffmpeg_timestamp(seconds: Number, decimals: int = 3) -> str:
    """Convenience: ``HH:MM:SS.sss`` suitable for ffmpeg ``-ss``/``-to``."""
    return seconds_to_timestamp(seconds, millis=True, decimals=decimals)


if __name__ == "__main__":  # tiny CLI: echo conversions
    import sys

    for arg in sys.argv[1:]:
        try:
            secs = float(arg)
            print(f"{arg} -> {seconds_to_timestamp(secs)}  ({to_ffmpeg_timestamp(secs)})")
        except ValueError:
            print(f"{arg} -> {timestamp_to_seconds(arg)}s")
