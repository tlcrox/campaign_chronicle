#!/usr/bin/env python3
"""
clean.py - drop hallucinated / low-confidence segments from WhisperX JSON.

Reads every *.json in an input dir, removes segments that look like silence,
low-confidence output, or loop hallucinations, and writes cleaned JSON (same
filename) to an output dir.

This is purely a confidence filter; it has nothing to do with speaker names.
Thresholds come from config.yaml (whisper.clean) and can be overridden per call.

Importable API:
    thresholds_from_config() -> dict
    clean_segments(segments, thresholds) -> list
    clean_file(src, dst, thresholds) -> (before, after)
    clean_dir(in_dir, out_dir, thresholds) -> totals dict

CLI:
    python3 -m pipeline.transcribe.clean <input_dir> <output_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Fallbacks if config.yaml has no whisper.clean section.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "no_speech_max": 0.6,
    "logprob_min": -1.0,
    "compression_max": 2.4,
}


def thresholds_from_config(config=None) -> Dict[str, float]:
    """Load clean thresholds from config.yaml (whisper.clean), with defaults.

    ``config`` import is lazy so this module has no hard dependency on config.py.
    """
    if config is None:
        try:
            from pipeline.config import get_config
            config = get_config()
        except Exception:
            return dict(DEFAULT_THRESHOLDS)

    raw = config.get("whisper", "clean", {}) or {}
    merged = dict(DEFAULT_THRESHOLDS)
    for key in DEFAULT_THRESHOLDS:
        if key in raw:
            merged[key] = float(raw[key])
    return merged


def clean_segments(segments: List[dict], thresholds: Optional[Dict[str, float]] = None) -> List[dict]:
    """Return only the segments that pass all thresholds and have text."""
    t = thresholds or DEFAULT_THRESHOLDS
    return [
        s for s in segments
        if s.get("no_speech_prob", 0) < t["no_speech_max"]
        and s.get("avg_logprob", 0) > t["logprob_min"]
        and s.get("compression_ratio", 0) < t["compression_max"]
        and s.get("text", "").strip()
    ]


def clean_file(src: Path, dst: Path, thresholds: Optional[Dict[str, float]] = None):
    """Clean one JSON file; returns (before_count, after_count)."""
    src, dst = Path(src), Path(dst)
    with src.open(encoding="utf-8") as f:
        data = json.load(f)

    before = len(data.get("segments", []))
    data["segments"] = clean_segments(data.get("segments", []), thresholds)
    after = len(data["segments"])
    data["text"] = " ".join(s["text"].strip() for s in data["segments"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return before, after


def clean_dir(in_dir: Path, out_dir: Path, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """Clean every *.json in in_dir into out_dir. Returns totals."""
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if not in_dir.is_dir():
        raise NotADirectoryError(f"Input is not a directory: {in_dir}")

    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files in {in_dir}")

    totals = {"in": 0, "out": 0, "files": 0}
    for src in files:
        before, after = clean_file(src, out_dir / src.name, thresholds)
        print(f"{src.name}: {before} -> {after} segments")
        totals["in"] += before
        totals["out"] += after
        totals["files"] += 1

    print(
        f"\n{totals['files']} file(s): {totals['in']} -> {totals['out']} segments "
        f"({totals['in'] - totals['out']} dropped)"
    )
    return totals


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Drop hallucinated/low-confidence segments from WhisperX JSON.",
    )
    parser.add_argument("input_dir", help="Directory of WhisperX *.json files")
    parser.add_argument("output_dir", help="Directory to write cleaned *.json into")
    parser.add_argument("--no-speech-max", type=float, default=None)
    parser.add_argument("--logprob-min", type=float, default=None)
    parser.add_argument("--compression-max", type=float, default=None)
    args = parser.parse_args(argv)

    thresholds = thresholds_from_config()
    if args.no_speech_max is not None:
        thresholds["no_speech_max"] = args.no_speech_max
    if args.logprob_min is not None:
        thresholds["logprob_min"] = args.logprob_min
    if args.compression_max is not None:
        thresholds["compression_max"] = args.compression_max

    try:
        clean_dir(args.input_dir, args.output_dir, thresholds)
    except (NotADirectoryError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
