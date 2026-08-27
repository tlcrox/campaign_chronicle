#!/usr/bin/env python3
"""
Reset the fixtures to a "not run" state.

Deletes pipeline-GENERATED artifacts, leaving the input fixtures (videos,
Audacity projects, ROI files, manual screen captures) intact. Under every session
directory below the root it removes:

  - transcriptions/     (WhisperX transcripts)
  - scenes_output/      (PySceneDetect CSVs + images)
  - combined_output/    (merged transcript/scenes)
  - *_storyboard*.docx  (generated storyboard documents)

Dry-run by DEFAULT — nothing is deleted until you pass --apply.

ScenesCraig ships a pre-populated scenes_output/ as a committed reference
(regression target), so that one folder is preserved unless --include-refs.

Usage:
    python reset_test_source.py                          # dry run against the default root
    python reset_test_source.py --apply                  # actually delete
    python reset_test_source.py --root D:/x --apply
    python reset_test_source.py --apply --include-refs    # also wipe ScenesCraig's reference
"""
import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_ROOT = Path("./")
# Everything the pipeline generates now lives under ONE folder per session
# (pipeline.common.mounts.OUTPUT_ROOT), so a reset is a single rmtree.
OUTPUT_DIRS = ("cc_output",)


def collect(root: Path):
    """Return (targets_to_remove, reference_dirs_kept)."""
    targets, kept = [], []
    # Layout is the single OUTPUT_ROOT folder
    for name in (OUTPUT_DIRS):
        for d in root.rglob(name):
            if not d.is_dir():
                continue
            targets.append(d)
    return targets, kept


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"Test source root (default: {DEFAULT_ROOT})")
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete (default: dry run)")
    args = ap.parse_args()

    if not args.root.exists():
        sys.exit(f"Root not found: {args.root}")

    targets, kept = collect(args.root)
    for d in kept:
        print(f"  KEEP (reference): {d}")

    if not targets:
        print("Nothing to remove - already at 'not run' state.")
        return

    verb = "Removing" if args.apply else "[dry run] would remove"
    for t in sorted(targets, key=lambda p: str(p).lower()):
        kind = "dir " if t.is_dir() else "file"
        print(f"  {verb} ({kind}): {t}")
        if args.apply:
            try:
                shutil.rmtree(t) if t.is_dir() else t.unlink()
            except OSError as e:
                print(f"    ! failed: {e}")

    print(f"\n{'Done' if args.apply else 'Dry run complete'} - {len(targets)} item(s).")
    if not args.apply:
        print("Re-run with --apply to delete.")


if __name__ == "__main__":
    main()
