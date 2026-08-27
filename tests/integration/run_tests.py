#!/usr/bin/env python3
"""
Master test runner for WhisperX pipeline on test directories.

Tests each tool (transcription, scene detection, merge) on all test directories.
Validates output structure and identifies mismatches.
"""

import os
import subprocess
import sys
from pathlib import Path
import json
import logging

from pipeline.config import find_repo_root

# `pipeline` and `cc_stages` come from the installed project (pyproject.toml);
# no sys.path manipulation is needed or wanted here.

from pipeline.common.logs import setup_logging

# Configured in main(), never here: importing a module must not reconfigure
# logging for whatever process happened to import it.
logger = logging.getLogger(__name__)

# Repo root, walking up to the directory containing pyproject.toml — the one
# implementation, in pipeline.config. required=True because a driver with no
# repo above it cannot do anything useful.
REPO_ROOT = find_repo_root(__file__, required=True)
WHISPERX_ROOT = REPO_ROOT          # kept: used as the subprocess cwd below
STAGES_DIR = REPO_ROOT / "scripts" / "cc_stages"

# Integration-test layout:
#   tests/test_source/  INPUTS  — media, Audacity projects, ROI files, manual
#                                 screen captures, plus the governing config.yaml
#   tests/expected/     GOLDENS — the outcomes a run is checked against
# config/config.yaml is the MANUAL-run config and is deliberately NOT consulted
# here, so a developer's own source_dir can never influence the suite.
FIXTURES = REPO_ROOT / "tests" / "test_source"
EXPECTED = REPO_ROOT / "tests" / "expected"

# This driver writes its results to a SEPARATE tree, not beside the inputs.
# run_orchestrate_tests.py deliberately does the opposite (output beside the
# source), so the two suites can run in any order without cleaning up between
# them — and, more importantly, neither can leave behind output that makes the
# other appear to pass. Both produce the same cc_output/... shape, so one
# directory comparison against tests/expected/ serves both.
TEST_OUTPUT = REPO_ROOT / "tests" / "integration" / "output"
os.environ.setdefault("CC_OUTPUT_BASE", str(TEST_OUTPUT))

# Historical fallback if config.yaml can't be loaded / has no session_dirs.
# The governing config is tests/test_source/config.yaml — change the tests there.
DEFAULT_TEST_DIRS = [
    "SingleVideo",      # Basic: 1 video, embedded audio
    "MultiVideo",       # Multiple videos, embedded audio (tests VIDEO_INDEX)
    "SingleCraig",      # Single video + Audacity audio
    "MultiCraig",       # Multiple videos + shared Audacity audio + per-video ROI
    "MultiROI",         # Multiple videos + time-based ROI regions
    "SingleROI",        # Single video default ROI from config.
    "ScenesCraig",      # Reference: already has scene output
    "Weeks/Week 13",    # Video only
    "Weeks/Week 14",    # Audio only
]

# Resolve the config BEFORE the import-time get_config() below, so a bare test
# run targets the in-repo fixtures and NEVER config/config.yaml (that one is the
# manual-run config and may point anywhere a developer likes). Honor --config,
# else <--source-dir>/config.yaml, with --source-dir defaulting to the fixtures.
import argparse as _argparse
_pre = _argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", default=None)
_pre.add_argument("--source-dir", default=str(FIXTURES))
_pre_args, _ = _pre.parse_known_args()
if _pre_args.config:
    os.environ.setdefault("WHISPERX_CONFIG", str(_pre_args.config))
elif not os.environ.get("WHISPERX_CONFIG"):
    _fixture_cfg = Path(_pre_args.source_dir) / "config.yaml"
    if _fixture_cfg.exists():
        os.environ["WHISPERX_CONFIG"] = str(_fixture_cfg)

# Base dir + session list come from the resolved config; the command line
# (--source-dir / --session-dirs / --dir) overrides them in main().
# Output layout constants — pure constants, no config dependency, so they are
# imported OUTSIDE the try below (the except branch references them too).
from pipeline.common.mounts import (  # noqa: E402
    output_dir_for,
    OUTPUT_ROOT,
    SESSION_OUTPUT_SUBDIR,
    SCENES_OUTPUT_SUBDIR,
    COMBINED_OUTPUT_SUBDIR,
)
from pipeline.common.scenes import iter_scene_images  # noqa: E402
# Imported, not restated: this driver's per-tool cap has to match the cap
# the Docker layer applies inside it.
from pipeline.common.docker import DOCKER_RUN_TIMEOUT  # noqa: E402

# Shared golden-comparison logic, used by BOTH integration drivers against their
# respective (differing) output hierarchies. Sibling import when run as a script
# (sys.path[0] is this dir); package import when run as -m tests.integration.*.
try:
    from output_compare import compare_output_tree, wipe_output_tree  # noqa: E402
except ImportError:  # pragma: no cover - only the -m invocation path
    from tests.integration.output_compare import compare_output_tree, wipe_output_tree

try:
    from pipeline.config import get_config
    cfg = get_config()
    DEFAULT_ROI = cfg.scene_roi or "400 445 925 745"  # Fallback to config ROI
    _src = cfg.source_dir
    TEST_BASE_DIR = _src if _src.is_absolute() else (REPO_ROOT / _src).resolve()
    # Entries exactly as configured, already relative to source_dir (e.g.
    # "Weeks/Week 13"). Do not reduce these to .name — nested paths matter.
    TEST_DIRS = [str(d) for d in cfg.session_dirs_config] or list(DEFAULT_TEST_DIRS)
    TEST_TRANSCRIPTS = SESSION_OUTPUT_SUBDIR
    TEST_SCENES = SCENES_OUTPUT_SUBDIR
except Exception as e:
    logger.warning(f"Could not load config, using defaults: {e}")
    DEFAULT_ROI = "400 445 925 745"  # Fallback default
    TEST_BASE_DIR = FIXTURES
    TEST_DIRS = list(DEFAULT_TEST_DIRS)
    TEST_TRANSCRIPTS = SESSION_OUTPUT_SUBDIR
    TEST_SCENES = SCENES_OUTPUT_SUBDIR

# Every stage a session runs, in order. A session passes only if all of them
# succeeded — the same rule run_orchestrate_tests.py applies to orchestrate's
# return code. Dry runs record True for each (nothing ran).
def _mark(ok) -> str:
    """Status glyph for the summary lines.

    Three states, not two: True passed, False failed, and None did not apply
    to this session — an audio-only session has no scenes to detect, and
    calling that a failure is how a green run gets reported red.
    """
    if ok is None:
        return "None"
    return "✓" if ok else "✗"


STAGE_KEYS = (
    "transcription",
    "speaker_mapping",
    "clean_transcription",
    "scene_detection",
    "merge_transcripts",
    "merge_scenes",
    "generate_storyboard",
)


class TestRunner:
    def __init__(self, compare: bool = True, clean: bool = True,
                 tool_timeout: int = DOCKER_RUN_TIMEOUT):
        self.results = {}
        self.errors = {}
        # When True (and not a dry run), each session's produced cc_output tree is
        # compared against tests/expected/<session>/cc_output after it runs.
        self.compare = compare
        # When True (and not a dry run), each session's output dir is wiped BEFORE
        # its stages run, so the comparison only ever sees this run's output.
        self.clean = clean
        # Per-tool wall-clock cap (seconds). Transcription is the long pole: a
        # multi-file session (e.g. ScenesCraig's 6 Audacity speakers) with
        # large-v3 + diarization, plus a possible first-run model download, runs
        # well past the old hardcoded 300s. Matches the Docker layer's own cap
        # (run_docker_command), so neither can cut the other short.
        self.tool_timeout = tool_timeout

    def run_tool(self, tool_name: str, session_dir: Path, args: list = None) -> bool:
        """Run a tool and return success/failure"""
        tool_script = STAGES_DIR / f"{tool_name}.py"

        if not tool_script.exists():
            logger.error(f"Stage not found: {tool_script}")
            return False

        cmd = [
            sys.executable,
            str(tool_script),
            "--session-dir", str(session_dir)
        ]

        if args:
            cmd.extend(args)

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                # Tools print UTF-8 status glyphs (✓ ⊘ ✗); decode explicitly so
                # the capture can't die with a UnicodeDecodeError under the
                # Windows cp1252 locale default.
                encoding="utf-8",
                errors="replace",
                cwd=str(WHISPERX_ROOT),
                timeout=self.tool_timeout
            )

            if result.returncode != 0:
                logger.error(f"Tool failed with return code {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
                return False

            logger.info(f"✓ {tool_name} succeeded")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"Tool timed out")
            return False
        except Exception as e:
            logger.error(f"Error running tool: {e}")
            return False

    def _manual_scenes_dir(self, session_dir: Path):
        """Return the pre-captured manual scenes folder if present, else None.

        Mirrors merge_scenes.find_scene_dirs_and_csvs: a session has manual
        scenes when the configured scenes.manual_source folder exists and holds
        per-video subdirs each with a scenes.manual_csv_name CSV.
        """
        try:
            name = cfg.get("scenes", "manual_source")
            csv_name = cfg.get("scenes", "manual_csv_name", "Scenes.csv")
        except Exception:
            return None
        if not name:
            return None
        folder = Path(session_dir) / name
        if folder.exists() and list(folder.glob(f"*/{csv_name}")):
            return folder
        return None

    def validate_output(self, session_dir: Path, expect_scenes: bool = True) -> dict:
        """Validate output structure for a session.

        ``expect_scenes`` is False for a session with no video and no manual
        capture folder: there is nothing to detect, so the scenes check records
        None (not applicable) rather than a failure. The counts are still
        reported in ``details`` either way.
        """
        validation = {
            "transcripts": False,
            "scenes": False,
            "storyboard": False,
            "details": {}
        }

        session_path = Path(session_dir)
        # Output does not live beside the input. Resolve it the same way the
        # stages do, so this validates wherever CC_OUTPUT_BASE sent it.
        out_root = output_dir_for(session_path, "", cfg).resolve()

        # Check transcripts
        transcripts_dir = out_root / TEST_TRANSCRIPTS
        if transcripts_dir.exists():
            json_files = list(transcripts_dir.glob("*.json"))
            validation["transcripts"] = len(json_files) > 0
            validation["details"]["transcripts"] = {
                "exists": True,
                "files": len(json_files),
                "names": [f.name for f in json_files]
            }
        else:
            validation["details"]["transcripts"] = {"exists": False}

        # Check scenes — validate the MERGED result in combined_output/, which is
        # source-agnostic: auto-detected scenes and pre-captured/manual scenes both
        # land here after merge_scenes. The question is simply "did it get scenes?"
        # == is there a merged "<session>_Scenes.csv" (with images) in combined_output/.
        combined_dir = out_root / COMBINED_OUTPUT_SUBDIR
        scene_csvs = list(combined_dir.glob("*_Scenes.csv")) if combined_dir.exists() else []
        scene_imgs = iter_scene_images(combined_dir, "Scene-*")
        validation["scenes"] = (len(scene_csvs) > 0) if expect_scenes else None
        validation["details"]["scenes"] = {
            "expected": expect_scenes,
            "combined_output": combined_dir.exists(),
            "csv_files": len(scene_csvs),
            "image_files": len(scene_imgs),
            "csv_names": [f.name for f in scene_csvs],
        }

        # Check storyboard — it lands in the OUTPUT tree at
        # out_root/OUTPUT_ROOT/<session>_storyboard.docx, NOT beside the input.
        # Resolve via the constants; a hand-written literal here would silently
        # report False for every session.
        storyboard = out_root / OUTPUT_ROOT / f"{session_path.name}_storyboard.docx"
        if storyboard.exists():
            validation["storyboard"] = True
            validation["details"]["storyboard"] = {
                "exists": True,
                "name": storyboard.name,
                "size": storyboard.stat().st_size,
            }
        else:
            validation["details"]["storyboard"] = {"exists": False}

        return validation

    def test_single_directory(self, test_dir_name: str, dry_run: bool = False):
        """Test a single directory through all tools"""
        test_path = TEST_BASE_DIR / test_dir_name
        logger.info(f"Test directory {test_path} from {test_dir_name}")

        if not test_path.exists():
            # Record it. Returning without an entry in self.results would leave
            # the session invisible to every gate below, so a suite pointed at a
            # missing fixture would report success having run nothing.
            logger.error(f"Test directory not found: {test_path}")
            self.results[test_dir_name] = {k: False for k in STAGE_KEYS}
            self.results[test_dir_name]["error"] = "test directory not found"
            return False

        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {test_dir_name}")
        logger.info(f"{'='*70}\n")

        # Start clean: wipe this session's output before any stage writes to it,
        # so the golden comparison below only ever sees this run's output.
        if self.clean and not dry_run:
            produced_cc = output_dir_for(test_path, OUTPUT_ROOT, cfg).resolve()
            logger.info(f"Cleaning output: {produced_cc}")
            wipe_output_tree(produced_cc, OUTPUT_ROOT)

        self.results[test_dir_name] = {k: False for k in STAGE_KEYS}

        # 1. Transcription
        logger.info(f"\n--- Transcription ---")

        # Detect audio source type to determine which transcription tool to use
        aup_files = list(test_path.glob("*.aup")) + list(test_path.glob("*.aup3"))
        aup_subdirs = [d for d in test_path.glob("*") if d.is_dir() and d.name.endswith(".aup")]
        video_exts = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
        audio_exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
        video_files_present = [f for f in test_path.glob("*") if f.suffix.lower() in video_exts]
        audio_files_present = [f for f in test_path.glob("*") if f.suffix.lower() in audio_exts]

        # Logged for the reader only — the transcribe stage detects the source
        # type itself and is called the same way regardless.
        if aup_files or aup_subdirs:
            logger.info("Audacity project detected (.aup)")
        elif video_files_present:
            logger.info("Video with embedded audio detected")
        elif audio_files_present:
            logger.info("Loose audio detected (no video)")
        else:
            logger.info("No recognizable media; transcribe will report why")
        transcribe_tool = "transcribe"

        if not dry_run:
            success = self.run_tool(transcribe_tool, test_path)
            self.results[test_dir_name]["transcription"] = success
        else:
            logger.info(f"(dry-run: would run {transcribe_tool})")
            self.results[test_dir_name]["transcription"] = True  # Mark as skipped

        # 1b/1c. Speaker Mapping. Auto-detects the resolver (filename for
        #     Audacity, diarization for video/audio), so it runs for every workflow.
        logger.info(f"\n--- Speaker Mapping ---")
        if not dry_run:
            success = self.run_tool("map_speakers", test_path)
            self.results[test_dir_name]["speaker_mapping"] = success
        else:
            logger.info("(dry-run: would run map_speakers)")
            self.results[test_dir_name]["speaker_mapping"] = True

        # 1d. Clean Transcription — confidence pass (opt-in) + filler removal,
        #     between map_speakers and merge. Runs for every workflow.
        logger.info(f"\n--- Clean Transcription ---")
        if not dry_run:
            success = self.run_tool("clean_transcription", test_path)
            self.results[test_dir_name]["clean_transcription"] = success
        else:
            logger.info("(dry-run: would run clean_transcription)")
            self.results[test_dir_name]["clean_transcription"] = True

        # 2. Scene Detection
        logger.info(f"\n--- Scene Detection ---")

        # Skip detection if the session already has manually captured scenes in
        # the configured manual_source folder (e.g. the ScenesCraig fixture) —
        # merge_scenes will use those instead of generating new screenshots.
        manual_scenes = self._manual_scenes_dir(test_path)

        # Determine if this uses multi-ROI
        roi_file = test_path / "roi_history.json"
        is_multi_roi = roi_file.exists()
        success = False  # Initialize

        # A session with no video has nothing to detect. Without this the run
        # falls through to single-ROI mode, the tool correctly reports that it
        # found no video, and an entirely healthy audio-only session is recorded
        # as a failed stage.
        expect_scenes = manual_scenes is not None or bool(video_files_present)

        if manual_scenes is not None:
            logger.info(f"Using pre-captured scenes in {manual_scenes.name}/ (skipping detection)")
            success = True
        elif not video_files_present:
            logger.info("No video in this session; scene detection does not apply")
            success = None
        elif is_multi_roi:
            logger.info("Using Multi-ROI mode (roi_history.json present)")
            # detect_scenes_multi_roi.py consumes the hierarchical (video-keyed)
            # roi_history.json via the RoiFile model. Run it against the fixture.
            if not dry_run:
                success = self.run_tool("detect_scenes_multi_roi", test_path)
            else:
                logger.info("(dry-run: would run detect_scenes_multi_roi)")
                success = True
        else:
            logger.info("Using Single-ROI mode")
            # Do NOT pass --roi: the tool resolves the ROI from config.yaml
            # (scene_detection.roi) when it is set, and falls back to full
            # frame otherwise. This exercises the config-driven path.
            if not dry_run:
                success = self.run_tool(
                    "detect_scenes_single_roi",
                    test_path
                )
            else:
                logger.info(f"(dry-run: would resolve ROI from config, currently={DEFAULT_ROI or 'full frame'})")
                success = True  # Mark dry-run as skipped

        self.results[test_dir_name]["scene_detection"] = success

        # 3. Merge Tools
        logger.info(f"\n--- Merge Tools ---")
        if not dry_run:
            # Each result is recorded. merge_transcripts and merge_scenes feed
            # generate_storyboard, but a failure in either is its own failure —
            # storyboard can still succeed against stale or partial input.
            for tool in ("merge_transcripts", "merge_scenes", "generate_storyboard"):
                self.results[test_dir_name][tool] = self.run_tool(tool, test_path)
        else:
            logger.info("(dry-run: skipping)")
            for tool in ("merge_transcripts", "merge_scenes", "generate_storyboard"):
                self.results[test_dir_name][tool] = True  # Mark as skipped

        # 4. Validation - skipped on a dry run, where nothing was written and
        # every check would report a false miss. Recorded only when it ran, so
        # validations_ok() can tell "passed" from "never looked" (same shape as
        # the comparison below).
        if not dry_run:
            logger.info(f"\n--- Validation ---")
            validation = self.validate_output(test_path, expect_scenes=expect_scenes)
            self.results[test_dir_name]["validation"] = validation

            logger.info(f"Transcripts: {_mark(validation['transcripts'])}")
            logger.info(f"Scenes: {_mark(validation['scenes'])}")
            logger.info(f"Storyboard: {_mark(validation['storyboard'])}")

            # Print details
            if validation["details"]:
                logger.info(f"\nDetails:")
                logger.info(json.dumps(validation["details"], indent=2))

        # 5. Golden comparison — the produced cc_output tree vs tests/expected/.
        # Skipped on a dry run (nothing was written). Produced output for THIS
        # driver lives under CC_OUTPUT_BASE; the same helper the stages use
        # resolves it, so we compare wherever it actually landed.
        if self.compare and not dry_run:
            logger.info(f"\n--- Golden Comparison ---")
            produced_cc = output_dir_for(test_path, OUTPUT_ROOT, cfg).resolve()
            expected_cc = EXPECTED / test_dir_name / OUTPUT_ROOT
            comparison = compare_output_tree(produced_cc, expected_cc)
            self.results[test_dir_name]["comparison"] = comparison.as_dict()
            for line in comparison.summary_lines():
                (logger.info if comparison.ok else logger.error)(line)

        return True

    def run_all_tests(self, dry_run: bool = False) -> bool:
        """Run tests on all test directories. Returns overall pass/fail."""
        logger.info(f"WhisperX Test Suite")
        logger.info(f"Test Base: {TEST_BASE_DIR}")
        logger.info(f"Dry Run: {dry_run}\n")

        for test_dir in TEST_DIRS:
            self.test_single_directory(test_dir, dry_run=dry_run)

        # Summary
        return self.print_summary()

    def stages_ok(self) -> bool:
        """True if no stage of any session failed.

        A stage recorded as None did not apply to that session (no video to
        detect scenes in) and does not count against it. Not vacuous: an absent
        key reads as False, which is what a missing fixture directory produces.
        """
        return all(
            all(r.get(k, False) is not False for k in STAGE_KEYS)
            for r in self.results.values()
        )

    def validations_ok(self) -> bool:
        """True if no validated session is missing output it should have produced.

        A check recorded as None did not apply (see validate_output). Vacuously
        true on a dry run, where no 'validation' key was recorded at all. This is
        the only content gate left when --no-compare is in force, which is what
        that flag's help text promises.
        """
        return all(
            v.get(check) is not False
            for v in (r.get("validation") for r in self.results.values())
            if v
            for check in ("transcripts", "scenes", "storyboard")
        )

    def comparisons_ok(self) -> bool:
        """True if every session that was compared matched its golden.

        Vacuously true when comparison is disabled or was a dry run (no
        'comparison' key was recorded for any session).
        """
        return all(
            r["comparison"]["ok"]
            for r in self.results.values()
            if "comparison" in r
        )

    def print_summary(self) -> bool:
        """Print test summary. Returns True if every gate passed."""
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST SUMMARY")
        logger.info(f"{'='*70}\n")

        for test_dir, results in self.results.items():
            logger.info(f"{test_dir}:")
            if results.get("error"):
                logger.error(f"  {results['error']}")
            for key in STAGE_KEYS:
                label = key.replace("_", " ").title()
                logger.info(f"  {label}: {_mark(results.get(key, False))}")

            validation = results.get("validation")
            if validation is not None:
                logger.info(f"  Output Validation:")
                logger.info(f"    Transcripts: {_mark(validation['transcripts'])}")
                logger.info(f"    Scenes: {_mark(validation['scenes'])}")
                logger.info(f"    Storyboard: {_mark(validation['storyboard'])}")

            comparison = results.get("comparison")
            if comparison is not None:
                detail = ("matches expected" if comparison["ok"]
                          else f"{len(comparison['diffs'])} difference(s)")
                logger.info(f"  Golden Comparison: {_mark(comparison['ok'])} ({detail})")

        # Save results to JSON. Explicit encoding and newline: the default is the
        # cp1252 locale on Windows, and the summary carries UTF-8 glyphs.
        results_file = WHISPERX_ROOT / "test_results.json"
        with open(results_file, 'w', encoding="utf-8", newline="\n") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nResults saved to: {results_file}")

        if not self.results:
            logger.error("No sessions ran - nothing was verified.")
            return False

        # Three independent gates. A session must clear all three: the stages
        # have to succeed, the output they produced has to have the expected
        # shape, and it has to match the golden. Validation and comparison are
        # vacuous when they did not run (dry run, --no-compare); stages are not.
        gates = {
            "stages": self.stages_ok(),
            "output validation": self.validations_ok(),
            "golden comparison": self.comparisons_ok(),
        }
        logger.info("")
        for name, ok in gates.items():
            (logger.info if ok else logger.error)(
                f"{name}: {'PASS' if ok else 'FAIL'}"
            )
        return all(gates.values())

def main():
    setup_logging()
    import argparse
    global TEST_BASE_DIR, TEST_DIRS

    parser = argparse.ArgumentParser(description="WhisperX test runner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making any changes"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Test only a specific directory (overrides the config/default list)"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Override the base directory (default: config.yaml orchestration.source_dir)"
    )
    parser.add_argument(
        "--session-dirs",
        nargs="+",
        default=None,
        help="Override the session dirs to test (default: config.yaml session_dirs)"
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip comparing produced output against tests/expected/ goldens "
             "(structure validation still runs)."
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not wipe each session's output dir before it runs "
             "(by default the output is cleaned first so the run starts fresh)."
    )
    parser.add_argument(
        "--tool-timeout",
        type=int,
        default=DOCKER_RUN_TIMEOUT,
        help=f"Per-tool wall-clock timeout in seconds (default: "
             f"{DOCKER_RUN_TIMEOUT}, matching the Docker layer's own cap). "
             f"Raise it for slow or large transcriptions."
    )

    args = parser.parse_args()

    # CLI overrides win over the config-derived defaults.
    if args.source_dir:
        TEST_BASE_DIR = Path(args.source_dir)
    if args.session_dirs:
        TEST_DIRS = args.session_dirs

    logger.info(f"Test base: {TEST_BASE_DIR}")
    logger.info(f"Session dirs: {args.dir or TEST_DIRS}")

    runner = TestRunner(compare=not args.no_compare, clean=not args.no_clean,
                        tool_timeout=args.tool_timeout)

    if args.dir:
        runner.test_single_directory(args.dir, dry_run=args.dry_run)
        ok = runner.print_summary()
    else:
        ok = runner.run_all_tests(dry_run=args.dry_run)

    # Exit non-zero when a golden comparison failed, so CI / the shell can see it.
    # (Vacuously passes on --dry-run or --no-compare: nothing was compared.)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
