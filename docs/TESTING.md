# TESTING

The test framework: the fixtures, how to run the suite, how to verify ROI
behaviour, and the current coverage gaps.

---

## 1. What the suite is

Two layers:

**Unit tests** (fast, deterministic, no video, no Docker) — live beside the code
they exercise and run with `unittest`:

| Test | Exercises |
|---|---|
| `tests/integration/test_config.py` | `config.Config` property access and path resolution (25 active properties, incl. hotwords file reading) |
| `tests/unit/pipeline/common/test_timecode.py` | timecode parsing/formatting |
| `tests/unit/pipeline/common/test_docker.py` | `pipeline.common.docker`: `-e`/`-v` flag injection, the transient-error classifier, retry behavior (subprocess is patched, no real Docker) |
| `tests/unit/pipeline/common/test_mounts.py` | `pipeline.common.mounts.session_mounts`: per-session bind mounts, the read-only guard, the escape-the-session safety check |
| `tests/unit/pipeline/common/test_sessions.py` | `pipeline.common.sessions.find_sessions`: explicit-list resolution, fail-loud with no `Week*` auto-discovery |
| `tests/unit/pipeline/common/test_scenes.py` | `pipeline.common.scenes`: scene-image naming (`Scene-{video:02d}-{scene:03d}.jpg`) and the host-side rename after the docker run |
| `tests/unit/pipeline/transcribe/test_remove_hallucination.py` | hallucination-cleanup thresholds |
| `tests/unit/pipeline/transcribe/test_clean_transcription.py` | `clean_transcript`: filler pass, opt-in confidence pass, combined-file exclusion, dry-run |
| `tests/unit/pipeline/transcribe/test_docker_env.py` | `pipeline.transcribe.docker_env.whisper_env`: the config → `WHISPER_*` env-var builder |
| `tests/unit/pipeline/scenes/test_roi.py` | `RoiFile` + `resolve_single_roi` |
| `tests/unit/pipeline/scenes/test_merge_segments.py` | multi-ROI segment CSV reassembly |
| `tests/unit/pipeline/merge/test_combine_transcripts.py` | `combine_transcripts.py`: `merge_transcripts` core (parallel/serial offsets), Audacity front end, filler detection |
| `tests/unit/pipeline/merge/test_combine_scenes.py` | `combine_scenes.py`: scene CSV + image merge (2-part video/scene key) |
| `tests/unit/pipeline/merge/test_storyboard.py` | `storyboard.py`: transcript parsing and document generation |
| `tests/unit/test_merge_scenes.py` | `merge_scenes.py`: scene discovery and merging |
| `tests/unit/test_merge_transcripts.py` | `build_merge_sources`: the `merge_transcripts` stage's source adapter (offset by workflow, no speaker override, bad-file skip) |
| `tests/unit/test_apply_speaker_mapping.py` | `map_speakers` (Workflow A filename + Workflow B diarization resolvers) and the `require_diarization_for_mapping` guard |

Run all of them from `tests/`:

```bash
cd tests
python3 -m unittest discover -p 'test_*.py'
```

**Integration runners** (slow; need Docker + fixture directories) — drive the
real tools/orchestrator against session directories on disk:

- `run_tests.py` — runs each stage tool (transcribe → `map_speakers` →
  `clean_transcription` → scene detect → merge) on a set of session directories
  and validates the output structure. Scene detection resolves its ROI from
  `config.yaml` (`scenes.roi`); it does not pass `--roi`.
- `run_orchestrate_tests.py` — runs `tests/orchestrate.py` against one or more
  session directories, **once per session, each with its own timeout** (`--timeout`,
  default 600s), reporting pass/fail per session. (Previously all sessions shared
  a single timeout, so whichever was running when the budget expired got killed.)
  Forwards `--source-dir`, `--session-dirs`, and `--dry-run` to `orchestrate.py`.

**Config lives with the source.** A `config.yaml` at a source root (e.g.
`tests\test_source\config.yaml`, next to its `speaker_config.json`)
governs runs against that source. Resolution order:

```
WHISPERX_CONFIG env  >  --config  >  (orchestrate) <--source-dir>/config.yaml
  >  (tools) nearest config.yaml walking up from --session-dir  >  walk up from the code dir
```

The tools' upward walk stops **two directories above the session** — by
convention a source root holds `config.yaml` and a session sits at most two under
it (`Weeks/Week 13`). Past that the search gives up rather than climbing out of
the source tree and adopting an unrelated `config.yaml`; `config.yaml` is a
common filename, and an unbounded walk reaches the drive root. Whichever file
wins is logged as `Config: <path> (<why>)` at the start of every stage.

Because `source_dir` defaults to the config file's own directory when
`orchestration.source_dir` is unset, a source folder is self-describing — drop a
`config.yaml` beside the fixtures and both `orchestrate` (`--source-dir …`) and
the individual tools (walking up from `--session-dir`) find it, with anything the
file omits falling back to `config.py`'s code defaults rather than the code-root
`config.yaml`. Point at a different config any time with `--config`:

```bash
cd tests\integration
python3 run_tests.py --dry-run                          # show what would run, using config.yaml's session_dirs
python3 run_tests.py --dir SingleVideo                   # one named fixture
python3 run_tests.py --source-dir <<project-root>>\tests\test_source --dir MultiVideo
python3 run_orchestrate_tests.py --session-dirs SingleVideo MultiVideo
python3 run_orchestrate_tests.py --dry-run
```

Fixture data lives at `<install>\campaign_chronical\tests\test_source`.

---

## 2. Fixtures

Seven directories, each covering a distinct input shape. Transcription tool is
handled by the single `transcribe.py` stage, which auto-detects the source
(embedded video audio, loose audio, or an Audacity `.aup`/`.aup3` project).

| Fixture | Shape | Exercises | Expected output |
|---|---|---|---|
| **SingleVideo** | 1 video, embedded audio | baseline: transcribe → single-ROI scenes → merge | `cc_output/transcriptions/*.json`, `cc_output/scenes_output/<video>/…-Scenes.csv` + `Scene-01-###.jpg`, `cc_output/…storyboard.docx` |
| **MultiVideo** | 3 videos, embedded audio | per-video indexing | separate CSV per video; images `Scene-01-*`, `Scene-02-*`, `Scene-03-*` |
| **SingleCraig** | 1 video + Audacity project | Audacity audio path | transcript from `.aup` audio; single-ROI scenes |
| **MultiCraig** | 2 videos + shared Audacity + `roi_history.json` | multi-video + per-video ROI | one CSV per video; per-video ROI applied |
| **MultiROI** *(sic)* | 3 videos + `roi_history.json` | time-based multi-ROI | multi-ROI scene detection per time segment, reassembled |
| **SingleROI** | 1 video | single-ROI scene detection using `scenes.roi` from `config.yaml` | single-ROI scenes |
| **ScenesCraig** | 1 video + Audacity, pre-populated manual `screens/` | reference / regression target | existing `Scene-01-001…055.jpg` to compare against |

Notes:
- The ROI file in `MultiROI` and `MultiCraig` is `roi_history.json` in the
  **hierarchical (video-keyed)** format — top-level video filenames, each with
  `_metadata.fps` and `HH:MM:SS → {frame, roi}`. ROI values are
  `"x1 y1 x2 y2"` (top-left & bottom-right corners). This is the only supported
  ROI-file format — a flat, non-video-keyed layout is rejected.
- Scene image naming is `Scene-{VIDEO:02d}-{SCENE:03d}.jpg` where `VIDEO` is the
  global video index and `SCENE` is the scene within that video.

---

## 3. Running a single tool by hand

```bash
# Transcription (pick by audio source; transcribe-only)
python3 tests/cc_stages/transcribe.py --session-dir tests/test_source/SingleVideo
python3 tests/cc_stages/transcribe.py --session-dir tests/test_source/SingleCraig

# Speaker mapping, then cleaning (transcript-side stages, after transcription)
python3 tests/unit/map_speakers.py        --session-dir TestWhisper/SingleVideo
python3 tests/unit/clean_transcription.py --session-dir TestWhisper/SingleVideo

# Scene detection
python3 tests/unit/detect_scenes_single_roi.py --session-dir TestWhisper/SingleVideo
python3 tests/unit/detect_scenes_multi_roi.py  --session-dir TestWhisper/MultiROI

# Merge → storyboard
python3 tests/unit/merge_transcripts.py   --session-dir TestWhisper/SingleVideo
python3 tests/unit/merge_scenes.py        --session-dir TestWhisper/SingleVideo
python3 tests/unit/generate_storyboard.py --session-dir TestWhisper/SingleVideo
```

Validation (what `run_tests.py` checks), all under the session's `cc_output/`:
a `transcriptions/` dir with JSON, a `scenes_output/<video>/` dir with a
`*-Scenes.csv` and `Scene-*.jpg` images, and
a `storyboard.docx`.

---

## 4. ROI verification

Three scene-detection modes:

- **Full frame** — no ROI; PySceneDetect analyses the whole frame.
- **Single ROI** — one fixed crop for the whole video. Resolved by
  `resolve_single_roi()`: CLI `--roi` > `config.yaml scenes.roi` > full frame.
  An explicit `--roi ""` forces full frame.
- **Multi-ROI** — time-varying, per-video crops from `roi_history.json`
  (`detect_scenes_multi_roi.py` / `detect_scenes_multi.sh`).

ROI is PySceneDetect's `--crop`, given as `x1 y1 x2 y2`. `detect_scenes.sh`
passes the value straight through, no width/height conversion.

Confirming a crop is actually applied: run the same fixture full-frame vs. with
an ROI and compare scene counts — a crop that excludes changing UI/chat should
yield **fewer, different** cuts than full frame. For multi-ROI, check the log:
the tool prints each video, its segments, and the active ROI per segment before
running.

```bash
# full-frame baseline
python3 tests/unit/detect_scenes_single_roi.py --session-dir TestWhisper/SingleVideo --roi ""
# config/CLI ROI
python3 tests/unit/detect_scenes_single_roi.py --session-dir TestWhisper/SingleVideo --roi "400 445 925 745"
```

---

## 5. Known coverage gaps

- **VIDEO_INDEX not incremented** (MultiVideo/MultiCraig/MultiROI) — multi-video
  runs may emit all images as `Scene-01-*` instead of `01/02/03`. Runners don't
  yet *assert* per-video indexing.
- **Per-video ROI** (MultiCraig) — not yet implemented.
- **Multi-ROI is not exercised by `run_tests.py`** — it currently skips that
  branch; needs a fixture run end-to-end.
- **Single-ROI source variation** — `run_tests.py` only exercises the
  config-ROI path; CLI-override and full-frame fallback aren't covered
  end-to-end (though `resolve_single_roi` itself is unit-tested).
- **No diarization-on / scene-parameter variation** in either integration
  runner.

Planned/deferred work beyond test coverage is tracked in `IMPLEMENTATION_PLAN.md`.
