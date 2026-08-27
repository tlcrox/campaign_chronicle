# Campaign Chronicle
This package allows you to automate conversion of captured video (or audio) into a transcription for use as
the basis of a chronicle of your exploits.  It started as a project to take Video recordings of Champions sessions
and turn them into Word documents with screen captures to give players a post-mortem read, and/or reference.
No creative re-construction is built in.  This stands on WhisperX to do the diarization and on PySceneDetect to
automate the detection of change in scenes.

I found PySceneDetect to be too fine grained even with constraining the area of the screen it paid attention to.
I built two other tools to help with Screen changes.  I highly recommend using CaptureScreens to pre-process your 
video for the scenes you care about.

There are many other packages out there that stand on WhixperX.  I use this one regularly to convert a Champions
and a Harn campaign into post-mortem issues.


## WhisperX + PySceneDetect Pipeline

A GPU-accelerated pipeline that turns a tabletop-session recording — a video,
and/or per-speaker Craig audio tracks — into a single speaker-tagged transcript
aligned with scene screenshots, assembled into a Word storyboard. It bundles:

- **[WhisperX](https://github.com/m-bain/whisperX)** — Whisper transcription with word-level alignment and optional speaker diarization.
- **[PySceneDetect](https://www.scenedetect.com/)** — scene-change detection with snapshot extraction, for aligning imagery to the transcript.

> **Hardware limits — user beware.** Nothing throttles concurrency. Launching
> many runs at once each claims a full GPU/CPU slice; oversubscribing a single
> GPU will OOM. Scale to your hardware.

This README is the entry point and covers architecture. See also:

| Doc | What's in it |
|---|---|
| **[ORCHESTRATION.md](ORCHESTRATION.md)** | How to run it — `orchestrate.py` and every step, file locations, integration points. |
| **[HISTORY.md](HISTORY.md)** | How it evolved, the refactor, bugs fixed, current state (what broke and how it's being restored). |
| **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** | Pending work, plus deferred work (embeddings, security, infrastructure). |
| **[TESTING.md](TESTING.md)** | What tests exist and how to run them.

---

## Prerequisites

1. **NVIDIA GPU** with recent drivers (WhisperX; PySceneDetect runs on CPU).
2. **Docker** / Docker Desktop, plus the **NVIDIA Container Toolkit** so Docker sees the GPU
   ([Linux guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html); on Windows enable WSL2 GPU support).
3. For diarization only (e.g. detecting speach): a free Hugging Face token (below).

GPU sanity check:
```
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## First-time setup

```
cd <<install dir>>
copy .env.example .env        :: Windows  (cp on Linux/macOS/WSL)
```

**Create and activate the environment.** `uv sync` builds `.venv` and installs
the `pipeline` and `cc_stages` packages into it; activating is what puts them on
your path. Every command below assumes an active venv.

```
uv sync
```
```powershell
.venv\Scripts\Activate.ps1        # PowerShell
.venv\Scripts\activate.bat        # cmd.exe
source .venv/bin/activate         # Git Bash / WSL / Linux / macOS
```

Without it you get `ModuleNotFoundError: No module named 'pipeline'`. (Don't want
to activate? Prefix any single command with `uv run`.)

**Build the two images** — each has its own compose file; there is no root
`docker-compose.yml`:

```
docker compose -f compose/whisperx/docker-compose.yml build whisperx
docker compose -f compose/pyscenedetect/docker-compose.yml build scenes
```

For diarization, put your token in `.env` (secrets only):
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```
enable it in `config.yaml` (`whisper.diarize: true`), and accept both model
licenses while logged into Hugging Face:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

cd config
copy config_sample.yaml config.yaml
make edits to reflect your directories, most specifically your source material

See Configuration in ORCHESTRATION.md for the different settings that will drive the tool

You can verify your configuration by running 
python scripts\verify_setup.py

Then jump to **[ORCHESTRATION.md](ORCHESTRATION.md)** to run it (the usual first
command is `python scripts/orchestrate.py --dry-run`, which uses the
`session_dirs` from `config/config.yaml`).

---

## Automating the processing of a whole hierarchy of recordings

`orchestrate.py` processes an **explicit list** of session folders — either
`--session-dirs` on the command line or `orchestration.session_dirs` in
`config.yaml`. It does **not** walk directory trees or auto-discover folders. At one
point it did, however over the expected use, auto-walking seemed counter-intuitive.
Discovery is deliberately the caller's job, so any naming scheme or nesting works.

To process a whole tree, iterate it in a small wrapper and call orchestrate once
per folder. Or you can specify multiple sub-folders within config.yaml. If you use 
external discovery, filtering (e.g. only `Week*`) is just a filter in the loop 
— no pipeline flags needed.  Or you can specify multiple sub-folders within config.yaml.

PowerShell:
```powershell
# every immediate subdirectory of a recordings tree
Get-ChildItem "D:\recordings" -Directory |
  ForEach-Object { python scripts\orchestrate.py --session-dirs $_.FullName }

# only folders named Week11..Week15
Get-ChildItem "D:\recordings" -Directory -Filter "Week1[1-5]" |
  ForEach-Object { python scripts\orchestrate.py --session-dirs $_.FullName }
```

Bash (Linux/macOS/WSL):
```bash
for d in /recordings/*/; do
  python scripts/orchestrate.py --session-dirs "$d"
done
```

Prefer one call per folder (isolated, restartable, parallel-friendly), or pass
several at once: `--session-dirs A B C`. You can also process the individual
stages (`transcribe.py`, `detect_scenes_*`, …) the same way — each takes a
single `--session-dir`.

---

## Architecture

### What it produces

From the session recording, a speaker-tagged transcript aligned with scene
screenshots, assembled into a Word storyboard. Alignment key: each scene's
`Start Time (seconds)` (now also `Start Timecode`) is matched against the
WhisperX segment `start`.

### Process flow

The pipeline supports **two distinct speaker workflows** that produce identical output:

**Workflow A: Audacity (Per-Speaker Audio)**
- Input: Audacity project (.aup3) with per-speaker audio exports
- Per-speaker JSON files → `map_speakers` resolves character names from filenames via `speaker_config.json` (`filename_mapping`) → `clean_transcription` → interleave by timestamp → unified transcript with character names
- developer's source for such was Craig added and run within a Discord gaming session.

**Workflow B: Video-only (Diarization)**
- Input: Single or multiple video files (no Audacity project)
- Per-video JSON files (with generic SPEAKER_XX IDs) → `map_speakers` applies speaker_config.json mapping → `clean_transcription` → interleave by timestamp → unified transcript with character names
- developer's source for such was running OBS to capture screen and video - generally the Discord window.

Both workflows produce identical final output: character names, filler removed, segments interleaved chronologically. (Confidence/hallucination filtering is also available but opt-in — see the `clean_transcription` stage.)

```
                 ┌──────────────── orchestrate.py (conductor) ────────────────┐
                 │ discover sessions → stage mounts → run stages → write back │
                 └────────────────────────────────────────────────────────────┘

  audio/video ──► TRANSCRIBE ─────► cc_output/transcriptions/*.json   (WhisperX, optional diarization)
   (session input)  (transcribe-only: transcribe.py, source auto-detected)
       │              │
       │              ▼
       │          MAP_SPEAKERS ── resolve speakers in place (auto-detects workflow)
       │              │             ├─ Workflow A: filename → name (speaker_config.json: filename_mapping)
       │              │             └─ Workflow B: SPEAKER_XX → name (speaker_config.json: global/session mapping)
       │              ▼
       │          CLEAN ── drop filler (speaker_config.json: filler_phrases)
       │              │      + confidence/hallucination pass (opt-in: whisper.clean.enabled)
       │
  video ───────►├─ SCENES ──────────► cc_output/scenes_output/<video>-Scenes.csv + Scene-01-NNN.jpg
                │   ├─ single ROI  (scenes)        one fixed crop
                │   └─ multi  ROI  (scenes_multi)  per-video time-based crops
                │        split → detect per segment → merge_segments (reassemble)
                └─ (or manual_source screens, when the session supplies them)
       │
  mapped+cleaned transcripts + CSV + images ──► MERGE ──► cc_output/<session>_storyboard.docx
                     ├─ combine_transcripts (build sources → interleave by timestamp)
                     └─ combine_scenes (merge multi-video CSVs + images, Scene-MM-NNN.jpg)
                        (merged results land in cc_output/combined_output/)
```

Every generated path above is under **`cc_output/`** — one folder per session
holding everything the tool produces (see "Where output goes"). Inside the
containers, the *subdirectory* being written is mounted as `/session_output`
(never `cc_output/` itself — each stage points that same container path at a
different subdirectory), and the session is mounted read-only as
`/session_input`. There are no shared `audio`/`video` folders. See
ORCHESTRATION.md, "Why the container path is not called `cc_output`".

### Execution model — two contexts

| Context | Runs | Configured by |
|---|---|---|
| **Host** (Windows) | `orchestrate.py`, the `pipeline.*` helpers, `speakers.*` | `config.py` reads `config.yaml` directly |
| **Container** (Linux/CUDA) | `entrypoint.sh` (whisperx), `detect_scenes.sh`, `detect_scenes_multi.sh` | env vars + a read-only mount of `config.yaml` |

Each image has its own compose file under `compose/<image>/`, so every
invocation names one with `-f` (there is no root `docker-compose.yml`). Static
mounts, relative to the compose file's own directory:

| host | container | why |
|---|---|---|
| `docker/<image>/` | `/usr/local/bin/scripts` | entrypoint + helper shell scripts |
| `src/` | `/usr/local/bin/pylib` | the importable `pipeline` package (`PYTHONPATH`) |
| `config/config.yaml` | `/usr/local/bin/config.yaml:ro` | settings |

Input and output are per-session bind mounts supplied at run time — the session read-only as
`/session_input`, its output dir writable as `/session_output` — which is what
lets sessions run concurrently without colliding.

Two images, not one: `whisperx:local` (GPU, transcription) and
`pyscenedetect:local` (CPU-only, scene detection), serving three services —
`whisperx`, `scenes`, `scenes_multi`.

### Configuration
#### `config.yaml` vs `.env`

Two files, one job each. They do **not** overlap:

- **`config.yaml` — the source of truth for all tunables.** Model, language,
  scene-detection settings, ROI, orchestration (source dir, session dirs), merge/storyboard, speaker settings, and the hotwords hint file. Read by `config.py`
  (`Config`), which every host-side script uses, and which is bind-mounted
  read-only into the containers. This is the file you edit to change behaviour.
  `whisper.hotwords_file` (relative to `source_dir`, default
  `whisperx_initial_prompt.txt`) supplies hint phrases passed to WhisperX as
  `--hotwords`; missing file → skipped.
- **`.env` — secrets only.** In practice that means `HF_TOKEN` (needed only when
  `whisper.diarize: true`). `.env` is gitignored; `config.yaml` is committed, so
  secrets must never go in it. Do not put tunables in `.env`.

`config.py` locates `config.yaml` via `WHISPERX_CONFIG` → walking up from
`__file__` → cwd, so the same code resolves it on the host and in the container.
The per-stage walk up from `--session-dir` is bounded at two directories (see
[TESTING.md](TESTING.md)), and the file that wins is logged.
Path-typed values (`source_dir`, mounts) are host-only. Env vars, if set, still
override individual keys inside `config.py` — but that's an advanced escape
hatch, not the normal path. (Full key reference in
[ORCHESTRATION.md](ORCHESTRATION.md#configuration).)

**How config reaches each container:** both are `config.yaml`-driven by the same
mechanism — the host tool reads `config.yaml` and passes the values to
`docker compose run` as `-e KEY=VALUE` flags, which override the compose
`environment:` block:

- **Scene detection** — the tool passes `SCENE_THRESHOLD`, `SCENE_ROI`, … from
  `config.yaml`.
- **Transcription** — the tool passes `WHISPER_MODEL`, `WHISPER_DIARIZE`, … from
  `config.yaml` (`pipeline/transcribe/docker_env.py::whisper_env`).

**No layer below `config.yaml` supplies a value.** Both compose files list the
config-owned variables by name only, so a bare `docker compose run` can still set
them from your shell, and neither file carries a copy of a setting. The
entrypoints refuse rather than guess: an unset `SCENE_THRESHOLD` or
`WHISPER_MODEL` exits naming the variable and the `config.yaml` key that owns it,
instead of quietly detecting scenes five times less sensitively. The exceptions
are the genuinely optional ones — `SCENE_ROI` (empty = full frame),
`WHISPER_LANGUAGE` (empty = auto-detect), `WHISPER_HOTWORDS` and
`WHISPER_INITIAL_PROMPT` (empty = flag omitted) — and the container's own layout
(`VIDEO_DIR`, `AUDIO_DIR`, `OUTPUT_DIR`, …), which is not configuration.

So editing `config.yaml` changes what every container does. `HF_TOKEN` is the
one exception — it flows from `.env` → `docker-compose.yml` → the `whisperx`
container (secret, consulted only when diarization is on).


### Code layout

```
src/                              installed as the `pipeline` package
  pipeline/
    config.py                     config surface (config.yaml loader + resolution)
    common/                       timecode, docker (compose wrapper), mounts, scenes, sessions
    transcribe/                   map_speakers (speaker-ID stage), clean_transcription
                                  (filler + opt-in confidence), remove_hallucination
    scenes/                       roi (RoiFile + resolve_single_roi), extract_segments,
                                  merge_segments
    merge/                        storyboard (docx), combine_transcripts (merge core +
                                  Audacity front end), combine_scenes (cross-video combine)

scripts/                          installed as the `cc_stages` package
  orchestrate.py                  thin conductor (resolve sessions -> call stages -> report)
  cc_stages/                      one module per pipeline stage, each runnable standalone
  verify_setup.py                 pre-flight environment check (cross-platform)

docker/<image>/                   Dockerfile + entrypoints + requirements per image
  whisperx/                       entrypoint.sh
  pyscenedetect/                  detect_scenes.sh, detect_scenes_multi.sh
compose/<image>/                  docker-compose.yml per image
config/                           config.yaml, config_sample.yaml, speaker_config.json
tests/                            unit/, integration/, test_source/ (inputs), expected/ (goldens)
```

Both `src` and `scripts` are import roots declared in `pyproject.toml`, so
`pipeline.*` and `cc_stages.*` resolve anywhere once `uv sync` has run. No module
manipulates `sys.path`.

Each stage is importable *and* independently runnable — all three work:

```
python -m cc_stages.merge_scenes --session-dir ...        # as a module
python scripts/cc_stages/merge_scenes.py --session-dir ...  # as a script
from cc_stages import merge_scenes; merge_scenes.run(...)  # composed by orchestrate
```

### Where output goes

Everything the tool generates lands under **one folder per session**:

```
Week 13/                     <- your source, untouched apart from this one folder
  Week13_a.mp4
  roi_history.json
  screens/                   <- OPTIONAL hand-captured scenes (INPUT, see below)
  cc_output/                 <- everything generated; safe to delete wholesale
    transcriptions/          per-source WhisperX transcripts
    scenes_output/           per-video PySceneDetect CSVs + images
    combined_output/         merged transcript + merged scenes
    Week 13_storyboard.docx  the deliverable
```

One root means "delete everything this tool made" is a single `rmtree`, and a
single `.gitignore` line (`cc_output/`) keeps results out of version control
regardless of which source tree produced them.

**Sending output elsewhere.** Set `output.base_dir` to write to a separate tree
instead, leaving the source completely untouched — useful when the source is
read-only or shared:

```yaml
output:
  base_dir: D:/Chronicle/output    # null (default) = beside the input
```

Output then lands at `<base_dir>/<session path relative to source_dir>/cc_output/...`.
The session's path is used rather than its folder name, because two sources can
each contain a `Week 13`. The `cc_output/` level is preserved either way, so both
layouts produce an identical tree and one comparison works for both.

`CC_OUTPUT_BASE` overrides the config for a single run. A config value resolves
against the config file's directory; the env var resolves against the current
directory, since whoever set it is standing somewhere specific.

### Hand-captured scenes are input, not output

If you capture scenes by hand (see the CaptureScreens tool), they go in the folder
named by `scenes.manual_source` (default `screens/`) at the **session root** —
deliberately *outside* `cc_output/`, because that folder gets deleted and your
captures are irreplaceable.

When that folder is present, orchestrate skips PySceneDetect entirely and the
merge consumes your captures instead. The layout is one subfolder per video, each
containing a `Scenes.csv` (name from `scenes.manual_csv_name`).

The folder name must match on both sides. CaptureScreens writes to
`output.scenes_dir_name`; the pipeline reads `scenes.manual_source`. If they
disagree, orchestrate refuses to run rather than quietly auto-detecting and
building a storyboard from the wrong scenes.

### Key design decisions

- **`RoiFile` is the single ROI parser** (`pipeline/scenes/roi.py`); the ROI (Region of Interest)
  file is read exactly once per run, in `extract_segments`.
- **Multi-ROI offsets are carried in the data, not re-derived** — each segment's
  absolute start is written to `segment_<i>/offset.txt` at split time, so
  `merge_segments` adds it back and never re-reads the ROI file (hence the merged
  CSV carries absolute `Start Timecode`/seconds).
- **One timecode implementation** (`pipeline/common/timecode.py`).
- **`python -m` + `PYTHONPATH=scripts`** is the uniform invocation, host and container.


#### JSON Processing Contract

Both workflows transform WhisperX output through the same pipeline:

**Input:** Raw WhisperX JSON (per-file)
```json
{
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello", "speaker": "SPEAKER_00", "words": [...]}
  ]
}
```

**Output:** Final merged JSON (unified)
```json
{
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello", "speaker": "GM", "words": [...]}
  ]
}
```

Transformations applied, in pipeline-stage order: speaker name mapping (`map_speakers` — A: filename-based, B: speaker_config.json) → cleaning (`clean_transcription` — filler removal, plus an opt-in confidence/hallucination pass) → timestamp interleaving (`merge_transcripts`). The merge does **not** coalesce consecutive same-speaker segments; the storyboard coalesces same-speaker runs within each inter-image window instead.

#### File Naming Convention (Multi-Video)

Scene images use standardized naming: **`Scene-MM-NNN.jpg`**
- **NNN** = scene number within this video (001-999, preserves original numbering)
- **MM** = video sequence number in timeline order (01-99)

Example: Week with 2 videos → Video1: `Scene-01-001.jpg` through `Scene-01-042.jpg` | Video2: `Scene-02-001.jpg` through `Scene-002-028.jpg`

This convention ensures both human readability and proper sorting (`ls`) by video first, then by scene within video.

### Pipeline unification (complete)

The pipeline was refactored into discrete, independently-runnable stages —
transcribe → `map_speakers` → `clean_transcription` → (scenes) → merge → storyboard
— so the Audacity and video paths share the same downstream stages instead of two
parallel code paths:
- **Speaker mapping** is its own stage (`map_speakers`), auto-detecting Workflow A
  (filename token) vs Workflow B (diarization) for every source type; the
  transcribe tools are transcribe-only.
- **Cleaning** is its own stage (`clean_transcription`): filler removal, plus an
  opt-in confidence/hallucination pass (`whisper.clean.enabled`, default off).
- The two per-workflow merges collapsed into one `merge_transcripts(sources)`
  core fed by a source adapter, and `combine_week.py` was split into
  `combine_transcripts.py` + `combine_scenes.py`.

See **[ORCHESTRATION.md](ORCHESTRATION.md)** for how to configure and run the tool
See **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** for the phase-by-phase
record and deferred work.
