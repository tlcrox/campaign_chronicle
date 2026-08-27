# Running the WhisperX Pipeline

How to run the pipeline — `scripts/orchestrate.py` first (the normal path),
then each individual step for when you need to run a piece on its own.
Parameters for every step are listed inline.

> Everything is driven by `config.yaml` (loaded by `config.py`). CLI flags and
> environment variables override it. See **[Configuration](#configuration)** at
> the end.

---

## Prerequisites

- Docker + Docker Compose, with the NVIDIA container toolkit (GPU).
- A `.env` with `HF_TOKEN` if you use diarization (see `.env.example`).
- `config/config.yaml` (mounted read-only into the containers).
- Source sessions on disk under `orchestration.source_dir`.

### Activate the environment first

**Every command in this document assumes the project virtualenv is active.** The
`pipeline` and `cc_stages` packages are installed into `.venv`, so without it
`python scripts/orchestrate.py` fails with `ModuleNotFoundError: No module named
'pipeline'`.

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```
```bat
:: Windows cmd.exe
.venv\Scripts\activate.bat
```
```bash
# Git Bash / WSL / Linux / macOS
source .venv/bin/activate      # .venv/Scripts/activate under Git Bash
```

Your prompt gains a `(campaign-chronicle)` prefix once it is active. If `.venv`
does not exist yet, or you have changed dependencies, create/refresh it with:

```
uv sync
```

(If you would rather not activate anything, prefix any single command with
`uv run` — `uv run python scripts/orchestrate.py --dry-run` — which resolves the
right interpreter without touching your shell.)

### Build the images

Two images, each with its own compose file — there is no root
`docker-compose.yml`:

```
docker compose -f compose/whisperx/docker-compose.yml build whisperx
docker compose -f compose/pyscenedetect/docker-compose.yml build scenes
```

### Quick sanity check

```
python scripts/verify_setup.py
```

It checks the Python version, that `pipeline`/`cc_stages` are importable, that
both compose files validate under `docker compose config`, that the images are
built, that `.env` carries an `HF_TOKEN` when `whisper.diarize` is on, and that
the configured `source_dir` and `session_dirs` actually exist. Exit status is 0
unless something is genuinely broken, so it works as a pre-flight check.

Add `--source-dir <path>` to check a directory other than the configured one, or
`--quiet` to show only problems.

---

## 1. scripts/orchestrate.py — the normal path

`scripts/orchestrate.py` processes an explicit list of session folders: each
tool bind-mounts the session directory (read-only for input, writing its own
output back into it), runs transcription then its downstream stages (speaker
mapping → cleaning) and/or scene detection, and (optionally) merges everything
into the storyboard.

Run it from the repo root:

```bash
python scripts/orchestrate.py --dry-run                       # preview (uses config session_dirs)
python scripts/orchestrate.py --session-dirs "SingleROI"        # process one folder
python scripts/orchestrate.py --session-dirs "MultiVideo" --roi-file roi_history2.json
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--source-dir PATH` | `orchestration.source_dir` | Base dir that relative `--session-dirs` resolve against. |
| `--session-dirs A B …` | `orchestration.session_dirs` | Explicit session folders to process (required; no auto-discovery). |
| `--parallel N` | `orchestration.parallel_workers` | Concurrent sessions (GPU-bound; start at 1). |
| `--roi-file NAME` | `$ROI_FILE`, else `scenes.roi_file` | ROI JSON filename, looked for in each session dir. If present there, triggers multi-ROI scene detection for that session; if absent, that session falls back to single-ROI. |
| `--no-scenes` | off | Skip scene detection. |
| `--no-transcription` | off | Skip transcription (scenes only). |
| `--merge-only` | off | Skip processing; only run the merge/combine step on existing outputs. |
| `--config PATH` | project root `config.yaml` | Use an alternate config file. |
| `--dry-run` | off | Show what would happen; change nothing. |
| `--verbose` | off | DEBUG logging. |

### Common recipes

```bash
# Preview (uses config.yaml session_dirs)
python scripts/orchestrate.py --dry-run

# Multi-ROI run (roi_history2.json must exist inside the session directory)
python scripts/orchestrate.py --session-dirs "MultiRoi" --roi-file roi_history.json

# Scenes only (no transcription)
python scripts/orchestrate.py --session-dirs "SingleROI" --no-transcription --roi-file roi_history.json

# Re-build the combined storyboard from outputs already on disk
python scripts/orchestrate.py --merge-only --session-dirs "SingleROI"
```

After a run, each session folder gains a single `cc_output/` folder holding
everything generated: `cc_output/transcriptions/` (transcripts),
`cc_output/scenes_output/` (`<video>-Scenes.csv` + `Scene-MM-NNN.jpg`),
`cc_output/combined_output/` (merged results) and the storyboard `.docx`.
MM is the Mth video, NNN the Nth captured scene. Deleting `cc_output/` removes
every generated artifact and leaves the source untouched; set `output.base_dir`
to write the whole tree elsewhere instead.

The final summary line reports counts (`completed`, `skipped`, `failed`); if
any session failed, a `Failures:` block below it lists which session and
which step (transcription / scene detection / merge) failed, and the process
exits non-zero.

---

## 2. Transcription (WhisperX)

Transcribes everything under the mounted session to JSON. There are no shared
`./audio` / `./output` folders any more: each run bind-mounts one session
read-only as `/session_input` and its own output dir as `/session_output`, which
is what makes concurrent sessions safe. The stages supply those mounts, so you
normally drive this through orchestrate rather than by hand.

```bash
# Normal use — the stage builds the mounts and env for you:
python scripts/cc_stages/transcribe.py --session-dir "<source>/Week 14"

# By hand, if you need to. Note the -f: there is no root compose file.
docker compose -f compose/whisperx/docker-compose.yml run --rm \
  -v "<abs path to session>:/session_input:ro" \
  -v "<abs path to session>/cc_output/transcriptions:/session_output" \
  -e AUDIO_DIR=/session_input -e OUTPUT_DIR=/session_output \
  whisperx
docker compose -f compose/whisperx/docker-compose.yml run --rm whisperx --help
```

For a specific session directory (the path `scripts/orchestrate.py` itself
uses), run the matching tool directly instead — see §1's flags and
`scripts/cc_stages/transcribe.py` — one stage for every source type, which it
detects itself (Audacity project / loose audio / video with embedded audio).

Configured by `config.yaml: whisper:` / env vars (override per run):

| Env | config key | Default |
|---|---|---|
| `WHISPER_MODEL` | `whisper.model` | `large-v3-turbo` |
| `WHISPER_LANGUAGE` | `whisper.language` | `en` (empty = auto-detect) |
| `WHISPER_COMPUTE_TYPE` | `whisper.compute_type` | `float16` |
| `WHISPER_BATCH_SIZE` | `whisper.batch_size` | `8` |
| `WHISPER_OUTPUT_FORMAT` | `whisper.output_format` | `json` |
| `WHISPER_DIARIZE` | `whisper.diarize` | `true` |
| `WHISPER_HOTWORDS` | `whisper.hotwords_file` (file contents) | `""` |
| `WHISPER_INITIAL_PROMPT` | `whisper.initial_prompt_file` (file contents) | `""` |
| `HF_TOKEN` | (from `.env`) | — |

The **Default** column is `config.py`'s, not the container's: nothing below
`config.yaml` supplies a value. `entrypoint.sh` exits naming the variable if one
it needs is missing, so a bare `docker compose run` must pass them itself. The
exceptions are the optional ones — an empty `WHISPER_LANGUAGE` means auto-detect,
and empty hint values omit their flags.

Diarization needs `HF_TOKEN` (read scope) with the `pyannote/speaker-diarization-3.1`
and `pyannote/segmentation-3.0` licenses accepted.

### Recognition hints — two files, two different flags

WhisperX offers two ways to steer recognition, and they are NOT interchangeable.
Each has its own config key and its own file beside the recordings, both relative
to `orchestration.source_dir`, both optional (missing or empty → flag skipped,
transcription runs unchanged).

| config key | default file | flag | scope |
|---|---|---|---|
| `whisper.hotwords_file` | `whisperx_hotwords.txt` | `--hotwords` | the **whole run** |
| `whisper.initial_prompt_file` | `whisperx_initial_prompt.txt` | `--initial_prompt` | the **first window only** |

**`--hotwords`** biases recognition toward specific terms for the entire
transcription. Rare proper nouns and campaign jargon belong here — "Mysteria",
which WhisperX otherwise hears as "Mysterio" or "mystery".

**`--initial_prompt`** seeds the decoder's first window, establishing spelling and
style conventions — "DEX" rather than "decks", "SPD" rather than "speed". Because
it primes only the opening, its influence fades across a long session, so any term
that must hold for three hours belongs in `hotwords_file` even if it reads like a
prompt.

Both files' contents are collapsed to a single line (whitespace normalised, double
quotes stripped) so the value survives shell quoting.

Name each file after the flag it feeds. These two were crossed for a long time —
`hotwords_file` pointed at `whisperx_initial_prompt.txt`, so prompt text was sent
to `--hotwords` and `--initial_prompt` was never passed at all. Matching names to
flags is what makes that kind of mistake visible.

The transcribe tools are **transcribe-only**. Two stages run between transcription
and the merge, each a standalone tool wired into orchestrate and `run_tests.py`:

### Speaker mapping — `map_speakers`

Resolves each per-source transcript's speakers in place, auto-detecting the
workflow (filename token for Audacity, `SPEAKER_XX` for diarization). See
§"speaker_config.json" below.

```bash
cd scripts
python -m tools.map_speakers --session-dir <session>      # or tools/map_speakers.py
```

### Clean transcription — `clean_transcription`

Runs after mapping, before merge. Two passes applied in place to each per-source
transcript:

- **filler** — drops throwaway utterances listed in `speaker_config.json`'s
  `filler_phrases`.
- **confidence / hallucination** — drops low-confidence / silence / loop segments
  using `config.yaml: whisper.clean` thresholds. **Opt-in**: only runs when
  `whisper.clean.enabled` is true (default off), so it's a no-op until enabled.

```bash
cd scripts
python -m tools.clean_transcription --session-dir <session>
```

The confidence pass reuses `pipeline.transcribe.remove_hallucination`, which can
also be run standalone as a pure confidence filter (no speaker/filler logic):

```bash
cd scripts
python -m pipeline.transcribe.remove_hallucination <input_dir> <output_dir>
python -m pipeline.transcribe.remove_hallucination output clean \
    --no-speech-max 0.6 --logprob-min -1.0 --compression-max 2.4
```

---

## 3. Scene detection

Two services, **distinct** and not interchangeable:

### Single ROI — `scenes`

One fixed crop from `scenes.roi` (or `SCENE_ROI`). Never reads an ROI file.

```bash
# Normal use:
python scripts/cc_stages/detect_scenes_single_roi.py --session-dir "<source>/Week 13"

# By hand: VIDEO_DIR/SCENES_DIR point at the per-session mounts.
docker compose -f compose/pyscenedetect/docker-compose.yml run --rm \
  -v "<abs path to session>:/session_input:ro" \
  -v "<abs path to session>/cc_output/scenes_output:/session_output" \
  -e VIDEO_DIR=/session_input -e SCENES_DIR=/session_output \
  scenes
```

| Env | config key | Default |
|---|---|---|
| `SCENE_THRESHOLD` | `scenes.threshold` | `5.0` (lower = more scenes) |
| `SCENE_MIN_LEN` | `scenes.min_length` | `1s` |
| `SCENE_NUM_IMAGES` | `scenes.num_images` | `1` |
| `SCENE_IMAGE_FORMAT` | `scenes.image_format` | `jpg` |
| `SCENE_ROI` | `scenes.roi` | `x1 y1 x2 y2`, currently `"400 445 925 745"` in `config.yaml` |

As above, the **Default** column is `config.py`'s. `detect_scenes.sh` refuses to
start without `SCENE_THRESHOLD`, `SCENE_MIN_LEN`, `SCENE_NUM_IMAGES` or
`SCENE_IMAGE_FORMAT`, naming the `config.yaml` key that owns each. `SCENE_ROI` is
optional: empty (or unset) means full frame.

### Multi ROI (time-varying crop) — `scenes_multi`

Reads an **ROI file** (per-video, time-based crops) — the only consumer of it.
The video is split into time segments, each detected with its own crop, then the
results are stitched back into one CSV + renumbered images.

```bash
# The ROI file sits in the SESSION directory (mounted read-only at
# /session_input), not in a shared ./video folder. Set ROI_FILE to its name:
docker compose -f compose/pyscenedetect/docker-compose.yml run --rm \
  -v "<abs path to session>:/session_input:ro" \
  -v "<abs path to session>/cc_output/scenes_output:/session_output" \
  -e VIDEO_DIR=/session_input -e SCENES_DIR=/session_output \
  -e ROI_FILE=roi_history2.json scenes_multi
# Usually you just let orchestrate drive it — --roi-file alone selects
# multi-ROI for any session where that file is present:
python scripts/orchestrate.py --session-dirs "SingleROI" --roi-file roi_history2.json
```

`SPLIT_ONLY=true` stops after the split: you get `segment_N.mp4` per time
segment under `TEMP_DIR` and no scene detection at all. It answers "are my
segment boundaries right?" without paying for detection on every segment. Note
it does not write the per-segment `offset.txt`, so that output is deliberately
not consumable by the host-side reassembly — it is for looking at.

`DEBUG=true` traces segment extraction: the ROI file it resolved, each segment's
range and crop, and the size of every split file.

Both are container-only switches with no `config.yaml` counterpart, so they have
to be passed by hand — no stage sets them:

```bash
docker compose -f compose/pyscenedetect/docker-compose.yml run --rm \
  -v "<abs path to session>:/session_input:ro" \
  -v "<abs path to session>/cc_output/scenes_output:/session_output" \
  -e VIDEO_DIR=/session_input -e SCENES_DIR=/session_output \
  -e ROI_FILE=roi_history2.json -e SPLIT_ONLY=true -e DEBUG=true scenes_multi
```

Both accept `true`, `1`, `yes` or `on` (any case); anything else is false.

ROI file format — **hierarchical only** (a flat, non-video-keyed layout is
rejected), `x1 y1 x2 y2`:

```json
{ "video.mkv": { "_metadata": {"fps": 60.0},
    "00:00:00": {"frame": 0,     "roi": "479 185 1374 817"},
    "00:04:10": {"frame": 15000, "roi": "909 493 1127 743"} } }
```

The merged `<video>-Scenes.csv` columns are: `Scene Number, Start Timecode,
Start Time (seconds), End Time (seconds), Duration (seconds), Length (frames)`
(timecodes/times are absolute across the whole video).

### Scene helpers (standalone, run from `scripts/`)

```bash
cd scripts
python -m pipeline.scenes.roi roi.json video.mkv              # segments: start|end|roi|desc
python -m pipeline.scenes.roi roi.json --list-videos           # videos in a hierarchical file
python -m pipeline.scenes.roi roi.json --validate               # check the file
python -m pipeline.scenes.extract_segments roi.json video.mkv  # same start|end|roi|desc (the .sh entry)
python -m pipeline.scenes.merge_segments <temp_dir> <out.csv>  # reassemble per-segment CSVs
```

---

## 4. Merge → storyboard

Aligns the transcript against the scene CSV and emits a Word storyboard
(scene image + dialogue per scene). The three merge steps are composed by
`merge_transcripts.run` / `merge_scenes.run` / `generate_storyboard.run` —
`scripts/orchestrate.py`'s `run_merge_tools()` calls all three per session (or
per session passed to `--merge-only`).

Every stage is reached the same way: `cc_stages.<name>.run(session_dir, config=,
dry_run=)`. The module names the stage, `run` is the verb, and whether the work
lives in the `cc_stages` module or is delegated to `pipeline/` is that module's
own business.

```bash
cd scripts
python -m pipeline.merge.storyboard <scene_dir> <session_text>
# scene_dir must contain a *.csv; transcript may be .json (whisperx) or .txt
```

Use `python scripts/orchestrate.py --merge-only` to re-run the merge on
existing outputs, or set `config.yaml: merge: auto_merge: true` to run it
automatically right after processing. Storyboard formatting (title, fonts,
image width, CSV columns) lives under `config.yaml: merge:`.

`pipeline/merge/combine_transcripts.py` holds the transcript merge logic and
`pipeline/merge/combine_scenes.py` the scene merge:
- `merge_transcripts(sources)` — the shared interleave core: shifts each source
  onto the session timeline by its offset and interleaves segments by time,
  producing the `{"segments": [...], "language": ...}` shape. The merge tool's
  `build_merge_sources()` adapter feeds it (parallel offset 0 for Audacity,
  serial cumulative offsets for multi-video). Speakers are already resolved by the
  `map_speakers` stage and filler already removed by `clean_transcription`, so the
  merge itself neither maps speakers nor drops filler.
- `merge_transcripts_audacity()` — a front end that resolves speakers from
  filenames for raw, *pre*-`map_speakers` files; kept only for the legacy
  standalone `scripts/merge_transcripts.py`.
- `merge_scene_csvs()` (combine_scenes.py) — merges per-video scene CSVs onto one
  timeline, keyed on (Video, per-video Scene Number).
- `merge_image_folders()` (combine_scenes.py) — discovers per-video image folders,
  orders them, and renames images to `Scene-NN-NNN.jpg` (the 2-part video+scene key).

---

## speaker_config.json

Located at project root, loaded by `config.yaml` (`speakers.config_file`).
Applied by the **`map_speakers` stage** (`cc_stages.map_speakers.run`, over `pipeline.transcribe.map_speakers.map_speakers`
/ `tools/map_speakers.py`), which runs after transcription and before cleaning/merge.
The stage auto-detects the workflow: **Workflow A** (Audacity per-player files)
resolves the character name from a filename token via `filename_mapping`, while
**Workflow B** (video/audio diarization) renames `SPEAKER_XX` IDs via the
`global_mapping` / `session_mappings` below. The transcribe tools no longer map
speakers themselves. The same file also holds `filler_phrases` (used by the
`clean_transcription` stage). The Workflow B structure:

```json
{
  "global_mapping": {
    "SPEAKER_00": "Craig",
    "SPEAKER_01": "Player1",
    "SPEAKER_02": "Player2"
  },
  "session_mappings": {
    "SingleROI/video": {
      "SPEAKER_00": "Craig",
      "SPEAKER_01": "Alice",
      "SPEAKER_02": "Bob"
    }
  }
}
```

Global mappings apply to all sessions; session-specific overrides apply only to
that session. If `whisper.diarize` is `false` and `speaker_config.json`
exists, mapping is refused with an error (there are no `SPEAKER_XX` tokens to
rename). If `whisper.diarize` is `true` and no `speaker_config.json` can be
found, mapping is skipped with a logged warning rather than an error — you
typically need to see a diarized transcript's `SPEAKER_XX` IDs before you know
what to map them to.

---

## How the containers are invoked

There is no `docker-compose.yml` at the repo root. Each image has its own
compose file, and every invocation names it explicitly:

```
docker compose --env-file "<repo>/.env" -f "compose/whisperx/docker-compose.yml" run --rm whisperx
docker compose --env-file "<repo>/.env" -f "compose/pyscenedetect/docker-compose.yml" run --rm scenes
```

`pipeline.common.docker.compose_run(service, repo_root)` builds that string; the
service -> compose-file mapping lives in `COMPOSE_FILES` in the same module. An
unregistered service raises `UnknownService` rather than producing a command that
fails obscurely.

Three details that are easy to get wrong:

- **Relative paths inside a compose file resolve against that file's directory,
  not the working directory.** That is why every path in `compose/*/` starts with
  `../../`, and why the build context is `../../docker/<image>` — the directory
  holding both the Dockerfile and the `requirements-docker.txt` it copies.
- **`--env-file` is required.** Compose loads `.env` from the *project directory*,
  which defaults to the compose file's folder — not the repo root. Without the
  explicit flag, `${HF_TOKEN:-}` silently resolves to empty and diarization fails
  with an unset token. (`--project-directory` would also find `.env`, but it
  re-bases every relative path above and breaks the mounts.)
- **`HF_TOKEN` is deliberately not passed as `-e`.** It is a secret and flows
  `.env` -> compose -> container. Every other setting comes from `config.yaml` via
  `-e KEY=VALUE`, which overrides the compose `environment:` block.

### Timeouts and retries

**7200 seconds — two hours — is the wall clock for one `docker compose run`**,
applied in `pipeline.common.docker.run_docker_command`. It has to cover the worst
realistic single call: a multi-file session (ScenesCraig's six Audacity speakers)
transcribed with `large-v3` and diarization, plus a first-run model download into
the `whisperx-models` volume. It is a cap on one container invocation, not on a
session or a run — orchestrate calls the containers many times.

The same number appears deliberately in one other place:

| where | value | what it bounds |
|---|---|---|
| `docker.DOCKER_RUN_TIMEOUT` | 7200 | one `docker compose run`, per attempt |
| `run_tests.py --tool-timeout` | imports it | one stage subprocess, which wraps the above |
| `run_orchestrate_tests.py --timeout` | 600 | one whole session through `orchestrate.py` |

The first two match by construction rather than by promise: `run_tests.py`
imports `DOCKER_RUN_TIMEOUT` instead of restating the number, so the driver
cannot end up killing a container the Docker layer is still waiting on. 
The third is a configuraton for the golden tests.  As it stands, none of the video
or audacity input for a single test set will exceed this on a reasonable GPU. The
test loops over the test sets, so each run of Docker is independent of the prior set.

Not part of this: `verify_setup.py` (30s for `docker info`, 60s for
`docker compose config`) and the `ffprobe` duration probe in
`pipeline/common/sessions.py` (60s). Those are quick metadata calls, and a slow
one means something is wrong rather than something is large.

**Retries are separate from the timeout.** A failed run is retried up to
`retries` times (default 2) with a linear backoff of `backoff × attempt` seconds
(default 3), and **only** when stderr matches `_TRANSIENT_DOCKER_MARKERS` — the
daemon 500ing on a project-network lookup during rapid back-to-back runs, and
similar. A real failure (bad input, a transcription error) is reported
immediately. Each attempt gets its own full 7200s, so the worst case is three
attempts, not a shared budget.

### What gets mounted

Per run, the stages pass the session in read-only and the output dir writable:

```
-v <session>:/session_input:ro      -e AUDIO_DIR=/session_input   (or VIDEO_DIR)
-v <output>:/session_output         -e OUTPUT_DIR=/session_output (or SCENES_DIR)
```

#### Why the container path is not called `cc_output`

Host-side, everything generated lands under **`cc_output/`**. Container-side, the
writable mount is always **`/session_output`**. These are not two names for one
thing, and the difference is deliberate:

```
cc_output/transcriptions  ->  /session_output     (transcribe stage)
cc_output/scenes_output   ->  /session_output     (scenes stage)
```

`/session_output` maps to a **subdirectory** of `cc_output`, never to `cc_output`
itself, and each stage points it at a different one. The container's whole
contract is "read `/session_input`, write `/session_output`" — it does not know
or care which subdirectory it was handed, which is what lets one image serve
every stage.

Calling the mount `/cc_output` would also be a lie under `output.base_dir`: with
an output base configured the host path is
`<base>/<session path>/cc_output/transcriptions`, nowhere near the session. The
generic name is the one thing that stays true across three stages and both
output modes.

Practical consequence: renaming the host folder is a config-and-docs change,
not an image rebuild. The two names are decoupled on purpose.

The source is mounted read-only so a container can never modify or delete the
originating material. `session_mounts()` refuses any output directory that would
resolve outside the output base, which is what protects source media from
`clear_mount`.

The compose files additionally mount, statically:

| host | container | why |
|---|---|---|
| `docker/<image>/` | `/usr/local/bin/scripts` | entrypoint + helper shell scripts |
| `src/` | `/usr/local/bin/pylib` | the importable `pipeline` package (`PYTHONPATH`) |
| `config/config.yaml` | `/usr/local/bin/config.yaml` | read-only |

The library and script mounts are separate on purpose: `detect_scenes_multi.sh`
runs `python3 -m pipeline.scenes.extract_segments`, so `PYTHONPATH` must point at
`src/`, not at the shell-script directory.

Because the working tree is bind-mounted rather than copied, the bytes on the
host are the bytes Linux executes — which is why `.gitattributes` pins everything
to LF. A CRLF `entrypoint.sh` fails with `/bin/bash^M: bad interpreter`.

## Configuration

`config/config.yaml` is the single source of truth, loaded by
`pipeline.config`. Resolution is `WHISPERX_CONFIG` (env), else
`<repo>/config/config.yaml` — no upward search, no cwd fallback, so a missing file
fails immediately naming the path it wanted. The containers set `WHISPERX_CONFIG`
to `/usr/local/bin/config.yaml`, where the file is mounted read-only.

A source tree may carry **its own** `config.yaml` at the source root; the stages
walk up from `--session-dir` to find it (`find_config_upward`). That per-source
config wins, and the repo's acts as the fallback when a source has none. Sections:

- `whisper:` — model, language, compute, batch, output format, diarize, `clean:` thresholds.
- `scenes:` — threshold, min length, num images, format, `roi`, `roi_file`, and
  `manual_source`/`manual_csv_name` (a folder of user-captured scenes to use
  instead of running PySceneDetect, when present in a session).
- `orchestration:` — `source_dir`, `session_dirs`, `parallel_workers`.
- `merge:` — `auto_merge`, CSV column names, image width, document title/fonts.
- `speakers:` — `config_file` (path to `speaker_config.json`).

Environment overrides exist for the common knobs (see the tables above and the
list at the bottom of `config.yaml`).

### Files that live WITH the recordings

Three files are user data, not repo configuration. They sit in the **source
directory** beside the recordings — not in `config/` — because each campaign has
its own cast and vocabulary. All three are optional and resolved relative to
`orchestration.source_dir`.

| file | config key | used for |
|---|---|---|
| `speaker_config.json` | `speakers.config_file` | speaker names + filler phrases |
| `whisperx_hotwords.txt` | `whisper.hotwords_file` | `--hotwords` |
| `whisperx_initial_prompt.txt` | `whisper.initial_prompt_file` | `--initial_prompt` |

Starting points for the last two are in `config/`:
`whisperx_hotwords_sample.txt` and `whisperx_initial_prompt_sample.txt`. Copy them
into your source directory and drop the `_sample` suffix.

`python scripts/verify_setup.py --source-dir <your source>` reports which
of the three it found, and for the hint files, how many terms each contributed.

#### What goes in which

The two hint files feed **different** WhisperX features, and the distinction
decides where a term belongs:

**`whisperx_hotwords.txt` → `--hotwords` → applies to the WHOLE run.**
One term per line. This is for words the model gets *wrong*: proper nouns and
jargon that have a plausible common-word competitor.

```
Mysteria          # else "Mysterio", "mystery", "mi steria"
Aberrancy
Mechanon
Omega Force
```

**`whisperx_initial_prompt.txt` → `--initial_prompt` → seeds the FIRST WINDOW.**
Prose, not a list — WhisperX continues *from* this text, so it works by example.
This is for conventions: capitalisation, abbreviation style, punctuation.

```
Omega Force campaign. Characters and terms: Max Miracle, Van Allen, Bruno,
Kincaid, Mechanon, Neutron. Use DEX, OCV, DCV, SPD, STUN rather than
"decks", "occ vee", "speed", "stunned".
```

**The rule of thumb:** if a term must be recognised correctly at the three-hour
mark, it belongs in **hotwords**. The prompt primes only the opening window, so
its influence decays across a long session. Anything that is purely a spelling or
formatting convention is fine in the prompt; anything load-bearing goes in
hotwords. Listing a term in both is harmless.

**Format for both files:** lines whose first non-space character is `#` are
comments and are stripped before the value reaches WhisperX, so the files can be
organised with headers. Blank lines are ignored. The remainder is collapsed to a
single line and double quotes are removed, so the value survives shell quoting.
A missing, empty, or comment-only file means that flag is simply not passed.

**Build them from your own corrections.** The most effective source of terms is a
diff of a hand-corrected storyboard against the raw WhisperX JSON — every fix you
made by hand is a word the model needs help with. That is how the shipped sample
was built.

---

## Selecting which sessions to process

Orchestrate processes an **explicit list** of session folders. Name the folders directly on 
the CLI, or set them in `config.yaml`. If neither is given, orchestrate **fails loudly** rather 
than processing nothing.

```bash
python scripts/orchestrate.py --session-dirs MultiVideo SingleVideo MultiROI
python scripts/orchestrate.py --session-dirs /abs/path/Session1 MultiVideo   # abs + relative mix
```

Or set them in `config.yaml` so no CLI flag is needed (the `--session-dirs`
default comes from here):

```yaml
orchestration:
  source_dir: D:/TestWhisper
  session_dirs:            # single string or a YAML list
    - MultiVideo
    - SingleVideo
    - MultiROI
```

Rules: relative paths resolve against `source_dir`; absolute paths are used
as-is; non-existent directories are skipped with a warning; the list is sorted
for stable ordering.

To process a whole tree, iterate it in a wrapper and call orchestrate once per
folder — see README "Automating the processing of a whole hierarchy of
recordings".

---

## Troubleshooting

- **`No session directories to process`** — pass `--session-dirs`, or set `orchestration.session_dirs` in `config.yaml`.
- **`could not select device driver "" with capabilities [[gpu]]`** — install/enable the nvidia-container-toolkit, restart Docker.
- **CUDA out of memory** — lower `WHISPER_BATCH_SIZE` (e.g. 8), use `int8_float16`, or a smaller model; run `--parallel 1`.
- **Diarization 401** — `HF_TOKEN` missing read scope or pyannote licenses not accepted.
- **Too few / too many scenes** — tune `SCENE_THRESHOLD` (lower = more).
- **Clear the model cache** — `docker compose down -v`.
- **A `torchcodec`/FFmpeg shared-lib traceback at startup** — cosmetic; WhisperX decodes via the `ffmpeg` binary, not torchcodec.
- **A file looks "truncated" when read from a Linux/WSL mount but is fine from the Windows side** — that's mount sync lag, not real file corruption; re-check from Windows before concluding a file is broken.
- **timeout - adjust the maximum timeout on the Docker layer in orchestrate.py
