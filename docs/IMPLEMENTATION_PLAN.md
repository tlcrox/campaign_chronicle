# Implementation Plan — outstanding work

What is left to do. Completed work lives in `HISTORY.md`; how things currently
work is in `README.md` (design) and `ORCHESTRATION.md` (running it).

Last updated: 2026-08-23.

---

## Near-term

### Files naming paths that do not exist

- [ ] **`TESTING.md:49`** says `run_orchestrate_tests.py` runs
  `tests/orchestrate.py`. It runs `scripts/orchestrate.py`.
- [ ] **`TESTING.md:77, :80`** spell the repo **campaign_chronical**.


---

## Correctness




### Issue #2 — per-video ROI not implemented (MultiCraig)

Orchestrate must apply each video's ROI when a session mixes videos with
different crops.


### Relative paths in config resolve against `config/`

`Config._config_dir` is the base for relative paths declared *inside* the config
(notably `orchestration.source_dir`). Since `config.yaml` lives in `config/`,
that base sits one level below the repo root: a relative `source_dir:
../TestWhisper` reads as repo-root-relative but resolves config-dir-relative.
Currently masked because the live config uses an absolute path. **Decide**
whether `_config_dir` stays "the config file's directory" or becomes "the repo
root".

---

## Testing

### Golden comparison

`output_compare.compare_output_tree` is called by both drivers
(`run_tests.py:445`, `run_orchestrate_tests.py:118`) and compares two `cc_output`
trees — JSON parsed, CSV newline-normalized, `.docx` by extracted text,
everything else by bytes.

- [ ] **Scene JPGs are still compared byte-for-byte.** `output_compare.py:102`
  falls through to `_bytes_equal` for any extension without a registered
  comparator, `.jpg` included, so an encoder or library change produces false
  failures. The ~28 KB of CSV/JSON carries the signal; images are better
  asserted by count and filename.

### Coverage gaps

- [ ] `src/pipeline/scenes/extract_segments.py` — no direct test, yet it runs
  in-container via `detect_scenes_multi.sh`.
- [ ] Single-ROI source variation end-to-end (CLI override, explicit full-frame).
  `resolve_single_roi` is unit-tested; the tool path is not.
- [ ] Assert VIDEO_INDEX image naming in the multi-video runner (ties to Issue #1).
- [ ] Exercise the config-driven `session_dirs` default — the runners pass
  explicit `--session-dirs`, bypassing it.
- [ ] Filler-phrase removal.
- [ ] **`--verbose` on the stage CLIs.** Only `orchestrate.py` declares it
  (checked across all nine stages and both integration drivers), yet running a
  stage directly is the debugging path — the place it would earn its keep most.
  `setup_logging(verbose=)` already takes the argument, so each stage needs a
  flag and one call-site change. Not a defect; orchestrate's own works.

---

## Deferred

### Speaker mapping

- [ ] **Finish the speaker-config audit.** Reconcile `config.yaml`
  (`speakers.config_file: speaker_config.json`) with `config_sample.yaml`
  (`../speakers/config.json`), and review the `speakers.*` knobs and
  `pipeline/transcribe/map_speakers.py` for remaining hardcoded names or path
  assumptions.
- [ ] Embedding-based speaker matching (currently manual config, by choice).
- [ ] Dynamic speaker profile generation from Audacity project metadata.
- [ ] Confidence-based fallback for unmapped speakers.

**Voice-based auto-detection.** The longer-term goal is replacing the fixed
`filename_mapping` with voice identification. Nothing usable exists in the tree —
treat this as a fresh design. It would need config-driven `build`/`match` runners
and a speaker-ID step wired into orchestrate, gated by `config.yaml: speakers:`.
*(Optional: a librosa-MFCC fallback in the embedding extractor for when a
SpeechBrain call fails.)*


### Infrastructure

- [ ] **Lock container deps.** The host is locked (`uv.lock`); the images are not.
  `docker compose -f compose/whisperx/docker-compose.yml run --rm --entrypoint pip
  whisperx freeze > requirements-docker.lock.txt`, then install from the lock.
- [ ] **Consider uv inside the containers.** A real speedup on this dependency
  set, but not a drop-in: the two-stage CUDA torch install needs uv's index
  pinning, and uv's resolver is stricter than pip's on the loosely-pinned
  whisperx/speechbrain/pyannote stack. Check current uv docs for the PyTorch
  index pattern rather than trusting a remembered recipe.


---

## Parked: Champions-specific storyboard styling & section detection

Source-specific refinements for the **Champions/GGG track only** — NOT the
generic pipeline. Everything here must live in that source's own config so other
sources (Harn, etc.) inherit none of it. Both items came from diffing tool output
(`Week 79_storyboard.docx`) against the hand-finished `Destroyers.docx`.

### A. Storyboard image styling

Today `generate_storyboard` emits each scene image as a 6" centered **inline**
picture, one large block per scene. The Champions final documents convert most to
small (~1.0–1.5"), right-anchored, tight-wrapped **floating** images so text flows
beside them and vertical space collapses — Destroyers has 33 inline + 47 floating
against the tool's 84 inline at 6".

Make image insertion configurable: width, wrap (inline vs tight), alignment
(center vs right-float). Needs raw-XML `<wp:anchor>` + `wrapTight`, as python-docx
has no floating-image API. Config-gated and off by default, so the Harn track
keeps the plain inline layout.

### B. Scene → Turn/Segment section headers

Today: 84 per-image `Scene NN-MMM` headers. Champions final: ~39 narrative
headers (`Recap`, `Heating up`, `Turn 0 – Facing Level 10`, `Segment 12`…),
collapsing images under sections.

The Turn/Segment structure is **not** in the transcript — whisperx finds zero
"segment N" / "turn N" mentions. It lives in the HERO System SPD chart plus
spoken cues. Approach, all source-config-driven:

- **Config (source dir):** per-character SPD chart (`Max: 6`, `Bazyn: 5`, villains
  4–6), cue phrases, roster, on/off switch.
- **SPD chart = deterministic backbone.** Given combat start and turn boundaries
  (post-12 "recoveries"), each character's acting segment is fixed by the rules
  (SPD 6 → 2,4,6,8,10,12; SPD 5 → 3,5,8,10,12). Detection needs only the turn
  boundaries and the order characters are prompted; segment numbers are then
  computed, not guessed.
- **Cues are too sparse and noisy for pure regex.** Week 79 (3h21m): "you're up"
  ×6, "what do you want to do" ×5, "go ahead / you're next" ×6, bare-name prompts
  ×32 but polluted with narration; "recoveries" marks turn ends. Intent
  classification is the hard part.
- **Proposed:** regex pulls candidate lines, then a local LLM (Ollama, on the
  host — not reachable from a sandbox) classifies each as
  `combat_start | turn_boundary | prompt:<character> | none`.
- **Human-in-the-loop:** emit a reviewable `(timestamp -> label)` section map the
  user corrects; `generate_storyboard` then places headers and groups images.
- Build as a **separate source-specific tool** (e.g. `detect_sections.py`) that
  outputs the map; the generic pipeline stays untouched.

Open: which Ollama model to target, and validating classification precision on a
Week-79 slice before building the full assembly.
