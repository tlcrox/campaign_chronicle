# Convenience wrappers around the documented commands. Nothing here is
# required — every target is a one-line command you can run directly, and
# ORCHESTRATION.md is the authority. `make` is not native to Windows; use Git
# Bash, WSL, or skip make entirely.
#
# ACTIVATE THE VENV FIRST (see ORCHESTRATION.md "Prerequisites"):
#     .venv\Scripts\Activate.ps1        PowerShell
#     source .venv/bin/activate         bash
# Or run without activating by overriding PYTHON:
#     make test PYTHON="uv run python"

PYTHON ?= python

COMPOSE_WHISPERX   := compose/whisperx/docker-compose.yml
COMPOSE_SCENES     := compose/pyscenedetect/docker-compose.yml

.PHONY: help verify build build-whisperx build-scenes test test-unit \
        orchestrate orchestrate-parallel orchestrate-dry clean-test

help:
	@echo "campaign_chronicle - common commands"
	@echo ""
	@echo "Usage: make [target]          (activate .venv first)"
	@echo ""
	@echo "Setup:"
	@echo "  verify                   Pre-flight check (env, Docker, config, hint files)"
	@echo "  build                    Build both container images"
	@echo "  build-whisperx           Build just the transcription image (GPU)"
	@echo "  build-scenes             Build just the scene-detection image (CPU)"
	@echo ""
	@echo "Run:"
	@echo "  orchestrate              Process the configured sessions"
	@echo "  orchestrate-parallel     Same, 2 concurrent sessions (see hardware note)"
	@echo "  orchestrate-dry          Show what would be processed, run nothing"
	@echo ""
	@echo "Test:"
	@echo "  test                     Full unit + integration suite"
	@echo "  clean-test               Delete generated test output"
	@echo ""
	@echo "Examples:"
	@echo "  make orchestrate-dry"
	@echo "  make orchestrate ARGS='--session-dirs \"Week 14\"'"
	@echo "  make test PYTHON='uv run python'"

# ---------------------------------------------------------------- setup

verify:
	$(PYTHON) scripts/verify_setup.py

build: build-whisperx build-scenes

build-whisperx:
	docker compose -f $(COMPOSE_WHISPERX) build whisperx

build-scenes:
	docker compose -f $(COMPOSE_SCENES) build scenes

# ---------------------------------------------------------------- run
# ARGS passes anything through, e.g. ARGS='--session-dirs "Week 14" --no-scenes'.
# Real flags: --source-dir --session-dirs --parallel --no-scenes --roi-file
#             --no-transcription --merge-only --config --dry-run --verbose

orchestrate:
	$(PYTHON) scripts/orchestrate.py --verbose $(ARGS)

# Each concurrent session takes a full GPU/CPU slice; the pipeline deliberately
# does not cap concurrency, so oversubscribing will OOM a single GPU.
orchestrate-parallel:
	$(PYTHON) scripts/orchestrate.py --parallel 2 --verbose $(ARGS)

orchestrate-dry:
	$(PYTHON) scripts/orchestrate.py --dry-run --verbose $(ARGS)

# ---------------------------------------------------------------- test

test:
	$(PYTHON) -m pytest

test-unit:
	$(PYTHON) -m pytest tests/unit

# Only the generated test trees. This never touches a real source directory —
# use tests/test_source/reset_test_source.py for that, deliberately, by hand.
clean-test:
	@echo "Removing generated test output..."
	@rm -rf tests/integration/output
	@find tests/test_source -type d -name cc_output -exec rm -rf {} + 2>/dev/null || true
	@echo "Done. tests/expected (goldens) and tests/test_source (inputs) untouched."

.DEFAULT_GOAL := help

