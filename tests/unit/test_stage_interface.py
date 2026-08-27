#!/usr/bin/env python3
"""Every pipeline stage is reached the same way: ``<module>.run(session_dir, …)``.

This replaced three conventions that had grown up around one idea. A stage was
named bare when its function name already matched its module, ``_tool`` when the
plain name would have collided with the pipeline function it wraps, and
``_stage`` when the work lived under ``pipeline/`` with a thin CLI wrapper in
``cc_stages``. Three situations, three spellings, and orchestrate's import block
showed all of them at once.

The module now carries the identity and ``run`` is the verb, so which layer
implements a stage is an implementation detail of its own module. These tests
exist because a convention with nothing checking it drifts back into three.
"""

import importlib
import inspect
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The stages orchestrate composes, in pipeline order. cc_stages also holds
# apply_speaker_mapping, which is a standalone utility for one transcript file
# rather than a session stage, so it is deliberately not here.
STAGE_MODULES = (
    "transcribe",
    "detect_scenes_single_roi",
    "detect_scenes_multi_roi",
    "map_speakers",
    "clean_transcription",
    "merge_transcripts",
    "merge_scenes",
    "generate_storyboard",
)


class EveryStageExposesRun(unittest.TestCase):
    def _run_callable(self, name):
        module = importlib.import_module(f"cc_stages.{name}")
        self.assertTrue(hasattr(module, "run"),
                        f"cc_stages.{name} has no run(); orchestrate calls it")
        return module.run

    def test_run_exists_and_is_callable(self):
        for name in STAGE_MODULES:
            with self.subTest(stage=name):
                self.assertTrue(callable(self._run_callable(name)))

    def test_session_dir_is_the_first_parameter(self):
        for name in STAGE_MODULES:
            with self.subTest(stage=name):
                params = list(inspect.signature(self._run_callable(name)).parameters)
                self.assertEqual(params[0], "session_dir")

    def test_config_and_dry_run_are_accepted_by_keyword(self):
        """orchestrate passes both to every stage; a stage that took neither
        would silently run against the process-wide config."""
        for name in STAGE_MODULES:
            with self.subTest(stage=name):
                params = inspect.signature(self._run_callable(name)).parameters
                for expected in ("config", "dry_run"):
                    self.assertIn(expected, params)
                    self.assertIn(params[expected].kind,
                                  (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY))

    def test_dry_run_defaults_to_false(self):
        """A stage that defaulted to dry_run=True would silently do nothing."""
        for name in STAGE_MODULES:
            with self.subTest(stage=name):
                params = inspect.signature(self._run_callable(name)).parameters
                self.assertIs(params["dry_run"].default, False)


class NoStageKeepsASuffix(unittest.TestCase):
    """The suffixes are what the single entry point replaced."""

    def test_no_tool_or_stage_functions_remain(self):
        folders = (REPO_ROOT / "scripts" / "cc_stages",
                   REPO_ROOT / "src" / "pipeline")
        offenders = []
        for folder in folders:
            for path in folder.rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped.startswith("def "):
                        continue
                    name = stripped[4:].split("(")[0]
                    # apply_speaker_mapping_tool is a standalone utility, not a
                    # stage; resolve_tool_config is unrelated naming.
                    if name in ("apply_speaker_mapping_tool", "resolve_tool_config"):
                        continue
                    if name.endswith("_tool") or name.endswith("_stage"):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}: {name}")
        self.assertEqual(offenders, [], "stages are reached through run()")


class OrchestrateImportsModules(unittest.TestCase):
    """Importing the module, not the name, is what removes the collisions:
    `run` is never a pipeline-layer name, so it cannot clash with one."""

    def test_no_stage_is_imported_by_name(self):
        source = (REPO_ROOT / "scripts" / "orchestrate.py").read_text(encoding="utf-8")
        self.assertIn("from cc_stages import (", source)
        for name in STAGE_MODULES:
            with self.subTest(stage=name):
                self.assertNotIn(f"from cc_stages.{name} import", source)


if __name__ == "__main__":
    unittest.main()
