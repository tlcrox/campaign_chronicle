#!/usr/bin/env python3
"""Logging is configured by entry points, not by importing one.

Every stage is both a CLI and a function orchestrate imports. While each of them
called ``logging.basicConfig`` at module level, importing one reconfigured root
logging for the whole process — including under pytest, where a test that
imports orchestrate handed the entire session a stderr handler at INFO.
"""

import logging
import re
import subprocess
import sys
import unittest
from pathlib import Path

from pipeline.common.logs import LOG_FORMAT, setup_logging

REPO_ROOT = Path(__file__).resolve().parents[4]

# Everything that is either a stage (imported by orchestrate) or an entry point.
MODULES = sorted(
    [p for p in (REPO_ROOT / "scripts" / "cc_stages").glob("*.py")
     if p.name != "__init__.py"]
    + [REPO_ROOT / "scripts" / "orchestrate.py"]
    + list((REPO_ROOT / "tests" / "integration").glob("run_*.py"))
)


class NoModuleLevelBasicConfig(unittest.TestCase):
    """Static: the call may not sit at column 0, where import runs it."""

    def test_no_stage_configures_logging_at_import(self):
        for path in MODULES:
            with self.subTest(module=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertNotRegex(
                    source, r"(?m)^logging\.basicConfig",
                    f"{path.name} configures root logging on import")

    def test_every_entry_point_configures_it_in_main(self):
        for path in MODULES:
            with self.subTest(module=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertRegex(source, r"(?m)^\s+setup_logging\(",
                                 f"{path.name} never configures logging")


class ImportingLeavesRootAlone(unittest.TestCase):
    """Behavioural, in a clean interpreter — this one cannot be faked.

    A subprocess because by the time this test runs, pytest and the rest of the
    suite have already touched root logging.
    """

    def test_a_fresh_interpreter_gains_no_handlers(self):
        probe = (
            "import logging;"
            "import cc_stages.transcribe, cc_stages.merge_scenes, orchestrate;"
            "root = logging.getLogger();"
            "print(len(root.handlers), root.level)"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=str(REPO_ROOT),
            capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        handlers, level = result.stdout.split()
        self.assertEqual(handlers, "0", "importing a stage added a root handler")
        self.assertEqual(int(level), logging.WARNING,
                         "importing a stage changed the root level")


class SetupLogging(unittest.TestCase):
    """The entry point gets the last word, whoever called basicConfig first."""

    def setUp(self):
        root = logging.getLogger()
        self._handlers = list(root.handlers)
        self._level = root.level

    def tearDown(self):
        root = logging.getLogger()
        root.handlers[:] = self._handlers
        root.setLevel(self._level)

    def test_default_is_info(self):
        setup_logging()
        self.assertEqual(logging.getLogger().level, logging.INFO)

    def test_verbose_is_debug(self):
        setup_logging(verbose=True)
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_it_overrides_an_earlier_basic_config(self):
        """Without force=True this is the silent no-op we just removed.

        whisperx and its dependencies call basicConfig on import, so an entry
        point that runs second would otherwise never get its own settings.
        """
        logging.basicConfig(level=logging.ERROR, format="%(message)s")
        setup_logging(verbose=True)
        root = logging.getLogger()
        self.assertEqual(root.level, logging.DEBUG)
        self.assertEqual(root.handlers[0].formatter._fmt, LOG_FORMAT)


if __name__ == "__main__":
    unittest.main()
