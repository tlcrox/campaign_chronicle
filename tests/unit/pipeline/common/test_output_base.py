#!/usr/bin/env python3
"""Configurable output location.

Default: output lands beside the input at <session>/cc_output/... — what a user
processing their own recordings wants. With an output base configured, the whole
tree moves and the source is never written to.

This is what lets run_tests.py and run_orchestrate_tests.py coexist:
run_tests.py sets CC_OUTPUT_BASE to tests/test_output, run_orchestrate_tests.py
leaves it unset. Two independent result trees, so neither suite's leftovers can
make the other appear to pass, and no cleanup is needed between them.
"""

import os
import tempfile
import unittest
from pathlib import Path

from pipeline.common.mounts import (
    OUTPUT_ROOT,
    SESSION_OUTPUT_SUBDIR,
    UnsafeOutputDir,
    output_dir_for,
    session_mounts,
    session_output_base,
)


class _Cfg:
    """Minimal config stub, as the stages' callers use."""

    def __init__(self, base=None, source=None):
        self.output_base_dir = base
        self._source = source

    def session_key(self, session_dir):
        return Path(session_dir).resolve().relative_to(Path(self._source).resolve())


class DefaultIsBesideTheSource(unittest.TestCase):
    def test_no_base_writes_under_the_session(self):
        with tempfile.TemporaryDirectory() as t:
            s = Path(t) / "Week 13"
            s.mkdir()
            out = output_dir_for(s, SESSION_OUTPUT_SUBDIR, None)
            self.assertEqual(out, s / SESSION_OUTPUT_SUBDIR)
            self.assertIn(OUTPUT_ROOT, out.parts)

    def test_config_without_the_attribute_still_works(self):
        """Duck-typed stubs predating output_base_dir must not explode."""
        class Ancient:
            pass
        with tempfile.TemporaryDirectory() as t:
            s = Path(t) / "S"
            s.mkdir()
            self.assertEqual(output_dir_for(s, SESSION_OUTPUT_SUBDIR, Ancient()),
                             s / SESSION_OUTPUT_SUBDIR)


class ConfiguredBaseRelocatesEverything(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.src = Path(self.t.name) / "src"
        self.sess = self.src / "Weeks" / "Week 13"
        self.sess.mkdir(parents=True)
        self.base = Path(self.t.name) / "out"

    def test_output_moves_and_keeps_the_same_shape(self):
        cfg = _Cfg(self.base, self.src)
        out = output_dir_for(self.sess, SESSION_OUTPUT_SUBDIR, cfg)
        self.assertEqual(out, self.base / "Weeks" / "Week 13" / SESSION_OUTPUT_SUBDIR)

    def test_shape_is_identical_to_the_default_layout(self):
        """One directory comparison must serve both modes."""
        beside = output_dir_for(self.sess, SESSION_OUTPUT_SUBDIR, None)
        moved = output_dir_for(self.sess, SESSION_OUTPUT_SUBDIR, _Cfg(self.base, self.src))
        self.assertEqual(beside.relative_to(self.sess), moved.relative_to(
            self.base / "Weeks" / "Week 13"))

    def test_source_tree_is_never_written_to(self):
        cfg = _Cfg(self.base, self.src)
        _, _, out = session_mounts(self.sess, SESSION_OUTPUT_SUBDIR, "A", "B",
                                   output_base=cfg.output_base_dir,
                                   session_key=cfg.session_key(self.sess))
        self.assertFalse(str(out).startswith(str(self.src)))

    def test_session_key_is_relative_path_not_leaf_name(self):
        """Two sources each with a 'Week 13' must not collide."""
        a = session_output_base(self.sess, self.base, Path("SourceA/Week 13"))
        b = session_output_base(self.sess, self.base, Path("SourceB/Week 13"))
        self.assertNotEqual(a, b)

    def test_absolute_session_key_is_rejected(self):
        with self.assertRaises(UnsafeOutputDir):
            session_output_base(self.sess, self.base, Path("/etc"))


class GuardIsAnchoredOnTheOutputBase(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.sess = Path(self.t.name) / "S"
        self.sess.mkdir()
        self.base = Path(self.t.name) / "out"

    def test_escapes_rejected_with_a_base(self):
        for bad in ("..", "../evil"):
            with self.assertRaises(UnsafeOutputDir):
                session_mounts(self.sess, bad, "A", "B",
                               output_base=self.base, session_key=Path("S"))

    def test_escapes_rejected_without_a_base(self):
        for bad in ("..", "../evil"):
            with self.assertRaises(UnsafeOutputDir):
                session_mounts(self.sess, bad, "A", "B")


class EnvOverride(unittest.TestCase):
    """CC_OUTPUT_BASE is how a driver redirects a run without editing config."""

    def test_env_beats_config_and_resolves_against_cwd(self):
        from pipeline.config import Config
        with tempfile.TemporaryDirectory() as t:
            cfgdir = Path(t) / "config"
            cfgdir.mkdir()
            (cfgdir / "config.yaml").write_text(
                "whisper: {}\nscenes: {}\noutput:\n  base_dir: from_config\n")
            c = Config(config_path=cfgdir / "config.yaml")
            self.assertEqual(c.output_base_dir.name, "from_config")

            os.environ["CC_OUTPUT_BASE"] = str(Path(t) / "from_env")
            self.addCleanup(os.environ.pop, "CC_OUTPUT_BASE", None)
            c2 = Config(config_path=cfgdir / "config.yaml")
            self.assertEqual(c2.output_base_dir, (Path(t) / "from_env").resolve())


if __name__ == "__main__":
    unittest.main()
