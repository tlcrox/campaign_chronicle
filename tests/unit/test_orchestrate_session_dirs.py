#!/usr/bin/env python3
"""Tests for orchestrate.resolve_session_dirs — the single resolution point.

--session-dirs defaults to Config.session_dirs_config (raw, unresolved) and is
resolved here, once, against the *effective* source dir.

That ordering is the whole point: resolving or existence-filtering config
session_dirs while argparse is still being constructed happens before
--source-dir is parsed, which silently drops every session that exists only
under the override and reports "No session directories to process".
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from orchestrate import resolve_session_dirs

REPO_ROOT = Path(__file__).resolve().parents[2]


class ImportingOrchestrateHasNoSideEffects(unittest.TestCase):
    """This module imports orchestrate for one pure function. That used to cost
    an argv scan and a config load, both at module scope.

    Subprocesses, because the damage is done at import: by the time a test body
    runs, this module has already been imported once.
    """

    def _import_with(self, argv_extra=(), env_extra=None):
        import os
        probe = (
            "import sys;"
            f"sys.argv = ['pytest', 'tests/unit'] + {list(argv_extra)!r};"
            "import orchestrate;"
            "print('ok')"
        )
        env = dict(os.environ)
        env.pop("WHISPERX_CONFIG", None)
        env.update(env_extra or {})
        return subprocess.run([sys.executable, "-c", probe], cwd=str(REPO_ROOT),
                              capture_output=True, text=True, encoding="utf-8",
                              timeout=120, env=env)

    def test_import_does_not_read_the_command_line(self):
        """--config naming a missing file used to sys.exit() during collection."""
        result = self._import_with(["--config", "does-not-exist.yaml"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_import_does_not_load_or_validate_a_config(self):
        """An unusable config must not stop tests that never read one."""
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "config.yaml"
            broken.write_text("whisper:\n  device: cpu\n  compute_type: float16\n",
                              encoding="utf-8")
            result = self._import_with(env_extra={"WHISPERX_CONFIG": str(broken)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_the_module_holds_no_config_global(self):
        import orchestrate
        self.assertFalse(hasattr(orchestrate, "cfg"),
                         "orchestrate keeps a module-level config again")


class ResolveSessionDirs(unittest.TestCase):
    def test_relative_entries_join_the_given_source_dir(self):
        out = resolve_session_dirs([Path("A"), Path("B")], Path("/src"))
        self.assertEqual(out, [Path("/src/A"), Path("/src/B")])

    def test_absolute_entries_pass_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            absolute = Path(tmp) / "Session9"
            out = resolve_session_dirs([absolute], Path(tmp) / "other")
            self.assertEqual(out, [absolute])

    def test_nested_relative_entries_keep_their_parent(self):
        """"Weeks/Week 13" must not collapse to "Week 13"."""
        out = resolve_session_dirs([Path("Weeks/Week 13")], Path("/src"))
        self.assertEqual(out, [Path("/src/Weeks/Week 13")])

    def test_empty_and_none_are_tolerated(self):
        self.assertEqual(resolve_session_dirs([], Path("/src")), [])
        self.assertEqual(resolve_session_dirs(None, Path("/src")), [])

    def test_source_dir_override_is_honoured(self):
        """The regression: the same configured names resolve under whichever
        source dir is passed, so --source-dir can redirect them."""
        names = [Path("MultiROI"), Path("MultiVideo")]
        from_config = resolve_session_dirs(names, Path("/configured"))
        from_cli = resolve_session_dirs(names, Path("/overridden"))
        self.assertEqual(from_config, [Path("/configured/MultiROI"),
                                       Path("/configured/MultiVideo")])
        self.assertEqual(from_cli, [Path("/overridden/MultiROI"),
                                    Path("/overridden/MultiVideo")])

    def test_resolution_does_not_touch_the_filesystem(self):
        """Non-existent paths are returned, not filtered — that's find_sessions' job."""
        out = resolve_session_dirs([Path("Ghost")], Path("/definitely/not/here"))
        self.assertEqual(out, [Path("/definitely/not/here/Ghost")])
        self.assertFalse(out[0].exists())


if __name__ == "__main__":
    unittest.main()
