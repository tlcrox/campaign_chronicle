#!/usr/bin/env python3
"""
Tests for pipeline.common.mounts.session_mounts — the per-session bind-mount +
safety-guard helper shared by the transcribe_* and detect_scenes_* tools.

Deterministic, no Docker. Verifies the source mount is READ-ONLY, output lands in
the session's own subdir, the env points the container's dir vars at the mounts,
and the guard refuses any output dir that would escape the session (protecting
the originating source material).

Run from scripts/:
    python3 -m unittest pipeline.common.test_mounts -v
"""

import tempfile
import unittest
from pathlib import Path


from pipeline.common.mounts import session_mounts, UnsafeOutputDir  # noqa: E402


class SessionMounts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Space in the name to catch quoting/handling bugs.
        self.session = Path(self.tmp.name) / "Sess A"
        self.session.mkdir()
        self.sess_abs = self.session.resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_happy_path_scenes(self):
        volumes, env, out = session_mounts(
            self.session, "scenes_output", "VIDEO_DIR", "SCENES_DIR"
        )
        self.assertEqual(out, self.sess_abs / "scenes_output")
        self.assertEqual(volumes, [
            f"{self.sess_abs}:/session_input:ro",
            f"{out}:/session_output",
        ])
        self.assertEqual(env, {"VIDEO_DIR": "/session_input", "SCENES_DIR": "/session_output"})

    def test_source_mount_is_read_only(self):
        volumes, _, _ = session_mounts(self.session, "transcriptions", "AUDIO_DIR", "OUTPUT_DIR")
        # First entry = source: MUST be read-only. Second = output: MUST be writable.
        self.assertTrue(volumes[0].endswith(":/session_input:ro"), volumes[0])
        self.assertFalse(volumes[1].endswith(":ro"), volumes[1])

    def test_env_keys_are_passed_through(self):
        _, env, _ = session_mounts(self.session, "transcriptions", "AUDIO_DIR", "OUTPUT_DIR")
        self.assertEqual(env, {"AUDIO_DIR": "/session_input", "OUTPUT_DIR": "/session_output"})

    def test_custom_mount_paths(self):
        volumes, env, _ = session_mounts(
            self.session, "o", "I", "O", in_mount="/in", out_mount="/out"
        )
        self.assertTrue(volumes[0].endswith(":/in:ro"))
        self.assertTrue(volumes[1].endswith(":/out"))
        self.assertEqual(env, {"I": "/in", "O": "/out"})

    def test_nested_subdir_allowed(self):
        _, _, out = session_mounts(self.session, "a/b", "I", "O")
        self.assertEqual(out, self.sess_abs / "a" / "b")

    # --- the safety guard: never let output escape the session -------------
    def test_rejects_session_itself(self):
        with self.assertRaises(UnsafeOutputDir):
            session_mounts(self.session, ".", "I", "O")

    def test_rejects_parent(self):
        with self.assertRaises(UnsafeOutputDir):
            session_mounts(self.session, "..", "I", "O")

    def test_rejects_sibling_escape(self):
        with self.assertRaises(UnsafeOutputDir):
            session_mounts(self.session, "../evil", "I", "O")

    def test_rejects_absolute_path(self):
        with self.assertRaises(UnsafeOutputDir):
            session_mounts(self.session, "/etc", "I", "O")


if __name__ == "__main__":
    unittest.main(verbosity=2)
