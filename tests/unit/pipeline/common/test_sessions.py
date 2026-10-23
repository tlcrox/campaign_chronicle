#!/usr/bin/env python3
"""
Tests for pipeline.common.sessions: media discovery (media_files) and
find_sessions — explicit-list resolution with fail-loud behaviour.

Run from scripts/:
    python3 -m unittest pipeline.common.test_sessions -v
"""

import tempfile
import unittest
from unittest import mock
from pathlib import Path


from pipeline.common.sessions import (  # noqa: E402
    media_files, find_sessions, detect_audio_source, NoSessionsError)


class MediaFiles(unittest.TestCase):
    """One walk filtering on suffix, where there was a glob per extension.

    detect_audio_source ran an rglob per extension per subdirectory — ten
    recursive walks of an Audacity project folder to answer one question.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _touch(self, rel):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p

    def test_selects_only_the_named_extensions(self):
        self._touch("a.wav")
        self._touch("b.mp3")
        self._touch("c.txt")
        found = media_files(self.root, {".wav", ".mp3"})
        self.assertEqual([p.name for p in found], ["a.wav", "b.mp3"])

    def test_is_case_insensitive_on_every_platform(self):
        """The old globs relied on the Windows filesystem for this, so a .MP4
        was invisible on Linux."""
        self._touch("LOUD.WAV")
        self.assertEqual([p.name for p in media_files(self.root, {".wav"})],
                         ["LOUD.WAV"])

    def test_a_directory_named_like_a_file_is_not_a_file(self):
        (self.root / "notreally.wav").mkdir()
        self._touch("real.wav")
        self.assertEqual([p.name for p in media_files(self.root, {".wav"})],
                         ["real.wav"])

    def test_shallow_by_default(self):
        self._touch("top.wav")
        self._touch("sub/deep.wav")
        self.assertEqual([p.name for p in media_files(self.root, {".wav"})],
                         ["top.wav"])

    def test_recursive_reaches_nested_exports(self):
        """Audacity exports sit several levels down."""
        self._touch("craig.aup/data/e00/one.flac")
        self._touch("craig.aup/data/e01/two.flac")
        found = media_files(self.root, {".flac"}, recursive=True)
        self.assertEqual([p.name for p in found], ["one.flac", "two.flac"])

    def test_sorted_not_filesystem_order(self):
        for name in ("c.wav", "a.wav", "b.wav"):
            self._touch(name)
        self.assertEqual([p.name for p in media_files(self.root, {".wav"})],
                         ["a.wav", "b.wav", "c.wav"])

    def test_walks_once_regardless_of_how_many_extensions(self):
        """The point of the change: cost is the walk, not the extension count."""
        self._touch("x.wav")
        calls = []
        real = Path.glob

        def counting(self, pattern, *a, **k):
            calls.append(pattern)
            return real(self, pattern, *a, **k)

        with mock.patch.object(Path, "glob", counting):
            media_files(self.root, {".wav", ".mp3", ".flac", ".ogg", ".m4a"})
        self.assertEqual(len(calls), 1)

    def test_missing_directory_is_empty_not_an_error(self):
        self.assertEqual(media_files(self.root / "nope", {".wav"}), [])


class DetectAudioSource(unittest.TestCase):
    """Classifying a session by what media it holds."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _touch(self, rel):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        return p

    def test_audacity_exports_are_found_under_the_project_dir(self):
        self._touch("2026-09-18.mkv")                       # session video
        self._touch("craig.aup/x_data/1-alice.flac")
        self._touch("craig.aup/x_data/2-bob.flac")
        kind, files = detect_audio_source(self.root)
        self.assertEqual(kind, "audacity")
        self.assertEqual(sorted(f.name for f in files),
                         ["1-alice.flac", "2-bob.flac"])

    def test_a_screen_capture_video_is_not_an_audacity_export(self):
        """A .mkv in a tool/screens subfolder must not be collected as export
        audio: it drags the exports' common parent up to the session root, and
        transcribe then drops the subdir narrowing and falls back to the root
        video — the Audacity audio silently ignored. (Week 18 live regression.)"""
        self._touch("2026-09-18.mkv")
        self._touch("craig.aup/x_data/1-alice.flac")
        self._touch("tool/screens/2026-09-18 20-43-55.mkv")
        kind, files = detect_audio_source(self.root)
        self.assertEqual(kind, "audacity")
        self.assertEqual([f.name for f in files], ["1-alice.flac"])
        # The exports' common parent stays inside the project dir.
        import os
        parent = Path(os.path.commonpath([str(f.parent) for f in files]))
        self.assertNotEqual(parent, self.root)

    def test_standalone_audio_without_a_project(self):
        self._touch("session.wav")
        kind, files = detect_audio_source(self.root)
        self.assertEqual(kind, "standalone")
        self.assertEqual([f.name for f in files], ["session.wav"])

    def test_video_only(self):
        self._touch("only.mp4")
        kind, files = detect_audio_source(self.root)
        self.assertEqual(kind, "video")
        self.assertEqual(files, [])

    def test_project_with_no_exports_falls_back_to_video(self):
        self._touch("only.mkv")
        (self.root / "craig.aup").mkdir()
        kind, files = detect_audio_source(self.root)
        self.assertEqual(kind, "video")


class FindSessions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        for n in ("B", "A", "C"):
            (self.base / n).mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_list_fails_loud(self):
        with self.assertRaises(NoSessionsError):
            find_sessions([])

    def test_none_fails_loud(self):
        with self.assertRaises(NoSessionsError):
            find_sessions(None)

    def test_returns_sorted_existing(self):
        dirs = [self.base / "C", self.base / "A", self.base / "B"]
        self.assertEqual(find_sessions(dirs), sorted(dirs))

    def test_skips_missing_dirs(self):
        dirs = [self.base / "A", self.base / "GONE", self.base / "B"]
        self.assertEqual(find_sessions(dirs), [self.base / "A", self.base / "B"])

    def test_skips_files(self):
        f = self.base / "note.txt"
        f.write_text("x")
        self.assertEqual(find_sessions([f, self.base / "A"]), [self.base / "A"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class SkipReporting(unittest.TestCase):
    """find_sessions is the single existence gate; it must not fail silently."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_all_missing_fails_loud(self):
        """Raises rather than returning [] — a caller cannot act on an empty list."""
        dirs = [self.base / "Nope1", self.base / "Nope2"]
        with self.assertRaises(NoSessionsError) as ctx:
            find_sessions(dirs)
        self.assertIn("None of the 2", str(ctx.exception))

    def test_partial_skip_still_returns_the_survivors(self):
        (self.base / "A").mkdir()
        dirs = [self.base / "A", self.base / "Missing"]
        self.assertEqual(find_sessions( dirs), [self.base / "A"])

    def test_skip_reason_distinguishes_file_from_missing(self):
        (self.base / "A").mkdir()
        f = self.base / "afile.txt"
        f.write_text("x")
        with self.assertLogs("pipeline.common.sessions", level="WARNING") as logs:
            find_sessions([f, self.base / "Gone", self.base / "A"])
        joined = "\n".join(logs.output)
        self.assertIn("not a directory", joined)
        self.assertIn("does not exist", joined)
        self.assertIn("2 of 3", joined)
