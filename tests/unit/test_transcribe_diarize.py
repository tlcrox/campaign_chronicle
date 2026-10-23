#!/usr/bin/env python3
"""Diarize and hotwords are overridden when each file is already one speaker.

Audacity per-player exports are one speaker per file, named from the filename,
and mostly silence while the others talk. Two whisper settings backfire there,
so neither reaches the container for that source whatever the config says:
  - diarize splits the single speaker into SPEAKER_XX that map_speakers must undo;
  - hotwords are injected into the decoder every chunk, so across the silences
    the decoder echoes them as name-salad hallucinations.
A mixed standalone recording or a video keeps both.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cc_stages import transcribe
from cc_stages.transcribe import TranscriptionSource, discover_source


class DiscoverSourceFlags(unittest.TestCase):
    """The decision: which sources are one-speaker-per-file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _touch(self, rel):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")

    def test_audacity_is_one_speaker_per_file(self):
        self._touch("session.mkv")
        self._touch("craig.aup/x_data/1-alice.flac")
        self._touch("craig.aup/x_data/2-bob.flac")
        src = discover_source(self.root)
        self.assertEqual(src.kind, "audacity")
        self.assertTrue(src.one_speaker_per_file)

    def test_video_is_not(self):
        self._touch("only.mp4")
        src = discover_source(self.root)
        self.assertEqual(src.kind, "video")
        self.assertFalse(src.one_speaker_per_file)
        self.assertTrue(src.needs_diarization)

    def test_standalone_audio_is_not(self):
        """One file can still hold several speakers, so diarization stays on."""
        self._touch("mixed.wav")
        src = discover_source(self.root)
        self.assertEqual(src.kind, "standalone")
        self.assertFalse(src.one_speaker_per_file)


class _Cfg:
    """Minimal config: the whisper_* properties whisper_env reads, plus the bits
    run() touches. diarize is ON, so the flip is what must turn it off."""
    whisper_model = "base"
    whisper_language = ""
    whisper_compute_type = "int8"
    whisper_device = "cpu"
    whisper_batch_size = 8
    whisper_output_format = "json"
    whisper_diarize = True
    whisper_hotwords = "Arlin Shara Hako"
    whisper_initial_prompt = ""
    output_base_dir = None
    repo_root = Path(".")


class OneSpeakerEnvOverrides(unittest.TestCase):
    """The consequence: what the container is actually handed. On a one-speaker
    source both diarize and hotwords are forced off, whatever the config says;
    on a mixed source both stand."""

    def _env_for(self, source: TranscriptionSource) -> dict:
        captured = {}

        def capture(_cmd, _dry, env_vars=None, **kw):
            captured.update(env_vars)
            return True

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(transcribe, "discover_source", return_value=source), \
             mock.patch.object(transcribe, "session_mounts",
                               return_value=([], {"AUDIO_DIR": "/a", "OUTPUT_DIR": "/o"},
                                             Path(d))), \
             mock.patch.object(transcribe, "clear_mount"), \
             mock.patch.object(transcribe, "compose_run", return_value=["cmd"]), \
             mock.patch.object(transcribe, "require_diarization_for_mapping"), \
             mock.patch.object(transcribe, "run_docker_command", side_effect=capture):
            self.assertTrue(transcribe.run(Path(d), config=_Cfg()))
        return captured

    def test_audacity_forces_diarize_and_hotwords_off(self):
        src = TranscriptionSource(kind="audacity", files=[Path("1-alice.flac")],
                                  one_speaker_per_file=True)
        env = self._env_for(src)
        self.assertEqual(env["WHISPER_DIARIZE"], "false")
        self.assertEqual(env["WHISPER_HOTWORDS"], "",
                         "hotwords must not reach the decoder for per-speaker audio")

    def test_video_keeps_diarize_and_hotwords(self):
        src = TranscriptionSource(kind="video", files=[Path("v.mp4")],
                                  needs_diarization=True)
        env = self._env_for(src)
        self.assertEqual(env["WHISPER_DIARIZE"], "true")
        self.assertEqual(env["WHISPER_HOTWORDS"], "Arlin Shara Hako")


if __name__ == "__main__":
    unittest.main()
