#!/usr/bin/env python3
"""
Tests for pipeline.transcribe.docker_env.whisper_env — the config -> WHISPER_*
env builder that makes config.yaml authoritative for transcription.

Run from scripts/:
    python3 -m unittest pipeline.transcribe.test_docker_env -v
"""

import unittest


from pipeline.transcribe.docker_env import whisper_env  # noqa: E402


class _Cfg:
    """Minimal stand-in exposing the whisper_* properties whisper_env reads."""
    def __init__(self, **over):
        base = dict(
            whisper_model="base",
            whisper_language="",
            whisper_compute_type="int8",
            whisper_device="cpu",
            whisper_batch_size=8,
            whisper_output_format="json",
            whisper_diarize=False,
            whisper_hotwords="",
        )
        base.update(over)
        self.__dict__.update(base)


class WhisperEnv(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(whisper_env(_Cfg()), {
            "WHISPER_MODEL": "base",
            "WHISPER_LANGUAGE": "",
            "WHISPER_COMPUTE_TYPE": "int8",
            "WHISPER_DEVICE": "cpu",
            "WHISPER_BATCH_SIZE": "8",
            "WHISPER_OUTPUT_FORMAT": "json",
            "WHISPER_DIARIZE": "false",
            "WHISPER_HOTWORDS": "",
            # --initial_prompt is a separate flag from --hotwords
            "WHISPER_INITIAL_PROMPT": "",
        })

    def test_diarize_true_is_lowercase_string(self):
        self.assertEqual(whisper_env(_Cfg(whisper_diarize=True))["WHISPER_DIARIZE"], "true")

    def test_all_values_are_strings(self):
        env = whisper_env(_Cfg(whisper_batch_size=16))
        self.assertTrue(all(isinstance(v, str) for v in env.values()))
        self.assertEqual(env["WHISPER_BATCH_SIZE"], "16")

    def test_no_hf_token_key(self):
        # HF_TOKEN is a secret from .env — must NOT be built here.
        self.assertNotIn("HF_TOKEN", whisper_env(_Cfg()))

    def test_none_language_becomes_empty(self):
        self.assertEqual(whisper_env(_Cfg(whisper_language=None))["WHISPER_LANGUAGE"], "")

    def test_hotwords_empty_by_default(self):
        self.assertEqual(whisper_env(_Cfg())["WHISPER_HOTWORDS"], "")

    def test_hotwords_passed_through(self):
        phrases = "Mechanon, Aberrancy, OCV, DCV"
        self.assertEqual(whisper_env(_Cfg(whisper_hotwords=phrases))["WHISPER_HOTWORDS"], phrases)

    def test_device_passed_through(self):
        # device travels the same path as compute_type; the two are coupled
        # (CPU has no efficient float16 path in CTranslate2).
        self.assertEqual(whisper_env(_Cfg(whisper_device="cuda"))["WHISPER_DEVICE"], "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
