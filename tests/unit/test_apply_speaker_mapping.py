#!/usr/bin/env python3
"""
Tests for the apply_speaker_mapping.py tool (Workflow B speaker mapping).

Deterministic, no Docker / GPU / real config.yaml needed: every case passes an
explicit speaker_config.json, so the tool's config default is never consulted.

Run from scripts/:
    python3 -m unittest tests.unit.test_apply_speaker_mapping -v
Or directly:
    python3 tools/test_apply_speaker_mapping.py
"""

import json
import os
import tempfile
import unittest
from pathlib import Path


# The tool calls get_config() at import; point it at the real config.yaml so it
# resolves regardless of cwd. Its contents are irrelevant here (tests pass an
# explicit speaker-config and a stub config to the tool function).
os.environ.setdefault(
    "WHISPERX_CONFIG", str(Path(__file__).resolve().parents[2] / "config" / "config.yaml")
)

from cc_stages.apply_speaker_mapping import apply_speaker_mapping_tool  # noqa: E402
# Both `pipeline` and `cc_stages` come from the installed project (pyproject.toml).
from pipeline.common.mounts import SESSION_OUTPUT_SUBDIR  # noqa: E402
from pipeline.transcribe.map_speakers import (  # noqa: E402
    require_diarization_for_mapping,
    map_speakers,
)


def _transcript():
    """Fresh transcript dict with segment- and word-level SPEAKER_XX IDs."""
    return {
        "segments": [
            {"speaker": "SPEAKER_00", "text": "a",
             "words": [{"word": "a", "speaker": "SPEAKER_00"}]},
            {"speaker": "SPEAKER_01", "text": "b",
             "words": [{"word": "b", "speaker": "SPEAKER_01"}]},
            {"speaker": "SPEAKER_99", "text": "c",
             "words": [{"word": "c", "speaker": "SPEAKER_99"}]},
        ]
    }


SPEAKER_CONFIG = {
    "global_mapping": {"SPEAKER_00": "Michael", "SPEAKER_01": "Lily"},
    "session_mappings": {"Week 13": {"SPEAKER_01": "Moria"}},
}


class _StubConfig:
    """Stand-in for config.Config; only speaker_config_file could be read, and
    the tests always pass an explicit speaker-config path so it never is.

    whisper_diarize=True because these tests simulate Workflow B: diarization
    has already run and produced the SPEAKER_XX tokens being remapped here.
    require_diarization_for_mapping() now checks this, so it must be set."""
    speaker_config_file = "unused-speaker_config.json"
    whisper_diarize = True


class ApplySpeakerMappingTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.cfg_path = self.d / "speaker_config.json"
        self.cfg_path.write_text(json.dumps(SPEAKER_CONFIG))
        # session dir with the transcriptions/<name>.json layout
        self.session = self.d / "Week 99"
        (self.session / SESSION_OUTPUT_SUBDIR).mkdir(parents=True)
        self.tpath = self.session / SESSION_OUTPUT_SUBDIR / "vid.json"
        self.tpath.write_text(json.dumps(_transcript()))

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, session_dir, out=None):
        ok = apply_speaker_mapping_tool(
            self.tpath, session_dir=session_dir, output_file=out,
            speaker_config_file=self.cfg_path, config=_StubConfig(),
        )
        self.assertTrue(ok)
        return json.loads(Path(out or self.tpath).read_text())

    def test_global_mapping_segments_and_words(self):
        res = self._run(self.session)  # "Week 99" not in session_mappings -> global only
        spk = [s["speaker"] for s in res["segments"]]
        self.assertEqual(spk, ["Michael", "Lily", "SPEAKER_99"])  # unmapped left as-is
        # word-level updated too
        self.assertEqual(res["segments"][0]["words"][0]["speaker"], "Michael")
        self.assertEqual(res["segments"][1]["words"][0]["speaker"], "Lily")
        self.assertEqual(res["segments"][2]["words"][0]["speaker"], "SPEAKER_99")

    def test_session_override_beats_global(self):
        session13 = self.d / "Week 13"
        (session13 / SESSION_OUTPUT_SUBDIR).mkdir(parents=True)
        # move transcript under the Week 13 session so session_id == "Week 13"
        self.tpath = session13 / SESSION_OUTPUT_SUBDIR / "vid.json"
        self.tpath.write_text(json.dumps(_transcript()))
        res = self._run(session13)
        spk = [s["speaker"] for s in res["segments"]]
        self.assertEqual(spk, ["Michael", "Moria", "SPEAKER_99"])

    def test_output_file_leaves_input_untouched(self):
        out = self.d / "mapped.json"
        res = self._run(self.session, out=out)
        self.assertEqual(res["segments"][0]["speaker"], "Michael")
        # original still has raw IDs
        orig = json.loads(self.tpath.read_text())
        self.assertEqual(orig["segments"][0]["speaker"], "SPEAKER_00")

    def test_missing_speaker_config_is_graceful(self):
        ok = apply_speaker_mapping_tool(
            self.tpath, session_dir=self.session,
            speaker_config_file=self.d / "does_not_exist.json",
            config=_StubConfig(),
        )
        self.assertTrue(ok)  # tool still succeeds (writes file)
        res = json.loads(self.tpath.read_text())
        self.assertEqual(res["segments"][0]["speaker"], "SPEAKER_00")  # unchanged

    def test_session_mappings_are_keyed_on_the_session_directory(self):
        """The session dir is now required, so this is what it buys.

        This replaced a test for the removed session_dir=None inference, which
        took the transcript's grandparent — `cc_output` on this very fixture's
        layout, not the session. It passed anyway: it asserted the SPEAKER_00
        global mapping, which resolves the same whichever wrong directory the
        inference produced, since neither is in session_mappings.
        """
        res = self._run(self.session)                    # "Week 99": global only
        self.assertEqual(res["segments"][0]["speaker"], "Michael")
        self.assertEqual(res["segments"][1]["speaker"], "Lily")

    def test_dry_run_writes_nothing(self):
        ok = apply_speaker_mapping_tool(
            self.tpath, session_dir=self.session, speaker_config_file=self.cfg_path,
            config=_StubConfig(), dry_run=True,
        )
        self.assertTrue(ok)
        res = json.loads(self.tpath.read_text())
        self.assertEqual(res["segments"][0]["speaker"], "SPEAKER_00")  # untouched

    def test_missing_transcript_fails(self):
        ok = apply_speaker_mapping_tool(
            self.d / "nope.json", session_dir=self.session,
            speaker_config_file=self.cfg_path, config=_StubConfig(),
        )
        self.assertFalse(ok)


class _DiarCfg:
    """Config stub exposing just what the guard reads."""
    def __init__(self, speaker_config_file, diarize):
        self.speaker_config_file = speaker_config_file
        self.whisper_diarize = diarize


class RequireDiarizationForMapping(unittest.TestCase):
    """Speaker-mapping guard behavior, by (config present?, diarize on?):

    - config present, diarize OFF: self-contradictory (mapping configured but
      diarization will never produce SPEAKER_XX tokens to rename) -> raises.
    - diarize ON, config missing/unresolvable: normal bootstrapping state (you
      need a diarized transcript's SPEAKER_XX IDs before you can write a
      config that maps them) -> does NOT raise, but warns loudly so it's not
      silently a no-op.
    - diarize OFF, config missing: fully consistent, nothing wanted, nothing
      happens -> silent, no raise, no warning.
    - config present, diarize ON: everything lines up -> silent.
    - Unreadable/unknown diarize state defaults to "off", so it's judged the
      same as diarize=False above."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.scf = self.d / "speaker_config.json"
        self.scf.write_text("{}")

    def tearDown(self):
        self.tmp.cleanup()

    def test_config_present_diarize_off_raises(self):
        with self.assertRaises(ValueError):
            require_diarization_for_mapping(_DiarCfg(self.scf, False))

    def test_config_present_diarize_on_ok(self):
        require_diarization_for_mapping(_DiarCfg(self.scf, True))  # no raise

    def test_config_absent_diarize_off_is_silent(self):
        # Nothing configured, nothing running -> consistent, no complaint.
        require_diarization_for_mapping(_DiarCfg(self.d / "nope.json", False))

    def test_config_absent_diarize_on_warns_not_raises(self):
        # Bootstrapping state: diarize on, no speaker_config.json yet. Must
        # not raise (that would block the very first diarization run used to
        # discover the SPEAKER_XX IDs), but should warn loudly that mapping
        # is a no-op right now.
        with self.assertLogs(level="WARNING") as cm:
            require_diarization_for_mapping(_DiarCfg(self.d / "nope.json", True))
        self.assertTrue(any("DID YOU REALLY MEAN TO" in msg for msg in cm.output))

    def test_explicit_override_path_off_raises(self):
        with self.assertRaises(ValueError):
            require_diarization_for_mapping(_DiarCfg(self.d / "unused.json", False), self.scf)

    def test_explicit_override_path_missing_diarize_on_warns_not_raises(self):
        with self.assertLogs(level="WARNING") as cm:
            require_diarization_for_mapping(
                _DiarCfg(self.d / "unused.json", True), self.d / "also_missing.json"
            )
        self.assertTrue(any("DID YOU REALLY MEAN TO" in msg for msg in cm.output))

    def test_unknown_diarize_state_defaults_off_and_raises_if_config_present(self):
        class _Raising:
            speaker_config_file = None  # set per-instance below
            @property
            def whisper_diarize(self):
                raise RuntimeError("cannot determine")
        stub = _Raising()
        stub.speaker_config_file = self.scf  # exists
        # Guard can't read diarize state -> defaults to "off" -> judged as
        # config-present-diarize-off -> raises.
        with self.assertRaises(ValueError):
            require_diarization_for_mapping(stub)

    def test_unknown_diarize_state_defaults_off_and_silent_if_config_absent(self):
        class _Raising:
            speaker_config_file = None  # set per-instance below
            @property
            def whisper_diarize(self):
                raise RuntimeError("cannot determine")
        stub = _Raising()
        stub.speaker_config_file = self.d / "nope.json"  # missing
        # Defaults to "off" -> config-absent-diarize-off -> silent.
        require_diarization_for_mapping(stub)

    def test_unresolvable_config_path_diarize_on_warns_not_crashes(self):
        # A config whose speaker_config_file property itself raises must still
        # produce a warning, not an AttributeError, when diarize is ON.
        class _BadConfigAttr:
            whisper_diarize = True
            @property
            def speaker_config_file(self):
                raise RuntimeError("not configured")
        with self.assertLogs(level="WARNING") as cm:
            require_diarization_for_mapping(_BadConfigAttr())
        self.assertTrue(any("DID YOU REALLY MEAN TO" in msg for msg in cm.output))

    def test_unresolvable_config_path_diarize_off_is_silent(self):
        class _BadConfigAttr:
            whisper_diarize = False
            @property
            def speaker_config_file(self):
                raise RuntimeError("not configured")
        require_diarization_for_mapping(_BadConfigAttr())  # no raise


class MapSpeakersStage(unittest.TestCase):
    """map_speakers maps a session's transcriptions/ in place."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.scf = self.d / "speaker_config.json"
        self.scf.write_text(json.dumps(SPEAKER_CONFIG))
        self.session = self.d / "Week 99"
        (self.session / SESSION_OUTPUT_SUBDIR).mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _cfg(self):
        return _DiarCfg(self.scf, True)

    def test_maps_all_transcripts_in_place(self):
        t = self.session / SESSION_OUTPUT_SUBDIR / "vid.json"
        t.write_text(json.dumps(_transcript()))
        self.assertTrue(map_speakers(self.session, self._cfg()))
        res = json.loads(t.read_text())
        self.assertEqual(
            [s["speaker"] for s in res["segments"]],
            ["Michael", "Lily", "SPEAKER_99"],
        )

    def test_excludes_combined_and_returns_false_when_no_sources(self):
        # only a combined file present -> nothing to map
        (self.session / SESSION_OUTPUT_SUBDIR / "x_transcript_combined.json").write_text(
            json.dumps(_transcript())
        )
        self.assertFalse(map_speakers(self.session, self._cfg()))

    def test_dry_run_leaves_files_untouched(self):
        t = self.session / SESSION_OUTPUT_SUBDIR / "vid.json"
        t.write_text(json.dumps(_transcript()))
        self.assertTrue(map_speakers(self.session, self._cfg(), dry_run=True))
        res = json.loads(t.read_text())
        self.assertEqual(res["segments"][0]["speaker"], "SPEAKER_00")  # untouched


if __name__ == "__main__":
    unittest.main(verbosity=2)
