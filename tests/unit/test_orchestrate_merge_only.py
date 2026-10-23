#!/usr/bin/env python3
"""--merge-only re-maps speaker names before merging.

Speaker names are stamped into the per-source transcripts by the map_speakers
stage, not by the merge. A full run does that right after transcription, but
--merge-only used to call only the merge tools — so a session whose
speaker_config.json arrived after transcription re-merged with raw SPEAKER_XX
and no names, and the only way to get names was a full (re-transcribing) run.
run_merge_only closes that: map first, then merge, no re-transcribe.
"""

import unittest
from pathlib import Path
from unittest import mock

import orchestrate

SESSION = Path("SomeSession")


class RunMergeOnly(unittest.TestCase):
    def test_maps_and_cleans_before_merging(self):
        calls = []
        with mock.patch.object(orchestrate.map_speakers, "run",
                               side_effect=lambda *a, **k: calls.append("map") or True), \
             mock.patch.object(orchestrate.clean_transcription, "run",
                               side_effect=lambda *a, **k: calls.append("clean") or True), \
             mock.patch.object(orchestrate, "run_merge_tools",
                               side_effect=lambda *a, **k: calls.append("merge") or True):
            self.assertTrue(orchestrate.run_merge_only(SESSION, config=None))
        self.assertEqual(calls, ["map", "clean", "merge"],
                         "names and corrections must be applied before the merge")

    def test_returns_the_merge_result_not_the_pre_steps(self):
        """The merge is the deliverable, so its success is the one that counts."""
        with mock.patch.object(orchestrate.map_speakers, "run", return_value=True), \
             mock.patch.object(orchestrate.clean_transcription, "run", return_value=True), \
             mock.patch.object(orchestrate, "run_merge_tools", return_value=False):
            self.assertFalse(orchestrate.run_merge_only(SESSION, config=None))

    def test_pre_steps_are_best_effort_merge_still_runs(self):
        """No config -> map/clean return False, but the merge must still run and
        decide the outcome."""
        merged = []
        with mock.patch.object(orchestrate.map_speakers, "run", return_value=False), \
             mock.patch.object(orchestrate.clean_transcription, "run", return_value=False), \
             mock.patch.object(orchestrate, "run_merge_tools",
                               side_effect=lambda *a, **k: merged.append(True) or True):
            self.assertTrue(orchestrate.run_merge_only(SESSION, config=None))
        self.assertEqual(merged, [True])

    def test_dry_run_is_passed_through_to_every_step(self):
        with mock.patch.object(orchestrate.map_speakers, "run", return_value=True) as m, \
             mock.patch.object(orchestrate.clean_transcription, "run", return_value=True) as c, \
             mock.patch.object(orchestrate, "run_merge_tools", return_value=True) as g:
            orchestrate.run_merge_only(SESSION, config="CFG", dry_run=True)
        self.assertEqual(m.call_args.kwargs.get("dry_run"), True)
        self.assertEqual(c.call_args.kwargs.get("dry_run"), True)
        self.assertEqual(g.call_args, mock.call(SESSION, "CFG", True))


if __name__ == "__main__":
    unittest.main()
