#!/usr/bin/env python3
"""Running a session and counting its outcome, now that they are separable.

The parallel and sequential branches each carried a verbatim copy of the status
tally and the exception fallback. They now share ``_run_session`` and ``_tally``,
which is what makes either testable at all — nothing exercises ``main()``, so
before this the counting had no coverage in any form.
"""

import unittest
from pathlib import Path
from unittest import mock

import orchestrate

SESSION = Path("SomeSession")


class RunSession(unittest.TestCase):
    """One session, with the exception policy in one place."""

    def _run(self):
        return orchestrate._run_session(SESSION, config=None, dry_run=True,
                                        no_scenes=False, no_transcription=False,
                                        roi_file="")

    def test_passes_the_status_through(self):
        with mock.patch.object(orchestrate, "process_session",
                               return_value=(True, "completed", [])):
            self.assertEqual(self._run(), ("completed", []))

    def test_passes_failures_through(self):
        with mock.patch.object(orchestrate, "process_session",
                               return_value=(False, "failed", ["merge failed"])):
            self.assertEqual(self._run(), ("failed", ["merge failed"]))

    def test_an_unhandled_exception_becomes_a_failed_status(self):
        """A raising session must not take the whole run down, in either branch —
        which is why the parallel path no longer needs a try of its own."""
        with mock.patch.object(orchestrate, "process_session",
                               side_effect=RuntimeError("boom")):
            status, failures = self._run()
        self.assertEqual(status, "failed")
        self.assertEqual(len(failures), 1)
        self.assertIn("boom", failures[0])
        self.assertIn("unhandled exception", failures[0])


class Tally(unittest.TestCase):
    """Counting, with no knowledge of how the session was run."""

    def setUp(self):
        self.counts = {"completed": 0, "failed": 0, "skipped": 0}
        self.session_failures = {}

    def _tally(self, status, failures=()):
        orchestrate._tally(SESSION, status, list(failures),
                           self.counts, self.session_failures)

    def test_completed(self):
        self._tally("completed")
        self.assertEqual(self.counts, {"completed": 1, "failed": 0, "skipped": 0})
        self.assertEqual(self.session_failures, {})

    def test_failed_records_the_messages_under_the_session_name(self):
        self._tally("failed", ["transcription failed", "merge failed"])
        self.assertEqual(self.counts["failed"], 1)
        self.assertEqual(self.session_failures[SESSION.name],
                         ["transcription failed", "merge failed"])

    def test_no_media_is_skipped(self):
        self._tally("no_media")
        self.assertEqual(self.counts["skipped"], 1)

    def test_an_unrecognised_status_is_skipped_not_dropped(self):
        """The old code had an explicit else for this; losing it would make a
        session vanish from the totals rather than be counted."""
        self._tally("something_new")
        self.assertEqual(self.counts["skipped"], 1)
        self.assertEqual(sum(self.counts.values()), 1)

    def test_every_session_lands_in_exactly_one_bucket(self):
        for status in ("completed", "failed", "no_media", "whatever"):
            self._tally(status)
        self.assertEqual(sum(self.counts.values()), 4)


if __name__ == "__main__":
    unittest.main()
