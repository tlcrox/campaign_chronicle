#!/usr/bin/env python3
"""
Tests for pipeline.common.docker: the -e/-v flag injection into the compose
argv, the transient-error classifier, the retry behaviour, and the logging
around a failed run.

No real Docker: subprocess.run (and time.sleep) are patched.

Run from scripts/:
    python3 -m unittest pipeline.common.test_docker -v
"""

import re
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


from pipeline.common import docker  # noqa: E402  (module object: tests patch docker.subprocess)
from pipeline.common.docker import (  # noqa: E402
    run_docker_command,
    printable_command,
    _is_transient_docker_error,
)

BASE = ["docker", "compose", "run", "--rm", "scenes"]


def _result(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# Compose narrates the container's lifecycle on stderr before the container's own
# output. These are the real shapes, copied from captured runs against the local
# pyscenedetect image — a container that ran and failed, and a daemon that was
# never reachable, produce the same exit code (1), so this is the only thing
# separating them.
_RAN_AND_FAILED = (
    " Container pyscenedetect-scenes-run-988292626c10  Creating\n"
    " Container pyscenedetect-scenes-run-988292626c10  Created\n"
    "ERR: unexpected EOF while reading input\n"
)
_REFUSED_AT_STARTUP = (
    " Container pyscenedetect-scenes-run-b33ad90956c9  Creating\n"
    " Container pyscenedetect-scenes-run-b33ad90956c9  Created\n"
    "/usr/local/bin/scripts/detect_scenes.sh: line 22: SCENE_THRESHOLD: "
    "config.yaml owns this (scenes.threshold)\n"
)
_NEVER_STARTED = (
    'error during connect: Get "http://127.0.0.1:9/v1.54/networks?filters=...": '
    "dial tcp 127.0.0.1:9: connectex: No connection could be made because the "
    "target machine actively refused it.\n"
)


class TransientClassifier(unittest.TestCase):
    def test_network_500_is_transient(self):
        s = ('request returned 500 Internal Server Error for API route and version '
             '.../networks?... check if the server supports the requested API version')
        self.assertTrue(_is_transient_docker_error(s))

    def test_cannot_connect_is_transient(self):
        self.assertTrue(_is_transient_docker_error("Cannot connect to the Docker daemon"))

    def test_real_failures_are_not_transient(self):
        self.assertFalse(_is_transient_docker_error("RuntimeError: CUDA out of memory"))
        self.assertFalse(_is_transient_docker_error("Error: No audio files found in /audio"))
        self.assertFalse(_is_transient_docker_error(""))
        self.assertFalse(_is_transient_docker_error(None))


class ContainerStartedMeansTheFailureIsReal(unittest.TestCase):
    """Once the container runs, whatever it says is the run's own verdict.

    Retrying is for the daemon failing to set the run up. Repeating a container
    that already answered just spends another DOCKER_RUN_TIMEOUT — two hours an
    attempt — reaching the same answer.
    """

    def test_a_container_that_ran_and_failed_is_not_retried(self):
        self.assertFalse(_is_transient_docker_error(_RAN_AND_FAILED))

    def test_a_container_that_refused_at_startup_is_not_retried(self):
        """No stdout at all, so 'did we get output?' would get this wrong."""
        self.assertFalse(_is_transient_docker_error(_REFUSED_AT_STARTUP))

    def test_a_daemon_that_never_started_it_is_retried(self):
        self.assertTrue(_is_transient_docker_error(_NEVER_STARTED))

    def test_a_started_container_wins_over_a_transient_marker(self):
        """The guard runs first, so a broad marker cannot swallow a real failure."""
        noisy = _RAN_AND_FAILED + "connection refused by the media server\n"
        self.assertFalse(_is_transient_docker_error(noisy))

    def test_the_lifecycle_words_compose_uses(self):
        for word in ("Creating", "Created", "Starting", "Started"):
            with self.subTest(word=word):
                self.assertFalse(_is_transient_docker_error(
                    f" Container demo-run-1  {word}\nboom\n"))


class EofIsNotAWordSearch(unittest.TestCase):
    """A bare "eof" substring retried four kinds of genuine failure.

    Each retry costs a full DOCKER_RUN_TIMEOUT, so a truncated input file was
    six hours from being reported. Docker writes the transient case as "...: EOF".
    """

    GENUINE = (
        "ffmpeg: Invalid data found: unexpected EOF while reading input",
        "EOFError: Ran out of input",
        "tar: Unexpected EOF in archive",
        "json.decoder.JSONDecodeError: Expecting value: unexpected EOF",
    )
    DAEMON = (
        'error during connect: Post "http://docker/v1.47/containers": EOF',
        "request returned Internal Server Error ... : EOF",
        "read tcp 127.0.0.1:2375: read: connection reset by peer: EOF",
    )

    def test_genuine_failures_carrying_eof_are_reported(self):
        for stderr in self.GENUINE:
            with self.subTest(stderr=stderr[:40]):
                self.assertFalse(_is_transient_docker_error(stderr))

    def test_the_daemon_form_is_still_caught(self):
        for stderr in self.DAEMON:
            with self.subTest(stderr=stderr[:40]):
                self.assertTrue(_is_transient_docker_error(stderr))


class FlagInjection(unittest.TestCase):
    def _run_capture(self, command=None, **kwargs):
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            captured["kwargs"] = kw
            return _result(returncode=0, stdout="", stderr="")

        with mock.patch.object(docker.subprocess, "run", fake_run):
            ok = run_docker_command(list(command or BASE), **kwargs)
        self.assertTrue(ok)
        return captured

    def test_env_and_volume_flags(self):
        cmd = self._run_capture(
            env_vars={"SCENE_ROI": "1 2 3 4"},
            volumes=["/host/sess:/session_input:ro", "/host/out:/session_output"],
        )["cmd"]
        # Each flag and its value are SEPARATE argv entries, unquoted.
        self.assertEqual(
            cmd,
            ["docker", "compose", "run", "--rm",
             "-e", "SCENE_ROI=1 2 3 4",
             "-v", "/host/sess:/session_input:ro",
             "-v", "/host/out:/session_output",
             "scenes"],
        )

    def test_no_flags_leaves_command_unchanged(self):
        self.assertEqual(self._run_capture()["cmd"], BASE)

    def test_caller_list_is_not_mutated(self):
        mine = list(BASE)
        self._run_capture(command=mine, env_vars={"A": "b"})
        self.assertEqual(mine, BASE)

    def test_dry_run_does_not_invoke_subprocess(self):
        with mock.patch.object(docker.subprocess, "run",
                               side_effect=AssertionError("should not run")):
            self.assertTrue(run_docker_command(list(BASE),
                                               dry_run=True, volumes=["/a:/b:ro"]))


class NoShell(unittest.TestCase):
    """The command is argv, and nothing re-parses it on the way to the process."""

    def _argv(self, **kwargs):
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            captured["kwargs"] = kw
            return _result(returncode=0)

        with mock.patch.object(docker.subprocess, "run", fake_run):
            run_docker_command(list(BASE), **kwargs)
        return captured

    def test_shell_is_never_requested(self):
        self.assertFalse(self._argv()["kwargs"].get("shell", False))

    def test_shell_metacharacters_reach_the_process_intact(self):
        """The values that a shell string mangles: quotes, %VAR%, $VAR, &."""
        values = {
            "QUOTED": 'Sir "Rod" Night',
            "PERCENT": "50%PATH%off",
            "DOLLAR": "$HOME/notes",
            "AMPERSAND": "Rock & Roll",
            "SPACED": "Weeks/Week 13",
            "BACKTICK": "a `whoami` b",
        }
        cmd = self._argv(env_vars=values)["cmd"]
        for key, value in values.items():
            self.assertIn(f"{key}={value}", cmd,
                          f"{key} did not survive as a single argv entry")

    def test_volume_path_with_spaces_is_one_entry(self):
        cmd = self._argv(volumes=["D:/GGG Issues/Week 13:/video:ro"])["cmd"]
        self.assertIn("D:/GGG Issues/Week 13:/video:ro", cmd)

    def test_missing_rm_refuses_rather_than_dropping_flags(self):
        """A string replace on a missing needle would silently drop every -e."""
        with mock.patch.object(docker.subprocess, "run",
                               side_effect=AssertionError("should not run")):
            with self.assertRaises(ValueError):
                run_docker_command(["docker", "ps"], env_vars={"A": "b"})


class PrintableCommand(unittest.TestCase):
    def test_round_trips_to_something_re_runnable(self):
        text = printable_command(["docker", "run", "-e", 'K=Sir "Rod" Night',
                                  "-v", "D:/Week 13:/video", "svc"])
        self.assertIn("docker run", text)
        self.assertIn("Week 13", text)
        self.assertIn("Rod", text)
        # not a python list repr
        self.assertNotIn("['docker'", text)

    def test_dry_run_logs_the_rendered_command(self):
        with self.assertLogs(docker.logger, level="INFO") as logs:
            run_docker_command(list(BASE), dry_run=True)
        joined = "\n".join(logs.output)
        self.assertIn("DRY RUN", joined)
        self.assertIn("docker compose run --rm scenes", joined)


class FailureLogging(unittest.TestCase):
    """A failed run says what ran, shows the tail, and files the rest.

    The terminal used to get every line of both streams — thousands, from a
    failed transcription, on the one path taken when something has already gone
    wrong. The error is nearly always the last thing said, so the tail goes to
    the screen and the whole thing goes to docker.log.
    """

    def _fail_with(self, stdout="", stderr="", **kwargs):
        def fake_run(cmd, **kw):
            return _result(returncode=1, stdout=stdout, stderr=stderr)

        with mock.patch.object(docker.subprocess, "run", fake_run):
            with self.assertLogs(docker.logger, level="ERROR") as logs:
                ok = run_docker_command(list(BASE), **kwargs)
        self.assertFalse(ok)
        return "\n".join(logs.output)

    def test_the_terminal_gets_only_the_tail(self):
        noisy = "\n".join(f"progress line {n}" for n in range(500))
        joined = self._fail_with(stderr=noisy + "\nRuntimeError: the real reason")
        self.assertIn("RuntimeError: the real reason", joined)     # last line kept
        self.assertIn("progress line 499", joined)                 # within the tail
        self.assertNotIn("progress line 0", joined)                # not the flood
        self.assertIn("earlier line(s) in the log", joined)        # says what it cut

    def test_the_tail_is_capped_at_the_documented_limit(self):
        noisy = "\n".join(f"line {n}" for n in range(100))
        joined = self._fail_with(stderr=noisy)
        # assertLogs prefixes every record, so match the payload rather than
        # the start of the line.
        shown = re.findall(r"\bline \d+\b", joined)
        self.assertEqual(len(shown), docker.FAILURE_TAIL_LINES)

    def test_stdout_is_used_when_stderr_is_silent(self):
        """The container explains itself on stdout; compose uses stderr."""
        joined = self._fail_with(stdout="the container's own reason", stderr="")
        self.assertIn("the container's own reason", joined)
        self.assertIn("stdout", joined)

    def test_the_exit_code_is_named(self):
        self.assertIn("exit 1", self._fail_with(stderr="boom"))

    def test_the_whole_output_goes_to_docker_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            joined = self._fail_with(
                stdout="\n".join(f"out {n}" for n in range(200)),
                stderr="\n".join(f"err {n}" for n in range(200)),
                log_dir=tmp)
            log = Path(tmp) / docker.DOCKER_LOG_NAME
            self.assertTrue(log.is_file())
            text = log.read_text(encoding="utf-8")
            self.assertIn("out 0", text)      # the middle the terminal cannot show
            self.assertIn("err 0", text)
            self.assertIn("out 199", text)
            self.assertIn("exit 1", text)
            self.assertIn("docker compose", text)   # the command that produced it
            self.assertIn(str(log), joined)         # and the terminal points at it

    def test_a_second_failure_appends_rather_than_replacing(self):
        """Normally there is one entry per file: each stage logs beside its own
        output, and clears that directory before it runs. Append covers the
        cases that break the assumption rather than losing the earlier failure."""
        with tempfile.TemporaryDirectory() as tmp:
            self._fail_with(stderr="first failure", log_dir=tmp)
            self._fail_with(stderr="second failure", log_dir=tmp)
            text = (Path(tmp) / docker.DOCKER_LOG_NAME).read_text(encoding="utf-8")
            self.assertIn("first failure", text)
            self.assertIn("second failure", text)

    def test_no_log_dir_still_bounds_the_terminal(self):
        joined = self._fail_with(stderr="\n".join(f"line {n}" for n in range(100)))
        self.assertNotIn("line 0", joined)
        self.assertNotIn("full output:", joined)

    def test_an_unwritable_log_does_not_replace_the_failure(self):
        """Writing the log happens while reporting a failure; it must not
        become the failure."""
        with mock.patch.object(docker.Path, "mkdir",
                               side_effect=OSError("read-only file system")):
            with mock.patch.object(docker.subprocess, "run",
                                   lambda cmd, **kw: _result(returncode=1,
                                                             stderr="the real reason")):
                with self.assertLogs(docker.logger, level="WARNING") as logs:
                    ok = run_docker_command(list(BASE), log_dir="/nope")
        self.assertFalse(ok)
        joined = "\n".join(logs.output)
        self.assertIn("the real reason", joined)          # still reported
        self.assertIn("Could not write", joined)          # and says why not filed

    def test_failure_names_the_command_that_ran(self):
        def fake_run(cmd, **kw):
            return _result(returncode=1, stderr="boom")

        with mock.patch.object(docker.subprocess, "run", fake_run):
            with self.assertLogs(docker.logger, level="ERROR") as logs:
                run_docker_command(list(BASE), env_vars={"K": "v"})
        joined = "\n".join(logs.output)
        self.assertIn("K=v", joined)
        self.assertIn("docker compose run --rm", joined)

    def test_missing_executable_names_the_command(self):
        """Without a shell there is no 'not recognized' on stderr to capture."""
        def fake_run(cmd, **kw):
            raise FileNotFoundError(2, "The system cannot find the file specified")

        with mock.patch.object(docker.subprocess, "run", fake_run):
            with self.assertLogs(docker.logger, level="ERROR") as logs:
                ok = run_docker_command(list(BASE))
        self.assertFalse(ok)
        joined = "\n".join(logs.output)
        self.assertIn("Command not found", joined)
        self.assertIn("docker compose", joined)

    def test_empty_streams_log_nothing_extra(self):
        def fake_run(cmd, **kw):
            return _result(returncode=1, stdout="", stderr="")

        with mock.patch.object(docker.subprocess, "run", fake_run):
            with self.assertLogs(docker.logger, level="ERROR") as logs:
                run_docker_command(list(BASE))
        self.assertEqual(len(logs.output), 1)  # just the "Command failed" line


class RetryBehaviour(unittest.TestCase):
    def test_retries_transient_then_succeeds(self):
        results = [
            _result(returncode=1, stderr="request returned 500 Internal Server Error"),
            _result(returncode=0, stdout="ok"),
        ]
        calls = {"n": 0}

        def fake_run(cmd, **kw):
            r = results[calls["n"]]
            calls["n"] += 1
            return r

        with mock.patch.object(docker.subprocess, "run", fake_run), \
             mock.patch.object(docker.time, "sleep", lambda *_a, **_k: None):
            ok = run_docker_command(list(BASE), retries=2, backoff=0)
        self.assertTrue(ok)
        self.assertEqual(calls["n"], 2)  # failed once, retried once, succeeded

    def test_does_not_retry_real_failure(self):
        calls = {"n": 0}

        def fake_run(cmd, **kw):
            calls["n"] += 1
            return _result(returncode=1, stderr="CUDA out of memory")

        with mock.patch.object(docker.subprocess, "run", fake_run), \
             mock.patch.object(docker.time, "sleep", lambda *_a, **_k: None):
            ok = run_docker_command(list(BASE), retries=3, backoff=0)
        self.assertFalse(ok)
        self.assertEqual(calls["n"], 1)  # no retries on a non-transient failure

    def test_gives_up_after_retries(self):
        calls = {"n": 0}

        def fake_run(cmd, **kw):
            calls["n"] += 1
            return _result(returncode=1, stderr="500 Internal Server Error")

        with mock.patch.object(docker.subprocess, "run", fake_run), \
             mock.patch.object(docker.time, "sleep", lambda *_a, **_k: None):
            ok = run_docker_command(list(BASE), retries=2, backoff=0)
        self.assertFalse(ok)
        self.assertEqual(calls["n"], 3)  # initial + 2 retries


if __name__ == "__main__":
    unittest.main(verbosity=2)
