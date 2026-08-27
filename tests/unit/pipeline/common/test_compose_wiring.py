#!/usr/bin/env python3
"""Compose wiring guards.

There is no docker-compose.yml at the repo root, so every invocation must pass
-f. These tests keep pipeline.common.docker.COMPOSE_FILES honest against the
compose files on disk, and keep those files' relative paths resolvable.

Worth having because none of this is exercised by import: a build context
pointing at a directory with no Dockerfile, or a volume pointing at a directory
that does not exist, fails only at `docker compose` time.
"""

import unittest
from pathlib import Path

import yaml

from pipeline.common.docker import COMPOSE_FILES, UnknownService, compose_run

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load(rel: str) -> dict:
    return yaml.safe_load((REPO_ROOT / rel).read_text(encoding="utf-8"))


class ComposeFileRegistry(unittest.TestCase):
    def test_every_registered_compose_file_exists(self):
        for service, rel in COMPOSE_FILES.items():
            self.assertTrue((REPO_ROOT / rel).is_file(),
                            f"{service}: missing compose file {rel}")

    def test_every_registered_service_is_defined_in_its_file(self):
        for service, rel in COMPOSE_FILES.items():
            services = _load(rel).get("services", {})
            self.assertIn(service, services,
                          f"{service} not defined in {rel}")

    def test_no_service_is_defined_in_two_compose_files(self):
        """Each service has exactly one definition; -f then picks it unambiguously."""
        seen = {}
        for rel in set(COMPOSE_FILES.values()):
            for name in _load(rel).get("services", {}):
                seen.setdefault(name, []).append(rel)
        dupes = {n: fs for n, fs in seen.items() if len(fs) > 1}
        self.assertEqual(dupes, {}, f"service(s) defined more than once: {dupes}")

    def test_every_defined_service_is_registered(self):
        for rel in set(COMPOSE_FILES.values()):
            for name in _load(rel).get("services", {}):
                self.assertIn(name, COMPOSE_FILES,
                              f"{name} is defined in {rel} but not registered")


class ComposeRunCommand(unittest.TestCase):
    def test_includes_the_right_file_and_service(self):
        cmd = compose_run("whisperx")
        self.assertIsInstance(cmd, list)
        self.assertIn("compose/whisperx/docker-compose.yml", cmd)
        self.assertEqual(cmd[-3:], ["run", "--rm", "whisperx"])

    def test_contains_rm_so_flags_can_be_inserted(self):
        """run_docker_command inserts -e/-v after the '--rm' element."""
        for service in COMPOSE_FILES:
            self.assertIn("--rm", compose_run(service))

    def test_unknown_service_fails_loud(self):
        with self.assertRaises(UnknownService):
            compose_run("nope")


class ComposePathsResolve(unittest.TestCase):
    """Relative paths in a compose file resolve against that FILE's directory."""

    def test_build_contexts_contain_their_dockerfile(self):
        for rel in set(COMPOSE_FILES.values()):
            base = (REPO_ROOT / rel).parent
            for name, svc in _load(rel)["services"].items():
                ctx = (base / svc["build"]["context"]).resolve()
                self.assertTrue(ctx.is_dir(), f"{name}: context {ctx} missing")
                dockerfile = ctx / svc["build"].get("dockerfile", "Dockerfile")
                self.assertTrue(dockerfile.is_file(),
                                f"{name}: {dockerfile} missing")

    def test_bind_mount_sources_exist(self):
        for rel in set(COMPOSE_FILES.values()):
            base = (REPO_ROOT / rel).parent
            for name, svc in _load(rel)["services"].items():
                for vol in svc.get("volumes", []):
                    host = vol.split(":")[0]
                    if not host.startswith("."):
                        continue  # named volume
                    self.assertTrue((base / host).resolve().exists(),
                                    f"{name}: bind source {host} missing")

    def test_entrypoint_script_exists_in_its_mounted_dir(self):
        for rel in set(COMPOSE_FILES.values()):
            base = (REPO_ROOT / rel).parent
            for name, svc in _load(rel)["services"].items():
                entrypoint = svc["entrypoint"][0]
                mounts = {v.split(":")[1]: v.split(":")[0]
                          for v in svc.get("volumes", []) if v.startswith(".")}
                match = next((c for c in mounts if entrypoint.startswith(c + "/")), None)
                self.assertIsNotNone(match, f"{name}: {entrypoint} is under no mount")
                script = (base / mounts[match] / entrypoint[len(match) + 1:]).resolve()
                self.assertTrue(script.is_file(), f"{name}: {script} missing")

    def test_no_fixed_container_names(self):
        """Fixed names make Docker refuse a second concurrent instance."""
        for rel in set(COMPOSE_FILES.values()):
            for name, svc in _load(rel)["services"].items():
                self.assertNotIn("container_name", svc,
                                 f"{name} pins container_name; blocks --parallel")

    def test_pythonpath_points_at_the_src_mount_not_the_scripts_mount(self):
        """`pipeline` lives in src/, not in the scripts mount — don't conflate them."""
        for rel in set(COMPOSE_FILES.values()):
            base = (REPO_ROOT / rel).parent
            for name, svc in _load(rel)["services"].items():
                pythonpath = svc["environment"]["PYTHONPATH"]
                host = next(v.split(":")[0] for v in svc["volumes"]
                            if v.split(":")[1] == pythonpath)
                self.assertTrue((base / host / "pipeline").is_dir(),
                                f"{name}: PYTHONPATH {pythonpath} has no pipeline/")


class EnvFileWiring(unittest.TestCase):
    """Compose loads .env from the project dir = the compose file's directory.

    Since the compose files live in compose/<svc>/, the repo-root .env is
    invisible without an explicit --env-file, and ${HF_TOKEN:-} silently
    resolves to empty. HF_TOKEN is deliberately not passed as -e (see
    pipeline.transcribe.docker_env), so this is its only route in.
    """

    def test_env_file_flag_added_when_dotenv_exists(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("HF_TOKEN=abc\n")
            cmd = compose_run("whisperx", root)
            self.assertIn("--env-file", cmd)
            self.assertIn(str(root / ".env"), cmd)
            self.assertEqual(cmd[-3:], ["run", "--rm", "whisperx"])

    def test_no_env_file_flag_when_absent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotIn("--env-file", compose_run("whisperx", Path(tmp)))

    def test_env_file_precedes_the_subcommand(self):
        """--env-file is a top-level flag; after `run` docker rejects it."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("HF_TOKEN=abc\n")
            cmd = compose_run("whisperx", root)
            self.assertLess(cmd.index("--env-file"), cmd.index("run"))
            self.assertLess(cmd.index("-f"), cmd.index("run"))

    def test_repo_dotenv_is_actually_found(self):
        """The real repo root has a .env (gitignored) — flag it if present."""
        cmd = compose_run("whisperx", REPO_ROOT)
        if (REPO_ROOT / ".env").is_file():
            self.assertIn("--env-file", cmd)
        else:
            self.skipTest("no .env in this checkout")


if __name__ == "__main__":
    unittest.main()
