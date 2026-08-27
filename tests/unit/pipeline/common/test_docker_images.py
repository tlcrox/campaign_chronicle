#!/usr/bin/env python3
"""Guards on the two container images.

These are static checks on Dockerfiles and requirements manifests — cheap, and
they cover three failures that all shipped silently because nothing imports a
Dockerfile:

1. `docker/pyscenedetect/requirements-docker.txt` was a UTF-16LE `pip freeze`
   dump. pip strips the BOM and installs it happily, so it failed at runtime
   rather than at build time.
2. That dump was a freeze of a *WhisperX* environment (whisperx, pyannote,
   torch==2.8.0) being installed into the scene-detection image.
3. `docker/whisperx/Dockerfile` pinned torch==2.5.1 while whisperx 3.8.6
   requires torch~=2.8.0, so the requirements step silently upgraded torch from
   PyPI and discarded the CUDA build the pin existed to create.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
WHISPERX = REPO_ROOT / "docker" / "whisperx"
PYSCENEDETECT = REPO_ROOT / "docker" / "pyscenedetect"


class ManifestsAreUtf8(unittest.TestCase):
    """A UTF-16 requirements file installs fine and breaks later. Catch it here."""

    def test_all_requirements_files_decode_as_utf8(self):
        for req in REPO_ROOT.glob("docker/*/requirements-docker.txt"):
            raw = req.read_bytes()
            self.assertNotEqual(raw[:2], b"\xff\xfe", f"{req} is UTF-16LE")
            self.assertNotEqual(raw[:3], b"\xef\xbb\xbf", f"{req} has a UTF-8 BOM")
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as e:
                self.fail(f"{req} is not valid UTF-8: {e}")

    def test_no_manifest_duplicates_under_compose(self):
        """Build contexts are docker/*, so compose/* must not carry manifests."""
        strays = list(REPO_ROOT.glob("compose/*/requirements-docker.txt"))
        self.assertEqual(strays, [], f"stale duplicate manifests: {strays}")


class PySceneDetectImageIsCpuOnly(unittest.TestCase):
    def setUp(self):
        self.dockerfile = (PYSCENEDETECT / "Dockerfile").read_text(encoding="utf-8")
        self.reqs = (PYSCENEDETECT / "requirements-docker.txt").read_text(encoding="utf-8")

    def test_no_torch_install(self):
        installs = [ln for ln in self.dockerfile.splitlines()
                    if re.search(r"^\s*torch(audio|vision)?==", ln)]
        self.assertEqual(installs, [], "pyscenedetect installs torch; it uses OpenCV")

    def test_requirements_have_no_ml_stack(self):
        for pkg in ("torch", "whisperx", "pyannote", "speechbrain"):
            self.assertNotRegex(
                self.reqs, rf"(?mi)^\s*{pkg}",
                f"{pkg} does not belong in the scene-detection image",
            )

    def test_requirements_have_what_the_scripts_need(self):
        self.assertRegex(self.reqs, r"(?mi)^\s*scenedetect")

    def test_base_image_is_not_cuda(self):
        base = re.search(r"(?m)^FROM\s+(\S+)", self.dockerfile).group(1)
        self.assertNotIn("cuda", base.lower(),
                         f"CPU-only image on a CUDA base: {base}")


class WhisperXImagePinsMatch(unittest.TestCase):
    def setUp(self):
        self.dockerfile = (WHISPERX / "Dockerfile").read_text(encoding="utf-8")
        self.reqs = (WHISPERX / "requirements-docker.txt").read_text(encoding="utf-8")

    def test_torch_is_pinned_from_a_cuda_index(self):
        self.assertRegex(self.dockerfile, r"(?m)^\s*torch==\d+\.\d+\.\d+")
        self.assertRegex(self.dockerfile, r"--index-url\s+https://download\.pytorch\.org/whl/cu\d+")

    def test_whisperx_is_pinned(self):
        """Unpinned, it tracks releases and drifts off the Dockerfile's torch pin."""
        self.assertRegex(self.reqs, r"(?mi)^\s*whisperx==\d+\.\d+\.\d+")

    def test_torch_is_not_also_listed_in_requirements(self):
        """Listing it twice lets the second install override the CUDA wheels."""
        self.assertNotRegex(self.reqs, r"(?mi)^\s*torch(audio|vision)?[=<>]")

    def test_scenedetect_moved_out(self):
        """Scene detection has its own image since the container split."""
        self.assertNotRegex(self.reqs, r"(?mi)^\s*scenedetect")


class EntrypointsGetEverythingTheyDemand(unittest.TestCase):
    """Every variable a container refuses to start without is one a stage sends.

    The entrypoints no longer invent a value for anything config.yaml owns — an
    unset SCENE_THRESHOLD exits naming the key instead of quietly detecting at a
    different sensitivity. That is only an improvement while the host actually
    supplies them, so this walks the `${VAR:?...}` list straight out of each
    script and checks it against what the stage builds.

    It fails if someone adds a required variable the stage does not send, or
    stops sending one a script requires.
    """

    def _required(self, script: Path) -> set:
        """The variables a script refuses to run without."""
        return set(re.findall(r"\$\{([A-Z_]+):\?", script.read_text(encoding="utf-8")))

    def setUp(self):
        from pipeline.config import get_config
        self.cfg = get_config()

    def test_whisperx_entrypoint(self):
        from pipeline.transcribe.docker_env import whisper_env
        required = self._required(WHISPERX / "entrypoint.sh")
        self.assertTrue(required, "entrypoint.sh requires nothing; did :? get lost?")
        self.assertEqual(required - set(whisper_env(self.cfg)), set())

    def test_single_roi_script(self):
        from cc_stages.detect_scenes_single_roi import set_env
        required = self._required(PYSCENEDETECT / "detect_scenes.sh")
        self.assertTrue(required)
        provided = set(set_env("400 445 925 745", self.cfg))
        self.assertEqual(required - provided, set())

    def test_multi_roi_script(self):
        from cc_stages.detect_scenes_multi_roi import set_env
        required = self._required(PYSCENEDETECT / "detect_scenes_multi.sh")
        self.assertTrue(required)
        # ROI_FILE is added by the caller rather than set_env: it is a per-run
        # filename, and the caller has already checked the file exists.
        provided = set(set_env(self.cfg)) | {"ROI_FILE"}
        self.assertEqual(required - provided, set())

    def test_the_caller_really_does_send_roi_file(self):
        source = (REPO_ROOT / "scripts" / "cc_stages"
                  / "detect_scenes_multi_roi.py").read_text(encoding="utf-8")
        self.assertIn('env_vars["ROI_FILE"]', source)

    def test_optional_variables_are_not_demanded(self):
        """Empty is meaningful for these; :? would reject it."""
        scenes = self._required(PYSCENEDETECT / "detect_scenes.sh")
        whisper = self._required(WHISPERX / "entrypoint.sh")
        self.assertNotIn("SCENE_ROI", scenes)          # empty = full frame
        self.assertNotIn("WHISPER_LANGUAGE", whisper)  # empty = auto-detect
        self.assertNotIn("WHISPER_HOTWORDS", whisper)  # empty = flag omitted
        self.assertNotIn("WHISPER_INITIAL_PROMPT", whisper)


class ComposeCarriesNoConfigValues(unittest.TestCase):
    """config.yaml owns these; a compose default would be a second copy.

    The keys stay listed (name only) so a bare `SCENE_THRESHOLD=10 docker
    compose run` still reaches the container — compose does not forward a host
    variable that is not named.
    """

    COMPOSE = (
        ("compose/pyscenedetect/docker-compose.yml", "SCENE_"),
        ("compose/whisperx/docker-compose.yml", "WHISPER_"),
    )

    def test_no_config_owned_variable_carries_a_default(self):
        for rel, prefix in self.COMPOSE:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped.startswith(prefix):
                    continue
                with self.subTest(compose=rel, line=stripped):
                    self.assertTrue(
                        stripped.endswith(":"),
                        f"{stripped} carries a value; config.yaml owns it")

    def test_the_keys_are_still_listed_for_pass_through(self):
        text = (REPO_ROOT / "compose/pyscenedetect/docker-compose.yml").read_text(
            encoding="utf-8")
        for key in ("SCENE_THRESHOLD", "SCENE_MIN_LEN", "SCENE_NUM_IMAGES",
                    "SCENE_IMAGE_FORMAT", "SCENE_ROI", "ROI_FILE"):
            self.assertIn(f"{key}:", text)


class EntrypointLanguageIsOptional(unittest.TestCase):
    """An empty language means auto-detect, which needs the FLAG omitted.

    `--language ""` is an explicit empty argument, not an absent flag. The host
    already produces "" for an unset language (docker_env.whisper_env), so the
    only thing standing between that and a bad whisperx invocation is the shell.
    """

    def setUp(self):
        self.script = (WHISPERX / "entrypoint.sh").read_text(encoding="utf-8")

    def test_language_is_not_in_the_unconditional_args(self):
        args_block = self.script.split("args=(", 1)[1].split(")", 1)[0]
        self.assertNotIn("--language", args_block)

    def test_language_is_added_only_when_non_empty(self):
        self.assertIn('if [ -n "$LANGUAGE" ]', self.script)
        self.assertIn("args+=(--language", self.script)

    def test_the_shell_does_not_substitute_its_own_language(self):
        """A `:-en` here would override the config's deliberate empty."""
        self.assertIn('LANGUAGE="${WHISPER_LANGUAGE:-}"', self.script)


class ContainerBooleanSwitches(unittest.TestCase):
    """The multi-ROI script's own switches, which no stage sets.

    The SPLIT_ONLY guard was `= "false"`, so SPLIT_ONLY=0 read as "not the
    string false" and silently disabled scene detection — the inverse of what
    anyone typing it would mean.
    """

    def setUp(self):
        self.script = (PYSCENEDETECT / "detect_scenes_multi.sh").read_text(encoding="utf-8")

    def test_debug_is_off_unless_asked_for(self):
        self.assertIn('DEBUG="${DEBUG:-false}"', self.script)

    def test_split_only_is_off_by_default(self):
        self.assertIn('SPLIT_ONLY="${SPLIT_ONLY:-false}"', self.script)

    def test_switches_go_through_one_truthiness_test(self):
        self.assertIn("is_true()", self.script)
        for spelling in ("true", "1", "yes", "on"):
            self.assertIn(spelling, self.script.split("is_true()", 1)[1][:200])

    def test_no_switch_is_compared_as_a_bare_string(self):
        for switch in ("$DEBUG", "$SPLIT_ONLY"):
            for comparison in (f'[ "{switch}" = "true" ]', f'[ "{switch}" = "false" ]'):
                self.assertNotIn(comparison, self.script,
                                 f"{switch} is string-compared; use is_true")


if __name__ == "__main__":
    unittest.main()
