#!/usr/bin/env python3
"""
Tests for config.Config property access and path resolution.

Deterministic: each test builds its own throwaway config.yaml in a tempdir and
constructs Config(config_path=...) directly, so nothing depends on the repo's
own config or on the filesystem layout around it.
"""

import io
import os
import tempfile
import unittest
from pathlib import Path


from pipeline.config import Config, find_repo_root

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    """Config properties: values from YAML, defaults, and overrides."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._saved = {k: os.environ.pop(k) for k in ("DEFAULT_SOURCE_DIR",)
                       if k in os.environ}

    def tearDown(self):
        os.environ.update(self._saved)
        self.tmp.cleanup()

    def _write_yaml(self, content: str) -> Path:
        """Write YAML config and return path."""
        p = self.root / "config.yaml"
        p.write_text(content)
        return p

    def _full_config(self) -> str:
        """Return complete YAML configuration."""
        lines = [
            "whisper:",
            "  model: large-v3-turbo",
            "  language: en",
            "  compute_type: float16",
            "  batch_size: 8",
            "  output_format: json",
            "  diarize: true",
            "",
            "scenes:",
            "  threshold: 5.0",
            "  min_length: 1s",
            "  num_images: 1",
            "  image_format: jpg",
            '  roi: "400 445 925 745"',
            "  roi_file: roi_history.json",
            "  manual_source: screens",
            "  manual_csv_name: Scenes.csv",
            "",
            "orchestration:",
            "  source_dir: data",
            "  session_dirs:",
            "    - Week1",
            "    - Week2",
            "  parallel_workers: 2",
            "",
            "merge:",
            "  auto_merge: true",
            "  csv:",
            '    scene_number_column: "Scene Number"',
            '    start_time_column: "Start Time (seconds)"',
            '    end_time_column: "End Time (seconds)"',
            "  images:",
            "    width: 6.0",
            "  document:",
            '    title: "Session Storyboard"',
            "",
            "speakers:",
            "  config_file: speaker_config.json",
        ]
        return "\n".join(lines)

    def test_whisper_properties(self):
        p = self._write_yaml(self._full_config())
        c = Config(config_path=p)
        self.assertEqual(c.whisper_model, "large-v3-turbo")
        self.assertEqual(c.whisper_language, "en")
        self.assertEqual(c.whisper_compute_type, "float16")
        self.assertEqual(c.whisper_batch_size, 8)
        self.assertEqual(c.whisper_output_format, "json")
        self.assertTrue(c.whisper_diarize)

    def test_whisper_defaults(self):
        yaml_lines = [
            "whisper: {}",
            "scenes: {}",
            "speakers:",
            "  config_file: speaker_config.json",
            "orchestration:",
            "  source_dir: data",
        ]
        yaml_text = "\n".join(yaml_lines)
        p = self._write_yaml(yaml_text)
        c = Config(config_path=p)
        self.assertEqual(c.whisper_model, "large-v3")
        self.assertEqual(c.whisper_batch_size, 16)
        self.assertFalse(c.whisper_diarize)

    def test_scene_properties(self):
        p = self._write_yaml(self._full_config())
        c = Config(config_path=p)
        self.assertEqual(c.scene_threshold, 5.0)
        self.assertEqual(c.scene_min_length, "1s")
        self.assertEqual(c.scene_num_images, 1)
        self.assertEqual(c.scene_image_format, "jpg")
        self.assertEqual(c.scene_roi, "400 445 925 745")
        self.assertEqual(c.scene_roi_file, "roi_history.json")
        self.assertEqual(c.scene_manual_source, "screens")
        self.assertEqual(c.scene_manual_csv_name, "Scenes.csv")

    def test_orchestration_properties(self):
        p = self._write_yaml(self._full_config())
        # The session directories are deliberately NOT created. Reading config
        # must not depend on the filesystem; existence is find_sessions' job.
        c = Config(config_path=p)
        self.assertTrue(c.source_dir.is_absolute())
        self.assertEqual(c.source_dir, self.root / "data")
        self.assertEqual(len(c.session_dirs), 2)
        self.assertEqual(c.parallel_workers, 2)

    def test_session_dirs_does_no_filesystem_io(self):
        """Non-existent session dirs are still returned, fully resolved."""
        p = self._write_yaml(self._full_config())
        c = Config(config_path=p)
        self.assertEqual(
            c.session_dirs,
            [self.root / "data" / "Week1", self.root / "data" / "Week2"],
        )
        self.assertFalse(any(d.exists() for d in c.session_dirs))

    def test_session_dirs_config_is_raw(self):
        """session_dirs_config returns the values verbatim, unresolved."""
        p = self._write_yaml(self._full_config())
        c = Config(config_path=p)
        self.assertEqual(c.session_dirs_config, [Path("Week1"), Path("Week2")])
        self.assertFalse(any(d.is_absolute() for d in c.session_dirs_config))

    def test_session_dirs_absolute_entries_pass_through(self):
        """Absolute entries are used as-is; relative ones join source_dir."""
        # Built from self.root so the "absolute" entry is genuinely absolute on
        # Windows too (a bare "/abs/x" has no drive and is NOT absolute there).
        abs_entry = self.root / "elsewhere" / "Session9"
        yaml_text = "\n".join([
            "whisper: {}",
            "scenes: {}",
            "orchestration:",
            "  source_dir: data",
            "  session_dirs:",
            f"    - {abs_entry.as_posix()}",
            "    - Relative9",
        ])
        c = Config(config_path=self._write_yaml(yaml_text))
        self.assertEqual(
            c.session_dirs,
            [Path(abs_entry.as_posix()), self.root / "data" / "Relative9"],
        )

    def test_merge_properties(self):
        p = self._write_yaml(self._full_config())
        c = Config(config_path=p)
        self.assertTrue(c.auto_merge)
        self.assertEqual(c.scene_number_column, "Scene Number")
        self.assertEqual(c.start_time_column, "Start Time (seconds)")
        self.assertEqual(c.end_time_column, "End Time (seconds)")
        self.assertEqual(c.image_width_inches, 6.0)
        self.assertEqual(c.document_title, "Session Storyboard")

    def test_speaker_config_file(self):
        p = self._write_yaml(self._full_config())
        c = Config(config_path=p)
        expected = self.root / "data" / "speaker_config.json"
        self.assertEqual(c.speaker_config_file, expected)

    def test_hotwords_file_default_path(self):
        p = self._write_yaml(self._full_config())
        c = Config(config_path=p)
        expected = self.root / "data" / "whisperx_hotwords.txt"
        self.assertEqual(c.whisper_hotwords_file, expected)

    def test_hotwords_reads_and_normalises_file(self):
        p = self._write_yaml(self._full_config())
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        # Multi-line and extra whitespace collapse; quotes are left alone, since
        # run_docker_command passes env values as argv rather than shell text.
        (self.root / "data" / "whisperx_hotwords.txt").write_text(
            'Characters:\n  Mechanon,   Aberrancy\n"OCV" DCV\n', encoding="utf-8"
        )
        c = Config(config_path=p)
        self.assertEqual(c.whisper_hotwords,
                         'Characters: Mechanon, Aberrancy "OCV" DCV')

    def test_hotwords_missing_file_is_empty(self):
        p = self._write_yaml(self._full_config())
        c = Config(config_path=p)  # data/whisperx_hotwords.txt does not exist
        self.assertEqual(c.whisper_hotwords, "")

    def test_hotwords_empty_when_source_dir_unset(self):
        yaml_text = "\n".join(["whisper: {}", "orchestration:", "  parallel_workers: 1"])
        p = self._write_yaml(yaml_text)
        c = Config(config_path=p)
        self.assertEqual(c.whisper_hotwords, "")

    def test_source_dir_resolves_relative(self):
        p = self._write_yaml(self._full_config())
        c = Config(config_path=p)
        self.assertTrue(c.source_dir.is_absolute())
        self.assertEqual(c.source_dir, self.root / "data")

    def test_source_dir_absolute_passthrough(self):
        abs_path = (self.root / "elsewhere").as_posix()
        yaml_text = self._full_config().replace(
            'source_dir: data',
            f'source_dir: {abs_path}'
        )
        p = self._write_yaml(yaml_text)
        c = Config(config_path=p)
        self.assertEqual(c.source_dir, Path(abs_path))

    def test_source_dir_defaults_to_config_dir(self):
        # With no orchestration.source_dir, source_dir defaults to the config
        # file's own directory (config.yaml lives at the source root).
        yaml_lines = [
            "orchestration:",
            "  parallel_workers: 1",
        ]
        yaml_text = "\n".join(yaml_lines)
        p = self._write_yaml(yaml_text)
        c = Config(config_path=p)
        self.assertEqual(c.source_dir, self.root)

    def test_find_config_upward_beside_the_session(self):
        """The flat layout: config.yaml at the source root, sessions under it."""
        from pipeline.config import find_config_upward
        self._write_yaml(self._full_config())
        session = self.root / "SingleVideo"
        session.mkdir(parents=True, exist_ok=True)
        self.assertEqual(find_config_upward(session), self.root / "config.yaml")

    def test_find_config_upward_two_levels(self):
        """The nested layout, "Weeks/Week 13" — exactly the convention's limit."""
        from pipeline.config import find_config_upward
        self._write_yaml(self._full_config())
        session = self.root / "Weeks" / "Week 14"
        session.mkdir(parents=True, exist_ok=True)
        self.assertEqual(find_config_upward(session), self.root / "config.yaml")

    def test_find_config_upward_stops_at_the_convention(self):
        """Three levels is past the cap: an unbounded walk climbs out of the
        source tree and adopts a config.yaml belonging to something else."""
        from pipeline.config import find_config_upward
        self._write_yaml(self._full_config())
        too_deep = self.root / "a" / "b" / "c"
        too_deep.mkdir(parents=True, exist_ok=True)
        self.assertIsNone(find_config_upward(too_deep))

    def test_find_config_upward_finds_nothing_when_there_is_nothing(self):
        from pipeline.config import find_config_upward
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            deep = Path(d).resolve()
            if any((x / "config.yaml").is_file() for x in [deep, *deep.parents]):
                self.skipTest("a stray config.yaml sits above the temp dir")
            self.assertIsNone(find_config_upward(deep))

    def test_defaults_when_sections_empty(self):
        yaml_lines = [
            "whisper: {}",
            "scenes: {}",
            "orchestration:",
            "  source_dir: data",
            "speakers:",
            "  config_file: speaker_config.json",
        ]
        yaml_text = "\n".join(yaml_lines)
        p = self._write_yaml(yaml_text)
        c = Config(config_path=p)
        self.assertEqual(c.scene_threshold, 5.0)
        self.assertEqual(c.parallel_workers, 1)
        self.assertTrue(c.auto_merge)


class FindRepoRoot(unittest.TestCase):
    """Walk up to pyproject.toml, never a hardcoded depth.

    config.py has already moved once (scripts/ -> src/pipeline/), which is
    exactly what silently invalidates a parents[N] index.
    """

    def test_finds_this_repo(self):
        self.assertEqual(find_repo_root(), REPO_ROOT)
        self.assertTrue((find_repo_root() / "pyproject.toml").is_file())

    def test_walks_past_intermediate_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            deep = root / "a" / "b" / "c"
            deep.mkdir(parents=True)
            self.assertEqual(find_repo_root(deep / "mod.py"), root)

    def test_none_when_there_is_no_repo_above(self):
        """The container case: only src/ is mounted, so no pyproject exists."""
        with tempfile.TemporaryDirectory() as tmp:
            deep = Path(tmp).resolve() / "x" / "y"
            deep.mkdir(parents=True)
            if any((d / "pyproject.toml").is_file() for d in deep.parents):
                self.skipTest("a stray pyproject.toml sits above the temp dir")
            self.assertIsNone(find_repo_root(deep / "mod.py"))

    def test_required_raises_naming_where_it_looked(self):
        with tempfile.TemporaryDirectory() as tmp:
            deep = Path(tmp).resolve() / "x"
            deep.mkdir()
            if any((d / "pyproject.toml").is_file() for d in deep.parents):
                self.skipTest("a stray pyproject.toml sits above the temp dir")
            with self.assertRaises(RuntimeError) as ctx:
                find_repo_root(deep / "mod.py", required=True)
            self.assertIn(str(deep), str(ctx.exception))

    def test_only_one_other_module_implements_the_walk(self):
        """verify_setup keeps a copy on purpose; nothing else may.

        That script's most useful check is whether `pipeline` imports at all, so
        it cannot import from the package to find the root — the tool that
        verifies the install cannot depend on the install.
        """
        needle = 'pyproject.toml").is_file()'
        implementers = set()
        for folder in ("src", "scripts", "tests"):
            for path in (REPO_ROOT / folder).rglob("*.py"):
                if "__pycache__" in path.parts or path == Path(__file__).resolve():
                    continue
                if needle in path.read_text(encoding="utf-8"):
                    implementers.add(path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(
            implementers,
            {"src/pipeline/config.py", "scripts/verify_setup.py"},
            "use pipeline.config.find_repo_root instead of walking again")

    def test_config_derives_no_root_by_index(self):
        source = (REPO_ROOT / "src" / "pipeline" / "config.py").read_text(encoding="utf-8")
        # ".parents[" — the subscript. The bare word appears in the prose
        # explaining why not to use one.
        self.assertNotIn(".parents[", source,
                         "a hardcoded depth breaks silently when the file moves")


class RepoRootWithoutARepo(unittest.TestCase):
    """Inside the containers there is no repo root, and that is not an error."""

    def _containerish(self) -> Config:
        cfg = Config(config_path=REPO_ROOT / "config" / "config.yaml")
        cfg._repo_root = None
        return cfg

    def test_the_property_refuses_and_explains(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._containerish().repo_root
        message = str(ctx.exception)
        self.assertIn("pyproject.toml", message)
        self.assertIn("containers", message)

    def test_validation_does_not_warn_about_an_impossible_dotenv(self):
        """The probe cannot look for .env when it does not know where to look."""
        cfg = self._containerish()
        cfg._config["whisper"]["diarize"] = True
        import os
        saved = os.environ.pop("HF_TOKEN", None)
        try:
            with self.assertNoLogs("pipeline.config", level="WARNING"):
                cfg._validate(Path("config.yaml"))
        finally:
            if saved is not None:
                os.environ["HF_TOKEN"] = saved


class RepoRootIsPublic(unittest.TestCase):
    """The cwd for `docker compose` is part of the contract, not an internal.

    Three stages need it to run a container at all: it is what makes the
    `-f compose/.../docker-compose.yml` relative path resolve, and what
    compose_run uses to find the .env carrying HF_TOKEN.
    """

    def test_it_is_the_directory_holding_pyproject(self):
        cfg = Config(config_path=REPO_ROOT / "config" / "config.yaml")
        self.assertTrue((cfg.repo_root / "pyproject.toml").is_file())

    def test_it_is_not_the_config_files_own_directory(self):
        """Anchored to the code, so an override config beside the source data
        does not move where `docker compose` runs."""
        cfg = Config(config_path=REPO_ROOT / "tests" / "test_source" / "config.yaml")
        self.assertTrue((cfg.repo_root / "pyproject.toml").is_file())
        self.assertNotEqual(cfg.repo_root, cfg._config_dir)

    def test_nothing_outside_config_reads_the_private_attribute(self):
        for folder in ("src", "scripts", "tests"):
            for path in (REPO_ROOT / folder).rglob("*.py"):
                # config.py owns the attribute; this file names it in the
                # assertion below and would otherwise flag itself.
                if (path.name == "config.py" or path == Path(__file__).resolve()
                        or "__pycache__" in path.parts):
                    continue
                with self.subTest(module=str(path.relative_to(REPO_ROOT))):
                    # "._repo_root", not the bare name: both integration
                    # drivers define a _repo_root() helper of their own.
                    self.assertNotIn("._repo_root", path.read_text(encoding="utf-8"),
                                     "use the public Config.repo_root")


class ConfigIsCompleteAfterLoad(unittest.TestCase):
    """config.yaml is the master source, so loading it fills in what it omits.

    Every one of these used to raise from the constructor — killing every entry
    point, not just the one that wanted the value — because the env-override
    pass indexed sections the file had not declared.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._saved = {k: os.environ.pop(k) for k in
                       ("DEFAULT_SOURCE_DIR", "PYANNOTE_METRICS_ENABLED",
                        "WHISPER_MODEL", "SCENE_THRESHOLD")
                       if k in os.environ}

    def tearDown(self):
        for key in ("PYANNOTE_METRICS_ENABLED", "WHISPER_MODEL", "SCENE_THRESHOLD"):
            os.environ.pop(key, None)
        os.environ.update(self._saved)
        self.tmp.cleanup()

    def _cfg(self, body: str) -> Config:
        p = self.root / "config.yaml"
        io.open(p, "w", encoding="utf-8", newline="\n").write(body)
        return Config(config_path=p)

    def test_empty_file_yields_a_working_config(self):
        """safe_load returns None for an empty file, not an empty dict."""
        c = self._cfg("")
        self.assertEqual(c.whisper_model, "large-v3")
        self.assertEqual(c.scene_threshold, 5.0)
        self.assertEqual(c.scene_image_format, "jpg")
        self.assertEqual(c.parallel_workers, 1)
        self.assertEqual(c.document_title, "Transcript")
        self.assertTrue(c.auto_merge)

    def test_empty_section_yields_defaults(self):
        """A section header with nothing under it also loads as None."""
        c = self._cfg("whisper:\n\nscenes:\n\nspeakers:\n")
        self.assertEqual(c.whisper_model, "large-v3")
        self.assertEqual(c.scene_num_images, 1)
        self.assertEqual(c.speaker_config_file.name, "speaker_config.json")

    def test_missing_section_yields_defaults(self):
        c = self._cfg("scenes:\n  threshold: 12.0\n")
        self.assertEqual(c.scene_threshold, 12.0)      # from the file
        self.assertEqual(c.whisper_batch_size, 16)     # section absent entirely
        self.assertEqual(c.video_column, "Video")

    def test_file_values_win_over_defaults(self):
        c = self._cfg("whisper:\n  model: tiny\n  batch_size: 4\n")
        self.assertEqual(c.whisper_model, "tiny")
        self.assertEqual(c.whisper_batch_size, 4)
        self.assertEqual(c.whisper_language, "en")     # sibling still defaulted

    def test_nested_sections_merge_key_by_key(self):
        c = self._cfg("merge:\n  csv:\n    video_column: Clip\n")
        self.assertEqual(c.video_column, "Clip")
        self.assertEqual(c.scene_number_column, "Scene Number")
        self.assertEqual(c.image_width_inches, 6.0)

    def test_keys_defaults_does_not_know_are_kept(self):
        """remove_hallucination's thresholds arrive this way."""
        c = self._cfg("whisper:\n  clean:\n    no_speech_max: 0.4\n")
        clean = c.get("whisper", "clean")
        self.assertEqual(clean["no_speech_max"], 0.4)
        self.assertFalse(clean["enabled"])             # default alongside it

    def test_env_override_against_an_undeclared_section(self):
        """Setting an override for a section the file omits used to be fatal."""
        os.environ["PYANNOTE_METRICS_ENABLED"] = "true"
        c = self._cfg("scenes:\n  image_format: png\n")
        self.assertTrue(c.get("speakers", "pyannote")["metrics_enabled"])
        self.assertEqual(c.scene_image_format, "png")

    def test_env_override_reaches_an_empty_file(self):
        os.environ["WHISPER_MODEL"] = "medium"
        self.assertEqual(self._cfg("").whisper_model, "medium")

    def test_a_scalar_where_a_section_belongs_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._cfg("whisper: large-v3\n")
        self.assertIn("whisper", str(ctx.exception))
        self.assertIn("mapping", str(ctx.exception))


class ConfigValidation(unittest.TestCase):
    """The completed config is verified to be runnable before anything uses it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _load(self, body: str) -> Config:
        p = self.root / "config.yaml"
        io.open(p, "w", encoding="utf-8", newline="\n").write(body)
        return Config(config_path=p)

    def _error(self, body: str) -> str:
        with self.assertRaises(ValueError) as ctx:
            self._load(body)
        return str(ctx.exception)

    def test_cpu_with_float16_is_rejected_with_the_reason(self):
        message = self._error("whisper:\n  device: cpu\n  compute_type: float16\n")
        self.assertIn("CPU", message)
        self.assertIn("int8", message)

    def test_cpu_with_int8_is_fine(self):
        c = self._load("whisper:\n  device: cpu\n  compute_type: int8\n")
        self.assertEqual(c.whisper_device, "cpu")

    def test_unknown_device_is_rejected(self):
        self.assertIn("whisper.device", self._error("whisper:\n  device: tpu\n"))

    def test_unknown_compute_type_is_rejected(self):
        self.assertIn("compute_type", self._error("whisper:\n  compute_type: fp16\n"))

    def test_unknown_output_format_is_rejected(self):
        self.assertIn("output_format", self._error("whisper:\n  output_format: docx\n"))

    def test_webp_is_rejected_with_its_own_explanation(self):
        message = self._error("scenes:\n  image_format: webp\n")
        self.assertIn("python-docx", message)

    def test_non_positive_numbers_are_rejected(self):
        self.assertIn("scenes.threshold", self._error("scenes:\n  threshold: 0\n"))
        self.assertIn("scenes.num_images", self._error("scenes:\n  num_images: 0\n"))
        self.assertIn("whisper.batch_size", self._error("whisper:\n  batch_size: 0\n"))
        self.assertIn("parallel_workers",
                      self._error("orchestration:\n  parallel_workers: 0\n"))
        self.assertIn("merge.images.width",
                      self._error("merge:\n  images:\n    width: 0\n"))

    def test_non_numeric_where_a_number_belongs_is_rejected(self):
        self.assertIn("must be a number",
                      self._error("scenes:\n  threshold: high\n"))

    def test_empty_min_length_is_rejected(self):
        self.assertIn("min_length", self._error('scenes:\n  min_length: ""\n'))

    def test_roi_must_be_four_integers(self):
        self.assertIn("scenes.roi", self._error('scenes:\n  roi: "400 445 925"\n'))
        self.assertIn("scenes.roi", self._error('scenes:\n  roi: "a b c d"\n'))
        # four integers, and an unset roi, are both fine
        self._load('scenes:\n  roi: "400 445 925 745"\n')
        self._load("scenes:\n  threshold: 5.0\n")

    def test_every_problem_is_reported_at_once(self):
        """Fixing a config one error per run is its own kind of misery."""
        message = self._error(
            "whisper:\n  device: cpu\n  compute_type: float16\n  batch_size: 0\n"
            "scenes:\n  image_format: webp\n  num_images: -1\n"
        )
        for expected in ("CPU", "batch_size", "python-docx", "num_images"):
            self.assertIn(expected, message)

    def test_a_malformed_env_override_is_reported(self):
        """It used to be `except ValueError: pass` — silently not applied."""
        os.environ["WHISPER_BATCH_SIZE"] = "sixteen"
        try:
            message = self._error("whisper:\n  model: tiny\n")
        finally:
            os.environ.pop("WHISPER_BATCH_SIZE", None)
        self.assertIn("batch_size", message)
        self.assertIn("sixteen", message)

    def test_a_valid_env_override_still_applies(self):
        os.environ["SCENE_THRESHOLD"] = "12.5"
        try:
            self.assertEqual(self._load("scenes:\n  threshold: 5.0\n").scene_threshold,
                             12.5)
        finally:
            os.environ.pop("SCENE_THRESHOLD", None)

    def test_the_message_names_the_file(self):
        message = self._error("scenes:\n  threshold: -1\n")
        self.assertIn(str(self.root / "config.yaml"), message)

    def test_the_shipped_configs_are_valid(self):
        """The three configs in the repo must survive their own validator."""
        repo = Path(__file__).resolve().parents[2]
        for rel in ("config/config.yaml", "config/config_sample.yaml",
                    "tests/test_source/config.yaml"):
            path = repo / rel
            if not path.is_file():
                continue
            with self.subTest(config=rel):
                Config(config_path=path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
