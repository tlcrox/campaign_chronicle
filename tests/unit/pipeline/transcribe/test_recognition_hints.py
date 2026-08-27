#!/usr/bin/env python3
"""The two recognition-hint files, and the flags they feed.

These are two different WhisperX features and must stay wired to two different
files — crossing them silently sends prompt text to --hotwords and never passes
--initial_prompt at all.

    --hotwords        biases recognition for the WHOLE run   (proper nouns, jargon)
    --initial_prompt  seeds only the FIRST window            (spelling, style)

Both files are configurable and source-dir-relative; a missing one means that
flag is simply omitted.
"""

import tempfile
import unittest
from pathlib import Path

from pipeline.config import Config
from pipeline.transcribe.docker_env import whisper_env


def _write_config(root: Path, extra: str = "") -> Path:
    (root / "config").mkdir(exist_ok=True)
    p = root / "config" / "config.yaml"
    p.write_text(
        "whisper:\n"
        "  model: tiny\n"
        "  diarize: false\n"
        f"{extra}"
        "scenes: {}\n"
        "orchestration:\n"
        f"  source_dir: {(root / 'src').as_posix()}\n",
        encoding="utf-8")
    return p


class HintFilesAreSeparate(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.src = self.root / "src"
        self.src.mkdir()

    def _cfg(self, extra=""):
        return Config(config_path=_write_config(self.root, extra))

    def test_defaults_point_at_distinct_files(self):
        cfg = self._cfg()
        self.assertEqual(cfg.whisper_hotwords_file.name, "whisperx_hotwords.txt")
        self.assertEqual(cfg.whisper_initial_prompt_file.name,
                         "whisperx_initial_prompt.txt")
        self.assertNotEqual(cfg.whisper_hotwords_file,
                            cfg.whisper_initial_prompt_file)

    def test_each_reads_its_own_file(self):
        (self.src / "whisperx_hotwords.txt").write_text("Mysteria Bazyn", encoding="utf-8")
        (self.src / "whisperx_initial_prompt.txt").write_text("Use DEX not decks.", encoding="utf-8")
        cfg = self._cfg()
        self.assertEqual(cfg.whisper_hotwords, "Mysteria Bazyn")
        self.assertEqual(cfg.whisper_initial_prompt, "Use DEX not decks.")

    def test_missing_files_yield_empty_not_error(self):
        cfg = self._cfg()
        self.assertEqual(cfg.whisper_hotwords, "")
        self.assertEqual(cfg.whisper_initial_prompt, "")

    def test_filenames_are_configurable(self):
        (self.src / "terms.txt").write_text("Kethira", encoding="utf-8")
        (self.src / "style.txt").write_text("SPD not speed", encoding="utf-8")
        cfg = self._cfg("  hotwords_file: terms.txt\n"
                        "  initial_prompt_file: style.txt\n")
        self.assertEqual(cfg.whisper_hotwords, "Kethira")
        self.assertEqual(cfg.whisper_initial_prompt, "SPD not speed")

    def test_text_is_collapsed_to_one_line(self):
        """Blank lines and indentation collapse; the terms themselves do not."""
        (self.src / "whisperx_hotwords.txt").write_text(
            'Mysteria\n  Bazyn\n\n  Kethira  \n', encoding="utf-8")
        cfg = self._cfg()
        self.assertEqual(cfg.whisper_hotwords, "Mysteria Bazyn Kethira")

    def test_quotes_are_preserved(self):
        """Env values are argv entries, not shell text, so a quoted term stands.

        This asserted the opposite until run_docker_command stopped building a
        shell string: the reader stripped every double quote so the command would
        survive quoting. Nothing re-parses the value now, so a hand-maintained
        vocabulary list reaches WhisperX as written.
        """
        (self.src / "whisperx_hotwords.txt").write_text(
            'Mysteria\n  "Sir" Roderick\n', encoding="utf-8")
        cfg = self._cfg()
        self.assertEqual(cfg.whisper_hotwords, 'Mysteria "Sir" Roderick')

    def test_empty_key_disables_the_flag(self):
        cfg = self._cfg("  hotwords_file: \"\"\n")
        self.assertEqual(cfg.whisper_hotwords, "")


class EnvPlumbing(unittest.TestCase):
    """whisper_env carries both to the container as separate variables."""

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.src = self.root / "src"
        self.src.mkdir()

    def test_both_vars_present_and_distinct(self):
        (self.src / "whisperx_hotwords.txt").write_text("Mysteria", encoding="utf-8")
        (self.src / "whisperx_initial_prompt.txt").write_text("Use DEX", encoding="utf-8")
        env = whisper_env(Config(config_path=_write_config(self.root)))
        self.assertEqual(env["WHISPER_HOTWORDS"], "Mysteria")
        self.assertEqual(env["WHISPER_INITIAL_PROMPT"], "Use DEX")

    def test_absent_files_give_empty_strings(self):
        """The entrypoint skips a flag when its variable is empty."""
        env = whisper_env(Config(config_path=_write_config(self.root)))
        self.assertEqual(env["WHISPER_HOTWORDS"], "")
        self.assertEqual(env["WHISPER_INITIAL_PROMPT"], "")

    def test_config_without_the_new_property_still_works(self):
        """Duck-typed configs lacking initial_prompt must not break."""
        class Old:
            whisper_model = "tiny"; whisper_language = "en"
            whisper_compute_type = "float16"; whisper_batch_size = 8
            whisper_output_format = "json"; whisper_diarize = False
            whisper_hotwords = "x"
        env = whisper_env(Old())
        self.assertEqual(env["WHISPER_INITIAL_PROMPT"], "")


class EntrypointPassesBothFlags(unittest.TestCase):
    """Static check on the shell that actually builds the whisperx command."""

    def setUp(self):
        root = Path(__file__).resolve().parents[4]
        self.sh = (root / "docker" / "whisperx" / "entrypoint.sh").read_text(
            encoding="utf-8")

    def test_hotwords_flag_is_wired(self):
        self.assertIn('args+=(--hotwords "$HOTWORDS")', self.sh)

    def test_initial_prompt_flag_is_wired(self):
        self.assertIn('args+=(--initial_prompt "$INITIAL_PROMPT")', self.sh)

    def test_both_are_opt_in(self):
        self.assertIn('if [ -n "$HOTWORDS" ]; then', self.sh)
        self.assertIn('if [ -n "$INITIAL_PROMPT" ]; then', self.sh)

    def test_initial_prompt_reads_its_env_var(self):
        self.assertIn('INITIAL_PROMPT="${WHISPER_INITIAL_PROMPT:-}"', self.sh)


class CommentsAreStripped(unittest.TestCase):
    """The shipped samples use '#' headers to organise the vocabulary.

    Without stripping, that commentary is handed to WhisperX as hint terms —
    'WhisperX hint terms Omega Force Season 2 Weeks 76-79 Built from manual
    corrections...' — diluting the real ones.
    """

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.src = self.root / "src"
        self.src.mkdir()

    def test_comment_lines_are_dropped(self):
        (self.src / "whisperx_hotwords.txt").write_text(
            "# WhisperX hint terms — Season 2\n"
            "# One term per line.\n"
            "\n"
            "# --- Names / proper nouns ---\n"
            "Omega Force\n"
            "Aberrancy\n"
            "   # indented comment\n"
            "Mechanon\n", encoding="utf-8")
        cfg = Config(config_path=_write_config(self.root))
        self.assertEqual(cfg.whisper_hotwords, "Omega Force Aberrancy Mechanon")
        for word in ("WhisperX", "term", "indented", "Names"):
            self.assertNotIn(word, cfg.whisper_hotwords)

    def test_hash_inside_a_line_is_kept(self):
        """Only leading '#' marks a comment; mid-line text is data."""
        (self.src / "whisperx_hotwords.txt").write_text("Agent #7\nGrond\n", encoding="utf-8")
        cfg = Config(config_path=_write_config(self.root))
        self.assertEqual(cfg.whisper_hotwords, "Agent #7 Grond")

    def test_comment_only_file_is_empty(self):
        (self.src / "whisperx_hotwords.txt").write_text("# nothing but notes\n#\n", encoding="utf-8")
        cfg = Config(config_path=_write_config(self.root))
        self.assertEqual(cfg.whisper_hotwords, "")

    def test_applies_to_the_prompt_file_too(self):
        (self.src / "whisperx_initial_prompt.txt").write_text(
            "# campaign prompt\nOmega Force campaign. Use DEX not decks.\n")
        cfg = Config(config_path=_write_config(self.root))
        self.assertEqual(cfg.whisper_initial_prompt,
                         "Omega Force campaign. Use DEX not decks.")


class NonUtf8HintFiles(unittest.TestCase):
    """Hint files are hand-edited; Windows editors still default to cp1252.

    An em-dash or smart quote then makes the file invalid UTF-8, and a strict
    read raises UnicodeDecodeError — which is a ValueError, so it slips past an
    `except OSError` and kills the run.
    """

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.src = self.root / "src"
        self.src.mkdir()

    def test_cp1252_hotwords_file_still_loads(self):
        (self.src / "whisperx_hotwords.txt").write_bytes(
            "# hint terms — Season 2\nOmega Force\nAberrancy\n".encode("cp1252"))
        cfg = Config(config_path=_write_config(self.root))
        self.assertEqual(cfg.whisper_hotwords, "Omega Force Aberrancy")

    def test_cp1252_prompt_file_still_loads(self):
        (self.src / "whisperx_initial_prompt.txt").write_bytes(
            "Use DEX — not decks.".encode("cp1252"))
        cfg = Config(config_path=_write_config(self.root))
        self.assertIn("DEX", cfg.whisper_initial_prompt)

    def test_utf8_is_still_preferred(self):
        (self.src / "whisperx_hotwords.txt").write_text(
            "Mysteria — Kethira", encoding="utf-8")
        cfg = Config(config_path=_write_config(self.root))
        self.assertEqual(cfg.whisper_hotwords, "Mysteria — Kethira")


if __name__ == "__main__":
    unittest.main()
