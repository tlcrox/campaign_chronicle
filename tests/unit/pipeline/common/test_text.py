#!/usr/bin/env python3
"""A cp1252 byte in a file must mean the same thing on Windows and on Linux.

Python's default text encoding is the locale's — cp1252 on most Windows
machines, UTF-8 on Linux — so an unpinned ``read_text()`` decodes the same
bytes two different ways depending on where it runs. Every read in this
codebase pins ``encoding="utf-8"``, which makes the platforms agree, but for
files a human hand-edits that agreement is "both crash": Notepad still saves
cp1252, and one em-dash in a comment then raises UnicodeDecodeError — a
ValueError, which slips past ``except OSError`` — out of config loading.

Each test here writes a byte sequence that is valid cp1252 and invalid UTF-8
(0x97, the em-dash) at one real read location and asserts the outcome, which
by construction cannot depend on the host locale.
"""

import ast
import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pipeline.common.text import read_text_lenient
from pipeline.config import Config
from pipeline.merge.combine_transcripts import load_merge_speaker_data
from pipeline.scenes.merge_segments import _read_offset

REPO_ROOT = Path(__file__).resolve().parents[4]

# "—" is U+2014: 0x97 in cp1252, three bytes (E2 80 94) in UTF-8. Encoded as
# cp1252 it is a single byte that no UTF-8 decoder accepts.
EM_DASH_CP1252 = "—".encode("cp1252")
assert EM_DASH_CP1252 == b"\x97"


class ReadTextLenient(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.path = Path(self.t.name) / "notes.txt"

    def test_utf8_is_read_as_utf8(self):
        self.path.write_bytes("Mysteria — Kethira".encode("utf-8"))
        with self.assertNoLogs("pipeline.common.text", level=logging.WARNING):
            self.assertEqual(read_text_lenient(self.path), "Mysteria — Kethira")

    def test_cp1252_falls_back_and_warns(self):
        self.path.write_bytes("Mysteria — Kethira".encode("cp1252"))
        with self.assertLogs("pipeline.common.text", level=logging.WARNING) as cm:
            self.assertEqual(read_text_lenient(self.path), "Mysteria — Kethira")
        self.assertIn("not valid UTF-8", cm.output[0])
        self.assertIn("notes.txt", cm.output[0])

    def test_a_missing_file_is_still_an_os_error(self):
        """Callers wrap this in ``except OSError``; the fallback must not turn a
        missing file into something else."""
        with self.assertRaises(OSError):
            read_text_lenient(self.path)


class ConfigYamlSavedAsCp1252(unittest.TestCase):
    """config.py: the file every entry point reads first."""

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        (self.root / "config").mkdir()
        (self.root / "src").mkdir()

    def _write(self, text: str, encoding: str) -> Path:
        p = self.root / "config" / "config.yaml"
        p.write_bytes(text.encode(encoding))
        return p

    def test_an_em_dash_in_a_comment_does_not_crash_loading(self):
        body = ("# Golden Gate Guardians — Season 2\n"
                "scenes:\n"
                "  threshold: 5.0\n"
                "orchestration:\n"
                f"  source_dir: {(self.root / 'src').as_posix()}\n")
        cfg = Config(config_path=self._write(body, "cp1252"))
        self.assertEqual(cfg.scene_threshold, 5.0)

    def test_the_same_document_as_utf8_reads_identically(self):
        body = ("# Golden Gate Guardians — Season 2\n"
                "scenes:\n"
                "  threshold: 5.0\n"
                "orchestration:\n"
                f"  source_dir: {(self.root / 'src').as_posix()}\n")
        cfg = Config(config_path=self._write(body, "utf-8"))
        self.assertEqual(cfg.scene_threshold, 5.0)


class SpeakerConfigSavedAsCp1252(unittest.TestCase):
    """combine_transcripts.py: character names are where accents turn up."""

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.path = Path(self.t.name) / "speaker_config.json"
        self.cfg = SimpleNamespace(speaker_config_file=self.path)

    def test_accented_names_survive_a_cp1252_save(self):
        doc = {"filename_mapping": {"rox": "José — the Bard"},
               "filler_phrases": ["um", "uh"]}
        self.path.write_bytes(json.dumps(doc, ensure_ascii=False).encode("cp1252"))
        mapping, fillers = load_merge_speaker_data(self.cfg)
        self.assertEqual(mapping, {"rox": "José — the Bard"})
        self.assertEqual(fillers, {"um", "uh"})

    def test_utf8_reads_identically(self):
        doc = {"filename_mapping": {"rox": "José — the Bard"}}
        self.path.write_bytes(json.dumps(doc, ensure_ascii=False).encode("utf-8"))
        mapping, _ = load_merge_speaker_data(self.cfg)
        self.assertEqual(mapping, {"rox": "José — the Bard"})


class OffsetFileWithACp1252Byte(unittest.TestCase):
    """merge_segments.py: tool-written, so no fallback — but the outcome must
    be the documented warn-and-0.0, not a traceback, and not a value that
    differs by host locale."""

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.seg = Path(self.t.name) / "segment_0"
        self.seg.mkdir()

    def test_a_clean_offset_is_read(self):
        (self.seg / "offset.txt").write_bytes(b"1800.5\n")
        self.assertEqual(_read_offset(self.seg), 1800.5)

    def test_a_cp1252_byte_is_a_bad_offset_on_every_platform(self):
        """0xA0 is a no-break space in cp1252 and ``strip()`` removes it, so
        the unpinned read this replaced returned 1800.5 on Windows and 0.0 on
        Linux from the very same bytes. Pinned, it is invalid UTF-8 everywhere,
        and the documented answer for a bad offset is 0.0."""
        (self.seg / "offset.txt").write_bytes(b"1800.5\xa0\n")
        self.assertEqual(_read_offset(self.seg), 0.0)


class EveryTextReadNamesItsEncoding(unittest.TestCase):
    """Static: no ``open()``, ``read_text()`` or ``write_text()`` in operational
    code without ``encoding=``. Binary opens are exempt. The one this caught
    (offset.txt) had been copied from a test fixture that also omitted it."""

    CALLS = {"open", "read_text", "write_text"}

    @staticmethod
    def _is_binary_open(call: ast.Call) -> bool:
        mode = call.args[1] if len(call.args) > 1 else None
        for kw in call.keywords:
            if kw.arg == "mode":
                mode = kw.value
        return (isinstance(mode, ast.Constant) and isinstance(mode.value, str)
                and "b" in mode.value)

    def test_src_and_scripts_pin_every_text_encoding(self):
        offenders = []
        for folder in ("src", "scripts"):
            for path in sorted((REPO_ROOT / folder).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    name = (func.id if isinstance(func, ast.Name)
                            else func.attr if isinstance(func, ast.Attribute)
                            else None)
                    if name not in self.CALLS:
                        continue
                    if name == "open" and self._is_binary_open(node):
                        continue
                    if any(kw.arg == "encoding" for kw in node.keywords):
                        continue
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}()")
        self.assertEqual(offenders, [],
                         "text I/O without encoding= decodes by host locale")


if __name__ == "__main__":
    unittest.main()
