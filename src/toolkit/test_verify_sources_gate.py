"""
test_verify_sources_gate.py
===========================
Fail-closed tests for scripts/verify_sources.py (PR #32, F32-2).

Before this change the gate parsed only ``src/toolkit`` rows, so the SHA pin
of the authoritative ``spec/`` contract was never checked. These tests build a
throw-away repository layout and assert the exit codes for the spec target.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

_GATE = Path(__file__).resolve().parents[2] / "scripts" / "verify_sources.py"
_spec = importlib.util.spec_from_file_location("verify_sources_gate", _GATE)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _sha16(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()[:16].upper()


MODULE = b"x = 1\n"
CONTRACT = b"# Contract v1\n\nD = max(E_s, eps) / max(E_0, eps)\n"


class TestVerifySourcesSpecTarget(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "src" / "toolkit").mkdir(parents=True)
        (self.root / "spec").mkdir()
        (self.root / "src" / "toolkit" / "__init__.py").write_bytes(b"")
        (self.root / "src" / "toolkit" / "mod.py").write_bytes(MODULE)
        (self.root / "spec" / "CONTRACT_v1.md").write_bytes(CONTRACT)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_sources(self, *rows: str) -> None:
        header = "| SHA256 (16) | mtime | Ziel | Quelle |\n|---|---|---|---|\n"
        (self.root / "SOURCES.md").write_text(
            header + "\n".join(rows) + "\n", encoding="utf-8"
        )

    def _toolkit_row(self) -> str:
        return f"| {_sha16(MODULE)} | 2026-09-23 00:00 | src/toolkit | in-repo: mod.py |"

    def _spec_row(self, sha: str) -> str:
        return f"| {sha} | 2026-09-23 00:00 | spec | in-repo: CONTRACT_v1.md |"

    def _run(self) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = gate.main(self.root)
        return rc, out.getvalue()

    def test_matching_spec_pin_passes(self) -> None:
        self._write_sources(self._toolkit_row(), self._spec_row(_sha16(CONTRACT)))
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        self.assertIn("1 Spec-Eintraege | PASS", out)

    def test_spec_drift_fails_closed(self) -> None:
        self._write_sources(self._toolkit_row(), self._spec_row(_sha16(CONTRACT)))
        (self.root / "spec" / "CONTRACT_v1.md").write_bytes(CONTRACT + b"edit\n")
        rc, out = self._run()
        self.assertEqual(rc, 1, out)
        self.assertIn("[FAIL] spec/CONTRACT_v1.md", out)

    def test_unpinned_spec_is_a_gap(self) -> None:
        self._write_sources(self._toolkit_row())
        rc, out = self._run()
        self.assertEqual(rc, 2, out)
        self.assertIn("[GAP ] spec/CONTRACT_v1.md", out)

    def test_last_spec_row_wins(self) -> None:
        self._write_sources(
            self._toolkit_row(),
            self._spec_row("0" * 16),
            self._spec_row(_sha16(CONTRACT)),
        )
        rc, out = self._run()
        self.assertEqual(rc, 0, out)

    def test_spec_row_without_file_fails(self) -> None:
        (self.root / "spec" / "CONTRACT_v1.md").unlink()
        self._write_sources(self._toolkit_row(), self._spec_row(_sha16(CONTRACT)))
        rc, out = self._run()
        self.assertEqual(rc, 1, out)
        self.assertIn("Datei fehlt", out)

    def test_nested_spec_file_is_not_silently_skipped(self) -> None:
        self._write_sources(self._toolkit_row(), self._spec_row(_sha16(CONTRACT)))
        (self.root / "spec" / "sub").mkdir()
        (self.root / "spec" / "sub" / "HIDDEN.md").write_bytes(b"x\n")
        rc, out = self._run()
        self.assertEqual(rc, 2, out)
        self.assertIn("spec/sub/HIDDEN.md", out)

    def test_spec_row_with_wrong_suffix_is_unparseable(self) -> None:
        self._write_sources(
            self._toolkit_row(),
            self._spec_row(_sha16(CONTRACT)),
            f"| {_sha16(MODULE)} | 2026-09-23 00:00 | spec | in-repo: mod.py |",
        )
        rc, out = self._run()
        self.assertEqual(rc, 3, out)


if __name__ == "__main__":
    unittest.main()
