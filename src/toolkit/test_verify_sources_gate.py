"""
test_verify_sources_gate.py
===========================
Fail-closed tests for scripts/verify_sources.py (PR #32, F32-2 and E7).

The gate must cover EVERY file under ``src/toolkit/`` and ``spec/`` -- not only
``*.py`` / ``*.md``. These tests build throw-away layouts, both as plain
directories (filesystem enumeration) and as git repositories (``git ls-files``
enumeration), and probe the dimensions file type / extension, upper/lower
case, sub-directory and symlink.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import shutil
import subprocess
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
SCHEMA = b'{"type": "object"}\n'


class _GateLayout(unittest.TestCase):
    """Minimal repo: src/toolkit/mod.py + spec/CONTRACT_v1.md, both pinned."""

    use_git = False

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "src" / "toolkit").mkdir(parents=True)
        (self.root / "spec").mkdir()
        (self.root / "src" / "toolkit" / "mod.py").write_bytes(MODULE)
        (self.root / "spec" / "CONTRACT_v1.md").write_bytes(CONTRACT)
        self.rows = [self._row("src/toolkit", MODULE, "mod.py"),
                     self._row("spec", CONTRACT, "CONTRACT_v1.md")]
        if self.use_git:
            if shutil.which("git") is None:
                self.fail("git executable required for git-mode gate tests")
            self._git("init", "-q")
            self._git("add", "-A")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            stdin=subprocess.DEVNULL, check=True, capture_output=True, text=True,
        ).stdout

    @staticmethod
    def _row(target: str, data: bytes, name: str) -> str:
        return f"| {_sha16(data)} | 2026-09-23 00:00 | {target} | in-repo: {name} |"

    def _run(self, *extra_rows: str) -> tuple[int, str]:
        header = "| SHA256 (16) | mtime | Ziel | Quelle |\n|---|---|---|---|\n"
        (self.root / "SOURCES.md").write_text(
            header + "\n".join([*self.rows, *extra_rows]) + "\n", encoding="utf-8"
        )
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = gate.main(self.root)
        return rc, out.getvalue()

    def _add(self, rel: str, data: bytes) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if self.use_git:
            self._git("add", "--", rel)


class TestVerifySourcesFilesystem(_GateLayout):
    use_git = False

    def test_matching_pins_pass(self) -> None:
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        self.assertIn("1 Spec-Eintraege", out)
        self.assertIn("Dateiliste: filesystem", out)

    def test_spec_drift_fails_closed(self) -> None:
        (self.root / "spec" / "CONTRACT_v1.md").write_bytes(CONTRACT + b"edit\n")
        rc, out = self._run()
        self.assertEqual(rc, 1, out)
        self.assertIn("[FAIL] spec/CONTRACT_v1.md", out)

    def test_unpinned_spec_md_is_a_gap(self) -> None:
        self.rows = self.rows[:1]
        rc, out = self._run()
        self.assertEqual(rc, 2, out)
        self.assertIn("[GAP ] spec/CONTRACT_v1.md", out)

    def test_last_spec_row_wins(self) -> None:
        self.rows.insert(1, f"| {'0' * 16} | 2026-09-23 00:00 | spec | in-repo: CONTRACT_v1.md |")
        rc, out = self._run()
        self.assertEqual(rc, 0, out)

    def test_spec_row_without_file_fails(self) -> None:
        (self.root / "spec" / "CONTRACT_v1.md").unlink()
        rc, out = self._run()
        self.assertEqual(rc, 1, out)
        self.assertIn("Datei fehlt", out)

    # --- dimension: file type / extension (E7) ---------------------------
    def test_unpinned_json_in_spec_is_a_gap(self) -> None:
        self._add("spec/schema.json", SCHEMA)
        rc, out = self._run()
        self.assertEqual(rc, 2, out)
        self.assertIn("[GAP ] spec/schema.json", out)

    def test_unpinned_arbitrary_extension_in_toolkit_is_a_gap(self) -> None:
        self._add("src/toolkit/weights.bin7", b"\x00\x01")
        rc, out = self._run()
        self.assertEqual(rc, 2, out)
        self.assertIn("[GAP ] src/toolkit/weights.bin7", out)

    def test_pinned_json_passes_and_drift_is_caught(self) -> None:
        self._add("spec/schema.json", SCHEMA)
        row = self._row("spec", SCHEMA, "schema.json")
        rc, out = self._run(row)
        self.assertEqual(rc, 0, out)
        (self.root / "spec" / "schema.json").write_bytes(SCHEMA + b" ")
        rc, out = self._run(row)
        self.assertEqual(rc, 1, out)
        self.assertIn("[FAIL] spec/schema.json", out)

    def test_file_without_extension_is_a_gap(self) -> None:
        self._add("spec/NOTICE", b"n\n")
        rc, out = self._run()
        self.assertEqual(rc, 2, out)
        self.assertIn("[GAP ] spec/NOTICE: Dateiname in SOURCES.md nicht ausdrueckbar", out)

    def test_file_name_with_space_is_a_gap(self) -> None:
        self._add("spec/my notes.md", b"n\n")
        rc, out = self._run()
        self.assertEqual(rc, 2, out)
        self.assertIn("[GAP ] spec/my notes.md: Dateiname in SOURCES.md nicht ausdrueckbar", out)

    # --- dimension: symlink (filesystem mode) ----------------------------
    def test_filesystem_symlink_is_a_gap(self) -> None:
        link = self.root / "spec" / "LINK.md"
        try:
            link.symlink_to(self.root / "spec" / "CONTRACT_v1.md")
        except (OSError, NotImplementedError) as exc:  # Windows w/o privilege
            self.skipTest(f"symlink creation not permitted here: {exc}")
        rc, out = self._run(self._row("spec", CONTRACT, "LINK.md"))
        self.assertEqual(rc, 2, out)
        self.assertIn("[GAP ] spec/LINK.md: symbolischer Link", out)

    # --- dimension: upper/lower case --------------------------------------
    def test_case_mismatch_between_row_and_file_is_not_accepted(self) -> None:
        self.rows[1] = self._row("spec", CONTRACT, "contract_v1.md")
        rc, out = self._run()
        self.assertEqual(rc, 2, out)
        self.assertIn("[GAP ] spec/CONTRACT_v1.md", out)

    # --- dimension: sub-directory -----------------------------------------
    def test_nested_file_is_not_silently_skipped(self) -> None:
        self._add("spec/sub/HIDDEN.yaml", b"x: 1\n")
        rc, out = self._run()
        self.assertEqual(rc, 2, out)
        self.assertIn("spec/sub/HIDDEN.yaml: verschachtelte Datei", out)

    def test_documented_exclusions_only(self) -> None:
        self._add("src/toolkit/__init__.py", b"")
        self._add("src/toolkit/__pycache__/mod.cpython-312.pyc", b"\x00")
        rc, out = self._run()
        self.assertEqual(rc, 0, out)


class TestVerifySourcesGit(_GateLayout):
    use_git = True

    def test_git_mode_matching_pins_pass(self) -> None:
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        self.assertIn("Dateiliste: git", out)

    def test_tracked_unpinned_json_is_a_gap(self) -> None:
        self._add("spec/schema.json", SCHEMA)
        rc, out = self._run()
        self.assertEqual(rc, 2, out)
        self.assertIn("[GAP ] spec/schema.json", out)

    def test_untracked_file_is_a_gap_before_git_add(self) -> None:
        (self.root / "src" / "toolkit" / "new_mod.cfg").write_bytes(b"a=1\n")
        rc, out = self._run()
        self.assertEqual(rc, 2, out)
        self.assertIn("[GAP ] src/toolkit/new_mod.cfg", out)

    def test_tracked_symlink_is_a_gap(self) -> None:
        # Mode 120000 via the index: platform-independent (no symlink privilege
        # needed); the gate must read the git mode, not only the filesystem.
        blob = subprocess.run(
            ["git", "-C", str(self.root), "hash-object", "-w", "--stdin"],
            input=b"CONTRACT_v1.md", check=True, capture_output=True,
        ).stdout.decode().strip()
        self._git("update-index", "--add", "--cacheinfo", f"120000,{blob},spec/LINK.md")
        rc, out = self._run(self._row("spec", CONTRACT, "LINK.md"))
        self.assertEqual(rc, 2, out)
        self.assertIn("[GAP ] spec/LINK.md: symbolischer Link", out)

    def test_nested_tracked_file_is_a_gap(self) -> None:
        self._add("src/toolkit/sub/extra.py", MODULE)
        rc, out = self._run()
        self.assertEqual(rc, 2, out)
        self.assertIn("src/toolkit/sub/extra.py: verschachtelte Datei", out)

    def test_tracked_pycache_is_documented_exception(self) -> None:
        """A tracked cache file remains excluded in Git mode."""
        self._add("src/toolkit/__pycache__/mod.cpython-312.pyc", b"\\x00")
        rc, out = self._run()
        self.assertEqual(rc, 0, out)



if __name__ == "__main__":
    unittest.main()
