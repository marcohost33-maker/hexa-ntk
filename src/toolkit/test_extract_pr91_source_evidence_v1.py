"""Isolated ephemeral Git test: NEVER uses or mutates user's repository."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
script = ROOT / "scripts/extract_pr91_source_evidence_v1.py"
spec = importlib.util.spec_from_file_location("extract_pr91", script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def run(repo: Path, *args):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
    return p.stdout


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.repo = self.dir / "isolated-repo"
        self.repo.mkdir()
        run(self.repo, "init", "-q")
        run(self.repo, "config", "user.name", "Local Test")
        run(self.repo, "config", "user.email", "local@example.invalid")
        (self.repo / "results").mkdir()
        self.data = [("results/a.json", b'{"a":1}\n'), ("results/b.json", b'{"b":2}\n')]
        for path, raw in self.data:
            (self.repo / path).write_bytes(raw)
        run(self.repo, "add", ".")
        run(self.repo, "commit", "-q", "-m", "local fixture")
        self.head = run(self.repo, "rev-parse", "HEAD").decode().strip()
        self.files = tuple(
            (
                path,
                mod.git_blob if False else hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
                hashlib.sha256(raw).hexdigest(),
            )
            for path, raw in self.data
        )

    def test_offline_positive_git_object_extraction(self):
        out = self.dir / "evidence"
        r = mod.extract(self.repo, out, head=self.head, files=self.files)
        self.assertEqual(len(r["files"]), 2)
        for path, raw in self.data:
            self.assertEqual((out / Path(path).name).read_bytes(), raw)
        self.assertFalse(r["owner_approved"])

    def test_dirty_worktree_does_not_contaminate_git_object(self):
        (self.repo / "results/a.json").write_bytes(b'changed\n')
        out = self.dir / "evidence"
        mod.extract(self.repo, out, head=self.head, files=self.files)
        self.assertEqual((out / "a.json").read_bytes(), b'{"a":1}\n')

    def test_wrong_sha256_before_any_write(self):
        bad = list(self.files)
        a, b, c = bad[1]
        bad[1] = (a, b, "0" * 64)
        out = self.dir / "evidence"
        with self.assertRaises(mod.ExtractError):
            mod.extract(self.repo, out, head=self.head, files=tuple(bad))
        self.assertFalse(out.exists())

    def test_wrong_git_blob(self):
        bad = list(self.files)
        a, b, c = bad[0]
        bad[0] = (a, "a" * 40, c)
        with self.assertRaises(mod.ExtractError):
            mod.extract(self.repo, self.dir / "no", head=self.head, files=tuple(bad))

    def test_wrong_head(self):
        with self.assertRaises(mod.ExtractError):
            mod.extract(self.repo, self.dir / "no", head="f" * 40, files=self.files)

    def test_existing_folder_never_overwritten(self):
        out = self.dir / "evidence"
        out.mkdir()
        (out / "a.json").write_text("keep")
        with self.assertRaises(mod.ExtractError):
            mod.extract(self.repo, out, head=self.head, files=self.files)
        self.assertEqual((out / "a.json").read_text(), "keep")

    def test_tree_mode_mismatch(self):
        bad = list(self.files)
        a, b, c = bad[0]
        bad[0] = ("results/missing.json", b, c)
        with self.assertRaises(mod.ExtractError):
            mod.extract(self.repo, self.dir / "out", head=self.head, files=tuple(bad))

    def test_reject_path_traversal(self):
        with self.assertRaises(mod.ExtractError):
            mod.extract(self.repo, self.dir / "out", head=self.head, files=(("results/../x", "a" * 40, "b" * 64),))

    def test_filename_collision(self):
        bad = self.files + (("results/sub/a.json", self.files[0][1], self.files[0][2]),)
        with self.assertRaises(mod.ExtractError):
            mod.extract(self.repo, self.dir / "out", head=self.head, files=bad)


if __name__ == "__main__":
    unittest.main()
