"""Source-bound generator: all 7 anchors must match exactly once."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "patch_synthetic_cost_oracle_v1.py"
spec = importlib.util.spec_from_file_location("local_patch_synthetic_cost_oracle", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.synthetic = (
            "#!/usr/bin/env python3\n" + "\n".join(old.strip("\n") for old, _ in module.REPLACEMENTS) + "\n"
        ).encode("utf-8")

    def test_reject_nonoriginal_blob(self):
        with self.assertRaises(module.PatchError):
            module.transform(self.synthetic)

    def test_transform_mock_source_changes_all_seven_anchors(self):
        out = module.transform(self.synthetic, require_blob=False).decode("utf-8")
        self.assertIn("T_max=args.schedule_epochs", out)
        self.assertIn("non-finite minibatch loss before optimizer update", out)
        self.assertIn("D4 draft requires --schedule-epochs=50", out)
        self.assertIn("non-finite measured minibatch loss", out)
        self.assertIn("non-finite warmup minibatch loss", out)
        self.assertIn("scheduler_t_max", out)
        self.assertIn("--schedule-epochs", out)
        self.assertNotIn("T_max=args.repeats", out)

    def test_duplicate_source_anchor_fails(self):
        a = self.synthetic.decode("utf-8")
        with self.assertRaises(module.PatchError):
            module.transform((a + a).encode("utf-8"), require_blob=False)

    def test_partial_source_fails(self):
        a = self.synthetic.decode("utf-8")
        with self.assertRaises(module.PatchError):
            module.transform(a.replace("CosineAnnealingLR", "CyclicLR").encode(), require_blob=False)

    def test_line_endings_fails(self):
        with self.assertRaises(module.PatchError):
            module.transform(self.synthetic.replace(b"\n", b"\r\n"), require_blob=False)

    def test_invalid_source_bytes_fail(self):
        with self.assertRaises(module.PatchError):
            module.transform(b"\xff")
        with self.assertRaises(module.PatchError):
            module.transform(b"1" * (module.CAP + 1))

    def test_blob_matches_git_hash_object(self):
        import hashlib

        self.assertEqual(module.blob(b"hello"), hashlib.sha1(b"blob 5\0hello").hexdigest())

    def test_inplace_write_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder) / "original.py"
            f.write_bytes(self.synthetic)
            with self.assertRaises(module.PatchError):
                module.apply(f, f)

    def test_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder) / "original.py"
            out = Path(folder) / "output.py"
            f.write_bytes(self.synthetic)
            out.write_bytes(b"keep")
            with self.assertRaises(module.PatchError):
                module.apply(f, out)
            self.assertEqual(out.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
