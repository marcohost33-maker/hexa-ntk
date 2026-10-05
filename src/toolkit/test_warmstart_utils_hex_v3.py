"""
test_warmstart_utils_hex_v3.py
==============================
Tests for weight-only warmstart loading.

Covers:
- payload formats (flat state_dict, "model_state_dict", "state_dict")
- prefix stripping / mapping, and fail-closed key collisions
- dtype casting and its count
- optional skipping of shape-mismatched tensors
- legacy numpy-scalar fallback via safe_globals
- diagnose_checkpoint_safety
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from warmstart_utils_hex_v3 import (
    diagnose_checkpoint_safety,
    load_partial_checkpoint,
)


class _Net(nn.Module):
    def __init__(self, hidden: int = 3, out: int = 2) -> None:
        super().__init__()
        self.enc = nn.Linear(4, hidden)
        self.head = nn.Linear(hidden, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


class _Unpicklable:
    """A custom class: weights_only=True must refuse it."""


class WarmstartCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _save(self, payload, name: str = "ckpt.pt") -> Path:
        path = self.dir / name
        torch.save(payload, path)
        return path


class TestPayloadFormats(WarmstartCase):
    def test_formats_load_the_same_weights(self) -> None:
        torch.manual_seed(0)
        src = _Net()
        for payload in (
            src.state_dict(),
            {"model_state_dict": src.state_dict(), "epoch": 3},
            {"state_dict": src.state_dict()},
        ):
            with self.subTest(keys=sorted(payload)[:2]):
                dst = _Net()
                report = load_partial_checkpoint(dst, self._save(payload), strict=True)
                self.assertEqual(report["checkpoint_semantics"], "warmstart")
                self.assertEqual(report["load_path_type"], "safe")
                self.assertEqual(report["missing_keys"], [])
                for k, v in src.state_dict().items():
                    self.assertTrue(torch.equal(v, dst.state_dict()[k]), msg=k)

    def test_non_dict_payload_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            load_partial_checkpoint(_Net(), self._save([torch.ones(1)]))

    def test_dict_without_tensors_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            load_partial_checkpoint(_Net(), self._save({"epoch": 1}))

    def test_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_partial_checkpoint(_Net(), self.dir / "nope.pt")

    def test_partial_load_reports_missing_and_unexpected(self) -> None:
        src = _Net()
        payload = {k: v for k, v in src.state_dict().items() if k.startswith("enc.")}
        payload["extra.weight"] = torch.zeros(1)
        report = load_partial_checkpoint(_Net(), self._save(payload))
        self.assertEqual(sorted(report["missing_keys"]), ["head.bias", "head.weight"])
        self.assertEqual(report["unexpected_keys"], ["extra.weight"])


class TestPrefixTransforms(WarmstartCase):
    def test_strip_ddp_prefix(self) -> None:
        src = _Net()
        payload = {f"module.{k}": v for k, v in src.state_dict().items()}
        dst = _Net()
        report = load_partial_checkpoint(dst, self._save(payload), strip_prefixes=["module."],
                                         strict=True)
        self.assertEqual(report["unexpected_keys"], [])
        self.assertTrue(torch.equal(src.enc.weight, dst.enc.weight))

    def test_prefix_map_for_architecture_migration(self) -> None:
        src = _Net()
        payload = {k.replace("enc.", "encoder."): v for k, v in src.state_dict().items()}
        dst = _Net()
        load_partial_checkpoint(dst, self._save(payload), prefix_map={"encoder.": "enc."},
                                strict=True)
        self.assertTrue(torch.equal(src.enc.bias, dst.enc.bias))

    def test_strip_collision_fails_closed(self) -> None:
        # Before: "module.enc.weight" and "enc.weight" both became
        # "enc.weight" and one tensor was dropped silently.
        src = _Net()
        payload = dict(src.state_dict())
        payload["module.enc.weight"] = torch.full_like(src.enc.weight, 9.0)
        dst = _Net()
        before = dst.enc.weight.clone()
        with self.assertRaises(ValueError) as ctx:
            load_partial_checkpoint(dst, self._save(payload), strip_prefixes=["module."])
        self.assertIn("'module.enc.weight'", str(ctx.exception))
        self.assertIn("'enc.weight'", str(ctx.exception))
        self.assertTrue(torch.equal(before, dst.enc.weight))

    def test_map_collision_fails_closed_and_names_original_keys(self) -> None:
        payload = {"a.w": torch.ones(1), "b.w": torch.zeros(1)}
        with self.assertRaises(ValueError) as ctx:
            load_partial_checkpoint(_Net(), self._save(payload), strip_prefixes=["x."],
                                    prefix_map={"a.": "c.", "b.": "c."})
        self.assertIn("'a.w'", str(ctx.exception))
        self.assertIn("'b.w'", str(ctx.exception))


class TestDtypeAndShape(WarmstartCase):
    def test_dtype_cast_is_counted(self) -> None:
        src = _Net().double()
        dst = _Net()
        report = load_partial_checkpoint(dst, self._save(src.state_dict()))
        self.assertEqual(report["dtype_casts"], 4)
        self.assertEqual(dst.enc.weight.dtype, torch.float32)
        self.assertTrue(torch.allclose(dst.enc.weight.double(), src.enc.weight))

    def test_dtype_mismatch_without_cast(self) -> None:
        src = _Net().double()
        report = load_partial_checkpoint(_Net(), self._save(src.state_dict()),
                                         allow_dtype_mismatch=False)
        self.assertEqual(report["dtype_casts"], 0)

    def test_shape_mismatch_raises_by_default(self) -> None:
        src = _Net(out=5)
        with self.assertRaises(RuntimeError):
            load_partial_checkpoint(_Net(out=2), self._save(src.state_dict()))

    def test_shape_mismatch_skipped_and_reported(self) -> None:
        torch.manual_seed(0)
        src = _Net(out=5)
        dst = _Net(out=2)
        head_before = dst.head.weight.clone()
        report = load_partial_checkpoint(dst, self._save(src.state_dict()),
                                         skip_shape_mismatch=True)
        self.assertEqual(
            report["shape_mismatches"],
            [
                {"key": "head.weight", "checkpoint_shape": [5, 3], "model_shape": [2, 3]},
                {"key": "head.bias", "checkpoint_shape": [5], "model_shape": [2]},
            ],
        )
        self.assertEqual(sorted(report["missing_keys"]), ["head.bias", "head.weight"])
        self.assertTrue(torch.equal(src.enc.weight, dst.enc.weight))
        self.assertTrue(torch.equal(head_before, dst.head.weight))

    def test_shape_skip_with_strict_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            load_partial_checkpoint(_Net(out=2), self._save(_Net(out=5).state_dict()),
                                    skip_shape_mismatch=True, strict=True)


class TestLegacyAndDiagnosis(WarmstartCase):
    def test_numpy_scalar_legacy_checkpoint_loads_via_whitelist(self) -> None:
        src = _Net()
        payload = {"model_state_dict": src.state_dict(), "best": np.float64(0.25)}
        path = self._save(payload)
        with self.assertRaises(Exception):
            torch.load(path, weights_only=True)  # precondition: plain safe load fails
        dst = _Net()
        report = load_partial_checkpoint(dst, path, strict=True)
        self.assertEqual(report["load_path_type"], "safe")
        self.assertTrue(torch.equal(src.head.bias, dst.head.bias))

    def test_arbitrary_object_is_still_refused(self) -> None:
        path = self._save({"model_state_dict": _Net().state_dict(), "obj": _Unpicklable()})
        with self.assertRaises(Exception):
            load_partial_checkpoint(_Net(), path)

    def test_diagnose_safe_and_unsafe(self) -> None:
        safe = diagnose_checkpoint_safety(self._save(_Net().state_dict(), "safe.pt"))
        self.assertTrue(safe["safe"])
        self.assertIsNone(safe["error"])
        unsafe = diagnose_checkpoint_safety(
            self._save({"obj": _Unpicklable()}, "unsafe.pt")
        )
        self.assertFalse(unsafe["safe"])
        self.assertTrue(any("_Unpicklable" in g for g in unsafe["unsafe_globals"]))

    def test_diagnose_missing_file(self) -> None:
        report = diagnose_checkpoint_safety(self.dir / "missing.pt")
        self.assertFalse(report["safe"])
        self.assertIn("File not found", report["error"])

    def test_diagnose_corrupt_file(self) -> None:
        path = self.dir / "corrupt.pt"
        path.write_bytes(b"not a checkpoint")
        report = diagnose_checkpoint_safety(path)
        self.assertIsNone(report["safe"])
        self.assertIsNotNone(report["error"])


if __name__ == "__main__":
    unittest.main()
