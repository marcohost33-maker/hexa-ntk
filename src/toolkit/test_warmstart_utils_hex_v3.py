"""
test_warmstart_utils_hex_v3.py
==============================
Tests for safe partial checkpoint loading (warmstart).

Covers the module design invariants:
- only model weights are loaded (optimizer/scheduler payload keys are ignored)
- safe load path (weights_only=True) with safe_globals fallback for legacy
  numpy-scalar checkpoints
- prefix stripping (DDP "module.") and prefix mapping (architecture migration)
- dtype mismatch tolerance
- diagnose_checkpoint_safety static analysis
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from warmstart_utils_hex_v3 import (
    _apply_prefix_transforms,
    _extract_state_dict,
    _handle_dtype_mismatches,
    diagnose_checkpoint_safety,
    load_partial_checkpoint,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _NotWhitelistable:
    """Custom class: never in torch's safe-globals allowlist."""

    def __init__(self) -> None:
        self.x = 1


class TestExtractStateDict(unittest.TestCase):
    def test_rejects_non_dict_payload(self) -> None:
        with self.assertRaises(TypeError):
            _extract_state_dict([1, 2, 3])

    def test_model_state_dict_key_preferred(self) -> None:
        sd = TinyModel().state_dict()
        payload = {"model_state_dict": sd, "epoch": 3}
        self.assertIs(_extract_state_dict(payload), sd)

    def test_state_dict_key_fallback(self) -> None:
        sd = TinyModel().state_dict()
        payload = {"state_dict": sd, "epoch": 3}
        self.assertIs(_extract_state_dict(payload), sd)

    def test_flat_tensor_dict_accepted(self) -> None:
        sd = TinyModel().state_dict()
        self.assertIs(_extract_state_dict(sd), sd)

    def test_dict_without_tensors_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _extract_state_dict({"epoch": 3, "note": "no tensors here"})


class TestPrefixTransforms(unittest.TestCase):
    def test_strip_ddp_module_prefix(self) -> None:
        sd = {"module.linear.weight": torch.zeros(1), "module.linear.bias": torch.zeros(1)}
        out = _apply_prefix_transforms(sd, strip_prefixes=["module."], prefix_map=None)
        self.assertEqual(set(out), {"linear.weight", "linear.bias"})

    def test_strip_leaves_unprefixed_keys(self) -> None:
        sd = {"module.a": torch.zeros(1), "b": torch.ones(1)}
        out = _apply_prefix_transforms(sd, strip_prefixes=["module."], prefix_map=None)
        self.assertEqual(set(out), {"a", "b"})

    def test_prefix_map_renames(self) -> None:
        sd = {"encoder.w": torch.zeros(1), "head.w": torch.ones(1)}
        out = _apply_prefix_transforms(
            sd, strip_prefixes=None, prefix_map={"encoder.": "backbone."}
        )
        self.assertEqual(set(out), {"backbone.w", "head.w"})

    def test_no_transforms_is_identity(self) -> None:
        sd = {"a": torch.zeros(1)}
        out = _apply_prefix_transforms(sd, strip_prefixes=None, prefix_map=None)
        self.assertEqual(set(out), {"a"})


class TestDtypeMismatches(unittest.TestCase):
    def test_casts_to_model_dtype(self) -> None:
        model = TinyModel()  # float32
        sd = {k: v.to(torch.float64) for k, v in model.state_dict().items()}
        out = _handle_dtype_mismatches(sd, model)
        for v in out.values():
            self.assertEqual(v.dtype, torch.float32)

    def test_matching_dtypes_untouched(self) -> None:
        model = TinyModel()
        sd = model.state_dict()
        out = _handle_dtype_mismatches(dict(sd), model)
        for k in sd:
            self.assertTrue(torch.equal(out[k], sd[k]))


class TestLoadPartialCheckpoint(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _save(self, payload: object, name: str = "ckpt.pt") -> Path:
        path = self.tmpdir / name
        torch.save(payload, path)
        return path

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_partial_checkpoint(TinyModel(), self.tmpdir / "nope.pt")

    def test_happy_path_wrapped_payload(self) -> None:
        src = TinyModel()
        path = self._save({"model_state_dict": src.state_dict(), "epoch": 7})
        dst = TinyModel()
        report = load_partial_checkpoint(dst, path)
        self.assertEqual(report["action"], "warmstart")
        self.assertEqual(report["checkpoint_semantics"], "warmstart")
        self.assertEqual(report["load_path_type"], "safe")
        self.assertFalse(report["strict_mode_used"])
        self.assertEqual(report["missing_keys"], [])
        self.assertEqual(report["unexpected_keys"], [])
        self.assertEqual(report["dtype_casts"], 0)
        self.assertIn("INTENTIONALLY dropped", report["warning"])
        for k, v in src.state_dict().items():
            self.assertTrue(torch.equal(v, dst.state_dict()[k]), msg=f"Mismatch in {k}")

    def test_optimizer_state_in_payload_is_never_loaded(self) -> None:
        src = TinyModel()
        opt = torch.optim.SGD(src.parameters(), lr=0.1)
        path = self._save(
            {"model_state_dict": src.state_dict(), "optimizer_state_dict": opt.state_dict()}
        )
        dst = TinyModel()
        report = load_partial_checkpoint(dst, path)
        # Weights transferred, optimizer payload silently ignored (design invariant).
        self.assertEqual(report["missing_keys"], [])
        for k, v in src.state_dict().items():
            self.assertTrue(torch.equal(v, dst.state_dict()[k]))

    def test_ddp_prefix_strip(self) -> None:
        src = TinyModel()
        ddp_sd = {f"module.{k}": v for k, v in src.state_dict().items()}
        path = self._save({"model_state_dict": ddp_sd})
        dst = TinyModel()
        report = load_partial_checkpoint(dst, path, strip_prefixes=["module."])
        self.assertEqual(report["missing_keys"], [])
        self.assertEqual(report["unexpected_keys"], [])
        for k, v in src.state_dict().items():
            self.assertTrue(torch.equal(v, dst.state_dict()[k]))

    def test_prefix_map_migration(self) -> None:
        src = TinyModel()
        old_sd = {k.replace("linear.", "encoder."): v for k, v in src.state_dict().items()}
        path = self._save({"model_state_dict": old_sd})
        dst = TinyModel()
        report = load_partial_checkpoint(dst, path, prefix_map={"encoder.": "linear."})
        self.assertEqual(report["missing_keys"], [])
        self.assertEqual(report["unexpected_keys"], [])

    def test_dtype_cast_counted(self) -> None:
        src = TinyModel()
        sd64 = {k: v.to(torch.float64) for k, v in src.state_dict().items()}
        path = self._save({"model_state_dict": sd64})
        dst = TinyModel()
        report = load_partial_checkpoint(dst, path, allow_dtype_mismatch=True)
        self.assertEqual(report["dtype_casts"], len(sd64))
        for v in dst.state_dict().values():
            self.assertEqual(v.dtype, torch.float32)

    def test_dtype_mismatch_disallowed_reports_zero_casts(self) -> None:
        # allow_dtype_mismatch=False: kein expliziter Cast (dtype_casts=0);
        # load_state_dict castet via copy_ trotzdem implizit auf Model-dtype.
        src = TinyModel()
        sd64 = {k: v.to(torch.float64) for k, v in src.state_dict().items()}
        path = self._save({"model_state_dict": sd64})
        dst = TinyModel()
        report = load_partial_checkpoint(dst, path, allow_dtype_mismatch=False)
        self.assertEqual(report["dtype_casts"], 0)
        for v in dst.state_dict().values():
            self.assertEqual(v.dtype, torch.float32)

    def test_partial_load_reports_missing_and_unexpected(self) -> None:
        src = TinyModel()
        sd = dict(src.state_dict())
        del sd["linear.bias"]
        sd["extra.weight"] = torch.zeros(1)
        path = self._save({"model_state_dict": sd})
        report = load_partial_checkpoint(TinyModel(), path, strict=False)
        self.assertIn("linear.bias", report["missing_keys"])
        self.assertIn("extra.weight", report["unexpected_keys"])

    def test_strict_mode_raises_on_mismatch(self) -> None:
        src = TinyModel()
        sd = dict(src.state_dict())
        del sd["linear.bias"]
        path = self._save({"model_state_dict": sd})
        with self.assertRaises(RuntimeError):
            load_partial_checkpoint(TinyModel(), path, strict=True)

    def test_legacy_numpy_scalar_fallback(self) -> None:
        # Legacy checkpoints carry numpy scalars -> weights_only=True fails,
        # the safe_globals whitelisting fallback must recover the load.
        src = TinyModel()
        path = self._save(
            {"model_state_dict": src.state_dict(), "best_metric": np.float64(0.125)}
        )
        dst = TinyModel()
        report = load_partial_checkpoint(dst, path)
        self.assertEqual(report["load_path_type"], "safe")
        for k, v in src.state_dict().items():
            self.assertTrue(torch.equal(v, dst.state_dict()[k]))

    def test_non_whitelistable_object_raises(self) -> None:
        # Arbitrary Python objects are NOT rescued by the numpy whitelist:
        # the fallback must fail closed and re-raise the original error.
        path = self._save(
            {"model_state_dict": TinyModel().state_dict(), "attack": _NotWhitelistable()}
        )
        with self.assertRaises(Exception):
            load_partial_checkpoint(TinyModel(), path)

    def test_corrupt_file_error_reraised(self) -> None:
        path = self.tmpdir / "corrupt.pt"
        path.write_bytes(b"this is not a torch checkpoint")
        with self.assertRaises(Exception):
            load_partial_checkpoint(TinyModel(), path)


class TestDiagnoseCheckpointSafety(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_missing_file(self) -> None:
        report = diagnose_checkpoint_safety(self.tmpdir / "nope.pt")
        self.assertFalse(report["safe"])
        self.assertIn("File not found", report["error"])

    def test_safe_checkpoint(self) -> None:
        path = self.tmpdir / "safe.pt"
        torch.save({"model_state_dict": TinyModel().state_dict()}, path)
        report = diagnose_checkpoint_safety(path)
        self.assertTrue(report["safe"])
        self.assertEqual(report["unsafe_globals"], [])
        self.assertIsNone(report["error"])

    def test_unsafe_checkpoint_detected(self) -> None:
        path = self.tmpdir / "legacy.pt"
        torch.save(
            {"model_state_dict": TinyModel().state_dict(), "metric": np.float64(1.0)},
            path,
        )
        report = diagnose_checkpoint_safety(path)
        self.assertFalse(report["safe"])
        self.assertGreater(len(report["unsafe_globals"]), 0)

    def test_diagnostic_error_reported_not_raised(self) -> None:
        path = self.tmpdir / "garbage.pt"
        path.write_bytes(b"garbage bytes, not a zip archive")
        report = diagnose_checkpoint_safety(path)
        self.assertIsNone(report["safe"])
        self.assertIsNotNone(report["error"])


if __name__ == "__main__":
    unittest.main()
