"""
test_checkpoint_integration_hex_v3.py
=====================================
Tests for the merged checkpoint integration manager.

Covers:
- BestCheckpointPolicy logic (min_delta, mode)
- AtomicCheckpointWriter (atomic write, crash cleanup)
- CheckpointIntegrationManager:
  - best model save/load roundtrip
  - best model min_delta policy
  - last model save
  - resume save/load roundtrip
  - warmstart does not restore optimizer
  - EMA state roundtrip
  - checkpoint_semantics field
  - is_main_process guard
  - best model loads with weights_only=True
"""
from __future__ import annotations

import os
import stat
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from checkpoint_integration_hex_v3 import (
    AtomicCheckpointWriter,
    BestCheckpointPolicy,
    CheckpointDurabilityError,
    CheckpointIntegrationManager,
)
from resume_utils_hex_v3 import save_resume_checkpoint


# ------------------------------------------------------------------
# Dummy model for testing
# ------------------------------------------------------------------


class DummyModel(nn.Module):
    """Minimal model for checkpoint tests."""

    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_test_model() -> DummyModel:
    return DummyModel(in_features=4, out_features=2)


def build_test_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def build_test_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.StepLR:
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


def run_one_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return loss."""
    model.train()
    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    optimizer.zero_grad()
    pred = model(x)
    loss = torch.mean((pred - target) ** 2)
    loss.backward()
    optimizer.step()
    return float(loss.item())


# ------------------------------------------------------------------
# Tests: BestCheckpointPolicy
# ------------------------------------------------------------------


class TestBestCheckpointPolicy(unittest.TestCase):
    def test_min_mode_first_value_always_improves(self) -> None:
        policy = BestCheckpointPolicy(mode="min", min_delta=0.0)
        self.assertTrue(policy.is_improved(1.0, None))

    def test_min_mode_lower_is_better(self) -> None:
        policy = BestCheckpointPolicy(mode="min", min_delta=0.0)
        self.assertTrue(policy.is_improved(0.5, 1.0))
        self.assertFalse(policy.is_improved(1.0, 0.5))

    def test_min_mode_min_delta_prevents_noise(self) -> None:
        policy = BestCheckpointPolicy(mode="min", min_delta=0.01)
        # 0.999 is NOT enough improvement over 1.0 with min_delta=0.01
        self.assertFalse(policy.is_improved(0.999, 1.0))
        # 0.989 IS enough
        self.assertTrue(policy.is_improved(0.989, 1.0))

    def test_max_mode_higher_is_better(self) -> None:
        policy = BestCheckpointPolicy(mode="max", min_delta=0.0)
        self.assertTrue(policy.is_improved(1.0, 0.5))
        self.assertFalse(policy.is_improved(0.5, 1.0))

    def test_max_mode_min_delta(self) -> None:
        policy = BestCheckpointPolicy(mode="max", min_delta=0.01)
        self.assertFalse(policy.is_improved(1.005, 1.0))
        self.assertTrue(policy.is_improved(1.011, 1.0))

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            BestCheckpointPolicy(mode="invalid")

    def test_negative_min_delta_raises(self) -> None:
        with self.assertRaises(ValueError):
            BestCheckpointPolicy(mode="min", min_delta=-0.1)


# ------------------------------------------------------------------
# Tests: AtomicCheckpointWriter
# ------------------------------------------------------------------


class TestAtomicCheckpointWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_save_torch_payload_atomic_creates_file(self) -> None:
        path = Path(self.tmpdir.name) / "test.pt"
        payload = {"key": torch.tensor([1.0, 2.0, 3.0])}

        result = AtomicCheckpointWriter.save_torch_payload_atomic(path, payload)

        self.assertTrue(Path(result).exists())
        loaded = torch.load(result, weights_only=True)
        self.assertTrue(torch.equal(loaded["key"], payload["key"]))

    def test_atomic_write_no_temp_files_on_success(self) -> None:
        path = Path(self.tmpdir.name) / "clean.pt"
        payload = {"x": torch.zeros(3)}

        AtomicCheckpointWriter.save_torch_payload_atomic(path, payload)

        files = list(Path(self.tmpdir.name).glob("*"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, "clean.pt")

    def test_atomic_write_creates_parent_dirs(self) -> None:
        path = Path(self.tmpdir.name) / "sub" / "dir" / "test.pt"
        payload = {"x": torch.ones(2)}

        AtomicCheckpointWriter.save_torch_payload_atomic(path, payload)

        self.assertTrue(path.exists())

    def test_save_model_state_dict_atomic(self) -> None:
        model = build_test_model()
        path = Path(self.tmpdir.name) / "model.pt"

        AtomicCheckpointWriter.save_model_state_dict_atomic(
            path, model=model, checkpoint_type="best_model"
        )

        loaded = torch.load(path, weights_only=True)
        self.assertEqual(loaded["checkpoint_type"], "best_model")
        self.assertIn("model_state_dict", loaded)
        self.assertEqual(loaded["schema_version"], "1.0")

    def test_save_model_state_dict_with_ema(self) -> None:
        model = build_test_model()
        ema_state = {k: v.clone() for k, v in model.state_dict().items()}
        path = Path(self.tmpdir.name) / "ema.pt"

        AtomicCheckpointWriter.save_model_state_dict_atomic(
            path, model=model, ema_model_state=ema_state
        )

        loaded = torch.load(path, weights_only=True)
        self.assertIn("ema_model_state_dict", loaded)

    def test_durable_write_does_not_raise_on_this_platform(self) -> None:
        """Regression: the fsync durability path must work cross-platform.

        Before the fix, save_torch_payload_atomic() reopened the temp file
        read-only ("rb") to fsync and unconditionally fsync'd the parent
        directory. On Windows the first raised OSError(EBADF, "Bad file
        descriptor") and the second OSError(EACCES); both aborted the save.
        This asserts the full durable path completes and yields a loadable
        file on whatever platform the suite runs on.
        """
        path = Path(self.tmpdir.name) / "durable.pt"
        payload = {"w": torch.arange(64, dtype=torch.float32)}

        # Must NOT raise (the bug surfaced here as OSError).
        result = AtomicCheckpointWriter.save_torch_payload_atomic(path, payload)

        self.assertTrue(Path(result).exists())
        loaded = torch.load(result, weights_only=True)
        self.assertTrue(torch.equal(loaded["w"], payload["w"]))

    def test_durable_write_preserves_large_payload_integrity(self) -> None:
        """Content integrity after the flush+fsync+os.replace durability path.

        Larger tensors are more likely to exercise OS write buffering, so this
        guards against the durability path truncating or corrupting data while
        still being byte-faithful after the atomic rename.
        """
        path = Path(self.tmpdir.name) / "large.pt"
        payload = {
            "big": torch.randn(256, 256),
            "meta": torch.tensor([1, 2, 3, 4, 5]),
        }

        AtomicCheckpointWriter.save_torch_payload_atomic(path, payload)

        loaded = torch.load(path, weights_only=True)
        self.assertTrue(torch.equal(loaded["big"], payload["big"]))
        self.assertTrue(torch.equal(loaded["meta"], payload["meta"]))


# ------------------------------------------------------------------
# Tests: CheckpointIntegrationManager
# ------------------------------------------------------------------


class TestCheckpointIntegrationManager(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name) / "checkpoints"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _make_manager(self, **kwargs: Any) -> CheckpointIntegrationManager:
        return CheckpointIntegrationManager(self.root, **kwargs)

    # --- Best model ---

    def test_save_best_model_first_always_saves(self) -> None:
        mgr = self._make_manager()
        model = build_test_model()

        report = mgr.save_best_model_checkpoint(
            model=model, metric_value=1.0, metric_name="val_loss", epoch=0
        )

        self.assertTrue(report["saved"])
        self.assertEqual(report["reason"], "improved")
        self.assertEqual(report["checkpoint_semantics"], "best_model")
        self.assertTrue(Path(mgr.best_model_path).exists())

    def test_save_best_model_min_delta_blocks_noise(self) -> None:
        mgr = self._make_manager(
            best_policy=BestCheckpointPolicy(mode="min", min_delta=0.01)
        )
        model = build_test_model()

        # First save always succeeds
        mgr.save_best_model_checkpoint(
            model=model, metric_value=1.0, metric_name="val_loss", epoch=0
        )

        # Noise-level improvement blocked
        report = mgr.save_best_model_checkpoint(
            model=model, metric_value=0.999, metric_name="val_loss", epoch=1
        )
        self.assertFalse(report["saved"])
        self.assertEqual(report["reason"], "not_improved")

        # Real improvement saves
        report = mgr.save_best_model_checkpoint(
            model=model, metric_value=0.98, metric_name="val_loss", epoch=2
        )
        self.assertTrue(report["saved"])

    def test_load_best_model_roundtrip(self) -> None:
        mgr = self._make_manager()
        src_model = build_test_model()
        optimizer = build_test_optimizer(src_model)
        run_one_step(src_model, optimizer)

        mgr.save_best_model_checkpoint(
            model=src_model, metric_value=0.5, metric_name="val_loss", epoch=3
        )

        dst_model = build_test_model()
        report = mgr.load_best_model_checkpoint(model=dst_model, strict=True)

        self.assertEqual(report["checkpoint_semantics"], "best_model")
        self.assertEqual(report["load_path_type"], "safe")
        self.assertEqual(len(report["model_load_result"]["missing_keys"]), 0)
        self.assertEqual(len(report["model_load_result"]["unexpected_keys"]), 0)

        # Verify weights match
        for key in src_model.state_dict():
            self.assertTrue(
                torch.equal(src_model.state_dict()[key], dst_model.state_dict()[key]),
                msg=f"Mismatch in {key}",
            )

    def test_load_best_model_with_ema(self) -> None:
        mgr = self._make_manager()
        model = build_test_model()
        ema_state = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}

        mgr.save_best_model_checkpoint(
            model=model,
            metric_value=0.5,
            metric_name="val_loss",
            epoch=1,
            ema_model_state=ema_state,
        )

        dst_model = build_test_model()
        report = mgr.load_best_model_checkpoint(
            model=dst_model, use_ema_if_available=True
        )

        self.assertTrue(report["used_ema"])
        # EMA state was all zeros
        for key in dst_model.state_dict():
            self.assertTrue(
                torch.all(dst_model.state_dict()[key] == 0),
                msg=f"EMA not loaded for {key}",
            )

    def test_best_model_loads_with_weights_only_true(self) -> None:
        """Best-model must always load via safe path."""
        mgr = self._make_manager()
        model = build_test_model()

        mgr.save_best_model_checkpoint(
            model=model, metric_value=0.5, metric_name="val_loss", epoch=0
        )

        # Verify the file can be loaded with weights_only=True
        payload = torch.load(mgr.best_model_path, weights_only=True)
        self.assertIsInstance(payload, dict)
        self.assertIn("model_state_dict", payload)

    # --- Last model ---

    def test_save_last_model(self) -> None:
        mgr = self._make_manager()
        model = build_test_model()

        report = mgr.save_last_model_checkpoint(model=model, epoch=5, global_step=100)

        self.assertTrue(report["saved"])
        self.assertEqual(report["checkpoint_semantics"], "last_model_only")
        self.assertTrue(Path(mgr.last_model_path).exists())

        # Verify checkpoint_type
        loaded = torch.load(mgr.last_model_path, weights_only=True)
        self.assertEqual(loaded["checkpoint_type"], "last_model_only")

    # --- Resume ---

    def test_save_and_load_resume_roundtrip(self) -> None:
        mgr = self._make_manager()
        model = build_test_model()
        optimizer = build_test_optimizer(model)
        scheduler = build_test_scheduler(optimizer)
        run_one_step(model, optimizer)
        scheduler.step()

        save_report = mgr.save_resume(
            model=model,
            optimizer=optimizer,
            epoch=3,
            global_step=30,
            scheduler=scheduler,
            include_python_rng=False,
            include_numpy_rng=False,
            include_torch_rng=True,
        )

        self.assertTrue(save_report["saved"])
        self.assertEqual(save_report["checkpoint_semantics"], "resume_full")
        self.assertFalse(save_report["requires_trusted_load"])

        # Load into new model/optimizer/scheduler
        dst_model = build_test_model()
        dst_optimizer = build_test_optimizer(dst_model)
        dst_scheduler = build_test_scheduler(dst_optimizer)

        load_report = mgr.load_resume(
            model=dst_model,
            optimizer=dst_optimizer,
            scheduler=dst_scheduler,
            restore_rng=False,
            trusted=False,
        )

        self.assertEqual(load_report["epoch"], 3)
        self.assertEqual(load_report["global_step"], 30)
        self.assertTrue(load_report["optimizer_restored"])
        self.assertTrue(load_report["scheduler_restored"])

        # Verify model weights match
        for key in model.state_dict():
            self.assertTrue(
                torch.equal(model.state_dict()[key], dst_model.state_dict()[key]),
                msg=f"Mismatch in {key}",
            )

    def test_load_resume_explicit_path_overrides_manager_default(self) -> None:
        """The existence check and actual load must refer to the same file."""
        mgr = self._make_manager()
        model = build_test_model()
        optimizer = build_test_optimizer(model)
        run_one_step(model, optimizer)

        # Manager-owned checkpoint intentionally carries a different epoch.
        mgr.save_resume(
            model=model,
            optimizer=optimizer,
            epoch=1,
            include_python_rng=False,
            include_numpy_rng=False,
        )

        foreign = Path(self.tmpdir.name) / "foreign" / "resume.pt"
        save_resume_checkpoint(
            foreign,
            model=model,
            optimizer=optimizer,
            epoch=9,
            include_python_rng=False,
            include_numpy_rng=False,
        )

        dst_model = build_test_model()
        dst_optimizer = build_test_optimizer(dst_model)
        report = mgr.load_resume(
            model=dst_model,
            optimizer=dst_optimizer,
            path=foreign,
            restore_rng=False,
        )
        self.assertEqual(report["epoch"], 9)
        self.assertEqual(report["checkpoint_path"], str(foreign))

    def test_resume_with_rng_requires_trusted(self) -> None:
        """Resume with Python/NumPy RNG needs trusted=True."""
        mgr = self._make_manager()
        model = build_test_model()
        optimizer = build_test_optimizer(model)

        save_report = mgr.save_resume(
            model=model,
            optimizer=optimizer,
            epoch=1,
            include_python_rng=True,
            include_numpy_rng=True,
        )
        self.assertTrue(save_report["requires_trusted_load"])

        # Safe path should fail
        dst_model = build_test_model()
        dst_optimizer = build_test_optimizer(dst_model)
        with self.assertRaises(Exception):
            mgr.load_resume(
                model=dst_model, optimizer=dst_optimizer, trusted=False
            )

        # Trusted path should succeed
        dst_model2 = build_test_model()
        dst_optimizer2 = build_test_optimizer(dst_model2)
        report = mgr.load_resume(
            model=dst_model2, optimizer=dst_optimizer2, trusted=True
        )
        self.assertTrue(report["trusted"])
        self.assertEqual(report["epoch"], 1)

    # --- Warmstart ---

    def test_warmstart_does_not_restore_optimizer(self) -> None:
        """Warmstart must NEVER touch optimizer state."""
        mgr = self._make_manager()
        model = build_test_model()
        optimizer = build_test_optimizer(model)

        # Train one step to change optimizer state
        run_one_step(model, optimizer)

        # Save best model (only weights)
        mgr.save_best_model_checkpoint(
            model=model, metric_value=0.5, metric_name="val_loss", epoch=1
        )

        # Create fresh model + optimizer
        dst_model = build_test_model()
        dst_optimizer = build_test_optimizer(dst_model)

        # Capture optimizer state BEFORE warmstart
        opt_state_before = {
            k: v for k, v in dst_optimizer.state_dict().items()
        }

        report = mgr.load_warmstart(
            model=dst_model, checkpoint_path=mgr.best_model_path
        )

        self.assertEqual(report["action"], "warmstart")
        self.assertEqual(report["checkpoint_semantics"], "warmstart")

        # Optimizer state should be unchanged
        opt_state_after = dst_optimizer.state_dict()
        self.assertEqual(
            opt_state_before["param_groups"],
            opt_state_after["param_groups"],
        )

    # --- DDP guard ---

    def test_is_main_process_false_skips_saves(self) -> None:
        mgr = self._make_manager(is_main_process=False)
        model = build_test_model()
        optimizer = build_test_optimizer(model)

        best_report = mgr.save_best_model_checkpoint(
            model=model, metric_value=0.5, metric_name="val_loss", epoch=0
        )
        self.assertFalse(best_report["saved"])

        last_report = mgr.save_last_model_checkpoint(model=model, epoch=0)
        self.assertFalse(last_report["saved"])

        resume_report = mgr.save_resume(
            model=model, optimizer=optimizer, epoch=0
        )
        self.assertFalse(resume_report["saved"])

        # No files created
        self.assertFalse(Path(mgr.best_model_path).exists())
        self.assertFalse(Path(mgr.last_model_path).exists())
        self.assertFalse(Path(mgr.resume_path).exists())

    # --- Export state ---

    def test_export_state(self) -> None:
        mgr = self._make_manager()
        state = mgr.export_state()

        self.assertIn("root_dir", state)
        self.assertIn("best_policy", state)
        self.assertIn("paths", state)
        self.assertIn("best_model_path", state["paths"])
        self.assertIn("last_model_path", state["paths"])
        self.assertIn("resume_path", state["paths"])

    # --- Missing checkpoint errors ---

    def test_load_best_model_missing_raises(self) -> None:
        mgr = self._make_manager()
        model = build_test_model()

        with self.assertRaises(FileNotFoundError):
            mgr.load_best_model_checkpoint(model=model)

    def test_load_resume_missing_raises(self) -> None:
        mgr = self._make_manager()
        model = build_test_model()
        optimizer = build_test_optimizer(model)

        with self.assertRaises(FileNotFoundError):
            mgr.load_resume(model=model, optimizer=optimizer)

    # --- checkpoint_semantics consistency ---

    def test_all_save_reports_contain_checkpoint_semantics(self) -> None:
        mgr = self._make_manager()
        model = build_test_model()
        optimizer = build_test_optimizer(model)

        best_report = mgr.save_best_model_checkpoint(
            model=model, metric_value=0.5, metric_name="val_loss", epoch=0
        )
        last_report = mgr.save_last_model_checkpoint(model=model, epoch=0)
        resume_report = mgr.save_resume(
            model=model, optimizer=optimizer, epoch=0,
            include_python_rng=False, include_numpy_rng=False,
        )

        self.assertEqual(best_report["checkpoint_semantics"], "best_model")
        self.assertEqual(last_report["checkpoint_semantics"], "last_model_only")
        self.assertEqual(resume_report["checkpoint_semantics"], "resume_full")


# ------------------------------------------------------------------
# Tests: best-metric input contract + state ordering (issue #37)
# ------------------------------------------------------------------


class TestBestMetricContract(unittest.TestCase):
    """save_best_model_checkpoint validates before it changes any state.

    Formerly a NaN metric was saved as the best model (saved=True) and then
    blocked every later improvement (NaN comparisons are always False), and
    best_metric_value was assigned before float()/int() conversion and before
    the write, so a rejected or failed save still moved the best value.
    """

    METRIC_MESSAGE = r"^metric_value must be a finite real number, got "

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name) / "checkpoints"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _make_manager(self, **kwargs: Any) -> CheckpointIntegrationManager:
        return CheckpointIntegrationManager(self.root, **kwargs)

    def _saved_metric(self, mgr: CheckpointIntegrationManager) -> float:
        payload = torch.load(mgr.best_model_path, weights_only=True)
        return payload["extra"]["metric_value"]

    def test_non_finite_metric_is_rejected_before_any_state_change(self) -> None:
        for mode in ("min", "max"):
            for case, metric in enumerate(
                (
                    float("nan"),
                    float("inf"),
                    float("-inf"),
                    torch.tensor(float("nan")),
                )
            ):
                with self.subTest(mode=mode, metric=repr(metric)):
                    mgr = CheckpointIntegrationManager(
                        self.root / f"{mode}_{case}",
                        best_policy=BestCheckpointPolicy(mode=mode),
                    )
                    # Capture any exception and check its exact type, so the
                    # pre-fix saved=True return is an assertion failure.
                    with self.assertRaises(Exception) as ctx:
                        mgr.save_best_model_checkpoint(
                            model=build_test_model(),
                            metric_value=metric,
                            metric_name="val_loss",
                            epoch=0,
                        )
                    exc = ctx.exception
                    self.assertIs(
                        type(exc),
                        ValueError,
                        f"expected ValueError, got {type(exc).__name__}: {exc}",
                    )
                    self.assertRegex(str(exc), self.METRIC_MESSAGE)
                    self.assertIsNone(mgr.best_metric_value)
                    self.assertFalse(Path(mgr.best_model_path).exists())

    def test_nan_first_metric_does_not_block_later_best(self) -> None:
        mgr = self._make_manager()
        model = build_test_model()
        with self.assertRaises(ValueError):
            mgr.save_best_model_checkpoint(
                model=model, metric_value=float("nan"), metric_name="val_loss", epoch=0
            )
        report = mgr.save_best_model_checkpoint(
            model=model, metric_value=1.0, metric_name="val_loss", epoch=1
        )
        self.assertTrue(report["saved"])
        self.assertEqual(mgr.best_metric_value, 1.0)
        self.assertEqual(self._saved_metric(mgr), 1.0)

    def test_metric_beyond_binary64_leaves_best_unchanged(self) -> None:
        # mode="max": 10**400 compares as an improvement over 0.5, so the
        # pre-fix code assigned it as best before float() overflowed.
        for metric in (10**400, Fraction(10**400)):
            with self.subTest(metric_bits=int(metric).bit_length(), kind=type(metric).__name__):
                mgr = self._make_manager(best_policy=BestCheckpointPolicy(mode="max"))
                model = build_test_model()
                mgr.save_best_model_checkpoint(
                    model=model, metric_value=0.5, metric_name="score", epoch=0
                )
                with self.assertRaises(Exception) as ctx:
                    mgr.save_best_model_checkpoint(
                        model=model, metric_value=metric, metric_name="score", epoch=1
                    )
                exc = ctx.exception
                self.assertIs(
                    type(exc),
                    ValueError,
                    f"expected ValueError, got {type(exc).__name__}: {exc}",
                )
                self.assertRegex(str(exc), self.METRIC_MESSAGE)
                self.assertIsInstance(exc.__cause__, OverflowError)
                self.assertEqual(mgr.best_metric_value, 0.5)
                self.assertEqual(self._saved_metric(mgr), 0.5)

    def test_non_main_process_rejects_non_finite_metric(self) -> None:
        # Every DDP rank must apply the same contract, otherwise rank 0 and
        # the other ranks disagree on the best value.
        mgr = self._make_manager(is_main_process=False)
        with self.assertRaises(ValueError):
            mgr.save_best_model_checkpoint(
                model=build_test_model(),
                metric_value=float("nan"),
                metric_name="val_loss",
                epoch=0,
            )
        self.assertIsNone(mgr.best_metric_value)
        # A valid improvement is still tracked (not saved) on this rank.
        report = mgr.save_best_model_checkpoint(
            model=build_test_model(),
            metric_value=0.5,
            metric_name="val_loss",
            epoch=1,
        )
        self.assertEqual(report["reason"], "not_main_process")
        self.assertEqual(mgr.best_metric_value, 0.5)
        self.assertFalse(Path(mgr.best_model_path).exists())

    def test_invalid_epoch_leaves_best_unchanged(self) -> None:
        mgr = self._make_manager()
        model = build_test_model()
        mgr.save_best_model_checkpoint(
            model=model, metric_value=1.0, metric_name="val_loss", epoch=0
        )
        with self.assertRaises(TypeError):
            mgr.save_best_model_checkpoint(
                model=model, metric_value=0.5, metric_name="val_loss", epoch=None
            )
        self.assertEqual(mgr.best_metric_value, 1.0)
        self.assertEqual(self._saved_metric(mgr), 1.0)

    def test_failed_write_leaves_best_unchanged(self) -> None:
        mgr = self._make_manager()
        model = build_test_model()
        mgr.save_best_model_checkpoint(
            model=model, metric_value=1.0, metric_name="val_loss", epoch=0
        )
        with mock.patch.object(
            AtomicCheckpointWriter,
            "save_model_state_dict_atomic",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(OSError):
                mgr.save_best_model_checkpoint(
                    model=model, metric_value=0.5, metric_name="val_loss", epoch=1
                )
        self.assertEqual(mgr.best_metric_value, 1.0)
        self.assertEqual(self._saved_metric(mgr), 1.0)
        # The same improvement is accepted once the write succeeds.
        report = mgr.save_best_model_checkpoint(
            model=model, metric_value=0.5, metric_name="val_loss", epoch=2
        )
        self.assertTrue(report["saved"])
        self.assertEqual(mgr.best_metric_value, 0.5)

    def test_valid_metric_types_are_normalized_to_float(self) -> None:
        for metric, expected in (
            (0.25, 0.25),
            (3, 3.0),
            (Fraction(1, 4), 0.25),
            (torch.tensor(0.25), 0.25),
        ):
            with self.subTest(metric=repr(metric)):
                root = self.root / type(metric).__name__
                mgr = CheckpointIntegrationManager(root)
                report = mgr.save_best_model_checkpoint(
                    model=build_test_model(),
                    metric_value=metric,
                    metric_name="val_loss",
                    epoch=0,
                )
                self.assertTrue(report["saved"])
                self.assertIs(type(mgr.best_metric_value), float)
                self.assertEqual(mgr.best_metric_value, expected)
                self.assertEqual(report["best_metric_value"], expected)


# ------------------------------------------------------------------
# Tests: FailureSignature (basic integration)
# ------------------------------------------------------------------


class TestFailureSignatureIntegration(unittest.TestCase):
    """Test that FailureSignature can be imported and used."""

    def test_failure_signature_roundtrip(self) -> None:
        from failure_signature_hex_v3 import FailureSignature

        with tempfile.TemporaryDirectory() as tmpdir:
            sig = FailureSignature(
                kind="NaN_or_Inf_Loss",
                step=42,
                locus="test",
                observable="loss=nan",
            )

            path = sig.register(tmpdir)
            self.assertTrue(os.path.exists(path))

            loaded = FailureSignature.from_json(path)
            self.assertEqual(loaded.kind, "NaN_or_Inf_Loss")
            self.assertEqual(loaded.step, 42)



# ------------------------------------------------------------------
# Issue #40 follow-ups: policy, counters, non-main rank, durability
# ------------------------------------------------------------------


class _TmpRootCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name) / "checkpoints"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _make_manager(self, root: Path | None = None, **kwargs: Any) -> CheckpointIntegrationManager:
        return CheckpointIntegrationManager(root or self.root, **kwargs)

    @staticmethod
    def _disk_metric(mgr: CheckpointIntegrationManager) -> float:
        return torch.load(mgr.best_model_path, weights_only=True)["extra"]["metric_value"]


class TestMinDeltaContract(unittest.TestCase):
    def test_non_finite_min_delta_rejected(self) -> None:
        # NaN passed "< 0" and then blocked every improvement.
        for value in (float("nan"), float("inf"), -float("inf"), -1e-9):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    BestCheckpointPolicy(min_delta=value)

    def test_non_numeric_min_delta_rejected(self) -> None:
        for value in ("0.1", None, True):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    BestCheckpointPolicy(min_delta=value)

    def test_valid_min_delta_values_accepted(self) -> None:
        for value in (0, 0.0, 1e-4, np.float32(0.5), Fraction(1, 10)):
            with self.subTest(value=repr(value)):
                self.assertTrue(BestCheckpointPolicy(min_delta=value).is_improved(0.0, 1.0))


class TestStepCounterContract(_TmpRootCase):
    """int(epoch)/int(global_step) raised raw OverflowError on inf and
    truncated 1.5 silently (issue #40); now a contract error, before any write."""

    BAD_VALUES = (float("inf"), float("nan"), 1.5, -1, 2.5)

    def _save_calls(self, mgr: CheckpointIntegrationManager, model: nn.Module, **kw: Any):
        optimizer = build_test_optimizer(model)
        return {
            "last": lambda: mgr.save_last_model_checkpoint(model=model, **kw),
            "resume": lambda: mgr.save_resume(
                model=model, optimizer=optimizer,
                include_python_rng=False, include_numpy_rng=False, **kw,
            ),
            "best": lambda: mgr.save_best_model_checkpoint(
                model=model, metric_value=0.5, metric_name="v", **kw
            ),
        }

    def test_invalid_epoch_and_step_raise_value_error_without_writing(self) -> None:
        model = build_test_model()
        for field in ("epoch", "global_step"):
            for bad in self.BAD_VALUES:
                kw = {"epoch": 0, "global_step": 0}
                kw[field] = bad
                mgr = self._make_manager()
                for name, call in self._save_calls(mgr, model, **kw).items():
                    with self.subTest(field=field, value=bad, save=name):
                        with self.assertRaises(Exception) as ctx:
                            call()
                        self.assertIs(type(ctx.exception), ValueError)
                        self.assertIn(f"{field} must be a non-negative integer", str(ctx.exception))
                self.assertEqual(sorted(p.name for p in self.root.iterdir()), [])
                self.assertIsNone(mgr.best_metric_value)

    def test_bool_and_non_numeric_counters_raise_type_error(self) -> None:
        model = build_test_model()
        mgr = self._make_manager()
        for bad in (True, "3", None):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(TypeError):
                    mgr.save_last_model_checkpoint(model=model, epoch=bad)

    def test_integral_counters_are_normalised_to_int(self) -> None:
        model = build_test_model()
        mgr = self._make_manager()
        for epoch, step in ((2.0, 5.0), (np.int64(2), np.int32(5)), (torch.tensor(2), Fraction(10, 2))):
            with self.subTest(epoch=repr(epoch)):
                report = mgr.save_last_model_checkpoint(model=model, epoch=epoch, global_step=step)
                self.assertEqual((report["epoch"], report["global_step"]), (2, 5))
                self.assertIs(type(report["epoch"]), int)
                extra = torch.load(mgr.last_model_path, weights_only=True)["extra"]
                self.assertEqual((extra["epoch"], extra["global_step"]), (2, 5))

    def test_non_main_rank_validates_counters_too(self) -> None:
        # All DDP ranks must accept/reject the same inputs.
        mgr = self._make_manager(is_main_process=False)
        with self.assertRaises(ValueError):
            mgr.save_last_model_checkpoint(model=build_test_model(), epoch=1.5)


class TestNonMainRankStoresValidatedFloat(_TmpRootCase):
    def test_non_main_rank_best_value_is_the_validated_float(self) -> None:
        # P3-2: all 37 checkpoint tests stayed green when non-main ranks
        # stored the raw metric object; pin the converted value and type.
        for raw in (torch.tensor(0.5), np.float32(0.5), Fraction(1, 2), 1):
            with self.subTest(raw=repr(raw)):
                mgr = self._make_manager(is_main_process=False)
                mgr.save_best_model_checkpoint(
                    model=build_test_model(), metric_value=raw, metric_name="v", epoch=0
                )
                self.assertIs(type(mgr.best_metric_value), float)
                self.assertEqual(mgr.best_metric_value, float(raw))


def _fail_directory_fsync(real_fsync):
    def fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(5, "simulated directory fsync failure")
        real_fsync(fd)
    return fsync


@unittest.skipUnless(os.name == "posix", "directory fsync is POSIX-only")
class TestDurabilityErrorAfterReplace(_TmpRootCase):
    """P3-3: an exception after os.replace left best.pt newer than memory."""

    def test_writer_reports_replaced_file(self) -> None:
        target = self.root / "x.pt"
        with mock.patch("os.fsync", _fail_directory_fsync(os.fsync)):
            with self.assertRaises(CheckpointDurabilityError) as ctx:
                AtomicCheckpointWriter.save_torch_payload_atomic(target, {"a": torch.ones(1)})
        self.assertIsInstance(ctx.exception, OSError)
        self.assertTrue(ctx.exception.replaced)
        self.assertEqual(ctx.exception.path, str(target))
        self.assertIsInstance(ctx.exception.__cause__, OSError)
        self.assertTrue(torch.equal(torch.load(target, weights_only=True)["a"], torch.ones(1)))
        self.assertEqual([p.name for p in self.root.iterdir()], ["x.pt"])

    def test_best_value_matches_file_after_directory_fsync_failure(self) -> None:
        mgr = self._make_manager()
        model = build_test_model()
        mgr.save_best_model_checkpoint(model=model, metric_value=1.0, metric_name="v", epoch=0)
        with mock.patch("os.fsync", _fail_directory_fsync(os.fsync)):
            with self.assertRaises(CheckpointDurabilityError):
                mgr.save_best_model_checkpoint(model=model, metric_value=0.5, metric_name="v", epoch=1)
        self.assertEqual(self._disk_metric(mgr), 0.5)
        self.assertEqual(mgr.best_metric_value, 0.5)
        # A caller that catches the error and continues must not overwrite
        # the better model with a worse one.
        report = mgr.save_best_model_checkpoint(model=model, metric_value=0.7, metric_name="v", epoch=2)
        self.assertFalse(report["saved"])
        self.assertEqual(self._disk_metric(mgr), 0.5)

    def test_failure_before_replace_is_not_a_durability_error(self) -> None:
        mgr = self._make_manager()
        with mock.patch("torch.save", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError) as ctx:
                mgr.save_best_model_checkpoint(
                    model=build_test_model(), metric_value=0.5, metric_name="v", epoch=0
                )
        self.assertNotIsInstance(ctx.exception, CheckpointDurabilityError)
        self.assertIsNone(mgr.best_metric_value)


# ------------------------------------------------------------------
# Resume restores best metric (issue #40) and EMA state
# ------------------------------------------------------------------


class TestResumeRestoresBestMetric(_TmpRootCase):
    def _run_with_best(self, mgr: CheckpointIntegrationManager, *metrics: float):
        model = build_test_model()
        optimizer = build_test_optimizer(model)
        for epoch, metric in enumerate(metrics):
            mgr.save_best_model_checkpoint(model=model, metric_value=metric, metric_name="v", epoch=epoch)
        mgr.save_resume(model=model, optimizer=optimizer, epoch=len(metrics) - 1,
                        include_python_rng=False, include_numpy_rng=False)
        return model

    def _resume(self, mgr: CheckpointIntegrationManager, path: Any = None) -> dict:
        model = build_test_model()
        return mgr.load_resume(model=model, optimizer=build_test_optimizer(model), path=path)

    def test_best_value_restored_and_protects_best_file(self) -> None:
        self._run_with_best(self._make_manager(), 1.0, 0.4, 0.6)
        fresh = self._make_manager()
        report = self._resume(fresh)
        self.assertTrue(report["best_metric_restored"])
        self.assertEqual(report["best_metric_restore_reason"], "restored")
        self.assertEqual(fresh.best_metric_value, 0.4)
        # Before the fix best was None after resume, so 0.6 replaced 0.4.
        after = fresh.save_best_model_checkpoint(
            model=build_test_model(), metric_value=0.6, metric_name="v", epoch=3
        )
        self.assertFalse(after["saved"])
        self.assertEqual(self._disk_metric(fresh), 0.4)

    def test_not_restored_when_best_file_missing(self) -> None:
        mgr = self._make_manager()
        self._run_with_best(mgr, 0.4)
        os.remove(mgr.best_model_path)
        fresh = self._make_manager()
        report = self._resume(fresh)
        self.assertFalse(report["best_metric_restored"])
        self.assertEqual(report["best_metric_restore_reason"], "best_checkpoint_missing")
        self.assertIsNone(fresh.best_metric_value)

    def test_not_restored_when_best_file_holds_other_value(self) -> None:
        mgr = self._make_manager()
        self._run_with_best(mgr, 0.4)
        other = self._make_manager(Path(self.tmpdir.name) / "other")
        other.save_best_model_checkpoint(model=build_test_model(), metric_value=0.9, metric_name="v", epoch=0)
        os.replace(other.best_model_path, mgr.best_model_path)
        report = self._resume(self._make_manager())
        self.assertEqual(report["best_metric_restore_reason"], "best_checkpoint_mismatch")

    def test_newer_better_best_from_same_run_is_adopted(self) -> None:
        # best.pt is written every epoch, resume only periodically: a later
        # epoch may have improved best.pt after the last resume save.
        mgr = self._make_manager()
        model = self._run_with_best(mgr, 1.0, 0.5)
        mgr.save_best_model_checkpoint(model=model, metric_value=0.3, metric_name="v", epoch=2)
        fresh = self._make_manager()
        report = self._resume(fresh)
        self.assertTrue(report["best_metric_restored"])
        self.assertEqual(report["best_metric_restore_reason"], "restored_from_newer_best")
        self.assertEqual(fresh.best_metric_value, 0.3)
        again = fresh.save_best_model_checkpoint(
            model=build_test_model(), metric_value=0.4, metric_name="v", epoch=3
        )
        self.assertFalse(again["saved"])
        self.assertEqual(self._disk_metric(fresh), 0.3)

    def test_newer_best_rule_respects_max_mode(self) -> None:
        policy = BestCheckpointPolicy(mode="max")
        mgr = self._make_manager(best_policy=policy)
        model = self._run_with_best(mgr, 0.2, 0.6)
        mgr.save_best_model_checkpoint(model=model, metric_value=0.8, metric_name="v", epoch=2)
        fresh = self._make_manager(best_policy=BestCheckpointPolicy(mode="max"))
        self.assertEqual(self._resume(fresh)["best_metric_restore_reason"], "restored_from_newer_best")
        self.assertEqual(fresh.best_metric_value, 0.8)

    def test_not_restored_on_policy_mode_mismatch(self) -> None:
        self._run_with_best(self._make_manager(), 0.4)
        fresh = self._make_manager(best_policy=BestCheckpointPolicy(mode="max"))
        report = self._resume(fresh)
        self.assertEqual(report["best_metric_restore_reason"], "policy_mode_mismatch")
        self.assertIsNone(fresh.best_metric_value)

    def test_foreign_resume_into_fresh_dir_does_not_block_first_best(self) -> None:
        self._run_with_best(self._make_manager(), 0.1)
        foreign = self.root / "resume_last.pt"
        fresh = self._make_manager(Path(self.tmpdir.name) / "new_run")
        report = self._resume(fresh, path=foreign)
        self.assertEqual(report["best_metric_restore_reason"], "best_checkpoint_missing")
        saved = fresh.save_best_model_checkpoint(
            model=build_test_model(), metric_value=0.5, metric_name="v", epoch=1
        )
        self.assertTrue(saved["saved"])

    def test_legacy_resume_without_record(self) -> None:
        model = build_test_model()
        save_resume_checkpoint(self.root / "resume_last.pt", model=model,
                               optimizer=build_test_optimizer(model), epoch=0,
                               include_python_rng=False, include_numpy_rng=False)
        report = self._resume(self._make_manager())
        self.assertEqual(report["best_metric_restore_reason"], "absent")
        self.assertFalse(report["best_metric_restored"])

    def test_resume_without_any_best_yet(self) -> None:
        mgr = self._make_manager()
        model = build_test_model()
        mgr.save_resume(model=model, optimizer=build_test_optimizer(model), epoch=0,
                        include_python_rng=False, include_numpy_rng=False)
        report = self._resume(self._make_manager())
        self.assertEqual(report["best_metric_restore_reason"], "no_best_recorded")

    def test_resume_payload_stays_weights_only_loadable(self) -> None:
        self._run_with_best(self._make_manager(), 0.4)
        payload = torch.load(self.root / "resume_last.pt", weights_only=True)
        self.assertEqual(
            payload["checkpoint_manager_state"],
            {"best_metric_value": 0.4, "best_policy_mode": "min"},
        )


class TestResumeEmaState(_TmpRootCase):
    @staticmethod
    def _ema(model: nn.Module) -> torch.optim.swa_utils.AveragedModel:
        return torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9), use_buffers=True
        )

    def test_ema_state_roundtrip_through_resume(self) -> None:
        torch.manual_seed(0)
        model = build_test_model()
        optimizer = build_test_optimizer(model)
        ema = self._ema(model)
        for _ in range(3):
            run_one_step(model, optimizer)
            ema.update_parameters(model)
        mgr = self._make_manager()
        report = mgr.save_resume(model=model, optimizer=optimizer, epoch=0,
                                 ema_state_dict=ema.state_dict(),
                                 include_python_rng=False, include_numpy_rng=False)
        self.assertTrue(report["has_ema_state"])

        torch.manual_seed(1)
        model2 = build_test_model()
        ema2 = self._ema(model2)
        loaded = mgr.load_resume(model=model2, optimizer=build_test_optimizer(model2), ema_model=ema2)
        self.assertTrue(loaded["ema_restored"])
        self.assertEqual(int(ema2.n_averaged.item()), 3)
        for key, value in ema.state_dict().items():
            self.assertTrue(torch.equal(value, ema2.state_dict()[key]), msg=key)

    def test_missing_ema_state_is_reported(self) -> None:
        model = build_test_model()
        mgr = self._make_manager()
        mgr.save_resume(model=model, optimizer=build_test_optimizer(model), epoch=0,
                        include_python_rng=False, include_numpy_rng=False)
        loaded = mgr.load_resume(model=model, optimizer=build_test_optimizer(model),
                                 ema_model=self._ema(model))
        self.assertFalse(loaded["has_ema_state"])
        self.assertFalse(loaded["ema_restored"])


if __name__ == "__main__":
    unittest.main()
