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
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from checkpoint_integration_hex_v3 import (
    AtomicCheckpointWriter,
    BestCheckpointPolicy,
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


if __name__ == "__main__":
    unittest.main()
