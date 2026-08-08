"""
test_resume_utils_hex_v3.py
===========================
Tests for training resume checkpoint utilities.

Covers:
- build_resume_payload composition (optional fields, RNG include flags)
- save_resume_checkpoint report + requires_trusted_load contract
- restore_rng_states determinism (python/numpy/torch)
- load_resume_checkpoint safe vs trusted paths, validation, restore order side effects
"""
from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from resume_utils_hex_v3 import (
    build_resume_payload,
    load_resume_checkpoint,
    restore_rng_states,
    save_resume_checkpoint,
)


def _make_training_state() -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # One real step so optimizer state is non-trivial.
    loss = model(torch.randn(8, 4)).sum()
    loss.backward()
    optimizer.step()
    return model, optimizer


class TestBuildResumePayload(unittest.TestCase):
    def test_required_fields(self) -> None:
        model, opt = _make_training_state()
        payload = build_resume_payload(model=model, optimizer=opt, epoch=3)
        self.assertEqual(payload["checkpoint_type"], "resume")
        self.assertEqual(payload["checkpoint_semantics"], "resume_full")
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["epoch"], 3)
        self.assertIn("model_state_dict", payload)
        self.assertIn("optimizer_state_dict", payload)

    def test_optional_fields_absent_by_default(self) -> None:
        model, opt = _make_training_state()
        payload = build_resume_payload(model=model, optimizer=opt, epoch=0)
        for key in (
            "global_step",
            "scheduler_state_dict",
            "grad_scaler_state_dict",
            "dataloader_state_dict",
            "extra",
        ):
            self.assertNotIn(key, payload)

    def test_optional_fields_present_when_given(self) -> None:
        model, opt = _make_training_state()
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
        scaler = torch.amp.GradScaler("cpu")
        payload = build_resume_payload(
            model=model,
            optimizer=opt,
            epoch=5,
            global_step=1234,
            scheduler=scheduler,
            grad_scaler=scaler,
            extra={"note": "hex"},
            include_dataloader_state={"batches_seen": 17},
        )
        self.assertEqual(payload["global_step"], 1234)
        self.assertIn("scheduler_state_dict", payload)
        self.assertIn("grad_scaler_state_dict", payload)
        self.assertEqual(payload["dataloader_state_dict"], {"batches_seen": 17})
        self.assertEqual(payload["extra"], {"note": "hex"})

    def test_rng_include_flags(self) -> None:
        model, opt = _make_training_state()
        payload = build_resume_payload(
            model=model,
            optimizer=opt,
            epoch=0,
            include_python_rng=False,
            include_numpy_rng=False,
            include_torch_rng=False,
        )
        self.assertNotIn("python_random_state", payload)
        self.assertNotIn("numpy_random_state", payload)
        self.assertNotIn("torch_random_state", payload)

        payload_full = build_resume_payload(model=model, optimizer=opt, epoch=0)
        self.assertIn("python_random_state", payload_full)
        self.assertIn("numpy_random_state", payload_full)
        self.assertIn("torch_random_state", payload_full)


class TestSaveResumeCheckpoint(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_save_report_and_file(self) -> None:
        model, opt = _make_training_state()
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
        path = self.tmpdir / "sub" / "resume.pt"
        report = save_resume_checkpoint(
            path,
            model=model,
            optimizer=opt,
            epoch=2,
            global_step=99,
            scheduler=scheduler,
        )
        self.assertTrue(path.exists())  # parent dir auto-created
        self.assertEqual(report["epoch"], 2)
        self.assertEqual(report["global_step"], 99)
        self.assertTrue(report["has_scheduler"])
        self.assertFalse(report["has_grad_scaler"])
        self.assertTrue(report["requires_trusted_load"])

    def test_requires_trusted_load_false_without_python_numpy_rng(self) -> None:
        model, opt = _make_training_state()
        path = self.tmpdir / "safe.pt"
        report = save_resume_checkpoint(
            path,
            model=model,
            optimizer=opt,
            epoch=0,
            include_python_rng=False,
            include_numpy_rng=False,
        )
        self.assertFalse(report["requires_trusted_load"])
        self.assertIsNone(report["global_step"])


class TestRestoreRngStates(unittest.TestCase):
    def test_python_rng_roundtrip(self) -> None:
        state = random.getstate()
        expected = random.random()
        restore_rng_states({"python_random_state": state})
        self.assertEqual(random.random(), expected)

    def test_numpy_rng_roundtrip(self) -> None:
        state = np.random.get_state()
        expected = np.random.rand()
        restore_rng_states({"numpy_random_state": state})
        self.assertEqual(np.random.rand(), expected)

    def test_torch_rng_roundtrip(self) -> None:
        state = torch.random.get_rng_state()
        expected = torch.rand(3)
        restore_rng_states({"torch_random_state": state})
        self.assertTrue(torch.equal(torch.rand(3), expected))

    def test_report_flags_reflect_payload(self) -> None:
        report = restore_rng_states({"torch_random_state": torch.random.get_rng_state()})
        self.assertTrue(report["torch_random_state"])
        self.assertFalse(report["python_random_state"])
        self.assertFalse(report["numpy_random_state"])

    def test_empty_payload_restores_nothing(self) -> None:
        report = restore_rng_states({})
        self.assertFalse(any(report.values()))


class TestLoadResumeCheckpoint(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _save_safe_checkpoint(self, **kwargs: object) -> tuple[Path, nn.Module, torch.optim.Optimizer]:
        """Checkpoint ohne python/numpy-RNG -> safe path (weights_only=True) laedt."""
        model, opt = _make_training_state()
        path = self.tmpdir / "resume.pt"
        save_resume_checkpoint(
            path,
            model=model,
            optimizer=opt,
            epoch=4,
            include_python_rng=False,
            include_numpy_rng=False,
            **kwargs,  # type: ignore[arg-type]
        )
        return path, model, opt

    def test_missing_file_raises(self) -> None:
        model, opt = _make_training_state()
        with self.assertRaises(FileNotFoundError):
            load_resume_checkpoint(self.tmpdir / "nope.pt", model=model, optimizer=opt)

    def test_non_dict_payload_rejected(self) -> None:
        path = self.tmpdir / "list.pt"
        torch.save([1, 2, 3], path)
        model, opt = _make_training_state()
        with self.assertRaises(TypeError):
            load_resume_checkpoint(path, model=model, optimizer=opt)

    def test_missing_model_state_rejected(self) -> None:
        path = self.tmpdir / "bad.pt"
        model, opt = _make_training_state()
        torch.save({"optimizer_state_dict": opt.state_dict(), "epoch": 0}, path)
        with self.assertRaises(ValueError):
            load_resume_checkpoint(path, model=model, optimizer=opt)

    def test_missing_optimizer_state_rejected(self) -> None:
        path = self.tmpdir / "bad.pt"
        model, opt = _make_training_state()
        torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, path)
        with self.assertRaises(ValueError):
            load_resume_checkpoint(path, model=model, optimizer=opt)

    def test_missing_epoch_rejected(self) -> None:
        path = self.tmpdir / "bad.pt"
        model, opt = _make_training_state()
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            },
            path,
        )
        with self.assertRaises(ValueError):
            load_resume_checkpoint(path, model=model, optimizer=opt)

    def test_safe_path_roundtrip(self) -> None:
        path, src_model, _ = self._save_safe_checkpoint()
        model, opt = _make_training_state()
        report = load_resume_checkpoint(path, model=model, optimizer=opt)
        self.assertEqual(report["epoch"], 4)
        self.assertFalse(report["trusted"])
        self.assertTrue(report["optimizer_restored"])
        self.assertFalse(report["scheduler_restored"])
        self.assertEqual(report["model_load_result"]["missing_keys"], [])
        self.assertTrue(report["rng_restored"]["torch_random_state"])
        for k, v in src_model.state_dict().items():
            self.assertTrue(torch.equal(v, model.state_dict()[k]), msg=f"Mismatch in {k}")

    def test_scheduler_and_grad_scaler_restored(self) -> None:
        model, opt = _make_training_state()
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        scheduler.step()
        scaler = torch.amp.GradScaler("cpu")
        path = self.tmpdir / "full.pt"
        save_resume_checkpoint(
            path,
            model=model,
            optimizer=opt,
            epoch=1,
            scheduler=scheduler,
            grad_scaler=scaler,
            include_python_rng=False,
            include_numpy_rng=False,
        )
        model2, opt2 = _make_training_state()
        scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=10, gamma=0.5)
        scaler2 = torch.amp.GradScaler("cpu")
        report = load_resume_checkpoint(
            path, model=model2, optimizer=opt2, scheduler=scheduler2, grad_scaler=scaler2
        )
        self.assertTrue(report["scheduler_restored"])
        self.assertTrue(report["grad_scaler_restored"])
        self.assertEqual(scheduler2.last_epoch, scheduler.last_epoch)

    def test_trusted_path_required_for_python_rng(self) -> None:
        model, opt = _make_training_state()
        path = self.tmpdir / "trusted.pt"
        save_resume_checkpoint(path, model=model, optimizer=opt, epoch=6)

        model2, opt2 = _make_training_state()
        # Safe path must fail closed on python/numpy RNG objects...
        with self.assertRaises(Exception):
            load_resume_checkpoint(path, model=model2, optimizer=opt2, trusted=False)
        # ...trusted path loads and restores all RNG streams.
        report = load_resume_checkpoint(path, model=model2, optimizer=opt2, trusted=True)
        self.assertTrue(report["trusted"])
        self.assertTrue(report["rng_restored"]["python_random_state"])
        self.assertTrue(report["rng_restored"]["numpy_random_state"])
        self.assertTrue(report["rng_restored"]["torch_random_state"])

    def test_restore_rng_false_skips_rng(self) -> None:
        path, _, _ = self._save_safe_checkpoint()
        model, opt = _make_training_state()
        report = load_resume_checkpoint(path, model=model, optimizer=opt, restore_rng=False)
        self.assertFalse(any(report["rng_restored"].values()))

    def test_extra_and_dataloader_state_passed_through(self) -> None:
        path, _, _ = self._save_safe_checkpoint(
            extra={"tag": "hex"}, include_dataloader_state={"batches_seen": 5}
        )
        model, opt = _make_training_state()
        report = load_resume_checkpoint(path, model=model, optimizer=opt)
        self.assertTrue(report["has_extra"])
        self.assertEqual(report["extra"], {"tag": "hex"})
        self.assertTrue(report["has_dataloader_state"])
        self.assertEqual(report["dataloader_state_dict"], {"batches_seen": 5})


if __name__ == "__main__":
    unittest.main()
