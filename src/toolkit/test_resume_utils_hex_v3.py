"""
test_resume_utils_hex_v3.py
===========================
Tests for the resume checkpoint utilities.

Covers:
- as_step_count: the epoch/global_step contract (issue #40)
- build/save/load roundtrip incl. scheduler, RNG and EMA state
- load_resume_checkpoint validates before it mutates model/optimizer
- safe (weights_only=True) vs trusted load path
"""
from __future__ import annotations

import random
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from resume_utils_hex_v3 import (
    as_step_count,
    build_resume_payload,
    load_resume_checkpoint,
    restore_rng_states,
    save_resume_checkpoint,
)


def _model() -> nn.Module:
    return nn.Linear(3, 2)


def _state_equal(a: nn.Module, b: nn.Module) -> bool:
    return all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


class TestAsStepCount(unittest.TestCase):
    def test_accepted_values(self) -> None:
        for value, expected in (
            (0, 0),
            (7, 7),
            (7.0, 7),
            (Fraction(14, 2), 7),
            (np.int64(7), 7),
            (np.float64(7.0), 7),
            (torch.tensor(7), 7),
            (np.array([7]), 7),
            (10**30, 10**30),
        ):
            with self.subTest(value=repr(value)):
                result = as_step_count(value, name="epoch")
                self.assertEqual(result, expected)
                self.assertIs(type(result), int)

    def test_value_errors(self) -> None:
        for value in (float("inf"), float("-inf"), float("nan"), 1.5, -1, -0.5,
                      np.float64(2.5), torch.tensor(1.5)):
            with self.subTest(value=repr(value)):
                with self.assertRaises(Exception) as ctx:
                    as_step_count(value, name="global_step")
                self.assertIs(type(ctx.exception), ValueError)
                self.assertIn("global_step must be a non-negative integer", str(ctx.exception))

    def test_type_errors(self) -> None:
        for value in (True, False, None, "3", b"3", [3]):
            with self.subTest(value=repr(value)):
                with self.assertRaises(TypeError):
                    as_step_count(value, name="epoch")

    def test_non_scalar_tensor(self) -> None:
        with self.assertRaises(ValueError):
            as_step_count(torch.tensor([1, 2]), name="epoch")


class TestRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sub" / "resume.pt"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_full_roundtrip_with_scheduler_and_rng(self) -> None:
        torch.manual_seed(0)
        model = _model()
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        model(torch.randn(4, 3)).sum().backward()
        opt.step()
        sched.step()
        random.seed(11)
        np.random.seed(11)
        report = save_resume_checkpoint(
            self.path, model=model, optimizer=opt, epoch=3, global_step=30,
            scheduler=sched, extra={"note": "x"}, include_dataloader_state={"pos": 5},
        )
        self.assertTrue(report["requires_trusted_load"])
        self.assertTrue(report["has_dataloader_state"])
        self.assertFalse(report["has_ema_state"])
        expected_py = random.random()
        expected_np = np.random.rand()

        model2 = _model()
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.1, momentum=0.9)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.5)
        random.seed(99)
        np.random.seed(99)
        loaded = load_resume_checkpoint(
            self.path, model=model2, optimizer=opt2, scheduler=sched2, trusted=True
        )
        self.assertEqual((loaded["epoch"], loaded["global_step"]), (3, 30))
        self.assertTrue(loaded["scheduler_restored"])
        self.assertEqual(loaded["extra"], {"note": "x"})
        self.assertEqual(loaded["dataloader_state_dict"], {"pos": 5})
        self.assertTrue(_state_equal(model, model2))
        self.assertEqual(sched2.get_last_lr(), sched.get_last_lr())
        self.assertEqual(random.random(), expected_py)
        self.assertEqual(np.random.rand(), expected_np)

    def test_safe_path_rejects_python_rng_payload(self) -> None:
        model = _model()
        save_resume_checkpoint(self.path, model=model,
                               optimizer=torch.optim.SGD(model.parameters(), lr=0.1), epoch=0)
        with self.assertRaises(Exception):
            load_resume_checkpoint(self.path, model=_model(),
                                   optimizer=torch.optim.SGD(_model().parameters(), lr=0.1))

    def test_safe_path_without_python_rng(self) -> None:
        model = _model()
        save_resume_checkpoint(self.path, model=model,
                               optimizer=torch.optim.SGD(model.parameters(), lr=0.1), epoch=2,
                               include_python_rng=False, include_numpy_rng=False)
        model2 = _model()
        loaded = load_resume_checkpoint(self.path, model=model2,
                                        optimizer=torch.optim.SGD(model2.parameters(), lr=0.1))
        self.assertEqual(loaded["epoch"], 2)
        self.assertIsNone(loaded["global_step"])
        self.assertTrue(loaded["rng_restored"]["torch_random_state"])
        self.assertFalse(loaded["rng_restored"]["python_random_state"])

    def test_ema_state_in_payload(self) -> None:
        model = _model()
        ema = torch.optim.swa_utils.AveragedModel(model)
        ema.update_parameters(model)
        payload = build_resume_payload(model=model,
                                       optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                                       epoch=0, ema_state_dict=ema.state_dict(),
                                       include_python_rng=False, include_numpy_rng=False)
        self.assertIn("ema_state_dict", payload)
        self.assertEqual(int(payload["ema_state_dict"]["n_averaged"]), 1)

    def test_missing_fields_raise(self) -> None:
        model = _model()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        self.path.parent.mkdir(parents=True)
        for missing in ("model_state_dict", "optimizer_state_dict", "epoch"):
            payload = {"model_state_dict": model.state_dict(),
                       "optimizer_state_dict": opt.state_dict(), "epoch": 0}
            del payload[missing]
            torch.save(payload, self.path)
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(ValueError, missing):
                    load_resume_checkpoint(self.path, model=_model(),
                                           optimizer=torch.optim.SGD(_model().parameters(), lr=0.1))

    def test_non_dict_payload_raises(self) -> None:
        self.path.parent.mkdir(parents=True)
        torch.save([1, 2], self.path)
        with self.assertRaises(TypeError):
            load_resume_checkpoint(self.path, model=_model(),
                                   optimizer=torch.optim.SGD(_model().parameters(), lr=0.1))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_resume_checkpoint(self.path, model=_model(),
                                   optimizer=torch.optim.SGD(_model().parameters(), lr=0.1))


class TestValidateBeforeMutation(unittest.TestCase):
    """A malformed epoch/global_step used to raise only AFTER the model and
    optimizer had been overwritten (half-restored state)."""

    def test_invalid_counters_leave_model_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pt"
            donor = _model()
            opt = torch.optim.SGD(donor.parameters(), lr=0.1)
            poisoned = {k: torch.full_like(v, 7.0) for k, v in donor.state_dict().items()}
            for field, bad in (("epoch", float("inf")), ("epoch", 1.5),
                               ("global_step", -3), ("global_step", float("nan"))):
                payload = {"model_state_dict": poisoned, "optimizer_state_dict": opt.state_dict(),
                           "epoch": 0, "global_step": 0}
                payload[field] = bad
                torch.save(payload, path)
                target = _model()
                before = {k: v.clone() for k, v in target.state_dict().items()}
                with self.subTest(field=field, value=bad):
                    with self.assertRaises(Exception) as ctx:
                        load_resume_checkpoint(path, model=target,
                                               optimizer=torch.optim.SGD(target.parameters(), lr=0.1))
                    self.assertIs(type(ctx.exception), ValueError)
                    for k, v in before.items():
                        self.assertTrue(torch.equal(v, target.state_dict()[k]), msg=k)

    def test_build_payload_rejects_invalid_counters(self) -> None:
        model = _model()
        with self.assertRaises(ValueError):
            build_resume_payload(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                                 epoch=float("inf"))


class TestRestoreRngStates(unittest.TestCase):
    def test_empty_payload_restores_nothing(self) -> None:
        self.assertEqual(
            restore_rng_states({}),
            {"python_random_state": False, "numpy_random_state": False,
             "torch_random_state": False, "torch_cuda_random_state_all": False},
        )


if __name__ == "__main__":
    unittest.main()
