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
    EMADecayError,
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
                                       ema_decay=0.9,
                                       include_python_rng=False, include_numpy_rng=False)
        self.assertIn("ema_state_dict", payload)
        self.assertEqual(int(payload["ema_state_dict"]["n_averaged"]), 1)
        self.assertEqual(payload["ema_decay"], 0.9)

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


class TestEmaDecayContract(unittest.TestCase):
    """Codex review #46 (PRRT_kwDOSwvSC86m2eei): EMA state and its decay
    travel together. Saving EMA state without a decay, restoring it without
    the current decay, or restoring a checkpoint that never recorded one
    would let a 0.9-average continue as a 0.5-average unnoticed."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "resume.pt"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _ema(model: nn.Module, decay: float) -> torch.optim.swa_utils.AveragedModel:
        return torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay),
            use_buffers=True)

    def _save(self, **kw):
        torch.manual_seed(0)
        model = _model()
        ema = self._ema(model, 0.9)
        ema.update_parameters(model)
        return save_resume_checkpoint(
            self.path, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            epoch=1, ema_state_dict=ema.state_dict(),
            include_python_rng=False, include_numpy_rng=False, **kw)

    def _load(self, **kw):
        torch.manual_seed(1)
        model = _model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        return model, before, lambda: load_resume_checkpoint(
            self.path, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            **kw)

    def test_build_payload_requires_decay_with_ema_state(self) -> None:
        model = _model()
        ema = self._ema(model, 0.9)
        with self.assertRaisesRegex(EMADecayError, "without ema_decay"):
            build_resume_payload(model=model,
                                 optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                                 epoch=0, ema_state_dict=ema.state_dict(),
                                 include_python_rng=False, include_numpy_rng=False)

    def test_save_requires_decay_and_writes_nothing(self) -> None:
        with self.assertRaisesRegex(EMADecayError, "without ema_decay"):
            self._save()
        self.assertFalse(self.path.exists())

    def test_invalid_decay_is_rejected_on_save(self) -> None:
        for bad in (float("nan"), float("inf"), -0.1, 1.5, True, "0.9"):
            with self.subTest(decay=bad):
                with self.assertRaises(EMADecayError):
                    self._save(ema_decay=bad)
                self.assertFalse(self.path.exists())

    def test_decay_without_ema_state_is_not_recorded(self) -> None:
        model = _model()
        payload = build_resume_payload(model=model,
                                       optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                                       epoch=0, ema_decay=0.9,
                                       include_python_rng=False, include_numpy_rng=False)
        self.assertNotIn("ema_decay", payload)
        self.assertNotIn("ema_state_dict", payload)

    def test_error_is_a_value_error(self) -> None:
        self.assertTrue(issubclass(EMADecayError, ValueError))

    def test_matching_decay_restores(self) -> None:
        self._save(ema_decay=0.9)
        _, _, load = self._load(ema_model=self._ema(_model(), 0.9), ema_decay=0.9)
        report = load()
        self.assertTrue(report["ema_restored"])
        self.assertEqual(report["ema_decay"], 0.9)

    def test_load_requires_current_decay(self) -> None:
        self._save(ema_decay=0.9)
        target_ema = self._ema(_model(), 0.5)
        ema_before = {k: v.clone() for k, v in target_ema.state_dict().items()}
        model, before, load = self._load(ema_model=target_ema)
        with self.assertRaisesRegex(EMADecayError, "ema_decay was not given"):
            load()
        for k, v in before.items():
            self.assertTrue(torch.equal(v, model.state_dict()[k]), msg=k)
        for k, v in ema_before.items():
            self.assertTrue(torch.equal(v, target_ema.state_dict()[k]), msg=k)

    def test_mismatching_decay_is_rejected(self) -> None:
        self._save(ema_decay=0.9)
        model, before, load = self._load(ema_model=self._ema(_model(), 0.5), ema_decay=0.5)
        with self.assertRaisesRegex(EMADecayError, "decay 0.9"):
            load()
        for k, v in before.items():
            self.assertTrue(torch.equal(v, model.state_dict()[k]), msg=k)

    def test_invalid_current_decay_is_rejected(self) -> None:
        self._save(ema_decay=0.9)
        for bad in (float("nan"), 2.0, "0.9"):
            with self.subTest(decay=bad):
                _, _, load = self._load(ema_model=self._ema(_model(), 0.9), ema_decay=bad)
                with self.assertRaises(EMADecayError):
                    load()

    def _set_recorded_decay(self, value=None) -> None:
        payload = torch.load(self.path, weights_only=True)
        if value is None:
            del payload["ema_decay"]
        else:
            payload["ema_decay"] = value
        torch.save(payload, self.path)

    def test_legacy_checkpoint_without_decay_fails_closed(self) -> None:
        # Decision: a checkpoint with EMA state but no recorded decay (written
        # before the decay was mandatory) cannot be continued with an EMA.
        self._save(ema_decay=0.9)
        self._set_recorded_decay()
        model, before, load = self._load(ema_model=self._ema(_model(), 0.9), ema_decay=0.9)
        with self.assertRaisesRegex(EMADecayError, "without the decay"):
            load()
        for k, v in before.items():
            self.assertTrue(torch.equal(v, model.state_dict()[k]), msg=k)
        # The documented way out: resume without ema_model.
        _, _, load = self._load()
        report = load()
        self.assertFalse(report["ema_restored"])
        self.assertIsNone(report["ema_decay"])

    def test_malformed_recorded_decay_fails_closed(self) -> None:
        self._save(ema_decay=0.9)
        self._set_recorded_decay(value=float("nan"))
        _, _, load = self._load(ema_model=self._ema(_model(), 0.9), ema_decay=0.9)
        with self.assertRaisesRegex(EMADecayError, "checkpoint ema_decay"):
            load()


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


class TestLegacyNegativeCounters(unittest.TestCase):
    """E3=A (2026-10-05): a schema-1.0 checkpoint whose writer stored a
    negative counter (e.g. epoch=-1 as a pre-training sentinel) is rejected
    before anything is restored, with a message naming file, field, value and
    the way out. No silent migration."""

    def test_negative_counter_fails_closed_with_explanation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            donor = _model()
            opt = torch.optim.SGD(donor.parameters(), lr=0.1)
            poisoned = {k: torch.full_like(v, 7.0) for k, v in donor.state_dict().items()}
            for field in ("epoch", "global_step"):
                payload = {"checkpoint_type": "resume", "schema_version": "1.0",
                           "model_state_dict": poisoned,
                           "optimizer_state_dict": opt.state_dict(),
                           "epoch": 0, "global_step": 0}
                payload[field] = -1
                torch.save(payload, path)
                target = _model()
                before = {k: v.clone() for k, v in target.state_dict().items()}
                with self.subTest(field=field):
                    with self.assertRaises(ValueError) as ctx:
                        load_resume_checkpoint(path, model=target,
                                               optimizer=torch.optim.SGD(target.parameters(), lr=0.1))
                    msg = str(ctx.exception)
                    self.assertIn(f"stores {field}=-1", msg)
                    self.assertIn("schema_version='1.0'", msg)
                    self.assertIn("nothing was restored", msg)
                    self.assertIn(str(path), msg)
                    for k, v in before.items():
                        self.assertTrue(torch.equal(v, target.state_dict()[k]), msg=k)


def _sched_factories():
    L = torch.optim.lr_scheduler
    return {
        "StepLR": lambda o: L.StepLR(o, step_size=2, gamma=0.5),
        "LambdaLR": lambda o: L.LambdaLR(o, lr_lambda=lambda e: 0.9 ** e),
        "CosineAnnealingLR": lambda o: L.CosineAnnealingLR(o, T_max=10),
        "OneCycleLR": lambda o: L.OneCycleLR(o, max_lr=0.5, total_steps=20),
        "CosineAnnealingWarmRestarts": lambda o: L.CosineAnnealingWarmRestarts(o, T_0=3),
        "SequentialLR": lambda o: L.SequentialLR(
            o, [L.LinearLR(o, start_factor=0.1, total_iters=3), L.ExponentialLR(o, gamma=0.8)],
            milestones=[3]),
        "CyclicLR": lambda o: L.CyclicLR(o, base_lr=0.01, max_lr=0.1, step_size_up=2),
    }


class TestRestoreOrderBehaviour(unittest.TestCase):
    """E4=B (2026-10-05), spec section 8.3: optimizer AND scheduler are
    constructed before any state is loaded; then optimizer -> scheduler.

    Behavioural oracle instead of a call-order mock: after resume the
    learning rate equals the checkpoint's, and the continued LR trajectory and
    weights are bit-identical to an uninterrupted run. The negative control
    (scheduler constructed AFTER optimizer.load_state_dict, whose constructor
    step overwrites the restored lr) must be detected by the same comparison.
    """

    N_PRE, N_POST = 7, 6

    @staticmethod
    def _build(factory):
        torch.manual_seed(0)
        model = nn.Linear(3, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        return model, opt, factory(opt)

    @staticmethod
    def _step(model, opt, sched, i):
        x = torch.full((4, 3), float(i + 1))
        loss = (model(x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        return [g["lr"] for g in opt.param_groups]

    def _uninterrupted(self, factory):
        model, opt, sched = self._build(factory)
        traj = [self._step(model, opt, sched, i) for i in range(self.N_PRE + self.N_POST)]
        return traj[self.N_PRE:], [p.detach().clone() for p in model.parameters()]

    def _resumed(self, factory, path, *, construct_scheduler_after_load=False):
        model, opt, sched = self._build(factory)
        for i in range(self.N_PRE):
            self._step(model, opt, sched, i)
        save_resume_checkpoint(path, model=model, optimizer=opt, epoch=self.N_PRE,
                               scheduler=sched, include_python_rng=False,
                               include_numpy_rng=False, include_torch_rng=False)
        ckpt_lr = [g["lr"] for g in opt.param_groups]
        model2, opt2, sched2 = self._build(factory)
        if construct_scheduler_after_load:
            # The hazard: build the scheduler only after the optimizer state is in.
            load_resume_checkpoint(path, model=model2, optimizer=opt2, restore_rng=False)
            sched2 = factory(opt2)
            sched2.load_state_dict(torch.load(path, weights_only=True)["scheduler_state_dict"])
        else:
            load_resume_checkpoint(path, model=model2, optimizer=opt2, scheduler=sched2,
                                   restore_rng=False)
        lr_after = [g["lr"] for g in opt2.param_groups]
        traj = [self._step(model2, opt2, sched2, self.N_PRE + i) for i in range(self.N_POST)]
        return ckpt_lr, lr_after, traj, [p.detach().clone() for p in model2.parameters()]

    def _matches(self, ref, resumed):
        ckpt_lr, lr_after, traj, weights = resumed
        return (lr_after == ckpt_lr and traj == ref[0]
                and all(torch.equal(a, b) for a, b in zip(weights, ref[1])))

    def test_resume_equals_uninterrupted_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name, factory in _sched_factories().items():
                with self.subTest(scheduler=name):
                    ref = self._uninterrupted(factory)
                    resumed = self._resumed(factory, Path(tmp) / f"{name}.pt")
                    self.assertEqual(resumed[1], resumed[0], "lr after resume != checkpoint lr")
                    self.assertEqual(resumed[2], ref[0], "LR trajectory diverged")
                    for a, b in zip(resumed[3], ref[1]):
                        self.assertTrue(torch.equal(a, b), "weights diverged")

    def test_oracle_detects_scheduler_constructed_after_optimizer_load(self) -> None:
        # Negative control: the comparison above must be able to fail.
        detected = []
        with tempfile.TemporaryDirectory() as tmp:
            for name, factory in _sched_factories().items():
                ref = self._uninterrupted(factory)
                bad = self._resumed(factory, Path(tmp) / f"{name}_bad.pt",
                                    construct_scheduler_after_load=True)
                if not self._matches(ref, bad):
                    detected.append(name)
        # Measured with torch 2.6.0: closed-form/stateful-cycle schedulers are hit.
        for name in ("LambdaLR", "OneCycleLR", "CosineAnnealingWarmRestarts",
                     "SequentialLR", "CyclicLR"):
            self.assertIn(name, detected)


class TestRestoreRngStates(unittest.TestCase):
    def test_empty_payload_restores_nothing(self) -> None:
        self.assertEqual(
            restore_rng_states({}),
            {"python_random_state": False, "numpy_random_state": False,
             "torch_random_state": False, "torch_cuda_random_state_all": False},
        )


if __name__ == "__main__":
    unittest.main()
