"""
test_ema_hex_v3.py
==================
Tests for the EMA wrapper module.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from ema_hex_v3 import create_ema_model, load_ema_state, save_ema_state


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.bn = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.linear(x))


class TestCreateEMA(unittest.TestCase):
    def test_ema_model_created(self) -> None:
        model = DummyModel()
        ema = create_ema_model(model, decay=0.999)
        self.assertIsNotNone(ema)
        self.assertIsInstance(ema, torch.optim.swa_utils.AveragedModel)

    def test_ema_initial_copy(self) -> None:
        model = DummyModel()
        ema = create_ema_model(model, decay=0.999)
        # First update_parameters copies
        ema.update_parameters(model)
        # Weights should match after first update
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), ema.module.state_dict().items()
        ):
            self.assertTrue(torch.equal(v1, v2), msg=f"Mismatch in {k1}")

    def test_ema_diverges_after_training_step(self) -> None:
        model = DummyModel()
        ema = create_ema_model(model, decay=0.999)
        ema.update_parameters(model)

        # Simulate training step
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.randn(8, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # EMA not updated yet — should differ from model
        ema_sd = ema.module.state_dict()
        model_sd = model.state_dict()
        any_different = False
        for k in model_sd:
            if not torch.equal(model_sd[k], ema_sd[k]):
                any_different = True
                break
        self.assertTrue(any_different)

        # Now update EMA
        ema.update_parameters(model)
        # EMA should be between old and new (decay=0.999, mostly old)


class TestSaveLoadEMA(unittest.TestCase):
    def test_save_ema_state_contains_keys(self) -> None:
        model = DummyModel()
        ema = create_ema_model(model, decay=0.999)
        ema.update_parameters(model)
        state = save_ema_state(ema)
        self.assertIn("ema_model_state_dict", state)
        self.assertIn("n_averaged", state)
        self.assertIn("ema_full_state_dict", state)

    def test_load_ema_state_roundtrip(self) -> None:
        model = DummyModel()
        ema = create_ema_model(model, decay=0.999)
        ema.update_parameters(model)
        state = save_ema_state(ema)

        # Create fresh EMA
        model2 = DummyModel()
        ema2 = create_ema_model(model2, decay=0.999)
        report = load_ema_state(ema2, state)

        self.assertIn("loaded_via", report)
        # Weights should match
        for k in ema.module.state_dict():
            self.assertTrue(
                torch.equal(
                    ema.module.state_dict()[k],
                    ema2.module.state_dict()[k],
                ),
                msg=f"Mismatch in {k}",
            )



class TestDecayContract(unittest.TestCase):
    def test_out_of_range_or_non_finite_decay_rejected(self) -> None:
        # decay > 1 diverges, < 0 oscillates, NaN poisons every parameter.
        for decay in (-0.1, 1.0001, 2.0, float("nan"), float("inf"), True, "0.9", None):
            with self.subTest(decay=repr(decay)):
                with self.assertRaises(ValueError):
                    create_ema_model(DummyModel(), decay=decay)

    def test_boundary_decays_accepted(self) -> None:
        for decay in (0.0, 1.0, 0.5):
            with self.subTest(decay=decay):
                self.assertIsNotNone(create_ema_model(DummyModel(), decay=decay))

    def test_decay_formula(self) -> None:
        torch.manual_seed(0)
        model = DummyModel()
        ema = create_ema_model(model, decay=0.75)
        ema.update_parameters(model)  # first update copies
        start = ema.module.linear.weight.detach().clone()
        with torch.no_grad():
            model.linear.weight.add_(1.0)
        ema.update_parameters(model)
        expected = 0.75 * start + 0.25 * model.linear.weight.detach()
        self.assertTrue(torch.allclose(ema.module.linear.weight, expected))


class TestLoadEmaStatePaths(unittest.TestCase):
    def _trained_ema(self):
        torch.manual_seed(0)
        model = DummyModel()
        ema = create_ema_model(model, decay=0.9)
        for _ in range(3):
            with torch.no_grad():
                model.linear.weight.add_(0.1)
            ema.update_parameters(model)
        return ema

    def test_full_state_roundtrip_reports_keys(self) -> None:
        ema = self._trained_ema()
        state = save_ema_state(ema)
        target = create_ema_model(DummyModel(), decay=0.9)
        report = load_ema_state(target, state)
        self.assertEqual(report["loaded_via"], "ema_full_state_dict")
        self.assertEqual(report["n_averaged"], 3)
        self.assertEqual((report["missing_keys"], report["unexpected_keys"]), ([], []))

    def test_full_state_honours_strict_false(self) -> None:
        # strict used to be ignored on this path.
        state = save_ema_state(self._trained_ema())
        full = dict(state["ema_full_state_dict"])
        full.pop("module.linear.bias")
        target = create_ema_model(DummyModel(), decay=0.9)
        with self.assertRaises(RuntimeError):
            load_ema_state(target, {"ema_full_state_dict": full})
        report = load_ema_state(target, {"ema_full_state_dict": full}, strict=False)
        self.assertEqual(report["missing_keys"], ["module.linear.bias"])

    def test_module_state_path_sets_counter(self) -> None:
        state = save_ema_state(self._trained_ema())
        target = create_ema_model(DummyModel(), decay=0.9)
        report = load_ema_state(
            target,
            {"ema_model_state_dict": state["ema_model_state_dict"], "n_averaged": 3},
        )
        self.assertEqual(report["loaded_via"], "ema_model_state_dict")
        self.assertEqual(report["n_averaged"], 3)

    def test_unknown_state_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_ema_state(create_ema_model(DummyModel()), {"weights": {}})


if __name__ == "__main__":
    unittest.main()
