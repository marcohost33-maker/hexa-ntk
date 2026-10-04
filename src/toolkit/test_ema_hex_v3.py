"""
test_ema_hex_v3.py
==================
Tests for the EMA wrapper module.
"""
from __future__ import annotations

import unittest
from unittest import mock

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


class TestScalarFallback(unittest.TestCase):
    """PyTorch < 2.0 Pfad: get_ema_multi_avg_fn fehlt -> scalar avg_fn Fallback."""

    def test_fallback_used_and_ema_math_correct(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        with mock.patch.object(
            torch.optim.swa_utils,
            "get_ema_multi_avg_fn",
            side_effect=AttributeError("PyTorch < 2.0"),
        ):
            ema = create_ema_model(model, decay=0.5, use_buffers=False)
        self.assertIsInstance(ema, torch.optim.swa_utils.AveragedModel)

        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update_parameters(model)  # first update copies
        self.assertTrue(torch.allclose(ema.module.weight, torch.ones_like(model.weight)))

        with torch.no_grad():
            model.weight.fill_(3.0)
        ema.update_parameters(model)  # 0.5 * 1.0 + 0.5 * 3.0 = 2.0
        self.assertTrue(
            torch.allclose(ema.module.weight, torch.full_like(model.weight, 2.0))
        )


class TestSaveLoadEMA(unittest.TestCase):
    def test_save_ema_state_contains_keys(self) -> None:
        model = DummyModel()
        ema = create_ema_model(model, decay=0.999)
        ema.update_parameters(model)
        state = save_ema_state(ema)
        self.assertIn("ema_model_state_dict", state)
        self.assertIn("n_averaged", state)
        self.assertIn("ema_full_state_dict", state)

    def test_load_via_ema_model_state_dict_only(self) -> None:
        model = DummyModel()
        ema = create_ema_model(model, decay=0.999)
        ema.update_parameters(model)
        ema.update_parameters(model)
        state = save_ema_state(ema)
        # Nur der Legacy-Teil (ohne ema_full_state_dict) -> zweiter Load-Pfad.
        legacy_state = {
            "ema_model_state_dict": state["ema_model_state_dict"],
            "n_averaged": state["n_averaged"],
        }

        model2 = DummyModel()
        ema2 = create_ema_model(model2, decay=0.999)
        report = load_ema_state(ema2, legacy_state)

        self.assertEqual(report["loaded_via"], "ema_model_state_dict")
        self.assertEqual(report["n_averaged"], state["n_averaged"])
        self.assertEqual(report["missing_keys"], [])
        self.assertEqual(report["unexpected_keys"], [])
        for k in ema.module.state_dict():
            self.assertTrue(
                torch.equal(ema.module.state_dict()[k], ema2.module.state_dict()[k]),
                msg=f"Mismatch in {k}",
            )

    def test_load_legacy_state_without_n_averaged(self) -> None:
        model = DummyModel()
        ema = create_ema_model(model, decay=0.999)
        ema.update_parameters(model)
        state = save_ema_state(ema)
        legacy_state = {"ema_model_state_dict": state["ema_model_state_dict"]}

        model2 = DummyModel()
        ema2 = create_ema_model(model2, decay=0.999)
        report = load_ema_state(ema2, legacy_state)
        self.assertEqual(report["loaded_via"], "ema_model_state_dict")
        # n_averaged bleibt beim Init-Wert der frischen EMA (0).
        self.assertEqual(report["n_averaged"], 0)

    def test_load_without_recognized_keys_raises(self) -> None:
        model = DummyModel()
        ema = create_ema_model(model, decay=0.999)
        with self.assertRaises(ValueError):
            load_ema_state(ema, {"unrelated": 1})

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


if __name__ == "__main__":
    unittest.main()
