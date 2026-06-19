"""
test_runner_hex_v3.py
=====================
Tests for the HexaNTK training runner components.

Focuses on testable units without requiring GPU or full model:
- TrainingAnomalyDetector
- dump_replay_capsule_safe
- RunnerConfig defaults
- FailureSignature integration
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from runner_hex_v3 import (
    RunnerConfig,
    TrainingAnomalyDetector,
    dump_replay_capsule_safe,
)


# ------------------------------------------------------------------
# Tests: TrainingAnomalyDetector
# ------------------------------------------------------------------


class TestTrainingAnomalyDetector(unittest.TestCase):
    def test_nan_detected(self) -> None:
        det = TrainingAnomalyDetector()
        is_anomaly, reason = det.check_loss(float("nan"))
        self.assertTrue(is_anomaly)
        self.assertEqual(reason, "NaN_or_Inf_Loss")

    def test_inf_detected(self) -> None:
        det = TrainingAnomalyDetector()
        is_anomaly, reason = det.check_loss(float("inf"))
        self.assertTrue(is_anomaly)
        self.assertEqual(reason, "NaN_or_Inf_Loss")

    def test_neg_inf_detected(self) -> None:
        det = TrainingAnomalyDetector()
        is_anomaly, reason = det.check_loss(float("-inf"))
        self.assertTrue(is_anomaly)

    def test_normal_loss_not_anomaly(self) -> None:
        det = TrainingAnomalyDetector(window_size=10)
        for _ in range(10):
            is_anomaly, _ = det.check_loss(1.0)
        self.assertFalse(is_anomaly)

    def test_spike_detected_after_warmup(self) -> None:
        det = TrainingAnomalyDetector(spike_multiplier=2.0, window_size=10)
        # Fill window with normal values
        for _ in range(10):
            det.check_loss(1.0)
        # Now a spike
        is_anomaly, reason = det.check_loss(5.0)
        self.assertTrue(is_anomaly)
        self.assertIn("Loss_Spike", reason)

    def test_no_spike_during_warmup(self) -> None:
        det = TrainingAnomalyDetector(spike_multiplier=2.0, window_size=20)
        # Only 3 values, below window_size // 2
        for _ in range(3):
            det.check_loss(1.0)
        # Even a large value shouldn't trigger (not enough history)
        is_anomaly, _ = det.check_loss(100.0)
        self.assertFalse(is_anomaly)

    def test_gradual_increase_no_spike(self) -> None:
        det = TrainingAnomalyDetector(spike_multiplier=3.0, window_size=10)
        # Gradually increasing — should not trigger spike
        for i in range(20):
            val = 1.0 + 0.05 * i
            is_anomaly, _ = det.check_loss(val)
        # The last value (1.95) should not be a spike relative to the recent median
        self.assertFalse(is_anomaly)

    def test_invalid_spike_multiplier_raises(self) -> None:
        with self.assertRaises(ValueError):
            TrainingAnomalyDetector(spike_multiplier=0.5)

    def test_median_robustness_to_outliers(self) -> None:
        """Median is robust: a single previous outlier shouldn't prevent spike detection."""
        det = TrainingAnomalyDetector(spike_multiplier=2.0, window_size=10)
        # 9 normal + 1 outlier
        for _ in range(9):
            det.check_loss(1.0)
        det.check_loss(100.0)  # One outlier enters the window
        # Normal value after that
        is_anomaly, _ = det.check_loss(1.0)
        self.assertFalse(is_anomaly)


# ------------------------------------------------------------------
# Tests: Replay capsule
# ------------------------------------------------------------------


class TestReplayCapsule(unittest.TestCase):
    def test_dump_creates_file(self) -> None:
        model = nn.Linear(4, 2)
        batch = {
            "input": torch.randn(2, 4),
            "target": torch.randn(2, 2),
            "metadata": "some_string",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = dump_replay_capsule_safe(
                batch, model, Path(tmpdir), step=42
            )
            self.assertTrue(Path(path).exists())

    def test_capsule_contains_model_and_batch(self) -> None:
        model = nn.Linear(4, 2)
        batch = {"input": torch.randn(2, 4)}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = dump_replay_capsule_safe(
                batch, model, Path(tmpdir), step=0
            )
            loaded = torch.load(path, weights_only=False)
            self.assertIn("batch_data", loaded)
            self.assertIn("model_state_dict", loaded)
            self.assertIn("input", loaded["batch_data"])
            self.assertEqual(loaded["step"], 0)

    def test_capsule_tensors_on_cpu(self) -> None:
        model = nn.Linear(4, 2)
        batch = {"input": torch.randn(2, 4)}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = dump_replay_capsule_safe(
                batch, model, Path(tmpdir), step=1
            )
            loaded = torch.load(path, weights_only=False)
            self.assertEqual(
                loaded["batch_data"]["input"].device, torch.device("cpu")
            )


# ------------------------------------------------------------------
# Tests: RunnerConfig
# ------------------------------------------------------------------


class TestRunnerConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = RunnerConfig()
        self.assertEqual(cfg.num_epochs, 100)
        self.assertEqual(cfg.grad_clip_max_norm, 1.0)
        self.assertTrue(cfg.use_ema)
        self.assertFalse(cfg.use_amp)
        self.assertTrue(cfg.is_main_process)
        self.assertIsNone(cfg.resume_from)
        self.assertEqual(cfg.warmup_steps, 0)
        self.assertEqual(cfg.warmup_start_factor, 0.01)

    def test_custom_values(self) -> None:
        cfg = RunnerConfig(
            num_epochs=50,
            use_amp=True,
            amp_dtype="bfloat16",
            ema_decay=0.999,
            warmup_steps=100,
        )
        self.assertEqual(cfg.num_epochs, 50)
        self.assertTrue(cfg.use_amp)
        self.assertEqual(cfg.amp_dtype, "bfloat16")
        self.assertEqual(cfg.ema_decay, 0.999)
        self.assertEqual(cfg.warmup_steps, 100)


# ------------------------------------------------------------------
# Tests: FailureSignature integration
# ------------------------------------------------------------------


class TestFailureSignatureInRunner(unittest.TestCase):
    def test_failure_signature_from_anomaly(self) -> None:
        from failure_signature_hex_v3 import FailureSignature

        sig = FailureSignature(
            kind="NaN_or_Inf_Loss",
            step=100,
            locus="train_step_loss_check",
            observable="loss=nan",
            replay_capsule="/tmp/capsule.pt",
        )
        d = sig.to_dict()
        self.assertEqual(d["kind"], "NaN_or_Inf_Loss")
        self.assertEqual(d["step"], 100)
        self.assertIn("timestamp", d)


if __name__ == "__main__":
    unittest.main()
