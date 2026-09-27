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

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from resume_utils_hex_v3 import save_resume_checkpoint
from runner_hex_v3 import (
    HexTrainingRunner,
    RunnerConfig,
    TrainingAnomalyDetector,
    dump_replay_capsule_safe,
)


# ------------------------------------------------------------------
# Test fixtures: tiny model + dict-batch DataLoader
# ------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal linear model; forward(x) -> (B, out_features)."""

    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _DictDataset(Dataset):
    """Dataset yielding {"input": ..., "target": ...} like the runner expects."""

    def __init__(self, n: int = 4, in_features: int = 4, out_features: int = 2) -> None:
        self.x = torch.randn(n, in_features)
        self.y = torch.randn(n, out_features)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return {"input": self.x[idx], "target": self.y[idx]}


def _make_loaders(
    n: int = 4, batch_size: int = 2, in_features: int = 4, out_features: int = 2
):
    train = DataLoader(
        _DictDataset(n, in_features, out_features), batch_size=batch_size
    )
    val = DataLoader(
        _DictDataset(n, in_features, out_features), batch_size=batch_size
    )
    return train, val


def _make_runner(output_dir: Path, config: RunnerConfig | None = None, **kwargs):
    """Build a CPU HexTrainingRunner with a tiny model over dict batches."""
    torch.manual_seed(0)
    cfg = config or RunnerConfig(output_dir=str(output_dir))
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_loader, val_loader = _make_loaders()
    return HexTrainingRunner(
        cfg,
        model,
        optimizer,
        train_loader,
        val_loader,
        device="cpu",
        **kwargs,
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


# ------------------------------------------------------------------
# Tests: dump_replay_capsule_safe error paths
# ------------------------------------------------------------------


class TestReplayCapsuleErrorPaths(unittest.TestCase):
    def test_batch_tensor_copy_failure_is_stringified(self) -> None:
        """A tensor whose .detach() raises must be stored as a diagnostic string,
        not crash the capsule dump (partial-failure resilience)."""

        class _Boom:
            def detach(self):
                raise RuntimeError("cannot copy")

        model = nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Force the is_tensor branch for our sentinel; real model tensors
            # still copy fine (they are genuine tensors).
            with mock.patch("runner_hex_v3.torch.is_tensor", return_value=True):
                path = dump_replay_capsule_safe(
                    {"bad": _Boom()}, model, Path(tmpdir), step=5
                )
            loaded = torch.load(path, weights_only=False)
            self.assertIsInstance(loaded["batch_data"]["bad"], str)
            self.assertIn("failed to copy", loaded["batch_data"]["bad"])

    def test_model_state_dict_failure_is_stringified(self) -> None:
        """If model.state_dict() raises, the capsule records a failure marker
        instead of propagating (post-mortem must not itself explode)."""

        class _BadStateModel(nn.Module):
            def state_dict(self, *args, **kwargs):
                raise RuntimeError("state export failed")

        batch = {"input": torch.randn(2, 4)}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = dump_replay_capsule_safe(
                batch, _BadStateModel(), Path(tmpdir), step=7
            )
            loaded = torch.load(path, weights_only=False)
            self.assertIsInstance(loaded["model_state_dict"], str)
            self.assertIn("failed", loaded["model_state_dict"])


# ------------------------------------------------------------------
# Tests: HexTrainingRunner construction
# ------------------------------------------------------------------


class TestRunnerConstruction(unittest.TestCase):
    def test_init_creates_output_and_ckpt_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "run"
            runner = _make_runner(out)
            self.assertTrue(out.exists())
            self.assertIsNotNone(runner.ckpt_mgr)
            self.assertIsNotNone(runner.anomaly_detector)
            self.assertEqual(runner.global_step, 0)
            self.assertEqual(runner.start_epoch, 0)

    def test_init_ema_enabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(Path(tmpdir))
            self.assertIsNotNone(runner.ema_model)

    def test_init_ema_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = RunnerConfig(output_dir=str(Path(tmpdir)), use_ema=False)
            runner = _make_runner(Path(tmpdir), config=cfg)
            self.assertIsNone(runner.ema_model)

    def test_init_no_grad_scaler_on_cpu(self) -> None:
        """AMP GradScaler must stay None on CPU even if use_amp is requested
        (GradScaler is CUDA-only)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = RunnerConfig(output_dir=str(Path(tmpdir)), use_amp=True)
            runner = _make_runner(Path(tmpdir), config=cfg)
            self.assertIsNone(runner.grad_scaler)


# ------------------------------------------------------------------
# Tests: single training step
# ------------------------------------------------------------------


class TestTrainOneStep(unittest.TestCase):
    def _batch(self):
        return {"input": torch.randn(2, 4), "target": torch.randn(2, 2)}

    def test_healthy_step_updates_and_returns_grad_norm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(Path(tmpdir))
            loss_val, is_healthy, grad_norm = runner._train_one_step(self._batch())
            self.assertTrue(is_healthy)
            self.assertIsNotNone(grad_norm)
            self.assertGreaterEqual(grad_norm, 0.0)
            self.assertTrue(torch.isfinite(torch.tensor(loss_val)))

    def test_nan_loss_skips_update_and_dumps_capsule(self) -> None:
        """NaN loss must NOT backprop/step (protect optimizer) and must dump a
        replay capsule + failure signature."""

        def nan_loss(pred, target):
            return pred.sum() * float("nan")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            runner = _make_runner(out, loss_fn=nan_loss)
            loss_val, is_healthy, grad_norm = runner._train_one_step(self._batch())
            self.assertFalse(is_healthy)
            self.assertIsNone(grad_norm)
            self.assertFalse(torch.isfinite(torch.tensor(loss_val)))
            capsules = list((out / "failures").glob("replay_capsule_*.pt"))
            self.assertTrue(capsules, "expected a replay capsule on NaN anomaly")

    def test_loss_spike_dumps_capsule_but_continues_update(self) -> None:
        """A (non-NaN) loss spike is logged + capsuled, but training still
        proceeds with a real optimizer step (spike != corruption)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            cfg = RunnerConfig(
                output_dir=str(out), spike_multiplier=2.0, spike_window_size=10
            )
            runner = _make_runner(out, config=cfg)
            # Prime detector so the rolling median is tiny -> next real MSE spikes.
            runner.anomaly_detector.loss_history.extend([0.001] * 10)
            loss_val, is_healthy, grad_norm = runner._train_one_step(self._batch())
            self.assertTrue(is_healthy)  # non-NaN anomaly still updates
            self.assertIsNotNone(grad_norm)
            capsules = list((out / "failures").glob("replay_capsule_*.pt"))
            self.assertTrue(capsules, "expected a replay capsule on loss spike")

    def test_oom_runtime_error_triggers_rescue_and_reraises(self) -> None:
        """An OOM RuntimeError must dump a capsule then re-raise (fail loud after
        rescue)."""

        class _OOMModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(4, 2)

            def forward(self, x):
                raise RuntimeError("CUDA out of memory. Tried to allocate 8 GiB")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            torch.manual_seed(0)
            model = _OOMModel()
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            tl, vl = _make_loaders()
            runner = HexTrainingRunner(
                RunnerConfig(output_dir=str(out)),
                model, opt, tl, vl, device="cpu",
            )
            with self.assertRaises(RuntimeError):
                runner._train_one_step(self._batch())
            capsules = list((out / "failures").glob("replay_capsule_*.pt"))
            self.assertTrue(capsules, "OOM rescue should have dumped a capsule")

    def test_non_oom_runtime_error_reraises_without_capsule(self) -> None:
        """A non-OOM RuntimeError propagates unchanged (no false OOM rescue)."""

        class _BrokenModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(4, 2)

            def forward(self, x):
                raise RuntimeError("shape mismatch: not an OOM")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            model = _BrokenModel()
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            tl, vl = _make_loaders()
            runner = HexTrainingRunner(
                RunnerConfig(output_dir=str(out)),
                model, opt, tl, vl, device="cpu",
            )
            with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
                runner._train_one_step(self._batch())
            self.assertFalse((out / "failures").exists())

    def test_warmup_scales_learning_rate(self) -> None:
        """During warmup (global_step < warmup_steps) the LR is scaled down from
        the base LR by the linear warmup factor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            cfg = RunnerConfig(
                output_dir=str(out), warmup_steps=10, warmup_start_factor=0.5
            )
            runner = _make_runner(out, config=cfg)
            runner.global_step = 0
            runner._train_one_step(self._batch())
            # base lr 0.1, factor at step 0 = 0.5 -> lr == 0.05
            lr = runner.optimizer.param_groups[0]["lr"]
            self.assertAlmostEqual(lr, 0.05, places=6)
            self.assertEqual(runner.optimizer.param_groups[0]["_base_lr"], 0.1)


# ------------------------------------------------------------------
# Tests: epoch / validation / full run
# ------------------------------------------------------------------


class TestEpochAndValidation(unittest.TestCase):
    def test_train_one_epoch_reports_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            cfg = RunnerConfig(output_dir=str(out), log_every_n_steps=1)
            runner = _make_runner(out, config=cfg)
            report = runner._train_one_epoch(epoch=0)
            self.assertEqual(report["epoch"], 0)
            self.assertIn("mean_loss", report)
            self.assertIn("mean_grad_norm", report)
            self.assertEqual(report["n_anomalies"], 0)
            self.assertEqual(report["n_batches"], 2)  # 4 samples / batch 2
            self.assertEqual(runner.global_step, 2)

    def test_train_one_epoch_steps_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            torch.manual_seed(0)
            model = _TinyModel()
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
            tl, vl = _make_loaders()
            runner = HexTrainingRunner(
                RunnerConfig(output_dir=str(out)),
                model, opt, tl, vl, scheduler=sched, device="cpu",
            )
            runner._train_one_epoch(epoch=0)
            # StepLR with gamma 0.5 after one epoch step -> lr halved
            self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.05, places=6)

    def test_validate_with_ema_returns_val_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(Path(tmpdir))
            report = runner._validate(epoch=0)
            self.assertIn("val_loss", report)
            self.assertIn("val_nrmse", report)
            self.assertTrue(report["used_ema"])
            self.assertEqual(report["n_batches"], 2)

    def test_validate_without_ema_uses_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            cfg = RunnerConfig(output_dir=str(out), use_ema=False)
            runner = _make_runner(out, config=cfg)
            report = runner._validate(epoch=0)
            self.assertFalse(report["used_ema"])
            self.assertGreaterEqual(report["val_loss"], 0.0)


class TestFullRun(unittest.TestCase):
    def test_run_end_to_end_writes_checkpoints_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            cfg = RunnerConfig(
                output_dir=str(out), num_epochs=2, log_every_n_steps=1
            )
            runner = _make_runner(out, config=cfg)
            summary = runner.run()
            self.assertEqual(summary["final_epoch"], 1)
            self.assertEqual(len(summary["history"]["train"]), 2)
            self.assertEqual(len(summary["history"]["val"]), 2)
            self.assertEqual(len(summary["history"]["checkpoints"]), 2)
            self.assertIsNone(summary["resume_report"])
            self.assertGreater(summary["global_step"], 0)
            # best/last/resume checkpoints written by _save_checkpoints
            ckpt_dir = out / "checkpoints"
            self.assertTrue((ckpt_dir / "best.pt").exists())
            self.assertTrue((ckpt_dir / "last.pt").exists())
            self.assertTrue((ckpt_dir / "resume_last.pt").exists())


class TestNonFiniteValidationMetric(unittest.TestCase):
    """A NaN/inf validation metric never becomes the best checkpoint (#37).

    The checkpoint manager rejects it with ValueError; the runner is the
    policy layer and skips the best-model save with a FailureSignature
    instead of aborting the run (last/resume checkpoints are still written).
    """

    @staticmethod
    def _val_report(value: float) -> dict:
        return {
            "epoch": 0,
            "val_loss": value,
            "val_nrmse": value,
            "n_batches": 2,
            "used_ema": True,
        }

    def test_save_checkpoints_skips_best_and_records_failure(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmpdir:
                out = Path(tmpdir)
                runner = _make_runner(out)
                reports = runner._save_checkpoints(
                    epoch=0,
                    val_report=self._val_report(value),
                    train_report={"mean_loss": 1.0},
                )
                self.assertFalse(reports["best"]["saved"])
                self.assertEqual(reports["best"]["reason"], "non_finite_metric")
                self.assertEqual(reports["best"]["checkpoint_semantics"], "best_model")
                self.assertIsNone(runner.ckpt_mgr.best_metric_value)
                self.assertFalse(Path(runner.ckpt_mgr.best_model_path).exists())
                self.assertTrue(reports["last"]["saved"])
                self.assertTrue(reports["resume"]["saved"])
                signatures = sorted((out / "failures").glob("*NaN_or_Inf_Metric*.json"))
                self.assertEqual(len(signatures), 1)
                record = json.loads(signatures[0].read_text(encoding="utf-8"))
                self.assertEqual(record["kind"], "NaN_or_Inf_Metric")
                self.assertEqual(record["locus"], "save_checkpoints_best_metric")
                self.assertEqual(record["extra"], {"epoch": 0, "metric_name": "val_nrmse"})

    def test_unrepresentable_metric_reaches_manager_contract(self) -> None:
        # Only NaN/inf are skipped; a value math.isfinite() cannot judge is
        # not masked as "non-finite" but rejected by the manager's contract.
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            runner = _make_runner(out)
            with self.assertRaisesRegex(
                ValueError, r"^metric_value must be a finite real number"
            ):
                runner._save_checkpoints(
                    epoch=0,
                    val_report=self._val_report(10**400),
                    train_report={"mean_loss": 1.0},
                )
            self.assertFalse((out / "failures").exists())
            self.assertIsNone(runner.ckpt_mgr.best_metric_value)

    def test_run_continues_past_nan_validation_and_later_saves_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            cfg = RunnerConfig(output_dir=str(out), num_epochs=2)
            runner = _make_runner(out, config=cfg)
            reports = iter([self._val_report(float("nan")), self._val_report(0.5)])
            with mock.patch.object(
                runner, "_validate", side_effect=lambda epoch: next(reports)
            ):
                summary = runner.run()
            best_reports = [c["best"] for c in summary["history"]["checkpoints"]]
            self.assertEqual(
                [(r["saved"], r["reason"]) for r in best_reports],
                [(False, "non_finite_metric"), (True, "improved")],
            )
            self.assertEqual(runner.ckpt_mgr.best_metric_value, 0.5)


# ------------------------------------------------------------------
# Tests: resume
# ------------------------------------------------------------------


class TestResume(unittest.TestCase):
    def test_maybe_resume_returns_none_when_not_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(Path(tmpdir))
            self.assertIsNone(runner.maybe_resume())

    def test_maybe_resume_nonexistent_path_fails_closed(self) -> None:
        """A configured but missing resume_from must raise, not retrain.

        Formerly this test expected None + start_epoch 0 (pinned by the
        coverage ratchet 65f765b), which legalised a silent full retraining
        on a mistyped resume path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            missing = out / "does_not_exist.pt"
            cfg = RunnerConfig(output_dir=str(out), resume_from=str(missing))
            runner = _make_runner(out, config=cfg)
            with self.assertRaises(Exception) as ctx:
                runner.maybe_resume()
            self.assertIsInstance(ctx.exception, FileNotFoundError)
            self.assertIn("does_not_exist.pt", str(ctx.exception))
            self.assertEqual(runner.start_epoch, 0)
            self.assertEqual(runner.global_step, 0)

    def test_run_with_missing_resume_from_does_not_train(self) -> None:
        """run() must propagate the error before any epoch is trained."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            cfg = RunnerConfig(
                output_dir=str(out),
                num_epochs=1,
                resume_from=str(out / "typo.pt"),
            )
            runner = _make_runner(out, config=cfg)
            with self.assertRaises(FileNotFoundError):
                runner.run()
            self.assertEqual(runner.train_losses, [])
            self.assertFalse(Path(runner.ckpt_mgr.resume_path).exists())

    def test_maybe_resume_restores_epoch_and_step(self) -> None:
        """End-to-end resume: run, then a fresh runner over the SAME run dir
        resumes from the written resume checkpoint and advances start_epoch past
        the saved epoch. The runner loads exactly resume_from."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            # 1. Produce a resume checkpoint via a real run.
            cfg1 = RunnerConfig(output_dir=str(out), num_epochs=1)
            runner1 = _make_runner(out, config=cfg1)
            runner1.run()
            resume_ckpt = runner1.ckpt_mgr.resume_path
            self.assertTrue(Path(resume_ckpt).exists())

            # 2. Fresh runner over the same run dir resumes from it (trusted:
            #    the resume payload carries Python/NumPy RNG state).
            cfg2 = RunnerConfig(
                output_dir=str(out),
                resume_from=resume_ckpt,
                resume_trusted=True,
            )
            runner2 = _make_runner(out, config=cfg2)
            report = runner2.maybe_resume()
            self.assertIsNotNone(report)
            self.assertEqual(runner2.start_epoch, report["epoch"] + 1)

    def test_maybe_resume_loads_named_foreign_checkpoint(self) -> None:
        """resume_from must not silently load output_dir/checkpoints/resume_last.pt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "run"
            foreign = Path(tmpdir) / "elsewhere" / "resume.pt"

            torch.manual_seed(0)
            src_model = _TinyModel()
            src_optimizer = torch.optim.SGD(src_model.parameters(), lr=0.1)
            save_resume_checkpoint(
                foreign,
                model=src_model,
                optimizer=src_optimizer,
                epoch=7,
                global_step=42,
                include_python_rng=False,
                include_numpy_rng=False,
            )

            cfg = RunnerConfig(output_dir=str(out), resume_from=str(foreign))
            runner = _make_runner(out, config=cfg)
            report = runner.maybe_resume()

            self.assertIsNotNone(report)
            self.assertEqual(report["checkpoint_path"], str(foreign))
            self.assertEqual(runner.start_epoch, 8)
            self.assertEqual(runner.global_step, 42)
            for key, value in src_model.state_dict().items():
                self.assertTrue(
                    torch.equal(value, runner.model.state_dict()[key]),
                    msg=f"Mismatch in {key}",
                )

    def test_maybe_resume_prefers_resume_from_over_own_checkpoint(self) -> None:
        """F31-1: the pre-fix failure mode. output_dir already holds the
        runner's OWN resume checkpoint (different epoch/step/weights) and
        resume_from names a foreign one. The foreign file must be loaded;
        silently loading output_dir/checkpoints/resume_last.pt is the bug."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "run"
            foreign = Path(tmpdir) / "elsewhere" / "resume.pt"

            torch.manual_seed(0)
            src_model = _TinyModel()
            src_optimizer = torch.optim.SGD(src_model.parameters(), lr=0.1)
            save_resume_checkpoint(
                foreign,
                model=src_model,
                optimizer=src_optimizer,
                epoch=7,
                global_step=42,
                include_python_rng=False,
                include_numpy_rng=False,
            )

            cfg = RunnerConfig(output_dir=str(out), resume_from=str(foreign))
            runner = _make_runner(out, config=cfg)

            # Own checkpoint in output_dir with distinct epoch/step/weights.
            torch.manual_seed(123)
            own_model = _TinyModel()
            own_optimizer = torch.optim.SGD(own_model.parameters(), lr=0.1)
            runner.ckpt_mgr.save_resume(
                model=own_model,
                optimizer=own_optimizer,
                epoch=1,
                global_step=5,
                include_python_rng=False,
                include_numpy_rng=False,
            )
            own_path = Path(runner.ckpt_mgr.resume_path)
            self.assertTrue(own_path.exists())
            self.assertNotEqual(own_path.resolve(), foreign.resolve())
            self.assertFalse(
                all(
                    torch.equal(v, own_model.state_dict()[k])
                    for k, v in src_model.state_dict().items()
                ),
                msg="precondition: own and foreign weights must differ",
            )

            report = runner.maybe_resume()

            self.assertIsNotNone(report)
            self.assertEqual(report["checkpoint_path"], str(foreign))
            self.assertEqual(report["epoch"], 7)
            self.assertEqual(runner.start_epoch, 8)
            self.assertEqual(runner.global_step, 42)
            for key, value in src_model.state_dict().items():
                self.assertTrue(
                    torch.equal(value, runner.model.state_dict()[key]),
                    msg=f"Mismatch in {key}",
                )



# ------------------------------------------------------------------
# Review 2026-09-27: config/detector contracts, empty validation,
# best every epoch, final-epoch state, full resume, real nRMSE
# ------------------------------------------------------------------


class _SeededDataset(Dataset):
    def __init__(self, n: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 4, generator=g)
        self.y = torch.randn(n, 2, generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return {"input": self.x[idx], "target": self.y[idx]}


def _seeded_runner(out: Path, cfg: RunnerConfig, *, val_n: int = 6) -> HexTrainingRunner:
    torch.manual_seed(0)
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    return HexTrainingRunner(
        cfg,
        model,
        optimizer,
        DataLoader(_SeededDataset(8, seed=0), batch_size=2),
        DataLoader(_SeededDataset(val_n, seed=1), batch_size=2),
        device="cpu",
    )


class TestDetectorContract(unittest.TestCase):
    def test_small_windows_do_not_crash(self) -> None:
        # window_size=1 called statistics.median([]) on the first loss.
        for window in (1, 2, 3):
            with self.subTest(window=window):
                det = TrainingAnomalyDetector(spike_multiplier=2.0, window_size=window)
                self.assertEqual(det.check_loss(1.0), (False, ""))
                self.assertTrue(det.check_loss(10.0)[0])

    def test_invalid_window_rejected(self) -> None:
        for window in (0, -1, 2.5, True):
            with self.subTest(window=window):
                with self.assertRaises(ValueError):
                    TrainingAnomalyDetector(window_size=window)

    def test_non_finite_spike_multiplier_rejected(self) -> None:
        for mult in (float("nan"), float("inf")):
            with self.subTest(mult=mult):
                with self.assertRaises(ValueError):
                    TrainingAnomalyDetector(spike_multiplier=mult)


class TestRunnerConfigContract(unittest.TestCase):
    def test_invalid_values_rejected_at_construction(self) -> None:
        for field, value in (
            ("save_every_n_epochs", 0),
            ("num_epochs", -1),
            ("num_epochs", 2.0),
            ("log_every_n_steps", -1),
            ("warmup_steps", -5),
            ("spike_window_size", 0),
            ("grad_clip_max_norm", float("nan")),
            ("grad_clip_max_norm", 0.0),
            ("warmup_start_factor", 0.0),
            ("warmup_start_factor", 1.5),
            ("ema_decay", 1.5),
            ("ema_decay", float("nan")),
            ("amp_dtype", "fp8"),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    RunnerConfig(**{field: value})

    def test_inf_clip_norm_allowed(self) -> None:
        self.assertEqual(RunnerConfig(grad_clip_max_norm=float("inf")).grad_clip_max_norm, float("inf"))


class TestEmptyValidationAndTrainEpoch(unittest.TestCase):
    def test_empty_validation_never_becomes_best(self) -> None:
        # Before: val_loss 0.0 became best and blocked every real value.
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            runner = _seeded_runner(out, RunnerConfig(output_dir=str(out), num_epochs=1), val_n=0)
            summary = runner.run()
            val = summary["history"]["val"][0]
            self.assertEqual(val["n_batches"], 0)
            self.assertTrue(math.isnan(val["val_loss"]))
            self.assertTrue(math.isnan(val["val_nrmse"]))
            best = summary["history"]["checkpoints"][0]["best"]
            self.assertEqual((best["saved"], best["reason"]), (False, "empty_validation"))
            self.assertIsNone(runner.ckpt_mgr.best_metric_value)
            self.assertFalse(Path(runner.ckpt_mgr.best_model_path).exists())
            sigs = list((out / "failures").glob("*Empty_Validation*.json"))
            self.assertEqual(len(sigs), 1)
            self.assertTrue(summary["history"]["checkpoints"][0]["resume"]["saved"])

    def test_epoch_without_healthy_step_reports_nan_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            runner = _seeded_runner(out, RunnerConfig(output_dir=str(out), num_epochs=1))
            runner.loss_fn = lambda pred, target: ((pred - target) ** 2).mean() * float("nan")
            report = runner._train_one_epoch(0)
            self.assertEqual(report["n_batches"], 0)
            self.assertEqual(report["n_anomalies"], 4)
            self.assertTrue(math.isnan(report["mean_loss"]))
            self.assertTrue(math.isnan(report["mean_grad_norm"]))


class TestCheckpointSchedule(unittest.TestCase):
    @staticmethod
    def _val(value: float) -> dict:
        return {"epoch": 0, "val_loss": value, "val_nrmse": value, "n_batches": 2, "used_ema": True}

    def _run(self, out: Path, cfg: RunnerConfig, metrics):
        runner = _seeded_runner(out, cfg)
        it = iter([self._val(m) for m in metrics])
        with mock.patch.object(runner, "_validate", side_effect=lambda epoch: next(it)):
            summary = runner.run()
        return runner, summary

    def test_best_is_evaluated_every_epoch(self) -> None:
        # Before: with save_every_n_epochs=2 the best model of epoch 1 was
        # never considered.
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            runner, summary = self._run(
                out, RunnerConfig(output_dir=str(out), num_epochs=3, save_every_n_epochs=2),
                [0.9, 0.5, 0.7],
            )
            ckpts = summary["history"]["checkpoints"]
            self.assertEqual(len(ckpts), 3)
            self.assertEqual([c["best"]["reason"] for c in ckpts], ["improved", "improved", "not_improved"])
            self.assertEqual(runner.ckpt_mgr.best_metric_value, 0.5)
            best_extra = torch.load(runner.ckpt_mgr.best_model_path, weights_only=True)["extra"]
            self.assertEqual(best_extra["epoch"], 1)
            self.assertEqual([c["last"]["saved"] for c in ckpts], [True, False, True])
            self.assertEqual(ckpts[1]["resume"]["reason"], "not_scheduled")

    def test_final_epoch_state_is_always_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            runner, summary = self._run(
                out, RunnerConfig(output_dir=str(out), num_epochs=5, save_every_n_epochs=3),
                [0.9, 0.8, 0.7, 0.6, 0.5],
            )
            # Epoch 4 is not on the 3-epoch schedule; it was dropped before.
            saved = [c["resume"]["saved"] for c in summary["history"]["checkpoints"]]
            self.assertEqual(saved, [True, False, False, True, True])
            payload = torch.load(runner.ckpt_mgr.resume_path, weights_only=False)
            self.assertEqual(payload["epoch"], 4)


class TestFullResumeEquivalence(unittest.TestCase):
    """Resume must continue the run, not restart parts of it.

    Before: the EMA average restarted from the resumed weights and the best
    value restarted at None (next epoch overwrote a better best.pt).
    """

    def _cfg(self, out: Path, epochs: int, **kw) -> RunnerConfig:
        return RunnerConfig(output_dir=str(out), num_epochs=epochs, ema_decay=0.9,
                            log_every_n_steps=0, **kw)

    def test_interrupted_run_equals_uninterrupted_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            full_dir, res_dir = Path(tmpdir) / "full", Path(tmpdir) / "res"
            full = _seeded_runner(full_dir, self._cfg(full_dir, 4))
            full.run()

            _seeded_runner(res_dir, self._cfg(res_dir, 2)).run()
            resumed = _seeded_runner(res_dir, self._cfg(
                res_dir, 4,
                resume_from=str(res_dir / "checkpoints" / "resume_last.pt"),
                resume_trusted=True,
            ))
            summary = resumed.run()

            self.assertTrue(summary["resume_report"]["ema_restored"])
            self.assertEqual(resumed.global_step, full.global_step)
            for name, value in full.model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.model.state_dict()[name]), msg=name)
            for name, value in full.ema_model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.ema_model.state_dict()[name]), msg=f"ema {name}")
            self.assertEqual(resumed.ckpt_mgr.best_metric_value, full.ckpt_mgr.best_metric_value)
            full_best = torch.load(full.ckpt_mgr.best_model_path, weights_only=True)["extra"]
            res_best = torch.load(resumed.ckpt_mgr.best_model_path, weights_only=True)["extra"]
            self.assertEqual(
                (res_best["metric_value"], res_best["epoch"]),
                (full_best["metric_value"], full_best["epoch"]),
            )

    def test_resume_with_other_ema_decay_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            first = _seeded_runner(out, self._cfg(out, 1))
            first.run()
            cfg = RunnerConfig(output_dir=str(out), num_epochs=2, ema_decay=0.5,
                               resume_from=first.ckpt_mgr.resume_path, resume_trusted=True)
            with self.assertRaisesRegex(ValueError, "decay"):
                _seeded_runner(out, cfg).maybe_resume()

    def test_resume_with_other_best_metric_does_not_adopt_foreign_best(self) -> None:
        # Runner half of the Codex-T1 fix on #46 (Equalita H7): the runner must
        # hand its best_metric_name to load_resume. With metric_name=None the
        # val_nrmse best of the first run was adopted into a val_loss run and
        # then blocked every val_loss best.pt (compared across metrics).
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            first = _seeded_runner(out, self._cfg(out, 2, best_metric_name="val_nrmse"))
            first.run()
            self.assertIsNotNone(first.ckpt_mgr.best_metric_value)

            second = _seeded_runner(out, self._cfg(
                out, 3, best_metric_name="val_loss",
                resume_from=first.ckpt_mgr.resume_path, resume_trusted=True))
            report = second.maybe_resume()
            self.assertFalse(report["best_metric_restored"])
            self.assertEqual(report["best_metric_restore_reason"], "metric_name_mismatch")
            self.assertIsNone(second.ckpt_mgr.best_metric_value)

            second.run()
            best_extra = torch.load(second.ckpt_mgr.best_model_path, weights_only=True)["extra"]
            self.assertEqual(best_extra["metric_name"], "val_loss")
            self.assertEqual(best_extra["epoch"], 2)

    def test_resume_report_exposes_best_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            first = _seeded_runner(out, self._cfg(out, 2))
            first.run()
            second = _seeded_runner(out, self._cfg(
                out, 3, resume_from=first.ckpt_mgr.resume_path, resume_trusted=True))
            report = second.maybe_resume()
            self.assertTrue(report["best_metric_restored"])
            self.assertEqual(second.ckpt_mgr.best_metric_value, first.ckpt_mgr.best_metric_value)


class TestValidationNrmse(unittest.TestCase):
    def test_val_nrmse_is_pooled_relative_l2(self) -> None:
        # Before: val_nrmse was a copy of the MSE val_loss ("placeholder").
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            runner = _seeded_runner(out, RunnerConfig(output_dir=str(out), use_ema=False))
            report = runner._validate(epoch=0)
            ds = runner.val_loader.dataset
            with torch.no_grad():
                pred = runner.model(ds.x).double()
            target = ds.y.double()
            expected = math.sqrt(float(((pred - target) ** 2).sum() / (target ** 2).sum()))
            self.assertAlmostEqual(report["val_nrmse"], expected, places=12)
            expected_mse = float(((pred - target) ** 2).mean())
            self.assertAlmostEqual(report["val_loss"], expected_mse, places=5)
            self.assertNotAlmostEqual(report["val_nrmse"], report["val_loss"], places=3)

    def test_perfect_prediction_of_zero_target_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            runner = _seeded_runner(out, RunnerConfig(output_dir=str(out), use_ema=False))
            ds = runner.val_loader.dataset
            ds.y = torch.zeros_like(ds.y)
            with torch.no_grad():
                runner.model.lin.weight.zero_()
                runner.model.lin.bias.zero_()
            self.assertEqual(runner._validate(epoch=0)["val_nrmse"], 0.0)
            with torch.no_grad():
                runner.model.lin.bias.fill_(1.0)
            self.assertEqual(runner._validate(epoch=0)["val_nrmse"], float("inf"))


if __name__ == "__main__":
    unittest.main()
