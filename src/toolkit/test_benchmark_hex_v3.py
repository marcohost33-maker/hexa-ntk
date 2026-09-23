"""
test_benchmark_hex_v3.py
========================
Tests for canonical benchmark metrics.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from benchmark_hex_v3 import (
    BenchmarkReport,
    aggregate_over_time,
    compute_divergence_step,
    compute_h1_rmse_per_step,
    compute_h1_nrmse_per_step,
    compute_nrmse_per_step,
    compute_rmse_per_step,
    evaluate_rollout,
)

T, N, C = 10, 7, 2


def make_target() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(T, N, C) + 2.0  # offset to avoid near-zero norms


def make_pred_close(target: torch.Tensor) -> torch.Tensor:
    return target + 0.01 * torch.randn_like(target)


def make_pred_diverging(target: torch.Tensor) -> torch.Tensor:
    pred = target.clone()
    # Make later timesteps diverge
    for t in range(T):
        pred[t] += 0.1 * (t ** 1.5) * torch.randn(N, C)
    return pred


def make_hex_neighbors() -> torch.Tensor:
    """Simple 7-node hex neighbor array with boundary padding."""
    # Node 0 is center, nodes 1-6 are neighbors
    neighbors = torch.full((N, 6), -1, dtype=torch.long)
    # Center node has all 6 neighbors
    neighbors[0] = torch.tensor([1, 2, 3, 4, 5, 6])
    # Each boundary node has 2 neighbors + center
    for i in range(1, 7):
        n1 = (i % 6) + 1
        n2 = ((i - 2) % 6) + 1
        neighbors[i, 0] = 0
        neighbors[i, 1] = n1
        neighbors[i, 2] = n2
    return neighbors


class TestRMSE(unittest.TestCase):
    def test_identical_predictions_give_zero(self) -> None:
        target = make_target()
        rmse = compute_rmse_per_step(target, target)
        self.assertEqual(rmse.shape, (T,))
        self.assertTrue(torch.allclose(rmse, torch.zeros(T), atol=1e-7))

    def test_rmse_is_positive_for_different_inputs(self) -> None:
        target = make_target()
        pred = make_pred_close(target)
        rmse = compute_rmse_per_step(pred, target)
        self.assertTrue(torch.all(rmse > 0))

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_rmse_per_step(torch.randn(T, N, C), torch.randn(T, N + 1, C))


class TestNRMSE(unittest.TestCase):
    def test_identical_predictions_give_zero(self) -> None:
        target = make_target()
        nrmse = compute_nrmse_per_step(target, target)
        self.assertTrue(torch.allclose(nrmse, torch.zeros(T), atol=1e-6))

    def test_nrmse_normalized_to_roughly_one_for_random(self) -> None:
        torch.manual_seed(0)
        target = torch.randn(T, N, C)
        pred = torch.randn(T, N, C)
        nrmse = compute_nrmse_per_step(pred, target)
        # nRMSE for independent random should be ~sqrt(2) ≈ 1.41
        mean_nrmse = nrmse.mean().item()
        self.assertGreater(mean_nrmse, 0.5)
        self.assertLess(mean_nrmse, 3.0)

    def test_close_predictions_give_small_nrmse(self) -> None:
        target = make_target()
        pred = make_pred_close(target)
        nrmse = compute_nrmse_per_step(pred, target)
        self.assertTrue(torch.all(nrmse < 0.1))


class TestH1Metrics(unittest.TestCase):
    def test_h1_rmse_without_neighbors_equals_l2(self) -> None:
        target = make_target()
        pred = make_pred_close(target)
        l2 = compute_rmse_per_step(pred, target)
        h1 = compute_h1_rmse_per_step(pred, target, hex_neighbors=None)
        self.assertTrue(torch.allclose(l2, h1))

    def test_h1_rmse_with_neighbors_geq_l2(self) -> None:
        target = make_target()
        pred = make_pred_close(target)
        neighbors = make_hex_neighbors()
        l2 = compute_rmse_per_step(pred, target)
        h1 = compute_h1_rmse_per_step(pred, target, neighbors, h=1.0)
        # H¹ ≥ L² by definition
        self.assertTrue(torch.all(h1 >= l2 - 1e-6))

    def test_h1_nrmse_is_bounded(self) -> None:
        target = make_target()
        pred = make_pred_close(target)
        neighbors = make_hex_neighbors()
        h1_nrmse = compute_h1_nrmse_per_step(pred, target, neighbors, h=1.0)
        self.assertTrue(torch.all(h1_nrmse >= 0))
        self.assertTrue(torch.all(h1_nrmse < 1.0))  # Close predictions

    def test_h1_constant_offset_has_zero_gradient_error(self) -> None:
        target = make_target()
        pred = target + 3.0
        neighbors = make_hex_neighbors()
        l2 = compute_rmse_per_step(pred, target)
        h1 = compute_h1_rmse_per_step(pred, target, neighbors, h=1.0)
        # A spatially constant error has zero H¹ seminorm contribution.
        self.assertTrue(torch.allclose(h1, l2, atol=1e-6))

    def test_h1_opposite_gradient_not_lost_by_magnitude_cancellation(self) -> None:
        neighbors = make_hex_neighbors()
        target = torch.zeros(1, N, 1)
        target[0, 1:, 0] = torch.tensor([1.0, 2.0, 1.0, -1.0, -2.0, -1.0])
        pred = -target

        l2 = compute_rmse_per_step(pred, target)
        h1 = compute_h1_rmse_per_step(pred, target, neighbors, h=1.0)

        # Old implementation compared |grad(pred)| with |grad(target)| and
        # therefore cancelled this case.  H¹ must use grad(pred-target).
        self.assertGreater(h1.item(), l2.item())


class TestDivergenceStep(unittest.TestCase):
    def test_no_divergence_returns_T(self) -> None:
        nrmse = torch.full((T,), 0.1)
        self.assertEqual(compute_divergence_step(nrmse, threshold=1.0), T)

    def test_immediate_divergence_returns_zero(self) -> None:
        nrmse = torch.full((T,), 2.0)
        self.assertEqual(compute_divergence_step(nrmse, threshold=1.0), 0)

    def test_divergence_at_step_5(self) -> None:
        nrmse = torch.cat([
            torch.full((5,), 0.5),
            torch.full((5,), 1.5),
        ])
        self.assertEqual(compute_divergence_step(nrmse, threshold=1.0), 5)


class TestAggregation(unittest.TestCase):
    def test_geometric_mean_of_ones(self) -> None:
        vals = torch.ones(10)
        self.assertAlmostEqual(aggregate_over_time(vals, "geometric_mean"), 1.0, places=5)

    def test_arithmetic_mean(self) -> None:
        vals = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(aggregate_over_time(vals, "arithmetic_mean"), 2.0, places=5)

    def test_max(self) -> None:
        vals = torch.tensor([1.0, 5.0, 3.0])
        self.assertAlmostEqual(aggregate_over_time(vals, "max"), 5.0, places=5)

    def test_geometric_mean_less_than_arithmetic(self) -> None:
        vals = torch.tensor([0.1, 1.0, 10.0])
        geo = aggregate_over_time(vals, "geometric_mean")
        arith = aggregate_over_time(vals, "arithmetic_mean")
        self.assertLess(geo, arith)

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_over_time(torch.ones(5), "median")


class TestEvaluateRollout(unittest.TestCase):
    def test_basic_rollout_without_hex(self) -> None:
        target = make_target()
        pred = make_pred_close(target)
        result = evaluate_rollout(pred, target)

        self.assertIn("nrmse_per_step", result)
        self.assertIn("rmse_per_step", result)
        self.assertIn("nrmse_mean", result)
        self.assertIn("divergence_step", result)
        self.assertEqual(result["num_steps"], T)
        self.assertEqual(len(result["nrmse_per_step"]), T)
        self.assertNotIn("h1_rmse_per_step", result)

    def test_rollout_with_hex_neighbors(self) -> None:
        target = make_target()
        pred = make_pred_close(target)
        neighbors = make_hex_neighbors()
        result = evaluate_rollout(pred, target, hex_neighbors=neighbors, h=1.0)

        self.assertIn("h1_rmse_per_step", result)
        self.assertIn("h1_nrmse_per_step", result)
        self.assertIn("h1_rmse_mean", result)
        self.assertIn("h1_nrmse_mean", result)
        self.assertEqual(len(result["h1_rmse_per_step"]), T)

    def test_diverging_rollout_detects_divergence(self) -> None:
        target = make_target()
        pred = make_pred_diverging(target)
        result = evaluate_rollout(pred, target, divergence_threshold=0.5)

        self.assertLess(result["divergence_step"], T)


class TestBenchmarkInputValidation(unittest.TestCase):
    def test_nonfinite_rollout_fails_closed(self) -> None:
        target = make_target()
        pred = target.clone()
        pred[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_nrmse_per_step(pred, target)

    def test_complex_rollout_requires_explicit_metric(self) -> None:
        target = make_target().to(torch.complex64)
        with self.assertRaisesRegex(TypeError, "real floating"):
            compute_rmse_per_step(target, target)

    def test_nonpositive_h_fails_closed(self) -> None:
        target = make_target()
        neighbors = make_hex_neighbors()
        with self.assertRaisesRegex(ValueError, "h must be finite and > 0"):
            compute_h1_rmse_per_step(target, target, neighbors, h=0.0)

    def test_out_of_range_neighbor_fails_closed(self) -> None:
        target = make_target()
        neighbors = make_hex_neighbors()
        neighbors[0, 0] = N
        with self.assertRaisesRegex(ValueError, "valid node indices"):
            compute_h1_rmse_per_step(target, target, neighbors, h=1.0)

    def test_duplicate_neighbor_fails_closed(self) -> None:
        target = make_target()
        neighbors = make_hex_neighbors()
        neighbors[0, 1] = neighbors[0, 0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compute_h1_rmse_per_step(target, target, neighbors, h=1.0)

    def test_self_neighbor_fails_closed(self) -> None:
        target = make_target()
        neighbors = make_hex_neighbors()
        neighbors[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "self-neighbour"):
            compute_h1_rmse_per_step(target, target, neighbors, h=1.0)

    def test_invalid_geometric_metric_not_silently_clamped(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            aggregate_over_time(torch.tensor([0.1, -0.2, 0.3]), "geometric_mean")
        with self.assertRaisesRegex(ValueError, "finite"):
            aggregate_over_time(
                torch.tensor([0.1, float("nan"), 0.3]),
                "geometric_mean",
            )

    def test_invalid_divergence_threshold_fails_closed(self) -> None:
        nrmse = torch.full((T,), 0.1)
        with self.assertRaisesRegex(ValueError, "finite and >= 0"):
            compute_divergence_step(nrmse, threshold=float("nan"))


class TestBenchmarkReport(unittest.TestCase):
    def test_save_and_load_json(self) -> None:
        target = make_target()
        pred = make_pred_close(target)
        metrics = evaluate_rollout(pred, target)

        report = BenchmarkReport(
            model_name="gino_hex_v3_v2",
            dataset_name="fhn_hex_7node_test",
            metrics=metrics,
            config={"hidden_dim": 64, "lr": 1e-3},
            metadata={"git_hash": "abc123"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save_json(path)
            self.assertTrue(path.exists())

            loaded = BenchmarkReport.load_json(path)
            self.assertEqual(loaded.model_name, "gino_hex_v3_v2")
            self.assertEqual(loaded.dataset_name, "fhn_hex_7node_test")
            self.assertEqual(loaded.config["hidden_dim"], 64)
            self.assertAlmostEqual(
                loaded.metrics["nrmse_mean"],
                report.metrics["nrmse_mean"],
                places=6,
            )

    def test_summary_string(self) -> None:
        report = BenchmarkReport(
            model_name="test",
            dataset_name="test_ds",
            metrics={"nrmse_mean": 0.05, "divergence_step": 10},
        )
        s = report.summary()
        self.assertIn("test", s)
        self.assertIn("0.0500", s)


if __name__ == "__main__":
    unittest.main()
