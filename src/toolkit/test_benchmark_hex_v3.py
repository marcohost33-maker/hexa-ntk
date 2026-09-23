"""
test_benchmark_hex_v3.py
========================
Tests for canonical benchmark metrics.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from unittest import mock

import numpy as np
from pathlib import Path

import torch

import benchmark_hex_v3 as benchmark
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

    def test_identical_zero_field_has_exact_zero_h1_error(self) -> None:
        target = torch.zeros(T, N, 1)
        neighbors = make_hex_neighbors()

        h1 = compute_h1_rmse_per_step(target, target, neighbors, h=1.0)
        h1_nrmse = compute_h1_nrmse_per_step(
            target,
            target,
            neighbors,
            h=1.0,
        )

        self.assertTrue(torch.equal(h1, torch.zeros_like(h1)))
        self.assertTrue(torch.equal(h1_nrmse, torch.zeros_like(h1_nrmse)))

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


def make_star_patch(h: float) -> tuple[list[tuple[float, float]], torch.Tensor]:
    """Regular hex patch: centre with a full six-ring at distance ``h``.

    Ring nodes list ONLY the centre as neighbour. For a linear field
    ``f(x) = g . x`` this makes the discrete gradient proxy an exact oracle
    (derivation in TestH1GradientScaling).
    """
    angles = [k * math.pi / 3.0 for k in range(6)]
    points = [(0.0, 0.0)] + [
        (h * math.cos(a), h * math.sin(a)) for a in angles
    ]
    neighbors = torch.full((7, 6), -1, dtype=torch.long)
    neighbors[0] = torch.tensor([1, 2, 3, 4, 5, 6])
    neighbors[1:, 0] = 0
    return points, neighbors


def linear_field(
    points: list[tuple[float, float]], gx: float, gy: float
) -> torch.Tensor:
    """Field ``g . x`` on ``points`` as a [1, N, 1] float64 rollout."""
    values = [gx * x + gy * y for x, y in points]
    return torch.tensor(values, dtype=torch.float64).reshape(1, len(points), 1)


class TestH1GradientScaling(unittest.TestCase):
    """Exact-magnitude oracle for the hex gradient proxy (PR #30, F30-1).

    For six neighbours at distance h in directions u_k (60 deg apart) and a
    linear field, each difference is h * (g . u_k) and
    sum_k (g . u_k)^2 = 3 |g|^2. The proxy (2 / h^2) * sum / n_valid therefore
    gives exactly |g|^2 at the centre. Each ring node sees only the centre:
    (2 / h^2) * h^2 (g . u_i)^2 = 2 (g . u_i)^2, which sums to 6 |g|^2 over
    the ring. The node-RMS of the gradient over all 7 nodes is thus exactly
    |g| -- independent of h. The pre-PR scaling 2 / (3 h^2) yields |g| / sqrt(3);
    a wrong h power (e.g. 2 / h) yields an h-dependent value. Both must fail.
    """

    GRADIENTS = ((3.0, -4.0), (0.0, 2.5), (-1.25, 0.75))

    def test_centre_gradient_magnitude_is_exact_for_linear_field(self) -> None:
        for h in (0.5, 2.0):
            points, neighbors = make_star_patch(h)
            for gx, gy in self.GRADIENTS:
                with self.subTest(h=h, g=(gx, gy)):
                    field = linear_field(points, gx, gy)[0]
                    grad = benchmark._hex_gradient_magnitude(field, neighbors, h)
                    self.assertAlmostEqual(
                        grad[0, 0].item(), math.hypot(gx, gy), places=12
                    )

    def test_h1_error_uses_exact_gradient_magnitude(self) -> None:
        for h in (0.5, 2.0):
            points, neighbors = make_star_patch(h)
            for gx, gy in self.GRADIENTS:
                with self.subTest(h=h, g=(gx, gy)):
                    target = linear_field(points, 1.0, 2.0) + 0.5
                    error = linear_field(points, gx, gy)
                    pred = target + error
                    l2 = torch.sqrt(torch.mean(error ** 2)).item()
                    expected = math.sqrt(l2 ** 2 + gx ** 2 + gy ** 2)
                    h1 = compute_h1_rmse_per_step(pred, target, neighbors, h=h)
                    self.assertEqual(tuple(h1.shape), (1,))
                    self.assertAlmostEqual(h1.item(), expected, places=12)

    def test_h1_nrmse_uses_exact_gradient_magnitude(self) -> None:
        h = 0.5
        points, neighbors = make_star_patch(h)
        target = linear_field(points, 1.0, 2.0) + 0.5
        error = linear_field(points, 3.0, -4.0)
        pred = target + error
        l2_err = torch.sqrt(torch.mean(error ** 2)).item()
        l2_tgt = torch.sqrt(torch.mean(target ** 2)).item()
        eps = 1e-8
        # |grad target| = |(1, 2)| -> squared 5; |grad error| squared 25.
        expected = math.sqrt(l2_err ** 2 + 25.0) / (
            math.sqrt(l2_tgt ** 2 + 5.0) + eps
        )
        h1n = compute_h1_nrmse_per_step(pred, target, neighbors, h=h, eps=eps)
        self.assertAlmostEqual(h1n.item(), expected, places=12)


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

    def test_hex_topology_is_prepared_once_per_evaluate_rollout(self) -> None:
        target = make_target()
        pred = make_pred_close(target)
        neighbors = make_hex_neighbors()

        with mock.patch.object(
            benchmark,
            "_prepare_hex_topology",
            wraps=benchmark._prepare_hex_topology,
        ) as prepare:
            evaluate_rollout(pred, target, hex_neighbors=neighbors, h=1.0)

        self.assertEqual(prepare.call_count, 1)

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

    def test_numpy_real_scalars_are_accepted(self) -> None:
        target = make_target()
        neighbors = make_hex_neighbors()

        compute_nrmse_per_step(target, target, eps=np.float32(1e-6))
        compute_h1_rmse_per_step(
            target,
            target,
            neighbors,
            h=np.float32(1.0),
        )
        step = compute_divergence_step(
            torch.full((T,), 0.1),
            threshold=np.float32(1.0),
        )
        self.assertEqual(step, T)

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
