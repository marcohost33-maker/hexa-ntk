"""
test_spectral_audit_hex_v3.py
=============================
Contract tests for the spectral audit (spec/SPECTRAL_AUDIT_v1.md): the T1
mode matrix (W-orthonormal eigenvectors of L_h, fail-closed on coarse or
aliasing lattices), the torch backend against a linear model with known
weights (oracle iv) and a float32 module smoke (v), the fail-closed paths
(non-finite output, shapes, budgets, orthonormality) and the verdict.
The heat oracles (i)-(iii) and the metamorphic relations live in
test_spectral_audit_oracles_hex_v3.py.
"""

from __future__ import annotations

import copy
import math
import unittest
from unittest import mock

import numpy as np
import torch

import spectral_audit_hex_v3 as sa
from heat_operator_task_hex_v3 import HeatOperatorTaskConfig, mode_basis
from hex_fv_solver_hex_v3 import apply_hex_laplacian, torus_hex_lattice, triangle_hex_lattice

TRI = HeatOperatorTaskConfig()
TORUS = HeatOperatorTaskConfig(domain="torus_periodic")
ORACLE_RTOL = 1e-12
EPS = float(np.finfo(np.float64).eps)


def _lattice(config, n):
    return triangle_hex_lattice(n) if config.domain == "triangle_dirichlet" else torus_hex_lattice(n)


def _setup(config, n):
    lattice = _lattice(config, n)
    return lattice, sa.heat_mode_matrix(config, lattice)


class TestHeatModeMatrix(unittest.TestCase):
    def test_orthonormal_eigenvectors_of_the_discrete_laplacian(self):
        for config in (TRI, TORUS):
            for n in (16, 32):
                with self.subTest(domain=config.domain, n=n):
                    lattice, hm = _setup(config, n)
                    q, w = hm.modes, hm.weights
                    self.assertEqual(q.shape, (lattice.num_nodes, len(mode_basis(config))))
                    self.assertLessEqual(hm.orthonormality_defect, hm.orthonormality_tol)
                    gram = q.T @ (w[:, None] * q)
                    self.assertLessEqual(np.max(np.abs(gram - np.eye(q.shape[1]))), 1e-13)
                    # The oracle precondition, checked against the solver operator itself:
                    # L_h q_k = -lambda_h q_k up to rounding of the stencil (scale lambda_h max|q|).
                    for k in range(q.shape[1]):
                        lq = apply_hex_laplacian(q[:, k], lattice)
                        scale = hm.discrete_eigenvalues[k] * np.max(np.abs(q[:, k]))
                        self.assertLessEqual(np.max(np.abs(lq + hm.discrete_eigenvalues[k] * q[:, k])) / scale,
                                             1e-11)
                    self.assertEqual(hm.keys, tuple(m.key for m in mode_basis(config)))
                    self.assertTrue(np.all(w == lattice.cell_area))

    def test_arrays_read_only(self):
        _, hm = _setup(TRI, 16)
        for arr in (hm.modes, hm.weights, hm.eigenvalues, hm.discrete_eigenvalues):
            with self.assertRaises(ValueError):
                arr[0] = 1.0

    def test_lattice_too_coarse_raises(self):
        with self.assertRaisesRegex(ValueError, "too coarse"):
            sa.heat_mode_matrix(TRI, triangle_hex_lattice(6))  # 10 nodes < 15 modes
        self.assertEqual(sa.heat_mode_matrix(TRI, triangle_hex_lattice(7)).modes.shape, (15, 15))

    def test_aliasing_raises(self):
        config = HeatOperatorTaskConfig(domain="torus_periodic", lambda_cutoff=2000.0)
        lattice = torus_hex_lattice(12)
        self.assertLessEqual(len(mode_basis(config)), lattice.num_nodes)  # not caught by the count check
        with self.assertRaisesRegex(ValueError, "not W-orthonormal"):
            sa.heat_mode_matrix(config, lattice)

    def test_vanishing_function_raises(self):
        class Zero:
            key = "zero"
            eigenvalue = 1.0

            def evaluate(self, coords):
                return np.zeros(len(coords))

        with mock.patch.object(sa, "mode_basis", return_value=(Zero(),)):
            with self.assertRaisesRegex(ValueError, "vanish"):
                sa.heat_mode_matrix(TRI, triangle_hex_lattice(8))

    def test_invalid_arguments(self):
        with self.assertRaises(TypeError):
            sa.heat_mode_matrix("triangle_dirichlet", triangle_hex_lattice(8))
        with self.assertRaises(ValueError):
            sa.heat_mode_matrix(TRI, torus_hex_lattice(8))


class TestTorchBackend(unittest.TestCase):
    """(iv) a torch linear model with known weights; (v) a float32 nn.Module smoke."""

    def setUp(self):
        self.lattice, self.hm = _setup(TRI, 16)
        self.q, self.w = self.hm.modes, self.hm.weights
        m = self.lattice.num_nodes
        rng = np.random.default_rng(21)
        self.matrix = rng.standard_normal((m, m))
        self.reference = self.q.T @ (self.w[:, None] * (self.matrix @ self.q))
        self.scale = np.max(np.abs(self.reference))
        self.linear = torch.nn.Linear(m, m, bias=True).double().eval()
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(self.matrix))
            self.linear.bias.copy_(torch.from_numpy(rng.standard_normal(m)))

    def test_linear_model_exact_in_both_modes(self):
        for mode in ("forward", "reverse"):
            with self.subTest(mode=mode):
                result = sa.input_jacobian_spectrum(self.linear, self.q, self.w, torch_mode=mode)
                self.assertEqual((result.backend, result.torch_mode, result.dtype), ("torch", mode, "float64"))
                self.assertLessEqual(np.max(np.abs(result.projection - self.reference)) / self.scale, 1e-13)
                self.assertEqual(result.evaluations,
                                 sa._REPEAT_EVALUATIONS + self.q.shape[1] + (mode == "reverse"))
        self.assertIsNone(sa.input_jacobian_spectrum(self.linear, self.q, self.w, torch_mode="reverse").out_of_span)

    def test_numpy_and_torch_agree(self):
        numpy_result = sa.input_jacobian_spectrum(lambda u: self.matrix @ u, self.q, self.w)
        torch_result = sa.input_jacobian_spectrum(
            lambda u: torch.as_tensor(self.matrix) @ u, self.q, self.w, backend="torch")
        self.assertLessEqual(np.max(np.abs(numpy_result.projection - torch_result.projection)) / self.scale, 1e-13)
        self.assertLessEqual(np.max(np.abs(numpy_result.out_of_span - torch_result.out_of_span)), 1e-12)

    def test_huge_finite_jacobian_keeps_derived_metrics_finite(self):
        # Codex P2 on #71: entries ~1e200 squared to inf, out_of_span became inf/inf = NaN
        # while only S was checked. Norms are max-scaled now; metrics equal the 1x case.
        base = sa.input_jacobian_spectrum(lambda u: self.matrix @ u, self.q, self.w)
        big = sa.input_jacobian_spectrum(lambda u: (1e200 * self.matrix) @ u, self.q, self.w)
        self.assertTrue(np.all(np.isfinite(big.out_of_span)))
        np.testing.assert_allclose(big.out_of_span, base.out_of_span, rtol=1e-9, atol=1e-12)
        self.assertTrue(np.isfinite(big.coupling_fraction))
        self.assertAlmostEqual(big.coupling_fraction, base.coupling_fraction, delta=1e-9)

    def test_float32_module_smoke(self):
        torch.manual_seed(0)
        m = self.lattice.num_nodes
        mlp = torch.nn.Sequential(torch.nn.Linear(m, 64), torch.nn.Tanh(), torch.nn.Linear(64, m)).eval()
        base = np.random.default_rng(22).standard_normal(m) * 0.5
        reference = sa.input_jacobian_spectrum(copy.deepcopy(mlp).double(), self.q, self.w, base_point=base)
        scale = np.max(np.abs(reference.projection))
        for mode in ("forward", "reverse"):
            with self.subTest(mode=mode):
                result = sa.input_jacobian_spectrum(mlp, self.q, self.w, base_point=base, torch_mode=mode)
                self.assertEqual(result.dtype, "float32")
                self.assertTrue(np.all(np.isfinite(result.projection)))
                self.assertLessEqual(np.max(np.abs(result.projection - reference.projection)) / scale, 1e-5)
        for p in mlp.parameters():
            self.assertIsNone(p.grad)  # the audit leaves no gradients behind

        def numpy_f32(u):
            with torch.no_grad():
                return mlp(torch.as_tensor(u, dtype=torch.float32)).numpy()

        fd = sa.input_jacobian_spectrum(numpy_f32, self.q, self.w, base_point=base)
        # The step follows the float32 output: eps32^(1/3) * max(1, ||u0||_W).
        expected_step = float(np.finfo(np.float32).eps) ** (1 / 3) * max(
            1.0, math.sqrt(float(np.sum(self.w * base * base))))
        self.assertAlmostEqual(fd.fd_step, expected_step, delta=1e-15)
        self.assertEqual(fd.dtype, "float32")
        # Measured 1.9e-5 with this step, 2.7e-3 with the float64 step.
        self.assertLessEqual(np.max(np.abs(fd.projection - reference.projection)) / scale, 2e-4)

    def test_train_mode_guard(self):
        model = torch.nn.Linear(self.lattice.num_nodes, self.lattice.num_nodes).double()
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, "training mode"):
            sa.input_jacobian_spectrum(model, self.q, self.w)
        result = sa.input_jacobian_spectrum(model, self.q, self.w, allow_train_mode=True)
        self.assertTrue(model.training)  # not switched behind the caller's back
        self.assertTrue(np.all(np.isfinite(result.projection)))

    def test_constant_module_gives_zero_s(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.ones_like(x)

        for mode in ("forward", "reverse"):
            with self.subTest(mode=mode):
                result = sa.input_jacobian_spectrum(Constant().eval(), self.q, self.w, torch_mode=mode,
                                                    oracle=np.ones(self.q.shape[1]))
                self.assertTrue(np.all(result.projection == 0.0))
                self.assertIsNone(result.coupling_fraction)
                self.assertEqual(result.dtype, "float64")  # no parameters -> float64
                self.assertEqual(sa.audit_verdict(result, diag_rtol=1.0, coupling_tol=1.0), "FAIL")

    def test_torch_fail_closed(self):
        m = self.lattice.num_nodes
        cases = {
            "nan": (lambda u: u * float("nan"), sa.NonFiniteOutputError),
            "inf_last": (lambda u: torch.cat([u[:-1], u[-1:] + float("inf")]), sa.NonFiniteOutputError),
            "shape": (lambda u: u[:-1], ValueError),
            "not_tensor": (lambda u: [0.0] * m, TypeError),
            "integer": (lambda u: torch.zeros(m, dtype=torch.int64), TypeError),
        }
        for name, (fn, exc) in cases.items():
            for mode in ("forward", "reverse"):
                with self.subTest(case=name, mode=mode):
                    with self.assertRaises(exc):
                        sa.input_jacobian_spectrum(fn, self.q, self.w, backend="torch", torch_mode=mode)
        with self.assertRaises(TypeError):
            sa.input_jacobian_spectrum(self.linear, self.q, self.w, torch_dtype=torch.int32)
        with self.assertRaises(TypeError):
            sa.input_jacobian_spectrum(self.linear, self.q, self.w, torch_dtype="float32")

    def test_nonfinite_jacobian_products_fail_closed(self):
        # Finite outputs, non-finite derivative: sqrt at 0 has an infinite slope.
        for mode in ("forward", "reverse"):
            with self.subTest(mode=mode):
                with self.assertRaises(sa.NonFiniteOutputError):
                    sa.input_jacobian_spectrum(lambda u: torch.sqrt(u * u * 0.0 + u.abs()), self.q, self.w,
                                               backend="torch", torch_mode=mode)

    def test_torch_budget(self):
        k = self.q.shape[1] + sa._REPEAT_EVALUATIONS  # Jacobian columns + determinism preflight
        with self.assertRaises(sa.BudgetExceededError):
            sa.input_jacobian_spectrum(self.linear, self.q, self.w, max_evaluations=k - 1)
        with self.assertRaises(sa.BudgetExceededError):
            sa.input_jacobian_spectrum(self.linear, self.q, self.w, torch_mode="reverse", max_evaluations=k)
        self.assertEqual(sa.input_jacobian_spectrum(self.linear, self.q, self.w, max_evaluations=k).evaluations, k)


class TestFailClosed(unittest.TestCase):
    def setUp(self):
        self.lattice, self.hm = _setup(TRI, 16)
        self.q, self.w = self.hm.modes, self.hm.weights
        self.m, self.k = self.q.shape

    def test_nonfinite_output(self):
        for position in (0, -1):
            for bad in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(position=position, bad=bad):
                    def op(u, position=position, bad=bad):
                        out = u.copy()
                        out[position] = bad
                        return out

                    with self.assertRaises(sa.NonFiniteOutputError):
                        sa.input_jacobian_spectrum(op, self.q, self.w)

    def test_nonfinite_only_at_the_base_point(self):
        # The base-point output enters no difference quotient: S would be finite
        # and the audit would pass silently over a model that is broken at u0.
        calls = []

        def op(u):
            calls.append(1)
            return u * float("nan") if len(calls) == 1 else u

        with self.assertRaises(sa.NonFiniteOutputError):
            sa.input_jacobian_spectrum(op, self.q, self.w)
        self.assertEqual(len(calls), 1)

    def test_nonfinite_only_off_the_base_point(self):
        calls = []

        def op(u):
            calls.append(1)
            return u if len(calls) < 5 else u * float("nan")

        with self.assertRaises(sa.NonFiniteOutputError):
            sa.input_jacobian_spectrum(op, self.q, self.w)

    def test_bad_output_shape_and_type(self):
        for op, exc in ((lambda u: u[:-1], ValueError), (lambda u: u[None, :], ValueError),
                        (lambda u: u > 0, TypeError), (lambda u: u.astype(complex), TypeError)):
            with self.assertRaises(exc):
                sa.input_jacobian_spectrum(op, self.q, self.w)

    def test_torch_tensor_from_numpy_callable_accepted(self):
        result = sa.input_jacobian_spectrum(lambda u: torch.from_numpy(2.0 * u), self.q, self.w)
        self.assertLessEqual(np.max(np.abs(result.projection - 2.0 * np.eye(self.k))), 1e-12)

    def test_evaluation_budget_checked_before_any_call(self):
        calls = []

        def op(u):
            calls.append(1)
            return u

        need = 2 * self.k + 1 + sa._REPEAT_EVALUATIONS  # FD stencil + determinism preflight
        with self.assertRaises(sa.BudgetExceededError):
            sa.input_jacobian_spectrum(op, self.q, self.w, max_evaluations=need - 1)
        self.assertEqual(calls, [])
        result = sa.input_jacobian_spectrum(op, self.q, self.w, max_evaluations=need)
        self.assertEqual((result.evaluations, len(calls)), (need, need))

    def test_time_budget(self):
        with self.assertRaises(sa.BudgetExceededError):
            sa.input_jacobian_spectrum(lambda u: u, self.q, self.w, max_seconds=1e-12)

    def test_not_orthonormal_modes(self):
        # Pins the tolerance from above: a column norm off by 64 M eps (~1.5e-12 here) is rejected.
        for factor in (1.0 + 1e-6, 1.0 + 64 * self.m * EPS):
            bad = self.q.copy()
            bad[:, 0] *= factor
            with self.subTest(factor=factor):
                with self.assertRaisesRegex(ValueError, "not W-orthonormal"):
                    sa.input_jacobian_spectrum(lambda u: u, bad, self.w)
        sheared = self.q.copy()
        sheared[:, 1] += 1e-6 * sheared[:, 0]
        with self.assertRaisesRegex(ValueError, "not W-orthonormal"):
            sa.input_jacobian_spectrum(lambda u: u, sheared, self.w)
        # A looser caller tolerance accepts the 1e-6 defect.
        result = sa.input_jacobian_spectrum(lambda u: u, sheared, self.w, orthonormality_tol=1e-5)
        self.assertEqual(result.orthonormality_tol, 1e-5)

    def test_invalid_inputs(self):
        q, w = self.q, self.w
        nan_q = q.copy()
        nan_q[0, 0] = np.nan
        cases = [
            (dict(modes=nan_q), ValueError),
            (dict(modes=q[:, :0]), ValueError),
            (dict(modes=q[0]), ValueError),
            (dict(modes=np.eye(3)), ValueError),
            (dict(modes=q.astype(bool)), TypeError),
            (dict(weights=np.where(np.arange(self.m) == 0, 0.0, w)), ValueError),
            (dict(weights=-w), ValueError),
            (dict(weights=np.full(self.m, np.nan)), ValueError),
            (dict(weights=w[None, :]), ValueError),
            (dict(weights=np.array([], dtype=float)), ValueError),
            (dict(base_point=np.zeros(self.m + 1)), ValueError),
            (dict(base_point=np.full(self.m, np.inf)), ValueError),
            (dict(oracle=np.ones(self.k + 1)), ValueError),
            (dict(oracle=np.full(self.k, np.nan)), ValueError),
            (dict(fd_step=0.0), ValueError),
            (dict(fd_step=-1e-6), ValueError),
            (dict(fd_step=float("nan")), ValueError),
            (dict(fd_step=True), TypeError),
            (dict(max_evaluations=0), ValueError),
            (dict(max_evaluations=True), TypeError),
            (dict(max_evaluations=10.0), TypeError),
            (dict(max_seconds=0.0), ValueError),
            (dict(orthonormality_tol=float("inf")), ValueError),
            (dict(backend="jax"), ValueError),
            (dict(torch_mode="sideways"), ValueError),
            (dict(torch_dtype=torch.float32), ValueError),  # numpy backend
        ]
        for overrides, exc in cases:
            kwargs = dict(modes=q, weights=w)
            kwargs.update(overrides)
            with self.subTest(overrides=list(overrides)):
                with self.assertRaises(exc):
                    sa.input_jacobian_spectrum(lambda u: u, **kwargs)
        with self.assertRaises(TypeError):
            sa.input_jacobian_spectrum("not callable", q, w)
        wide = np.eye(4, 5)
        with self.assertRaisesRegex(ValueError, "cannot be orthonormal"):
            sa.input_jacobian_spectrum(lambda u: u, wide, np.ones(4))

    def test_results_read_only(self):
        result = sa.input_jacobian_spectrum(lambda u: u, self.q, self.w, oracle=np.ones(self.k))
        for arr in (result.projection, result.diagonal, result.out_of_span, result.oracle,
                    result.diag_abs_error, result.diag_rel_error):
            with self.assertRaises(ValueError):
                arr[0] = 0.0


class TestDeterminismPreflight(unittest.TestCase):
    """B6 (2026-10-15): .training alone does not prove a deterministic operator.
    Two forwards at the base point must agree bitwise and leave the global RNG
    state (torch, numpy, random) unchanged."""

    def setUp(self):
        self.lattice, self.hm = _setup(TRI, 16)
        self.q, self.w = self.hm.modes, self.hm.weights

    def _refused(self, operator, needle, **kwargs):
        with self.assertRaises(Exception) as caught:
            sa.input_jacobian_spectrum(operator, self.q, self.w, **kwargs)
        got = type(caught.exception).__name__
        self.assertIsInstance(caught.exception, ValueError, f"expected ValueError, got {got}")
        self.assertIn(needle, str(caught.exception))

    def test_eval_module_with_noise_is_refused(self):
        class Noisy(torch.nn.Module):
            def forward(self, x):
                return x + 1e-3 * torch.rand_like(x)

        self._refused(Noisy().eval(), "RNG", backend="torch")

    def test_rng_consumer_with_identical_output_is_refused(self):
        # Output is deterministic, but the global torch RNG advances: only the RNG check sees it.
        class Draws(torch.nn.Module):
            def forward(self, x):
                torch.rand(1)
                return 2.0 * x

        self._refused(Draws().eval(), "RNG", backend="torch")

    def test_numpy_rng_consumer_is_refused(self):
        def op(u):
            np.random.random()
            return 2.0 * u

        self._refused(op, "RNG")

    def test_stateful_operator_without_rng_is_refused(self):
        # No RNG at all, but two forwards differ in the last bit: only the bitwise check sees it.
        class Drifting(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                return x * (1.0 + self.calls * 2.0 ** -40)

        # Non-zero base point: at the default base 0 the drift would vanish (0 * anything = 0).
        self._refused(Drifting().eval(), "bitwise", backend="torch", torch_dtype=torch.float64,
                      base_point=np.ones(self.q.shape[0]))

    def test_deterministic_operators_pass_and_keep_the_rng(self):
        # Positive control: deterministic module and numpy callable, RNG untouched.
        linear = torch.nn.Linear(self.q.shape[0], self.q.shape[0]).double().eval()  # init draws RNG
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        result = sa.input_jacobian_spectrum(linear, self.q, self.w)
        self.assertTrue(np.all(np.isfinite(result.projection)))
        numpy_result = sa.input_jacobian_spectrum(lambda u: 2.0 * u, self.q, self.w)
        self.assertTrue(np.all(np.isfinite(numpy_result.projection)))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        after = np.random.get_state()
        self.assertTrue(np.array_equal(numpy_state[1], after[1]) and numpy_state[2:] == after[2:])


class TestAuditVerdict(unittest.TestCase):
    def setUp(self):
        lattice, hm = _setup(TRI, 16)
        self.q, self.w = hm.modes, hm.weights
        self.k = self.q.shape[1]
        self.ok = sa.input_jacobian_spectrum(lambda u: u, self.q, self.w, oracle=np.ones(self.k))

    def test_pass_and_tolerance_validation(self):
        self.assertEqual(sa.audit_verdict(self.ok, diag_rtol=1e-12, coupling_tol=1e-12), "PASS")
        for bad in (None, float("nan"), float("inf"), -1e-12, True, "1e-12"):
            with self.subTest(bad=bad):
                self.assertEqual(sa.audit_verdict(self.ok, diag_rtol=bad, coupling_tol=1e-12), "ABSTAIN")
                self.assertEqual(sa.audit_verdict(self.ok, diag_rtol=1e-12, coupling_tol=bad), "ABSTAIN")
                self.assertEqual(sa.audit_verdict(self.ok, diag_rtol=1e-12, coupling_tol=1e-12, diag_atol=bad),
                                 "ABSTAIN")
                if bad is not None:
                    self.assertEqual(sa.audit_verdict(self.ok, diag_rtol=1e-12, coupling_tol=1e-12,
                                                      out_of_span_tol=bad), "ABSTAIN")

    def test_abstain_without_oracle_or_residual(self):
        no_oracle = sa.input_jacobian_spectrum(lambda u: u, self.q, self.w)
        self.assertIsNone(no_oracle.diag_rel_error)
        self.assertEqual(sa.audit_verdict(no_oracle, diag_rtol=1.0, coupling_tol=1.0), "ABSTAIN")
        reverse = sa.input_jacobian_spectrum(lambda u: u, self.q, self.w, backend="torch", torch_mode="reverse",
                                             oracle=np.ones(self.k))
        self.assertEqual(sa.audit_verdict(reverse, diag_rtol=1e-12, coupling_tol=1e-12), "PASS")
        self.assertEqual(sa.audit_verdict(reverse, diag_rtol=1e-12, coupling_tol=1e-12, out_of_span_tol=1.0),
                         "ABSTAIN")
        with self.assertRaises(TypeError):
            sa.audit_verdict({"projection": np.eye(2)}, diag_rtol=1.0, coupling_tol=1.0)

    def test_atol_and_zero_oracle(self):
        oracle = np.ones(self.k)
        oracle[0] = 0.0
        result = sa.input_jacobian_spectrum(lambda u: u, self.q, self.w, oracle=oracle)
        self.assertEqual(result.diag_rel_error[0], math.inf)
        self.assertEqual(sa.audit_verdict(result, diag_rtol=1e-12, coupling_tol=1e-12), "FAIL")
        self.assertEqual(sa.audit_verdict(result, diag_rtol=1e-12, coupling_tol=1e-12, diag_atol=1.0 + 1e-12),
                         "PASS")
        zero = sa.input_jacobian_spectrum(lambda u: 0.0 * u, self.q, self.w, oracle=np.zeros(self.k))
        self.assertTrue(np.all(zero.diag_rel_error == 0.0))
        self.assertEqual(zero.spectral_rel_error, 0.0)
        self.assertEqual(sa.audit_verdict(zero, diag_rtol=1e-12, coupling_tol=1e-12), "FAIL")  # no coupling value
        off = sa.input_jacobian_spectrum(lambda u: u, self.q, self.w, oracle=np.zeros(self.k))
        self.assertEqual(off.spectral_rel_error, math.inf)


class TestResponses(unittest.TestCase):
    def test_closed_forms(self):
        lam = np.array([0.0, 1.0, 50.0, 400.0])
        np.testing.assert_allclose(sa.semigroup_response(lam, kappa=2.0, t=0.01),
                                   [math.exp(-2.0 * x * 0.01) for x in lam], rtol=4 * EPS, atol=0.0)
        cn = sa.crank_nicolson_response(lam, kappa=1.0, dt=0.005, steps=3)
        expected = [((1 - x * 0.0025) / (1 + x * 0.0025)) ** 3 for x in lam]
        np.testing.assert_allclose(cn, expected, rtol=4 * EPS, atol=0.0)
        self.assertEqual(sa.crank_nicolson_response(np.array([400.0]), kappa=1.0, dt=0.005, steps=1)[0], 0.0)

    def test_invalid(self):
        for fn in (lambda lam: sa.semigroup_response(lam, kappa=1.0, t=1.0),
                   lambda lam: sa.crank_nicolson_response(lam, kappa=1.0, dt=1.0, steps=1)):
            for bad in (np.array([-1.0]), np.array([[1.0]]), np.array([np.nan]), np.array(["a"])):
                with self.subTest(bad=bad):
                    with self.assertRaises((ValueError, TypeError)):
                        fn(bad)
        with self.assertRaises(ValueError):
            sa.semigroup_response([1.0], kappa=0.0, t=1.0)
        with self.assertRaises(ValueError):
            sa.semigroup_response([1.0], kappa=1.0, t=float("inf"))
        with self.assertRaises(ValueError):
            sa.crank_nicolson_response([1.0], kappa=1.0, dt=1.0, steps=0)
        with self.assertRaises(TypeError):
            sa.crank_nicolson_response([1.0], kappa=1.0, dt=1.0, steps=True)
        with self.assertRaises(TypeError):
            sa.semigroup_response([1.0], kappa="1", t=1.0)

    def test_default_orthonormality_tol(self):
        self.assertEqual(sa.default_orthonormality_tol(105), 4.0 * 105 * EPS)
        with self.assertRaises(ValueError):
            sa.default_orthonormality_tol(0)


if __name__ == "__main__":
    unittest.main()
