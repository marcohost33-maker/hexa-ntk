"""
test_spectral_audit_oracles_hex_v3.py
=====================================
Oracle tests for the spectral audit (spec/SPECTRAL_AUDIT_v1.md, section 4):
(i) the exact semidiscrete semigroup exp(kappa T L_h), (ii) the Crank-Nicolson
mechanistic baseline, (iii) negative controls (mode swap, wrong frequency
response, out-of-span leakage, sensitivity just above the tolerance) and the
metamorphic relations (base-point and operator scale, node permutation, mode
order, rotation inside a degenerate eigenspace). The operators are built from
the dense assembled L_h, independent of the mode basis under test.

Oracle tolerances: the measured worst case of every oracle below is <= 3.2e-14
relative (probe of 2026-10-13, N = 16 and 32, both domains); the tests allow
1e-12, about 30x margin, and a sensitivity test pins that a 1e-10 error in the
decay rate is still detected (the tolerance is not blind).
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import spectral_audit_hex_v3 as sa
from heat_operator_task_hex_v3 import HeatOperatorTaskConfig, mechanistic_baseline
from hex_fv_solver_hex_v3 import assemble_hex_operator, torus_hex_lattice, triangle_hex_lattice

TRI = HeatOperatorTaskConfig()
TORUS = HeatOperatorTaskConfig(domain="torus_periodic")
ORACLE_RTOL = 1e-12
EPS = float(np.finfo(np.float64).eps)


def _lattice(config, n):
    return triangle_hex_lattice(n) if config.domain == "triangle_dirichlet" else torus_hex_lattice(n)


def _dense_laplacian(lattice):
    """L_h as a dense matrix from the assembled stencil (independent of the mode basis)."""
    diag, off = assemble_hex_operator(lattice, 1.0)
    dense = np.diag(np.array(diag, dtype=np.float64))
    for i in range(lattice.num_nodes):
        for s in range(6):
            j = lattice.neighbors[i, s]
            if j >= 0:
                dense[i, j] += off[i, s]
    return dense


def _semigroup_matrix(lattice, kappa, t, rate_factor=1.0):
    """exp(rate_factor * kappa * t * L_h) via the symmetric eigendecomposition of the dense L_h."""
    mu, vec = np.linalg.eigh(_dense_laplacian(lattice))
    return (vec * np.exp(rate_factor * kappa * t * mu)) @ vec.T


def _setup(config, n):
    lattice = _lattice(config, n)
    return lattice, sa.heat_mode_matrix(config, lattice)


class TestOracleSemidiscrete(unittest.TestCase):
    """(i) exp(kappa T L_h): S = diag(exp(-kappa lambda_h T))."""

    def test_exact_semidiscrete_operator(self):
        for config, n in ((TRI, 16), (TRI, 32), (TORUS, 16)):
            with self.subTest(domain=config.domain, n=n):
                lattice, hm = _setup(config, n)
                expm = _semigroup_matrix(lattice, config.kappa, config.t_final)
                oracle = sa.semigroup_response(hm.discrete_eigenvalues, kappa=config.kappa, t=config.t_final)
                result = sa.input_jacobian_spectrum(lambda u: expm @ u, hm.modes, hm.weights, oracle=oracle)
                self.assertLessEqual(np.max(result.diag_rel_error), ORACLE_RTOL)
                self.assertLessEqual(result.coupling_fraction, ORACLE_RTOL)
                self.assertLessEqual(result.out_of_span_max, ORACLE_RTOL)
                self.assertLessEqual(result.spectral_rel_error, ORACLE_RTOL)
                self.assertEqual(sa.audit_verdict(result, diag_rtol=ORACLE_RTOL, coupling_tol=ORACLE_RTOL,
                                                  out_of_span_tol=ORACLE_RTOL), "PASS")
                self.assertEqual(result.backend, "numpy")
                self.assertEqual(result.evaluations, 2 * hm.modes.shape[1] + 1)

    def test_semidiscrete_is_not_the_continuous_t1_truth(self):
        # The T1 truth decays with lambda, the semidiscrete operator with lambda_h:
        # on N = 16 the highest mode differs by ~97 % (spatial discretisation error).
        lattice, hm = _setup(TRI, 16)
        expm = _semigroup_matrix(lattice, TRI.kappa, TRI.t_final)
        truth = sa.semigroup_response(hm.eigenvalues, kappa=TRI.kappa, t=TRI.t_final)
        result = sa.input_jacobian_spectrum(lambda u: expm @ u, hm.modes, hm.weights, oracle=truth)
        self.assertGreater(np.max(result.diag_rel_error), 0.5)
        self.assertLessEqual(result.coupling_fraction, ORACLE_RTOL)
        self.assertEqual(sa.audit_verdict(result, diag_rtol=1e-3, coupling_tol=ORACLE_RTOL), "FAIL")


class TestOracleCrankNicolson(unittest.TestCase):
    """(ii) mechanistic_baseline: S = diag(((1 - z/2) / (1 + z/2))^steps), z = kappa lambda_h dt."""

    def test_crank_nicolson_baseline(self):
        for config, steps in ((TRI, 1), (TRI, 4), (TRI, 32), (TORUS, 4)):
            with self.subTest(domain=config.domain, steps=steps):
                lattice, hm = _setup(config, 16)
                dt = config.t_final / steps
                oracle = sa.crank_nicolson_response(hm.discrete_eigenvalues, kappa=config.kappa, dt=dt, steps=steps)
                result = sa.input_jacobian_spectrum(
                    lambda u: mechanistic_baseline(config, u, lattice, steps=steps)[0],
                    hm.modes, hm.weights, oracle=oracle)
                self.assertLessEqual(np.max(result.diag_rel_error), ORACLE_RTOL)
                self.assertLessEqual(result.coupling_fraction, ORACLE_RTOL)
                # CG stops at rtol = 1e-12 per step: the out-of-span residual sits at that level.
                self.assertLessEqual(result.out_of_span_max, 1e-10)
                self.assertEqual(sa.audit_verdict(result, diag_rtol=ORACLE_RTOL, coupling_tol=ORACLE_RTOL), "PASS")
                if steps == 1:
                    # z > 2 for the high modes: CN flips their sign (no L-stability); the audit sees it.
                    self.assertTrue(np.any(result.diagonal < 0.0))

    def test_closed_form_against_dense_crank_nicolson(self):
        # The oracle formula itself, against a dense matrix power independent of the mode basis.
        lattice, hm = _setup(TRI, 16)
        dense = _dense_laplacian(lattice)
        steps, dt = 3, TRI.t_final / 3
        eye = np.eye(lattice.num_nodes)
        step_matrix = np.linalg.solve(eye - 0.5 * dt * dense, eye + 0.5 * dt * dense)
        power = np.linalg.matrix_power(step_matrix, steps)
        s = hm.modes.T @ (hm.weights[:, None] * (power @ hm.modes))
        oracle = sa.crank_nicolson_response(hm.discrete_eigenvalues, kappa=1.0, dt=dt, steps=steps)
        self.assertLessEqual(np.max(np.abs(np.diag(s) - oracle) / np.abs(oracle)), ORACLE_RTOL)


class TestNegativeControls(unittest.TestCase):
    def setUp(self):
        self.lattice, self.hm = _setup(TRI, 16)
        self.q, self.w = self.hm.modes, self.hm.weights
        self.oracle = sa.semigroup_response(self.hm.discrete_eigenvalues, kappa=1.0, t=TRI.t_final)

    def _audit(self, matrix):
        return sa.input_jacobian_spectrum(lambda u: matrix @ u, self.q, self.w, oracle=self.oracle)

    def test_mode_swap_detected_as_coupling(self):
        perm = np.arange(self.q.shape[1])
        perm[[0, 1]] = [1, 0]
        swap = self.q[:, perm] @ np.diag(self.oracle[perm]) @ self.q.T @ np.diag(self.w)
        result = self._audit(swap)
        self.assertGreater(result.coupling_fraction, 0.5)
        self.assertEqual(sa.audit_verdict(result, diag_rtol=ORACLE_RTOL, coupling_tol=ORACLE_RTOL), "FAIL")
        # A pure coupling defect is visible only in the coupling: the other modes' diagonal is exact.
        self.assertLessEqual(np.max(result.diag_rel_error[2:]), ORACLE_RTOL)

    def test_wrong_frequency_response_detected(self):
        result = self._audit(_semigroup_matrix(self.lattice, 1.0, TRI.t_final, rate_factor=2.0))
        self.assertGreater(np.min(result.diag_rel_error), 0.3)
        self.assertLessEqual(result.coupling_fraction, ORACLE_RTOL)
        self.assertEqual(sa.audit_verdict(result, diag_rtol=ORACLE_RTOL, coupling_tol=ORACLE_RTOL), "FAIL")

    def test_tolerance_not_blind_to_small_errors(self):
        # Pins the upper end of the tolerance: a 1e-10 relative rate error and a 1e-9
        # admixture of a swapped pair must both be measurable at ORACLE_RTOL.
        rate = self._audit(_semigroup_matrix(self.lattice, 1.0, TRI.t_final, rate_factor=1.0 + 1e-10))
        self.assertEqual(sa.audit_verdict(rate, diag_rtol=ORACLE_RTOL, coupling_tol=ORACLE_RTOL), "FAIL")
        exact = _semigroup_matrix(self.lattice, 1.0, TRI.t_final)
        leak = 1e-9 * np.outer(self.q[:, 1], self.w * self.q[:, 0])
        mixed = self._audit(exact + leak)
        self.assertGreater(mixed.coupling_fraction, 1e-10)
        self.assertEqual(sa.audit_verdict(mixed, diag_rtol=ORACLE_RTOL, coupling_tol=ORACLE_RTOL), "FAIL")

    def test_out_of_span_leak_invisible_in_s_but_measured(self):
        exact = _semigroup_matrix(self.lattice, 1.0, TRI.t_final)
        rng = np.random.default_rng(7)
        v = rng.standard_normal(self.lattice.num_nodes)
        v -= self.q @ (self.q.T @ (self.w * v))  # W-orthogonal to span(Q)
        v /= math.sqrt(float(np.sum(self.w * v * v)))
        leaky = exact + 0.1 * np.outer(v, self.w * self.q[:, 3])
        result = self._audit(leaky)
        # S cannot see it ...
        self.assertEqual(sa.audit_verdict(result, diag_rtol=ORACLE_RTOL, coupling_tol=ORACLE_RTOL), "PASS")
        # ... the out-of-span residual does, for exactly that input mode.
        self.assertGreater(result.out_of_span[3], 0.5)
        self.assertLessEqual(np.max(np.delete(result.out_of_span, 3)), ORACLE_RTOL)
        self.assertEqual(sa.audit_verdict(result, diag_rtol=ORACLE_RTOL, coupling_tol=ORACLE_RTOL,
                                          out_of_span_tol=1e-6), "FAIL")


class TestMetamorphic(unittest.TestCase):
    def setUp(self):
        self.lattice, self.hm = _setup(TRI, 16)
        self.q, self.w = self.hm.modes, self.hm.weights
        rng = np.random.default_rng(11)
        self.matrix = rng.standard_normal((self.lattice.num_nodes, self.lattice.num_nodes))
        self.reference = self.q.T @ (self.w[:, None] * (self.matrix @ self.q))
        self.scale = np.max(np.abs(self.reference))

    def test_base_point_scale_does_not_change_s_of_a_linear_operator(self):
        rng = np.random.default_rng(12)
        for amplitude in (0.0, 1.0, 1e3, 1e6):
            with self.subTest(amplitude=amplitude):
                base = amplitude * rng.standard_normal(self.lattice.num_nodes)
                result = sa.input_jacobian_spectrum(lambda u: self.matrix @ u, self.q, self.w, base_point=base)
                # h scales with ||u0||_W, so the rounding error is ~eps^(2/3) ~ 4e-11 at worst.
                self.assertLessEqual(np.max(np.abs(result.projection - self.reference)) / self.scale, 1e-9)

    def test_operator_scale_covariance(self):
        base = sa.input_jacobian_spectrum(lambda u: self.matrix @ u, self.q, self.w)
        for factor in (1e-30, 1e30):
            with self.subTest(factor=factor):
                result = sa.input_jacobian_spectrum(lambda u: factor * (self.matrix @ u), self.q, self.w)
                self.assertLessEqual(np.max(np.abs(result.projection / factor - base.projection)) / self.scale, 1e-13)
                self.assertAlmostEqual(result.coupling_fraction, base.coupling_fraction, delta=1e-13)

    def test_node_permutation_invariance(self):
        rng = np.random.default_rng(13)
        perm = rng.permutation(self.lattice.num_nodes)
        permuted = self.matrix[np.ix_(perm, perm)]
        result = sa.input_jacobian_spectrum(lambda u: permuted @ u, self.q[perm], self.w[perm])
        self.assertLessEqual(np.max(np.abs(result.projection - self.reference)) / self.scale, 1e-13)

    def test_mode_order_permutation_permutes_s(self):
        perm = np.random.default_rng(14).permutation(self.q.shape[1])
        result = sa.input_jacobian_spectrum(lambda u: self.matrix @ u, self.q[:, perm], self.w)
        self.assertLessEqual(np.max(np.abs(result.projection - self.reference[np.ix_(perm, perm)])) / self.scale,
                             1e-13)

    def test_rotation_inside_a_degenerate_eigenspace_keeps_the_truth_diagonal(self):
        lam = self.hm.discrete_eigenvalues
        pairs = [(i, i + 1) for i in range(len(lam) - 1) if lam[i] == lam[i + 1]]
        self.assertTrue(pairs, "T1 triangle basis has sin/cos pairs with equal lambda")
        i, j = pairs[0]
        c, s = math.cos(0.7), math.sin(0.7)
        rotated = self.q.copy()
        rotated[:, i], rotated[:, j] = c * self.q[:, i] - s * self.q[:, j], s * self.q[:, i] + c * self.q[:, j]
        expm = _semigroup_matrix(self.lattice, 1.0, TRI.t_final)
        oracle = sa.semigroup_response(lam, kappa=1.0, t=TRI.t_final)
        for basis in (self.q, rotated):
            result = sa.input_jacobian_spectrum(lambda u: expm @ u, basis, self.w, oracle=oracle)
            self.assertEqual(sa.audit_verdict(result, diag_rtol=ORACLE_RTOL, coupling_tol=ORACLE_RTOL), "PASS")


if __name__ == "__main__":
    unittest.main()
