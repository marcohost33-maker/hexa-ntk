"""
test_linear_baseline_hex_v3.py
==============================
Contract tests for the linear POD-ridge baseline (spec/SPECTRAL_AUDIT_v1.md,
section 6). The checker is the spectral audit: on exactly linear synthetic
data with enough samples the fitted map must reproduce the semidiscrete
semigroup on the mode basis. Synthetic only: t_final = 0.005 (not the T1
value) and coefficients from a private RNG, so no T1 data is fitted
(spec/PREREGISTRATION_PHASE1_v1.md).

Measured on 2026-10-13 (triangle and torus, N = 16): alpha_rel = 0 recovers
S to <= 8.5e-14; the default alpha_rel = 1e-10 with 2K..4K samples to
<= 9.3e-10 (ridge bias ~ alpha_rel * cond^2); fewer samples than modes fail.
"""

from __future__ import annotations

import unittest

import numpy as np

import spectral_audit_hex_v3 as sa
from heat_operator_task_hex_v3 import HeatOperatorTaskConfig
from hex_fv_solver_hex_v3 import assemble_hex_operator, torus_hex_lattice, triangle_hex_lattice
from linear_baseline_hex_v3 import LeakageError, LinearBaselineConfig, PODRidgeBaseline

T_SYNTH = 0.005


def _semigroup(lattice, t):
    diag, off = assemble_hex_operator(lattice, 1.0)
    dense = np.diag(np.array(diag, dtype=np.float64))
    for i in range(lattice.num_nodes):
        for s in range(6):
            j = lattice.neighbors[i, s]
            if j >= 0:
                dense[i, j] += off[i, s]
    mu, vec = np.linalg.eigh(dense)
    return (vec * np.exp(t * mu)) @ vec.T


class _Synthetic:
    def __init__(self, domain="triangle_dirichlet", n=16):
        self.config = HeatOperatorTaskConfig(domain=domain, t_final=T_SYNTH)
        self.lattice = triangle_hex_lattice(n) if domain == "triangle_dirichlet" else torus_hex_lattice(n)
        self.hm = sa.heat_mode_matrix(self.config, self.lattice)
        self.q, self.w = self.hm.modes, self.hm.weights
        self.k = self.q.shape[1]
        self.expm = _semigroup(self.lattice, T_SYNTH)
        self.oracle = sa.semigroup_response(self.hm.discrete_eigenvalues, kappa=1.0, t=T_SYNTH)

    def data(self, count, seed=123):
        coeff = np.random.default_rng(seed).standard_normal((count, self.k))
        u0 = coeff @ self.q.T
        return u0, u0 @ self.expm.T

    def audit(self, model):
        return sa.input_jacobian_spectrum(model, self.q, self.w, oracle=self.oracle)


TRI = _Synthetic()


class TestRecovery(unittest.TestCase):
    def test_recovers_the_semigroup_on_linear_data(self):
        for syn, factor in ((TRI, 2), (TRI, 4), (_Synthetic("torus_periodic"), 2)):
            with self.subTest(domain=syn.config.domain, samples=factor * syn.k):
                u0, ut = syn.data(factor * syn.k)
                model = PODRidgeBaseline(syn.w).fit(u0, ut, range(len(u0)))
                self.assertEqual((model.info["rank_in"], model.info["rank_out"]), (syn.k, syn.k))
                result = syn.audit(model)
                self.assertLessEqual(np.max(result.diag_rel_error), 1e-8)
                self.assertLessEqual(result.coupling_fraction, 1e-8)
                self.assertLessEqual(result.out_of_span_max, 1e-12)
                self.assertEqual(sa.audit_verdict(result, diag_rtol=1e-8, coupling_tol=1e-8,
                                                  out_of_span_tol=1e-12), "PASS")

    def test_unregularised_fit_is_exact_to_rounding(self):
        u0, ut = TRI.data(TRI.k)
        model = PODRidgeBaseline(TRI.w, LinearBaselineConfig(alpha_rel=0.0)).fit(u0, ut, range(TRI.k))
        result = TRI.audit(model)
        self.assertLessEqual(np.max(result.diag_rel_error), 1e-12)
        self.assertLessEqual(result.coupling_fraction, 1e-12)
        self.assertEqual(model.info["alpha"], 0.0)

    def test_ridge_bias_is_linear_in_alpha(self):
        u0, ut = TRI.data(2 * TRI.k)
        errors = []
        for alpha_rel in (1e-10, 1e-6):
            model = PODRidgeBaseline(TRI.w, LinearBaselineConfig(alpha_rel=alpha_rel)).fit(u0, ut, range(len(u0)))
            s_in = model.info["singular_values_in"]
            self.assertAlmostEqual(model.info["alpha"] / (alpha_rel * s_in[0] ** 2), 1.0, delta=1e-12)
            errors.append(np.max(TRI.audit(model).diag_rel_error))
        self.assertTrue(5e3 <= errors[1] / errors[0] <= 2e4, errors)
        self.assertEqual(sa.audit_verdict(TRI.audit(model), diag_rtol=1e-8, coupling_tol=1e-8), "FAIL")

    def test_too_few_samples_or_truncated_rank_fail_the_audit(self):
        u0, ut = TRI.data(TRI.k // 2)
        few = PODRidgeBaseline(TRI.w).fit(u0, ut, range(len(u0)))
        self.assertEqual(few.info["rank_in"], TRI.k // 2)
        self.assertEqual(sa.audit_verdict(TRI.audit(few), diag_rtol=1e-3, coupling_tol=1e-3), "FAIL")
        u0, ut = TRI.data(4 * TRI.k)
        capped = PODRidgeBaseline(TRI.w, LinearBaselineConfig(max_rank=5)).fit(u0, ut, range(len(u0)))
        self.assertEqual((capped.info["rank_in"], capped.info["rank_out"]), (5, 5))
        self.assertEqual(sa.audit_verdict(TRI.audit(capped), diag_rtol=1e-3, coupling_tol=1e-3), "FAIL")

    def test_rank_rtol_rule(self):
        u0, ut = TRI.data(4 * TRI.k)
        noisy = u0 + 1e-9 * np.random.default_rng(5).standard_normal(u0.shape)
        default = PODRidgeBaseline(TRI.w).fit(noisy, ut, range(len(u0)))
        self.assertGreater(default.info["rank_in"], TRI.k)  # numerical rank sees the noise
        loose = PODRidgeBaseline(TRI.w, LinearBaselineConfig(rank_rtol=1e-6)).fit(noisy, ut, range(len(u0)))
        self.assertEqual(loose.info["rank_in"], TRI.k)


class TestViewsAndInvariance(unittest.TestCase):
    def setUp(self):
        self.u0, self.ut = TRI.data(3 * TRI.k)
        self.ids = list(range(len(self.u0)))
        self.model = PODRidgeBaseline(TRI.w).fit(self.u0, self.ut, self.ids)

    def test_views_agree(self):
        test_u0, _ = TRI.data(5, seed=999)
        g = self.model.matrix()
        pred = self.model.predict(test_u0, [1000 + i for i in range(5)])
        np.testing.assert_allclose(pred, test_u0 @ g.T, rtol=0, atol=1e-13 * np.max(np.abs(pred)))
        np.testing.assert_allclose(self.model(test_u0[0]), pred[0], rtol=0, atol=1e-13 * np.max(np.abs(pred)))

    def test_deterministic(self):
        again = PODRidgeBaseline(TRI.w).fit(self.u0, self.ut, self.ids)
        np.testing.assert_array_equal(again.matrix(), self.model.matrix())

    def test_scale_equivariance(self):
        g = self.model.matrix()
        scale = np.max(np.abs(g))
        for c in (1e-30, 1e30):
            with self.subTest(c=c):
                both = PODRidgeBaseline(TRI.w).fit(c * self.u0, c * self.ut, self.ids)
                self.assertLessEqual(np.max(np.abs(both.matrix() - g)) / scale, 1e-12)
                inputs = PODRidgeBaseline(TRI.w).fit(c * self.u0, self.ut, self.ids)
                self.assertLessEqual(np.max(np.abs(c * inputs.matrix() - g)) / scale, 1e-12)

    def test_training_order_invariance(self):
        perm = np.random.default_rng(3).permutation(len(self.u0))
        shuffled = PODRidgeBaseline(TRI.w).fit(self.u0[perm], self.ut[perm], [self.ids[i] for i in perm])
        g = self.model.matrix()
        self.assertLessEqual(np.max(np.abs(shuffled.matrix() - g)) / np.max(np.abs(g)), 1e-12)

    def test_fitted_arrays_read_only(self):
        with self.assertRaises(ValueError):
            self.model.weights[0] = 1.0
        info = self.model.info
        info["singular_values_in"][0] = -1.0  # a copy
        self.assertGreater(self.model.info["singular_values_in"][0], 0.0)


class TestLeakageGuard(unittest.TestCase):
    def setUp(self):
        self.u0, self.ut = TRI.data(2 * TRI.k)
        self.model = PODRidgeBaseline(TRI.w).fit(self.u0, self.ut, list(range(100, 100 + len(self.u0))))

    def test_trained_ids_are_refused(self):
        self.assertEqual(self.model.train_ids, frozenset(range(100, 100 + len(self.u0))))
        with self.assertRaises(LeakageError):
            self.model.predict(self.u0[:1], [100])
        with self.assertRaisesRegex(LeakageError, "1 scored sample ids"):
            self.model.predict(self.u0[:3], [7, 8, 100 + 5])
        self.assertTrue(issubclass(LeakageError, ValueError))
        # Fresh content under fresh ids (training rows under fresh ids are a content leak since 2026-10-15).
        fresh, _ = TRI.data(2, seed=778)
        self.assertEqual(self.model.predict(fresh, [7, 8]).shape, (2, TRI.lattice.num_nodes))

    def test_content_duplicate_with_a_new_id_is_leakage(self):
        # B9 (2026-10-15): the guard compared ids only; a training row scored
        # under a fresh id, or with -0.0 where training had 0.0, slipped through.
        with self.assertRaises(Exception) as caught:
            self.model.predict(self.u0[3:4].copy(), [999])
        got = type(caught.exception).__name__
        self.assertIsInstance(caught.exception, LeakageError, f"expected LeakageError, got {got}")

    def test_signed_zero_duplicate_is_leakage(self):
        u0 = self.u0.copy()
        u0[2, :7] = 0.0
        model = PODRidgeBaseline(TRI.w).fit(u0, self.ut, list(range(len(u0))))
        probe = u0[2:3].copy()
        probe[0, :7] = -0.0
        self.assertTrue(np.all(np.signbit(probe[0, :7])))
        self.assertNotEqual(probe.tobytes(), u0[2:3].tobytes())  # differs bytewise, equal as numbers
        with self.assertRaises(Exception) as caught:
            model.predict(probe, [500])
        got = type(caught.exception).__name__
        self.assertIsInstance(caught.exception, LeakageError, f"expected LeakageError, got {got}")

    def test_disjoint_content_with_new_ids_is_scored(self):
        # Positive control: fresh samples (other seed) under fresh ids are scored.
        fresh, _ = TRI.data(3, seed=777)
        out = self.model.predict(fresh, [7, 8, 9])
        self.assertEqual(out.shape, (3, TRI.lattice.num_nodes))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_refit_replaces_the_training_ids(self):
        self.model.fit(self.u0, self.ut, list(range(len(self.u0))))
        fresh, _ = TRI.data(1, seed=779)
        self.assertEqual(self.model.predict(fresh, [100]).shape, (1, TRI.lattice.num_nodes))
        with self.assertRaises(LeakageError):
            self.model.predict(self.u0[:1], [0])

    def test_id_validation(self):
        for bad, exc in (([True], TypeError), ([-1], ValueError), ([1.0], TypeError), ([3, 3], ValueError),
                         ([1, 2], ValueError), ("1", TypeError), ([np.int64(4)], None)):
            with self.subTest(bad=bad):
                rows = self.u0[:len(bad)] if not isinstance(bad, str) else self.u0[:1]
                if exc is None:
                    fresh, _ = TRI.data(1, seed=780)  # training rows are a content leak since 2026-10-15
                    self.assertEqual(self.model.predict(fresh, bad).shape[0], 1)
                    continue
                if bad == [1, 2]:
                    rows = self.u0[:1]  # length mismatch
                with self.assertRaises(exc):
                    self.model.predict(rows, bad)
                with self.assertRaises(exc):
                    PODRidgeBaseline(TRI.w).fit(rows, rows, bad)


class TestFailClosed(unittest.TestCase):
    def test_config_validation(self):
        for kwargs, exc in ((dict(alpha_rel=-1e-10), ValueError), (dict(alpha_rel=float("nan")), ValueError),
                            (dict(alpha_rel=float("inf")), ValueError), (dict(alpha_rel=True), TypeError),
                            (dict(rank_rtol=0.0), ValueError), (dict(rank_rtol=1.0), ValueError),
                            (dict(rank_rtol=float("nan")), ValueError), (dict(rank_rtol="1e-6"), TypeError),
                            (dict(max_rank=0), ValueError), (dict(max_rank=2.0), TypeError),
                            (dict(max_rank=True), TypeError)):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(exc):
                    LinearBaselineConfig(**kwargs)

    def test_weights_and_config_type(self):
        for bad, exc in ((np.zeros(3), ValueError), (-np.ones(3), ValueError), (np.full(3, np.nan), ValueError),
                         (np.ones((2, 2)), ValueError), (np.array([], dtype=float), ValueError),
                         (np.ones(3, dtype=bool), TypeError)):
            with self.subTest(bad=bad):
                with self.assertRaises(exc):
                    PODRidgeBaseline(bad)
        with self.assertRaises(TypeError):
            PODRidgeBaseline(np.ones(3), config={"alpha_rel": 0.0})

    def test_all_zero_training_fails_by_assertion(self):
        # Fails by assertion for ANY other outcome (no exception, or a crash such as IndexError).
        m = TRI.lattice.num_nodes
        u0, ut = TRI.data(4)
        for x, y in ((np.zeros((4, m)), ut), (u0, np.zeros((4, m)))):
            outcome = None
            try:
                PODRidgeBaseline(TRI.w).fit(x, y, range(4))
            except Exception as exc:  # noqa: BLE001 - the type is what is checked
                outcome = exc
            self.assertIsInstance(outcome, ValueError, repr(outcome))
            self.assertIn("all zero", str(outcome))

    def test_data_validation(self):
        model = PODRidgeBaseline(TRI.w)
        u0, ut = TRI.data(4)
        m = TRI.lattice.num_nodes
        with self.assertRaises(RuntimeError):
            model.predict(u0, range(4))
        with self.assertRaises(RuntimeError):
            model.matrix()
        bad_nan = u0.copy()
        bad_nan[-1, -1] = np.nan
        cases = ((bad_nan, ut, ValueError), (u0, ut[:3], ValueError), (u0[:, :-1], ut[:, :-1], ValueError),
                 (u0[0], ut[0], ValueError), (np.zeros((4, m)), ut, ValueError),
                 (u0, np.zeros((4, m)), ValueError), (u0.astype(complex), ut, TypeError),
                 (np.zeros((0, m)), np.zeros((0, m)), ValueError))
        for x, y, exc in cases:
            with self.subTest(shape=np.shape(x), dtype=np.asarray(x).dtype):
                with self.assertRaises(exc):
                    model.fit(x, y, range(len(x)))
        model.fit(u0, ut, range(4))
        with self.assertRaises(ValueError):
            model(u0[:2])
        with self.assertRaises(ValueError):
            model(np.full(m, np.inf))


if __name__ == "__main__":
    unittest.main()
