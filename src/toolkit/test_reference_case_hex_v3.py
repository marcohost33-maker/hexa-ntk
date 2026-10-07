"""
test_reference_case_hex_v3.py
=============================
Contract tests for the Reality Anchor Phase 0a reference case ``hex-heat-tri``
(spec/REFERENCE_CASE_HEX_HEAT_TRI_v1.md). The full N = 16..128 study runs in
well under a second, so CI re-derives the gate on every run.
"""

from __future__ import annotations

import json
import math
import unittest
from unittest import mock

import numpy as np

import reference_case_hex_v3 as rc
from reference_case_hex_v3 import (
    HexHeatTriConfig,
    LameMode,
    apply_dirichlet_hex_laplacian,
    assemble_dirichlet_hex_operator,
    celik_three_grid,
    explicit_euler_heat,
    hex_heat_tri_reference_spec,
    observed_order,
    positivity_max_dt,
    run_hex_heat_tri_study,
    triangle_hex_lattice,
)
from reference_spec_hex_v3 import ClaimCeiling, EvidenceStatus

_SQRT3 = math.sqrt(3.0)


def _boundary_coords(n: int) -> np.ndarray:
    h = 1.0 / n
    pts = [(h * (i + j / 2), h * j * _SQRT3 / 2)
           for j in range(n + 1) for i in range(n + 1 - j) if i == 0 or j == 0 or i + j == n]
    return np.asarray(pts)


class TestTriangleLattice(unittest.TestCase):
    def test_node_count_and_spacing(self) -> None:
        for n in (3, 4, 7, 16, 33):
            with self.subTest(n=n):
                lat = triangle_hex_lattice(n)
                self.assertEqual(lat.num_nodes, (n - 1) * (n - 2) // 2)
                self.assertEqual(lat.h, 1.0 / n)
                self.assertEqual(lat.neighbors.shape, (lat.num_nodes, 6))

    def test_smallest_lattice_has_only_dirichlet_neighbours(self) -> None:
        lat = triangle_hex_lattice(3)
        self.assertEqual(lat.neighbors.tolist(), [[-1] * 6])

    def test_neighbour_geometry_and_symmetry(self) -> None:
        lat = triangle_hex_lattice(9)
        h = lat.h
        for s, angle in enumerate(np.arange(6) * math.pi / 3):
            step = h * np.array([math.cos(angle), math.sin(angle)])
            rows = np.nonzero(lat.neighbors[:, s] >= 0)[0]
            nb = lat.neighbors[rows, s]
            np.testing.assert_allclose(lat.coords[nb] - lat.coords[rows], np.tile(step, (rows.size, 1)), atol=1e-14)
            # opposite slot points back
            np.testing.assert_array_equal(lat.neighbors[nb, (s + 3) % 6], rows)

    def test_dirichlet_slots_point_to_boundary_nodes(self) -> None:
        lat = triangle_hex_lattice(8)
        boundary = {tuple(np.round(p, 12)) for p in _boundary_coords(8)}
        for s, (di, dj) in enumerate(rc.LATTICE_SLOTS):
            step = lat.h * np.array([di + dj / 2, dj * _SQRT3 / 2])
            for row in np.nonzero(lat.neighbors[:, s] < 0)[0]:
                self.assertIn(tuple(np.round(lat.coords[row] + step, 12)), boundary)

    def test_arrays_are_read_only(self) -> None:
        lat = triangle_hex_lattice(5)
        with self.assertRaises(ValueError):
            lat.neighbors[0, 0] = 0
        with self.assertRaises(ValueError):
            lat.coords[0, 0] = 0.0

    def test_invalid_n(self) -> None:
        for bad, exc in ((2, ValueError), (0, ValueError), (True, TypeError), (16.0, TypeError), ("16", TypeError)):
            with self.subTest(n=bad), self.assertRaises(exc):
                triangle_hex_lattice(bad)


class TestLameModesAreExactDiscreteEigenvectors(unittest.TestCase):
    MODES = (LameMode(1, 1, "sin"), LameMode(1, 2, "sin"), LameMode(1, 2, "cos"),
             LameMode(2, 3, "sin"), LameMode(3, 1, "cos"))

    def test_modes_vanish_on_the_boundary(self) -> None:
        for n in (8, 15):
            pts = _boundary_coords(n)
            for mode in self.MODES:
                with self.subTest(n=n, mode=mode):
                    self.assertLess(np.max(np.abs(mode.evaluate(pts))), 1e-12)

    def test_exact_eigen_residual_on_odd_and_even_lattices(self) -> None:
        for n in (7, 15, 16, 31):
            lat = triangle_hex_lattice(n)
            for mode in self.MODES:
                with self.subTest(n=n, mode=mode):
                    phi = mode.evaluate(lat.coords)
                    lam_h = mode.discrete_eigenvalue(lat.h)
                    residual = apply_dirichlet_hex_laplacian(phi, lat) + lam_h * phi
                    self.assertLess(np.max(np.abs(residual)) / (lam_h * np.max(np.abs(phi))), 1e-12)

    def test_continuous_eigenvalue_formula(self) -> None:
        # lambda = |k|^2 = (16 pi^2 / 9)(m^2 + mn + n^2); fundamental 16 pi^2 / 3
        self.assertAlmostEqual(LameMode(1, 1).eigenvalue, 16 * math.pi**2 / 3, places=12)
        for mode in self.MODES:
            with self.subTest(mode=mode):
                self.assertAlmostEqual(float(mode.wavevector @ mode.wavevector), mode.eigenvalue, places=9)

    def test_isotropic_leading_error_term(self) -> None:
        for mode in self.MODES:
            for h in (1 / 64, 1 / 256):
                with self.subTest(mode=mode, h=h):
                    lam = mode.eigenvalue
                    ratio = (lam - mode.discrete_eigenvalue(h)) / (h * h * lam * lam / 16)
                    self.assertAlmostEqual(ratio, 1.0, delta=0.02 if h == 1 / 64 else 0.002)

    def test_fundamental_matches_the_form_in_issue_60(self) -> None:
        lat = triangle_hex_lattice(32)
        x, y = lat.coords[:, 0], lat.coords[:, 1]
        issue60 = np.sin(4 * np.pi * y / _SQRT3) - np.sin(2 * np.pi * (x + y / _SQRT3)) \
            + np.sin(2 * np.pi * (x - y / _SQRT3))
        np.testing.assert_allclose(LameMode(1, 1).evaluate(lat.coords), -2.0 * issue60, atol=1e-12)

    def test_fundamental_has_one_sign(self) -> None:
        phi = LameMode(1, 1).evaluate(triangle_hex_lattice(40).coords)
        self.assertTrue(np.all(phi < 0.0))

    def test_invalid_modes(self) -> None:
        for args, exc in (((1, 1, "cos"), ValueError), ((0, 1), ValueError), ((1, 0), ValueError),
                          ((1, 1, "tan"), ValueError), ((True, 1), TypeError), ((1.0, 1), TypeError)):
            with self.subTest(args=args), self.assertRaises(exc):
                LameMode(*args)


class TestOperatorUsesEngineWeight(unittest.TestCase):
    def test_weight_comes_from_the_engine(self) -> None:
        lat = triangle_hex_lattice(6)
        u = np.linspace(-1.0, 2.0, lat.num_nodes)
        with mock.patch.object(rc._engine(), "hex_laplacian_stencil_weight", return_value=1.0) as weight:
            unit = apply_dirichlet_hex_laplacian(u, lat)
        weight.assert_called_once_with(lat.h)
        expected = np.where(lat.neighbors >= 0, u[np.clip(lat.neighbors, 0, None)], 0.0).sum(1) - 6 * u
        np.testing.assert_allclose(unit, expected, rtol=0, atol=0)
        np.testing.assert_allclose(apply_dirichlet_hex_laplacian(u, lat), unit * 2 / (3 * lat.h**2), rtol=1e-14)

    def test_field_contract(self) -> None:
        lat = triangle_hex_lattice(5)
        with self.assertRaises(ValueError):
            apply_dirichlet_hex_laplacian(np.zeros(lat.num_nodes + 1), lat)
        with self.assertRaises(ValueError):
            apply_dirichlet_hex_laplacian(np.full(lat.num_nodes, np.nan), lat)
        with self.assertRaises(TypeError):
            apply_dirichlet_hex_laplacian(np.zeros(lat.num_nodes), object())


class TestConvergenceAnalysis(unittest.TestCase):
    def test_observed_order_and_gci_on_synthetic_second_order_data(self) -> None:
        f0, c = 3.0, 0.7
        f = [f0 + c * h**2 for h in (0.25, 0.5, 1.0)]  # finest first
        out = celik_three_grid(*f, ratio=2.0)
        self.assertAlmostEqual(out["observed_order"], 2.0, places=12)
        self.assertFalse(out["oscillatory"])
        self.assertAlmostEqual(out["extrapolated"], f0, places=12)
        e_a21 = abs((f[0] - f[1]) / f[0])
        self.assertAlmostEqual(out["gci_fine"], 1.25 * e_a21 / 3.0, places=15)
        self.assertAlmostEqual(observed_order(4.0, 1.0, 2.0), 2.0, places=15)

    def test_oscillatory_convergence_is_flagged(self) -> None:
        self.assertTrue(celik_three_grid(1.0, 1.1, 0.7)["oscillatory"])
        out = celik_three_grid(1.0, 2.0, 0.0)  # eps32/eps21 = -2: flagged, not an error
        self.assertTrue(out["oscillatory"])
        self.assertAlmostEqual(out["observed_order"], 1.0, places=15)

    def test_degenerate_inputs_fail_closed(self) -> None:
        for args in ((1.0, 1.0, 2.0), (1.0, 2.0, 2.0), (0.0, 1.0, 3.0), (1.0, 2.0, 3.0, 1.0),
                     (float("nan"), 1.0, 2.0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                celik_three_grid(*args)
        for args in ((0.0, 1.0), (1.0, -1.0), (1.0, 2.0, 1.0), (float("inf"), 1.0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                observed_order(*args)


class TestPositivityGateIssue60(unittest.TestCase):
    def test_bound_matches_the_engine_positivity_contract(self) -> None:
        engine = rc._engine()
        for n in (8, 16, 33):
            for kappa in (1.0, 0.3, 7.5):
                with self.subTest(n=n, kappa=kappa):
                    lat = triangle_hex_lattice(n)
                    dt = positivity_max_dt(*assemble_dirichlet_hex_operator(lat, kappa))
                    ref = engine.hex_positivity_max_dt(lat.h, kappa)
                    self.assertLessEqual(abs(dt - ref), 4 * np.finfo(float).eps * ref)

    def test_sign_structure_violations_raise(self) -> None:
        diag, off = assemble_dirichlet_hex_operator(triangle_hex_lattice(6))
        bad_off = off.copy()
        bad_off[0, 0] = -1e-9
        with self.assertRaisesRegex(ValueError, "M-matrix"):
            positivity_max_dt(diag, bad_off)
        bad_diag = diag.copy()
        bad_diag[-1] = 0.0
        with self.assertRaisesRegex(ValueError, "diagonal"):
            positivity_max_dt(bad_diag, off)
        with self.assertRaises(ValueError):
            positivity_max_dt(diag, off[:-1])
        with self.assertRaises(ValueError):
            positivity_max_dt(np.array([]), np.zeros((0, 6)))

    def test_bound_is_safe_in_binary64(self) -> None:
        diag, off = assemble_dirichlet_hex_operator(triangle_hex_lattice(27), 3.3)
        dt = positivity_max_dt(diag, off)
        self.assertGreaterEqual(1.0 + dt * float(np.min(diag)), 0.0)

    def test_positive_data_stays_non_negative_at_the_bound(self) -> None:
        lat = triangle_hex_lattice(16)
        rng = np.random.default_rng(0)
        u0 = rng.random(lat.num_nodes)
        u0[rng.random(lat.num_nodes) < 0.5] = 0.0  # rough, non-smooth data
        dt = positivity_max_dt(*assemble_dirichlet_hex_operator(lat, 2.0))
        u = explicit_euler_heat(u0, lat, kappa=2.0, dt=dt, steps=200)
        self.assertGreaterEqual(float(u.min()), 0.0)
        self.assertLess(float(u.max()), float(u0.max()))

    def test_step_above_the_bound_is_refused(self) -> None:
        lat = triangle_hex_lattice(16)
        u0 = np.ones(lat.num_nodes)
        bound = positivity_max_dt(*assemble_dirichlet_hex_operator(lat))
        with self.assertRaisesRegex(ValueError, "positivity bound"):
            explicit_euler_heat(u0, lat, kappa=1.0, dt=bound * (1 + 1e-9), steps=1)
        # Without the positivity requirement the engine's sharp L2 bound applies.
        u = explicit_euler_heat(u0, lat, kappa=1.0, dt=bound * 1.2, steps=3, require_positivity=False)
        self.assertTrue(np.all(np.isfinite(u)))
        l2 = rc._engine().hex_cfl_max_dt(lat.h, 1.0, safety=1.0)
        with self.assertRaisesRegex(ValueError, "L2 stability bound"):
            explicit_euler_heat(u0, lat, kappa=1.0, dt=l2 * (1 + 1e-9), steps=1, require_positivity=False)

    def test_single_positive_node_shows_the_bound_is_sharp(self) -> None:
        lat = triangle_hex_lattice(16)
        diag, off = assemble_dirichlet_hex_operator(lat)
        bound = positivity_max_dt(diag, off)
        u0 = np.zeros(lat.num_nodes)
        u0[lat.num_nodes // 2] = 1.0
        nb = np.clip(lat.neighbors, 0, None)
        above = u0 + 1.01 * bound * (diag * u0 + (off * u0[nb]).sum(1))
        self.assertLess(float(above.min()), 0.0)  # the bound is necessary, not only sufficient

    def test_euler_converges_first_order_in_time(self) -> None:
        lat = triangle_hex_lattice(16)
        mode = LameMode(1, 1)
        phi = mode.evaluate(lat.coords)
        lam_h = mode.discrete_eigenvalue(lat.h)
        t_final = 0.02
        dt0 = rc._engine().hex_positivity_max_dt(lat.h, 1.0)
        errors = []
        for k in range(3):
            steps = math.ceil(t_final / dt0) * 2**k
            u = explicit_euler_heat(phi, lat, kappa=1.0, dt=t_final / steps, steps=steps)
            errors.append(np.linalg.norm(u - math.exp(-lam_h * t_final) * phi))
        for coarse, fine in zip(errors, errors[1:]):
            self.assertAlmostEqual(observed_order(coarse, fine), 1.0, delta=0.05)

    def test_euler_argument_contract(self) -> None:
        lat = triangle_hex_lattice(5)
        u0 = np.zeros(lat.num_nodes)
        for kwargs, exc in (({"kappa": 0.0, "dt": 1e-3, "steps": 1}, ValueError),
                            ({"kappa": 1.0, "dt": -1e-3, "steps": 1}, ValueError),
                            ({"kappa": 1.0, "dt": 1e-3, "steps": -1}, ValueError),
                            ({"kappa": 1.0, "dt": 1e-3, "steps": 1.5}, TypeError),
                            ({"kappa": 1.0, "dt": 1e-3, "steps": 1, "require_positivity": 1}, TypeError)):
            with self.subTest(kwargs=kwargs), self.assertRaises(exc):
                explicit_euler_heat(u0, lat, **kwargs)


class TestHexHeatTriStudyGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.study = run_hex_heat_tri_study()

    def test_default_study_passes_every_criterion(self) -> None:
        self.assertEqual(len(self.study["criteria"]), 9)
        self.assertEqual([k for k, ok in self.study["criteria"].items() if not ok], [])
        self.assertTrue(self.study["code_verification_pass"])
        self.assertTrue(self.study["solution_verification_pass"])
        self.assertTrue(self.study["passed"])

    def test_reproduces_the_numbers_decided_in_issue_60(self) -> None:
        # #60 (2026-10-06): lambda_h = 51.9649 / 52.4690 / 52.5956 / 52.6273,
        # observed order 1.994 / 1.999 / 2.000, three-grid Richardson 1.993 / 1.998.
        lams = [round(level["lambda_rq"], 4) for level in self.study["levels"]]
        self.assertEqual(lams, [51.9649, 52.469, 52.5956, 52.6273])
        exact_orders = [observed_order(a["lambda_exact"] - a["lambda_rq"], b["lambda_exact"] - b["lambda_rq"])
                        for a, b in zip(self.study["levels"], self.study["levels"][1:])]
        self.assertEqual([round(p, 3) for p in exact_orders], [1.994, 1.999, 2.0])
        self.assertEqual(round(self.study["celik_coarse"]["observed_order"], 3), 1.993)
        self.assertEqual(round(self.study["celik_fine"]["observed_order"], 3), 1.998)

    def test_gci_is_conservative_against_the_exact_error(self) -> None:
        gci = self.study["celik_fine"]["gci_fine"]
        self.assertGreaterEqual(gci, self.study["qoi_exact_rel_error_fine"])
        self.assertLess(gci, 2 * self.study["qoi_exact_rel_error_fine"])  # informative, not vacuous

    def test_evidence_is_strict_json(self) -> None:
        payload = json.dumps(self.study, allow_nan=False, sort_keys=True)
        self.assertEqual(json.loads(payload)["reference_id"], "hex-heat-tri/v1")

    def test_reference_spec_derives_the_verified_ceiling(self) -> None:
        spec = hex_heat_tri_reference_spec(self.study, evidence_ref="results/hex-heat-tri.json")
        self.assertEqual(spec.claim_ceiling(), ClaimCeiling.VERIFIED_NUMERICAL_REFERENCE)
        self.assertEqual(spec.spatial_study.level_count, 4)
        self.assertEqual([lv.level_id for lv in spec.spatial_study.levels], ["N=128", "N=64", "N=32", "N=16"])
        self.assertTrue(spec.spatial_study.supports_verified_reference)
        self.assertIs(spec.temporal_verification.status, EvidenceStatus.NOT_APPLICABLE)
        self.assertIn("N = 16, 32, 64, 128", spec.context_of_use)
        self.assertEqual(spec.to_dict()["claim_ceiling"], "verified_numerical_reference")

    def test_reference_spec_rejects_foreign_input(self) -> None:
        with self.assertRaises(ValueError):
            hex_heat_tri_reference_spec({"schema": "other"}, evidence_ref="x")


class TestGateDiscrimination(unittest.TestCase):
    """A wrong operator must fail the gate and cap the claim at comparison_only."""

    def _assert_capped(self, study, failing: str) -> None:
        self.assertFalse(study["passed"])
        self.assertFalse(study["criteria"][failing])
        spec = hex_heat_tri_reference_spec(study, evidence_ref="results/x.json")
        self.assertEqual(spec.claim_ceiling(), ClaimCeiling.COMPARISON_ONLY)

    def test_wrong_stencil_weight_fails(self) -> None:
        engine = rc._engine()
        true_weight = engine.hex_laplacian_stencil_weight
        with mock.patch.object(engine, "hex_laplacian_stencil_weight", side_effect=lambda h: 1.01 * true_weight(h)):
            study = run_hex_heat_tri_study()
        self._assert_capped(study, "C1_eigen_residual")
        self.assertFalse(study["code_verification_pass"])

    def test_first_order_weight_error_fails_the_order_gate(self) -> None:
        # w * (1 + h): consistent, but only first-order accurate.
        engine = rc._engine()
        true_weight = engine.hex_laplacian_stencil_weight
        with mock.patch.object(engine, "hex_laplacian_stencil_weight",
                               side_effect=lambda h: (1 + h) * true_weight(h)):
            study = run_hex_heat_tri_study()
        self._assert_capped(study, "C5_observed_order")
        self.assertFalse(study["criteria"]["C4_heat_error_order"])

    def test_neumann_like_boundary_handling_fails(self) -> None:
        # Dropping Dirichlet slots (the benchmark H1 convention) is a different operator.
        def dropped(u, lattice):
            arr = np.asarray(u, dtype=np.float64)
            w = 2.0 / (3.0 * lattice.h**2)
            valid = lattice.neighbors >= 0
            gathered = np.where(valid, arr[np.clip(lattice.neighbors, 0, None)], 0.0)
            return w * (gathered.sum(1) - valid.sum(1) * arr)

        with mock.patch.object(rc, "apply_dirichlet_hex_laplacian", side_effect=dropped):
            study = run_hex_heat_tri_study()
        self._assert_capped(study, "C1_eigen_residual")

    def test_gci_that_does_not_cover_the_exact_error_fails(self) -> None:
        with mock.patch.object(rc, "_GCI_SAFETY_FACTOR", 0.5):
            study = run_hex_heat_tri_study()
        self._assert_capped(study, "C9_gci_covers_exact_error")
        self.assertTrue(study["code_verification_pass"])

    def test_each_failed_criterion_caps_the_ceiling(self) -> None:
        passing = run_hex_heat_tri_study()
        code_keys = {"C1_eigen_residual", "C2_rayleigh_matches_closed_form", "C3_isotropic_coefficient",
                     "C4_heat_error_order"}
        for key in passing["criteria"]:
            with self.subTest(criterion=key):
                study = json.loads(json.dumps(passing))
                study["criteria"][key] = False
                spec = hex_heat_tri_reference_spec(study, evidence_ref="results/x.json")
                self.assertEqual(spec.claim_ceiling(), ClaimCeiling.COMPARISON_ONLY)
                check = spec.code_verification if key in code_keys else spec.spatial_verification
                self.assertIs(check.status, EvidenceStatus.FAIL)
                self.assertIn(key, check.rationale)


class TestConfigContract(unittest.TestCase):
    def test_invalid_configs(self) -> None:
        for kwargs in ({"levels": (16, 32, 64)}, {"levels": (16, 32, 64, 100)}, {"levels": [16, 32, 64, 128]},
                       {"levels": (2, 4, 8, 16)}, {"kappa": 0.0}, {"t_final": float("inf")}, {"modes": ()},
                       {"modes": ((LameMode(1, 1), 0.0),)}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                HexHeatTriConfig(**kwargs)
        with self.assertRaises(TypeError):
            HexHeatTriConfig(modes=((LameMode(1, 1), True),))
        with self.assertRaises(TypeError):
            run_hex_heat_tri_study(config={"levels": (16, 32, 64, 128)})

    def test_other_valid_levels_and_data_still_pass(self) -> None:
        modes = ((LameMode(1, 1), 2.0), (LameMode(2, 1, "cos"), -1.0))
        study = run_hex_heat_tri_study(HexHeatTriConfig(levels=(12, 24, 48, 96), kappa=0.5, t_final=0.05, modes=modes))
        self.assertTrue(study["passed"], study["criteria"])
        self.assertEqual(study["config"]["levels"], [12, 24, 48, 96])


if __name__ == "__main__":
    unittest.main()
