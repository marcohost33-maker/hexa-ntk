"""
test_reference_case_hex_v3.py
=============================
Contract tests for the Reality Anchor Phase 0 reference cases:
``hex-heat-tri`` (spec/REFERENCE_CASE_HEX_HEAT_TRI_v1.md), ``hex-heat-torus``
and ``hex-mms-tri`` (spec/REFERENCE_CASES_HEX_PHASE0B_v1.md). The full default
studies run in a few seconds, so CI re-derives every gate on every run. The
lattice, operator, positivity gate and CG tests live in
test_hex_fv_solver_hex_v3.py.
"""

from __future__ import annotations

import json
import math
import unittest
from unittest import mock

import numpy as np

import hex_fv_solver_hex_v3 as fv
import reference_case_hex_v3 as rc
from hex_fv_solver_hex_v3 import apply_hex_laplacian, torus_hex_lattice, triangle_hex_lattice
from reference_case_hex_v3 import (
    HexHeatTorusConfig,
    HexHeatTriConfig,
    HexMmsTriConfig,
    LameMode,
    TorusMode,
    celik_three_grid,
    hex_heat_torus_reference_spec,
    hex_heat_tri_reference_spec,
    hex_mms_tri_reference_specs,
    observed_order,
    run_hex_heat_torus_study,
    run_hex_heat_tri_study,
    run_hex_mms_tri_study,
)
from reference_spec_hex_v3 import ClaimCeiling, EvidenceStatus, ReferenceKind

_SQRT3 = math.sqrt(3.0)


def _boundary_coords(n: int) -> np.ndarray:
    h = 1.0 / n
    pts = [(h * (i + j / 2), h * j * _SQRT3 / 2)
           for j in range(n + 1) for i in range(n + 1 - j) if i == 0 or j == 0 or i + j == n]
    return np.asarray(pts)


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
                    residual = apply_hex_laplacian(phi, lat) + lam_h * phi
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
        engine = fv._engine()
        true_weight = engine.hex_laplacian_stencil_weight
        with mock.patch.object(engine, "hex_laplacian_stencil_weight", side_effect=lambda h: 1.01 * true_weight(h)):
            study = run_hex_heat_tri_study()
        self._assert_capped(study, "C1_eigen_residual")
        self.assertFalse(study["code_verification_pass"])

    def test_first_order_weight_error_fails_the_order_gate(self) -> None:
        # w * (1 + h): consistent, but only first-order accurate.
        engine = fv._engine()
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

        with mock.patch.object(rc, "apply_hex_laplacian", side_effect=dropped):
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
                       {"levels": (16, 20, 25, 31.25)}, {"levels": (64, 80, 100, 125)},
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



def _fd_gradient(func, coords: np.ndarray, step: float = 1e-5) -> np.ndarray:
    out = np.zeros_like(coords)
    for axis in range(2):
        e = np.zeros(2)
        e[axis] = step
        out[:, axis] = (func(coords + e) - func(coords - e)) / (2 * step)
    return out


class TestModeGradients(unittest.TestCase):
    def test_lame_gradient_matches_central_differences(self) -> None:
        xy = np.random.default_rng(5).random((40, 2)) * [0.8, 0.6] + [0.1, 0.05]
        for mode in (LameMode(1, 1), LameMode(1, 2, "cos"), LameMode(2, 3, "sin")):
            with self.subTest(mode=mode):
                np.testing.assert_allclose(mode.gradient(xy), _fd_gradient(mode.evaluate, xy),
                                           atol=1e-6 * mode.eigenvalue)


class TestTorusModes(unittest.TestCase):
    MODES = (TorusMode(1, 0, "sin"), TorusMode(1, 1, "cos"), TorusMode(2, -1, "sin"), TorusMode(0, 2, "cos"),
             TorusMode(3, 2, "sin"))

    def test_plane_waves_are_exact_discrete_eigenvectors(self) -> None:
        for n in (6, 16, 30):
            lat = torus_hex_lattice(n)
            for mode in self.MODES:
                with self.subTest(n=n, mode=mode):
                    phi = mode.evaluate(lat.coords)
                    lam_h = mode.discrete_eigenvalue(lat.h)
                    residual = apply_hex_laplacian(phi, lat) + lam_h * phi
                    self.assertLess(np.max(np.abs(residual)) / (lam_h * np.max(np.abs(phi))), 1e-12)

    def test_constant_mode_is_in_the_kernel(self) -> None:
        lat = torus_hex_lattice(12)
        const = TorusMode(0, 0, "cos")
        self.assertEqual(const.eigenvalue, 0.0)
        self.assertEqual(const.discrete_eigenvalue(lat.h), 0.0)
        np.testing.assert_allclose(apply_hex_laplacian(const.evaluate(lat.coords), lat), 0.0, atol=1e-9)

    def test_eigenvalue_and_wavevector(self) -> None:
        for mode in self.MODES:
            with self.subTest(mode=mode):
                k = mode.wavevector
                self.assertAlmostEqual(float(k @ k), mode.eigenvalue, places=9)
                # periodic under both torus periods
                self.assertAlmostEqual(math.cos(k[0]), 1.0, places=12)
                self.assertAlmostEqual(math.cos(k[1] * _SQRT3 / 2), 1.0, places=12)

    def test_invalid_modes(self) -> None:
        for args, exc in (((0, 0, "sin"), ValueError), ((1, 1, "tan"), ValueError), ((1.0, 1), TypeError),
                          ((1, True), TypeError)):
            with self.subTest(args=args), self.assertRaises(exc):
                TorusMode(*args)


class TestHexHeatTorusStudyGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.study = run_hex_heat_torus_study()

    def test_default_study_passes_every_criterion(self) -> None:
        self.assertEqual(len(self.study["criteria"]), 10)
        self.assertEqual([k for k, ok in self.study["criteria"].items() if not ok], [])
        self.assertTrue(self.study["passed"])

    def test_isotropy_across_non_equivalent_directions(self) -> None:
        # Reduced into the fundamental sector [0, 30] degrees the default
        # modes point at 0, 10.9 and 30 degrees; each coefficient ratio -> 1.
        sectors = set()
        for mode, _ in HexHeatTorusConfig().modes:
            if mode.eigenvalue == 0.0:
                continue
            angle = math.degrees(math.atan2(mode.wavevector[1], mode.wavevector[0])) % 60.0
            sectors.add(round(min(angle, 60.0 - angle), 1))
        self.assertEqual(sectors, {0.0, 10.9, 30.0})
        ratios = [c for c in self.study["levels"][-1]["coefficient_ratio"].values() if c is not None]
        self.assertEqual(len(ratios), 3)
        for ratio in ratios:
            self.assertAlmostEqual(ratio, 1.0, delta=1e-3)

    def test_conservation_and_strict_json(self) -> None:
        for level in self.study["levels"]:
            self.assertLessEqual(level["conservation_residual"], 1e-12)
        self.assertEqual(json.loads(json.dumps(self.study, allow_nan=False))["reference_id"], "hex-heat-torus/v1")

    def test_reference_spec_derives_the_verified_ceiling(self) -> None:
        spec = hex_heat_torus_reference_spec(self.study, evidence_ref="results/torus.json")
        self.assertEqual(spec.claim_ceiling(), ClaimCeiling.VERIFIED_NUMERICAL_REFERENCE)
        self.assertIn("periodic hex torus", spec.context_of_use)
        with self.assertRaises(ValueError):
            hex_heat_torus_reference_spec(run_hex_heat_tri_study(), evidence_ref="x")

    def test_non_conservative_operator_fails_c10(self) -> None:
        real = rc.apply_hex_laplacian

        def leaky(u, lattice):  # adds a uniform sink: the column sums no longer vanish
            return real(u, lattice) - 1e-9 * np.asarray(u) * 2.0 / (3.0 * lattice.h**2)

        with mock.patch.object(rc, "apply_hex_laplacian", side_effect=leaky):
            study = run_hex_heat_torus_study()
        self.assertFalse(study["criteria"]["C10_conservation"])
        spec = hex_heat_torus_reference_spec(study, evidence_ref="x")
        self.assertEqual(spec.claim_ceiling(), ClaimCeiling.COMPARISON_ONLY)
        self.assertIn("C10_conservation", spec.code_verification.rationale)

    def test_config_contract(self) -> None:
        for kwargs in ({"levels": (15, 30, 60, 120)}, {"levels": (16, 32, 64)}, {"modes": ((LameMode(1, 1), 1.0),)},
                       {"modes": ((TorusMode(0, 0), 1.0),)}, {"kappa": -1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                HexHeatTorusConfig(**kwargs)
        with self.assertRaises(TypeError):
            run_hex_heat_torus_study(HexHeatTriConfig())


class TestHexMmsTriStudyGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.study = run_hex_mms_tri_study()

    def test_default_study_passes_every_criterion(self) -> None:
        self.assertEqual(len(self.study["criteria"]), 12)
        self.assertEqual([k for k, ok in self.study["criteria"].items() if not ok], [])
        for key in ("oracle_verification_pass", "code_verification_pass", "solution_verification_pass",
                    "iterative_verification_pass", "passed"):
            self.assertTrue(self.study[key], key)

    def test_manufactured_solution_and_source(self) -> None:
        cfg = HexMmsTriConfig()
        boundary = _boundary_coords(48)
        self.assertLess(np.max(np.abs(cfg.solution(boundary))), 1e-12)
        self.assertLess(self.study["oracle"]["source_rel_error_vs_fd"], 1e-10)
        xy = np.random.default_rng(6).random((30, 2)) * [0.6, 0.4] + [0.2, 0.1]
        np.testing.assert_allclose(cfg.kappa_gradient(xy), _fd_gradient(cfg.kappa, xy), atol=1e-7)
        np.testing.assert_allclose(cfg.solution_gradient(xy), _fd_gradient(cfg.solution, xy), atol=1e-6)

    def test_second_order_error_and_negligible_iterative_error(self) -> None:
        for p in self.study["rel_l2_error_orders"][-2:] + self.study["max_error_orders"][-2:]:
            self.assertAlmostEqual(p, 2.0, delta=0.05)
        for level in self.study["levels"]:
            self.assertLess(level["iterative_to_discretisation_ratio"], 1e-3)
            self.assertTrue(level["conductance_symmetric"])
            self.assertAlmostEqual(level["kappa_min"], 0.6, delta=0.01)
        self.assertGreaterEqual(self.study["celik_fine"]["gci_fine"], self.study["qoi_exact_rel_error_fine"])
        self.assertAlmostEqual(self.study["qoi_exact"], 1.5 * _SQRT3, places=12)  # -1/2 * phi_11(centroid)

    def test_reference_specs_derive_oracle_and_verified_ceilings(self) -> None:
        oracle, solver = hex_mms_tri_reference_specs(self.study, evidence_ref="results/mms.json")
        self.assertIs(oracle.kind, ReferenceKind.MANUFACTURED)
        self.assertEqual(oracle.claim_ceiling(), ClaimCeiling.MATHEMATICAL_ORACLE)
        self.assertEqual(solver.claim_ceiling(), ClaimCeiling.VERIFIED_NUMERICAL_REFERENCE)
        self.assertIs(solver.iterative_verification.status, EvidenceStatus.PASS)
        self.assertIn("kappa in [0.6, 1.4]", solver.context_of_use)
        self.assertEqual(json.loads(json.dumps(self.study, allow_nan=False))["oracle_id"], "hex-mms-tri/oracle")
        with self.assertRaises(ValueError):
            hex_mms_tri_reference_specs(run_hex_heat_tri_study(), evidence_ref="x")

    def test_each_failed_criterion_caps_the_solver_ceiling(self) -> None:
        oracle_keys = {"O1_boundary_zero", "O2_source_matches_flux_divergence", "O3_kappa_positive"}
        for key in self.study["criteria"]:
            with self.subTest(criterion=key):
                study = json.loads(json.dumps(self.study))
                study["criteria"][key] = False
                oracle, solver = hex_mms_tri_reference_specs(study, evidence_ref="x")
                self.assertEqual(solver.claim_ceiling(), ClaimCeiling.COMPARISON_ONLY)
                expected = ClaimCeiling.COMPARISON_ONLY if key in oracle_keys else ClaimCeiling.MATHEMATICAL_ORACLE
                self.assertEqual(oracle.claim_ceiling(), expected)

    def test_config_contract(self) -> None:
        for kwargs in ({"levels": (16, 32, 64, 128)}, {"levels": (24, 48, 96)}, {"kappa_amplitude": 1.0},
                       {"kappa_amplitude": float("nan")}, {"kappa_amplitude": True}, {"modes": ()},
                       {"modes": ((TorusMode(1, 1), 1.0),)}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                HexMmsTriConfig(**kwargs)
        with self.assertRaises(TypeError):
            run_hex_mms_tri_study(HexHeatTriConfig())


class TestMmsGateDiscrimination(unittest.TestCase):
    """Realistic defects must fail the MMS gate; a valid alternative must not."""

    LEVELS = (12, 24, 48, 96)

    def _study(self):
        return run_hex_mms_tri_study(HexMmsTriConfig(levels=self.LEVELS))

    def test_kappa_at_the_node_instead_of_the_midpoint_fails(self) -> None:
        def node_kappa(lattice, kappa):
            c = np.empty(lattice.neighbors.shape)
            for s in range(6):
                c[:, s] = kappa(lattice.coords)
            for s in range(3):
                rows = np.nonzero(lattice.neighbors[:, s] >= 0)[0]
                c[lattice.neighbors[rows, s], s + 3] = c[rows, s]
            return c

        with mock.patch.object(rc, "edge_conductances", side_effect=node_kappa):
            study = self._study()
        self.assertFalse(study["criteria"]["M1_error_order"])  # first order only
        self.assertTrue(study["oracle_verification_pass"])
        oracle, solver = hex_mms_tri_reference_specs(study, evidence_ref="x")
        self.assertEqual(solver.claim_ceiling(), ClaimCeiling.COMPARISON_ONLY)
        self.assertEqual(oracle.claim_ceiling(), ClaimCeiling.MATHEMATICAL_ORACLE)

    def test_source_without_the_kappa_gradient_term_fails_the_oracle(self) -> None:
        def incomplete(self, coords):
            xy = np.asarray(coords, dtype=float)
            return self.kappa(xy) * sum(float(a) * m.eigenvalue * m.evaluate(xy) for m, a in self.modes)

        with mock.patch.object(HexMmsTriConfig, "source", incomplete):
            study = self._study()
        self.assertFalse(study["criteria"]["O2_source_matches_flux_divergence"])
        oracle, solver = hex_mms_tri_reference_specs(study, evidence_ref="x")
        self.assertEqual(oracle.claim_ceiling(), ClaimCeiling.COMPARISON_ONLY)
        self.assertEqual(solver.claim_ceiling(), ClaimCeiling.COMPARISON_ONLY)

    def test_wrong_stencil_weight_fails(self) -> None:
        engine = fv._engine()
        true_weight = engine.hex_laplacian_stencil_weight
        with mock.patch.object(engine, "hex_laplacian_stencil_weight", side_effect=lambda h: 1.01 * true_weight(h)):
            study = self._study()
        self.assertFalse(study["criteria"]["M1_error_order"])
        self.assertFalse(study["passed"])

    def test_non_converged_cg_fails_i1_instead_of_crashing(self) -> None:
        # spec Phase 0b I1: "On every level CG converges at rtol = 1e-13". A
        # ConvergenceError is evidence (I1 false), not a passed level.
        real = rc.solve_steady_diffusion

        def stalls_on_the_finest(lattice, kappa, source, *, rtol):
            u, info = real(lattice, kappa, source, rtol=rtol)
            if lattice.n == self.LEVELS[-1] and rtol == rc._MMS_CG_RTOL:
                raise fv.ConvergenceError("stalled", best=u, best_relative_residual=info["relative_residual"],
                                          iterations=info["iterations"])
            return u, info

        with mock.patch.object(rc, "solve_steady_diffusion", side_effect=stalls_on_the_finest):
            study = self._study()
        self.assertEqual([level["cg_converged"] for level in study["levels"]], [True, True, True, False])
        self.assertEqual([k for k, ok in study["criteria"].items() if not ok], ["I1_iterative_error_negligible"])
        _, solver = hex_mms_tri_reference_specs(study, evidence_ref="x")
        self.assertIs(solver.iterative_verification.status, EvidenceStatus.FAIL)
        with mock.patch.object(rc, "solve_steady_diffusion", side_effect=fv.ConvergenceError("no iterate")):
            with self.assertRaisesRegex(fv.ConvergenceError, "no iterate"):
                self._study()

    def test_early_cg_stop_fails_the_iterative_criterion(self) -> None:
        with mock.patch.object(rc, "_MMS_CG_RTOL", 1e-3):
            study = self._study()
        self.assertFalse(study["criteria"]["I1_iterative_error_negligible"])
        _, solver = hex_mms_tri_reference_specs(study, evidence_ref="x")
        self.assertIs(solver.iterative_verification.status, EvidenceStatus.FAIL)
        self.assertEqual(solver.claim_ceiling(), ClaimCeiling.COMPARISON_ONLY)

    def test_arithmetic_mean_conductance_is_a_valid_second_order_alternative(self) -> None:
        def mean_kappa(lattice, kappa):
            c = np.empty(lattice.neighbors.shape)
            for s in range(6):
                c[:, s] = 0.5 * (kappa(lattice.coords) + kappa(lattice.coords + lattice.h * fv.SLOT_DIRECTIONS[s]))
            for s in range(3):
                rows = np.nonzero(lattice.neighbors[:, s] >= 0)[0]
                c[lattice.neighbors[rows, s], s + 3] = c[rows, s]
            return c

        with mock.patch.object(rc, "edge_conductances", side_effect=mean_kappa):
            study = run_hex_mms_tri_study()
        self.assertTrue(study["criteria"]["M1_error_order"])  # the gate is not over-fitted to one scheme
        self.assertTrue(study["passed"], [k for k, ok in study["criteria"].items() if not ok])


if __name__ == "__main__":
    unittest.main()
