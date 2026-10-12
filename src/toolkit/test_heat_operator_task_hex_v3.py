"""
test_heat_operator_task_hex_v3.py
=================================
Contract tests for the Phase 1 instrument T1 ``hex-heat-operator``
(spec/HEAT_OPERATOR_TASK_v1.md): basis, prior, exact truth, manifest and
digests, split compatibility, the three metrics and the mechanistic baseline.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

import heat_operator_task_hex_v3 as task
from dataset_split_hex_v3 import split_dataset_by_trajectory, summarize_group_split
from hex_fv_solver_hex_v3 import HexLattice, apply_hex_laplacian, torus_hex_lattice, triangle_hex_lattice
from heat_operator_task_hex_v3 import (
    HeatOperatorDataset,
    HeatOperatorTaskConfig,
    dataset_manifest,
    evaluate_predictions,
    evaluate_sample,
    generate_dataset,
    mechanistic_baseline,
    mode_basis,
    relative_h1_seminorm,
    relative_l2,
    relative_max,
    sample_coefficients,
    verify_dataset,
)
from reference_case_hex_v3 import LameMode, TorusMode

TRI = HeatOperatorTaskConfig()
TORUS = HeatOperatorTaskConfig(domain="torus_periodic")


class TestBasis(unittest.TestCase):
    def test_triangle_basis_is_complete_sorted_and_without_duplicates(self) -> None:
        basis = mode_basis(TRI)
        expected = []
        for m in range(1, 12):
            for n in range(m, 12):
                for parity in (("sin",) if m == n else ("sin", "cos")):
                    mode = LameMode(m, n, parity)
                    if mode.eigenvalue <= TRI.lambda_cutoff:
                        expected.append(mode.key)
        self.assertEqual(sorted(m.key for m in basis), sorted(expected))
        self.assertEqual(len(basis), 15)
        eigenvalues = [m.eigenvalue for m in basis]
        self.assertEqual(eigenvalues, sorted(eigenvalues))
        lat = triangle_hex_lattice(48)
        gram = np.array([[a.evaluate(lat.coords) @ b.evaluate(lat.coords) for b in basis] for a in basis])
        d = np.sqrt(np.diag(gram))
        np.testing.assert_allclose(gram / np.outer(d, d), np.eye(len(basis)), atol=1e-9)

    def test_torus_basis_uses_the_half_plane_and_excludes_the_constant(self) -> None:
        basis = mode_basis(TORUS)
        self.assertEqual(len(basis), 40)
        keys = {(m.p, m.q) for m in basis}
        self.assertNotIn((0, 0), keys)
        for p, q in keys:
            self.assertTrue(p > 0 or (p == 0 and q > 0))
            self.assertNotIn((-p, -q), keys)
        self.assertTrue(all(isinstance(m, TorusMode) and m.eigenvalue <= TORUS.lambda_cutoff for m in basis))

    def test_config_contract(self) -> None:
        for kwargs, exc in (({"domain": "square"}, ValueError), ({"kappa": 0.0}, ValueError),
                            ({"t_final": float("nan")}, ValueError), ({"lambda_cutoff": 10.0}, ValueError),
                            ({"spectral_decay": "1"}, TypeError), ({"kappa": True}, TypeError)):
            with self.subTest(kwargs=kwargs), self.assertRaises(exc):
                HeatOperatorTaskConfig(**kwargs)
        self.assertEqual(len(TRI.digest()), 64)
        self.assertNotEqual(TRI.digest(), TORUS.digest())


class TestPriorAndTruth(unittest.TestCase):
    def test_coefficients_depend_only_on_seed_and_sample_id(self) -> None:
        a = sample_coefficients(TRI, 7, 3)
        np.testing.assert_array_equal(a, sample_coefficients(TRI, 7, 3))
        self.assertFalse(np.array_equal(a, sample_coefficients(TRI, 7, 4)))
        self.assertFalse(np.array_equal(a, sample_coefficients(TRI, 8, 3)))
        d1 = generate_dataset(TRI, n=16, sample_ids=[5, 3, 9], dataset_seed=7)
        d2 = generate_dataset(TRI, n=32, sample_ids=[3], dataset_seed=7)
        np.testing.assert_array_equal(d1["coefficients"][1], d2["coefficients"][0])

    def test_prior_scale_follows_the_spectral_decay(self) -> None:
        basis = mode_basis(TRI)
        lam = np.array([m.eigenvalue for m in basis])
        sigma = lam.min() / lam
        draws = np.stack([sample_coefficients(TRI, 1, s) for s in range(4000)])
        np.testing.assert_allclose(draws.std(axis=0) / sigma, 1.0, atol=0.06)
        for bad in (-1, 1.5, True):
            with self.assertRaises((TypeError, ValueError)):
                sample_coefficients(TRI, bad, 0)

    def test_truth_is_the_exact_semigroup_in_the_mode_basis(self) -> None:
        lat = triangle_hex_lattice(20)
        a = sample_coefficients(TRI, 2, 11)
        u0, u_t = evaluate_sample(TRI, a, lat)
        basis = mode_basis(TRI)
        phis = np.stack([m.evaluate(lat.coords) for m in basis])
        np.testing.assert_allclose(u0, a @ phis, atol=1e-12)
        decay = np.exp(-np.array([m.eigenvalue for m in basis]) * TRI.t_final)
        np.testing.assert_allclose(u_t, (a * decay) @ phis, atol=1e-12)
        with self.assertRaises(ValueError):
            evaluate_sample(TRI, a[:-1], lat)
        with self.assertRaises(ValueError):
            evaluate_sample(TRI, a, torus_hex_lattice(8))

    def test_resolution_axis_samples_the_same_function(self) -> None:
        coarse = generate_dataset(TRI, n=16, sample_ids=[0, 1], dataset_seed=3)
        fine = generate_dataset(TRI, n=32, sample_ids=[0, 1], dataset_seed=3)
        index = {tuple(np.round(c, 10)): k for k, c in enumerate(fine["lattice"].coords)}
        rows = [index[tuple(np.round(c, 10))] for c in coarse["lattice"].coords]
        np.testing.assert_allclose(coarse["u0"], fine["u0"][:, rows], atol=1e-12)
        np.testing.assert_allclose(coarse["uT"], fine["uT"][:, rows], atol=1e-12)


class TestManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = generate_dataset(TRI, n=24, sample_ids=range(12), dataset_seed=5)

    def test_manifest_fields_and_reproducibility(self) -> None:
        m = self.data["manifest"]
        self.assertEqual(m["task_id"], "hex-heat-operator/v1")
        self.assertEqual((m["n"], m["num_nodes"], m["dataset_seed"]), (24, 253, 5))
        self.assertEqual(len(m["sample_sha256"]), 12)
        self.assertEqual(m["config_sha256"], TRI.digest())
        self.assertEqual(len(m["generator_sha256"]), 64)
        report = verify_dataset(m, self.data)
        self.assertTrue(report["integrity"] and report["reproducible"])
        self.assertLessEqual(report["max_relative_deviation"], 1e-12)

    def test_any_changed_byte_breaks_integrity(self) -> None:
        tampered = dict(self.data, uT=self.data["uT"].copy())
        tampered["uT"][3, 7] = np.nextafter(tampered["uT"][3, 7], np.inf)
        report = verify_dataset(self.data["manifest"], tampered)
        self.assertFalse(report["integrity"])
        self.assertTrue(report["reproducible"])  # one ulp is within the reproducibility tolerance
        tampered["uT"][3, 7] += 1e-6
        self.assertFalse(verify_dataset(self.data["manifest"], tampered)["reproducible"])

    def test_lattice_and_coefficients_are_integrity_bound(self) -> None:
        tampered_coeff = dict(self.data, coefficients=self.data["coefficients"].copy())
        tampered_coeff["coefficients"][0, 0] = np.nextafter(
            tampered_coeff["coefficients"][0, 0], np.inf
        )
        report = verify_dataset(self.data["manifest"], tampered_coeff)
        self.assertFalse(report["integrity"])
        self.assertTrue(report["reproducible"])

        lattice = self.data["lattice"]
        coords = lattice.coords.copy()
        coords[0, 0] += 1e-6
        coords.flags.writeable = False
        tampered_lattice = HexLattice(
            n=lattice.n,
            h=lattice.h,
            domain=lattice.domain,
            coords=coords,
            neighbors=lattice.neighbors,
        )
        report = verify_dataset(
            self.data["manifest"],
            dict(self.data, lattice=tampered_lattice),
        )
        self.assertFalse(report["integrity"])
        self.assertFalse(report["reproducible"])

    def test_sample_ids_are_verified_exactly(self) -> None:
        # Codex P2 on #61 @266900e: int() turned a changed id 0.2 back into 0, and
        # integrity, digest and regeneration all agreed again.
        manifest = self.data["manifest"]
        self.assertTrue(verify_dataset(manifest, self.data)["integrity"])
        for label, ids in (
            ("non-integral id", np.concatenate([[0.2], np.arange(1, 12)])),
            ("integral float ids", np.arange(12, dtype=np.float64)),
            ("bool id", [True] + list(range(1, 12))),
        ):
            with self.subTest(data_ids=label):
                report = verify_dataset(manifest, dict(self.data, sample_ids=ids))
                self.assertFalse(report["integrity"], report)
                self.assertFalse(report["reproducible"], report)
        for field, value in (("sample_ids", [0.2] + list(range(1, 12))),
                             ("sample_ids", [0.0] + list(range(1, 12))),
                             ("dataset_seed", 5.0), ("dataset_seed", 5.2)):
            with self.subTest(manifest_field=field, value=value), self.assertRaises(TypeError):
                verify_dataset(dict(manifest, **{field: value}), self.data)
        with self.assertRaises(TypeError):
            dataset_manifest(TRI, dict(self.data, sample_ids=np.concatenate([[0.2], np.arange(1, 12)])),
                             dataset_seed=5)
        with self.assertRaises(TypeError):
            HeatOperatorDataset(dict(self.data, sample_ids=np.concatenate([[0.2], np.arange(1, 12)])))

    def test_manifest_for_another_task_is_rejected(self) -> None:
        # Codex P2 on #61 @266900e: task_id is in no digest, so a manifest relabelled
        # for another task verified with full integrity and reproducibility.
        for task_id in ("hex-heat-operator/v2", "other-task/v1", None):
            with self.subTest(task_id=task_id), self.assertRaisesRegex(ValueError, "task_id"):
                verify_dataset(dict(self.data["manifest"], task_id=task_id), self.data)
        missing = {k: v for k, v in self.data["manifest"].items() if k != "task_id"}
        with self.assertRaisesRegex(ValueError, "task_id"):
            verify_dataset(missing, self.data)

    def test_digests_are_order_and_resolution_sensitive(self) -> None:
        reordered = generate_dataset(TRI, n=24, sample_ids=list(range(12))[::-1], dataset_seed=5)
        self.assertNotEqual(reordered["manifest"]["dataset_sha256"], self.data["manifest"]["dataset_sha256"])
        self.assertEqual(reordered["manifest"]["sample_sha256"][::-1], self.data["manifest"]["sample_sha256"])
        other_n = generate_dataset(TRI, n=12, sample_ids=range(12), dataset_seed=5)
        self.assertNotEqual(other_n["manifest"]["dataset_sha256"], self.data["manifest"]["dataset_sha256"])

    def test_manifest_contract(self) -> None:
        with self.assertRaises(ValueError):
            verify_dataset({"schema": "other"}, self.data)
        bad = dict(self.data["manifest"], config_sha256="0" * 64)
        with self.assertRaises(ValueError):
            verify_dataset(bad, self.data)
        for ids in ([], [1, 1], [-1]):
            with self.assertRaises((ValueError, TypeError)):
                generate_dataset(TRI, n=8, sample_ids=ids, dataset_seed=0)
        with self.assertRaises(TypeError):
            generate_dataset({"domain": "triangle_dirichlet"}, n=8, sample_ids=[0], dataset_seed=0)
        self.assertEqual(dataset_manifest(TRI, self.data, dataset_seed=5), self.data["manifest"])


class TestSplitAndLoader(unittest.TestCase):
    def test_leakage_safe_split_and_dataloader(self) -> None:
        data = generate_dataset(TRI, n=12, sample_ids=range(20), dataset_seed=1)
        ds = HeatOperatorDataset(data)
        train, val = split_dataset_by_trajectory(ds, val_fraction=0.25, seed=0)
        self.assertEqual(len(train) + len(val), 20)
        self.assertEqual(summarize_group_split(train, val)["overlap_trajectory_ids"], [])
        batch = next(iter(DataLoader(train, batch_size=4)))
        self.assertEqual(tuple(batch["u0"].shape), (4, data["lattice"].num_nodes))
        self.assertEqual(batch["u0"].dtype, torch.float64)
        with self.assertRaises(ValueError):
            HeatOperatorDataset(dict(data, uT=data["uT"][:-1]))


class TestMetrics(unittest.TestCase):
    def test_simple_values(self) -> None:
        lat = triangle_hex_lattice(16)
        truth = np.stack([LameMode(1, 1).evaluate(lat.coords), LameMode(1, 2, "cos").evaluate(lat.coords)])
        for fn in (relative_l2, relative_h1_seminorm, relative_max):
            with self.subTest(metric=fn.__name__):
                np.testing.assert_allclose(fn(truth, truth, lat), 0.0, atol=1e-15)
                np.testing.assert_allclose(fn(1.1 * truth, truth, lat), 0.1, rtol=1e-12)
        report = evaluate_predictions(0.9 * truth, truth, lat)
        self.assertEqual(sorted(report), ["rel_h1", "rel_l2", "rel_max"])
        np.testing.assert_allclose(report["rel_l2"], [0.1, 0.1], rtol=1e-12)

    def test_h1_is_the_energy_form_with_dirichlet_edges(self) -> None:
        lat = triangle_hex_lattice(13)
        v = np.random.default_rng(3).standard_normal(lat.num_nodes)
        total = 0.0
        for row in range(lat.num_nodes):
            for slot in range(6):
                j = lat.neighbors[row, slot]
                if j < 0:
                    total += v[row] ** 2  # edge to a Dirichlet node (value 0), seen once
                elif j > row:
                    total += (v[row] - v[j]) ** 2
        energy = task._energy_seminorm(v[None], lat)[0] ** 2
        self.assertAlmostEqual(energy, total / math.sqrt(3.0), delta=1e-9 * energy)
        phi = LameMode(1, 1).evaluate(lat.coords)
        h1_sq = task._energy_seminorm(phi[None], lat)[0] ** 2
        self.assertAlmostEqual(h1_sq / (lat.cell_area * float(phi @ phi)), LameMode(1, 1).discrete_eigenvalue(lat.h),
                               delta=1e-9)
        torus = torus_hex_lattice(12)
        w = TorusMode(1, 1).evaluate(torus.coords)
        self.assertAlmostEqual(-torus.cell_area * float(w @ apply_hex_laplacian(w, torus)),
                               task._energy_seminorm(w[None], torus)[0] ** 2, places=9)

    def test_fail_closed(self) -> None:
        lat = triangle_hex_lattice(8)
        good = LameMode(1, 1).evaluate(lat.coords)
        for pred, truth in ((good[:-1], good), (np.full_like(good, np.nan), good), (good, np.zeros_like(good))):
            for fn in (relative_l2, relative_h1_seminorm, relative_max):
                with self.subTest(metric=fn.__name__), self.assertRaises(ValueError):
                    fn(pred, truth, lat)

    def test_inputs_with_extra_dimensions_are_rejected(self) -> None:
        # Codex P2 on #61 @266900e: [S, M, C] passed the shape[1] check and the
        # metrics returned C values per sample instead of one.
        lat = triangle_hex_lattice(8)
        good = np.stack([LameMode(1, 1).evaluate(lat.coords), LameMode(1, 2, "cos").evaluate(lat.coords)])
        extra = np.repeat(good[:, :, None], 3, axis=2)
        for fn in (relative_l2, relative_h1_seminorm, relative_max):
            with self.subTest(metric=fn.__name__):
                with self.assertRaisesRegex(ValueError, "same shape"):
                    fn(1.1 * extra, extra, lat)
                with self.assertRaisesRegex(ValueError, "same shape"):
                    fn(1.1 * good[None], good[None], lat)
                self.assertEqual(fn(1.1 * good[0], good[0], lat).shape, (1,))  # one 1-D field = one sample
        with self.assertRaisesRegex(ValueError, "same shape"):
            evaluate_predictions(1.1 * extra, extra, lat)

    def test_l2_and_max_are_exact_at_extreme_finite_scales(self) -> None:
        # Codex P2 on #61 @266900e: norm(axis=1) underflowed a nonzero truth to 0
        # (1e-300: "zero truth") or overflowed numerator and denominator to inf
        # (1e308: inf / inf = NaN); p - t itself overflows for p = -t = 1e308.
        lat = triangle_hex_lattice(8)
        truth = np.stack([LameMode(1, 1).evaluate(lat.coords), LameMode(1, 2, "cos").evaluate(lat.coords)])
        reference = {fn: fn(1.1 * truth, truth, lat) for fn in (relative_l2, relative_max)}
        for scale in (2.0**-990, 1e-300, 1e300, 2.0**1000, 1e308 / float(np.max(np.abs(truth)))):
            for fn, expected in reference.items():
                with self.subTest(scale=scale, metric=fn.__name__):
                    t = scale * truth
                    self.assertTrue(np.all(np.isfinite(t)) and np.all(np.isfinite(1.1 * t)))
                    got = fn(1.1 * t, t, lat)
                    self.assertTrue(np.all(np.isfinite(got)), got)
                    np.testing.assert_allclose(got, expected, rtol=1e-12)
        big = 1e308 * np.sign(truth + 0.5)
        for fn in (relative_l2, relative_max):
            with self.subTest(case="p = -t near the overflow threshold", metric=fn.__name__):
                got = fn(-big, big, lat)
                np.testing.assert_allclose(got, 2.0, rtol=1e-12)
        # A ratio that binary64 cannot hold fails closed instead of reporting inf.
        for fn in (relative_l2, relative_max):
            with self.subTest(case="unrepresentable ratio", metric=fn.__name__):
                with self.assertRaisesRegex(ValueError, "not representable"):
                    fn(1e300 * truth, 1e-300 * truth, lat)

    def test_h1_rejects_materially_negative_discrete_energy(self) -> None:
        lat = triangle_hex_lattice(8)
        v = LameMode(1, 1).evaluate(lat.coords)
        original = task.apply_hex_laplacian
        try:
            task.apply_hex_laplacian = lambda row, lattice: np.asarray(row)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                task._energy_seminorm(v[None], lat)
        finally:
            task.apply_hex_laplacian = original

    def test_negative_energy_band_is_per_sample(self) -> None:
        # Codex P2 on #61: the band was 64 eps * max over the WHOLE batch, so a
        # high-energy sample hid a materially negative energy of another.
        lat = triangle_hex_lattice(8)
        loud = np.full(lat.num_nodes, 1e6)  # energy ~ 2.8e11 -> batch band ~ 4e-3
        quiet = np.ones(lat.num_nodes)
        original = task.apply_hex_laplacian

        def fake_laplacian(row, lattice, factor):
            row = np.asarray(row)
            return -row if row[0] > 1.0 else factor * row

        try:
            task.apply_hex_laplacian = lambda row, lattice: fake_laplacian(row, lattice, 1e-3)
            self.assertLess(-1e-3 * lat.cell_area * float(quiet @ quiet), -1e-4)  # materially negative
            with self.assertRaisesRegex(ValueError, "non-negative"):
                task._energy_seminorm(np.stack([loud, quiet]), lat)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                task._energy_seminorm(np.stack([quiet, loud]), lat)
            # Within its OWN band (|energy| < 1 -> 64 eps) a negative energy is clamped to 0.
            task.apply_hex_laplacian = lambda row, lattice: fake_laplacian(row, lattice, 1e-14)
            energies = task._energy_seminorm(np.stack([loud, quiet]), lat)
            self.assertEqual(float(energies[1]), 0.0)
            self.assertGreater(float(energies[0]), 0.0)
        finally:
            task.apply_hex_laplacian = original


class TestMechanisticBaseline(unittest.TestCase):
    def test_second_order_in_space_and_time(self) -> None:
        errors = []
        for n, steps in ((16, 4), (32, 8), (64, 16)):
            data = generate_dataset(TRI, n=n, sample_ids=range(3), dataset_seed=9)
            preds, infos = zip(*(mechanistic_baseline(TRI, u0, data["lattice"], steps=steps) for u0 in data["u0"]))
            errors.append(float(np.mean(relative_l2(np.stack(preds), data["uT"], data["lattice"]))))
            self.assertEqual(infos[0]["n"], n)
            self.assertGreaterEqual(infos[0]["wall_time_s"], 0.0)
            self.assertGreater(infos[0]["cg_iterations"], 0)
        for coarse, fine in zip(errors, errors[1:]):
            self.assertAlmostEqual(math.log2(coarse / fine), 2.0, delta=0.1)

    def test_torus_baseline_and_contract(self) -> None:
        data = generate_dataset(TORUS, n=16, sample_ids=[0], dataset_seed=2)
        pred, _ = mechanistic_baseline(TORUS, data["u0"][0], data["lattice"], steps=8)
        self.assertLess(relative_l2(pred, data["uT"][0], data["lattice"])[0], 0.05)
        with self.assertRaises(ValueError):
            mechanistic_baseline(TRI, data["u0"][0], data["lattice"], steps=8)
        tri = generate_dataset(TRI, n=8, sample_ids=[0], dataset_seed=2)
        for steps, exc in ((0, ValueError), (-1, ValueError), (2.0, TypeError)):
            with self.subTest(steps=steps), self.assertRaises(exc):
                mechanistic_baseline(TRI, tri["u0"][0], tri["lattice"], steps=steps)


if __name__ == "__main__":
    unittest.main()
