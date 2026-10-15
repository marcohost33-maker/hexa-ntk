"""
test_hex_fv_solver_hex_v3.py
============================
Contract tests for the mechanistic hex FV layer (spec/HEX_FV_SOLVER_v1.md):
lattices, operator, conductances, positivity gate (issue #60 item 1), explicit
Euler and conjugate gradients.
"""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

import hex_fv_solver_hex_v3 as fv
from hex_fv_solver_hex_v3 import (
    ConvergenceError,
    HexLattice,
    apply_hex_laplacian,
    assemble_hex_operator,
    conjugate_gradient,
    crank_nicolson_heat,
    edge_conductances,
    explicit_euler_heat,
    positivity_max_dt,
    solve_steady_diffusion,
    torus_hex_lattice,
    triangle_hex_lattice,
)

_SQRT3 = math.sqrt(3.0)


def _boundary_coords(n: int) -> np.ndarray:
    h = 1.0 / n
    pts = [(h * (i + j / 2), h * j * _SQRT3 / 2)
           for j in range(n + 1) for i in range(n + 1 - j) if i == 0 or j == 0 or i + j == n]
    return np.asarray(pts)


def _smooth_kappa(xy: np.ndarray) -> np.ndarray:
    return 1.0 + 0.4 * np.sin(2 * np.pi * xy[:, 0] + 0.3) * np.cos(3 * np.pi * xy[:, 1])


class TestTriangleLattice(unittest.TestCase):
    def test_node_count_spacing_and_domain(self) -> None:
        for n in (3, 4, 7, 16, 33):
            with self.subTest(n=n):
                lat = triangle_hex_lattice(n)
                self.assertEqual(lat.num_nodes, (n - 1) * (n - 2) // 2)
                self.assertEqual(lat.h, 1.0 / n)
                self.assertEqual(lat.domain, "triangle_dirichlet")
                self.assertEqual(lat.neighbors.shape, (lat.num_nodes, 6))
                self.assertAlmostEqual(lat.cell_area, _SQRT3 / 2 / n**2, places=15)

    def test_smallest_lattice_has_only_dirichlet_neighbours(self) -> None:
        self.assertEqual(triangle_hex_lattice(3).neighbors.tolist(), [[-1] * 6])

    def test_neighbour_geometry_and_symmetry(self) -> None:
        lat = triangle_hex_lattice(9)
        for s in range(6):
            step = lat.h * fv.SLOT_DIRECTIONS[s]
            rows = np.nonzero(lat.neighbors[:, s] >= 0)[0]
            nb = lat.neighbors[rows, s]
            np.testing.assert_allclose(lat.coords[nb] - lat.coords[rows], np.tile(step, (rows.size, 1)), atol=1e-14)
            np.testing.assert_array_equal(lat.neighbors[nb, (s + 3) % 6], rows)

    def test_dirichlet_slots_point_to_boundary_nodes(self) -> None:
        lat = triangle_hex_lattice(8)
        boundary = {tuple(np.round(p, 12)) for p in _boundary_coords(8)}
        for s in range(6):
            for row in np.nonzero(lat.neighbors[:, s] < 0)[0]:
                self.assertIn(tuple(np.round(lat.coords[row] + lat.h * fv.SLOT_DIRECTIONS[s], 12)), boundary)

    def test_slot_offsets_match_slot_directions(self) -> None:
        for (di, dj), direction in zip(fv.LATTICE_SLOTS, fv.SLOT_DIRECTIONS):
            np.testing.assert_allclose([di + dj / 2, dj * _SQRT3 / 2], direction, atol=1e-15)

    def test_arrays_are_read_only(self) -> None:
        lat = triangle_hex_lattice(5)
        for arr in (lat.neighbors, lat.coords, fv.SLOT_DIRECTIONS):
            with self.assertRaises(ValueError):
                arr[0, 0] = 0

    def test_invalid_n(self) -> None:
        for bad, exc in ((2, ValueError), (0, ValueError), (True, TypeError), (16.0, TypeError), ("16", TypeError)):
            with self.subTest(n=bad), self.assertRaises(exc):
                triangle_hex_lattice(bad)


class TestTorusLattice(unittest.TestCase):
    PERIODS = np.array([[1.0, 0.0], [0.0, _SQRT3 / 2]])

    def _reduced(self, d: np.ndarray) -> np.ndarray:
        """Reduce displacements modulo the torus periods into (-1/2, 1/2] x (-sqrt3/4, sqrt3/4]."""
        out = d.copy()
        out[:, 1] -= np.round(out[:, 1] / self.PERIODS[1, 1]) * self.PERIODS[1, 1]
        out[:, 0] -= np.round(out[:, 0])
        return out

    def test_counts_domain_and_coordinates(self) -> None:
        for n in (4, 6, 16):
            with self.subTest(n=n):
                lat = torus_hex_lattice(n)
                self.assertEqual(lat.num_nodes, n * n)
                self.assertEqual(lat.domain, "torus_periodic")
                self.assertTrue(np.all(lat.neighbors >= 0))
                self.assertTrue(np.all((lat.coords[:, 0] >= 0) & (lat.coords[:, 0] < 1)))
                self.assertTrue(np.all((lat.coords[:, 1] >= 0) & (lat.coords[:, 1] < _SQRT3 / 2)))
                self.assertAlmostEqual(lat.num_nodes * lat.cell_area, _SQRT3 / 2, places=12)

    def test_neighbours_are_the_six_lattice_steps_modulo_the_periods(self) -> None:
        for n in (4, 8, 12):
            lat = torus_hex_lattice(n)
            for s in range(6):
                with self.subTest(n=n, slot=s):
                    d = self._reduced(lat.coords[lat.neighbors[:, s]] - lat.coords)
                    np.testing.assert_allclose(d, np.tile(lat.h * fv.SLOT_DIRECTIONS[s], (lat.num_nodes, 1)),
                                               atol=1e-12)
                    np.testing.assert_array_equal(lat.neighbors[lat.neighbors[:, s], (s + 3) % 6],
                                                  np.arange(lat.num_nodes))

    def test_six_distinct_neighbours(self) -> None:
        lat = torus_hex_lattice(6)
        for row in range(lat.num_nodes):
            self.assertEqual(len(set(lat.neighbors[row].tolist()) | {row}), 7)

    def test_even_n_is_required(self) -> None:
        for bad, exc in ((5, ValueError), (2, ValueError), (7, ValueError), (8.0, TypeError)):
            with self.subTest(n=bad), self.assertRaises(exc):
                torus_hex_lattice(bad)


class TestOperator(unittest.TestCase):
    def test_weight_comes_from_the_engine(self) -> None:
        lat = triangle_hex_lattice(6)
        u = np.linspace(-1.0, 2.0, lat.num_nodes)
        with mock.patch.object(fv._engine(), "hex_laplacian_stencil_weight", return_value=1.0) as weight:
            unit = apply_hex_laplacian(u, lat)
        weight.assert_called_once_with(lat.h)
        expected = np.where(lat.neighbors >= 0, u[np.clip(lat.neighbors, 0, None)], 0.0).sum(1) - 6 * u
        np.testing.assert_allclose(unit, expected, rtol=0, atol=0)
        np.testing.assert_allclose(apply_hex_laplacian(u, lat), unit * 2 / (3 * lat.h**2), rtol=1e-14)

    def test_scalar_and_unit_conductance_paths_agree(self) -> None:
        for lat in (triangle_hex_lattice(11), torus_hex_lattice(10)):
            u = np.random.default_rng(1).standard_normal(lat.num_nodes)
            ones = np.ones(lat.neighbors.shape)
            with self.subTest(domain=lat.domain):
                np.testing.assert_allclose(apply_hex_laplacian(u, lat, ones), apply_hex_laplacian(u, lat), rtol=1e-12,
                                           atol=1e-9)
                np.testing.assert_allclose(apply_hex_laplacian(u, lat, 2.5), 2.5 * apply_hex_laplacian(u, lat),
                                           rtol=1e-14)
                diag, off = assemble_hex_operator(lat, ones)
                nb = np.clip(lat.neighbors, 0, None)
                np.testing.assert_allclose(diag * u + (off * u[nb]).sum(1), apply_hex_laplacian(u, lat), rtol=1e-12,
                                           atol=1e-9)

    def test_operator_is_symmetric_and_conservative_on_the_torus(self) -> None:
        lat = torus_hex_lattice(12)
        c = edge_conductances(lat, _smooth_kappa)
        rng = np.random.default_rng(2)
        u, v = rng.standard_normal((2, lat.num_nodes))
        for kappa in (1.0, c):
            lu, lv = apply_hex_laplacian(u, lat, kappa), apply_hex_laplacian(v, lat, kappa)
            self.assertAlmostEqual(float(v @ lu), float(u @ lv), delta=1e-9 * abs(float(v @ lu)))
            self.assertLess(abs(float(lu.sum())), 1e-10 * float(np.abs(lu).sum()))

    def test_field_and_lattice_contract(self) -> None:
        lat = triangle_hex_lattice(5)
        with self.assertRaises(ValueError):
            apply_hex_laplacian(np.zeros(lat.num_nodes + 1), lat)
        with self.assertRaises(ValueError):
            apply_hex_laplacian(np.full(lat.num_nodes, np.nan), lat)
        with self.assertRaises(TypeError):
            apply_hex_laplacian(np.zeros(lat.num_nodes), object())
        with self.assertRaises(ValueError):
            apply_hex_laplacian(np.zeros(lat.num_nodes), lat, 0.0)
        with self.assertRaises(TypeError):
            apply_hex_laplacian(np.zeros(lat.num_nodes), lat, "1")


class TestEdgeConductances(unittest.TestCase):
    def test_values_are_kappa_at_the_edge_midpoints(self) -> None:
        lat = triangle_hex_lattice(13)
        c = edge_conductances(lat, _smooth_kappa)
        for s in range(6):
            with self.subTest(slot=s):
                midpoints = lat.coords + 0.5 * lat.h * fv.SLOT_DIRECTIONS[s]
                np.testing.assert_allclose(c[:, s], _smooth_kappa(midpoints), rtol=1e-14)

    def test_exact_symmetry_and_read_only(self) -> None:
        for lat in (triangle_hex_lattice(17), torus_hex_lattice(14)):
            c = edge_conductances(lat, _smooth_kappa)
            for s in range(3):
                rows = np.nonzero(lat.neighbors[:, s] >= 0)[0]
                np.testing.assert_array_equal(c[rows, s], c[lat.neighbors[rows, s], s + 3])
            with self.assertRaises(ValueError):
                c[0, 0] = 1.0

    def test_invalid_kappa_is_rejected(self) -> None:
        lat = triangle_hex_lattice(7)
        for bad in (lambda xy: np.zeros(len(xy)), lambda xy: -np.ones(len(xy)),
                    lambda xy: np.full(len(xy), np.nan), lambda xy: np.ones(len(xy) + 1)):
            with self.assertRaises(ValueError):
                edge_conductances(lat, bad)
        with self.assertRaises(TypeError):
            edge_conductances(lat, 1.0)

    def test_asymmetric_or_malformed_arrays_are_rejected(self) -> None:
        lat = triangle_hex_lattice(7)
        c = np.array(edge_conductances(lat, _smooth_kappa))
        row = int(np.nonzero(lat.neighbors[:, 0] >= 0)[0][0])
        c[row, 0] *= 1.0 + 1e-15
        u = np.ones(lat.num_nodes)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            apply_hex_laplacian(u, lat, c)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            assemble_hex_operator(lat, c)
        with self.assertRaises(ValueError):
            apply_hex_laplacian(u, lat, np.ones((lat.num_nodes, 5)))
        with self.assertRaises(ValueError):
            apply_hex_laplacian(u, lat, np.full(lat.neighbors.shape, np.inf))

    def test_non_positive_conductance_arrays_are_rejected(self) -> None:
        # Codex P2 on #61: finite + symmetric is not enough; c <= 0 is
        # anti-diffusive and breaks the SPD claim of Crank-Nicolson.
        lat = triangle_hex_lattice(7)
        base = np.array(edge_conductances(lat, _smooth_kappa))
        row = int(np.nonzero(lat.neighbors[:, 0] >= 0)[0][0])
        partner = int(lat.neighbors[row, 0])
        one_zero = base.copy()
        one_zero[row, 0] = one_zero[partner, 3] = 0.0
        boundary_negative = base.copy()
        boundary_row = int(np.nonzero(lat.neighbors[:, 0] < 0)[0][0])
        boundary_negative[boundary_row, 0] = -1.0
        u = np.ones(lat.num_nodes)
        for name, c in (("all negative", -base), ("one symmetric zero edge", one_zero),
                        ("negative Dirichlet slot", boundary_negative)):
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, "> 0"):
                    apply_hex_laplacian(u, lat, c)
                with self.assertRaisesRegex(ValueError, "> 0"):
                    assemble_hex_operator(lat, c)
                with self.assertRaisesRegex(ValueError, "> 0"):
                    crank_nicolson_heat(u, lat, kappa=c, dt=1e-3, steps=1)


class TestPositivityGateIssue60(unittest.TestCase):
    def test_bound_matches_the_engine_positivity_contract(self) -> None:
        engine = fv._engine()
        for n in (8, 16, 33):
            for kappa in (1.0, 0.3, 7.5):
                with self.subTest(n=n, kappa=kappa):
                    lat = triangle_hex_lattice(n)
                    dt = positivity_max_dt(*assemble_hex_operator(lat, kappa))
                    ref = engine.hex_positivity_max_dt(lat.h, kappa)
                    self.assertLessEqual(abs(dt - ref), 4 * np.finfo(float).eps * ref)

    def test_variable_kappa_bound_is_row_dependent(self) -> None:
        lat = triangle_hex_lattice(20)
        c = edge_conductances(lat, _smooth_kappa)
        diag, off = assemble_hex_operator(lat, c)
        w = 2 / (3 * lat.h**2)
        expected = 1.0 / float(np.max(w * c.sum(1)))
        self.assertAlmostEqual(positivity_max_dt(diag, off), expected, delta=1e-14 * expected)

    def test_sign_structure_violations_raise(self) -> None:
        diag, off = assemble_hex_operator(triangle_hex_lattice(6))
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
        with self.assertRaises(ValueError):
            positivity_max_dt(diag, np.full(off.shape, np.nan))

    def test_bound_is_safe_in_binary64(self) -> None:
        diag, off = assemble_hex_operator(triangle_hex_lattice(27), 3.3)
        dt = positivity_max_dt(diag, off)
        self.assertGreaterEqual(1.0 + dt * float(np.min(diag)), 0.0)

    def test_positive_data_stays_non_negative_at_the_bound(self) -> None:
        for lat in (triangle_hex_lattice(16), torus_hex_lattice(16)):
            rng = np.random.default_rng(0)
            u0 = rng.random(lat.num_nodes)
            u0[rng.random(lat.num_nodes) < 0.5] = 0.0  # rough, non-smooth data
            for kappa in (2.0, edge_conductances(lat, _smooth_kappa)):
                with self.subTest(domain=lat.domain, scalar=not isinstance(kappa, np.ndarray)):
                    dt = positivity_max_dt(*assemble_hex_operator(lat, kappa))
                    u = explicit_euler_heat(u0, lat, kappa=kappa, dt=dt, steps=200)
                    self.assertGreaterEqual(float(u.min()), 0.0)
                    self.assertLessEqual(float(u.max()), float(u0.max()))

    def test_mass_is_conserved_on_the_torus(self) -> None:
        lat = torus_hex_lattice(16)
        c = edge_conductances(lat, _smooth_kappa)
        u0 = np.random.default_rng(3).random(lat.num_nodes)
        dt = positivity_max_dt(*assemble_hex_operator(lat, c))
        u = explicit_euler_heat(u0, lat, kappa=c, dt=dt, steps=100)
        self.assertAlmostEqual(float(u.sum()), float(u0.sum()), delta=1e-10 * float(u0.sum()))

    def test_step_above_the_bound_is_refused(self) -> None:
        lat = triangle_hex_lattice(16)
        u0 = np.ones(lat.num_nodes)
        bound = positivity_max_dt(*assemble_hex_operator(lat))
        with self.assertRaisesRegex(ValueError, "positivity bound"):
            explicit_euler_heat(u0, lat, kappa=1.0, dt=bound * (1 + 1e-9), steps=1)
        # Without the positivity requirement the engine's sharp L2 bound applies.
        u = explicit_euler_heat(u0, lat, kappa=1.0, dt=bound * 1.2, steps=3, require_positivity=False)
        self.assertTrue(np.all(np.isfinite(u)))
        l2 = fv._engine().hex_cfl_max_dt(lat.h, 1.0, safety=1.0)
        with self.assertRaisesRegex(ValueError, "L2 stability bound"):
            explicit_euler_heat(u0, lat, kappa=1.0, dt=l2 * (1 + 1e-9), steps=1, require_positivity=False)
        c = edge_conductances(lat, _smooth_kappa)
        with self.assertRaisesRegex(ValueError, "require_positivity=True"):
            explicit_euler_heat(u0, lat, kappa=c, dt=bound / 10, steps=1, require_positivity=False)

    def test_single_positive_node_shows_the_bound_is_sharp(self) -> None:
        lat = triangle_hex_lattice(16)
        diag, off = assemble_hex_operator(lat)
        bound = positivity_max_dt(diag, off)
        u0 = np.zeros(lat.num_nodes)
        u0[lat.num_nodes // 2] = 1.0
        nb = np.clip(lat.neighbors, 0, None)
        above = u0 + 1.01 * bound * (diag * u0 + (off * u0[nb]).sum(1))
        self.assertLess(float(above.min()), 0.0)  # the bound is necessary, not only sufficient

    def test_euler_converges_first_order_in_time(self) -> None:
        lat = triangle_hex_lattice(16)
        k = 2 * math.pi * np.array([1.0, 1 / _SQRT3])  # Lame fundamental, exact discrete eigenvector
        group = [np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]) for t in (0, 2 * math.pi / 3,
                                                                                            4 * math.pi / 3)]
        phi = sum(np.sin(lat.coords @ (g @ k)) for g in group)
        phi -= sum(np.sin(lat.coords @ (g @ np.array([k[0], -k[1]]))) for g in group)
        lam_h = -float(phi @ apply_hex_laplacian(phi, lat)) / float(phi @ phi)
        np.testing.assert_allclose(apply_hex_laplacian(phi, lat), -lam_h * phi, atol=1e-9 * lam_h)
        t_final = 0.02
        dt0 = fv._engine().hex_positivity_max_dt(lat.h, 1.0)
        errors = []
        for k_level in range(3):
            steps = math.ceil(t_final / dt0) * 2**k_level
            u = explicit_euler_heat(phi, lat, kappa=1.0, dt=t_final / steps, steps=steps)
            errors.append(np.linalg.norm(u - math.exp(-lam_h * t_final) * phi))
        for coarse, fine in zip(errors, errors[1:]):
            self.assertAlmostEqual(math.log2(coarse / fine), 1.0, delta=0.05)

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


def _lame_fundamental(lat):
    """Lame mode (1,1,sin) on the triangle lattice and its discrete eigenvalue (exact eigenvector)."""
    k = 2 * math.pi * np.array([1.0, 1 / _SQRT3])
    rotations = [np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
                 for t in (0, 2 * math.pi / 3, 4 * math.pi / 3)]
    phi = sum(np.sin(lat.coords @ (g @ k)) for g in rotations)
    phi -= sum(np.sin(lat.coords @ (g @ np.array([k[0], -k[1]]))) for g in rotations)
    lam_h = 16 / (3 * lat.h**2) * math.sin(math.pi * lat.h) ** 2
    return phi, lam_h


class TestCrankNicolson(unittest.TestCase):
    def test_one_step_factor_on_exact_eigenvectors(self) -> None:
        tri = triangle_hex_lattice(24)
        phi, lam_h = _lame_fundamental(tri)
        torus = torus_hex_lattice(24)
        k = 2 * math.pi * np.array([1.0, 2 / _SQRT3])
        wave = np.cos(torus.coords @ k)
        lam_t = 8 / (3 * torus.h**2) * sum(math.sin(torus.h * float(k @ fv.SLOT_DIRECTIONS[s]) / 2) ** 2
                                           for s in range(3))
        for lat, mode, lam in ((tri, phi, lam_h), (torus, wave, lam_t)):
            for dt in (1e-4, 1e-2):
                with self.subTest(domain=lat.domain, dt=dt):
                    u, info = crank_nicolson_heat(mode, lat, kappa=1.0, dt=dt, steps=5)
                    factor = (1 - dt * lam / 2) / (1 + dt * lam / 2)
                    np.testing.assert_allclose(u, factor**5 * mode, atol=1e-10 * np.max(np.abs(mode)))
                    self.assertEqual(info["steps"], 5)
                    self.assertLessEqual(info["max_relative_residual"], 1e-12)

    def test_second_order_in_time(self) -> None:
        lat = triangle_hex_lattice(16)
        phi, lam_h = _lame_fundamental(lat)
        t_final = 0.02
        errors = []
        for steps in (4, 8, 16, 32):
            u, _ = crank_nicolson_heat(phi, lat, kappa=1.0, dt=t_final / steps, steps=steps)
            errors.append(np.linalg.norm(u - math.exp(-lam_h * t_final) * phi))
        orders = [math.log2(a / b) for a, b in zip(errors, errors[1:])]
        for order in orders[-2:]:
            self.assertAlmostEqual(order, 2.0, delta=0.05)

    def test_unconditionally_l2_stable_and_conservative(self) -> None:
        lat = torus_hex_lattice(16)
        c = edge_conductances(lat, _smooth_kappa)
        u0 = np.random.default_rng(7).standard_normal(lat.num_nodes)
        dt = 100 * positivity_max_dt(*assemble_hex_operator(lat, c))  # far beyond any explicit bound
        u, info = crank_nicolson_heat(u0, lat, kappa=c, dt=dt, steps=20)
        self.assertLessEqual(np.linalg.norm(u), np.linalg.norm(u0))
        self.assertAlmostEqual(float(u.sum()), float(u0.sum()), delta=1e-9 * float(np.abs(u0).sum()))
        self.assertGreater(info["cg_iterations"], 0)

    def test_zero_steps_and_contract(self) -> None:
        lat = triangle_hex_lattice(6)
        u0 = np.arange(lat.num_nodes, dtype=float)
        u, info = crank_nicolson_heat(u0, lat, kappa=1.0, dt=1e-3, steps=0)
        np.testing.assert_array_equal(u, u0)
        self.assertEqual(info, {"steps": 0, "cg_iterations": 0, "max_relative_residual": 0.0})
        for kwargs, exc in (({"kappa": 1.0, "dt": 0.0, "steps": 1}, ValueError),
                            ({"kappa": 1.0, "dt": 1e-3, "steps": -1}, ValueError),
                            ({"kappa": -1.0, "dt": 1e-3, "steps": 1}, ValueError)):
            with self.subTest(kwargs=kwargs), self.assertRaises(exc):
                crank_nicolson_heat(u0, lat, **kwargs)


class TestConjugateGradient(unittest.TestCase):
    @staticmethod
    def _spd(n: int, seed: int) -> np.ndarray:
        a = np.random.default_rng(seed).standard_normal((n, n))
        return a @ a.T + n * np.eye(n)

    def test_solves_a_dense_spd_system(self) -> None:
        a = self._spd(40, 0)
        b = np.random.default_rng(1).standard_normal(40)
        x, info = conjugate_gradient(lambda v: a @ v, b, rtol=1e-13, maxiter=400)
        np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-10, atol=1e-12)
        self.assertLessEqual(info["relative_residual"], 1e-13)
        self.assertLessEqual(np.linalg.norm(b - a @ x), 2e-13 * np.linalg.norm(b))
        self.assertGreater(info["iterations"], 0)

    def test_warm_start_converges_with_fewer_iterations(self) -> None:
        a = self._spd(50, 3)
        b = np.random.default_rng(4).standard_normal(50)
        cold, info_cold = conjugate_gradient(lambda v: a @ v, b, rtol=1e-12, maxiter=500)
        near = cold + 1e-6 * np.random.default_rng(5).standard_normal(50)
        warm, info_warm = conjugate_gradient(lambda v: a @ v, b, rtol=1e-12, maxiter=500, x0=near)
        np.testing.assert_allclose(warm, cold, rtol=1e-9, atol=1e-11)
        self.assertLess(info_warm["iterations"], info_cold["iterations"])
        exact = np.linalg.solve(a, b)
        _, info_done = conjugate_gradient(lambda v: a @ v, b, rtol=1e-8, maxiter=5, x0=exact)
        self.assertEqual(info_done["iterations"], 0)
        with self.assertRaises(ValueError):
            conjugate_gradient(lambda v: a @ v, b, rtol=1e-8, maxiter=5, x0=np.ones(49))

    def test_zero_rhs_returns_zero_without_iterations(self) -> None:
        x, info = conjugate_gradient(lambda v: v, np.zeros(5), rtol=1e-10, maxiter=1)
        self.assertEqual(info, {"iterations": 0, "relative_residual": 0.0})
        self.assertTrue(np.all(x == 0.0))

    def test_indefinite_operator_raises(self) -> None:
        a = np.diag([1.0, -1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "not SPD"):
            conjugate_gradient(lambda v: a @ v, np.array([0.0, 1.0, 0.0]), rtol=1e-10, maxiter=10)

    def test_non_convergence_raises(self) -> None:
        a = self._spd(60, 2)
        with self.assertRaises(ConvergenceError):
            conjugate_gradient(lambda v: a @ v, np.ones(60), rtol=1e-14, maxiter=2)

    @staticmethod
    def _poisson(n: int):
        lat = triangle_hex_lattice(n)
        diag, off = assemble_hex_operator(lat, edge_conductances(lat, _smooth_kappa))

        def apply_a(v: np.ndarray) -> np.ndarray:
            return -fv._apply_assembled(diag, off, lat.neighbors, v)

        return lat, apply_a

    def test_stopping_rule_is_decided_on_the_true_residual(self) -> None:
        # spec section 5: ||b - A x||_2 <= rtol ||b||_2. On these systems the
        # recursive residual passes 1e-13 while the true one is still above it
        # (measured with the old rule: 1.12e-13 and 1.34e-13).
        lat, apply_a = self._poisson(48)
        x_coord, y_coord = lat.coords[:, 0], lat.coords[:, 1]
        for name, b in (("ones", np.ones(lat.num_nodes)),
                        ("sin", np.sin(np.pi * x_coord) * np.sin(2 * np.pi * y_coord) + 1.0)):
            with self.subTest(rhs=name):
                x, info = conjugate_gradient(apply_a, b, rtol=1e-13, maxiter=10 * lat.num_nodes + 100)
                true_relative = float(np.linalg.norm(b - apply_a(x)) / np.linalg.norm(b))
                self.assertLessEqual(true_relative, 1e-13)
                self.assertAlmostEqual(info["relative_residual"] / true_relative, 1.0, places=9)

    def test_stagnating_true_residual_raises_with_the_best_iterate(self) -> None:
        # rtol = 1e-15 lies below the binary64 floor of ||b - A x|| here.
        lat, apply_a = self._poisson(48)
        b = np.ones(lat.num_nodes)
        maxiter = 10 * lat.num_nodes + 100
        with self.assertRaisesRegex(ConvergenceError, "stagnated") as caught:
            conjugate_gradient(apply_a, b, rtol=1e-15, maxiter=maxiter)
        exc = caught.exception
        self.assertLess(exc.iterations, maxiter)
        best_true = float(np.linalg.norm(b - apply_a(exc.best)) / np.linalg.norm(b))
        self.assertAlmostEqual(exc.best_relative_residual / best_true, 1.0, places=9)
        self.assertGreater(exc.best_relative_residual, 1e-15)

    def test_unrepresentable_rhs_norm_is_solved_by_exact_rescaling(self) -> None:
        # Codex P2 on #61: ||b||_2 of finite entries can overflow (1e200 -> inf:
        # x = 0 after 0 iterations with a NaN residual) or underflow (1e-200 ->
        # 0: the zero vector returned as "converged").
        a = self._spd(30, 7)
        v = np.random.default_rng(8).standard_normal(30)
        x_ref, info_ref = conjugate_gradient(lambda w: a @ w, v, rtol=1e-12, maxiter=300)
        for scale in (2.0**700, 1e200, 1e-200, 2.0**-1000):
            with self.subTest(scale=scale):
                b = scale * v
                self.assertTrue(np.all(np.isfinite(b)) and np.all(b != 0.0))
                try:
                    x, info = conjugate_gradient(lambda w: a @ w, b, rtol=1e-12, maxiter=300)
                except Exception as exc:  # the solve itself is the contract under test
                    self.fail(f"finite b of scale {scale!r} is not solved: {type(exc).__name__}: {exc}")
                self.assertGreater(info["iterations"], 0)
                self.assertTrue(math.isfinite(info["relative_residual"]))
                self.assertLessEqual(info["relative_residual"], 1e-12)
                np.testing.assert_allclose(x / scale, x_ref, rtol=1e-12, atol=0.0)
                if math.frexp(scale)[0] == 0.5:  # exact power of two: bit-identical to the unscaled solve
                    self.assertTrue(np.array_equal(x, scale * x_ref))
                    self.assertEqual(info, info_ref)

    def test_solution_outside_binary64_raises(self) -> None:
        # A = 1e-300 I, b = 1e10: x = 1e310 overflows; x = 1e-320 would be subnormal.
        b = np.full(4, 1e10)
        with self.assertRaisesRegex(ConvergenceError, "not representable"):
            conjugate_gradient(lambda w: 1e-300 * w, b, rtol=1e-12, maxiter=10)
        with self.assertRaisesRegex(ConvergenceError, "not representable"):
            conjugate_gradient(lambda w: 1e300 * w, np.full(4, 1e-20), rtol=1e-12, maxiter=10)
        with self.assertRaisesRegex(ValueError, "x0 is not representable"):
            conjugate_gradient(lambda w: w, np.full(4, 1e-300), rtol=1e-12, maxiter=10, x0=np.full(4, 1e300))

    def test_true_residual_norm_does_not_underflow(self) -> None:
        # Codex P2 on #61 @266900e: A = diag(1, 2), b = [1, 1e-200], rtol = 1e-250.
        # After one step the residual is ~5e-201 (scaled system), r . r underflows
        # to 0 and the old rule returned "converged" with relative_residual 0.
        a = np.diag([1.0, 2.0])
        b = np.array([1.0, 1e-200])
        try:
            x, info = conjugate_gradient(lambda v: a @ v, b, rtol=1e-250, maxiter=10)
        except Exception as exc:  # any other exception type is a failed contract, not a crash
            self.assertIsInstance(exc, ConvergenceError, f"{type(exc).__name__}: {exc}")
            self.assertIn("below the binary64 range", str(exc))
            true_relative = float(np.linalg.norm(b - a @ exc.best) / np.linalg.norm(b))
            self.assertGreater(exc.best_relative_residual, 1e-250)
            self.assertGreaterEqual(true_relative, 0.0)
        else:
            true = b - a @ x
            scale = float(np.max(np.abs(true)))
            true_relative = 0.0 if scale == 0.0 else scale * float(np.linalg.norm(true / scale)) / float(
                np.linalg.norm(b))
            self.fail(f"reported convergence (relative_residual={info['relative_residual']!r}) although the "
                      f"true relative residual is {true_relative!r} > rtol 1e-250")

    def test_scaled_norm_agrees_bitwise_with_the_unscaled_one_in_range(self) -> None:
        # The pre-check and the true-residual decision keep their old arithmetic
        # wherever r . r is safely in range, so converged solves are unchanged.
        v = np.random.default_rng(3).standard_normal(500)
        for scale in (1.0, 2.0**-300, 2.0**-440):
            with self.subTest(scale=scale):
                w = scale * v
                squared = float(w @ w)
                self.assertGreaterEqual(squared, fv._CG_SAFE_SQUARED_NORM)
                self.assertEqual(fv._norm2(w, squared), math.sqrt(squared))
        tiny = 2.0**-600 * v  # r . r underflows: the scaled norm takes over
        self.assertEqual(float(tiny @ tiny), 0.0)
        self.assertEqual(fv._norm2(tiny, 0.0), math.ldexp(float(np.linalg.norm(v)), -600))
        self.assertEqual(fv._norm2(np.zeros(3), 0.0), 0.0)

    def test_maxiter_error_carries_the_true_residual_of_the_start(self) -> None:
        a = self._spd(60, 2)
        b = np.ones(60)
        with self.assertRaises(ConvergenceError) as caught:
            conjugate_gradient(lambda v: a @ v, b, rtol=1e-14, maxiter=2)
        self.assertEqual(caught.exception.iterations, 2)
        self.assertTrue(np.all(caught.exception.best == 0.0))
        self.assertEqual(caught.exception.best_relative_residual, 1.0)

    def test_argument_contract(self) -> None:
        for kwargs, exc in (({"rtol": 1.0, "maxiter": 5}, ValueError), ({"rtol": 0.0, "maxiter": 5}, ValueError),
                            ({"rtol": 1e-8, "maxiter": 0}, ValueError), ({"rtol": 1e-8, "maxiter": 2.0}, TypeError)):
            with self.subTest(kwargs=kwargs), self.assertRaises(exc):
                conjugate_gradient(lambda v: v, np.ones(3), **kwargs)
        with self.assertRaises(ValueError):
            conjugate_gradient(lambda v: v, np.array([1.0, np.nan]), rtol=1e-8, maxiter=5)
        with self.assertRaises(TypeError):
            conjugate_gradient(np.eye(2), np.ones(2), rtol=1e-8, maxiter=5)


class TestSteadySolve(unittest.TestCase):
    def test_recovers_a_discrete_solution(self) -> None:
        lat = triangle_hex_lattice(30)
        c = edge_conductances(lat, _smooth_kappa)
        u_true = np.random.default_rng(4).standard_normal(lat.num_nodes)
        f = -apply_hex_laplacian(u_true, lat, c)
        u, info = solve_steady_diffusion(lat, c, f)
        np.testing.assert_allclose(u, u_true, atol=1e-9)
        self.assertLessEqual(info["relative_residual"], 1e-13)

    def test_torus_is_refused(self) -> None:
        lat = torus_hex_lattice(8)
        with self.assertRaisesRegex(ValueError, "singular"):
            solve_steady_diffusion(lat, 1.0, np.zeros(lat.num_nodes))

    def test_contract(self) -> None:
        lat = triangle_hex_lattice(6)
        with self.assertRaises(ValueError):
            solve_steady_diffusion(lat, 1.0, np.zeros(lat.num_nodes + 1))
        with self.assertRaises(TypeError):
            solve_steady_diffusion(object(), 1.0, np.zeros(3))
        with self.assertRaises(ConvergenceError):
            solve_steady_diffusion(triangle_hex_lattice(40), 1.0, np.ones(triangle_hex_lattice(40).num_nodes),
                                   maxiter=3)

    def test_forged_lattice_topology_is_rejected_w9_f01(self) -> None:
        # GPT6 W9-F01: HexLattice is a plain dataclass; a lattice whose
        # neighbours all point to node 0 was accepted by the operator.
        lat = triangle_hex_lattice(8)
        u = np.random.default_rng(9).standard_normal(lat.num_nodes)

        def forged(**fields):
            values = {"n": lat.n, "h": lat.h, "domain": lat.domain, "coords": lat.coords, "neighbors": lat.neighbors}
            values.update(fields)
            return HexLattice(**values)

        other = triangle_hex_lattice(9)
        swapped = np.array(lat.neighbors)
        swapped[[0, 1]] = swapped[[1, 0]]
        cases = {
            "neighbours all node 0": forged(neighbors=np.zeros_like(lat.neighbors)),
            "two rows swapped": forged(neighbors=swapped),
            "neighbours as a list": forged(neighbors=lat.neighbors.tolist()),
            "neighbours int32": forged(neighbors=lat.neighbors.astype(np.int32)),
            "shifted coords": forged(coords=lat.coords + 1e-12),
            "wrong h": forged(h=0.1),
            "nan h": forged(h=float("nan")),
            "arrays of n=9": forged(coords=other.coords, neighbors=other.neighbors),
            "n as float": forged(n=8.0),
            "n as bool": forged(n=True),
            "n too small": forged(n=2),
            "torus with odd n": HexLattice(n=7, h=1 / 7, domain="torus_periodic", coords=lat.coords,
                                           neighbors=lat.neighbors),
        }
        for name, fake in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    apply_hex_laplacian(u, fake)
                with self.assertRaises(ValueError):
                    assemble_hex_operator(fake)
                with self.assertRaises(ValueError):
                    edge_conductances(fake, _smooth_kappa)
                with self.assertRaises(ValueError):
                    solve_steady_diffusion(fake, 1.0, np.ones(lat.num_nodes))
        # An equal lattice (numpy int n, fresh arrays) is accepted and gives the same operator.
        twin = forged(n=np.int64(8), coords=lat.coords.copy(), neighbors=lat.neighbors.copy())
        self.assertTrue(np.array_equal(apply_hex_laplacian(u, twin), apply_hex_laplacian(u, lat)))

    def test_lattice_mutated_after_construction_is_rejected(self) -> None:
        lat = triangle_hex_lattice(8)
        u = np.ones(lat.num_nodes)
        apply_hex_laplacian(u, lat)
        original = int(lat.neighbors[0, 0])
        lat.neighbors.flags.writeable = True  # owned array: the read-only flag is advisory
        lat.neighbors[0, 0] = -1 if original != -1 else 0
        with self.assertRaisesRegex(ValueError, "canonical"):
            apply_hex_laplacian(u, lat)
        with self.assertRaisesRegex(ValueError, "canonical"):
            explicit_euler_heat(u, lat, kappa=1.0, dt=1e-6, steps=1)
        with self.assertRaisesRegex(ValueError, "canonical"):
            crank_nicolson_heat(u, lat, kappa=1.0, dt=1e-3, steps=1)
        self.assertEqual(int(triangle_hex_lattice(8).neighbors[0, 0]), original)  # factory output unaffected

    def test_lattice_type_is_checked(self) -> None:
        fake = HexLattice(n=3, h=1 / 3, domain="hexagon", coords=np.zeros((1, 2)), neighbors=-np.ones((1, 6), int))
        with self.assertRaises(TypeError):
            apply_hex_laplacian(np.zeros(1), fake)


_U = 2.0**-53

#: An rtol that CG must NOT reach on the n = 12 systems below, so V1 gives up
#: and V2 has to decide at the rounding floor. 1e-16 (~0.9 u) is not safe:
#: the attainable true relative residual is itself ~1e-16 and depends on the
#: summation order of the BLAS ddot kernel. Measured on PR #61 head 6d219d9
#: with OPENBLAS_CORETYPE in {SkylakeX, Sandybridge, Haswell, Zen, Prescott}:
#: with rtol=1e-16, V1 CONVERGES on Haswell/Zen/Prescott (9.9e-17 / 9.8e-17;
#: the CI run of 6d219d9 failed exactly these tests) and stagnates only on
#: SkylakeX/Sandybridge. With
#: 1e-17 (~u/11) every kernel stagnates (best 9.4e-17 .. 3.8e-16, at most 683
#: of 2000 iterations), i.e. the premise holds with a margin of ~10.
_RTOL_BELOW_ROUNDING = 1e-17


def _hex_system(n: int, *, seed: int = 3):
    """-L u = f on the n-triangle (kappa = 1): (apply_a, apply_abs_a, f)."""
    lat = triangle_hex_lattice(n)
    diag, off = assemble_hex_operator(lat, 1.0)
    nb = lat.neighbors

    def apply_a(v: np.ndarray) -> np.ndarray:
        return -fv._apply_assembled(diag, off, nb, v)

    def apply_abs_a(v: np.ndarray) -> np.ndarray:
        return fv._apply_assembled(np.abs(diag), np.abs(off), nb, v)

    return apply_a, apply_abs_a, np.random.default_rng(seed).standard_normal(lat.num_nodes)


class TestConjugateGradientV2(unittest.TestCase):
    """CG stopping rule V2 (spec/HEX_FV_SOLVER_v2.md, E1012-07)."""

    def _v2(self, apply_a, apply_abs_a, b, *, rtol, maxiter=2000, c=fv.HEX_RESIDUAL_FLOOR_FACTOR, **kwargs):
        return conjugate_gradient(apply_a, b, rtol=rtol, maxiter=maxiter, stopping="v2", apply_abs_a=apply_abs_a,
                                  floor_factor=c, **kwargs)

    def _expected_terms(self, apply_abs_a, x, b, relative, c):
        """Independent recomputation of the V2 terms in the original scale."""
        s = float(np.linalg.norm(apply_abs_a(np.abs(x))))
        bn = float(np.linalg.norm(b))
        return c * _U * (s + bn) / bn, relative * bn / (s + bn)

    def test_converged_is_bit_identical_to_v1_and_reports_eta(self) -> None:
        apply_a, apply_abs_a, f = _hex_system(9)
        x1, i1 = conjugate_gradient(apply_a, f, rtol=1e-12, maxiter=500)
        x2, i2 = self._v2(apply_a, apply_abs_a, f, rtol=1e-12, maxiter=500)
        self.assertTrue(np.array_equal(x1, x2))
        self.assertEqual((i2["iterations"], i2["relative_residual"]), (i1["iterations"], i1["relative_residual"]))
        self.assertEqual(i2["status"], "converged")
        self.assertNotIn("status", i1)  # V1 output is unchanged
        floor, eta = self._expected_terms(apply_abs_a, x2, f, i2["relative_residual"], 8)
        np.testing.assert_allclose([i2["rounding_floor"], i2["backward_error"]], [floor, eta], rtol=1e-12)
        self.assertEqual(i2["tol_eff"], max(1e-12, i2["rounding_floor"]))
        np.testing.assert_allclose(np.linalg.norm(f - apply_a(x2)) / np.linalg.norm(f), i2["relative_residual"],
                                   rtol=1e-12)

    def test_v2_checks_the_last_iterate_before_maxiter_give_up(self) -> None:
        # Codex P2 on #61 (dd24538): the recursive residual of the final iterate stayed
        # above a tiny rtol, so its true residual was never computed and give_up judged
        # the initial iterate (best_relative = 1) instead of the far better last one.
        d = np.array([1.0, 1.0 + 2.0**-50])
        apply_a = lambda v: d * v  # noqa: E731
        apply_abs_a = lambda v: np.abs(d) * v  # noqa: E731
        b = np.array([1.0, 1.0])
        x, info = self._v2(apply_a, apply_abs_a, b, rtol=1e-20, maxiter=1)
        self.assertEqual((info["status"], info["stop_reason"]), ("at_rounding_floor", "maxiter"))
        self.assertLess(info["relative_residual"], 1e-12)
        np.testing.assert_allclose(x, b / d, rtol=1e-12)

    def test_at_rounding_floor_where_v1_stagnates(self) -> None:
        apply_a, apply_abs_a, f = _hex_system(12)
        with self.assertRaisesRegex(ConvergenceError, "stagnated") as ctx:
            conjugate_gradient(apply_a, f, rtol=_RTOL_BELOW_ROUNDING, maxiter=2000)
        x, info = self._v2(apply_a, apply_abs_a, f, rtol=_RTOL_BELOW_ROUNDING)
        self.assertEqual((info["status"], info["stop_reason"]), ("at_rounding_floor", "stagnation"))
        # The returned iterate is V1's best iterate, and rtol was NOT reached.
        self.assertTrue(np.array_equal(x, ctx.exception.best))
        self.assertEqual(info["relative_residual"], ctx.exception.best_relative_residual)
        self.assertGreater(info["relative_residual"], _RTOL_BELOW_ROUNDING)
        self.assertLessEqual(info["relative_residual"], info["tol_eff"])
        self.assertLessEqual(info["backward_error"], 8 * _U)  # at_rounding_floor implies eta <= c u
        floor, eta = self._expected_terms(apply_abs_a, x, f, info["relative_residual"], 8)
        np.testing.assert_allclose([info["rounding_floor"], info["backward_error"]], [floor, eta], rtol=1e-12)

    def test_maxiter_and_underflow_give_ups_also_consult_the_floor(self) -> None:
        apply_a, apply_abs_a, f = _hex_system(9)
        with self.assertRaisesRegex(ConvergenceError, "no convergence"):
            conjugate_gradient(apply_a, f, rtol=1e-18, maxiter=380)
        _, info = self._v2(apply_a, apply_abs_a, f, rtol=1e-18, maxiter=380)
        self.assertEqual((info["status"], info["stop_reason"]), ("at_rounding_floor", "maxiter"))
        # maxiter before any true-residual check: the best iterate is x0 = 0, far above the floor.
        with self.assertRaisesRegex(ConvergenceError, "no convergence"):
            self._v2(apply_a, apply_abs_a, f, rtol=1e-12, maxiter=2)
        # r . r underflow: V2 accepts only through the floor test as well.
        b = np.array([1.0, 2.0**-600])
        _, info = self._v2(lambda v: v, lambda v: v, b, rtol=1e-300, x0=np.array([1.0, 0.0]))
        self.assertEqual((info["status"], info["stop_reason"]), ("at_rounding_floor", "underflow"))

    def test_a_real_residual_above_the_floor_is_rejected(self) -> None:
        # Defect: the operator is applied in float32. The true residual then
        # stagnates near 1e-8, far above the binary64 rounding floor: V2 must
        # keep V1's ConvergenceError.
        apply_a, apply_abs_a, f = _hex_system(9)

        def apply_a32(v: np.ndarray) -> np.ndarray:
            return apply_a(v).astype(np.float32).astype(np.float64)

        with self.assertRaisesRegex(ConvergenceError, "stagnated"):
            self._v2(apply_a32, apply_abs_a, f, rtol=1e-10)

    def test_the_acceptance_boundary_is_pinned(self) -> None:
        # Choose c so that tol_eff sits 1e-6 (relative) below / above the best
        # residual: below must reject, above must accept (obere Grenze genagelt).
        apply_a, apply_abs_a, f = _hex_system(9)

        def apply_a32(v: np.ndarray) -> np.ndarray:
            return apply_a(v).astype(np.float32).astype(np.float64)

        with self.assertRaises(ConvergenceError) as ctx:
            conjugate_gradient(apply_a32, f, rtol=1e-10, maxiter=2000)
        best, relative = ctx.exception.best, ctx.exception.best_relative_residual
        s, bn = float(np.linalg.norm(apply_abs_a(np.abs(best)))), float(np.linalg.norm(f))
        c_crit = relative * bn / (_U * (s + bn))
        self.assertGreater(c_crit, 1.0)
        with self.assertRaisesRegex(ConvergenceError, "stagnated"):
            self._v2(apply_a32, apply_abs_a, f, rtol=1e-10, c=c_crit * (1 - 1e-6))
        _, info = self._v2(apply_a32, apply_abs_a, f, rtol=1e-10, c=c_crit * (1 + 1e-6))
        self.assertEqual(info["status"], "at_rounding_floor")

    def test_status_is_invariant_under_power_of_two_scaling(self) -> None:
        # One case per V2 outcome; the status is pinned so that a platform on
        # which rtol becomes reachable cannot silently test the wrong path.
        for n, rtol, status in ((9, 1e-12, "converged"), (12, _RTOL_BELOW_ROUNDING, "at_rounding_floor")):
            apply_a, apply_abs_a, f = _hex_system(n)
            x_ref, ref = self._v2(apply_a, apply_abs_a, f, rtol=rtol)
            self.assertEqual(ref["status"], status)
            for k, j in ((0, 600), (0, -600), (-300, 200), (250, -250), (40, 3)):
                with self.subTest(n=n, k=k, j=j):
                    def scaled_a(v, k=k, base=apply_a):
                        return np.ldexp(base(v), k)

                    def scaled_abs(v, k=k, base=apply_abs_a):
                        return np.ldexp(base(v), k)

                    x, info = self._v2(scaled_a, scaled_abs, np.ldexp(f, j), rtol=rtol)
                    for key in ("status", "iterations", "relative_residual", "tol_eff", "rounding_floor",
                                "backward_error"):
                        self.assertEqual(info[key], ref[key], key)
                    self.assertTrue(np.array_equal(x, np.ldexp(x_ref, j - k)))

    def test_non_finite_inputs_fail_closed(self) -> None:
        apply_a, apply_abs_a, f = _hex_system(9)
        for bad in (np.nan, np.inf, -np.inf):
            for pos in (0, -1):
                with self.subTest(bad=bad, pos=pos):
                    b = f.copy()
                    b[pos] = bad
                    with self.assertRaises(ValueError):
                        self._v2(apply_a, apply_abs_a, b, rtol=1e-12)
                    x0 = np.zeros_like(f)
                    x0[pos] = bad
                    with self.assertRaises(ValueError):
                        self._v2(apply_a, apply_abs_a, f, rtol=1e-12, x0=x0)

    def test_a_non_finite_floor_never_accepts(self) -> None:
        apply_a, apply_abs_a, f = _hex_system(12)
        for bad in (np.nan, np.inf):
            with self.subTest(bad=bad):
                def poisoned(v, bad=bad):
                    out = apply_abs_a(v)
                    out[-1] = bad
                    return out

                # on the floor path (rtol unreachable) and on the converged path
                for rtol in (_RTOL_BELOW_ROUNDING, 1e-10):
                    # Any exception type is caught, so a wrong class fails by assertion, not by crash.
                    with self.assertRaises(Exception) as ctx:
                        self._v2(apply_a, poisoned, f, rtol=rtol)
                    self.assertIsInstance(ctx.exception, ConvergenceError)
                    self.assertIn("not computable", str(ctx.exception))
        # || |A||x| || overflowing despite finite entries
        with self.assertRaisesRegex(ConvergenceError, "not computable"):
            self._v2(apply_a, lambda v: np.full_like(v, 1e308), f, rtol=_RTOL_BELOW_ROUNDING)

    def test_acceptance_hangs_on_true_comparisons(self) -> None:
        # NaN orientation: if the floor terms were NaN, the floor branch must not accept.
        apply_a, apply_abs_a, f = _hex_system(12)
        nan_terms = {"rounding_floor": math.nan, "tol_eff": math.nan, "backward_error": math.nan}
        with mock.patch.object(fv, "_floor_terms", return_value=nan_terms):
            with self.assertRaisesRegex(ConvergenceError, "stagnated"):
                self._v2(apply_a, apply_abs_a, f, rtol=_RTOL_BELOW_ROUNDING)
        # A floor >= 1 carries no information and never accepts.
        with mock.patch.object(fv, "_floor_terms", return_value={"rounding_floor": 1.0, "tol_eff": 1.0,
                                                                   "backward_error": 0.0}):
            with self.assertRaisesRegex(ConvergenceError, "stagnated"):
                self._v2(apply_a, apply_abs_a, f, rtol=_RTOL_BELOW_ROUNDING)

    def test_abs_operator_contract(self) -> None:
        apply_a, apply_abs_a, f = _hex_system(9)
        with self.assertRaisesRegex(ValueError, "shape"):
            self._v2(apply_a, lambda v: apply_abs_a(v)[:-1], f, rtol=1e-12)
        with self.assertRaisesRegex(ValueError, ">= 0"):
            self._v2(apply_a, lambda v: -apply_abs_a(v), f, rtol=1e-12)

    def test_zero_rhs_is_exact(self) -> None:
        apply_a, apply_abs_a, f = _hex_system(9)
        x, info = self._v2(apply_a, apply_abs_a, np.zeros_like(f), rtol=1e-12)
        self.assertTrue(np.array_equal(x, np.zeros_like(f)))
        self.assertEqual(info, {"iterations": 0, "relative_residual": 0.0, "status": "converged",
                                "rounding_floor": 0.0, "tol_eff": 1e-12, "backward_error": 0.0})

    def test_argument_contract(self) -> None:
        apply_a, apply_abs_a, f = _hex_system(9)
        for stopping in ("v3", "V2", "", None, 2):
            with self.subTest(stopping=stopping), self.assertRaises(ValueError):
                conjugate_gradient(apply_a, f, rtol=1e-12, maxiter=50, stopping=stopping)
        with self.assertRaises(TypeError):
            conjugate_gradient(apply_a, f, rtol=1e-12, maxiter=50, stopping="v2", floor_factor=8)
        for c in (None, 0, -1.0, 0.5, math.nan, math.inf, True, "8"):
            with self.subTest(c=c), self.assertRaises((TypeError, ValueError)):
                self._v2(apply_a, apply_abs_a, f, rtol=1e-12, c=c)
        for extra in ({"apply_abs_a": apply_abs_a}, {"floor_factor": 8}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "v2"):
                conjugate_gradient(apply_a, f, rtol=1e-12, maxiter=50, **extra)

    def test_steady_solve_v2_on_the_hex_operator(self) -> None:
        lat = triangle_hex_lattice(12)
        c = edge_conductances(lat, _smooth_kappa)
        f = np.random.default_rng(5).standard_normal(lat.num_nodes)
        with self.assertRaises(ConvergenceError):
            solve_steady_diffusion(lat, c, f, rtol=_RTOL_BELOW_ROUNDING)
        u, info = solve_steady_diffusion(lat, c, f, rtol=_RTOL_BELOW_ROUNDING, stopping="v2")
        self.assertEqual(info["status"], "at_rounding_floor")
        self.assertGreater(info["relative_residual"], _RTOL_BELOW_ROUNDING)
        diag, off = assemble_hex_operator(lat, c)
        abs_au = (np.abs(diag) * np.abs(u) + (np.abs(off) * np.abs(u)[np.clip(lat.neighbors, 0, None)]).sum(axis=1))
        expected = 8 * _U * (np.linalg.norm(abs_au) + np.linalg.norm(f)) / np.linalg.norm(f)
        np.testing.assert_allclose(info["rounding_floor"], expected, rtol=1e-12)
        u1, i1 = solve_steady_diffusion(lat, c, f, rtol=1e-10)
        u2, i2 = solve_steady_diffusion(lat, c, f, rtol=1e-10, stopping="v2")
        self.assertTrue(np.array_equal(u1, u2))
        self.assertEqual(i2["status"], "converged")
        with self.assertRaises(ValueError):
            solve_steady_diffusion(lat, c, f, stopping="v3")


if __name__ == "__main__":
    unittest.main()
