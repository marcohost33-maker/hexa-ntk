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

    def test_lattice_type_is_checked(self) -> None:
        fake = HexLattice(n=3, h=1 / 3, domain="hexagon", coords=np.zeros((1, 2)), neighbors=-np.ones((1, 6), int))
        with self.assertRaises(TypeError):
            apply_hex_laplacian(np.zeros(1), fake)


if __name__ == "__main__":
    unittest.main()
