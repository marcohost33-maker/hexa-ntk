"""
hex_fv_solver_hex_v3.py
=======================
Mechanistic finite-volume layer on regular hex lattices (contract:
spec/HEX_FV_SOLVER_v1.md). Verified by the Reality Anchor reference cases
(reference_case_hex_v3.py) and meant as the mechanistic baseline of Phase 1
(issue #60).

Lattices: the unit equilateral triangle with homogeneous Dirichlet boundary
nodes (slot -1 = boundary node with value 0) and the doubly periodic hex torus.
Operator: two-point-flux FV Laplacian

    (L_c u)_i = w * sum_s c_{i,s} (u_{n(i,s)} - u_i),   w = 2 / (3 h^2)

with symmetric edge conductances c (a scalar kappa, or kappa at the edge
midpoints). ``w`` comes from the engine's ``hex_laplacian_stencil_weight``.

Claim boundary: regular lattices, scalar isotropic kappa, two-point flux. No
irregular/anisotropic meshes, no discrete maximum principle beyond the
positivity gate, no performance claim.
"""

from __future__ import annotations

__all__ = [
    "LATTICE_SLOTS",
    "SLOT_DIRECTIONS",
    "ConvergenceError",
    "HexLattice",
    "triangle_hex_lattice",
    "torus_hex_lattice",
    "edge_conductances",
    "apply_hex_laplacian",
    "assemble_hex_operator",
    "positivity_max_dt",
    "explicit_euler_heat",
    "crank_nicolson_heat",
    "conjugate_gradient",
    "solve_steady_diffusion",
]

import functools
import importlib.util
import math
import numbers
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np

#: Neighbour slots, counter-clockwise from east, as (di, dj) index offsets of
#: the node x_ij = h * (i + j/2, j * sqrt(3)/2). Slot s and s + 3 are opposite.
LATTICE_SLOTS: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))

#: Unit vector of each slot (angle s * 60 degrees).
SLOT_DIRECTIONS = np.array([[math.cos(s * math.pi / 3.0), math.sin(s * math.pi / 3.0)] for s in range(6)])
SLOT_DIRECTIONS.flags.writeable = False

_SQRT3 = math.sqrt(3.0)
_DOMAINS = ("triangle_dirichlet", "torus_periodic")
_ENGINE_PATH = Path(__file__).resolve().parents[1] / "hexantk_v3.py"

Coefficient = Union[float, np.ndarray]


#: Consecutive true-residual checks without a new minimum after which CG stops
#: as stagnated (the binary64 floor of ||b - A x|| lies above rtol * ||b||).
_CG_STAGNATION_CHECKS = 50


class ConvergenceError(RuntimeError):
    """An iterative solve did not reach its tolerance (fail-closed).

    ``best`` is the iterate with the smallest TRUE relative residual
    ``best_relative_residual`` seen before the stop, ``iterations`` the
    iterations spent. All three are diagnostics (``None`` when unknown), never
    a converged result.
    """

    def __init__(
        self,
        message: str,
        *,
        best: np.ndarray | None = None,
        best_relative_residual: float | None = None,
        iterations: int | None = None,
    ) -> None:
        super().__init__(message)
        self.best = best
        self.best_relative_residual = best_relative_residual
        self.iterations = iterations


@functools.lru_cache(maxsize=1)
def _engine() -> Any:
    """Load the canonical engine src/hexantk_v3.py (the code under test)."""
    spec = importlib.util.spec_from_file_location("hexantk_v3_hex_fv_solver", _ENGINE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise ImportError(f"cannot load engine from {_ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module via sys.modules during class creation
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _int_at_least(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must fit in a finite float") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return result


# ------------------------------------------------------------------
# Lattices (spec section 2)
# ------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class HexLattice:
    """Unknown nodes of a regular hex lattice with spacing h = 1/n.

    ``neighbors[i, s] == -1`` (triangle only) means the slot-``s`` neighbour is
    a homogeneous Dirichlet boundary node (value 0), NOT a missing neighbour.
    Arrays are read-only.
    """

    n: int
    h: float
    domain: str
    coords: np.ndarray
    neighbors: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.coords.shape[0])

    @property
    def cell_area(self) -> float:
        return _SQRT3 / 2.0 * self.h * self.h


def _freeze(*arrays: np.ndarray) -> None:
    for arr in arrays:
        arr.flags.writeable = False


def triangle_hex_lattice(n: int) -> HexLattice:
    """Interior nodes of the unit equilateral triangle: i, j >= 1, i + j <= n - 1."""
    n = _int_at_least(n, name="n", minimum=3)
    index = np.full((n + 1, n + 1), -1, dtype=np.int64)
    ii, jj = [], []
    for j in range(1, n - 1):
        for i in range(1, n - j):
            index[i, j] = len(ii)
            ii.append(i)
            jj.append(j)
    i_arr = np.asarray(ii, dtype=np.int64)
    j_arr = np.asarray(jj, dtype=np.int64)
    h = 1.0 / n
    coords = np.stack([h * (i_arr + 0.5 * j_arr), h * j_arr * (_SQRT3 / 2.0)], axis=1)
    neighbors = np.empty((i_arr.size, 6), dtype=np.int64)
    for slot, (di, dj) in enumerate(LATTICE_SLOTS):
        ni, nj = i_arr + di, j_arr + dj
        interior = (ni >= 1) & (nj >= 1) & (ni + nj <= n - 1)
        neighbors[:, slot] = np.where(interior, index[np.clip(ni, 0, n), np.clip(nj, 0, n)], -1)
    _freeze(coords, neighbors)
    return HexLattice(n=n, h=h, domain="triangle_dirichlet", coords=coords, neighbors=neighbors)


def torus_hex_lattice(n: int) -> HexLattice:
    """All n^2 nodes of the hex torus with periods (1, 0) and (0, sqrt(3)/2); n even.

    Index identification (i, j) ~ (i + n, j) ~ (i - n/2, j + n).
    """
    n = _int_at_least(n, name="n", minimum=4)
    if n % 2:
        raise ValueError(f"the hex torus needs an even n (period (0, sqrt(3)/2) is (i - n/2, j + n)), got {n}")
    i_arr, j_arr = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    h = 1.0 / n
    coords = np.stack([np.mod(h * (i_arr + 0.5 * j_arr), 1.0), h * j_arr * (_SQRT3 / 2.0)], axis=1)
    neighbors = np.empty((n * n, 6), dtype=np.int64)
    for slot, (di, dj) in enumerate(LATTICE_SLOTS):
        ni, nj = i_arr + di, j_arr + dj
        ni = np.where(nj == n, ni + n // 2, ni)
        ni = np.where(nj == -1, ni - n // 2, ni)
        nj = np.mod(nj, n)
        neighbors[:, slot] = np.mod(ni, n) * n + nj
    _freeze(coords, neighbors)
    return HexLattice(n=n, h=h, domain="torus_periodic", coords=coords, neighbors=neighbors)


@functools.lru_cache(maxsize=8)
def _canonical_lattice(domain: str, n: int) -> HexLattice:
    """Private reference copy of a factory lattice (never handed to callers)."""
    return triangle_hex_lattice(n) if domain == "triangle_dirichlet" else torus_hex_lattice(n)


def _same_array(value: Any, reference: np.ndarray) -> bool:
    return (
        isinstance(value, np.ndarray)
        and value.shape == reference.shape
        and value.dtype == reference.dtype
        and bool(np.array_equal(value, reference))
    )


def _check_lattice(lattice: Any) -> HexLattice:
    """Accept only a lattice equal to its factory output; return the private copy.

    HexLattice is a plain dataclass, so n, h, coords and neighbours can be
    forged (W9-F01: a lattice with every neighbour set to node 0 was accepted).
    The operator trusts the topology, so it is compared field by field with
    the canonical lattice of (domain, n), and the read-only canonical copy is
    what the caller computes on.
    """
    if not isinstance(lattice, HexLattice) or lattice.domain not in _DOMAINS:
        raise TypeError("lattice must be a HexLattice")
    n = lattice.n
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise ValueError(f"lattice.n must be an int, got {n!r}")
    try:
        canonical = _canonical_lattice(lattice.domain, int(n))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"lattice is not a valid {lattice.domain} lattice: {exc}") from exc
    h = lattice.h
    if (
        isinstance(h, bool)
        or not isinstance(h, numbers.Real)
        or not float(h) == canonical.h
        or not _same_array(lattice.coords, canonical.coords)
        or not _same_array(lattice.neighbors, canonical.neighbors)
    ):
        raise ValueError(
            f"lattice does not match the canonical {lattice.domain} lattice for n={int(n)} "
            "(h, coords or neighbors differ); build it with triangle_hex_lattice or torus_hex_lattice"
        )
    return canonical


def _check_field(u: Any, lattice: HexLattice, *, name: str = "u") -> np.ndarray:
    arr = np.asarray(u, dtype=np.float64)
    if arr.shape != (lattice.num_nodes,):
        raise ValueError(f"{name} must have shape ({lattice.num_nodes},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


# ------------------------------------------------------------------
# Conductances and operator (spec section 3)
# ------------------------------------------------------------------


def edge_conductances(lattice: HexLattice, kappa: Callable[[np.ndarray], Any]) -> np.ndarray:
    """kappa at every edge midpoint, exactly symmetric, as a read-only (M, 6) array.

    ``kappa`` maps midpoint coordinates (K, 2) to K values; every value must be
    finite and > 0.
    """
    lattice = _check_lattice(lattice)
    if not callable(kappa):
        raise TypeError("kappa must be callable on (K, 2) coordinates")
    c = np.empty(lattice.neighbors.shape)
    for slot in range(6):
        midpoints = lattice.coords + 0.5 * lattice.h * SLOT_DIRECTIONS[slot]
        values = np.asarray(kappa(midpoints), dtype=np.float64)
        if values.shape != (lattice.num_nodes,):
            raise ValueError(f"kappa must return shape ({lattice.num_nodes},), got {values.shape}")
        c[:, slot] = values
    # One value per interior edge: computed from slots 0-2, copied to the partner.
    for slot in range(3):
        rows = np.nonzero(lattice.neighbors[:, slot] >= 0)[0]
        c[lattice.neighbors[rows, slot], slot + 3] = c[rows, slot]
    if not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        raise ValueError("kappa must be finite and > 0 at every edge midpoint")
    _freeze(c)
    return c


def _check_conductance(conductance: Any, lattice: HexLattice) -> np.ndarray:
    c = np.asarray(conductance, dtype=np.float64)
    if c.shape != lattice.neighbors.shape:
        raise ValueError(f"conductance must have shape {lattice.neighbors.shape}, got {c.shape}")
    if not np.all(np.isfinite(c)):
        raise ValueError("conductance must be finite")
    # Same positivity contract as the scalar and callable paths (spec section 3):
    # c <= 0 makes L_c anti-diffusive and I - dt/2 L_c no longer SPD.
    if np.any(c <= 0.0):
        raise ValueError("conductance must be > 0 on every slot")
    for slot in range(3):
        rows = np.nonzero(lattice.neighbors[:, slot] >= 0)[0]
        if not np.array_equal(c[rows, slot], c[lattice.neighbors[rows, slot], slot + 3]):
            raise ValueError("conductance must be symmetric: c[i, s] == c[n(i, s), s + 3] on every edge")
    return c


def _coefficient(kappa: Any, lattice: HexLattice) -> Tuple[np.ndarray, bool]:
    """Return (conductance array, is_scalar)."""
    if isinstance(kappa, np.ndarray):
        return _check_conductance(kappa, lattice), False
    value = _positive_finite(kappa, name="kappa")
    return np.full(lattice.neighbors.shape, value), True


def apply_hex_laplacian(u: Any, lattice: HexLattice, kappa: Coefficient = 1.0) -> np.ndarray:
    """(L_c u)_i = w * sum_s c_{i,s} (u_{n(i,s)} - u_i), u_{-1} = 0 (Dirichlet zero)."""
    lattice = _check_lattice(lattice)
    arr = _check_field(u, lattice)
    c, scalar = _coefficient(kappa, lattice)
    weight = float(_engine().hex_laplacian_stencil_weight(lattice.h))
    nb = lattice.neighbors
    gathered = np.where(nb >= 0, arr[np.clip(nb, 0, None)], 0.0)
    if scalar:
        # Constant coefficient: the 7-point form, bit-identical to Phase 0a.
        return weight * float(kappa) * (gathered.sum(axis=1) - 6.0 * arr)
    return weight * (c * (gathered - arr[:, None])).sum(axis=1)


def assemble_hex_operator(lattice: HexLattice, kappa: Coefficient = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble L_c as (diag[M], off[M, 6]).

    (L_c u)_i = diag_i * u_i + sum_s off[i, s] * u_{n(i,s)}; a slot pointing to
    a Dirichlet node carries off = 0 (its value is 0) but still contributes to
    the diagonal.
    """
    lattice = _check_lattice(lattice)
    c, scalar = _coefficient(kappa, lattice)
    weight = float(_engine().hex_laplacian_stencil_weight(lattice.h))
    if scalar:
        coupling = float(kappa) * weight
        off = np.where(lattice.neighbors >= 0, coupling, 0.0)
        return np.full(lattice.num_nodes, -6.0 * coupling), off
    coupling = weight * c
    off = np.where(lattice.neighbors >= 0, coupling, 0.0)
    return -coupling.sum(axis=1), off


def _apply_assembled(diag: np.ndarray, off: np.ndarray, neighbors: np.ndarray, u: np.ndarray) -> np.ndarray:
    return diag * u + (off * u[np.clip(neighbors, 0, None)]).sum(axis=1)


# ------------------------------------------------------------------
# Positivity gate and explicit Euler (spec section 4)
# ------------------------------------------------------------------


def positivity_max_dt(diag: Any, off: Any) -> float:
    """Largest dt with u + dt * A u non-negative for every non-negative u (#60 item 1).

    Gate on the ASSEMBLED operator A = (diag, off): requires a negative
    diagonal and non-negative off-diagonals (-A is then an M-matrix), and
    returns min_i 1 / |diag_i|, rounded so 1 + dt * diag_i >= 0 holds in
    binary64. A violated sign structure raises ValueError (contract error).
    Positivity only; not a discrete maximum principle for other schemes.
    """
    diag_arr = np.asarray(diag, dtype=np.float64)
    off_arr = np.asarray(off, dtype=np.float64)
    if diag_arr.ndim != 1 or diag_arr.size == 0:
        raise ValueError("diag must be a non-empty 1-D array")
    if off_arr.ndim != 2 or off_arr.shape[0] != diag_arr.size:
        raise ValueError("off must have shape (len(diag), slots)")
    if not (np.all(np.isfinite(diag_arr)) and np.all(np.isfinite(off_arr))):
        raise ValueError("operator entries must be finite")
    if np.any(diag_arr >= 0.0):
        raise ValueError("positivity gate: every diagonal entry must be negative")
    if np.any(off_arr < 0.0):
        raise ValueError(
            "positivity gate: negative off-diagonal entry, the assembled operator "
            "does not have M-matrix sign structure"
        )
    largest = float(np.max(-diag_arr))
    dt = 1.0 / largest
    while 1.0 - dt * largest < 0.0:
        dt = float(np.nextafter(dt, 0.0))
    return dt


def explicit_euler_heat(
    u0: Any,
    lattice: HexLattice,
    *,
    kappa: Coefficient,
    dt: float,
    steps: int,
    require_positivity: bool = True,
) -> np.ndarray:
    """Forward Euler for u_t = L_c u, fail-closed on the step size (spec section 4).

    Scalar kappa: dt must respect the positivity bound of the assembled operator
    and the engine's hex_positivity_max_dt (or, with require_positivity=False,
    the engine's sharp L2 bound hex_cfl_max_dt(h, kappa, safety=1)). Conductance
    array: only the assembled positivity bound applies.
    """
    lattice = _check_lattice(lattice)
    u = _check_field(u0, lattice, name="u0").copy()
    dt = _positive_finite(dt, name="dt")
    steps = _int_at_least(steps, name="steps", minimum=0)
    if not isinstance(require_positivity, bool):
        raise TypeError("require_positivity must be bool")
    _, scalar = _coefficient(kappa, lattice)
    diag, off = assemble_hex_operator(lattice, kappa)
    engine = _engine()
    if require_positivity:
        bound = positivity_max_dt(diag, off)
        if scalar:
            bound = min(bound, engine.hex_positivity_max_dt(lattice.h, float(kappa)))
        kind = "positivity"
    elif scalar:
        bound = engine.hex_cfl_max_dt(lattice.h, float(kappa), safety=1.0)
        kind = "L2 stability"
    else:
        raise ValueError("a conductance array requires require_positivity=True (no sharp L2 bound in v1)")
    if dt > bound:
        raise ValueError(f"dt={dt!r} exceeds the {kind} bound {bound!r} of this operator")
    for _ in range(steps):
        u = u + dt * _apply_assembled(diag, off, lattice.neighbors, u)
    return u


# ------------------------------------------------------------------
# Linear solves (spec section 5)
# ------------------------------------------------------------------


def _rescaled_solution(y: np.ndarray, exponent: int) -> np.ndarray:
    """x = y * 2^exponent, fail-closed unless x represents y exactly."""
    x = np.ldexp(y, exponent)
    if not (np.all(np.isfinite(x)) and np.array_equal(np.ldexp(x, -exponent), y)):
        raise ConvergenceError(
            "conjugate_gradient: the solution is not representable in binary64 at the scale of b "
            "(overflow or underflow)"
        )
    return x


# Below this, r . r may have lost the squares of residual components to
# underflow (|r_i| < 2^-511 squares below the normal range); sqrt(r . r) is then
# replaced by the power-of-two-scaled norm. Above it the two agree bit for bit:
# scaling by 2^k commutes exactly with every binary64 operation, and the
# underflowed squares are below 2^-1022 each, negligible against 2^-900.
_CG_SAFE_SQUARED_NORM = 2.0**-900


def _norm2(v: np.ndarray, squared: float) -> float:
    """||v||_2 without underflow (PR #61 Codex P2): sqrt(v . v) where that is exact, else scaled.

    ``squared`` is v . v as already computed by the caller (finite).
    """
    if squared >= _CG_SAFE_SQUARED_NORM:
        return math.sqrt(squared)
    peak = float(np.max(np.abs(v)))
    if peak == 0.0:
        return 0.0
    exponent = math.frexp(peak)[1]
    return math.ldexp(float(np.linalg.norm(np.ldexp(v, -exponent))), exponent)


def _cg_underflow(best: np.ndarray, exponent: int, best_relative: float, iteration: int) -> ConvergenceError:
    return ConvergenceError(
        "conjugate_gradient: the residual is below the binary64 range of the CG recurrence "
        "(r . r underflows to 0 while ||r|| > rtol ||b||); the requested rtol is not reachable",
        best=np.ldexp(best, exponent), best_relative_residual=best_relative, iterations=iteration,
    )


def conjugate_gradient(
    apply_a: Callable[[np.ndarray], np.ndarray],
    b: Any,
    *,
    rtol: float,
    maxiter: int,
    x0: Any = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve A x = b for symmetric positive definite A, ||b - A x|| <= rtol ||b||.

    The stopping rule is decided on the TRUE residual b - A x (spec section 5).
    The recursively updated residual drifts from it in floating point, so it
    only serves as a cheap pre-check: once it passes, b - A x is evaluated; if
    that fails, CG restarts from the true residual (residual replacement).
    The returned ``relative_residual`` is the true ||b - A x|| / ||b||.

    ``x0`` is an optional initial guess (warm start); the stopping rule stays
    relative to ||b||. Fail-closed: non-positive curvature -> ValueError (A is
    not SPD); no convergence within maxiter, a stagnating true residual
    (no new minimum in ``_CG_STAGNATION_CHECKS`` consecutive checks) or a
    non-finite iterate -> ConvergenceError.
    """
    if not callable(apply_a):
        raise TypeError("apply_a must be callable")
    rhs = np.asarray(b, dtype=np.float64)
    if rhs.ndim != 1 or rhs.size == 0 or not np.all(np.isfinite(rhs)):
        raise ValueError("b must be a non-empty finite 1-D array")
    rtol = _positive_finite(rtol, name="rtol")
    if rtol >= 1.0:
        raise ValueError("rtol must be < 1")
    maxiter = _int_at_least(maxiter, name="maxiter", minimum=1)
    peak = float(np.max(np.abs(rhs)))
    if peak == 0.0:
        return np.zeros_like(rhs), {"iterations": 0, "relative_residual": 0.0}
    # Finite entries do not imply a representable ||b||_2 (1e200 overflows,
    # 1e-200 underflows to 0). CG runs on the linear system scaled by a power
    # of two, A (x / 2^e) = b / 2^e with max|b / 2^e| in [0.5, 1): the scaling
    # is exact in binary64 and the relative residual is scale invariant.
    exponent = math.frexp(peak)[1]
    rhs = np.ldexp(rhs, -exponent)
    b_norm = float(np.linalg.norm(rhs))
    if x0 is None:
        x = np.zeros_like(rhs)
        r = rhs.copy()
    else:
        x = np.array(x0, dtype=np.float64)
        if x.shape != rhs.shape or not np.all(np.isfinite(x)):
            raise ValueError("x0 must be a finite array with the shape of b")
        x = np.ldexp(x, -exponent)
        if not np.all(np.isfinite(x)):
            raise ValueError("x0 is not representable at the scale of b")
        r = rhs - np.asarray(apply_a(x), dtype=np.float64)
    # Here r is the true residual of x.
    rr = float(r @ r)
    if not math.isfinite(rr):
        raise ConvergenceError("conjugate_gradient: non-finite residual")
    best, best_relative = x, _norm2(r, rr) / b_norm
    if best_relative <= rtol:
        return _rescaled_solution(x, exponent), {"iterations": 0, "relative_residual": best_relative}
    if rr == 0.0:
        raise _cg_underflow(best, exponent, best_relative, 0)
    p = r.copy()
    stale = 0
    for iteration in range(1, maxiter + 1):
        ap = np.asarray(apply_a(p), dtype=np.float64)
        curvature = float(p @ ap)
        if not curvature > 0.0:
            raise ValueError("conjugate_gradient: non-positive curvature, the operator is not SPD")
        alpha = rr / curvature
        x = x + alpha * p
        r = r - alpha * ap
        rr_new = float(r @ r)
        if not math.isfinite(rr_new):
            raise ConvergenceError("conjugate_gradient: non-finite residual", best=np.ldexp(best, exponent),
                                   best_relative_residual=best_relative, iterations=iteration)
        if _norm2(r, rr_new) > rtol * b_norm:
            if rr_new == 0.0:
                raise _cg_underflow(best, exponent, best_relative, iteration)
            p = r + (rr_new / rr) * p
            rr = rr_new
            continue
        # Pre-check passed: the true residual decides.
        r = rhs - np.asarray(apply_a(x), dtype=np.float64)
        rr = float(r @ r)
        if not math.isfinite(rr):
            raise ConvergenceError("conjugate_gradient: non-finite residual", best=np.ldexp(best, exponent),
                                   best_relative_residual=best_relative, iterations=iteration)
        relative = _norm2(r, rr) / b_norm
        if relative <= rtol:
            return _rescaled_solution(x, exponent), {"iterations": iteration, "relative_residual": relative}
        if relative < best_relative:
            best, best_relative, stale = x, relative, 0
        else:
            stale += 1
            if stale >= _CG_STAGNATION_CHECKS:
                raise ConvergenceError(
                    f"conjugate_gradient: true residual stagnated at {best_relative:.3e} > rtol={rtol!r} "
                    f"(no new minimum in {stale} checks, {iteration} iterations)",
                    best=np.ldexp(best, exponent), best_relative_residual=best_relative, iterations=iteration,
                )
        if rr == 0.0:
            raise _cg_underflow(best, exponent, best_relative, iteration)
        p = r.copy()  # residual replacement: restart from the true residual
    raise ConvergenceError(
        f"conjugate_gradient: no convergence to rtol={rtol!r} in {maxiter} iterations",
        best=np.ldexp(best, exponent), best_relative_residual=best_relative, iterations=maxiter,
    )


def crank_nicolson_heat(
    u0: Any,
    lattice: HexLattice,
    *,
    kappa: Coefficient,
    dt: float,
    steps: int,
    rtol: float = 1e-12,
    maxiter: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Crank-Nicolson for u_t = L_c u (spec section 4):

        (I - dt/2 L_c) u^{n+1} = (I + dt/2 L_c) u^n,

    each step solved with CG warm-started from u^n. The left operator is SPD
    for every dt > 0 on both lattices, so the scheme is unconditionally L2
    stable and second order in time; it is NOT positivity preserving.
    Returns (u, info) with the total CG iterations and the worst relative
    residual over all steps.
    """
    lattice = _check_lattice(lattice)
    u = _check_field(u0, lattice, name="u0").copy()
    dt = _positive_finite(dt, name="dt")
    steps = _int_at_least(steps, name="steps", minimum=0)
    diag, off = assemble_hex_operator(lattice, kappa)
    nb = lattice.neighbors
    half = 0.5 * dt
    limit = 10 * lattice.num_nodes + 100 if maxiter is None else maxiter

    def lhs(v: np.ndarray) -> np.ndarray:
        return v - half * _apply_assembled(diag, off, nb, v)

    iterations = 0
    worst = 0.0
    for _ in range(steps):
        rhs = u + half * _apply_assembled(diag, off, nb, u)
        u, info = conjugate_gradient(lhs, rhs, rtol=rtol, maxiter=limit, x0=u)
        iterations += info["iterations"]
        worst = max(worst, info["relative_residual"])
    return u, {"steps": steps, "cg_iterations": iterations, "max_relative_residual": worst}


def solve_steady_diffusion(
    lattice: HexLattice,
    kappa: Coefficient,
    source: Any,
    *,
    rtol: float = 1e-13,
    maxiter: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve -L_c u = f on the Dirichlet triangle with CG (spec section 5).

    The torus is refused: there -L_c has the constants as null space.
    """
    lattice = _check_lattice(lattice)
    if lattice.domain != "triangle_dirichlet":
        raise ValueError("solve_steady_diffusion supports the Dirichlet triangle only (singular on the torus)")
    f = _check_field(source, lattice, name="source")
    diag, off = assemble_hex_operator(lattice, kappa)
    nb = lattice.neighbors
    limit = 10 * lattice.num_nodes + 100 if maxiter is None else maxiter
    return conjugate_gradient(lambda v: -_apply_assembled(diag, off, nb, v), f, rtol=rtol, maxiter=limit)
