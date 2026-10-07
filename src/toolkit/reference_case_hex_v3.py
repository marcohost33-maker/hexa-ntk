"""
reference_case_hex_v3.py
========================
Reality Anchor Phase 0a (issue #60, wave plan W1-K08): the canonical hex-PDE
reference case ``hex-heat-tri`` and its verification study, expressed as a
ReferenceSpec v1. Contract: spec/REFERENCE_CASE_HEX_HEAT_TRI_v1.md (case,
gate) and spec/REFERENCE_SPEC_v1.md (claim ceilings).

Problem: u_t = kappa * Lap(u) on the equilateral triangle with vertices
(0, 0), (1, 0), (1/2, sqrt(3)/2), homogeneous Dirichlet BC, initial data a sum
of Lame modes.

Why it is an oracle: every Lame mode restricted to the aligned hex lattice
(h = 1/N) is an EXACT eigenvector of the 7-point hex FV Laplacian with
Dirichlet-zero boundary nodes, with the closed-form eigenvalue

    lambda_h = (8 / (3 h^2)) * sum_{a in {0, 60, 120 deg}} sin^2(h <k, a> / 2)
             = lambda - h^2 lambda^2 / 16 + O(h^4).

Continuous and semi-discrete solutions are therefore both known exactly; the
only error is the spatial error in lambda_h.

Code under verification: the engine's ``hex_laplacian_stencil_weight`` (used by
the operator), ``hex_positivity_max_dt`` and ``hex_cfl_max_dt`` (used by the
explicit step gate). The engine is loaded from src/hexantk_v3.py.

Claim boundary: verifies the regular hex Laplacian, its Dirichlet assembly and
second-order convergence for one smooth linear isotropic problem. No physics
validation, no irregular/anisotropic grids, no neural-operator assessment.
"""

from __future__ import annotations

__all__ = [
    "LATTICE_SLOTS",
    "ORDER_BAND",
    "TriangleHexLattice",
    "triangle_hex_lattice",
    "apply_dirichlet_hex_laplacian",
    "assemble_dirichlet_hex_operator",
    "positivity_max_dt",
    "explicit_euler_heat",
    "LameMode",
    "observed_order",
    "celik_three_grid",
    "HexHeatTriConfig",
    "run_hex_heat_tri_study",
    "hex_heat_tri_reference_spec",
]

import functools
import importlib.util
import math
import numbers
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from reference_spec_hex_v3 import (
    ConvergenceAxis,
    ConvergenceStudy,
    EvidenceCheck,
    EvidenceStatus,
    ReferenceKind,
    ReferenceSpec,
    RefinementLevel,
    UncertaintyKind,
    UncertaintyStatement,
)

# ------------------------------------------------------------------
# Contract constants (spec section 5; deliberately not configurable)
# ------------------------------------------------------------------

#: Neighbour slots, counter-clockwise from east, as (di, dj) index offsets of
#: the node x_ij = h * (i + j/2, j * sqrt(3)/2). Slot s and s + 3 are opposite.
LATTICE_SLOTS: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))

#: Gate band for every observed order (decision #60).
ORDER_BAND: Tuple[float, float] = (1.95, 2.05)

_EIGEN_RESIDUAL_TOL = 1e-10
_RAYLEIGH_TOL = 1e-10
_COEFFICIENT_BAND = (0.99, 1.01)
_ASYMPTOTIC_TOL = 0.05
_GCI_SAFETY_FACTOR = 1.25  # Celik et al. 2008, three or more grids

_SQRT3 = math.sqrt(3.0)
_REFERENCE_ID = "hex-heat-tri/v1"
_SCHEMA = "hexantk_reference_case_hex_heat_tri_v1"
_ENGINE_PATH = Path(__file__).resolve().parents[1] / "hexantk_v3.py"


@functools.lru_cache(maxsize=1)
def _engine() -> Any:
    """Load the canonical engine src/hexantk_v3.py (the code under test)."""
    spec = importlib.util.spec_from_file_location("hexantk_v3_reference_case", _ENGINE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise ImportError(f"cannot load engine from {_ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module via sys.modules during class creation
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _positive_int(value: Any, *, name: str, minimum: int) -> int:
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
# Lattice and operator
# ------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TriangleHexLattice:
    """Interior nodes of the aligned hex lattice in the unit equilateral triangle.

    ``neighbors[i, s] == -1`` means the slot-``s`` neighbour is a homogeneous
    Dirichlet boundary node (value 0), NOT a missing neighbour. Arrays are
    read-only.
    """

    n: int
    h: float
    coords: np.ndarray
    neighbors: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.coords.shape[0])


def triangle_hex_lattice(n: int) -> TriangleHexLattice:
    """Build the lattice with spacing h = 1/n; unknowns are i, j >= 1, i + j <= n - 1."""
    n = _positive_int(n, name="n", minimum=3)
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
    coords.flags.writeable = False
    neighbors.flags.writeable = False
    return TriangleHexLattice(n=n, h=h, coords=coords, neighbors=neighbors)


def _check_field(u: Any, lattice: TriangleHexLattice, *, name: str = "u") -> np.ndarray:
    if not isinstance(lattice, TriangleHexLattice):
        raise TypeError("lattice must be a TriangleHexLattice")
    arr = np.asarray(u, dtype=np.float64)
    if arr.shape != (lattice.num_nodes,):
        raise ValueError(f"{name} must have shape ({lattice.num_nodes},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def apply_dirichlet_hex_laplacian(u: Any, lattice: TriangleHexLattice) -> np.ndarray:
    """(L_h u)_i = w * sum_s (u_{n(i,s)} - u_i) with u_{-1} = 0 (Dirichlet zero).

    ``w`` is the engine's ``hex_laplacian_stencil_weight(h)``.
    """
    arr = _check_field(u, lattice)
    weight = float(_engine().hex_laplacian_stencil_weight(lattice.h))
    nb = lattice.neighbors
    gathered = np.where(nb >= 0, arr[np.clip(nb, 0, None)], 0.0)
    return weight * (gathered.sum(axis=1) - 6.0 * arr)


def assemble_dirichlet_hex_operator(
    lattice: TriangleHexLattice,
    kappa: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble kappa * L_h as (diag[M], off[M, 6]).

    (kappa L_h u)_i = diag_i * u_i + sum_s off[i, s] * u_{n(i,s)}; a slot that
    points to a Dirichlet node carries off = 0 (its value is 0) but still
    contributes to the diagonal.
    """
    if not isinstance(lattice, TriangleHexLattice):
        raise TypeError("lattice must be a TriangleHexLattice")
    kappa = _positive_finite(kappa, name="kappa")
    coupling = kappa * float(_engine().hex_laplacian_stencil_weight(lattice.h))
    off = np.where(lattice.neighbors >= 0, coupling, 0.0)
    diag = np.full(lattice.num_nodes, -6.0 * coupling)
    return diag, off


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
    lattice: TriangleHexLattice,
    *,
    kappa: float,
    dt: float,
    steps: int,
    require_positivity: bool = True,
) -> np.ndarray:
    """Forward Euler for u_t = kappa * L_h u, fail-closed on the step size.

    With ``require_positivity`` dt must not exceed the positivity bound of the
    assembled operator (checked against the engine's hex_positivity_max_dt);
    otherwise it must not exceed the engine's sharp L2 bound
    hex_cfl_max_dt(h, kappa, safety=1).
    """
    u = _check_field(u0, lattice, name="u0").copy()
    kappa = _positive_finite(kappa, name="kappa")
    dt = _positive_finite(dt, name="dt")
    steps = _positive_int(steps, name="steps", minimum=0)
    if not isinstance(require_positivity, bool):
        raise TypeError("require_positivity must be bool")
    diag, off = assemble_dirichlet_hex_operator(lattice, kappa)
    if require_positivity:
        bound = min(positivity_max_dt(diag, off), _engine().hex_positivity_max_dt(lattice.h, kappa))
        kind = "positivity"
    else:
        bound = _engine().hex_cfl_max_dt(lattice.h, kappa, safety=1.0)
        kind = "L2 stability"
    if dt > bound:
        raise ValueError(f"dt={dt!r} exceeds the {kind} bound {bound!r} of this operator")
    nb = lattice.neighbors
    safe = np.clip(nb, 0, None)
    for _ in range(steps):
        u = u + dt * (diag * u + (off * u[safe]).sum(axis=1))
    return u


# ------------------------------------------------------------------
# Lame modes (exact eigenfunctions, spec section 4)
# ------------------------------------------------------------------


def _rotation(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def _reflection(angle: float) -> np.ndarray:
    c, s = math.cos(2.0 * angle), math.sin(2.0 * angle)
    return np.array([[c, s], [s, -c]])


# Dihedral group of order 6 generated by the reflections across the lines at
# 0 and 60 degrees through the origin, with det(g).
_GROUP: Tuple[Tuple[np.ndarray, int], ...] = (
    (_rotation(0.0), 1),
    (_rotation(2.0 * math.pi / 3.0), 1),
    (_rotation(4.0 * math.pi / 3.0), 1),
    (_reflection(0.0), -1),
    (_reflection(math.pi / 3.0), -1),
    (_reflection(2.0 * math.pi / 3.0), -1),
)
_LATTICE_DIRECTIONS = tuple(
    np.array([math.cos(a), math.sin(a)]) for a in (0.0, math.pi / 3.0, 2.0 * math.pi / 3.0)
)


@dataclass(frozen=True)
class LameMode:
    """Dirichlet eigenfunction (m, n, parity) of the unit equilateral triangle.

    lambda = (16 pi^2 / 9) * (m^2 + m n + n^2); parity "cos" with m == n
    vanishes identically and is rejected.
    """

    m: int
    n: int
    parity: str = "sin"

    def __post_init__(self) -> None:
        _positive_int(self.m, name="m", minimum=1)
        _positive_int(self.n, name="n", minimum=1)
        if self.parity not in ("sin", "cos"):
            raise ValueError(f"parity must be 'sin' or 'cos', got {self.parity!r}")
        if self.parity == "cos" and self.m == self.n:
            raise ValueError("the cos mode with m == n vanishes identically")

    @property
    def wavevector(self) -> np.ndarray:
        p, q = self.m + self.n, self.n
        return 2.0 * math.pi * np.array([2.0 * p / 3.0 - q / 3.0, q / _SQRT3])

    @property
    def eigenvalue(self) -> float:
        return 16.0 * math.pi**2 / 9.0 * (self.m**2 + self.m * self.n + self.n**2)

    def discrete_eigenvalue(self, h: float) -> float:
        h = _positive_finite(h, name="h")
        k = self.wavevector
        total = sum(math.sin(h * float(k @ a) / 2.0) ** 2 for a in _LATTICE_DIRECTIONS)
        return 8.0 / (3.0 * h * h) * total

    def evaluate(self, coords: Any) -> np.ndarray:
        xy = np.asarray(coords, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("coords must have shape (M, 2)")
        func = np.sin if self.parity == "sin" else np.cos
        k = self.wavevector
        return sum(sign * func(xy @ (g @ k)) for g, sign in _GROUP)


# ------------------------------------------------------------------
# Convergence analysis (Celik et al. 2008, J. Fluids Eng. 130(7) 078001)
# ------------------------------------------------------------------


def observed_order(error_coarse: float, error_fine: float, ratio: float = 2.0) -> float:
    """p = ln(e_coarse / e_fine) / ln(r) against a known exact solution."""
    e_c = _positive_finite(error_coarse, name="error_coarse")
    e_f = _positive_finite(error_fine, name="error_fine")
    r = _positive_finite(ratio, name="ratio")
    if r <= 1.0:
        raise ValueError("ratio must be > 1")
    return math.log(e_c / e_f) / math.log(r)


def celik_three_grid(f1: float, f2: float, f3: float, ratio: float = 2.0) -> Dict[str, Any]:
    """Three-grid observed order and GCI for a constant refinement ratio.

    f1 is the FINEST solution. eps21 = f2 - f1, eps32 = f3 - f2,
    p = |ln|eps32 / eps21|| / ln r, e_a21 = |(f1 - f2) / f1|,
    GCI_fine = 1.25 * e_a21 / (r^p - 1). A negative eps32/eps21 flags
    oscillatory convergence. Degenerate inputs raise ValueError.
    """
    values = [float(v) for v in (f1, f2, f3)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("f1, f2, f3 must be finite")
    f1, f2, f3 = values
    r = _positive_finite(ratio, name="ratio")
    if r <= 1.0:
        raise ValueError("ratio must be > 1")
    eps21, eps32 = f2 - f1, f3 - f2
    if eps21 == 0.0 or eps32 == 0.0 or f1 == 0.0:
        raise ValueError("degenerate refinement data: zero difference or zero finest value")
    quotient = eps32 / eps21
    p = abs(math.log(abs(quotient))) / math.log(r)
    if p == 0.0:
        raise ValueError("observed order is zero; GCI undefined")
    r_p = r**p
    e_a21 = abs((f1 - f2) / f1)
    return {
        "observed_order": p,
        "oscillatory": quotient < 0.0,
        "eps21": eps21,
        "eps32": eps32,
        "relative_change_fine": e_a21,
        "gci_fine": _GCI_SAFETY_FACTOR * e_a21 / (r_p - 1.0),
        "extrapolated": (r_p * f1 - f2) / (r_p - 1.0),
    }


# ------------------------------------------------------------------
# Study (spec sections 5 and 6)
# ------------------------------------------------------------------


def _default_modes() -> Tuple[Tuple[LameMode, float], ...]:
    return ((LameMode(1, 1, "sin"), 1.0), (LameMode(1, 2, "sin"), 0.5), (LameMode(1, 2, "cos"), 0.25))


_QOI_MODE = LameMode(1, 1, "sin")


@dataclass(frozen=True)
class HexHeatTriConfig:
    """Problem data and levels. Gate thresholds are contract constants."""

    levels: Tuple[int, ...] = (16, 32, 64, 128)
    kappa: float = 1.0
    t_final: float = 0.01
    modes: Tuple[Tuple[LameMode, float], ...] = field(default_factory=_default_modes)

    def __post_init__(self) -> None:
        if not isinstance(self.levels, tuple) or len(self.levels) < 4:
            raise ValueError("levels must be a tuple of at least four N values")
        levels = [_positive_int(n, name="level", minimum=3) for n in self.levels]
        ratios = {levels[i + 1] / levels[i] for i in range(len(levels) - 1)}
        if len(ratios) != 1 or next(iter(ratios)) <= 1.0:
            raise ValueError("levels must increase with one constant refinement ratio")
        _positive_finite(self.kappa, name="kappa")
        _positive_finite(self.t_final, name="t_final")
        if not isinstance(self.modes, tuple) or not self.modes:
            raise ValueError("modes must be a non-empty tuple of (LameMode, amplitude)")
        for entry in self.modes:
            if not (isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], LameMode)):
                raise TypeError("each mode must be a (LameMode, amplitude) pair")
            amplitude = entry[1]
            if isinstance(amplitude, bool) or not isinstance(amplitude, numbers.Real):
                raise TypeError("mode amplitude must be a real number")
            if not math.isfinite(float(amplitude)) or float(amplitude) == 0.0:
                raise ValueError("mode amplitude must be finite and non-zero")

    @property
    def refinement_ratio(self) -> float:
        return self.levels[1] / self.levels[0]


def _rayleigh_quotient(phi: np.ndarray, lattice: TriangleHexLattice) -> float:
    return float(-(phi @ apply_dirichlet_hex_laplacian(phi, lattice)) / (phi @ phi))


def _mode_evidence(mode: LameMode, lattice: TriangleHexLattice) -> Dict[str, Any]:
    """Exactness evidence of one mode on one lattice (spec section 5, C1-C3)."""
    phi = mode.evaluate(lattice.coords)
    scale = float(np.max(np.abs(phi)))
    if not scale > 1e-12:
        raise ValueError(f"mode {mode} vanishes on the N={lattice.n} lattice")
    h = lattice.h
    lam_h = mode.discrete_eigenvalue(h)
    lam = mode.eigenvalue
    residual = apply_dirichlet_hex_laplacian(phi, lattice) + lam_h * phi
    rayleigh = _rayleigh_quotient(phi, lattice)
    return {
        "phi": phi,
        "eigen_residual": float(np.max(np.abs(residual))) / (lam_h * scale),
        "lambda_rq": rayleigh,
        "lambda_h_closed_form": lam_h,
        "lambda_exact": lam,
        "coefficient_ratio": (lam - rayleigh) / (h * h * lam * lam / 16.0),
    }


def _level_evidence(n: int, config: HexHeatTriConfig) -> Dict[str, Any]:
    lattice = triangle_hex_lattice(n)
    qoi = _mode_evidence(_QOI_MODE, lattice)
    residuals: List[float] = [qoi["eigen_residual"]]
    coefficients: Dict[str, float] = {"1,1,sin": qoi["coefficient_ratio"]}
    exact = np.zeros(lattice.num_nodes)
    semi_discrete = np.zeros(lattice.num_nodes)
    for mode, amplitude in config.modes:
        ev = qoi if mode == _QOI_MODE else _mode_evidence(mode, lattice)
        residuals.append(ev["eigen_residual"])
        coefficients[f"{mode.m},{mode.n},{mode.parity}"] = ev["coefficient_ratio"]
        exact += amplitude * math.exp(-config.kappa * ev["lambda_exact"] * config.t_final) * ev["phi"]
        semi_discrete += amplitude * math.exp(-config.kappa * ev["lambda_rq"] * config.t_final) * ev["phi"]
    error = semi_discrete - exact
    return {
        "n": n,
        "h": lattice.h,
        "num_nodes": lattice.num_nodes,
        "eigen_residual_max": max(residuals),
        "lambda_rq": qoi["lambda_rq"],
        "lambda_h_closed_form": qoi["lambda_h_closed_form"],
        "lambda_exact": qoi["lambda_exact"],
        "coefficient_ratio": coefficients,
        "heat_rel_l2_error": float(np.linalg.norm(error) / np.linalg.norm(exact)),
        "heat_max_error": float(np.max(np.abs(error))),
    }


def _within(value: float, band: Tuple[float, float]) -> bool:
    return math.isfinite(value) and band[0] <= value <= band[1]


def run_hex_heat_tri_study(config: HexHeatTriConfig | None = None) -> Dict[str, Any]:
    """Run the hex-heat-tri verification study; returns strict-JSON evidence.

    ``passed`` is True only if every gate criterion of the spec holds.
    """
    config = HexHeatTriConfig() if config is None else config
    if not isinstance(config, HexHeatTriConfig):
        raise TypeError("config must be a HexHeatTriConfig")
    r = config.refinement_ratio
    levels = [_level_evidence(n, config) for n in config.levels]
    heat_orders = [
        observed_order(a["heat_rel_l2_error"], b["heat_rel_l2_error"], r) for a, b in zip(levels, levels[1:])
    ]
    qoi = [level["lambda_rq"] for level in levels]
    celik_fine = celik_three_grid(qoi[-1], qoi[-2], qoi[-3], r)
    celik_coarse = celik_three_grid(qoi[-2], qoi[-3], qoi[-4], r)
    finest = levels[-1]
    exact_rel_error_fine = abs(finest["lambda_exact"] - finest["lambda_rq"]) / finest["lambda_exact"]
    criteria = {
        "C1_eigen_residual": all(level["eigen_residual_max"] <= _EIGEN_RESIDUAL_TOL for level in levels),
        "C2_rayleigh_matches_closed_form": all(
            abs(level["lambda_rq"] - level["lambda_h_closed_form"]) <= _RAYLEIGH_TOL * level["lambda_h_closed_form"]
            for level in levels
        ),
        "C3_isotropic_coefficient": all(_within(c, _COEFFICIENT_BAND) for c in finest["coefficient_ratio"].values()),
        "C4_heat_error_order": all(_within(p, ORDER_BAND) for p in heat_orders[-2:]),
        "C5_observed_order": _within(celik_fine["observed_order"], ORDER_BAND),
        "C6_monotone": not celik_fine["oscillatory"],
        "C7_asymptotic_range": abs(celik_fine["observed_order"] - celik_coarse["observed_order"]) <= _ASYMPTOTIC_TOL,
        "C8_gci_reported": math.isfinite(celik_fine["gci_fine"]) and celik_fine["gci_fine"] >= 0.0,
        "C9_gci_covers_exact_error": celik_fine["gci_fine"] >= exact_rel_error_fine,
    }
    code_ok = all(criteria[k] for k in ("C1_eigen_residual", "C2_rayleigh_matches_closed_form",
                                        "C3_isotropic_coefficient", "C4_heat_error_order"))
    solution_ok = all(criteria[k] for k in ("C5_observed_order", "C6_monotone", "C7_asymptotic_range",
                                            "C8_gci_reported", "C9_gci_covers_exact_error"))
    return {
        "schema": _SCHEMA,
        "reference_id": _REFERENCE_ID,
        "spec": "spec/REFERENCE_CASE_HEX_HEAT_TRI_v1.md",
        "code_under_test": "src/hexantk_v3.py:hex_laplacian_stencil_weight",
        "config": {
            "levels": list(config.levels),
            "refinement_ratio": r,
            "kappa": config.kappa,
            "t_final": config.t_final,
            "modes": [[m.m, m.n, m.parity, float(a)] for m, a in config.modes],
            "qoi_mode": [_QOI_MODE.m, _QOI_MODE.n, _QOI_MODE.parity],
        },
        "thresholds": {
            "order_band": list(ORDER_BAND),
            "eigen_residual_tol": _EIGEN_RESIDUAL_TOL,
            "rayleigh_tol": _RAYLEIGH_TOL,
            "coefficient_band": list(_COEFFICIENT_BAND),
            "asymptotic_tol": _ASYMPTOTIC_TOL,
            "gci_safety_factor": _GCI_SAFETY_FACTOR,
        },
        "levels": levels,
        "heat_error_orders": heat_orders,
        "celik_fine": celik_fine,
        "celik_coarse": celik_coarse,
        "qoi_exact_rel_error_fine": exact_rel_error_fine,
        "criteria": criteria,
        "code_verification_pass": code_ok,
        "solution_verification_pass": solution_ok,
        "passed": code_ok and solution_ok,
    }


def _failed(criteria: Dict[str, bool], keys: Tuple[str, ...]) -> str:
    return "failed gate criteria: " + ", ".join(k for k in keys if not criteria[k])


def hex_heat_tri_reference_spec(study: Dict[str, Any], *, evidence_ref: str) -> ReferenceSpec:
    """Express a study result as ReferenceSpec v1 (spec section 6).

    The claim ceiling is derived by ReferenceSpec; a failed criterion keeps it
    at comparison_only.
    """
    if not isinstance(study, dict) or study.get("schema") != _SCHEMA:
        raise ValueError(f"study must be a {_SCHEMA} result of run_hex_heat_tri_study")
    criteria = study["criteria"]
    code_keys = ("C1_eigen_residual", "C2_rayleigh_matches_closed_form", "C3_isotropic_coefficient",
                 "C4_heat_error_order")
    solution_keys = ("C5_observed_order", "C6_monotone", "C7_asymptotic_range", "C8_gci_reported",
                     "C9_gci_covers_exact_error")
    code_check = (
        EvidenceCheck(EvidenceStatus.PASS, evidence_ref=evidence_ref)
        if all(criteria[k] for k in code_keys)
        else EvidenceCheck(EvidenceStatus.FAIL, rationale=_failed(criteria, code_keys))
    )
    spatial_ok = all(criteria[k] for k in solution_keys)
    spatial_check = (
        EvidenceCheck(EvidenceStatus.PASS, evidence_ref=evidence_ref)
        if spatial_ok
        else EvidenceCheck(EvidenceStatus.FAIL, rationale=_failed(criteria, solution_keys))
    )
    qoi_name = "decay rate lambda_RQ of the fundamental Dirichlet Lame mode (1,1,sin)"
    celik = study["celik_fine"]
    order = celik["observed_order"]
    gci = celik["gci_fine"]
    levels = tuple(
        RefinementLevel(f"N={level['n']}", level["h"], level["lambda_rq"]) for level in reversed(study["levels"])
    )
    spatial_study = ConvergenceStudy(
        axis=ConvergenceAxis.SPATIAL,
        quantity_of_interest=qoi_name,
        levels=levels,
        evidence_ref=evidence_ref,
        observed_order=order if math.isfinite(order) and order > 0.0 else None,
        fine_grid_relative_error_estimate=gci if math.isfinite(gci) and gci >= 0.0 else None,
        asymptotic_range_checked=bool(criteria["C7_asymptotic_range"]),
    )
    n_values = ", ".join(str(level["n"]) for level in study["levels"])
    context = (
        "u_t = kappa*Lap(u) on the unit equilateral triangle, homogeneous Dirichlet BC, regular 7-point "
        f"hex FV Laplacian (engine weight 2/(3h^2)), h = 1/N for N = {n_values}, smooth Lame-mode data"
    )
    return ReferenceSpec(
        reference_id=study["reference_id"],
        kind=ReferenceKind.NUMERICAL,
        context_of_use=context,
        quantity_of_interest=qoi_name,
        code_verification=code_check,
        spatial_verification=spatial_check,
        temporal_verification=EvidenceCheck(
            EvidenceStatus.NOT_APPLICABLE,
            rationale="integrated exactly in time along verified discrete eigenvectors; no time-step error",
        ),
        iterative_verification=EvidenceCheck(
            EvidenceStatus.NOT_APPLICABLE,
            rationale="explicit operator application and closed-form eigenvectors; no iterative solve",
        ),
        spatial_study=spatial_study,
        numerical_uncertainty=(
            UncertaintyStatement(UncertaintyKind.RELATIVE_FRACTION, gci, evidence_ref)
            if math.isfinite(gci) and gci >= 0.0
            else None
        ),
    )
