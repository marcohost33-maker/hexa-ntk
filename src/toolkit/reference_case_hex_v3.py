"""
reference_case_hex_v3.py
========================
Reality Anchor Phase 0 (issue #60, wave plan W1-K08): the canonical hex-PDE
reference cases and their verification studies, expressed as ReferenceSpec v1.

- ``hex-heat-tri`` (Phase 0a, spec/REFERENCE_CASE_HEX_HEAT_TRI_v1.md):
  u_t = kappa * Lap(u) on the unit equilateral triangle, homogeneous Dirichlet
  BC, Lame-mode data.
- ``hex-heat-torus`` (Phase 0b, spec/REFERENCE_CASES_HEX_PHASE0B_v1.md): the
  same equation on the periodic hex torus, plane-wave data, conservation.
- ``hex-mms-tri`` (Phase 0b): -div(kappa grad u) = f on the triangle with a
  manufactured solution, variable kappa, source term and a CG solve.

Why the heat cases are oracles: every Lame mode (triangle) and every
lattice-compatible plane wave (torus) is an EXACT eigenvector of the 7-point
hex FV Laplacian, with the closed-form eigenvalue

    lambda_h = (8 / (3 h^2)) * sum_{a in {0, 60, 120 deg}} sin^2(h <k, a> / 2)
             = lambda - h^2 lambda^2 / 16 + O(h^4).

Code under verification: the mechanistic layer hex_fv_solver_hex_v3 (lattices,
operator, conductances, positivity gate, CG) and through it the engine's
``hex_laplacian_stencil_weight``.

Claim boundary: regular hex lattice, smooth data, scalar isotropic kappa. No
physics validation, no irregular/anisotropic grids, no neural-operator
assessment; every derived ceiling holds only for its stated context of use.
"""

from __future__ import annotations

__all__ = [
    "ORDER_BAND",
    "LameMode",
    "TorusMode",
    "observed_order",
    "celik_three_grid",
    "HexHeatTriConfig",
    "run_hex_heat_tri_study",
    "hex_heat_tri_reference_spec",
    "HexHeatTorusConfig",
    "run_hex_heat_torus_study",
    "hex_heat_torus_reference_spec",
    "HexMmsTriConfig",
    "run_hex_mms_tri_study",
    "hex_mms_tri_reference_specs",
]

import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from hex_fv_solver_hex_v3 import (
    CG_STOPPING_RULES,
    HEX_RESIDUAL_FLOOR_FACTOR,
    ConvergenceError,
    HexLattice,
    apply_hex_laplacian,
    assemble_hex_operator,
    edge_conductances,
    positivity_max_dt,
    solve_steady_diffusion,
    torus_hex_lattice,
    triangle_hex_lattice,
)
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
# Contract constants (specs, section "gate"; deliberately not configurable)
# ------------------------------------------------------------------

#: Gate band for every observed order (decision #60).
ORDER_BAND: Tuple[float, float] = (1.95, 2.05)

_EIGEN_RESIDUAL_TOL = 1e-10
_RAYLEIGH_TOL = 1e-10
_COEFFICIENT_BAND = (0.99, 1.01)
_ASYMPTOTIC_TOL = 0.05
_GCI_SAFETY_FACTOR = 1.25  # Celik et al. 2008, three or more grids
_MIN_REFINEMENT_RATIO = 1.3  # Celik et al. 2008: r should exceed 1.3
_CONSERVATION_TOL = 1e-12
_MMS_BOUNDARY_TOL = 1e-12
_MMS_SOURCE_TOL = 1e-8
_MMS_FD_STEP = 1e-4
_MMS_FD_POINTS = 64
_MMS_CG_RTOL = 1e-13
_MMS_CG_RTOL_LOOSE = 1e-10
_MMS_ITERATIVE_RATIO = 1e-3

_SQRT3 = math.sqrt(3.0)
_REFERENCE_ID = "hex-heat-tri/v1"
_SCHEMA = "hexantk_reference_case_hex_heat_tri_v1"
_TORUS_REFERENCE_ID = "hex-heat-torus/v1"
_TORUS_SCHEMA = "hexantk_reference_case_hex_heat_torus_v1"
_MMS_REFERENCE_ID = "hex-mms-tri/v1"
_MMS_ORACLE_ID = "hex-mms-tri/oracle"
_MMS_SCHEMA = "hexantk_reference_case_hex_mms_tri_v1"

_HEAT_CODE_KEYS = ("C1_eigen_residual", "C2_rayleigh_matches_closed_form", "C3_isotropic_coefficient",
                   "C4_heat_error_order")
_SOLUTION_KEYS = ("C5_observed_order", "C6_monotone", "C7_asymptotic_range", "C8_gci_reported",
                  "C9_gci_covers_exact_error")
_TORUS_CODE_KEYS = _HEAT_CODE_KEYS + ("C10_conservation",)
_MMS_ORACLE_KEYS = ("O1_boundary_zero", "O2_source_matches_flux_divergence", "O3_kappa_positive")
_MMS_CODE_KEYS = _MMS_ORACLE_KEYS + ("M1_error_order", "M2_conductance_symmetric", "M3_positivity_gate")
_MMS_ITERATIVE_KEYS = ("I1_iterative_error_negligible",)


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


def _check_coords(coords: Any) -> np.ndarray:
    xy = np.asarray(coords, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coords must have shape (M, 2)")
    return xy


# ------------------------------------------------------------------
# Exact modes (Phase 0a spec section 4; Phase 0b spec section 2)
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


def _discrete_eigenvalue(k: np.ndarray, h: Any) -> float:
    h = _positive_finite(h, name="h")
    total = sum(math.sin(h * float(k @ a) / 2.0) ** 2 for a in _LATTICE_DIRECTIONS)
    return 8.0 / (3.0 * h * h) * total


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
        _int_at_least(self.m, name="m", minimum=1)
        _int_at_least(self.n, name="n", minimum=1)
        if self.parity not in ("sin", "cos"):
            raise ValueError(f"parity must be 'sin' or 'cos', got {self.parity!r}")
        if self.parity == "cos" and self.m == self.n:
            raise ValueError("the cos mode with m == n vanishes identically")

    @property
    def key(self) -> str:
        return f"{self.m},{self.n},{self.parity}"

    @property
    def wavevector(self) -> np.ndarray:
        p, q = self.m + self.n, self.n
        return 2.0 * math.pi * np.array([2.0 * p / 3.0 - q / 3.0, q / _SQRT3])

    @property
    def eigenvalue(self) -> float:
        return 16.0 * math.pi**2 / 9.0 * (self.m**2 + self.m * self.n + self.n**2)

    def discrete_eigenvalue(self, h: float) -> float:
        return _discrete_eigenvalue(self.wavevector, h)

    def evaluate(self, coords: Any) -> np.ndarray:
        xy = _check_coords(coords)
        func = np.sin if self.parity == "sin" else np.cos
        k = self.wavevector
        return sum(sign * func(xy @ (g @ k)) for g, sign in _GROUP)

    def gradient(self, coords: Any) -> np.ndarray:
        """Closed-form gradient, shape (M, 2)."""
        xy = _check_coords(coords)
        k = self.wavevector
        grad = np.zeros_like(xy)
        for g, sign in _GROUP:
            gk = g @ k
            arg = xy @ gk
            slope = np.cos(arg) if self.parity == "sin" else -np.sin(arg)
            grad += sign * slope[:, None] * gk
        return grad


@dataclass(frozen=True)
class TorusMode:
    """Plane wave cos/sin(<k, x>) on the hex torus, k = 2 pi (p, 2q / sqrt(3)).

    lambda = |k|^2 = 4 pi^2 (p^2 + 4 q^2 / 3). (0, 0, "cos") is the constant
    mode (lambda = 0); (0, 0, "sin") vanishes identically and is rejected.
    """

    p: int
    q: int
    parity: str = "cos"

    def __post_init__(self) -> None:
        for name, value in (("p", self.p), ("q", self.q)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be an int, got {value!r}")
        if self.parity not in ("sin", "cos"):
            raise ValueError(f"parity must be 'sin' or 'cos', got {self.parity!r}")
        if self.parity == "sin" and self.p == 0 and self.q == 0:
            raise ValueError("the sin mode with p == q == 0 vanishes identically")

    @property
    def key(self) -> str:
        return f"{self.p},{self.q},{self.parity}"

    @property
    def wavevector(self) -> np.ndarray:
        return 2.0 * math.pi * np.array([float(self.p), 2.0 * self.q / _SQRT3])

    @property
    def eigenvalue(self) -> float:
        return 4.0 * math.pi**2 * (self.p**2 + 4.0 * self.q**2 / 3.0)

    def discrete_eigenvalue(self, h: float) -> float:
        return _discrete_eigenvalue(self.wavevector, h)

    def evaluate(self, coords: Any) -> np.ndarray:
        xy = _check_coords(coords)
        arg = xy @ self.wavevector
        return np.sin(arg) if self.parity == "sin" else np.cos(arg)


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


def _within(value: float, band: Tuple[float, float]) -> bool:
    return math.isfinite(value) and band[0] <= value <= band[1]


def _validate_levels(levels: Any, *, minimum: int, multiple_of: int = 1) -> Tuple[int, ...]:
    if not isinstance(levels, tuple) or len(levels) < 4:
        raise ValueError("levels must be a tuple of at least four N values")
    values = [_int_at_least(n, name="level", minimum=minimum) for n in levels]
    if any(n % multiple_of for n in values):
        raise ValueError(f"every level must be a multiple of {multiple_of}")
    ratios = {values[i + 1] / values[i] for i in range(len(values) - 1)}
    if len(ratios) != 1 or next(iter(ratios)) <= _MIN_REFINEMENT_RATIO:
        raise ValueError(
            f"levels must increase with one constant refinement ratio > {_MIN_REFINEMENT_RATIO} "
            "(Celik et al. 2008)"
        )
    return tuple(values)


def _validate_modes(modes: Any, mode_type: type) -> None:
    if not isinstance(modes, tuple) or not modes:
        raise ValueError(f"modes must be a non-empty tuple of ({mode_type.__name__}, amplitude)")
    for entry in modes:
        if not (isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], mode_type)):
            raise TypeError(f"each mode must be a ({mode_type.__name__}, amplitude) pair")
        amplitude = entry[1]
        if isinstance(amplitude, bool) or not isinstance(amplitude, numbers.Real):
            raise TypeError("mode amplitude must be a real number")
        if not math.isfinite(float(amplitude)) or float(amplitude) == 0.0:
            raise ValueError("mode amplitude must be finite and non-zero")


def _failed(criteria: Dict[str, bool], keys: Tuple[str, ...]) -> str:
    return "failed gate criteria: " + ", ".join(k for k in keys if not criteria[k])


def _check_from(criteria: Dict[str, bool], keys: Tuple[str, ...], evidence_ref: str) -> EvidenceCheck:
    if all(criteria[k] for k in keys):
        return EvidenceCheck(EvidenceStatus.PASS, evidence_ref=evidence_ref)
    return EvidenceCheck(EvidenceStatus.FAIL, rationale=_failed(criteria, keys))


def _spatial_study(study: Dict[str, Any], qoi_name: str, qoi_key: str, evidence_ref: str) -> ConvergenceStudy:
    celik = study["celik_fine"]
    order, gci = celik["observed_order"], celik["gci_fine"]
    # An oscillatory finest triplet (C6 fails) is negative evidence: celik_three_grid
    # still returns a finite p, which ConvergenceStudy would reject as non-monotone.
    # Omit the order, keep the raw levels and the failed criterion (PR #61 Codex P2).
    if celik["oscillatory"]:
        order = math.nan
    levels = tuple(
        RefinementLevel(f"N={level['n']}", level["h"], level[qoi_key]) for level in reversed(study["levels"])
    )
    return ConvergenceStudy(
        axis=ConvergenceAxis.SPATIAL,
        quantity_of_interest=qoi_name,
        levels=levels,
        evidence_ref=evidence_ref,
        observed_order=order if math.isfinite(order) and order > 0.0 else None,
        fine_grid_relative_error_estimate=gci if math.isfinite(gci) and gci >= 0.0 else None,
        asymptotic_range_checked=bool(study["criteria"]["C7_asymptotic_range"]),
    )


def _uncertainty(study: Dict[str, Any], evidence_ref: str) -> Optional[UncertaintyStatement]:
    gci = study["celik_fine"]["gci_fine"]
    if math.isfinite(gci) and gci >= 0.0:
        return UncertaintyStatement(UncertaintyKind.RELATIVE_FRACTION, gci, evidence_ref)
    return None


def _solution_criteria(qoi: List[float], exact: float, r: float) -> Tuple[Dict[str, Any], Dict[str, Any],
                                                                           Dict[str, bool], float]:
    """Celik on the three finest levels, the coarse triplet for the asymptotic check, C5-C9."""
    celik_fine = celik_three_grid(qoi[-1], qoi[-2], qoi[-3], r)
    celik_coarse = celik_three_grid(qoi[-2], qoi[-3], qoi[-4], r)
    exact_rel_error_fine = abs(exact - qoi[-1]) / abs(exact)
    criteria = {
        "C5_observed_order": _within(celik_fine["observed_order"], ORDER_BAND),
        "C6_monotone": not celik_fine["oscillatory"],
        "C7_asymptotic_range": abs(celik_fine["observed_order"] - celik_coarse["observed_order"]) <= _ASYMPTOTIC_TOL,
        "C8_gci_reported": math.isfinite(celik_fine["gci_fine"]) and celik_fine["gci_fine"] >= 0.0,
        "C9_gci_covers_exact_error": celik_fine["gci_fine"] >= exact_rel_error_fine,
    }
    return celik_fine, celik_coarse, criteria, exact_rel_error_fine


# ------------------------------------------------------------------
# Eigen-heat studies: hex-heat-tri (0a) and hex-heat-torus (0b)
# ------------------------------------------------------------------


def _rayleigh_quotient(phi: np.ndarray, lattice: HexLattice) -> float:
    return float(-(phi @ apply_hex_laplacian(phi, lattice)) / (phi @ phi))


def _mode_evidence(mode: Any, lattice: HexLattice) -> Dict[str, Any]:
    """Exactness evidence of one mode on one lattice (C1-C3)."""
    phi = mode.evaluate(lattice.coords)
    scale = float(np.max(np.abs(phi)))
    if not scale > 1e-12:
        raise ValueError(f"mode {mode} vanishes on the N={lattice.n} lattice")
    h = lattice.h
    lam_h = mode.discrete_eigenvalue(h)
    lam = mode.eigenvalue
    residual = apply_hex_laplacian(phi, lattice) + lam_h * phi
    rayleigh = _rayleigh_quotient(phi, lattice)
    # The constant torus mode (lambda = 0) is normalised by the stencil weight.
    denominator = lam_h if lam_h > 0.0 else 2.0 / (3.0 * h * h)
    return {
        "phi": phi,
        "eigen_residual": float(np.max(np.abs(residual))) / (denominator * scale),
        "lambda_rq": rayleigh,
        "lambda_h_closed_form": lam_h,
        "lambda_exact": lam,
        "coefficient_ratio": (lam - rayleigh) / (h * h * lam * lam / 16.0) if lam > 0.0 else None,
    }


def _conservation_residual(lattice: HexLattice) -> float:
    """|sum_i (L_h v)_i| / (w * sum_i |v_i|) for a seeded field (Phase 0b C10)."""
    v = np.random.default_rng(20261007).standard_normal(lattice.num_nodes)
    weight = 2.0 / (3.0 * lattice.h * lattice.h)
    return abs(float(apply_hex_laplacian(v, lattice).sum())) / (weight * float(np.abs(v).sum()))


def _heat_level_evidence(lattice: HexLattice, qoi_mode: Any, config: Any, conservation: bool) -> Dict[str, Any]:
    qoi = _mode_evidence(qoi_mode, lattice)
    residuals: List[float] = [qoi["eigen_residual"]]
    coefficients: Dict[str, Optional[float]] = {qoi_mode.key: qoi["coefficient_ratio"]}
    exact = np.zeros(lattice.num_nodes)
    semi_discrete = np.zeros(lattice.num_nodes)
    for mode, amplitude in config.modes:
        ev = qoi if mode == qoi_mode else _mode_evidence(mode, lattice)
        residuals.append(ev["eigen_residual"])
        coefficients[mode.key] = ev["coefficient_ratio"]
        exact += amplitude * math.exp(-config.kappa * ev["lambda_exact"] * config.t_final) * ev["phi"]
        semi_discrete += amplitude * math.exp(-config.kappa * ev["lambda_rq"] * config.t_final) * ev["phi"]
    error = semi_discrete - exact
    evidence = {
        "n": lattice.n,
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
    if conservation:
        evidence["conservation_residual"] = _conservation_residual(lattice)
    return evidence


def _heat_study(
    config: Any,
    *,
    lattice_fn: Callable[[int], HexLattice],
    qoi_mode: Any,
    schema: str,
    reference_id: str,
    spec: str,
    conservation: bool,
    mode_fields: Callable[[Any], List[Any]],
) -> Dict[str, Any]:
    r = config.refinement_ratio
    levels = [_heat_level_evidence(lattice_fn(n), qoi_mode, config, conservation) for n in config.levels]
    heat_orders = [
        observed_order(a["heat_rel_l2_error"], b["heat_rel_l2_error"], r) for a, b in zip(levels, levels[1:])
    ]
    qoi = [level["lambda_rq"] for level in levels]
    finest = levels[-1]
    celik_fine, celik_coarse, solution, exact_rel_error_fine = _solution_criteria(qoi, finest["lambda_exact"], r)
    criteria = {
        "C1_eigen_residual": all(level["eigen_residual_max"] <= _EIGEN_RESIDUAL_TOL for level in levels),
        "C2_rayleigh_matches_closed_form": all(
            abs(level["lambda_rq"] - level["lambda_h_closed_form"]) <= _RAYLEIGH_TOL * level["lambda_h_closed_form"]
            for level in levels
        ),
        "C3_isotropic_coefficient": all(
            _within(c, _COEFFICIENT_BAND) for c in finest["coefficient_ratio"].values() if c is not None
        ),
        "C4_heat_error_order": all(_within(p, ORDER_BAND) for p in heat_orders[-2:]),
        **solution,
    }
    code_keys = _HEAT_CODE_KEYS
    if conservation:
        criteria["C10_conservation"] = all(level["conservation_residual"] <= _CONSERVATION_TOL for level in levels)
        code_keys = _TORUS_CODE_KEYS
    code_ok = all(criteria[k] for k in code_keys)
    solution_ok = all(criteria[k] for k in _SOLUTION_KEYS)
    thresholds = {
        "order_band": list(ORDER_BAND),
        "eigen_residual_tol": _EIGEN_RESIDUAL_TOL,
        "rayleigh_tol": _RAYLEIGH_TOL,
        "coefficient_band": list(_COEFFICIENT_BAND),
        "asymptotic_tol": _ASYMPTOTIC_TOL,
        "gci_safety_factor": _GCI_SAFETY_FACTOR,
    }
    if conservation:
        thresholds["conservation_tol"] = _CONSERVATION_TOL
    return {
        "schema": schema,
        "reference_id": reference_id,
        "spec": spec,
        "code_under_test": "src/hexantk_v3.py:hex_laplacian_stencil_weight",
        "config": {
            "levels": list(config.levels),
            "refinement_ratio": r,
            "kappa": config.kappa,
            "t_final": config.t_final,
            "modes": [mode_fields(m) + [float(a)] for m, a in config.modes],
            "qoi_mode": mode_fields(qoi_mode),
        },
        "thresholds": thresholds,
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


def _heat_reference_spec(
    study: Dict[str, Any], *, evidence_ref: str, code_keys: Tuple[str, ...], qoi_name: str, context: str
) -> ReferenceSpec:
    criteria = study["criteria"]
    return ReferenceSpec(
        reference_id=study["reference_id"],
        kind=ReferenceKind.NUMERICAL,
        context_of_use=context,
        quantity_of_interest=qoi_name,
        code_verification=_check_from(criteria, code_keys, evidence_ref),
        spatial_verification=_check_from(criteria, _SOLUTION_KEYS, evidence_ref),
        temporal_verification=EvidenceCheck(
            EvidenceStatus.NOT_APPLICABLE,
            rationale="integrated exactly in time along verified discrete eigenvectors; no time-step error",
        ),
        iterative_verification=EvidenceCheck(
            EvidenceStatus.NOT_APPLICABLE,
            rationale="explicit operator application and closed-form eigenvectors; no iterative solve",
        ),
        spatial_study=_spatial_study(study, qoi_name, "lambda_rq", evidence_ref),
        numerical_uncertainty=_uncertainty(study, evidence_ref),
    )


# --- hex-heat-tri (Phase 0a) ---------------------------------------


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
        _validate_levels(self.levels, minimum=3)
        _positive_finite(self.kappa, name="kappa")
        _positive_finite(self.t_final, name="t_final")
        _validate_modes(self.modes, LameMode)

    @property
    def refinement_ratio(self) -> float:
        return self.levels[1] / self.levels[0]


def run_hex_heat_tri_study(config: HexHeatTriConfig | None = None) -> Dict[str, Any]:
    """Run the hex-heat-tri verification study; returns strict-JSON evidence.

    ``passed`` is True only if every gate criterion of the spec holds.
    """
    config = HexHeatTriConfig() if config is None else config
    if not isinstance(config, HexHeatTriConfig):
        raise TypeError("config must be a HexHeatTriConfig")
    return _heat_study(
        config,
        lattice_fn=triangle_hex_lattice,
        qoi_mode=_QOI_MODE,
        schema=_SCHEMA,
        reference_id=_REFERENCE_ID,
        spec="spec/REFERENCE_CASE_HEX_HEAT_TRI_v1.md",
        conservation=False,
        mode_fields=lambda m: [m.m, m.n, m.parity],
    )


def hex_heat_tri_reference_spec(study: Dict[str, Any], *, evidence_ref: str) -> ReferenceSpec:
    """Express a hex-heat-tri study result as ReferenceSpec v1 (Phase 0a spec section 6).

    The claim ceiling is derived by ReferenceSpec; a failed criterion keeps it
    at comparison_only.
    """
    if not isinstance(study, dict) or study.get("schema") != _SCHEMA:
        raise ValueError(f"study must be a {_SCHEMA} result of run_hex_heat_tri_study")
    n_values = ", ".join(str(level["n"]) for level in study["levels"])
    context = (
        "u_t = kappa*Lap(u) on the unit equilateral triangle, homogeneous Dirichlet BC, regular 7-point "
        f"hex FV Laplacian (engine weight 2/(3h^2)), h = 1/N for N = {n_values}, smooth Lame-mode data"
    )
    return _heat_reference_spec(
        study,
        evidence_ref=evidence_ref,
        code_keys=_HEAT_CODE_KEYS,
        qoi_name="decay rate lambda_RQ of the fundamental Dirichlet Lame mode (1,1,sin)",
        context=context,
    )


# --- hex-heat-torus (Phase 0b) --------------------------------------


def _default_torus_modes() -> Tuple[Tuple[TorusMode, float], ...]:
    return (
        (TorusMode(0, 0, "cos"), 0.3),
        (TorusMode(1, 1, "cos"), 1.0),
        (TorusMode(2, -1, "sin"), 0.5),
        (TorusMode(1, 0, "sin"), 0.25),
    )


_TORUS_QOI_MODE = TorusMode(1, 1, "cos")


@dataclass(frozen=True)
class HexHeatTorusConfig:
    """Problem data and levels (even N). Gate thresholds are contract constants."""

    levels: Tuple[int, ...] = (16, 32, 64, 128)
    kappa: float = 1.0
    t_final: float = 0.01
    modes: Tuple[Tuple[TorusMode, float], ...] = field(default_factory=_default_torus_modes)

    def __post_init__(self) -> None:
        _validate_levels(self.levels, minimum=4, multiple_of=2)
        _positive_finite(self.kappa, name="kappa")
        _positive_finite(self.t_final, name="t_final")
        _validate_modes(self.modes, TorusMode)
        if all(mode.eigenvalue == 0.0 for mode, _ in self.modes):
            raise ValueError("at least one mode must decay (lambda > 0)")

    @property
    def refinement_ratio(self) -> float:
        return self.levels[1] / self.levels[0]


def run_hex_heat_torus_study(config: HexHeatTorusConfig | None = None) -> Dict[str, Any]:
    """Run the hex-heat-torus verification study (Phase 0b spec section 2)."""
    config = HexHeatTorusConfig() if config is None else config
    if not isinstance(config, HexHeatTorusConfig):
        raise TypeError("config must be a HexHeatTorusConfig")
    return _heat_study(
        config,
        lattice_fn=torus_hex_lattice,
        qoi_mode=_TORUS_QOI_MODE,
        schema=_TORUS_SCHEMA,
        reference_id=_TORUS_REFERENCE_ID,
        spec="spec/REFERENCE_CASES_HEX_PHASE0B_v1.md",
        conservation=True,
        mode_fields=lambda m: [m.p, m.q, m.parity],
    )


def hex_heat_torus_reference_spec(study: Dict[str, Any], *, evidence_ref: str) -> ReferenceSpec:
    """Express a hex-heat-torus study result as ReferenceSpec v1 (Phase 0b spec section 2)."""
    if not isinstance(study, dict) or study.get("schema") != _TORUS_SCHEMA:
        raise ValueError(f"study must be a {_TORUS_SCHEMA} result of run_hex_heat_torus_study")
    n_values = ", ".join(str(level["n"]) for level in study["levels"])
    context = (
        "u_t = kappa*Lap(u) on the periodic hex torus with periods (1, 0) and (0, sqrt(3)/2), regular 7-point "
        f"hex FV Laplacian (engine weight 2/(3h^2)), h = 1/N for N = {n_values}, smooth plane-wave data"
    )
    return _heat_reference_spec(
        study,
        evidence_ref=evidence_ref,
        code_keys=_TORUS_CODE_KEYS,
        qoi_name="decay rate lambda_RQ of the torus plane wave (1,1,cos)",
        context=context,
    )


# ------------------------------------------------------------------
# hex-mms-tri (Phase 0b spec section 3)
# ------------------------------------------------------------------


def _default_mms_modes() -> Tuple[Tuple[LameMode, float], ...]:
    return ((LameMode(1, 1, "sin"), -0.5), (LameMode(1, 2, "cos"), 0.2))


@dataclass(frozen=True)
class HexMmsTriConfig:
    """Manufactured solution, coefficient amplitude and levels (multiples of 3).

    kappa(x, y) = 1 + kappa_amplitude * sin(2 pi x + 0.3) * cos(3 pi y),
    |kappa_amplitude| < 1 keeps kappa > 0.

    ``cg_stopping`` selects the I1 contract (spec Phase 0b section 3): "v2"
    (contractual since E1012-07) or "v1" (historical, BLOCKED on N=192).
    """

    levels: Tuple[int, ...] = (24, 48, 96, 192)
    modes: Tuple[Tuple[LameMode, float], ...] = field(default_factory=_default_mms_modes)
    kappa_amplitude: float = 0.4
    cg_stopping: str = "v2"

    def __post_init__(self) -> None:
        _validate_levels(self.levels, minimum=3, multiple_of=3)
        _validate_modes(self.modes, LameMode)
        if not isinstance(self.cg_stopping, str) or self.cg_stopping not in CG_STOPPING_RULES:
            raise ValueError(f"cg_stopping must be one of {CG_STOPPING_RULES}, got {self.cg_stopping!r}")
        amplitude = self.kappa_amplitude
        if isinstance(amplitude, bool) or not isinstance(amplitude, numbers.Real):
            raise TypeError("kappa_amplitude must be a real number")
        if not math.isfinite(float(amplitude)) or abs(float(amplitude)) >= 1.0:
            raise ValueError("kappa_amplitude must be finite with |kappa_amplitude| < 1 (kappa > 0)")

    @property
    def refinement_ratio(self) -> float:
        return self.levels[1] / self.levels[0]

    def kappa(self, coords: Any) -> np.ndarray:
        xy = _check_coords(coords)
        a = float(self.kappa_amplitude)
        return 1.0 + a * np.sin(2.0 * math.pi * xy[:, 0] + 0.3) * np.cos(3.0 * math.pi * xy[:, 1])

    def kappa_gradient(self, coords: Any) -> np.ndarray:
        xy = _check_coords(coords)
        a = float(self.kappa_amplitude)
        sx, cx = np.sin(2.0 * math.pi * xy[:, 0] + 0.3), np.cos(2.0 * math.pi * xy[:, 0] + 0.3)
        sy, cy = np.sin(3.0 * math.pi * xy[:, 1]), np.cos(3.0 * math.pi * xy[:, 1])
        return np.stack([a * 2.0 * math.pi * cx * cy, -a * 3.0 * math.pi * sx * sy], axis=1)

    def solution(self, coords: Any) -> np.ndarray:
        return sum(float(a) * mode.evaluate(coords) for mode, a in self.modes)

    def solution_gradient(self, coords: Any) -> np.ndarray:
        return sum(float(a) * mode.gradient(coords) for mode, a in self.modes)

    def source(self, coords: Any) -> np.ndarray:
        """f = -div(kappa grad u*) = kappa * sum a_i lambda_i phi_i - grad(kappa) . grad(u*)."""
        xy = _check_coords(coords)
        neg_laplacian = sum(float(a) * mode.eigenvalue * mode.evaluate(xy) for mode, a in self.modes)
        return self.kappa(xy) * neg_laplacian - (self.kappa_gradient(xy) * self.solution_gradient(xy)).sum(axis=1)


def _fd_flux_divergence(config: HexMmsTriConfig, points: np.ndarray, step: float) -> np.ndarray:
    """-div(kappa grad u*) by fourth-order central differences of the flux (independent of the
    closed-form source; uses only kappa and grad u*)."""
    out = np.zeros(points.shape[0])
    for axis in range(2):
        shift = np.zeros(2)
        shift[axis] = step

        def flux(offset: float) -> np.ndarray:
            p = points + offset * shift
            return config.kappa(p) * config.solution_gradient(p)[:, axis]

        out += (-flux(2.0) + 8.0 * flux(1.0) - 8.0 * flux(-1.0) + flux(-2.0)) / (12.0 * step)
    return -out


def _triangle_boundary_coords(n: int) -> np.ndarray:
    h = 1.0 / n
    pts = [(h * (i + j / 2.0), h * j * _SQRT3 / 2.0)
           for j in range(n + 1) for i in range(n + 1 - j) if i == 0 or j == 0 or i + j == n]
    return np.asarray(pts, dtype=np.float64)


def _interior_sample_points(count: int) -> np.ndarray:
    """Seeded points at distance >= 0.05 from every edge of the unit triangle."""
    rng = np.random.default_rng(60)
    points: List[np.ndarray] = []
    while len(points) < count:
        x, y = rng.random(2) * np.array([1.0, _SQRT3 / 2.0])
        distances = (y, (_SQRT3 * x - y) / 2.0, (_SQRT3 * (1.0 - x) - y) / 2.0)
        if min(distances) >= 0.05:
            points.append(np.array([x, y]))
    return np.stack(points)


def _centroid_index(lattice: HexLattice) -> int:
    target = np.array([0.5, _SQRT3 / 6.0])
    hits = np.nonzero(np.all(np.abs(lattice.coords - target) <= lattice.h / 4.0, axis=1))[0]
    if hits.size != 1:
        raise ValueError(f"the centroid is not a node of the N={lattice.n} lattice")
    return int(hits[0])


def _mms_solve(lattice: HexLattice, conductance: np.ndarray, source: np.ndarray,
               rtol: float, stopping: str) -> Tuple[np.ndarray, Dict[str, Any], bool]:
    """CG solve for I1: a non-converged solve is evidence (I1 fails), not a crash.

    On ConvergenceError the best iterate and its TRUE relative residual are
    reported with converged=False and status "not_converged", so the remaining
    criteria stay measurable. A returned V1 solve has status "converged"
    (V1 returns only at rtol); V2 reports its own status, tol_eff and eta.
    """
    try:
        u, info = solve_steady_diffusion(lattice, conductance, source, rtol=rtol, stopping=stopping)
    except ConvergenceError as exc:
        if exc.best is None or exc.best_relative_residual is None:
            raise
        return exc.best, {"iterations": exc.iterations, "relative_residual": exc.best_relative_residual,
                          "status": "not_converged"}, False
    if stopping == "v1":
        info = {**info, "status": "converged", "tol_eff": rtol}
    return u, info, True


def _mms_level_evidence(n: int, config: HexMmsTriConfig) -> Dict[str, Any]:
    lattice = triangle_hex_lattice(n)
    conductance = edge_conductances(lattice, config.kappa)
    symmetric = all(
        np.array_equal(conductance[rows, slot], conductance[lattice.neighbors[rows, slot], slot + 3])
        for slot in range(3)
        for rows in (np.nonzero(lattice.neighbors[:, slot] >= 0)[0],)
    )
    diag, off = assemble_hex_operator(lattice, conductance)
    try:
        dt_bound: Optional[float] = positivity_max_dt(diag, off)
    except ValueError:
        dt_bound = None
    source = config.source(lattice.coords)
    stopping = config.cg_stopping
    u_tight, info_tight, converged_tight = _mms_solve(lattice, conductance, source, _MMS_CG_RTOL, stopping)
    u_loose, info_loose, converged_loose = _mms_solve(lattice, conductance, source, _MMS_CG_RTOL_LOOSE, stopping)
    exact = config.solution(lattice.coords)
    error = u_tight - exact
    discretisation_error = float(np.max(np.abs(error)))
    iterative_error = float(np.max(np.abs(u_loose - u_tight)))
    centroid = _centroid_index(lattice)
    return {
        "n": n,
        "h": lattice.h,
        "num_nodes": lattice.num_nodes,
        "kappa_min": float(conductance.min()),
        "conductance_symmetric": bool(symmetric),
        "positivity_dt_bound": dt_bound,
        # "cg_converged": both solves returned (V2: converged or at_rounding_floor).
        "cg_converged": converged_tight and converged_loose,
        "cg_status": info_tight["status"],
        "cg_status_loose": info_loose["status"],
        "cg_iterations": info_tight["iterations"],
        "cg_relative_residual": info_tight["relative_residual"],
        # None where a solve raised (no accepted iterate, so no tol_eff).
        "cg_tol_eff": info_tight.get("tol_eff"),
        "cg_rounding_floor": info_tight.get("rounding_floor"),
        "cg_backward_error": info_tight.get("backward_error"),
        "cg_iterations_loose": info_loose["iterations"],
        "rel_l2_error": float(np.linalg.norm(error) / np.linalg.norm(exact)),
        "max_error": discretisation_error,
        "iterative_error_estimate": iterative_error,
        "iterative_to_discretisation_ratio": iterative_error / discretisation_error,
        "u_centroid": float(u_tight[centroid]),
    }


#: Tight-solve statuses that I1 accepts, per CG stopping rule (Phase 0b spec section 3).
_I1_ACCEPTED_STATUS = {"v1": ("converged",), "v2": ("converged", "at_rounding_floor")}


def _i1_level_holds(level: Dict[str, Any], stopping: str) -> bool:
    """I1 on one level: V1 needs the true residual <= rtol, V2 <= tol_eff of the iterate.

    The PASS branch hangs on comparisons that must be TRUE, so a NaN fails.
    """
    if not level["cg_converged"] or level["cg_status"] not in _I1_ACCEPTED_STATUS[stopping]:
        return False
    bound = _MMS_CG_RTOL if stopping == "v1" else level["cg_tol_eff"]
    return bool(level["cg_relative_residual"] <= bound
                and level["iterative_to_discretisation_ratio"] <= _MMS_ITERATIVE_RATIO)


def run_hex_mms_tri_study(config: HexMmsTriConfig | None = None) -> Dict[str, Any]:
    """Run the hex-mms-tri verification study (Phase 0b spec section 3)."""
    config = HexMmsTriConfig() if config is None else config
    if not isinstance(config, HexMmsTriConfig):
        raise TypeError("config must be a HexMmsTriConfig")
    r = config.refinement_ratio
    finest_n = config.levels[-1]
    # Oracle checks (independent of the solver).
    interior_exact = config.solution(triangle_hex_lattice(finest_n).coords)
    boundary_max = float(np.max(np.abs(config.solution(_triangle_boundary_coords(finest_n)))))
    boundary_ratio = boundary_max / float(np.max(np.abs(interior_exact)))
    points = _interior_sample_points(_MMS_FD_POINTS)
    closed = config.source(points)
    finite_difference = _fd_flux_divergence(config, points, _MMS_FD_STEP)
    source_error = float(np.max(np.abs(closed - finite_difference)) / np.max(np.abs(closed)))
    levels = [_mms_level_evidence(n, config) for n in config.levels]
    l2_orders = [observed_order(a["rel_l2_error"], b["rel_l2_error"], r) for a, b in zip(levels, levels[1:])]
    max_orders = [observed_order(a["max_error"], b["max_error"], r) for a, b in zip(levels, levels[1:])]
    exact_centroid = float(config.solution(np.array([[0.5, _SQRT3 / 6.0]]))[0])
    celik_fine, celik_coarse, solution, exact_rel_error_fine = _solution_criteria(
        [level["u_centroid"] for level in levels], exact_centroid, r
    )
    criteria = {
        "O1_boundary_zero": boundary_ratio <= _MMS_BOUNDARY_TOL,
        "O2_source_matches_flux_divergence": source_error <= _MMS_SOURCE_TOL,
        "O3_kappa_positive": all(level["kappa_min"] > 0.0 for level in levels),
        "M1_error_order": all(_within(p, ORDER_BAND) for p in l2_orders[-2:] + max_orders[-2:]),
        "M2_conductance_symmetric": all(level["conductance_symmetric"] for level in levels),
        "M3_positivity_gate": all(level["positivity_dt_bound"] is not None for level in levels),
        **solution,
        "I1_iterative_error_negligible": all(_i1_level_holds(level, config.cg_stopping) for level in levels),
    }
    oracle_ok = all(criteria[k] for k in _MMS_ORACLE_KEYS)
    code_ok = all(criteria[k] for k in _MMS_CODE_KEYS)
    solution_ok = all(criteria[k] for k in _SOLUTION_KEYS)
    iterative_ok = all(criteria[k] for k in _MMS_ITERATIVE_KEYS)
    return {
        "schema": _MMS_SCHEMA,
        "reference_id": _MMS_REFERENCE_ID,
        "oracle_id": _MMS_ORACLE_ID,
        "spec": "spec/REFERENCE_CASES_HEX_PHASE0B_v1.md",
        "code_under_test": "src/toolkit/hex_fv_solver_hex_v3.py (conductances, operator, CG)",
        "config": {
            "levels": list(config.levels),
            "refinement_ratio": r,
            "modes": [[m.m, m.n, m.parity, float(a)] for m, a in config.modes],
            "kappa": f"1 + {float(config.kappa_amplitude)!r} * sin(2 pi x + 0.3) * cos(3 pi y)",
            "kappa_range": [1.0 - abs(float(config.kappa_amplitude)), 1.0 + abs(float(config.kappa_amplitude))],
            "qoi_point": [0.5, _SQRT3 / 6.0],
        },
        "thresholds": {
            "order_band": list(ORDER_BAND),
            "asymptotic_tol": _ASYMPTOTIC_TOL,
            "gci_safety_factor": _GCI_SAFETY_FACTOR,
            "boundary_tol": _MMS_BOUNDARY_TOL,
            "source_tol": _MMS_SOURCE_TOL,
            "fd_step": _MMS_FD_STEP,
            "fd_points": _MMS_FD_POINTS,
            "cg_rtol": _MMS_CG_RTOL,
            "cg_rtol_loose": _MMS_CG_RTOL_LOOSE,
            "iterative_ratio": _MMS_ITERATIVE_RATIO,
            "cg_stopping": config.cg_stopping,
            "cg_floor_factor": HEX_RESIDUAL_FLOOR_FACTOR if config.cg_stopping == "v2" else None,
        },
        "i1_contract": ("I1 V2 (spec/HEX_FV_SOLVER_v2.md, E1012-07)" if config.cg_stopping == "v2"
                        else "I1 V1 (historical, rtol only; BLOCKED on the default levels since 2026-10-12)"),
        "oracle": {"boundary_ratio": boundary_ratio, "source_rel_error_vs_fd": source_error},
        "levels": levels,
        "rel_l2_error_orders": l2_orders,
        "max_error_orders": max_orders,
        "qoi_exact": exact_centroid,
        "celik_fine": celik_fine,
        "celik_coarse": celik_coarse,
        "qoi_exact_rel_error_fine": exact_rel_error_fine,
        "criteria": criteria,
        "oracle_verification_pass": oracle_ok,
        "code_verification_pass": code_ok,
        "solution_verification_pass": solution_ok,
        "iterative_verification_pass": iterative_ok,
        "passed": code_ok and solution_ok and iterative_ok,
    }


def hex_mms_tri_reference_specs(
    study: Dict[str, Any], *, evidence_ref: str
) -> Tuple[ReferenceSpec, ReferenceSpec]:
    """Express a hex-mms-tri study as two ReferenceSpecs: (manufactured oracle, verified solver)."""
    if not isinstance(study, dict) or study.get("schema") != _MMS_SCHEMA:
        raise ValueError(f"study must be a {_MMS_SCHEMA} result of run_hex_mms_tri_study")
    criteria = study["criteria"]
    oracle_context = (
        "manufactured solution u* = sum a_i phi_i (Lame modes) and kappa = " + study["config"]["kappa"]
        + " on the unit equilateral triangle, homogeneous Dirichlet BC, source f = -div(kappa grad u*)"
    )
    oracle = ReferenceSpec(
        reference_id=study["oracle_id"],
        kind=ReferenceKind.MANUFACTURED,
        context_of_use=oracle_context,
        quantity_of_interest="manufactured solution u* and source f",
        oracle_verification=_check_from(criteria, _MMS_ORACLE_KEYS, evidence_ref),
    )
    n_values = ", ".join(str(level["n"]) for level in study["levels"])
    low, high = study["config"]["kappa_range"]
    qoi_name = "u_h at the centroid (1/2, sqrt(3)/6)"
    solver = ReferenceSpec(
        reference_id=study["reference_id"],
        kind=ReferenceKind.NUMERICAL,
        context_of_use=(
            "-div(kappa grad u) = f on the unit equilateral triangle, homogeneous Dirichlet BC, smooth "
            f"kappa in [{low!r}, {high!r}], two-point-flux hex FV with midpoint conductances, CG, "
            f"h = 1/N for N = {n_values}"
        ),
        quantity_of_interest=qoi_name,
        code_verification=_check_from(criteria, _MMS_CODE_KEYS, evidence_ref),
        spatial_verification=_check_from(criteria, _SOLUTION_KEYS, evidence_ref),
        temporal_verification=EvidenceCheck(EvidenceStatus.NOT_APPLICABLE, rationale="steady problem"),
        iterative_verification=_check_from(criteria, _MMS_ITERATIVE_KEYS, evidence_ref),
        spatial_study=_spatial_study(study, qoi_name, "u_centroid", evidence_ref),
        numerical_uncertainty=_uncertainty(study, evidence_ref),
    )
    return oracle, solver
