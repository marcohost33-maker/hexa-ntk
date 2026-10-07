"""
heat_operator_task_hex_v3.py
============================
Phase 1 instrument (issue #60): task T1 ``hex-heat-operator`` -- learn the heat
semigroup u0 -> u(., T) on the regular hex lattice. Contract:
spec/HEAT_OPERATOR_TASK_v1.md.

Fixes, before any model exists: the data distribution (exact eigenfunction
basis + spectral prior, seeded per sample), the exact truth, the three
preregistered metrics (relative L2 with cell areas, relative H1 seminorm via
the energy form of the verified operator, relative max norm), the dataset
manifest with SHA-256 digests, and the mechanistic baseline (Crank-Nicolson,
spec/HEX_FV_SOLVER_v1.md).

Claim boundary: instruments only -- no model, no training, no statement that
any learner is better or worse. Wall times are hardware-specific measurements.
"""

from __future__ import annotations

__all__ = [
    "TASK_ID",
    "HeatOperatorTaskConfig",
    "mode_basis",
    "sample_coefficients",
    "evaluate_sample",
    "generate_dataset",
    "dataset_manifest",
    "verify_dataset",
    "HeatOperatorDataset",
    "relative_l2",
    "relative_h1_seminorm",
    "relative_max",
    "evaluate_predictions",
    "mechanistic_baseline",
]

import functools
import hashlib
import json
import math
import numbers
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from hex_fv_solver_hex_v3 import (
    HexLattice,
    apply_hex_laplacian,
    crank_nicolson_heat,
    torus_hex_lattice,
    triangle_hex_lattice,
)
from reference_case_hex_v3 import LameMode, TorusMode

TASK_ID = "hex-heat-operator/v1"
_SCHEMA = "hexantk_heat_operator_dataset_v1"
_DOMAINS = ("triangle_dirichlet", "torus_periodic")
_REPRODUCIBILITY_RTOL = 1e-12
_MODULE_PATH = Path(__file__).resolve()

Mode = Union[LameMode, TorusMode]


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return result


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class HeatOperatorTaskConfig:
    """Task data distribution (spec section 2). Defaults are the T1 proposal."""

    domain: str = "triangle_dirichlet"
    kappa: float = 1.0
    t_final: float = 0.01
    lambda_cutoff: float = 600.0
    spectral_decay: float = 1.0

    def __post_init__(self) -> None:
        if self.domain not in _DOMAINS:
            raise ValueError(f"domain must be one of {_DOMAINS}, got {self.domain!r}")
        _positive_finite(self.kappa, name="kappa")
        _positive_finite(self.t_final, name="t_final")
        _positive_finite(self.lambda_cutoff, name="lambda_cutoff")
        _positive_finite(self.spectral_decay, name="spectral_decay")
        if not mode_basis(self):
            raise ValueError("lambda_cutoff admits no basis function")

    def canonical(self) -> Dict[str, Any]:
        out = asdict(self)
        for key in ("kappa", "t_final", "lambda_cutoff", "spectral_decay"):
            out[key] = float(out[key])
        return out

    def digest(self) -> str:
        return _sha256_json(self.canonical())


def _sha256_json(value: Any) -> str:
    data = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@functools.lru_cache(maxsize=32)
def mode_basis(config: HeatOperatorTaskConfig) -> Tuple[Mode, ...]:
    """All exact eigenfunctions with lambda <= lambda_cutoff, sorted by (lambda, key)."""
    cutoff = float(config.lambda_cutoff)
    modes: List[Mode] = []
    if config.domain == "triangle_dirichlet":
        # (16 pi^2 / 9)(m^2 + m n + n^2) <= cutoff with 1 <= m <= n
        bound = int(math.isqrt(int(cutoff * 9.0 / (16.0 * math.pi**2)))) + 1
        for m in range(1, bound + 1):
            for n in range(m, bound + 1):
                for parity in (("sin",) if m == n else ("sin", "cos")):
                    mode = LameMode(m, n, parity)
                    if mode.eigenvalue <= cutoff:
                        modes.append(mode)
    else:
        # 4 pi^2 (p^2 + 4 q^2 / 3) <= cutoff, half plane p > 0 or (p == 0, q > 0)
        bound = int(math.sqrt(cutoff) / (2.0 * math.pi)) + 1
        for p in range(0, bound + 1):
            for q in range(-2 * bound, 2 * bound + 1):
                if p == 0 and q <= 0:
                    continue
                for parity in ("cos", "sin"):
                    mode = TorusMode(p, q, parity)
                    if mode.eigenvalue <= cutoff:
                        modes.append(mode)
    return tuple(sorted(modes, key=lambda md: (md.eigenvalue, md.key)))


def sample_coefficients(config: HeatOperatorTaskConfig, dataset_seed: int, sample_id: int) -> np.ndarray:
    """a_i = sigma_i * xi_i, xi from SeedSequence([dataset_seed, sample_id]) (spec section 2)."""
    seed = _non_negative_int(dataset_seed, name="dataset_seed")
    sid = _non_negative_int(sample_id, name="sample_id")
    basis = mode_basis(config)
    eigenvalues = np.array([mode.eigenvalue for mode in basis])
    sigma = (eigenvalues.min() / eigenvalues) ** float(config.spectral_decay)
    xi = np.random.default_rng(np.random.SeedSequence([seed, sid])).standard_normal(len(basis))
    return sigma * xi


def _lattice(config: HeatOperatorTaskConfig, n: int) -> HexLattice:
    return triangle_hex_lattice(n) if config.domain == "triangle_dirichlet" else torus_hex_lattice(n)


def evaluate_sample(config: HeatOperatorTaskConfig, coefficients: Any, lattice: HexLattice) -> Tuple[np.ndarray,
                                                                                                   np.ndarray]:
    """Exact (u0, u(T)) at the lattice nodes for one coefficient vector."""
    if not isinstance(lattice, HexLattice) or lattice.domain != config.domain:
        raise ValueError(f"lattice must be a {config.domain} HexLattice")
    basis = mode_basis(config)
    a = np.asarray(coefficients, dtype=np.float64)
    if a.shape != (len(basis),) or not np.all(np.isfinite(a)):
        raise ValueError(f"coefficients must be {len(basis)} finite values")
    u0 = np.zeros(lattice.num_nodes)
    u_t = np.zeros(lattice.num_nodes)
    for coefficient, mode in zip(a, basis):
        phi = mode.evaluate(lattice.coords)
        u0 += coefficient * phi
        u_t += coefficient * math.exp(-float(config.kappa) * mode.eigenvalue * float(config.t_final)) * phi
    return u0, u_t


def _sample_ids(sample_ids: Sequence[int]) -> List[int]:
    ids = [_non_negative_int(s, name="sample_id") for s in sample_ids]
    if not ids:
        raise ValueError("sample_ids must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    return ids


def generate_dataset(
    config: HeatOperatorTaskConfig, *, n: int, sample_ids: Sequence[int], dataset_seed: int
) -> Dict[str, Any]:
    """Arrays u0, uT [S, M] and coefficients [S, K] for the given samples on lattice N, plus the manifest."""
    if not isinstance(config, HeatOperatorTaskConfig):
        raise TypeError("config must be a HeatOperatorTaskConfig")
    ids = _sample_ids(sample_ids)
    lattice = _lattice(config, n)
    coefficients = np.stack([sample_coefficients(config, dataset_seed, sid) for sid in ids])
    pairs = [evaluate_sample(config, a, lattice) for a in coefficients]
    data = {
        "lattice": lattice,
        "sample_ids": np.asarray(ids, dtype=np.int64),
        "coefficients": coefficients,
        "u0": np.stack([p[0] for p in pairs]),
        "uT": np.stack([p[1] for p in pairs]),
    }
    data["manifest"] = dataset_manifest(config, data, dataset_seed=dataset_seed)
    return data


def _lattice_digest(lattice: HexLattice) -> str:
    digest = hashlib.sha256()
    digest.update(lattice.domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray([lattice.n], dtype="<i8").tobytes())
    digest.update(np.asarray([lattice.h], dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(lattice.coords, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(lattice.neighbors, dtype="<i8").tobytes())
    return digest.hexdigest()


def _sample_digest(coefficients: np.ndarray, u0: np.ndarray, u_t: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(coefficients, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(u0, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(u_t, dtype="<f8").tobytes())
    return digest.hexdigest()


def _module_digest() -> str:
    return hashlib.sha256(_MODULE_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def dataset_manifest(config: HeatOperatorTaskConfig, data: Dict[str, Any], *, dataset_seed: int) -> Dict[str, Any]:
    """Canonical manifest with per-sample and dataset SHA-256 digests (spec section 4)."""
    lattice = data["lattice"]
    sample_digests = [
        _sample_digest(coeff, u0, u_t)
        for coeff, u0, u_t in zip(data["coefficients"], data["u0"], data["uT"])
    ]
    config_sha = config.digest()
    lattice_sha = _lattice_digest(lattice)
    dataset_seed = _non_negative_int(dataset_seed, name="dataset_seed")
    sample_ids = [int(s) for s in data["sample_ids"]]
    basis = mode_basis(config)
    return {
        "schema": _SCHEMA,
        "task_id": TASK_ID,
        "domain": config.domain,
        "n": lattice.n,
        "num_nodes": lattice.num_nodes,
        "lattice_sha256": lattice_sha,
        "config": config.canonical(),
        "config_sha256": config_sha,
        "dataset_seed": dataset_seed,
        "sample_ids": sample_ids,
        "basis": [[mode.key, mode.eigenvalue] for mode in basis],
        "sample_sha256": sample_digests,
        "dataset_sha256": _sha256_json({
            "config_sha256": config_sha,
            "dataset_seed": dataset_seed,
            "sample_ids": sample_ids,
            "lattice_sha256": lattice_sha,
            "samples": sample_digests,
        }),
        "generator_sha256": _module_digest(),
        "numpy_version": np.__version__,
    }


def verify_dataset(manifest: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Verify content integrity and cross-platform reproducibility fail-closed."""
    if not isinstance(manifest, dict) or manifest.get("schema") != _SCHEMA:
        raise ValueError(f"manifest must be a {_SCHEMA} manifest")
    config = HeatOperatorTaskConfig(**manifest["config"])
    if config.digest() != manifest["config_sha256"]:
        raise ValueError("config digest mismatch")

    lattice = data.get("lattice")
    if not isinstance(lattice, HexLattice):
        raise ValueError("data must contain a HexLattice")
    coords = np.asarray(lattice.coords)
    neighbors = np.asarray(lattice.neighbors)
    if (
        coords.ndim != 2
        or coords.shape[1:] != (2,)
        or neighbors.shape != (lattice.num_nodes, 6)
        or not np.all(np.isfinite(coords))
        or not np.issubdtype(neighbors.dtype, np.integer)
        or np.any(neighbors < -1)
        or np.any(neighbors >= lattice.num_nodes)
    ):
        return {
            "integrity": False,
            "reproducible": False,
            "max_relative_deviation": math.inf,
            "bitwise_identical": False,
        }

    sample_ids = [int(s) for s in data["sample_ids"]]
    coefficients = np.asarray(data["coefficients"], dtype=np.float64)
    u0 = np.asarray(data["u0"], dtype=np.float64)
    u_t = np.asarray(data["uT"], dtype=np.float64)
    expected_shape = (len(sample_ids), lattice.num_nodes)
    expected_coeff_shape = (len(sample_ids), len(mode_basis(config)))
    if (
        u0.shape != expected_shape
        or u_t.shape != expected_shape
        or coefficients.shape != expected_coeff_shape
        or not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(u0))
        or not np.all(np.isfinite(u_t))
    ):
        return {
            "integrity": False,
            "reproducible": False,
            "max_relative_deviation": math.inf,
            "bitwise_identical": False,
        }

    lattice_sha = _lattice_digest(lattice)
    digests = [
        _sample_digest(coeff, initial, final)
        for coeff, initial, final in zip(coefficients, u0, u_t)
    ]
    dataset_sha = _sha256_json({
        "config_sha256": manifest["config_sha256"],
        "dataset_seed": int(manifest["dataset_seed"]),
        "sample_ids": sample_ids,
        "lattice_sha256": lattice_sha,
        "samples": digests,
    })
    expected_basis = [[mode.key, mode.eigenvalue] for mode in mode_basis(config)]
    integrity = (
        manifest.get("domain") == config.domain == lattice.domain
        and manifest.get("n") == lattice.n
        and manifest.get("num_nodes") == lattice.num_nodes
        and manifest.get("lattice_sha256") == lattice_sha
        and sample_ids == manifest["sample_ids"]
        and manifest.get("basis") == expected_basis
        and digests == manifest["sample_sha256"]
        and dataset_sha == manifest["dataset_sha256"]
    )

    regenerated = generate_dataset(
        config,
        n=manifest["n"],
        sample_ids=manifest["sample_ids"],
        dataset_seed=manifest["dataset_seed"],
    )
    fresh_lattice = regenerated["lattice"]
    if not np.array_equal(lattice.neighbors, fresh_lattice.neighbors):
        return {
            "integrity": bool(integrity),
            "reproducible": False,
            "max_relative_deviation": math.inf,
            "bitwise_identical": False,
        }

    worst = 0.0
    coord_scale = max(float(np.max(np.abs(fresh_lattice.coords))), np.finfo(float).tiny)
    worst = max(
        worst,
        float(np.max(np.abs(lattice.coords - fresh_lattice.coords)) / coord_scale),
    )
    for key, given in (("coefficients", coefficients), ("u0", u0), ("uT", u_t)):
        fresh = np.asarray(regenerated[key])
        if given.shape != fresh.shape:
            return {
                "integrity": bool(integrity),
                "reproducible": False,
                "max_relative_deviation": math.inf,
                "bitwise_identical": False,
            }
        scale = np.maximum(np.max(np.abs(fresh), axis=1), np.finfo(float).tiny)
        worst = max(
            worst,
            float(np.max(np.max(np.abs(given - fresh), axis=1) / scale)),
        )
    return {
        "integrity": bool(integrity),
        "reproducible": worst <= _REPRODUCIBILITY_RTOL,
        "max_relative_deviation": worst,
        "bitwise_identical": regenerated["manifest"]["dataset_sha256"] == manifest["dataset_sha256"],
    }


class HeatOperatorDataset(Dataset):
    """Samples {"trajectory_id": sample_id, "u0": ..., "uT": ...} for dataset_split_hex_v3 and DataLoader.

    ``samples`` is the metadata view the splitter groups on (one trajectory per
    sample_id), so splits are leakage-safe by construction (spec section 5).
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self.sample_ids = [int(s) for s in data["sample_ids"]]
        self.u0 = np.asarray(data["u0"], dtype=np.float64)
        self.u_t = np.asarray(data["uT"], dtype=np.float64)
        if self.u0.shape != self.u_t.shape or self.u0.shape[0] != len(self.sample_ids):
            raise ValueError("u0, uT and sample_ids must agree in length and shape")
        self.samples = [{"trajectory_id": sid} for sid in self.sample_ids]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"trajectory_id": self.sample_ids[index], "u0": self.u0[index], "uT": self.u_t[index]}


# ------------------------------------------------------------------
# Metrics (spec section 3)
# ------------------------------------------------------------------


def _pair(pred: Any, truth: Any, lattice: HexLattice) -> Tuple[np.ndarray, np.ndarray]:
    p = np.atleast_2d(np.asarray(pred, dtype=np.float64))
    t = np.atleast_2d(np.asarray(truth, dtype=np.float64))
    if p.shape != t.shape or p.shape[1] != lattice.num_nodes:
        raise ValueError(f"pred and truth must have the same shape [S, {lattice.num_nodes}]")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(t))):
        raise ValueError("pred and truth must be finite")
    return p, t


def _nonzero(norms: np.ndarray, name: str) -> np.ndarray:
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} of the truth is zero for at least one sample")
    return norms


def relative_l2(pred: Any, truth: Any, lattice: HexLattice) -> np.ndarray:
    """||e||_A / ||u||_A per sample (equal cell areas: an RMS ratio)."""
    p, t = _pair(pred, truth, lattice)
    return np.linalg.norm(p - t, axis=1) / _nonzero(np.linalg.norm(t, axis=1), "L2 norm")


def _energy_seminorm(v: np.ndarray, lattice: HexLattice) -> np.ndarray:
    values = np.asarray(
        [-lattice.cell_area * float(row @ apply_hex_laplacian(row, lattice)) for row in v],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("discrete H1 energy must be finite")
    scale = max(1.0, float(np.max(np.abs(values))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if np.any(values < -tolerance):
        raise ValueError("discrete H1 energy must be non-negative")
    return np.sqrt(np.maximum(values, 0.0))


def relative_h1_seminorm(pred: Any, truth: Any, lattice: HexLattice) -> np.ndarray:
    """|e|_{1,h} / |u|_{1,h} with |v|_{1,h}^2 = -A v . (L_h v) (Dirichlet edges included)."""
    p, t = _pair(pred, truth, lattice)
    return _energy_seminorm(p - t, lattice) / _nonzero(_energy_seminorm(t, lattice), "H1 seminorm")


def relative_max(pred: Any, truth: Any, lattice: HexLattice) -> np.ndarray:
    """max|e| / max|u| per sample."""
    p, t = _pair(pred, truth, lattice)
    return np.max(np.abs(p - t), axis=1) / _nonzero(np.max(np.abs(t), axis=1), "max norm")


def evaluate_predictions(pred: Any, truth: Any, lattice: HexLattice) -> Dict[str, List[float]]:
    """All three preregistered metrics per sample."""
    return {
        "rel_l2": relative_l2(pred, truth, lattice).tolist(),
        "rel_h1": relative_h1_seminorm(pred, truth, lattice).tolist(),
        "rel_max": relative_max(pred, truth, lattice).tolist(),
    }


# ------------------------------------------------------------------
# Mechanistic baseline (spec section 6)
# ------------------------------------------------------------------


def mechanistic_baseline(
    config: HeatOperatorTaskConfig, u0: Any, lattice: HexLattice, *, steps: int, rtol: float = 1e-12
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """u0 -> u_h(T) with Crank-Nicolson (dt = T / steps) on the input lattice; returns (u, info)."""
    if not isinstance(lattice, HexLattice) or lattice.domain != config.domain:
        raise ValueError(f"lattice must be a {config.domain} HexLattice")
    steps = _non_negative_int(steps, name="steps")
    if steps == 0:
        raise ValueError("steps must be >= 1")
    start = time.perf_counter()
    u, info = crank_nicolson_heat(u0, lattice, kappa=float(config.kappa), dt=float(config.t_final) / steps,
                                  steps=steps, rtol=rtol)
    info = dict(info, wall_time_s=time.perf_counter() - start, n=lattice.n, num_nodes=lattice.num_nodes)
    return u, info
