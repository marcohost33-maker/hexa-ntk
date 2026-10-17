"""
linear_baseline_hex_v3.py
=========================
Linear learning baseline for operator tasks on the hex lattice (contract:
spec/SPECTRAL_AUDIT_v1.md, section 6): ridge regression between W-weighted
POD coordinates of the inputs and of the outputs (the linear case of
PCA-Net, Bhattacharya, Hosseini, Kovachki & Stuart 2021, SMAI J. Comput.
Math. 7, 121-157).

    a = P_in^T W u0,  b = B^T a,  uT = P_out b,
    B = argmin ||A B - C||_F^2 + alpha ||B||_F^2,

with P_in, P_out W-orthonormal POD bases of the training inputs and outputs,
A, C the training coordinates. Everything is fixed BEFORE any test data is
seen, by ``LinearBaselineConfig``:
- POD rank: singular values above ``rank_rtol * s_max`` (default
  ``max(N, M) * eps``, the numpy.linalg.matrix_rank rule, i.e. numerical
  rank), capped at ``max_rank``;
- ridge: ``alpha = alpha_rel * s_max(A)^2``, so the fit is equivariant under
  scaling of the data (scaling inputs by c scales the map by 1/c).
No centring: the target operators here are linear, not affine.

Leakage guard: ``fit`` records the training sample ids; ``predict`` takes
the ids of the samples it scores and raises if any of them was trained on.
Since 2026-10-15 ``fit`` also records a SHA-256 per training input row in
canonical form (float64 little-endian, -0.0 -> 0.0), and ``predict`` raises
if a scored row repeats a training input under another id.
The operator view (``__call__``, ``matrix``) carries no ids: it is for
instruments such as the spectral audit, not for scoring samples.

Deterministic: no randomness; a fit is a function of the training arrays.

Claim boundary: a baseline instrument only. It is not fitted to T1 data here
(spec/PREREGISTRATION_PHASE1_v1.md: no model on T1 before approval); its
tests use synthetic linear data.
"""

from __future__ import annotations

__all__ = ["LinearBaselineConfig", "PODRidgeBaseline", "LeakageError"]

import hashlib
import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

_EPS64 = float(np.finfo(np.float64).eps)


class LeakageError(ValueError):
    """A sample scored by ``predict`` was part of the training set."""


@dataclass(frozen=True)
class LinearBaselineConfig:
    """Fixed-in-advance rules for rank and regularisation (no test-data choice)."""

    alpha_rel: float = 1e-10
    rank_rtol: Optional[float] = None
    max_rank: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.alpha_rel, bool) or not isinstance(self.alpha_rel, numbers.Real):
            raise TypeError(f"alpha_rel must be a real number, got {self.alpha_rel!r}")
        if not (math.isfinite(float(self.alpha_rel)) and float(self.alpha_rel) >= 0.0):
            raise ValueError(f"alpha_rel must be finite and >= 0, got {self.alpha_rel!r}")
        if self.rank_rtol is not None:
            if isinstance(self.rank_rtol, bool) or not isinstance(self.rank_rtol, numbers.Real):
                raise TypeError(f"rank_rtol must be a real number or None, got {self.rank_rtol!r}")
            if not (math.isfinite(float(self.rank_rtol)) and 0.0 < float(self.rank_rtol) < 1.0):
                raise ValueError(f"rank_rtol must be in (0, 1), got {self.rank_rtol!r}")
        if self.max_rank is not None:
            if isinstance(self.max_rank, bool) or not isinstance(self.max_rank, numbers.Integral):
                raise TypeError(f"max_rank must be an int or None, got {self.max_rank!r}")
            if self.max_rank < 1:
                raise ValueError(f"max_rank must be >= 1, got {self.max_rank!r}")


def _ids(sample_ids: Any, n: int) -> Tuple[int, ...]:
    if isinstance(sample_ids, (str, bytes)):
        raise TypeError("sample_ids must be a sequence of ints")
    ids = list(sample_ids)
    for sid in ids:
        if isinstance(sid, bool) or not isinstance(sid, numbers.Integral):
            raise TypeError(f"sample ids must be ints, got {sid!r}")
        if sid < 0:
            raise ValueError(f"sample ids must be >= 0, got {sid!r}")
    ids = [int(s) for s in ids]
    if len(ids) != n:
        raise ValueError(f"need one sample id per row: {len(ids)} ids for {n} rows")
    if len(set(ids)) != len(ids):
        raise ValueError("sample ids must be unique")
    return tuple(ids)


def _fields(value: Any, num_nodes: Optional[int], *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.bool_ or not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)):
        raise TypeError(f"{name} must be a real numeric array, got dtype {arr.dtype}")
    arr = arr.astype(np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} must have shape [S, M] with S, M >= 1, got {arr.shape}")
    if num_nodes is not None and arr.shape[1] != num_nodes:
        raise ValueError(f"{name} must have {num_nodes} columns, got {arr.shape[1]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _row_hashes(fields: np.ndarray) -> Tuple[bytes, ...]:
    """SHA-256 of each row in canonical form: float64, little-endian, -0.0 -> 0.0.

    Inputs are finite (checked by ``_fields``), so adding 0.0 maps exactly the
    signed zero to +0.0 and leaves every other value unchanged.
    """
    canonical = np.ascontiguousarray(fields.astype("<f8") + 0.0)
    return tuple(hashlib.sha256(row.tobytes()).digest() for row in canonical)


def _pod(fields: np.ndarray, sqrt_w: np.ndarray, config: LinearBaselineConfig, *, name: str
         ) -> Tuple[np.ndarray, np.ndarray]:
    """W-orthonormal POD basis P (M, r) and all singular values of fields * sqrt(W)."""
    _, s, vt = np.linalg.svd(fields * sqrt_w, full_matrices=False)
    rtol = max(fields.shape) * _EPS64 if config.rank_rtol is None else float(config.rank_rtol)
    if not s[0] > 0.0:
        raise ValueError(f"training {name} are all zero: POD rank 0")
    rank = int(np.sum(s > rtol * s[0]))
    if config.max_rank is not None:
        rank = min(rank, config.max_rank)
    return vt[:rank].T / sqrt_w[:, None], s


class PODRidgeBaseline:
    """Ridge regression between W-weighted POD coordinates (see module docstring)."""

    def __init__(self, weights: Any, config: Optional[LinearBaselineConfig] = None) -> None:
        w = np.asarray(weights)
        if w.dtype == np.bool_ or not (np.issubdtype(w.dtype, np.floating) or np.issubdtype(w.dtype, np.integer)):
            raise TypeError(f"weights must be a real numeric array, got dtype {w.dtype}")
        w = w.astype(np.float64)
        if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)) or not np.all(w > 0.0):
            raise ValueError("weights must be a non-empty 1-D array of finite values > 0")
        if config is None:
            config = LinearBaselineConfig()
        if not isinstance(config, LinearBaselineConfig):
            raise TypeError("config must be a LinearBaselineConfig")
        w.flags.writeable = False
        self.weights = w
        self.config = config
        self._fitted: Optional[Dict[str, Any]] = None

    @property
    def num_nodes(self) -> int:
        return int(self.weights.size)

    @property
    def train_ids(self) -> FrozenSet[int]:
        return self._require_fit()["train_ids"]

    @property
    def info(self) -> Dict[str, Any]:
        """Ranks, the ridge alpha actually used and the singular values (copies).

        ``alpha`` is None when alpha_rel * s_max^2 leaves float64 (the fit never forms it);
        ``log10_alpha`` always holds the value (-inf for alpha_rel = 0).
        """
        fit = self._require_fit()
        return {
            "rank_in": fit["p_in"].shape[1],
            "rank_out": fit["p_out"].shape[1],
            "alpha": fit["alpha"],
            "log10_alpha": fit["log10_alpha"],
            "singular_values_in": fit["s_in"].copy(),
            "singular_values_out": fit["s_out"].copy(),
            "num_train": len(fit["train_ids"]),
        }

    def _require_fit(self) -> Dict[str, Any]:
        if self._fitted is None:
            raise RuntimeError("PODRidgeBaseline is not fitted; call fit() first")
        return self._fitted

    def fit(self, u0: Any, u_t: Any, sample_ids: Sequence[int]) -> "PODRidgeBaseline":
        """Fit on training pairs [S, M]; records the training ids for the leakage guard."""
        x = _fields(u0, self.num_nodes, name="u0")
        y = _fields(u_t, self.num_nodes, name="uT")
        if x.shape != y.shape:
            raise ValueError(f"u0 and uT must have the same shape, got {x.shape} and {y.shape}")
        ids = _ids(sample_ids, x.shape[0])
        sqrt_w = np.sqrt(self.weights)
        p_in, s_in = _pod(x, sqrt_w, self.config, name="inputs")
        p_out, s_out = _pod(y, sqrt_w, self.config, name="outputs")
        a = x @ (self.weights[:, None] * p_in)  # (S, r_in)
        c = y @ (self.weights[:, None] * p_out)  # (S, r_out)
        ua, sa, vta = np.linalg.svd(a, full_matrices=False)
        # Report the alpha actually used without ever storing inf (Codex P2 on #71): the fit
        # below never forms alpha, so it stays scale-equivariant even where alpha itself
        # leaves float64; then "alpha" is None and "log10_alpha" carries the value.
        rel, s_max = float(self.config.alpha_rel), float(sa[0])
        alpha = (rel * s_max) * s_max  # (rel * s) * s overflows only if alpha does
        log10_alpha = math.log10(rel) + 2.0 * math.log10(s_max) if rel > 0.0 else -math.inf
        if not math.isfinite(alpha) or (rel > 0.0 and alpha == 0.0):  # overflow or underflow (Codex P2 on #77)
            alpha = None
        # B = V diag(s / (s^2 + alpha)) U^T C, formed with s normalized by s_max:
        # s/(s^2 + alpha) = (t / (t^2 + alpha_rel)) / s_max with t = s / s_max, so neither
        # s^2 nor alpha can overflow or underflow, and the fit stays scale-equivariant
        # (Codex P2 on #71). Stable for alpha = 0 as long as s > 0 (POD rank ensures it).
        t = sa / sa[0]
        filt = (t / (t * t + float(self.config.alpha_rel))) / sa[0]
        b = vta.T @ (filt[:, None] * (ua.T @ c))
        if not np.all(np.isfinite(b)):
            raise ValueError("ridge solution is not finite")
        for arr in (p_in, p_out, b, s_in, s_out):
            arr.flags.writeable = False
        self._fitted = {"p_in": p_in, "p_out": p_out, "b": b, "alpha": alpha, "log10_alpha": log10_alpha,
                        "s_in": s_in, "s_out": s_out,
                        "train_ids": frozenset(ids), "train_rows": frozenset(_row_hashes(x))}
        return self

    def _apply(self, x: np.ndarray) -> np.ndarray:
        fit = self._require_fit()
        coords = x @ (self.weights[:, None] * fit["p_in"])
        return (coords @ fit["b"]) @ fit["p_out"].T

    def predict(self, u0: Any, sample_ids: Sequence[int]) -> np.ndarray:
        """Predictions [S, M] for scored samples; raises LeakageError if any id was trained on."""
        fit = self._require_fit()
        x = _fields(u0, self.num_nodes, name="u0")
        ids = _ids(sample_ids, x.shape[0])
        leaked = sorted(set(ids) & fit["train_ids"])
        if leaked:
            raise LeakageError(f"{len(leaked)} scored sample ids were trained on, e.g. {leaked[:5]}")
        duplicates = [sid for sid, h in zip(ids, _row_hashes(x)) if h in fit["train_rows"]]
        if duplicates:
            raise LeakageError(
                f"{len(duplicates)} scored samples repeat a training input under another id, "
                f"e.g. {duplicates[:5]} (content match after -0.0 -> 0.0)"
            )
        return self._apply(x)

    def __call__(self, u: Any) -> np.ndarray:
        """Operator view u (M,) -> (M,) for instruments; scores no sample, so it carries no ids."""
        x = _fields(np.asarray(u)[None, :] if np.ndim(u) == 1 else u, self.num_nodes, name="u")
        if x.shape[0] != 1:
            raise ValueError("the operator view maps one field of shape (M,)")
        return self._apply(x)[0]

    def matrix(self) -> np.ndarray:
        """The fitted linear map as a dense (M, M) matrix G with uT = G u0."""
        fit = self._require_fit()
        return fit["p_out"] @ fit["b"].T @ (fit["p_in"].T * self.weights[None, :])
