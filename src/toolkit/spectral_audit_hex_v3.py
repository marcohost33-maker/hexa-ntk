"""
spectral_audit_hex_v3.py
========================
Spectral audit of an operator ``u0 -> uT`` on the hex lattice (contract:
spec/SPECTRAL_AUDIT_v1.md): the input Jacobian ``J = dF/du0`` at a base point,
projected onto a W-orthonormal mode basis ``Q``,

    S = Q^T W J Q,

with ``W = diag(weights)`` (cell areas). For the heat semigroup of task T1
the truth is known in closed form: the T1 basis functions are exact
eigenvectors of the discrete hex Laplacian (Phase 0 C1), so the semidiscrete
operator ``exp(kappa T L_h)`` gives ``S = diag(exp(-kappa lambda_h T))``,
Crank-Nicolson gives ``S = diag(r(kappa lambda_h dt)^steps)`` with
``r(z) = (1 - z/2) / (1 + z/2)``, and the continuous T1 truth gives
``S = diag(exp(-kappa lambda T))``. Deviations of a learned operator show a
frequency-response error (diagonal), wrong mode coupling (off-diagonal) and
response outside the span of the basis (out-of-span residual, invisible in S).

Two backends, one definition:
- numpy callables: central finite differences along each mode (2K + 1
  operator calls, step documented in ``_default_fd_step``);
- torch callables and ``nn.Module``: forward-mode ``torch.func.jvp`` (K calls,
  gives ``J q_k`` and hence the out-of-span residual) or reverse mode
  (``torch.autograd.grad`` with cotangent ``W q_k``: K backward passes, gives
  the rows of ``Q^T W J`` only, no out-of-span residual).

Claim boundary: this is the INPUT Jacobian of one operator at one base point.
It is not the parameter NTK ``J_theta J_theta^T`` (issue #60) and says nothing
about training dynamics. For a nonlinear operator S depends on the base point.
"""

from __future__ import annotations

__all__ = [
    "BudgetExceededError",
    "NonFiniteOutputError",
    "NondeterministicOperatorError",
    "HeatModeMatrix",
    "SpectralAuditResult",
    "heat_mode_matrix",
    "orthonormality_defect",
    "default_orthonormality_tol",
    "semigroup_response",
    "crank_nicolson_response",
    "input_jacobian_spectrum",
    "audit_verdict",
]

import math
import numbers
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

from heat_operator_task_hex_v3 import HeatOperatorTaskConfig, mode_basis
from hex_fv_solver_hex_v3 import HexLattice

_EPS64 = float(np.finfo(np.float64).eps)
_DEFAULT_MAX_EVALUATIONS = 4096
_BACKENDS = ("auto", "numpy", "torch")
_TORCH_MODES = ("forward", "reverse")


class BudgetExceededError(RuntimeError):
    """The audit would exceed (or exceeded) its evaluation or time budget."""


class NonFiniteOutputError(ValueError):
    """The operator, or a quantity derived from it, is not finite."""


class NondeterministicOperatorError(ValueError):
    """Two forwards at the base point differ bitwise, or the operator moved the global RNG."""


# Forwards spent on the determinism preflight (counted in the evaluation budget).
_REPEAT_EVALUATIONS = 2


def _rng_snapshot() -> Tuple[Any, ...]:
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else []
    return torch.get_rng_state(), np.random.get_state(), random.getstate(), cuda


def _rng_unchanged(before: Tuple[Any, ...], after: Tuple[Any, ...]) -> bool:
    np_before, np_after = before[1], after[1]
    return (
        torch.equal(before[0], after[0])
        and np_before[0] == np_after[0]
        and np.array_equal(np_before[1], np_after[1])
        and tuple(np_before[2:]) == tuple(np_after[2:])
        and before[2] == after[2]
        and len(before[3]) == len(after[3])
        and all(torch.equal(a, b) for a, b in zip(before[3], after[3]))
    )


def _repeat_check(operator: Callable[[Any], Any], base: np.ndarray, m: int, use_torch: bool,
                  dtype: Optional[torch.dtype], device: Optional[torch.device], budget: "_Budget") -> None:
    """Two forwards at ``base`` must agree bitwise and leave the global RNG state unchanged."""
    before = _rng_snapshot()
    outputs = []
    for _ in range(_REPEAT_EVALUATIONS):
        if use_torch:
            with torch.no_grad():
                y = _torch_output(operator(torch.as_tensor(base, dtype=dtype, device=device)), m)
            y = y.detach().to("cpu").contiguous()
            outputs.append((str(y.dtype), y.view(torch.uint8).numpy().tobytes()))
        else:
            y = _numpy_output(operator(base.copy()), m)
            outputs.append((y.dtype.str, np.ascontiguousarray(y).tobytes()))
        budget.tick()
    if not _rng_unchanged(before, _rng_snapshot()):
        raise NondeterministicOperatorError(
            "operator changed the global RNG state (torch/numpy/random) during two forwards at the base point; "
            "dropout or sampling would make J random")
    if not outputs[0] == outputs[1]:
        raise NondeterministicOperatorError(
            "two forwards at the base point differ bitwise; J would not be reproducible")


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or not result > 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return int(value)


def _real_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.bool_ or not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)):
        raise TypeError(f"{name} must be a real numeric array, got dtype {arr.dtype}")
    arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _check_weights(weights: Any) -> np.ndarray:
    w = _real_array(weights, name="weights")
    if w.ndim != 1 or w.size == 0:
        raise ValueError(f"weights must be a non-empty 1-D array, got shape {w.shape}")
    if not np.all(w > 0.0):
        raise ValueError("weights must be > 0 on every node")
    return w


def _check_modes(modes: Any, num_nodes: int) -> np.ndarray:
    q = _real_array(modes, name="modes")
    if q.ndim != 2 or q.shape[0] != num_nodes or q.shape[1] < 1:
        raise ValueError(f"modes must have shape ({num_nodes}, K) with K >= 1, got {q.shape}")
    if q.shape[1] > num_nodes:
        raise ValueError(f"modes: K = {q.shape[1]} columns cannot be orthonormal in {num_nodes} dimensions")
    return q


# ------------------------------------------------------------------
# Orthonormality (Bezugsgroesse: the Cauchy-Schwarz bound 1, see spec section 3)
# ------------------------------------------------------------------


def default_orthonormality_tol(num_nodes: int) -> float:
    """Rounding bound for max|Q^T W Q - I| of a W-orthonormal Q with M rows.

    Each entry is a W-weighted dot product of two W-unit vectors; its rounding
    error is at most gamma_M * sum_i w_i |q_ik q_il| <= gamma_M * 1 (Higham
    2002, section 3.1, and Cauchy-Schwarz), gamma_M ~ M eps. The reference size is
    therefore 1, the exact value of the diagonal. Factor 4 covers the rounding
    of the normalisation that produced Q.
    """
    m = _positive_int(num_nodes, name="num_nodes")
    return 4.0 * m * _EPS64


def orthonormality_defect(modes: Any, weights: Any) -> float:
    """max |Q^T W Q - I| (entrywise)."""
    w = _check_weights(weights)
    q = _check_modes(modes, w.size)
    gram = q.T @ (w[:, None] * q)
    return float(np.max(np.abs(gram - np.eye(q.shape[1]))))


# ------------------------------------------------------------------
# The T1 mode basis and its closed-form responses
# ------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class HeatModeMatrix:
    """W-orthonormal T1 basis on one lattice. Arrays are read-only."""

    keys: Tuple[str, ...]
    modes: np.ndarray  # Q, shape (M, K), columns W-orthonormal
    weights: np.ndarray  # W diagonal, shape (M,)
    eigenvalues: np.ndarray  # continuous lambda, shape (K,)
    discrete_eigenvalues: np.ndarray  # lambda_h on this lattice, shape (K,)
    orthonormality_defect: float
    orthonormality_tol: float


def heat_mode_matrix(config: HeatOperatorTaskConfig, lattice: HexLattice) -> HeatModeMatrix:
    """Normalise the sampled T1 basis functions in the W-norm; fail closed if they are not W-orthonormal.

    The sampled functions are exact eigenvectors of L_h and W-orthogonal on
    every lattice that resolves them (measured: correlation <= 1e-15 up to
    N = 128), but not W-normalised. On a lattice too coarse for the basis a
    function can vanish or alias onto another one; that raises ValueError.
    """
    if not isinstance(config, HeatOperatorTaskConfig):
        raise TypeError("config must be a HeatOperatorTaskConfig")
    if not isinstance(lattice, HexLattice) or lattice.domain != config.domain:
        raise ValueError(f"lattice must be a {config.domain} HexLattice")
    basis = mode_basis(config)
    if len(basis) > lattice.num_nodes:
        raise ValueError(
            f"{len(basis)} basis functions cannot be orthonormal on {lattice.num_nodes} nodes (lattice too coarse)"
        )
    weights = np.full(lattice.num_nodes, float(lattice.cell_area))
    phi = np.stack([mode.evaluate(lattice.coords) for mode in basis], axis=1)
    norms = np.sqrt(np.sum(weights[:, None] * phi * phi, axis=0))
    if not np.all(norms > 0.0):
        bad = [mode.key for mode, nrm in zip(basis, norms) if not nrm > 0.0]
        raise ValueError(f"basis functions vanish on this lattice (too coarse): {bad}")
    q = phi / norms
    defect = orthonormality_defect(q, weights)
    tol = default_orthonormality_tol(lattice.num_nodes)
    if not defect <= tol:
        raise ValueError(
            f"T1 basis is not W-orthonormal on this lattice: max|Q^T W Q - I| = {defect:.3e} > {tol:.3e} "
            "(aliasing on a lattice too coarse for lambda_cutoff)"
        )
    lam = np.array([mode.eigenvalue for mode in basis])
    lam_h = np.array([mode.discrete_eigenvalue(lattice.h) for mode in basis])
    for arr in (q, weights, lam, lam_h):
        arr.flags.writeable = False
    return HeatModeMatrix(
        keys=tuple(mode.key for mode in basis),
        modes=q,
        weights=weights,
        eigenvalues=lam,
        discrete_eigenvalues=lam_h,
        orthonormality_defect=defect,
        orthonormality_tol=tol,
    )


def semigroup_response(eigenvalues: Any, *, kappa: float, t: float) -> np.ndarray:
    """exp(-kappa * lambda * t) per mode: the diagonal of S for the exact semigroup."""
    lam = _real_array(eigenvalues, name="eigenvalues")
    if lam.ndim != 1 or not np.all(lam >= 0.0):
        raise ValueError("eigenvalues must be a 1-D array of values >= 0")
    k = _positive_finite(kappa, name="kappa")
    tt = _positive_finite(t, name="t")
    return np.exp(-k * lam * tt)


def crank_nicolson_response(eigenvalues: Any, *, kappa: float, dt: float, steps: int) -> np.ndarray:
    """((1 - z/2) / (1 + z/2))^steps with z = kappa * lambda * dt: S of Crank-Nicolson on an eigenvector."""
    lam = _real_array(eigenvalues, name="eigenvalues")
    if lam.ndim != 1 or not np.all(lam >= 0.0):
        raise ValueError("eigenvalues must be a 1-D array of values >= 0")
    kap = _positive_finite(kappa, name="kappa")
    delta = _positive_finite(dt, name="dt")
    n = _positive_int(steps, name="steps")
    # z from mantissas and exponents: kappa * lambda may overflow although z itself is
    # moderate (tiny dt), and a z that truly overflows has the finite limit r = -1
    # instead of inf/inf = NaN (Codex P2 on #71).
    (mk, ek), (md, ed) = math.frexp(kap), math.frexp(delta)
    ml, el = np.frexp(lam)
    with np.errstate(over="ignore", invalid="ignore"):
        z = np.ldexp(ml * (mk * md), el + (ek + ed))
        r = np.where(np.isinf(z), -1.0, (1.0 - 0.5 * z) / (1.0 + 0.5 * z))
    return r ** n


# ------------------------------------------------------------------
# The audit
# ------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SpectralAuditResult:
    """Output of ``input_jacobian_spectrum``. Arrays are read-only float64."""

    projection: np.ndarray  # S = Q^T W J Q, shape (K, K); column j = response to input mode j
    diagonal: np.ndarray  # diag(S): frequency response per mode
    coupling_fraction: Optional[float]  # ||S - diag(S)||_F / ||S||_F; None if S == 0
    offdiag_abs_max: float
    out_of_span: Optional[np.ndarray]  # ||J q_k - Q S[:, k]||_W / ||J q_k||_W per mode; None in reverse mode
    oracle: Optional[np.ndarray]
    diag_abs_error: Optional[np.ndarray]  # |S_kk - o_k|
    diag_rel_error: Optional[np.ndarray]  # |S_kk - o_k| / |o_k| (0 if both 0, inf if only o_k == 0)
    spectral_rel_error: Optional[float]  # ||S - diag(o)||_F / ||o||_2
    orthonormality_defect: float
    orthonormality_tol: float
    backend: str
    torch_mode: Optional[str]
    dtype: str
    fd_step: Optional[float]
    evaluations: int
    wall_time_s: float

    @property
    def out_of_span_max(self) -> Optional[float]:
        return None if self.out_of_span is None else float(np.max(self.out_of_span))


def _default_fd_step(base: np.ndarray, weights: np.ndarray, output_dtype: np.dtype) -> float:
    """h = eps_out^(1/3) * max(1, ||u0||_W) for central differences along a W-unit direction.

    Central differences have truncation error O(h^2 |F'''|) and rounding error
    O(eps |F| / h); balancing them gives h ~ eps^(1/3) times the scale of the
    argument (Nocedal & Wright 2006, section 8.1; Numerical Recipes 3rd ed.,
    section 5.7). The scale is the W-norm of the base point, at least 1 (the
    W-norm of a perturbation direction). ``eps_out`` is the machine epsilon of
    the operator's output dtype, so a float32 operator gets ~5e-3, not ~6e-6.
    For a linear operator the step only affects rounding.
    """
    eps_out = float(np.finfo(output_dtype).eps) if np.issubdtype(output_dtype, np.floating) else _EPS64
    scale = max(1.0, float(math.sqrt(float(np.sum(weights * base * base)))))
    return eps_out ** (1.0 / 3.0) * scale


def _numpy_output(value: Any, num_nodes: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.shape != (num_nodes,):
        raise ValueError(f"operator must return shape ({num_nodes},), got {arr.shape}")
    if arr.dtype == np.bool_ or not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)):
        raise TypeError(f"operator must return real numbers, got dtype {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise NonFiniteOutputError("operator returned a non-finite value")
    return arr


class _Budget:
    def __init__(self, max_evaluations: int, max_seconds: Optional[float]) -> None:
        self.max_evaluations = max_evaluations
        self.max_seconds = max_seconds
        self.count = 0
        self.start = time.perf_counter()

    def require(self, needed: int) -> None:
        if needed > self.max_evaluations:
            raise BudgetExceededError(f"audit needs {needed} evaluations > max_evaluations={self.max_evaluations}")

    def tick(self) -> None:
        self.count += 1
        if self.max_seconds is not None and not (time.perf_counter() - self.start <= self.max_seconds):
            raise BudgetExceededError(f"audit exceeded max_seconds={self.max_seconds} after {self.count} evaluations")


def _jq_numpy(operator: Callable[[np.ndarray], Any], q: np.ndarray, w: np.ndarray, base: np.ndarray,
              fd_step: Optional[float], budget: _Budget) -> Tuple[np.ndarray, float, str]:
    m, k = q.shape
    budget.require(2 * k + 1)
    reference = _numpy_output(operator(base.copy()), m)
    budget.tick()
    step = _default_fd_step(base, w, reference.dtype) if fd_step is None else fd_step
    jq = np.empty((m, k))
    for j in range(k):
        plus = _numpy_output(operator(base + step * q[:, j]), m).astype(np.float64)
        budget.tick()
        minus = _numpy_output(operator(base - step * q[:, j]), m).astype(np.float64)
        budget.tick()
        jq[:, j] = (plus - minus) / (2.0 * step)
    return jq, step, str(reference.dtype)


def _torch_dtype(operator: Any, torch_dtype: Optional[torch.dtype]) -> torch.dtype:
    if torch_dtype is not None:
        if not isinstance(torch_dtype, torch.dtype) or not torch_dtype.is_floating_point:
            raise TypeError(f"torch_dtype must be a floating torch.dtype, got {torch_dtype!r}")
        return torch_dtype
    if isinstance(operator, torch.nn.Module):
        for tensor in list(operator.parameters()) + list(operator.buffers()):
            if tensor.is_floating_point():
                return tensor.dtype
    return torch.float64


def _torch_device(operator: Any, torch_device: Any = None) -> torch.device:
    """The module's own device; for a plain callable ``torch_device`` (default CPU).

    A plain torch callable gives no device to inspect, so a closure around a CUDA
    operator needs ``torch_device`` (Codex P2 on #71). For a module an override that
    conflicts with its parameters is rejected rather than silently moving inputs.
    """
    wanted = None if torch_device is None else torch.device(torch_device)
    if isinstance(operator, torch.nn.Module):
        for tensor in list(operator.parameters()) + list(operator.buffers()):
            if wanted is not None and wanted != tensor.device:
                raise ValueError(f"torch_device {wanted} conflicts with the module device {tensor.device}")
            return tensor.device
    return wanted if wanted is not None else torch.device("cpu")


def _torch_output(value: Any, num_nodes: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"torch operator must return a torch.Tensor, got {type(value).__name__}")
    if tuple(value.shape) != (num_nodes,):
        raise ValueError(f"operator must return shape ({num_nodes},), got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"operator must return a floating tensor, got {value.dtype}")
    if not bool(torch.all(torch.isfinite(value.detach()))):
        raise NonFiniteOutputError("operator returned a non-finite value")
    return value


def _jq_torch(operator: Callable[[torch.Tensor], torch.Tensor], q: np.ndarray, w: np.ndarray, base: np.ndarray,
              dtype: torch.dtype, device: torch.device, mode: str,
              budget: _Budget) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Forward mode: (J Q, None). Reverse mode: (None, Q^T W J as (K, M))."""
    m, k = q.shape
    x = torch.as_tensor(base, dtype=dtype, device=device)
    if mode == "forward":
        budget.require(k)
        cols = []
        for j in range(k):
            tangent = torch.as_tensor(q[:, j], dtype=dtype, device=device)
            primal_out, tangent_out = torch.func.jvp(lambda v: _torch_output(operator(v), m), (x,), (tangent,))
            budget.tick()
            if not bool(torch.all(torch.isfinite(tangent_out))):
                raise NonFiniteOutputError("Jacobian-vector product is not finite")
            cols.append(tangent_out.detach().to("cpu", torch.float64).numpy())
        return np.stack(cols, axis=1), None
    budget.require(k + 1)
    x = x.clone().requires_grad_(True)
    y = _torch_output(operator(x), m)
    budget.tick()
    rows = []
    for j in range(k):
        if y.requires_grad:
            cotangent = torch.as_tensor(w * q[:, j], dtype=y.dtype, device=y.device)
            (grad,) = torch.autograd.grad(y, x, grad_outputs=cotangent, retain_graph=True, allow_unused=True)
        else:
            grad = None
        budget.tick()
        row = np.zeros(m) if grad is None else grad.detach().to("cpu", torch.float64).numpy()
        if not np.all(np.isfinite(row)):
            raise NonFiniteOutputError("vector-Jacobian product is not finite")
        rows.append(row)
    return None, np.stack(rows, axis=0)


def _relative_errors(diag: np.ndarray, oracle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    abs_err = np.abs(diag - oracle)
    denom = np.abs(oracle)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0.0, abs_err / np.where(denom > 0.0, denom, 1.0),
                       np.where(abs_err > 0.0, np.inf, 0.0))
    return abs_err, rel


def _scaled_column_norms(a: np.ndarray) -> np.ndarray:
    """Euclidean column norms without overflow/underflow: scale by the column max."""
    scale = np.max(np.abs(a), axis=0)
    safe = np.where(scale > 0.0, scale, 1.0)
    return np.where(scale > 0.0, scale * np.sqrt(np.sum((a / safe) ** 2, axis=0)), 0.0)


def input_jacobian_spectrum(
    operator: Any,
    modes: Any,
    weights: Any,
    *,
    base_point: Any = None,
    oracle: Any = None,
    backend: str = "auto",
    torch_mode: str = "forward",
    torch_dtype: Optional[torch.dtype] = None,
    torch_device: Any = None,
    fd_step: Optional[float] = None,
    max_evaluations: int = _DEFAULT_MAX_EVALUATIONS,
    max_seconds: Optional[float] = None,
    orthonormality_tol: Optional[float] = None,
    allow_train_mode: bool = False,
) -> SpectralAuditResult:
    """Project the input Jacobian of ``operator`` at ``base_point`` (default 0) onto ``modes``.

    ``operator`` maps a field of shape (M,) to a field of shape (M,): a numpy
    callable (``backend="numpy"``), a torch callable (``backend="torch"``) or
    an ``nn.Module`` (``"auto"`` picks torch for modules, numpy otherwise).
    ``modes`` is Q with shape (M, K) and W-orthonormal columns, ``weights`` the
    diagonal of W. ``oracle`` (K,) is the expected diagonal of S, if known.
    ``torch_device`` places the inputs of a torch callable (default CPU); a
    module always runs on its own device and rejects a conflicting override.

    Fail-closed: invalid input raises TypeError/ValueError; Q not W-orthonormal
    within ``orthonormality_tol`` raises ValueError; a non-finite operator
    output or Jacobian product raises NonFiniteOutputError; more than
    ``max_evaluations`` operator calls (checked BEFORE the first call) or more
    than ``max_seconds`` wall time raise BudgetExceededError. A module in
    training mode raises ValueError unless ``allow_train_mode`` (dropout and
    batch statistics would make J random and mutate buffers).

    Determinism preflight (2026-10-15): before any Jacobian work the operator
    is called twice at the base point (counted in ``evaluations`` and in the
    budget); the two outputs must agree bitwise and the global RNG state
    (torch, numpy, random, initialised CUDA) must be unchanged, otherwise
    NondeterministicOperatorError (a ValueError) is raised. ``.training`` alone
    does not prove determinism: an eval-mode module may still sample.
    """
    start = time.perf_counter()
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    if torch_mode not in _TORCH_MODES:
        raise ValueError(f"torch_mode must be one of {_TORCH_MODES}, got {torch_mode!r}")
    if not callable(operator):
        raise TypeError("operator must be callable")
    w = _check_weights(weights)
    q = _check_modes(modes, w.size)
    m, k = q.shape
    base = np.zeros(m) if base_point is None else _real_array(base_point, name="base_point")
    if base.shape != (m,):
        raise ValueError(f"base_point must have shape ({m},), got {base.shape}")
    o = None
    if oracle is not None:
        o = _real_array(oracle, name="oracle")
        if o.shape != (k,):
            raise ValueError(f"oracle must have shape ({k},), got {o.shape}")
    step = None if fd_step is None else _positive_finite(fd_step, name="fd_step")
    budget = _Budget(_positive_int(max_evaluations, name="max_evaluations"),
                     None if max_seconds is None else _positive_finite(max_seconds, name="max_seconds"))
    tol = default_orthonormality_tol(m) if orthonormality_tol is None else _positive_finite(
        orthonormality_tol, name="orthonormality_tol")
    defect = orthonormality_defect(q, w)
    if not defect <= tol:
        raise ValueError(f"modes are not W-orthonormal: max|Q^T W Q - I| = {defect:.3e} > {tol:.3e}")

    use_torch = backend == "torch" or (backend == "auto" and isinstance(operator, torch.nn.Module))
    mode_used: Optional[str] = None
    if use_torch:
        if isinstance(operator, torch.nn.Module) and operator.training and not allow_train_mode:
            raise ValueError("module is in training mode; call .eval() or pass allow_train_mode=True")
        dtype = _torch_dtype(operator, torch_dtype)
        device = _torch_device(operator, torch_device)
        budget.require(_REPEAT_EVALUATIONS + (k if torch_mode == "forward" else k + 1))
        _repeat_check(operator, base, m, True, dtype, device, budget)
        jq, rows = _jq_torch(operator, q, w, base, dtype, device, torch_mode, budget)
        mode_used = torch_mode
        dtype_name = str(dtype).replace("torch.", "")
        step = None
    else:
        if torch_dtype is not None:
            raise ValueError("torch_dtype applies to the torch backend only")
        budget.require(_REPEAT_EVALUATIONS + 2 * k + 1)
        _repeat_check(operator, base, m, False, None, None, budget)
        jq, step, dtype_name = _jq_numpy(operator, q, w, base, step, budget)
        rows = None

    if jq is not None:
        s = q.T @ (w[:, None] * jq)
        residual = jq - q @ s
        # Weighted column norms via a max-scaled sum: squaring entries above ~1e154
        # overflowed to inf and gave out_of_span = inf/inf = NaN (Codex P2 on #71).
        sw = np.sqrt(w)[:, None]
        num = _scaled_column_norms(sw * residual)
        den = _scaled_column_norms(sw * jq)
        with np.errstate(divide="ignore", invalid="ignore"):
            out_of_span = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)
    else:
        s = rows @ q
        out_of_span = None
    if not np.all(np.isfinite(s)):
        raise NonFiniteOutputError("projected Jacobian S is not finite")
    diag = np.diag(s).copy()
    off = s - np.diag(diag)
    total = float(_scaled_column_norms(s.reshape(-1, 1))[0])  # Frobenius, max-scaled
    coupling = float(_scaled_column_norms(off.reshape(-1, 1))[0]) / total if total > 0.0 else None
    # Fail closed on every derived metric, not only on S (Codex P2 on #71).
    if (out_of_span is not None and not np.all(np.isfinite(out_of_span))) or \
            not math.isfinite(total) or (coupling is not None and not math.isfinite(coupling)):
        raise NonFiniteOutputError("derived spectral metrics (out-of-span, coupling) are not finite")
    abs_err = rel_err = spectral = None
    if o is not None:
        abs_err, rel_err = _relative_errors(diag, o)
        o_norm = float(np.linalg.norm(o))
        diff = float(np.linalg.norm(s - np.diag(o)))
        spectral = diff / o_norm if o_norm > 0.0 else (0.0 if diff == 0.0 else math.inf)
    arrays = [a for a in (s, diag, out_of_span, o, abs_err, rel_err) if a is not None]
    for arr in arrays:
        arr.flags.writeable = False
    return SpectralAuditResult(
        projection=s,
        diagonal=diag,
        coupling_fraction=coupling,
        offdiag_abs_max=float(np.max(np.abs(off))),
        out_of_span=out_of_span,
        oracle=o,
        diag_abs_error=abs_err,
        diag_rel_error=rel_err,
        spectral_rel_error=spectral,
        orthonormality_defect=defect,
        orthonormality_tol=tol,
        backend="torch" if use_torch else "numpy",
        torch_mode=mode_used,
        dtype=dtype_name,
        fd_step=step,
        evaluations=budget.count,
        wall_time_s=time.perf_counter() - start,
    )


def _tolerance(value: Any) -> Optional[float]:
    """A usable tolerance, or None (-> ABSTAIN): missing, bool, non-real, NaN, inf or negative."""
    if value is None or isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0.0 else None


def audit_verdict(
    result: SpectralAuditResult,
    *,
    diag_rtol: Any,
    coupling_tol: Any,
    diag_atol: Any = 0.0,
    out_of_span_tol: Any = None,
) -> str:
    """"PASS", "FAIL" or "ABSTAIN" for an audit against its oracle.

    PASS needs every mode within ``|S_kk - o_k| <= diag_atol + diag_rtol |o_k|``
    (numpy.isclose form), ``coupling_fraction <= coupling_tol`` and, if
    ``out_of_span_tol`` is given, every out-of-span residual within it. Each
    acceptance is a comparison that must be TRUE, so NaN fails. ABSTAIN when
    there is no oracle, a tolerance is missing or not a finite value >= 0, or
    an out-of-span tolerance is given but the residual was not measured
    (reverse mode). A zero S has no coupling fraction and fails.
    """
    if not isinstance(result, SpectralAuditResult):
        raise TypeError("result must be a SpectralAuditResult")
    rtol, ctol, atol = _tolerance(diag_rtol), _tolerance(coupling_tol), _tolerance(diag_atol)
    if result.oracle is None or rtol is None or ctol is None or atol is None:
        return "ABSTAIN"
    otol = None
    if out_of_span_tol is not None:
        otol = _tolerance(out_of_span_tol)
        if otol is None or result.out_of_span is None:
            return "ABSTAIN"
    diag_ok = bool(np.all(result.diag_abs_error <= atol + rtol * np.abs(result.oracle)))
    coupling_ok = result.coupling_fraction is not None and result.coupling_fraction <= ctol
    span_ok = otol is None or bool(np.all(result.out_of_span <= otol))
    return "PASS" if diag_ok and coupling_ok and span_ok else "FAIL"
