"""Empirical parameter NTK of a model, measured in a W-weighted mode basis (PARAM_NTK_v1).

The parameter-space neural tangent kernel of an output field y(theta) is
Theta = J J^T with J = dy/dtheta. On a lattice the field is compared in the
discrete L2(W) inner product, so the kernel is reported in a W-orthonormal mode
basis phi_k (the same basis as spectral_audit_hex_v3):

    Theta_kl = < dP_k/dtheta , dP_l/dtheta >,   P_k(theta) = phi_k^T W y(theta)

One reverse-mode pass (VJP) per mode gives the row g_k = dP_k/dtheta. Complex
parameters (FNO spectral weights) are resolved into real coordinates
(torch.view_as_real), so Theta is the real Gram matrix over every real degree of
freedom and is symmetric positive semidefinite by construction.

Measurement policy (owner decision E1015M-03):
* float64 shim -- the kernel is measured on a float64/complex128 copy of the
  model, never on the caller's model, and assembled in float64;
* float32 cross-check -- the same kernel from the float32 model is reported as a
  relative Frobenius deviation, so rounding effects are visible, not hidden.

Fail-closed guards:
* every trainable parameter of the model must be measured (a partial parameter
  list would silently report the NTK of a sub-model);
* a directional finite-difference check of the projected output must agree with
  autograd, so a forward that detaches or swaps parameters outside autograd
  cannot produce a kernel that looks valid.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

POLICY = "PARAM_NTK_v1"
FD_REL_TOL = 1e-6  # float64 central difference vs autograd, relative to |J v|
# float32 models (backends without float64, e.g. neuraloperator 2.0.0 GINO): central
# differences at step ~ u^(1/3) have error ~ u^(2/3) ~ 1.5e-5; 1e-3 leaves margin and
# still refuses a detached parameter (relative error 1).
FD_REL_TOL_FLOAT32 = 1e-3
_FD_STEP = {torch.float64: 1e-6, torch.float32: 4e-3}


class ParamNTKError(ValueError):
    """The measurement is not a valid parameter NTK of the given model."""


@dataclass(frozen=True)
class ParamNTKResult:
    kernel: np.ndarray            # (K, K) float64, symmetric PSD
    eigenvalues: np.ndarray       # ascending, float64
    n_real_params: int            # real degrees of freedom (complex counted twice)
    fd_relative_error: float      # directional FD vs autograd (float64 model)
    float32_relative_deviation: float | None  # ||Theta32 - Theta64||_F / ||Theta64||_F
    policy: str = POLICY


def _trainable(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def _check_parameters(model: torch.nn.Module, params: Sequence[torch.Tensor] | None) -> list[torch.Tensor]:
    full = _trainable(model)
    if not full:
        raise ParamNTKError("model has no trainable parameters")
    if params is None:
        return full
    given = list(params)
    if len(given) != len(full) or any(a is not b for a, b in zip(given, full)):
        raise ParamNTKError(
            f"params must be exactly the model's {len(full)} trainable parameters in order "
            f"(got {len(given)}): a partial list measures the NTK of a sub-model"
        )
    return given


def _as_real(t: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(t).reshape(-1) if t.is_complex() else t.reshape(-1)


def _cast_copy(model: torch.nn.Module, real: torch.dtype, cplx: torch.dtype) -> torch.nn.Module:
    """A copy with every floating/complex parameter and buffer cast; the caller's model
    is never changed."""
    twin = copy.deepcopy(model)
    with torch.no_grad():
        for t in [*twin.parameters(), *twin.buffers()]:
            if t.is_complex():
                t.data = t.data.to(cplx)
            elif t.is_floating_point():
                t.data = t.data.to(real)
    return twin


def _projector(modes: Any, weights: Any) -> np.ndarray:
    phi = np.asarray(modes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if phi.ndim != 2 or w.shape != (phi.shape[0],):
        raise ValueError("modes must be (M, K) and weights (M,)")
    if not (np.all(np.isfinite(phi)) and np.all(np.isfinite(w)) and np.all(w > 0)):
        raise ValueError("modes must be finite and weights finite and positive")
    return (w[:, None] * phi).T  # (K, M): P = projector @ y


def _gram_rows(model, forward, projector) -> tuple[torch.Tensor, list[torch.Tensor]]:
    params = _trainable(model)
    y = forward(model)
    if not isinstance(y, torch.Tensor) or y.dim() != 1 or y.shape[0] != projector.shape[1]:
        raise ParamNTKError(f"forward must return a 1-D field of length {projector.shape[1]}")
    if not bool(torch.isfinite(y).all()):
        raise ParamNTKError("forward output must be finite")
    proj = torch.as_tensor(projector, dtype=y.dtype, device=y.device)
    p_all = proj @ y
    rows = []
    for k in range(p_all.shape[0]):
        grads = torch.autograd.grad(p_all[k], params, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([
            _as_real(g) if g is not None else torch.zeros(_as_real(p).numel(), dtype=_as_real(p).dtype,
                                                          device=p.device)
            for g, p in zip(grads, params)
        ]))
    return torch.stack(rows), params


def _fd_check(model, forward, projector, seed: int, step: float) -> float:
    """Central difference of one random projection along one random parameter direction."""
    gen = torch.Generator().manual_seed(seed)
    params = _trainable(model)
    coeff = torch.randn(projector.shape[0], generator=gen, dtype=torch.float64)
    proj = torch.as_tensor(projector, dtype=torch.float64)

    def scalar() -> torch.Tensor:
        return coeff @ (proj @ forward(model).to(torch.float64))

    directions = [torch.randn(p.shape, generator=gen, dtype=p.dtype) for p in params]
    value = scalar()
    grads = torch.autograd.grad(value, params, allow_unused=True)
    jv = sum(float((_as_real(g) * _as_real(d)).sum()) for g, d in zip(grads, directions) if g is not None)
    scale = math.sqrt(sum(float(_as_real(p.detach()).pow(2).sum()) for p in params)) or 1.0
    eps = step * scale / math.sqrt(sum(float(_as_real(d).pow(2).sum()) for d in directions))
    with torch.no_grad():
        for p, d in zip(params, directions):
            p.add_(eps * d)
        plus = float(scalar())
        for p, d in zip(params, directions):
            p.sub_(2 * eps * d)
        minus = float(scalar())
        for p, d in zip(params, directions):
            p.add_(eps * d)
    fd = (plus - minus) / (2 * eps)
    return abs(fd - jv) / max(abs(fd), abs(jv), 1e-300)


def param_ntk(
    model: torch.nn.Module,
    forward: Callable[[torch.nn.Module], torch.Tensor],
    modes: Any,
    weights: Any,
    *,
    params: Sequence[torch.Tensor] | None = None,
    float32_crosscheck: bool = True,
    model_dtype: str = "float64",
    seed: int = 0,
) -> ParamNTKResult:
    """Measure Theta in the W-weighted mode basis; ``forward(m)`` maps a model to its field (M,).

    ``model_dtype="float64"``: the model copy runs in float64/complex128 and the
    cross-check reruns it in float32. ``"float32"`` (backends without float64): the
    model copy runs in float32, the kernel is assembled from its gradients in
    float64, and the cross-check is the same Gram matrix assembled in float32.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if model_dtype not in ("float64", "float32"):
        raise ValueError("model_dtype must be 'float64' or 'float32'")
    _check_parameters(model, params)
    projector = _projector(modes, weights)

    if model_dtype == "float64":
        twin = _cast_copy(model, torch.float64, torch.complex128)
        step, tol = _FD_STEP[torch.float64], FD_REL_TOL
    else:
        twin = _cast_copy(model, torch.float32, torch.complex64)
        step, tol = _FD_STEP[torch.float32], FD_REL_TOL_FLOAT32
    g, twin_params = _gram_rows(twin, forward, projector)
    fd_error = _fd_check(twin, forward, projector, seed, step)
    if not fd_error <= tol:  # NaN fails closed
        raise ParamNTKError(
            f"autograd and finite differences disagree (relative {fd_error:.3e} > {tol}): "
            "the forward detaches or swaps parameters outside autograd"
        )
    g64 = g.detach().to(torch.float64)
    kernel = (g64 @ g64.T).cpu().numpy()
    kernel = 0.5 * (kernel + kernel.T)

    deviation = None
    if float32_crosscheck:
        if model_dtype == "float64":
            g32, _ = _gram_rows(_cast_copy(model, torch.float32, torch.complex64), forward, projector)
        else:
            g32 = g.detach()
        k32 = (g32 @ g32.T).to(torch.float64).cpu().numpy()
        norm = np.linalg.norm(kernel)
        deviation = float(np.linalg.norm(k32 - kernel) / norm) if norm > 0 else 0.0

    eigenvalues = np.linalg.eigvalsh(kernel)
    n_real = int(sum(_as_real(p).numel() for p in twin_params))
    return ParamNTKResult(kernel, eigenvalues, n_real, float(fd_error), deviation)
