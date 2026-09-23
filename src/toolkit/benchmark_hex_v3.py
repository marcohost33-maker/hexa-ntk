"""
benchmark_hex_v3.py
===================
Canonical benchmark metrics for HexaNTK PDE operator evaluation.

Metric conventions:
- nRMSE(t): normalized RMSE per rollout timestep (APEBench Eq.31)
- H1 RMSE: discrete hex H¹ proxy using the gradient seminorm of the error field
- H1 nRMSE: normalized discrete H¹ proxy (not a continuum Sobolev certificate)
- divergence_step: first timestep where nRMSE exceeds threshold
- Aggregation over time: geometric mean (APEBench Eq.32)

References:
- APEBench (NeurIPS 2024 D&B): nRMSE, Sobolev metrics, rollout evaluation
- PDEBench (NeurIPS 2022): normalized metrics, trajectory evaluation
- HexaNTK review: h1_rmse_mean, h1_nrmse_mean as canonical metric names

Design:
- Pure functions operating on tensors (no model dependency)
- Structured BenchmarkReport for JSON serialization
- Hex-Laplacian-aware H1 approximation via finite differences on hex grid
"""
from __future__ import annotations

__all__ = [
    "compute_nrmse_per_step",
    "compute_rmse_per_step",
    "compute_h1_rmse_per_step",
    "compute_h1_nrmse_per_step",
    "compute_divergence_step",
    "aggregate_over_time",
    "evaluate_rollout",
    "BenchmarkReport",
]

import json
import math
import numbers
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


# ------------------------------------------------------------------
# Core metric functions
# ------------------------------------------------------------------


def _validate_rollout_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    """Fail closed on rollout tensors that cannot yield meaningful real metrics."""
    if not torch.is_tensor(pred) or not torch.is_tensor(target):
        raise TypeError("pred and target must be torch.Tensor instances")
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )
    if pred.ndim not in {3, 4}:
        raise ValueError(
            "pred and target must have shape [T, N, C] or [T, B, N, C]"
        )
    if any(dim <= 0 for dim in pred.shape):
        raise ValueError("pred and target dimensions must all be non-empty")
    if pred.device != target.device:
        raise ValueError("pred and target must be on the same device")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError(
            "benchmark metrics currently support real floating tensors only; "
            "complex-valued fields require an explicit complex-field metric"
        )
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("pred and target must contain only finite values")


def _coerce_real_scalar(value: Any, *, name: str) -> float:
    """Normalize Python/NumPy real scalars without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_positive_finite(value: Any, *, name: str) -> float:
    result = _coerce_real_scalar(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _validate_nonnegative_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _validate_nonnegative_metric_vector(
    metric_per_step: torch.Tensor,
    *,
    name: str,
) -> None:
    if not torch.is_tensor(metric_per_step):
        raise TypeError(f"{name} must be a torch.Tensor")
    if metric_per_step.ndim != 1 or metric_per_step.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 1D tensor")
    if not metric_per_step.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if not torch.isfinite(metric_per_step).all():
        raise ValueError(f"{name} must contain only finite values")
    if torch.any(metric_per_step < 0):
        raise ValueError(f"{name} must be non-negative")


def _validate_real_field(field: torch.Tensor) -> None:
    if not torch.is_tensor(field):
        raise TypeError("field must be a torch.Tensor")
    if field.ndim < 2 or not field.is_floating_point():
        raise TypeError("field must be a real floating tensor with node/channel axes")
    if not torch.isfinite(field).all():
        raise ValueError("field must contain only finite values")


@dataclass(frozen=True)
class _PreparedHexTopology:
    """Validated topology tensors reusable across an entire rollout."""

    safe_idx: torch.Tensor
    valid_mask: torch.Tensor
    n_valid: torch.Tensor
    n_nodes: int
    h: float


def _prepare_hex_topology(
    field: torch.Tensor,
    hex_neighbors: torch.Tensor,
    h: Any,
    *,
    validate_field: bool = True,
) -> _PreparedHexTopology:
    """Validate/normalize a regular six-neighbour topology exactly once.

    ``validate_field=False`` is only for callers that already ran
    ``_validate_rollout_pair`` on this exact tensor (evaluate_rollout); it
    skips a redundant full-tensor finiteness scan, not the topology checks.
    """
    if validate_field:
        _validate_real_field(field)
    h_value = _validate_positive_finite(h, name="h")
    if not torch.is_tensor(hex_neighbors):
        raise TypeError("hex_neighbors must be a torch.Tensor")

    n_nodes = field.shape[-2]
    if hex_neighbors.shape != (n_nodes, 6):
        raise ValueError(
            f"hex_neighbors must have shape [{n_nodes}, 6], "
            f"got {tuple(hex_neighbors.shape)}"
        )
    if hex_neighbors.dtype not in {torch.int32, torch.int64}:
        raise TypeError("hex_neighbors must use torch.int32 or torch.int64")

    neighbors = hex_neighbors.to(device=field.device, dtype=torch.long)
    if torch.any(neighbors < -1) or torch.any(neighbors >= n_nodes):
        raise ValueError("hex_neighbors entries must be -1 or valid node indices")

    valid_mask = neighbors >= 0
    if torch.any(valid_mask.sum(dim=-1) == 0):
        raise ValueError("every hex node must have at least one valid neighbour")

    node_ids = torch.arange(n_nodes, device=field.device).unsqueeze(1)
    if torch.any(valid_mask & (neighbors == node_ids)):
        raise ValueError("hex topology contains a self-neighbour")

    # Detect duplicates on-device. Invalid -1 slots are replaced by a unique
    # sentinel outside the valid node range before sorting.
    sentinel = torch.full_like(neighbors, n_nodes)
    sortable = torch.where(valid_mask, neighbors, sentinel)
    sorted_neighbors = torch.sort(sortable, dim=1).values
    duplicate_pairs = (
        (sorted_neighbors[:, 1:] == sorted_neighbors[:, :-1])
        & (sorted_neighbors[:, 1:] != n_nodes)
    )
    if torch.any(duplicate_pairs):
        raise ValueError("hex topology contains duplicate neighbours")

    return _PreparedHexTopology(
        safe_idx=neighbors.clamp(min=0),
        valid_mask=valid_mask,
        n_valid=valid_mask.sum(dim=-1),
        n_nodes=n_nodes,
        h=h_value,
    )


def compute_rmse_per_step(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute RMSE per rollout timestep.

    Parameters
    ----------
    pred : shape [T, N, C] or [T, B, N, C]
        Predicted fields over T timesteps.
    target : shape [T, N, C] or [T, B, N, C]
        Ground truth fields.

    Returns
    -------
    torch.Tensor : shape [T]
        RMSE for each timestep, averaged over batch/nodes/channels.
    """
    _validate_rollout_pair(pred, target)
    return _rmse_per_step_unchecked(pred, target)


def _rmse_per_step_unchecked(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """RMSE kernel for callers that already validated the rollout pair.

    Half-precision inputs are accumulated in float32 (``_accumulation_dtype``).
    """
    acc_dtype = _accumulation_dtype(pred.dtype)
    diff = pred.to(acc_dtype) - target.to(acc_dtype)
    # Flatten all dims except T
    T = diff.shape[0]
    diff_flat = diff.reshape(T, -1)
    rmse = torch.sqrt(torch.mean(diff_flat ** 2, dim=-1))
    return rmse


def compute_nrmse_per_step(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute normalized RMSE per rollout timestep.

    nRMSE(t) = ||pred(t) - target(t)||_2 / (||target(t)||_2 + eps)

    This is the standard metric used in APEBench (Eq.31) and PDEBench.

    Parameters
    ----------
    pred : shape [T, N, C] or [T, B, N, C]
    target : shape [T, N, C] or [T, B, N, C]
    eps : small constant to avoid division by zero

    Returns
    -------
    torch.Tensor : shape [T]
    """
    _validate_rollout_pair(pred, target)
    eps_value = _validate_positive_finite(eps, name="eps")
    return _nrmse_per_step_unchecked(pred, target, eps_value)


def _nrmse_per_step_unchecked(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps_value: float,
) -> torch.Tensor:
    """nRMSE kernel for callers that already validated pair and eps.

    Half-precision inputs are accumulated in float32 (``_accumulation_dtype``);
    in fp16 the default eps=1e-8 would itself round to zero.
    """
    acc_dtype = _accumulation_dtype(pred.dtype)
    pred = pred.to(acc_dtype)
    target = target.to(acc_dtype)
    T = pred.shape[0]
    diff_flat = (pred - target).reshape(T, -1)
    target_flat = target.reshape(T, -1)

    numerator = torch.sqrt(torch.mean(diff_flat ** 2, dim=-1))
    denominator = torch.sqrt(torch.mean(target_flat ** 2, dim=-1)) + eps_value

    return numerator / denominator


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """float16/bfloat16 -> float32, float32/float64 unchanged.

    Same policy as torch.autocast, which runs sum/pow/norm/mse_loss in
    float32 because the half-precision range (fp16 max 65504) is too small
    for sums of squares.
    """
    return torch.promote_types(dtype, torch.float32)


def _hex_gradient_magnitude_prepared(
    field: torch.Tensor,
    topology: _PreparedHexTopology,
) -> torch.Tensor:
    """Directional gradient proxy using prevalidated topology.

    The stencil is accumulated in ``_accumulation_dtype(field.dtype)``: a
    neighbour difference of 260 is representable in fp16, its square is not.
    The result keeps the accumulation dtype (float32 for half inputs).
    """
    if field.shape[-2] != topology.n_nodes:
        raise ValueError(
            f"field has {field.shape[-2]} nodes but topology has "
            f"{topology.n_nodes}"
        )

    acc_dtype = _accumulation_dtype(field.dtype)
    n_nodes = topology.n_nodes
    n_channels = field.shape[-1]
    batch_shape = field.shape[:-2]
    field_flat = field.reshape(-1, n_nodes, n_channels).to(acc_dtype)

    neighbor_vals = field_flat[:, topology.safe_idx, :]
    center_vals = field_flat.unsqueeze(2)
    diffs = neighbor_vals - center_vals

    mask = topology.valid_mask.unsqueeze(0).unsqueeze(-1).to(acc_dtype)
    diffs = diffs * mask
    n_valid = topology.n_valid.unsqueeze(0).unsqueeze(-1).to(acc_dtype)

    grad_sq = (
        (2.0 / (topology.h * topology.h))
        * torch.sum(diffs ** 2, dim=2)
        / n_valid
    )

    # grad_sq is a sum-of-squares estimator and therefore non-negative.
    # Do NOT add an epsilon here: an exactly constant/error-zero field must
    # retain an exactly zero H1 seminorm.
    grad_mag = torch.sqrt(torch.clamp(grad_sq, min=0.0))
    return grad_mag.reshape(*batch_shape, n_nodes, n_channels)


def _hex_gradient_magnitude(
    field: torch.Tensor,
    hex_neighbors: torch.Tensor,
    h: Any,
) -> torch.Tensor:
    """Validated convenience wrapper around the prepared-topology kernel."""
    topology = _prepare_hex_topology(field, hex_neighbors, h)
    return _hex_gradient_magnitude_prepared(field, topology)


def _compute_h1_terms_prepared(
    pred: torch.Tensor,
    target: torch.Tensor,
    topology: _PreparedHexTopology,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-step H1 error and target norm, one timestep at a time.

    The neighbour gather materialises [B, N, 6, C] temporaries; looping over
    the rollout axis keeps that peak independent of T (as on main before the
    H1 rework) instead of [T*B, N, 6, C]. Accumulation dtype as in
    ``_accumulation_dtype``; the returned [T] tensors use it as well.
    """
    acc_dtype = _accumulation_dtype(pred.dtype)
    h1_error_steps = []
    h1_target_steps = []
    for t in range(pred.shape[0]):
        pred_t = pred[t].to(acc_dtype)
        target_t = target[t].to(acc_dtype)
        error_t = pred_t - target_t

        l2_error = torch.sqrt(torch.mean(error_t ** 2))
        grad_error = _hex_gradient_magnitude_prepared(error_t, topology)
        grad_error_rmse = torch.sqrt(torch.mean(grad_error ** 2))
        h1_error_steps.append(torch.sqrt(l2_error ** 2 + grad_error_rmse ** 2))

        l2_target = torch.sqrt(torch.mean(target_t ** 2))
        grad_target = _hex_gradient_magnitude_prepared(target_t, topology)
        grad_target_rmse = torch.sqrt(torch.mean(grad_target ** 2))
        h1_target_steps.append(
            torch.sqrt(l2_target ** 2 + grad_target_rmse ** 2)
        )

    return torch.stack(h1_error_steps), torch.stack(h1_target_steps)


def compute_h1_rmse_per_step(
    pred: torch.Tensor,
    target: torch.Tensor,
    hex_neighbors: Optional[torch.Tensor] = None,
    h: Any = 1.0,
) -> torch.Tensor:
    """Compute the discrete H1 error proxy per rollout timestep."""
    _validate_rollout_pair(pred, target)
    if hex_neighbors is None:
        return _rmse_per_step_unchecked(pred, target)

    topology = _prepare_hex_topology(pred, hex_neighbors, h, validate_field=False)
    h1_error, _ = _compute_h1_terms_prepared(pred, target, topology)
    return h1_error


def compute_h1_nrmse_per_step(
    pred: torch.Tensor,
    target: torch.Tensor,
    hex_neighbors: Optional[torch.Tensor] = None,
    h: Any = 1.0,
    eps: Any = 1e-8,
) -> torch.Tensor:
    """Compute normalized discrete H1 error per rollout timestep."""
    _validate_rollout_pair(pred, target)
    eps_value = _validate_positive_finite(eps, name="eps")

    if hex_neighbors is None:
        return _nrmse_per_step_unchecked(pred, target, eps_value)

    topology = _prepare_hex_topology(pred, hex_neighbors, h, validate_field=False)
    h1_error, h1_target = _compute_h1_terms_prepared(pred, target, topology)
    return h1_error / (h1_target + eps_value)


def compute_divergence_step(
    nrmse_per_step: torch.Tensor,
    threshold: float = 1.0,
) -> int:
    """
    Find the first timestep where nRMSE exceeds the threshold.

    Parameters
    ----------
    nrmse_per_step : shape [T]
    threshold : divergence threshold (default 1.0 = 100% relative error)

    Returns
    -------
    int
        First timestep index where nRMSE > threshold.
        Returns T (total steps) if no divergence detected.
    """
    _validate_nonnegative_metric_vector(nrmse_per_step, name="nrmse_per_step")
    threshold_value = _validate_nonnegative_finite(threshold, name="threshold")

    exceeded = (nrmse_per_step > threshold_value).nonzero(as_tuple=False)
    if exceeded.numel() == 0:
        return int(nrmse_per_step.shape[0])
    return int(exceeded[0, 0].item())


def aggregate_over_time(
    metric_per_step: torch.Tensor,
    method: str = "geometric_mean",
) -> float:
    """
    Aggregate a per-step metric over time.

    Parameters
    ----------
    metric_per_step : shape [T]
    method :
        "geometric_mean" (APEBench convention, Eq.32)
        "arithmetic_mean"
        "max"

    Returns
    -------
    float
    """
    _validate_nonnegative_metric_vector(metric_per_step, name="metric_per_step")

    if method == "geometric_mean":
        # APEBench Eq.32 uses log aggregation. Exact zeros make log undefined;
        # clamp only those valid zero values to a documented numerical floor.
        log_vals = torch.log(metric_per_step.clamp(min=1e-12))
        return float(torch.exp(torch.mean(log_vals)).item())
    elif method == "arithmetic_mean":
        return float(torch.mean(metric_per_step).item())
    elif method == "max":
        return float(torch.max(metric_per_step).item())
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ------------------------------------------------------------------
# Full rollout evaluation
# ------------------------------------------------------------------


def evaluate_rollout(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    hex_neighbors: Optional[torch.Tensor] = None,
    h: float = 1.0,
    divergence_threshold: float = 1.0,
    aggregation: str = "geometric_mean",
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Evaluate a full rollout with all canonical metrics.

    Parameters
    ----------
    pred : shape [T, N, C] or [T, B, N, C]
    target : shape [T, N, C] or [T, B, N, C]
    hex_neighbors : optional [N, 6] for H¹ metrics
    h : hex grid spacing
    divergence_threshold : nRMSE threshold for divergence detection
    aggregation : time aggregation method
    eps : regularization constant

    Returns
    -------
    Dict with:
        nrmse_per_step: list[float]
        rmse_per_step: list[float]
        h1_rmse_per_step: list[float] (if hex_neighbors provided)
        h1_nrmse_per_step: list[float] (if hex_neighbors provided)
        nrmse_mean: float (time-aggregated)
        rmse_mean: float
        h1_rmse_mean: float (if hex_neighbors)
        h1_nrmse_mean: float (if hex_neighbors)
        divergence_step: int
        num_steps: int
        aggregation_method: str
    """
    eps_value = _validate_positive_finite(eps, name="eps")
    divergence_threshold_value = _validate_nonnegative_finite(
        divergence_threshold,
        name="divergence_threshold",
    )
    # Validate the rollout pair ONCE (full-tensor finiteness scans cost a
    # device sync each); every kernel below reuses the validated tensors.
    _validate_rollout_pair(pred, target)
    nrmse = _nrmse_per_step_unchecked(pred, target, eps_value)
    rmse = _rmse_per_step_unchecked(pred, target)

    result: Dict[str, Any] = {
        "nrmse_per_step": nrmse.tolist(),
        "rmse_per_step": rmse.tolist(),
        "nrmse_mean": aggregate_over_time(nrmse, aggregation),
        "rmse_mean": aggregate_over_time(rmse, aggregation),
        "divergence_step": compute_divergence_step(
            nrmse,
            divergence_threshold_value,
        ),
        "num_steps": int(pred.shape[0]),
        "aggregation_method": aggregation,
        "divergence_threshold": divergence_threshold_value,
    }

    if hex_neighbors is not None:
        # Prepare/validate the topology once for the whole rollout and reuse it
        # for both error and target-gradient terms.
        topology = _prepare_hex_topology(
            pred, hex_neighbors, h, validate_field=False
        )
        h1_rmse, h1_target = _compute_h1_terms_prepared(
            pred,
            target,
            topology,
        )
        h1_nrmse = h1_rmse / (h1_target + eps_value)

        result["h1_rmse_per_step"] = h1_rmse.tolist()
        result["h1_nrmse_per_step"] = h1_nrmse.tolist()
        result["h1_rmse_mean"] = aggregate_over_time(h1_rmse, aggregation)
        result["h1_nrmse_mean"] = aggregate_over_time(h1_nrmse, aggregation)

    return result


# ------------------------------------------------------------------
# Structured report
# ------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    """
    Structured benchmark report for JSON serialization and comparison.

    Attributes
    ----------
    model_name : identifier of the evaluated model
    dataset_name : identifier of the evaluation dataset
    metrics : output from evaluate_rollout()
    config : optional model/training configuration snapshot
    metadata : optional extra metadata (git hash, timestamp, etc.)
    """

    model_name: str
    dataset_name: str
    metrics: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Union[str, Path]) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        return str(path)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "BenchmarkReport":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            model_name=data["model_name"],
            dataset_name=data["dataset_name"],
            metrics=data["metrics"],
            config=data.get("config"),
            metadata=data.get("metadata"),
        )

    def summary(self) -> str:
        """One-line summary string."""
        m = self.metrics
        parts = [
            f"model={self.model_name}",
            f"dataset={self.dataset_name}",
            f"nRMSE={m.get('nrmse_mean', '?'):.4f}" if isinstance(m.get("nrmse_mean"), float) else "",
            f"div_step={m.get('divergence_step', '?')}",
        ]
        if "h1_nrmse_mean" in m:
            parts.append(f"H1_nRMSE={m['h1_nrmse_mean']:.4f}")
        return " | ".join(p for p in parts if p)
