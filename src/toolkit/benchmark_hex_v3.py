"""
benchmark_hex_v3.py
===================
Canonical benchmark metrics for HexaNTK PDE operator evaluation.

Metric conventions:
- Legacy nRMSE(t): historical ratio-of-aggregate-RMS semantics retained for reproducibility.
- MetricSpecV4 mean-nRMSE: normalize each sample first, then average samples (APEBench Eq.31).
- H1 RMSE: discrete hex H¹ proxy using the gradient seminorm of the error field.
- H1 nRMSE: normalized discrete H¹ proxy (not a continuum Sobolev certificate).
- Legacy divergence_step remains for old reports; v4 uses threshold_crossing_step.
- MetricSpecV4 forecast aggregation excludes t=0 when it is the shared initial state (APEBench Eq.32).

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
    "compute_mean_nrmse_per_step",
    "compute_rmse_per_step",
    "compute_h1_rmse_per_step",
    "compute_h1_nrmse_per_step",
    "compute_divergence_step",
    "compute_threshold_crossing_step",
    "aggregate_over_time",
    "aggregate_forecast_metric",
    "MetricSpecV4",
    "metric_spec_digest",
    "evaluate_rollout",
    "evaluate_rollout_v4",
    "preserve_legacy_report_without_relabeling",
    "BenchmarkReport",
]

import hashlib
import json
import math
import numbers
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

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
    # Exact Reals beyond binary64 range (Fraction(10**400), 10**400) raise
    # OverflowError instead of returning inf; keep the ValueError contract.
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
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
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and >= 0") from exc
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
    acc_dtype = _accumulation_dtype(pred.dtype, target.dtype)
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
    acc_dtype = _accumulation_dtype(pred.dtype, target.dtype)
    pred = pred.to(acc_dtype)
    target = target.to(acc_dtype)
    T = pred.shape[0]
    diff_flat = (pred - target).reshape(T, -1)
    target_flat = target.reshape(T, -1)

    numerator = torch.sqrt(torch.mean(diff_flat ** 2, dim=-1))
    denominator = torch.sqrt(torch.mean(target_flat ** 2, dim=-1)) + eps_value

    return numerator / denominator


def _accumulation_dtype(*dtypes: torch.dtype) -> torch.dtype:
    """Widest of all input dtypes, but at least float32.

    float16/bfloat16 -> float32 (same policy as torch.autocast, which runs
    sum/pow/norm/mse_loss in float32 because the half-precision range, fp16
    max 65504, is too small for sums of squares). Mixed inputs promote to the
    widest one, so a float64 target is never rounded down to a float32
    prediction's precision before the subtraction (Codex on 8b51cda).
    """
    result = torch.float32
    for dtype in dtypes:
        result = torch.promote_types(result, dtype)
    return result


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


def _compute_h1_error_prepared(
    pred: torch.Tensor,
    target: torch.Tensor,
    topology: _PreparedHexTopology,
) -> torch.Tensor:
    """Return only the per-step H1 error, without target-norm work."""
    acc_dtype = _accumulation_dtype(pred.dtype, target.dtype)
    h1_error_steps = []
    for t in range(pred.shape[0]):
        pred_t = pred[t].to(acc_dtype)
        target_t = target[t].to(acc_dtype)
        error_t = pred_t - target_t
        l2_error = torch.sqrt(torch.mean(error_t ** 2))
        grad_error = _hex_gradient_magnitude_prepared(error_t, topology)
        grad_error_rmse = torch.sqrt(torch.mean(grad_error ** 2))
        h1_error_steps.append(torch.sqrt(l2_error ** 2 + grad_error_rmse ** 2))
    return torch.stack(h1_error_steps)


def _compute_h1_terms_prepared(
    pred: torch.Tensor,
    target: torch.Tensor,
    topology: _PreparedHexTopology,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-step H1 error and target norm, one timestep at a time.

    The neighbour gather materialises [B, N, 6, C] temporaries; looping over
    the rollout axis keeps that peak independent of T (as on main before the
    H1 rework) instead of [T*B, N, 6, C]. The unnormalised H1 API uses the
    error-only helper above so it does not pay for an unused target gradient.
    """
    h1_error = _compute_h1_error_prepared(pred, target, topology)
    acc_dtype = _accumulation_dtype(pred.dtype, target.dtype)
    h1_target_steps = []
    for t in range(target.shape[0]):
        target_t = target[t].to(acc_dtype)
        l2_target = torch.sqrt(torch.mean(target_t ** 2))
        grad_target = _hex_gradient_magnitude_prepared(target_t, topology)
        grad_target_rmse = torch.sqrt(torch.mean(grad_target ** 2))
        h1_target_steps.append(
            torch.sqrt(l2_target ** 2 + grad_target_rmse ** 2)
        )
    return h1_error, torch.stack(h1_target_steps)


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
    return _compute_h1_error_prepared(pred, target, topology)


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
# MetricSpec v4: explicit sample reduction + forecast aggregation
# ------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpecV4:
    """Versioned scientific metric contract.

    sample_reduction='mean_of_ratios' normalizes each batch sample by its own
    reference norm before averaging samples. exclude_initial_state=True excludes
    rollout index 0 from time aggregation when it is the shared initial condition.
    """

    name: str = "mean_nrmse"
    version: str = "4.0"
    sample_reduction: str = "mean_of_ratios"
    time_aggregation: str = "geometric_mean"
    exclude_initial_state: bool = True
    zero_reference_policy: str = "error"

    def __post_init__(self) -> None:
        # Identity fields are part of the digest; a report must not declare a
        # metric or version other than the one it computes (Codex #52 T2).
        if self.name != "mean_nrmse":
            raise ValueError("MetricSpecV4 requires name='mean_nrmse'")
        if self.version != "4.0":
            raise ValueError("MetricSpecV4 requires version='4.0'")
        # A truthy non-bool ("false", 1) would exclude index 0 while the
        # serialized spec still shows the raw value (Codex PR #58).
        if type(self.exclude_initial_state) is not bool:
            raise TypeError(
                "MetricSpecV4.exclude_initial_state must be a bool, got "
                f"{type(self.exclude_initial_state).__name__} "
                f"{self.exclude_initial_state!r}"
            )
        if self.sample_reduction != "mean_of_ratios":
            raise ValueError("MetricSpecV4 requires sample_reduction='mean_of_ratios'")
        if self.time_aggregation not in {"geometric_mean", "arithmetic_mean", "max"}:
            raise ValueError("unsupported MetricSpecV4 time_aggregation")
        if self.zero_reference_policy not in {"error", "zero_if_both_zero"}:
            raise ValueError("unsupported MetricSpecV4 zero_reference_policy")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def metric_spec_digest(spec: MetricSpecV4) -> str:
    payload = json.dumps(
        spec.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _v4_promote_single_sample(field: Any) -> Any:
    """MetricSpec v4 accepts a single-sample rollout [T, DOF]; it is the
    explicit [T, 1, DOF] form (spec/METRIC_SPEC_v4.md section 2). Other ranks
    pass through unchanged and are validated by _validate_rollout_pair."""
    if torch.is_tensor(field) and field.ndim == 2:
        return field.unsqueeze(1)
    return field


def _as_t_b_dof(field: torch.Tensor) -> torch.Tensor:
    if field.ndim == 3:
        return field.unsqueeze(1).reshape(field.shape[0], 1, -1)
    return field.reshape(field.shape[0], field.shape[1], -1)


def compute_mean_nrmse_per_step(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    zero_reference_policy: str = "error",
) -> torch.Tensor:
    """Return mean-of-per-sample relative L2 error for each timestep."""
    pred = _v4_promote_single_sample(pred)
    target = _v4_promote_single_sample(target)
    _validate_rollout_pair(pred, target)
    if zero_reference_policy not in {"error", "zero_if_both_zero"}:
        raise ValueError("unsupported zero_reference_policy")
    dtype = _accumulation_dtype(pred.dtype, target.dtype)
    p = _as_t_b_dof(pred.to(dtype))
    y = _as_t_b_dof(target.to(dtype))
    err_norm = torch.linalg.vector_norm(p - y, dim=-1)
    ref_norm = torch.linalg.vector_norm(y, dim=-1)
    zero_ref = ref_norm == 0

    if bool(zero_ref.any().item()):
        if zero_reference_policy == "error":
            loc = zero_ref.nonzero(as_tuple=False)[0].tolist()
            raise ZeroDivisionError(
                f"mean nRMSE undefined for zero reference norm at [t,b]={loc}"
            )
        bad = zero_ref & (err_norm != 0)
        if bool(bad.any().item()):
            loc = bad.nonzero(as_tuple=False)[0].tolist()
            raise ZeroDivisionError(
                f"mean nRMSE undefined for nonzero error over zero reference at [t,b]={loc}"
            )
        ratios = torch.where(
            zero_ref,
            torch.zeros_like(err_norm),
            err_norm / ref_norm,
        )
    else:
        ratios = err_norm / ref_norm
    return ratios.mean(dim=1)


def _geometric_mean_nonnegative(values: torch.Tensor) -> float:
    _validate_nonnegative_metric_vector(values, name="values")
    if bool((values == 0).any().item()):
        return 0.0
    return float(torch.exp(torch.mean(torch.log(values))).item())


def aggregate_forecast_metric(
    metric_per_step: torch.Tensor,
    *,
    method: str = "geometric_mean",
    exclude_initial_state: bool = True,
) -> float:
    _validate_nonnegative_metric_vector(metric_per_step, name="metric_per_step")
    values = metric_per_step[1:] if exclude_initial_state else metric_per_step
    if values.numel() == 0:
        raise ValueError("no forecast timesteps remain after excluding initial state")
    if method == "geometric_mean":
        return _geometric_mean_nonnegative(values)
    if method == "arithmetic_mean":
        return float(torch.mean(values).item())
    if method == "max":
        return float(torch.max(values).item())
    raise ValueError(f"Unknown aggregation method: {method}")


def compute_threshold_crossing_step(
    metric_per_step: torch.Tensor,
    threshold: float = 1.0,
) -> Optional[int]:
    """Return first strict threshold crossing, or None if no crossing occurs."""
    _validate_nonnegative_metric_vector(metric_per_step, name="metric_per_step")
    threshold_value = _validate_nonnegative_finite(threshold, name="threshold")
    exceeded = (metric_per_step > threshold_value).nonzero(as_tuple=False)
    return None if exceeded.numel() == 0 else int(exceeded[0, 0].item())


def metric_semantics_label(spec: MetricSpecV4) -> str:
    """Provenance label derived from the spec, never hard-coded (Codex #52 T5).

    APEBench Eq. 32 is the geometric mean over forecast steps; only that
    combination carries the Eq32 label.
    """
    steps = "forecast_only" if spec.exclude_initial_state else "all_steps"
    time_part = f"{spec.time_aggregation}_{steps}_time_aggregation"
    if spec.time_aggregation == "geometric_mean" and spec.exclude_initial_state:
        time_part = "Eq32_" + time_part
    else:
        time_part = "not_Eq32_" + time_part
    return f"APEBench_Eq31_mean_of_sample_ratios; {time_part}"


def evaluate_rollout_v4(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    spec: Optional[MetricSpecV4] = None,
    threshold: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate a rollout under MetricSpec v4 (spec/METRIC_SPEC_v4.md).

    With ``exclude_initial_state=True`` index 0 is dropped from time
    aggregation, which is only valid if index 0 is the shared initial
    condition. That is checked, fail-closed: ``pred[0]`` must equal
    ``target[0]`` exactly, otherwise ValueError (Codex #52 T1). Forecast-only
    rollouts pass ``MetricSpecV4(exclude_initial_state=False)``.
    """
    spec = spec or MetricSpecV4()
    pred = _v4_promote_single_sample(pred)
    target = _v4_promote_single_sample(target)
    mean_nrmse = compute_mean_nrmse_per_step(
        pred,
        target,
        zero_reference_policy=spec.zero_reference_policy,
    )
    if spec.exclude_initial_state and not torch.equal(pred[0], target[0]):
        raise ValueError(
            "MetricSpecV4.exclude_initial_state=True drops index 0, which requires "
            "a shared initial state (pred[0] == target[0] exactly); index 0 of this "
            "rollout differs. For forecast-only rollouts use "
            "MetricSpecV4(exclude_initial_state=False)."
        )
    rmse = compute_rmse_per_step(pred, target)
    return {
        "metric_spec": spec.to_dict(),
        "metric_spec_digest": metric_spec_digest(spec),
        "metric_spec_version": spec.version,
        "metric_semantics": metric_semantics_label(spec),
        "mean_nrmse_per_step": [float(x) for x in mean_nrmse.tolist()],
        "rmse_per_step": [float(x) for x in rmse.tolist()],
        "aggregated_mean_nrmse": aggregate_forecast_metric(
            mean_nrmse,
            method=spec.time_aggregation,
            exclude_initial_state=spec.exclude_initial_state,
        ),
        "aggregated_rmse": aggregate_forecast_metric(
            rmse,
            method=spec.time_aggregation,
            exclude_initial_state=spec.exclude_initial_state,
        ),
        "threshold_crossing_step": compute_threshold_crossing_step(
            mean_nrmse,
            threshold,
        ),
        "threshold": float(threshold),
        "num_timesteps": int(pred.shape[0]),
    }


def preserve_legacy_report_without_relabeling(
    report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Preserve legacy aggregates without pretending they are MetricSpec-v4 values."""
    metrics_obj = report.get("metrics", {})
    metrics = metrics_obj if isinstance(metrics_obj, Mapping) else {}
    legacy_step = metrics.get("divergence_step")
    legacy_num_steps = metrics.get("num_steps")
    no_crossing = (
        None
        if legacy_step is None or legacy_num_steps is None
        else legacy_step == legacy_num_steps
    )
    return {
        "migration_status": "legacy_non_convertible_without_per_sample_raw_values",
        "legacy_metric_spec": "hexantk_v3_ratio_of_aggregate_rms",
        "legacy_nrmse_mean": metrics.get("nrmse_mean"),
        "legacy_divergence_step": legacy_step,
        # Parameters needed to interpret the copied values (Codex #52 T3).
        # Legacy divergence_step == num_steps is the old no-crossing sentinel.
        "legacy_num_steps": legacy_num_steps,
        "legacy_aggregation_method": metrics.get("aggregation_method"),
        "legacy_divergence_threshold": metrics.get("divergence_threshold"),
        "legacy_divergence_step_is_no_crossing_sentinel": no_crossing,
        "corrected_aggregated_mean_nrmse": None,
        "threshold_crossing_step": None,
    }

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
        if "aggregated_mean_nrmse" in m:  # MetricSpec v4 result (Codex #52 T4)
            value = m["aggregated_mean_nrmse"]
            crossing = m.get("threshold_crossing_step")
            parts = [
                f"model={self.model_name}",
                f"dataset={self.dataset_name}",
                f"mean_nRMSE_v{m.get('metric_spec_version', '?')}={value:.4f}"
                if isinstance(value, float) else "",
                f"crossing_step={'none' if crossing is None else crossing}",
            ]
            return " | ".join(p for p in parts if p)
        parts = [
            f"model={self.model_name}",
            f"dataset={self.dataset_name}",
            f"nRMSE={m.get('nrmse_mean', '?'):.4f}" if isinstance(m.get("nrmse_mean"), float) else "",
            f"div_step={m.get('divergence_step', '?')}",
        ]
        if "h1_nrmse_mean" in m:
            parts.append(f"H1_nRMSE={m['h1_nrmse_mean']:.4f}")
        return " | ".join(p for p in parts if p)
