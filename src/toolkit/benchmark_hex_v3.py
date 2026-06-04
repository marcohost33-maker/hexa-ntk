"""
benchmark_hex_v3.py
===================
Canonical benchmark metrics for HexaNTK PDE operator evaluation.

Metric conventions:
- nRMSE(t): normalized RMSE per rollout timestep (APEBench Eq.31)
- H1 RMSE: Sobolev H¹ RMSE (gradient error)
- H1 nRMSE: normalized H¹ RMSE
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


# ------------------------------------------------------------------
# Core metric functions
# ------------------------------------------------------------------


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
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )

    diff = pred - target
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
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )

    T = pred.shape[0]
    diff_flat = (pred - target).reshape(T, -1)
    target_flat = target.reshape(T, -1)

    numerator = torch.sqrt(torch.mean(diff_flat ** 2, dim=-1))
    denominator = torch.sqrt(torch.mean(target_flat ** 2, dim=-1)) + eps

    return numerator / denominator


def _hex_gradient_magnitude(
    field: torch.Tensor,
    hex_neighbors: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """
    Approximate gradient magnitude on hexagonal grid via finite differences.

    Uses the standard hex-FV stencil: for each node, compute differences
    to its 6 neighbors and estimate |∇u| ≈ sqrt(2/(3h²) * Σ(u_j - u_0)²).

    Parameters
    ----------
    field : shape [..., N, C]
        Field values at each node.
    hex_neighbors : shape [N, 6]
        Index array: hex_neighbors[i, j] = index of j-th neighbor of node i.
        Use -1 for boundary nodes with fewer than 6 neighbors.
    h : float
        Hex grid spacing.

    Returns
    -------
    torch.Tensor : shape [..., N, C]
        Approximate gradient magnitude at each node.
    """
    N = field.shape[-2]
    C = field.shape[-1]

    # Gather neighbor values: [..., N, 6, C]
    # Clamp -1 indices to 0, then mask
    valid_mask = hex_neighbors >= 0  # [N, 6]
    safe_idx = hex_neighbors.clamp(min=0)  # [N, 6]

    # field has shape [..., N, C], gather along N dimension
    batch_shape = field.shape[:-2]
    field_flat = field.reshape(-1, N, C)  # [B, N, C]
    B = field_flat.shape[0]

    # Expand safe_idx for gather: [N, 6] -> [B, N, 6, C]
    idx_expanded = safe_idx.unsqueeze(0).unsqueeze(-1).expand(B, N, 6, C)
    field_expanded = field_flat.unsqueeze(2).expand(B, N, 6, C)

    # Gather neighbor values
    neighbor_vals = torch.gather(
        field_flat.unsqueeze(2).expand(B, N, N, C),
        dim=2,
        index=idx_expanded,
    )
    # Actually, simpler: index directly
    neighbor_vals = field_flat[:, safe_idx, :]  # [B, N, 6, C]

    # Center values
    center_vals = field_flat.unsqueeze(2)  # [B, N, 1, C]

    # Differences
    diffs = neighbor_vals - center_vals  # [B, N, 6, C]

    # Mask invalid neighbors
    mask = valid_mask.unsqueeze(0).unsqueeze(-1).float()  # [1, N, 6, 1]
    diffs = diffs * mask

    # Count valid neighbors per node
    n_valid = valid_mask.sum(dim=-1, keepdim=True).unsqueeze(0).unsqueeze(-1).float().clamp(min=1)

    # Approximate |∇u|² ≈ (2/(3h²)) * mean(Δu²)
    # Using sum/n_valid instead of fixed /6 to handle boundary nodes
    grad_sq = (2.0 / (3.0 * h * h)) * torch.sum(diffs ** 2, dim=2) / n_valid.squeeze(2)

    grad_mag = torch.sqrt(grad_sq + 1e-12)

    return grad_mag.reshape(*batch_shape, N, C)


def compute_h1_rmse_per_step(
    pred: torch.Tensor,
    target: torch.Tensor,
    hex_neighbors: Optional[torch.Tensor] = None,
    h: float = 1.0,
) -> torch.Tensor:
    """
    Compute H¹ Sobolev RMSE per rollout timestep.

    H¹ error = sqrt(L² error² + gradient error²)

    If hex_neighbors is not provided, falls back to pure L² RMSE.

    Parameters
    ----------
    pred : shape [T, N, C] or [T, B, N, C]
    target : shape [T, N, C] or [T, B, N, C]
    hex_neighbors : shape [N, 6], optional
        Hex neighbor index array for gradient computation.
    h : float
        Hex grid spacing for gradient scaling.

    Returns
    -------
    torch.Tensor : shape [T]
        H¹ RMSE for each timestep.
    """
    l2_rmse = compute_rmse_per_step(pred, target)

    if hex_neighbors is None:
        return l2_rmse

    T = pred.shape[0]

    grad_errors = []
    for t in range(T):
        grad_pred = _hex_gradient_magnitude(pred[t], hex_neighbors, h)
        grad_target = _hex_gradient_magnitude(target[t], hex_neighbors, h)
        diff = (grad_pred - grad_target).reshape(-1)
        grad_rmse_t = torch.sqrt(torch.mean(diff ** 2))
        grad_errors.append(grad_rmse_t)

    grad_rmse = torch.stack(grad_errors)

    # H¹ = sqrt(L² + |∇|²)
    h1_rmse = torch.sqrt(l2_rmse ** 2 + grad_rmse ** 2)
    return h1_rmse


def compute_h1_nrmse_per_step(
    pred: torch.Tensor,
    target: torch.Tensor,
    hex_neighbors: Optional[torch.Tensor] = None,
    h: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute normalized H¹ RMSE per rollout timestep.

    H¹ nRMSE(t) = H¹ RMSE(t) / (H¹ norm of target(t) + eps)

    Parameters
    ----------
    pred, target : shape [T, N, C] or [T, B, N, C]
    hex_neighbors : optional [N, 6]
    h : hex grid spacing
    eps : regularization

    Returns
    -------
    torch.Tensor : shape [T]
    """
    h1_rmse = compute_h1_rmse_per_step(pred, target, hex_neighbors, h)

    # Compute H¹ norm of target
    T = target.shape[0]
    target_flat = target.reshape(T, -1)
    l2_norm = torch.sqrt(torch.mean(target_flat ** 2, dim=-1))

    if hex_neighbors is not None:
        grad_norms = []
        for t in range(T):
            grad_t = _hex_gradient_magnitude(target[t], hex_neighbors, h)
            grad_norms.append(torch.sqrt(torch.mean(grad_t.reshape(-1) ** 2)))
        grad_norm = torch.stack(grad_norms)
        h1_norm = torch.sqrt(l2_norm ** 2 + grad_norm ** 2)
    else:
        h1_norm = l2_norm

    return h1_rmse / (h1_norm + eps)


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
    exceeded = (nrmse_per_step > threshold).nonzero(as_tuple=False)
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
    if method == "geometric_mean":
        # geometric mean = exp(mean(log(x)))
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
    nrmse = compute_nrmse_per_step(pred, target, eps=eps)
    rmse = compute_rmse_per_step(pred, target)

    result: Dict[str, Any] = {
        "nrmse_per_step": nrmse.tolist(),
        "rmse_per_step": rmse.tolist(),
        "nrmse_mean": aggregate_over_time(nrmse, aggregation),
        "rmse_mean": aggregate_over_time(rmse, aggregation),
        "divergence_step": compute_divergence_step(nrmse, divergence_threshold),
        "num_steps": int(pred.shape[0]),
        "aggregation_method": aggregation,
        "divergence_threshold": divergence_threshold,
    }

    if hex_neighbors is not None:
        h1_rmse = compute_h1_rmse_per_step(pred, target, hex_neighbors, h)
        h1_nrmse = compute_h1_nrmse_per_step(pred, target, hex_neighbors, h, eps)

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
