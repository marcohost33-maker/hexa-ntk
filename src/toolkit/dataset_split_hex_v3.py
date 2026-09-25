"""
dataset_split_hex_v3.py
=======================
Leakage-safe dataset splitting for HexaNTK operator learning.

Key design decisions:
- Non-mutating: SampleSubsetDataset wraps without modifying base dataset
- Group-based: splits by trajectory_id, not individual samples
- Optional stricter grouping by (trajectory_id, dt)
- Optional dt rounding for floating-point normalization
- Optional deterministic override for paper-reproducible splits
- Attribute passthrough for grid, config, etc.

Prevents data leakage by ensuring:
- No trajectory appears in both train and val splits
- No temporal correlation across the split boundary
"""
from __future__ import annotations

__all__ = [
    "TrajectoryGroupKey",
    "SampleSubsetDataset",
    "split_dataset_by_trajectory",
    "summarize_group_split",
]

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------------
# Group key
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryGroupKey:
    """
    Group key for leakage-safe grouped splitting.

    Default: trajectory_id only
    Optional stricter mode: (trajectory_id, dt)
    """

    trajectory_id: int
    dt: Optional[float] = None


# ------------------------------------------------------------------
# Dataset wrapper
# ------------------------------------------------------------------


class SampleSubsetDataset(Dataset):
    """
    Non-mutating subset wrapper over an existing dataset.

    Notes
    -----
    - __getitem__ delegates to the base dataset (canonical data path).
    - .samples is a projected metadata view only.
    - Unknown attributes are forwarded to the base dataset.
    """

    def __init__(self, base_dataset: Dataset, sample_indices: Sequence[int]) -> None:
        if base_dataset is None:
            raise ValueError("base_dataset must not be None")

        self.base_dataset = base_dataset
        self.sample_indices = list(sample_indices)

        n = len(base_dataset)
        for idx in self.sample_indices:
            if idx < 0 or idx >= n:
                raise IndexError(
                    f"Subset index out of bounds: {idx} for dataset length {n}"
                )

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Any:
        base_idx = self.sample_indices[idx]
        return self.base_dataset[base_idx]

    @property
    def samples(self) -> List[Any]:
        """Project base_dataset.samples onto the subset."""
        base_samples = getattr(self.base_dataset, "samples", None)
        if base_samples is None:
            raise AttributeError("base_dataset has no attribute 'samples'")
        return [base_samples[i] for i in self.sample_indices]

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the base dataset."""
        try:
            return getattr(self.base_dataset, name)
        except AttributeError as exc:
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute '{name}', "
                f"and base_dataset also has no such attribute."
            ) from exc


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _extract_dt_value(sample: Mapping[str, Any]) -> Optional[float]:
    if "dt" not in sample:
        return None

    dt = sample["dt"]

    if isinstance(dt, (float, int)):
        # An int beyond binary64 range (10**400) raises OverflowError here;
        # report it as an invalid value rather than an arithmetic crash.
        try:
            return float(dt)
        except OverflowError as exc:
            raise ValueError("Sample 'dt' must fit in a finite float") from exc

    if torch.is_tensor(dt):
        if dt.numel() != 1:
            raise ValueError("Sample 'dt' tensor must be scalar-like")
        return float(dt.item())

    raise TypeError(f"Unsupported dt type: {type(dt)}")


def _normalize_dt_value(
    dt: Optional[float],
    dt_round_ndigits: Optional[int],
) -> Optional[float]:
    if dt is None:
        return None
    if dt_round_ndigits is None:
        return float(dt)
    return round(float(dt), dt_round_ndigits)


def _extract_group_key(
    sample: Mapping[str, Any],
    *,
    group_by_dt: bool,
    dt_round_ndigits: Optional[int] = None,
) -> TrajectoryGroupKey:
    if "trajectory_id" not in sample:
        raise KeyError("Sample is missing required key 'trajectory_id'")

    trajectory_id = int(sample["trajectory_id"])
    dt = _extract_dt_value(sample) if group_by_dt else None
    dt = _normalize_dt_value(dt, dt_round_ndigits)

    return TrajectoryGroupKey(trajectory_id=trajectory_id, dt=dt)


def _build_group_to_indices(
    dataset: Dataset,
    *,
    group_by_dt: bool,
    dt_round_ndigits: Optional[int] = None,
) -> Dict[TrajectoryGroupKey, List[int]]:
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise AttributeError(
            "Dataset must expose a 'samples' attribute for grouped splitting"
        )

    group_to_indices: Dict[TrajectoryGroupKey, List[int]] = {}

    for idx, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError(
                f"Dataset sample metadata at index {idx} must be a mapping-like object"
            )

        key = _extract_group_key(
            sample, group_by_dt=group_by_dt, dt_round_ndigits=dt_round_ndigits
        )
        group_to_indices.setdefault(key, []).append(idx)

    if len(group_to_indices) == 0:
        raise ValueError("No groups found in dataset; cannot perform grouped split")

    return group_to_indices


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def split_dataset_by_trajectory(
    dataset: Dataset,
    *,
    val_fraction: float,
    seed: int,
    group_by_dt: bool = False,
    dt_round_ndigits: Optional[int] = None,
    min_val_groups: int = 1,
    val_group_keys_override: Optional[Sequence[TrajectoryGroupKey]] = None,
) -> Tuple[SampleSubsetDataset, SampleSubsetDataset]:
    """
    Leakage-safe grouped split.

    Parameters
    ----------
    dataset :
        Base dataset exposing `.samples`
    val_fraction :
        Fraction of groups allocated to validation (if no override)
    seed :
        Reproducible shuffle seed
    group_by_dt :
        If True, groups are (trajectory_id, dt)
    dt_round_ndigits :
        Optional rounding for dt normalization
    min_val_groups :
        Minimum number of validation groups
    val_group_keys_override :
        Optional explicit validation groups (bypasses shuffle/fraction)

    Returns
    -------
    train_subset, val_subset
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must satisfy 0.0 < val_fraction < 1.0")

    if min_val_groups < 1:
        raise ValueError("min_val_groups must be >= 1")

    group_to_indices = _build_group_to_indices(
        dataset, group_by_dt=group_by_dt, dt_round_ndigits=dt_round_ndigits
    )
    group_keys = list(group_to_indices.keys())

    if val_group_keys_override is not None:
        val_group_keys = set(val_group_keys_override)
        unknown = [g for g in val_group_keys if g not in group_to_indices]
        if unknown:
            raise ValueError(
                f"val_group_keys_override contains unknown groups: {unknown}"
            )
        train_group_keys = set(group_keys) - val_group_keys
        if len(train_group_keys) == 0:
            raise ValueError(
                "Training split would be empty under val_group_keys_override"
            )
        if len(val_group_keys) == 0:
            raise ValueError(
                "Validation split would be empty under val_group_keys_override"
            )
    else:
        group_keys_sorted = sorted(
            group_keys,
            key=lambda g: (
                g.trajectory_id,
                float("-inf") if g.dt is None else g.dt,
            ),
        )

        rng = random.Random(seed)
        rng.shuffle(group_keys_sorted)

        n_groups = len(group_keys_sorted)
        n_val_groups = max(min_val_groups, int(round(val_fraction * n_groups)))

        if n_val_groups >= n_groups:
            n_val_groups = n_groups - 1

        if n_val_groups <= 0:
            raise ValueError(
                "Validation split would be empty after constraints; "
                "increase dataset size or adjust val_fraction."
            )

        val_group_keys = set(group_keys_sorted[:n_val_groups])
        train_group_keys = set(group_keys_sorted[n_val_groups:])

    if len(train_group_keys) == 0:
        raise ValueError("Training split would be empty")

    train_indices: List[int] = []
    val_indices: List[int] = []

    for key, idxs in group_to_indices.items():
        if key in val_group_keys:
            val_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    train_indices.sort()
    val_indices.sort()

    if len(train_indices) == 0:
        raise ValueError("Training subset has zero samples")
    if len(val_indices) == 0:
        raise ValueError("Validation subset has zero samples")

    train_subset = SampleSubsetDataset(dataset, train_indices)
    val_subset = SampleSubsetDataset(dataset, val_indices)

    return train_subset, val_subset


def summarize_group_split(
    train_dataset: SampleSubsetDataset,
    val_dataset: SampleSubsetDataset,
    *,
    group_by_dt: bool = False,
    dt_round_ndigits: Optional[int] = None,
) -> Dict[str, Any]:
    """Compact summary of a grouped split with leakage diagnostics."""
    train_groups = _build_group_to_indices(
        train_dataset, group_by_dt=group_by_dt, dt_round_ndigits=dt_round_ndigits
    )
    val_groups = _build_group_to_indices(
        val_dataset, group_by_dt=group_by_dt, dt_round_ndigits=dt_round_ndigits
    )

    train_group_keys = set(train_groups.keys())
    val_group_keys = set(val_groups.keys())
    overlap_groups = train_group_keys.intersection(val_group_keys)

    train_trajectory_ids = sorted({k.trajectory_id for k in train_group_keys})
    val_trajectory_ids = sorted({k.trajectory_id for k in val_group_keys})
    overlap_trajectory_ids = sorted(
        set(train_trajectory_ids).intersection(val_trajectory_ids)
    )

    train_dt_values = sorted({k.dt for k in train_group_keys if k.dt is not None})
    val_dt_values = sorted({k.dt for k in val_group_keys if k.dt is not None})

    return {
        "group_by_dt": group_by_dt,
        "dt_round_ndigits": dt_round_ndigits,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "num_train_groups": len(train_group_keys),
        "num_val_groups": len(val_group_keys),
        "num_overlap_groups": len(overlap_groups),
        "train_trajectory_ids": train_trajectory_ids,
        "val_trajectory_ids": val_trajectory_ids,
        "overlap_trajectory_ids": overlap_trajectory_ids,
        "train_dt_values": train_dt_values,
        "val_dt_values": val_dt_values,
    }
