"""
dataset_split_hex_v3.py
=======================
Leakage-safe dataset splitting for HexaNTK operator learning.

Key design decisions:
- Non-mutating: SampleSubsetDataset wraps without modifying base dataset
- Group-based: splits by trajectory_id, not individual samples
- Optional finer grouping by (trajectory_id, dt)
- Optional dt rounding for floating-point normalization
- Optional deterministic override for paper-reproducible splits
- Attribute passthrough for grid, config, etc.

Leakage guarantee (owner decision E0927-03 on issue #42): no trajectory_id
appears in both train and val -- ALWAYS, whatever the dt values or types and
whether group_by_dt is set. The unit of the split is the trajectory.
- Default (group_by_dt=False): one group per trajectory_id.
- group_by_dt=True: groups are (trajectory_id, dt) for reporting
  (summarize_group_split counts and lists them), but all dt groups of one
  trajectory are allocated to the same side; val_fraction and
  min_val_groups count trajectories. An explicit val_group_keys_override
  that names only some dt groups of a trajectory raises ValueError.
  summarize_group_split()["overlap_trajectory_ids"] stays [] for every split
  produced here; it is the diagnostic for hand-built subsets.
  dt keys of one trajectory that differ only below float32 resolution
  (0.1 as float vs. float32 tensor) are one step size in two precisions and
  still raise ValueError (inconsistent metadata, issue #42). Set
  dt_round_ndigits to merge them.
With one dt per trajectory the split for a given seed is identical to the
one before E0927-03.

Input contract (fail-closed, ValueError/TypeError instead of silent keys):
- trajectory_id: an exact integer (int, integral float/Decimal/Fraction,
  single-element tensor or numpy array/scalar) or a canonical decimal
  string ("12", "-3"). bool, bytes and non-canonical strings ("012",
  "+12", " 12") are rejected: they collide with other ids.
- dt (only read under group_by_dt): a finite real scalar, not bool.
"""
from __future__ import annotations

__all__ = [
    "TrajectoryGroupKey",
    "SampleSubsetDataset",
    "split_dataset_by_trajectory",
    "summarize_group_split",
]

import math
import numbers
import random
import re
import struct
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

# The quantisation contract for "same step size reported in two precisions"
# (issues #42, #59) is float32 resolution: two dt keys of one trajectory are
# the same step size iff they round to the same IEEE-754 binary32 value
# (e.g. 0.1 as a Python float and a float32 tensor's 0.10000000149...). The
# earlier relative tolerance (2**-20) compared with abs_tol=0.0, so 0.0 and
# 1e-46 were treated as different although both are float32 0.0 (#59 P3).
def _as_float32(value: float) -> float:
    """Return ``value`` rounded to IEEE-754 binary32, as a Python float.

    Values beyond the binary32 range round to +/-inf (round-to-nearest),
    matching how a float32 tensor would store them; sub-binary32 magnitudes
    (e.g. 1e-46) underflow to 0.0 exactly as float32 does.
    """
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.inf if value > 0 else -math.inf

# Canonical decimal integer: no sign "+", no leading zeros, no "-0", no
# whitespace, ASCII digits only (int() also accepts "٣" and "1_2").
_CANONICAL_INT_STR = re.compile(r"0|-?[1-9][0-9]*")


def _unwrap_scalar(value: Any, *, name: str) -> Any:
    """Return the Python scalar of a single-element tensor/numpy value.

    torch.Tensor, numpy.ndarray and numpy scalars all expose ``shape`` and
    ``item()``; numpy values are not ``numbers.Real`` subclasses in general
    (a 0-d array is not), so without unwrapping they bypass exactness checks
    (#40 P3-1: np.array(1.5) was truncated to 1).
    """
    item = getattr(value, "item", None)
    shape = getattr(value, "shape", None)
    if callable(item) and shape is not None:
        if math.prod(shape) != 1:
            raise ValueError(f"Sample '{name}' tensor must be scalar-like")
        return item()
    return value


def _validate_dt_round_ndigits(dt_round_ndigits: Optional[int]) -> None:
    if dt_round_ndigits is None:
        return
    if isinstance(dt_round_ndigits, bool) or not isinstance(
        dt_round_ndigits, numbers.Integral
    ):
        raise TypeError(
            f"dt_round_ndigits must be None or an int, got {dt_round_ndigits!r}"
        )


def _extract_dt_value(sample: Mapping[str, Any]) -> Optional[float]:
    """Return ``sample["dt"]`` as a finite float, None if absent, or raise.

    A non-finite dt is a contract violation (#39): nan != nan, so every
    sample with a fresh nan formed its own (trajectory_id, dt) group and one
    trajectory could land in both train and val. inf/-inf are rejected by
    the same rule. The check runs per sample, before the key is grouped.
    """
    if "dt" not in sample:
        return None

    dt = _unwrap_scalar(sample["dt"], name="dt")

    if isinstance(dt, bool) or not isinstance(dt, numbers.Real):
        raise TypeError(f"Unsupported dt type: {type(dt)}")
    # An int beyond binary64 range (10**400) raises OverflowError here;
    # report it as an invalid value rather than an arithmetic crash.
    try:
        value = float(dt)
    except OverflowError as exc:
        raise ValueError("Sample 'dt' must fit in a finite float") from exc

    if not math.isfinite(value):
        raise ValueError(f"Sample 'dt' must be finite, got {value!r}")
    return value


def _normalize_dt_value(
    dt: Optional[float],
    dt_round_ndigits: Optional[int],
) -> Optional[float]:
    if dt is None:
        return None
    if dt_round_ndigits is None:
        return float(dt)
    try:
        return round(float(dt), dt_round_ndigits)
    except OverflowError as exc:
        # round(1.797e308, -308) rounds past binary64 max.
        raise ValueError(
            f"Sample 'dt' {dt!r} cannot be rounded to {dt_round_ndigits} digits "
            "within the finite float range"
        ) from exc


def _extract_trajectory_id(value: Any) -> int:
    """Return ``value`` as an exact int group id, or raise ValueError.

    int() alone raised OverflowError for inf and silently truncated
    non-integral reals (1.5 and 1.7 both became group 1, merging two
    trajectories). Exact int-convertible values keep their former key.

    bool, bytes and non-canonical strings are rejected (#40 P3-5): True is
    not an id, and int() maps "012", "+12", " 12" and b"12" all to 12.
    """
    value = _unwrap_scalar(value, name="trajectory_id")

    message = f"Sample 'trajectory_id' must be a finite integer, got {value!r}"
    if isinstance(value, (bool, bytes, bytearray)):
        raise ValueError(message)
    if isinstance(value, str) and not _CANONICAL_INT_STR.fullmatch(value):
        raise ValueError(message)
    try:
        trajectory_id = int(value)
    except (OverflowError, ValueError) as exc:
        # int(inf) -> OverflowError, int(nan) -> ValueError.
        raise ValueError(message) from exc

    # Exact comparison (no float round-trip), so 10**400 stays accepted.
    if isinstance(value, (numbers.Real, Decimal)) and trajectory_id != value:
        raise ValueError(message)
    return trajectory_id


def _extract_group_key(
    sample: Mapping[str, Any],
    *,
    group_by_dt: bool,
    dt_round_ndigits: Optional[int] = None,
) -> TrajectoryGroupKey:
    if "trajectory_id" not in sample:
        raise KeyError("Sample is missing required key 'trajectory_id'")

    trajectory_id = _extract_trajectory_id(sample["trajectory_id"])
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

    _validate_dt_round_ndigits(dt_round_ndigits)
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

    if group_by_dt:
        _reject_precision_split_dt_keys(group_to_indices.keys())

    return group_to_indices


def _group_sort_key(g: TrajectoryGroupKey) -> Tuple[int, float]:
    return (g.trajectory_id, float("-inf") if g.dt is None else g.dt)


def _reject_precision_split_dt_keys(keys: Any) -> None:
    """Fail closed when one trajectory carries one step size twice (#42).

    0.1 (float) and tensor(0.1) (float32 -> 0.10000000149...) are different
    keys; before E0927-03 the trajectory then split into two groups and landed
    in train and val in 41/50 seeds. The split now keeps a trajectory whole,
    so this guard no longer carries the leakage invariant; it stays as a
    metadata-consistency check (one step size must not be reported as two).
    Deliberately different step sizes (0.1 vs 0.2) stay separate groups.
    """
    dts_by_trajectory: Dict[int, List[float]] = {}
    for key in keys:
        if key.dt is not None:
            dts_by_trajectory.setdefault(key.trajectory_id, []).append(key.dt)

    for trajectory_id, dts in dts_by_trajectory.items():
        dts.sort()
        for lower, upper in zip(dts, dts[1:]):
            if _as_float32(lower) == _as_float32(upper):
                raise ValueError(
                    f"trajectory_id {trajectory_id} has dt keys {lower!r} and "
                    f"{upper!r} that round to the same float32 value, i.e. "
                    "one step size in two precisions, which would be grouped "
                    "and reported as two step sizes. Set "
                    "dt_round_ndigits (e.g. 6) to normalise dt before grouping."
                )


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
        Fraction of trajectories allocated to validation (if no override)
    seed :
        Reproducible shuffle seed
    group_by_dt :
        If True, groups are (trajectory_id, dt), but a trajectory is never
        split across train and val (E0927-03, see module docstring);
        same-step keys that differ only by precision raise ValueError.
    dt_round_ndigits :
        Optional rounding for dt normalization (None or int)
    min_val_groups :
        Minimum number of validation trajectories
    val_group_keys_override :
        Optional explicit validation groups (bypasses shuffle/fraction);
        must contain all dt groups of every trajectory it names

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

    # E0927-03 (owner decision on issue #42): the unit of the split is the
    # TRAJECTORY, always. Under group_by_dt one trajectory may own several
    # (trajectory_id, dt) groups; they are allocated together, never apart.
    groups_by_trajectory: Dict[int, List[TrajectoryGroupKey]] = {}
    for key in group_keys:
        groups_by_trajectory.setdefault(key.trajectory_id, []).append(key)

    if val_group_keys_override is not None:
        val_group_keys = set(val_group_keys_override)
        unknown = [g for g in val_group_keys if g not in group_to_indices]
        if unknown:
            raise ValueError(
                f"val_group_keys_override contains unknown groups: {unknown}"
            )
        # An explicit override is the caller's declared split: it is not
        # silently widened. Naming only some dt groups of a trajectory would
        # put that trajectory on both sides -> fail closed.
        for trajectory_id in sorted({g.trajectory_id for g in val_group_keys}):
            missing = [
                g for g in groups_by_trajectory[trajectory_id] if g not in val_group_keys
            ]
            if missing:
                raise ValueError(
                    f"val_group_keys_override splits trajectory_id {trajectory_id}: "
                    f"groups {sorted(missing, key=_group_sort_key)} would stay in "
                    "train; a trajectory must never be in both train and val "
                    "(E0927-03). List all of its dt groups or none."
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
        # Shuffle trajectories, not groups. With one group per trajectory
        # (group_by_dt=False, or one dt per trajectory) the sorted list has
        # the same length and order as the former group list, so
        # random.Random(seed).shuffle yields the same permutation and the
        # split is unchanged; only multi-dt trajectories are now kept whole.
        trajectories_sorted = sorted(groups_by_trajectory)

        rng = random.Random(seed)
        rng.shuffle(trajectories_sorted)

        n_units = len(trajectories_sorted)
        n_val_units = max(min_val_groups, int(round(val_fraction * n_units)))

        if n_val_units >= n_units:
            n_val_units = n_units - 1

        if n_val_units <= 0:
            raise ValueError(
                "Validation split would be empty after constraints; "
                "increase dataset size or adjust val_fraction."
            )

        val_group_keys = {
            key
            for trajectory_id in trajectories_sorted[:n_val_units]
            for key in groups_by_trajectory[trajectory_id]
        }
        train_group_keys = set(group_keys) - val_group_keys

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
