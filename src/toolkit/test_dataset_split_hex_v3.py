"""
test_dataset_split_hex_v3.py
============================
Tests for leakage-safe grouped dataset splitting.

Focus: the module's core promise — no trajectory appears in both train and
val splits — plus determinism, constraint validation and the non-mutating
subset wrapper.
"""
from __future__ import annotations

import random
import sys
import unittest
from decimal import Decimal
from fractions import Fraction
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

import dataset_split_hex_v3 as dataset_split

from dataset_split_hex_v3 import (
    SampleSubsetDataset,
    TrajectoryGroupKey,
    split_dataset_by_trajectory,
    summarize_group_split,
)


class FakeTrajectoryDataset(Dataset):
    """Minimal dataset exposing `.samples` metadata, as the splitter requires."""

    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        self.samples = samples
        self.grid = "hex-grid-marker"  # for attribute-passthrough tests

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def make_dataset(
    n_trajectories: int,
    samples_per_trajectory: int = 3,
    dt: float = 0.1,
) -> FakeTrajectoryDataset:
    samples = [
        {"trajectory_id": tid, "dt": dt, "step": s}
        for tid in range(n_trajectories)
        for s in range(samples_per_trajectory)
    ]
    return FakeTrajectoryDataset(samples)


# ------------------------------------------------------------------
# SampleSubsetDataset
# ------------------------------------------------------------------


class TestSampleSubsetDataset(unittest.TestCase):
    def test_rejects_none_base(self) -> None:
        with self.assertRaises(ValueError):
            SampleSubsetDataset(None, [0])

    def test_rejects_out_of_bounds_indices(self) -> None:
        ds = make_dataset(2)  # 6 samples
        with self.assertRaises(IndexError):
            SampleSubsetDataset(ds, [0, 6])
        with self.assertRaises(IndexError):
            SampleSubsetDataset(ds, [-1])

    def test_len_and_getitem_delegate(self) -> None:
        ds = make_dataset(2)
        subset = SampleSubsetDataset(ds, [1, 4])
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset[0], ds[1])
        self.assertEqual(subset[1], ds[4])

    def test_samples_projection(self) -> None:
        ds = make_dataset(2)
        subset = SampleSubsetDataset(ds, [2, 3])
        self.assertEqual(subset.samples, [ds.samples[2], ds.samples[3]])

    def test_samples_requires_base_attribute(self) -> None:
        class Bare(Dataset):
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int) -> int:
                return idx

        subset = SampleSubsetDataset(Bare(), [0])
        with self.assertRaises(AttributeError):
            _ = subset.samples

    def test_attribute_passthrough(self) -> None:
        ds = make_dataset(1)
        subset = SampleSubsetDataset(ds, [0])
        self.assertEqual(subset.grid, "hex-grid-marker")

    def test_missing_attribute_raises(self) -> None:
        ds = make_dataset(1)
        subset = SampleSubsetDataset(ds, [0])
        with self.assertRaises(AttributeError):
            _ = subset.does_not_exist

    def test_base_dataset_not_mutated(self) -> None:
        ds = make_dataset(3)
        before = list(ds.samples)
        split_dataset_by_trajectory(ds, val_fraction=0.34, seed=7)
        self.assertEqual(ds.samples, before)


# ------------------------------------------------------------------
# split_dataset_by_trajectory — validation
# ------------------------------------------------------------------


class TestSplitValidation(unittest.TestCase):
    def test_val_fraction_bounds(self) -> None:
        ds = make_dataset(4)
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                split_dataset_by_trajectory(ds, val_fraction=bad, seed=0)

    def test_min_val_groups_must_be_positive(self) -> None:
        ds = make_dataset(4)
        with self.assertRaises(ValueError):
            split_dataset_by_trajectory(
                ds, val_fraction=0.25, seed=0, min_val_groups=0
            )

    def test_single_group_cannot_split(self) -> None:
        ds = make_dataset(1)
        with self.assertRaises(ValueError):
            split_dataset_by_trajectory(ds, val_fraction=0.5, seed=0)

    def test_dataset_without_samples_attribute(self) -> None:
        class Bare(Dataset):
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int) -> int:
                return idx

        with self.assertRaises(AttributeError):
            split_dataset_by_trajectory(Bare(), val_fraction=0.5, seed=0)

    def test_empty_dataset(self) -> None:
        ds = FakeTrajectoryDataset([])
        with self.assertRaises(ValueError):
            split_dataset_by_trajectory(ds, val_fraction=0.5, seed=0)

    def test_sample_missing_trajectory_id(self) -> None:
        ds = FakeTrajectoryDataset([{"dt": 0.1}])
        with self.assertRaises(KeyError):
            split_dataset_by_trajectory(ds, val_fraction=0.5, seed=0)

    def test_non_mapping_sample_metadata(self) -> None:
        ds = FakeTrajectoryDataset([("trajectory_id", 0)])
        with self.assertRaises(TypeError):
            split_dataset_by_trajectory(ds, val_fraction=0.5, seed=0)

    def test_non_scalar_dt_tensor(self) -> None:
        ds = FakeTrajectoryDataset(
            [{"trajectory_id": 0, "dt": torch.tensor([0.1, 0.2])}]
        )
        with self.assertRaises(ValueError):
            split_dataset_by_trajectory(
                ds, val_fraction=0.5, seed=0, group_by_dt=True
            )

    def test_unsupported_dt_type(self) -> None:
        ds = FakeTrajectoryDataset([{"trajectory_id": 0, "dt": "0.1"}])
        with self.assertRaises(TypeError):
            split_dataset_by_trajectory(
                ds, val_fraction=0.5, seed=0, group_by_dt=True
            )

    # Smallest int that float() rounds past binary64 max (tie -> 2**1024).
    FIRST_OVERFLOWING_INT = 2**1024 - 2**970

    def test_int_dt_beyond_binary64_raises_value_error(self) -> None:
        for dt in (10**400, 2**1024, self.FIRST_OVERFLOWING_INT):
            with self.subTest(bits=dt.bit_length()):
                ds = FakeTrajectoryDataset(
                    [
                        {"trajectory_id": 0, "dt": dt},
                        {"trajectory_id": 1, "dt": 0.1},
                    ]
                )
                # Capture any exception and check its exact type, so the
                # pre-fix OverflowError is an assertion failure.
                with self.assertRaises(Exception) as ctx:
                    split_dataset_by_trajectory(
                        ds, val_fraction=0.5, seed=0, group_by_dt=True
                    )
                exc = ctx.exception
                self.assertIs(
                    type(exc),
                    ValueError,
                    f"expected ValueError, got {type(exc).__name__}: {exc}",
                )
                self.assertRegex(str(exc), "^Sample 'dt' must fit in a finite float$")
                self.assertIsInstance(exc.__cause__, OverflowError)

    def test_int_dt_at_or_just_below_binary64_max_is_accepted(self) -> None:
        max_float = sys.float_info.max
        for dt in (int(max_float), self.FIRST_OVERFLOWING_INT - 1):
            with self.subTest(bits=dt.bit_length()):
                value = dataset_split._extract_dt_value({"dt": dt})
                self.assertEqual(value, max_float)
                self.assertIs(type(value), float)
                ds = FakeTrajectoryDataset(
                    [
                        {"trajectory_id": 0, "dt": dt},
                        {"trajectory_id": 1, "dt": dt},
                    ]
                )
                train, val = split_dataset_by_trajectory(
                    ds, val_fraction=0.5, seed=0, group_by_dt=True
                )
                self.assertEqual(len(train) + len(val), 2)

    def test_huge_dt_is_ignored_without_group_by_dt(self) -> None:
        ds = FakeTrajectoryDataset(
            [
                {"trajectory_id": 0, "dt": 10**400},
                {"trajectory_id": 1, "dt": 10**400},
            ]
        )
        train, val = split_dataset_by_trajectory(ds, val_fraction=0.5, seed=0)
        self.assertEqual(len(train) + len(val), 2)

    # --- dt finiteness contract (issue #39) ---

    DT_NON_FINITE_MESSAGE = r"^Sample 'dt' must be finite, got "

    # Each factory returns a FRESH object per call. Python compares tuple
    # members by identity first, so one shared nan object still forms a
    # single group and hides the bug; real metadata (tensor.item(),
    # arithmetic, deserialization) yields a new nan per sample, and
    # nan != nan then made every sample its own group (the #39 leak).
    NON_FINITE_DT_FACTORIES = (
        ("nan", lambda: float("nan")),
        ("+inf", lambda: float("inf")),
        ("-inf", lambda: float("-inf")),
        ("tensor nan", lambda: torch.tensor(float("nan"))),
        ("tensor +inf", lambda: torch.tensor(float("inf"))),
        ("float64 tensor -inf", lambda: torch.tensor(float("-inf"), dtype=torch.float64)),
        ("numpy float64 nan", lambda: np.float64("nan")),
    )

    def _assert_dt_contract_error(self, ds: Dataset, **kwargs: Any) -> None:
        # Capture any exception and check its exact type; before the fix the
        # split succeeded, so "not raised" is the pre-fix assertion failure.
        with self.assertRaises(Exception) as ctx:
            split_dataset_by_trajectory(
                ds, val_fraction=0.5, seed=0, group_by_dt=True, **kwargs
            )
        exc = ctx.exception
        self.assertIs(
            type(exc),
            ValueError,
            f"expected ValueError, got {type(exc).__name__}: {exc}",
        )
        self.assertRegex(str(exc), self.DT_NON_FINITE_MESSAGE)

    def test_non_finite_dt_raises_value_error(self) -> None:
        for label, make_dt in self.NON_FINITE_DT_FACTORIES:
            for ndigits in (None, 6):
                with self.subTest(dt=label, dt_round_ndigits=ndigits):
                    ds = FakeTrajectoryDataset(
                        [
                            {"trajectory_id": tid, "dt": make_dt(), "step": s}
                            for tid in range(3)
                            for s in range(3)
                        ]
                    )
                    self._assert_dt_contract_error(ds, dt_round_ndigits=ndigits)

    def test_single_non_finite_dt_in_valid_trajectory_raises(self) -> None:
        # One bad sample inside an otherwise valid trajectory formerly became
        # an extra group of that trajectory. Positions: first, middle, last.
        for label, make_dt in self.NON_FINITE_DT_FACTORIES:
            for bad_index in (0, 4, 8):
                with self.subTest(dt=label, bad_index=bad_index):
                    samples = [
                        {"trajectory_id": tid, "dt": 0.1, "step": s}
                        for tid in range(3)
                        for s in range(3)
                    ]
                    samples[bad_index]["dt"] = make_dt()
                    ds = FakeTrajectoryDataset(samples)
                    self._assert_dt_contract_error(ds)

    def test_non_finite_dt_raises_in_summary(self) -> None:
        # summarize_group_split groups through the same path; with fresh nan
        # objects it formerly reported zero overlap for a leaking split.
        ds = FakeTrajectoryDataset(
            [
                {"trajectory_id": tid, "dt": float("nan"), "step": s}
                for tid in range(2)
                for s in range(2)
            ]
        )
        train, val = split_dataset_by_trajectory(ds, val_fraction=0.5, seed=0)
        with self.assertRaises(ValueError) as ctx:
            summarize_group_split(train, val, group_by_dt=True)
        self.assertRegex(str(ctx.exception), self.DT_NON_FINITE_MESSAGE)

    def test_non_finite_dt_is_ignored_without_group_by_dt(self) -> None:
        # Same scope as the 10**400 case above: dt is read only when it is
        # part of the group key.
        ds = FakeTrajectoryDataset(
            [
                {"trajectory_id": 0, "dt": float("nan")},
                {"trajectory_id": 1, "dt": float("inf")},
            ]
        )
        train, val = split_dataset_by_trajectory(ds, val_fraction=0.5, seed=0)
        self.assertEqual(len(train) + len(val), 2)

    def test_finite_zero_and_negative_dt_are_valid_group_keys(self) -> None:
        # The splitter uses dt only as a group key; sign and zero are not
        # its contract. Signed zeros compare and hash equal -> one group.
        for dt_values in ((0.0, -0.0), (-0.05, -0.05), (5e-324, 5e-324)):
            with self.subTest(dt=dt_values):
                ds = FakeTrajectoryDataset(
                    [
                        {"trajectory_id": 0, "dt": dt_values[0]},
                        {"trajectory_id": 0, "dt": dt_values[1]},
                        {"trajectory_id": 1, "dt": dt_values[0]},
                    ]
                )
                train, val = split_dataset_by_trajectory(
                    ds, val_fraction=0.5, seed=0, group_by_dt=True
                )
                self.assertEqual(
                    {s["trajectory_id"] for s in train.samples}
                    & {s["trajectory_id"] for s in val.samples},
                    set(),
                )

    # --- trajectory_id contract (issue #36) ---

    TRAJECTORY_ID_MESSAGE = r"^Sample 'trajectory_id' must be a finite integer, got "

    def _split_with_trajectory_id(self, trajectory_id: Any) -> None:
        ds = FakeTrajectoryDataset(
            [
                {"trajectory_id": trajectory_id, "dt": 0.1},
                {"trajectory_id": 1, "dt": 0.1},
            ]
        )
        split_dataset_by_trajectory(ds, val_fraction=0.5, seed=0)

    def test_non_finite_trajectory_id_raises_value_error(self) -> None:
        # int(inf) raises OverflowError, int(nan) ValueError with an
        # unrelated message; both must surface as the contract ValueError.
        cases = [
            (float("inf"), OverflowError),
            (float("-inf"), OverflowError),
            (float("nan"), ValueError),
            (torch.tensor(float("inf")), OverflowError),
            (Decimal("Infinity"), OverflowError),
            (Decimal("NaN"), ValueError),
        ]
        for trajectory_id, cause_type in cases:
            with self.subTest(trajectory_id=repr(trajectory_id)):
                # Capture any exception and check its exact type, so the
                # pre-fix OverflowError is an assertion failure.
                with self.assertRaises(Exception) as ctx:
                    self._split_with_trajectory_id(trajectory_id)
                exc = ctx.exception
                self.assertIs(
                    type(exc),
                    ValueError,
                    f"expected ValueError, got {type(exc).__name__}: {exc}",
                )
                self.assertRegex(str(exc), self.TRAJECTORY_ID_MESSAGE)
                self.assertIsInstance(exc.__cause__, cause_type)

    def test_non_integral_trajectory_id_raises_value_error(self) -> None:
        # Formerly int() truncated silently: 1.5 and 1.7 both became group 1,
        # merging two distinct trajectories into one split group.
        for trajectory_id in (
            1.5,
            -0.5,
            torch.tensor(1.5),
            Fraction(3, 2),
            Decimal("1.5"),
        ):
            with self.subTest(trajectory_id=repr(trajectory_id)):
                with self.assertRaises(ValueError) as ctx:
                    self._split_with_trajectory_id(trajectory_id)
                self.assertRegex(str(ctx.exception), self.TRAJECTORY_ID_MESSAGE)

    def test_non_scalar_trajectory_id_tensor(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._split_with_trajectory_id(torch.tensor([0, 1]))
        self.assertEqual(
            str(ctx.exception), "Sample 'trajectory_id' tensor must be scalar-like"
        )

    def test_integral_trajectory_ids_keep_their_group_key(self) -> None:
        # Differential against the pre-fix int() path: every accepted value
        # must yield exactly the key int() produced before.
        for trajectory_id, expected in (
            (7, 7),
            (-3, -3),
            (7.0, 7),
            (-0.0, 0),
            (1e300, int(1e300)),
            (10**400, 10**400),
            (Fraction(14, 2), 7),
            (Decimal("7"), 7),
            (torch.tensor(7), 7),
            (torch.tensor(7.0), 7),
            ("7", 7),
        ):
            with self.subTest(trajectory_id=repr(trajectory_id)):
                key = dataset_split._extract_group_key(
                    {"trajectory_id": trajectory_id}, group_by_dt=False
                )
                self.assertEqual(key.trajectory_id, expected)
                self.assertIs(type(key.trajectory_id), int)


# ------------------------------------------------------------------
# split_dataset_by_trajectory — core behavior
# ------------------------------------------------------------------


class TestSplitBehavior(unittest.TestCase):
    def _trajectory_ids(self, subset: SampleSubsetDataset) -> set:
        return {s["trajectory_id"] for s in subset.samples}

    def test_no_trajectory_leakage(self) -> None:
        ds = make_dataset(10)
        train, val = split_dataset_by_trajectory(ds, val_fraction=0.3, seed=42)
        self.assertEqual(
            self._trajectory_ids(train) & self._trajectory_ids(val), set()
        )

    def test_no_trajectory_leakage_invariant_across_seeds(self) -> None:
        # Invariant (module docstring): no trajectory_id in both train and
        # val, and train/val partition all indices. Checked for both
        # grouping modes, with and without rounding, over 50 seeded random
        # datasets with ONE finite dt per trajectory (zero, signed zero,
        # negative, subnormal and huge included; float and float64-tensor
        # values, fresh objects per sample). Several dt values inside one
        # trajectory are split by design under group_by_dt (see
        # test_group_by_dt_separates_dt_variants) and are not covered here.
        dt_pool = (0.1, 0.2, 1e-3, 0.0, -0.05, 1e300, 5e-324)
        for seed in range(50):
            rng = random.Random(seed)
            samples: List[Dict[str, Any]] = []
            for tid in range(rng.randint(2, 12)):
                dt = rng.choice(dt_pool)
                for step in range(rng.randint(1, 5)):
                    value: Any = -dt if (dt == 0.0 and step % 2) else float(dt)
                    if rng.random() < 0.5:
                        value = torch.tensor(value, dtype=torch.float64)
                    samples.append({"trajectory_id": tid, "dt": value, "step": step})
            ds = FakeTrajectoryDataset(samples)
            val_fraction = rng.choice((0.1, 0.25, 0.5))
            for group_by_dt in (False, True):
                for ndigits in (None, 6):
                    with self.subTest(
                        seed=seed, group_by_dt=group_by_dt, dt_round_ndigits=ndigits
                    ):
                        train, val = split_dataset_by_trajectory(
                            ds,
                            val_fraction=val_fraction,
                            seed=seed,
                            group_by_dt=group_by_dt,
                            dt_round_ndigits=ndigits,
                        )
                        self.assertEqual(
                            self._trajectory_ids(train) & self._trajectory_ids(val),
                            set(),
                        )
                        train_idx = set(train.sample_indices)
                        val_idx = set(val.sample_indices)
                        self.assertEqual(train_idx & val_idx, set())
                        self.assertEqual(train_idx | val_idx, set(range(len(ds))))
                        summary = summarize_group_split(
                            train,
                            val,
                            group_by_dt=group_by_dt,
                            dt_round_ndigits=ndigits,
                        )
                        self.assertEqual(summary["overlap_trajectory_ids"], [])
                        self.assertEqual(summary["num_overlap_groups"], 0)

    def test_partition_is_complete_and_disjoint(self) -> None:
        ds = make_dataset(10, samples_per_trajectory=4)
        train, val = split_dataset_by_trajectory(ds, val_fraction=0.3, seed=42)
        train_idx = set(train.sample_indices)
        val_idx = set(val.sample_indices)
        self.assertEqual(train_idx & val_idx, set())
        self.assertEqual(train_idx | val_idx, set(range(len(ds))))

    def test_same_seed_same_split(self) -> None:
        ds = make_dataset(10)
        t1, v1 = split_dataset_by_trajectory(ds, val_fraction=0.3, seed=42)
        t2, v2 = split_dataset_by_trajectory(ds, val_fraction=0.3, seed=42)
        self.assertEqual(t1.sample_indices, t2.sample_indices)
        self.assertEqual(v1.sample_indices, v2.sample_indices)

    def test_different_seed_different_split(self) -> None:
        ds = make_dataset(50)
        _, v1 = split_dataset_by_trajectory(ds, val_fraction=0.3, seed=1)
        _, v2 = split_dataset_by_trajectory(ds, val_fraction=0.3, seed=2)
        self.assertNotEqual(v1.sample_indices, v2.sample_indices)

    def test_val_fraction_respected(self) -> None:
        ds = make_dataset(10)
        _, val = split_dataset_by_trajectory(ds, val_fraction=0.3, seed=0)
        self.assertEqual(len(self._trajectory_ids(val)), 3)

    def test_min_val_groups_enforced(self) -> None:
        # 0.01 * 10 groups rounds to 0 -> min_val_groups=1 must win.
        ds = make_dataset(10)
        _, val = split_dataset_by_trajectory(
            ds, val_fraction=0.01, seed=0, min_val_groups=1
        )
        self.assertEqual(len(self._trajectory_ids(val)), 1)

    def test_val_groups_clamped_below_total(self) -> None:
        # min_val_groups >= n_groups is clamped so train keeps >= 1 group.
        ds = make_dataset(3)
        train, val = split_dataset_by_trajectory(
            ds, val_fraction=0.5, seed=0, min_val_groups=99
        )
        self.assertEqual(len(self._trajectory_ids(train)), 1)
        self.assertEqual(len(self._trajectory_ids(val)), 2)

    def test_indices_sorted(self) -> None:
        ds = make_dataset(6)
        train, val = split_dataset_by_trajectory(ds, val_fraction=0.34, seed=3)
        self.assertEqual(train.sample_indices, sorted(train.sample_indices))
        self.assertEqual(val.sample_indices, sorted(val.sample_indices))

    def test_group_by_dt_separates_dt_variants(self) -> None:
        samples = [
            {"trajectory_id": 0, "dt": 0.1},
            {"trajectory_id": 0, "dt": 0.2},
        ]
        ds = FakeTrajectoryDataset(samples)
        # With group_by_dt these are 2 groups -> a split is possible;
        # without it they collapse into 1 group -> split must fail.
        train, val = split_dataset_by_trajectory(
            ds, val_fraction=0.5, seed=0, group_by_dt=True
        )
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        with self.assertRaises(ValueError):
            split_dataset_by_trajectory(ds, val_fraction=0.5, seed=0)

    def test_dt_rounding_merges_float_noise(self) -> None:
        samples = [
            {"trajectory_id": 0, "dt": 0.1},
            {"trajectory_id": 0, "dt": 0.1 + 1e-12},
        ]
        ds = FakeTrajectoryDataset(samples)
        # Rounded, both samples form ONE group -> split impossible.
        with self.assertRaises(ValueError):
            split_dataset_by_trajectory(
                ds, val_fraction=0.5, seed=0, group_by_dt=True, dt_round_ndigits=6
            )

    def test_scalar_tensor_dt_accepted(self) -> None:
        samples = [
            {"trajectory_id": 0, "dt": torch.tensor(0.1)},
            {"trajectory_id": 1, "dt": torch.tensor(0.1)},
        ]
        ds = FakeTrajectoryDataset(samples)
        train, val = split_dataset_by_trajectory(
            ds, val_fraction=0.5, seed=0, group_by_dt=True
        )
        self.assertEqual(len(train) + len(val), 2)


# ------------------------------------------------------------------
# split_dataset_by_trajectory — explicit override
# ------------------------------------------------------------------


class TestSplitOverride(unittest.TestCase):
    def test_override_selects_exact_groups(self) -> None:
        ds = make_dataset(5)
        override = [TrajectoryGroupKey(trajectory_id=2)]
        train, val = split_dataset_by_trajectory(
            ds, val_fraction=0.2, seed=0, val_group_keys_override=override
        )
        val_ids = {s["trajectory_id"] for s in val.samples}
        train_ids = {s["trajectory_id"] for s in train.samples}
        self.assertEqual(val_ids, {2})
        self.assertEqual(train_ids, {0, 1, 3, 4})

    def test_override_unknown_group_rejected(self) -> None:
        ds = make_dataset(3)
        with self.assertRaises(ValueError):
            split_dataset_by_trajectory(
                ds,
                val_fraction=0.5,
                seed=0,
                val_group_keys_override=[TrajectoryGroupKey(trajectory_id=99)],
            )

    def test_override_covering_all_groups_rejected(self) -> None:
        ds = make_dataset(2)
        override = [
            TrajectoryGroupKey(trajectory_id=0),
            TrajectoryGroupKey(trajectory_id=1),
        ]
        with self.assertRaises(ValueError):
            split_dataset_by_trajectory(
                ds, val_fraction=0.5, seed=0, val_group_keys_override=override
            )


# ------------------------------------------------------------------
# summarize_group_split
# ------------------------------------------------------------------


class TestSummarizeGroupSplit(unittest.TestCase):
    def test_summary_reports_no_leakage(self) -> None:
        ds = make_dataset(10, samples_per_trajectory=2)
        train, val = split_dataset_by_trajectory(ds, val_fraction=0.3, seed=42)
        summary = summarize_group_split(train, val)
        self.assertEqual(summary["num_overlap_groups"], 0)
        self.assertEqual(summary["overlap_trajectory_ids"], [])
        self.assertEqual(summary["num_train_samples"], len(train))
        self.assertEqual(summary["num_val_samples"], len(val))
        self.assertEqual(
            summary["num_train_groups"] + summary["num_val_groups"], 10
        )

    def test_summary_detects_engineered_overlap(self) -> None:
        # Adversarial check: hand-built overlapping subsets MUST be flagged.
        ds = make_dataset(4)
        overlapping_a = SampleSubsetDataset(ds, [0, 1, 2, 3, 4, 5])  # traj 0,1
        overlapping_b = SampleSubsetDataset(ds, [3, 4, 5, 6, 7, 8])  # traj 1,2
        summary = summarize_group_split(overlapping_a, overlapping_b)
        self.assertEqual(summary["num_overlap_groups"], 1)
        self.assertEqual(summary["overlap_trajectory_ids"], [1])

    def test_summary_reports_dt_values(self) -> None:
        samples = [
            {"trajectory_id": 0, "dt": 0.1},
            {"trajectory_id": 1, "dt": 0.2},
        ]
        ds = FakeTrajectoryDataset(samples)
        train, val = split_dataset_by_trajectory(
            ds, val_fraction=0.5, seed=0, group_by_dt=True
        )
        summary = summarize_group_split(train, val, group_by_dt=True)
        self.assertEqual(summary["group_by_dt"], True)
        self.assertEqual(
            sorted(summary["train_dt_values"] + summary["val_dt_values"]),
            [0.1, 0.2],
        )


if __name__ == "__main__":
    unittest.main()
