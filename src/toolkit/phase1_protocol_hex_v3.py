"""
phase1_protocol_hex_v3.py
=========================
Phase 1 evaluation-protocol instruments (contract: spec/PHASE1_PROTOCOL_v1.md)
for sections 4-7 of spec/PREREGISTRATION_PHASE1_v1.md:

- the timing protocol (warm-up, repeats, median, percentile bootstrap);
- the mechanistic cost-accuracy curve (Crank-Nicolson baseline over a
  ``steps`` grid, accuracy and latency per point);
- the Pareto-efficient frontier with log-log interpolation and NO
  extrapolation;
- the IQM (trimmed mean, Agarwal et al. 2021) with its two-stage paired
  bootstrap (seeds first, sample ids second, one index set shared by every
  test set so the resolution axis stays paired);
- the preregistered decision rules (equal runtime, equal accuracy, baseline
  floor, no claim outside the measured range);
- the break-even query count;
- the run manifest with its SHA-256.

numpy only. Instruments only: nothing here trains, tunes or scores a model on
T1 data. Every result of these functions is a measurement of the instrument
on the numbers it was given, never a scientific claim.
"""

from __future__ import annotations

__all__ = [
    "PROTOCOL_ID",
    "PREREGISTERED_CONFIDENCE",
    "PREREGISTERED_BOOTSTRAP_RESAMPLES",
    "PREREGISTERED_IQM_PROPORTION",
    "TimingProtocol",
    "LatencyMeasurement",
    "measure_latency",
    "CostAccuracyPoint",
    "baseline_cost_accuracy_curve",
    "pareto_frontier",
    "error_at_time",
    "time_at_error",
    "interquartile_mean",
    "BootstrapSummary",
    "PairingError",
    "paired_bootstrap_iqm",
    "PREREGISTERED_SEEDS",
    "PREREGISTERED_TEST_SAMPLES",
    "PREREGISTERED_TRAINING_SEEDS",
    "PREREGISTERED_TEST_SETS",
    "RuleVerdict",
    "DecisionSummary",
    "DecisionInputError",
    "DecisionProvenanceError",
    "decide_test_set",
    "PreregisteredDecision",
    "decide_preregistered",
    "verify_preregistered_decision",
    "break_even_queries",
    "RunManifestError",
    "file_sha256",
    "source_revision",
    "build_run_manifest",
    "FLOAT32_UNIT_ROUNDOFF",
    "FLOAT64_UNIT_ROUNDOFF",
    "cdist_rounding_margin",
    "RadiusStabilityVerdict",
    "radius_graph_stability",
]

import dataclasses
import hashlib
import json
import math
import numbers
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from heat_operator_task_hex_v3 import (
    HeatOperatorTaskConfig,
    evaluate_predictions,
    mechanistic_baseline,
)
from hex_fv_solver_hex_v3 import HexLattice

PROTOCOL_ID = "PHASE1_PROTOCOL_v1"
_MANIFEST_SCHEMA = "hexantk_phase1_run_manifest_v1"
_DEFAULT_STEPS_GRID = (1, 2, 4, 8, 16, 32, 64)
# The CG tolerance of the baseline. The preregistration names the method (:128-133,
# ``mechanistic_baseline`` as verified in Phase 0) but writes out no tolerance; this is
# the verified default, and decide_preregistered requires it (Equalita R5-F1; whether
# the preregistration should name it is an owner question).
_REGISTERED_BASELINE_RTOL = 1e-12
# The wall clock of the timing protocol (spec section 1, ``measure_latency``'s default). The
# producer takes any ``clock`` (tests pass a fake one); a point timed with another clock carries
# its name and is refused for a decision (self-check of the R5-F1 fix, same class: a producer
# parameter that sets the time values).
_REGISTERED_CLOCK = "time.perf_counter"
# The preregistered statistic, single source for every default below and for
# the binding in ``decide_test_set``: spec/PREREGISTRATION_PHASE1_v1.md
# section 5 ("the IQM of the five seed-level mean errors; its 95 % interval
# uses a two-stage paired bootstrap with 10 000 resamples") and section 6 ("the
# upper 95 % CI bound"); spec/PHASE1_PROTOCOL_v1.md sections 1, 4 and 5.
PREREGISTERED_CONFIDENCE = 0.95
PREREGISTERED_BOOTSTRAP_RESAMPLES = 10_000
PREREGISTERED_IQM_PROPORTION = 0.25
# Section 3 [D2] seeds 0-4 and [D3] 500 samples per test set: a summary over a
# subset of seeds or samples is another statistic (Equalita R2-P2-1).
PREREGISTERED_SEEDS = 5
PREREGISTERED_TEST_SAMPLES = 500
# The seeds and test sets themselves (preregistration section 2, [D2] training
# seeds 0-4, [D3] sample-id table): the decision-grade entry
# ``decide_preregistered`` accepts raw errors only for these.
PREREGISTERED_TRAINING_SEEDS = (0, 1, 2, 3, 4)
PREREGISTERED_TEST_SETS: Mapping[str, Tuple[str, Tuple[int, ...], range]] = MappingProxyType({
    "id": ("triangle", (32,), range(100000, 100500)),
    "resolution_ood": ("triangle", (16, 64, 128), range(100000, 100500)),
    "composite_ood": ("torus", (32,), range(200000, 200500)),
})
# The registered domain names above as the lattice spells them (HexLattice.domain,
# heat_operator_task_hex_v3._DOMAINS): a baseline point carries the lattice's spelling.
_LATTICE_DOMAIN: Mapping[str, str] = MappingProxyType({"triangle": "triangle_dirichlet", "torus": "torus_periodic"})
_DECISION_SCHEMA = "hexantk_phase1_decision_record_v1"
_AUTHENTICITY = ("unattested: the record_sha256 is a self-digest; it shows that the record follows from "
                 "the named raw bytes, not who produced those bytes (no signature, no pipeline attestation)")
# \Z, not $: `$` also matches before a final "\n", so "a" * 40 + "\n" passed (HNT-R86-03, #82).
_HEX40 = re.compile(r"^[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"^[0-9a-f]{64}\Z")
_ROOT = Path(__file__).resolve().parents[2]
# Run-manifest content fixed by spec/PREREGISTRATION_PHASE1_v1.md (Equalita R3 F2): the
# file itself (:222), the train pool 0..1999 of the [D3] table (:66) and D7 (:94).
_PREREGISTRATION_FILE = _ROOT / "spec" / "PREREGISTRATION_PHASE1_v1.md"
_REGISTERED_TRAIN_POOL = range(0, 2000)
_REGISTERED_NEURALOPERATOR = "2.0.0"


# ------------------------------------------------------------------
# Validation helpers (fail closed, spec section 8)
# ------------------------------------------------------------------


def _is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _finite_float(value: Any, *, name: str) -> float:
    if not _is_real(value):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return result


def _non_negative_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return result


def _int_at_least(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


def _confidence(value: Any) -> float:
    result = _finite_float(value, name="confidence")
    if not 0.0 < result < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {value!r}")
    return result


def _proportion(value: Any) -> float:
    result = _finite_float(value, name="proportion")
    if not 0.0 <= result < 0.5:
        raise ValueError(f"proportion must lie in [0, 0.5), got {value!r}")
    return result


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.bool_ or np.iscomplexobj(arr) or not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _positive_vector(value: Any, *, name: str) -> np.ndarray:
    arr = _finite_vector(value, name=name)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be > 0 (values are interpolated on a log scale)")
    return arr


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


def _percentile_interval(stats: np.ndarray, confidence: float) -> Tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(low), float(high)


# ------------------------------------------------------------------
# Section 1: timing protocol
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TimingProtocol:
    """Warm-up, repeats and the bootstrap of the median latency (spec section 1)."""

    warmup: int = 3
    repeats: int = 10
    bootstrap_resamples: int = PREREGISTERED_BOOTSTRAP_RESAMPLES
    seed: int = 0
    confidence: float = PREREGISTERED_CONFIDENCE

    def __post_init__(self) -> None:
        _int_at_least(self.warmup, name="warmup", minimum=0)
        _int_at_least(self.repeats, name="repeats", minimum=1)
        _int_at_least(self.bootstrap_resamples, name="bootstrap_resamples", minimum=1)
        _int_at_least(self.seed, name="seed", minimum=0)
        _confidence(self.confidence)

    def canonical(self) -> Dict[str, Any]:
        return {
            "warmup": int(self.warmup),
            "repeats": int(self.repeats),
            "bootstrap_resamples": int(self.bootstrap_resamples),
            "seed": int(self.seed),
            "confidence": float(self.confidence),
        }

    def digest(self) -> str:
        """SHA-256 of ``canonical()`` (as ``HeatOperatorTaskConfig.digest``); a baseline point carries it."""
        return hashlib.sha256(_canonical_json(self.canonical())).hexdigest()


@dataclass(frozen=True)
class LatencyMeasurement:
    """Timed calls of one callable: the median and its percentile bootstrap interval."""

    times_s: Tuple[float, ...]
    median_s: float
    ci_low_s: float
    ci_high_s: float
    protocol: TimingProtocol

    def to_dict(self) -> Dict[str, Any]:
        return {
            "times_s": list(self.times_s),
            "median_s": self.median_s,
            "ci_low_s": self.ci_low_s,
            "ci_high_s": self.ci_high_s,
            "protocol": self.protocol.canonical(),
        }


def _bootstrap_median_rows(rows: np.ndarray, protocol: TimingProtocol) -> Tuple[float, float]:
    """Cluster bootstrap of the median of all entries of ``rows`` (C clusters x R calls).

    Clusters are resampled with replacement; each drawn cluster contributes all
    its calls. With one cluster per call (C = repeats, R = 1) this is the plain
    bootstrap of the median over the calls.
    """
    clusters, calls = rows.shape
    rng = _rng(protocol.seed)
    index = rng.integers(0, clusters, size=(protocol.bootstrap_resamples, clusters))
    stats = np.empty(protocol.bootstrap_resamples, dtype=np.float64)
    chunk = max(1, 2_000_000 // (clusters * calls))
    for start in range(0, protocol.bootstrap_resamples, chunk):
        block = index[start:start + chunk]
        stats[start:start + block.shape[0]] = np.median(
            rows[block].reshape(block.shape[0], clusters * calls), axis=1
        )
    return _percentile_interval(stats, protocol.confidence)


def measure_latency(
    fn: Callable[[], Any],
    protocol: TimingProtocol = TimingProtocol(),
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> LatencyMeasurement:
    """``warmup`` untimed calls, then ``repeats`` timed calls; median + bootstrap interval."""
    if not callable(fn):
        raise TypeError("fn must be callable")
    if not isinstance(protocol, TimingProtocol):
        raise TypeError("protocol must be a TimingProtocol")
    if not callable(clock):
        raise TypeError("clock must be callable")
    for _ in range(protocol.warmup):
        fn()
    times = []
    for _ in range(protocol.repeats):
        start = clock()
        fn()
        stop = clock()
        elapsed = _finite_float(stop, name="clock reading") - _finite_float(start, name="clock reading")
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError(f"timed call reported a duration of {elapsed!r} s; the clock is not monotone")
        times.append(elapsed)
    arr = np.asarray(times, dtype=np.float64)
    low, high = _bootstrap_median_rows(arr[:, None], protocol)
    return LatencyMeasurement(
        times_s=tuple(times), median_s=float(np.median(arr)), ci_low_s=low, ci_high_s=high,
        protocol=protocol,
    )


# ------------------------------------------------------------------
# Section 2: mechanistic cost-accuracy curve
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CostAccuracyPoint:
    """One ``(N, steps)`` point of the baseline's cost-accuracy curve (spec section 2).

    The point carries its own measurement coordinates (lattice ``n`` and
    ``domain``, ``steps``, ``sample_count``, ``timed_samples``), written by
    ``baseline_cost_accuracy_curve`` from the lattice and inputs it ran on.
    ``decide_preregistered`` checks them against the preregistration
    (Equalita R4-F1). Since Equalita R5-F1 the point also carries the settings
    that determine its values besides the lattice and the inputs: the task
    configuration (``kappa``, ``t_final`` and ``task_config_sha256``, the
    digest of the whole ``HeatOperatorTaskConfig``), the solver tolerance
    ``rtol``, and the timing protocol (``warmup``, ``repeats``,
    ``timing_protocol_sha256`` and ``clock``, the name of the clock the timed
    calls were read with: ``"time.perf_counter"`` or ``"custom:<name>"``).
    The fields after ``cg_iterations_mean`` have
    no default, so a construction from before they were added fails loudly.
    """

    n: int
    steps: int
    dt: float
    sample_count: int
    timed_samples: int
    mean_rel_l2: float
    median_rel_l2: float
    mean_rel_h1: float
    mean_rel_max: float
    median_wall_time_s: float
    wall_time_ci_s: Tuple[float, float]
    cg_iterations_mean: float
    domain: str
    kappa: float
    t_final: float
    task_config_sha256: str
    rtol: float
    warmup: int
    repeats: int
    timing_protocol_sha256: str
    clock: str

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["wall_time_ci_s"] = list(self.wall_time_ci_s)
        return out


def _fields(value: Any, lattice: HexLattice, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.bool_ or np.iscomplexobj(arr) or not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array")
    arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != lattice.num_nodes:
        raise ValueError(f"{name} must have shape [S, {lattice.num_nodes}] with S >= 1, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _steps_grid(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) == 0:
        raise TypeError("steps_grid must be a non-empty sequence of ints")
    grid = tuple(_int_at_least(s, name="steps", minimum=1) for s in value)
    if len(set(grid)) != len(grid):
        raise ValueError("steps_grid must not repeat a value")
    return grid


def _clock_name(clock: Callable[[], float]) -> str:
    """``"time.perf_counter"`` for the wall clock of section 1, else ``"custom:<name>"`` (self-check of R5-F1)."""
    if clock is time.perf_counter:
        return _REGISTERED_CLOCK
    return f"custom:{getattr(clock, '__qualname__', type(clock).__qualname__)}"


def baseline_cost_accuracy_curve(
    config: HeatOperatorTaskConfig,
    u0: Any,
    truth: Any,
    lattice: HexLattice,
    *,
    steps_grid: Sequence[int] = _DEFAULT_STEPS_GRID,
    protocol: TimingProtocol = TimingProtocol(),
    timed_samples: Optional[int] = None,
    rtol: float = _REGISTERED_BASELINE_RTOL,
    clock: Callable[[], float] = time.perf_counter,
) -> Tuple[CostAccuracyPoint, ...]:
    """The mechanistic baseline on every input, for every ``steps``: accuracy and latency per point."""
    if not isinstance(config, HeatOperatorTaskConfig):
        raise TypeError("config must be a HeatOperatorTaskConfig")
    if not isinstance(lattice, HexLattice) or lattice.domain != config.domain:
        raise ValueError(f"lattice must be a {config.domain} HexLattice")
    if not isinstance(protocol, TimingProtocol):
        raise TypeError("protocol must be a TimingProtocol")
    inputs = _fields(u0, lattice, name="u0")
    targets = _fields(truth, lattice, name="truth")
    if inputs.shape != targets.shape:
        raise ValueError("u0 and truth must have the same shape")
    grid = _steps_grid(steps_grid)
    samples = inputs.shape[0]
    timed = samples if timed_samples is None else _int_at_least(timed_samples, name="timed_samples", minimum=1)
    if timed > samples:
        raise ValueError(f"timed_samples={timed} exceeds the {samples} samples")
    rtol = _positive_float(rtol, name="rtol")
    # Equalita R5-F1: the settings the values depend on travel with every point, read once here.
    settings = {
        "kappa": float(config.kappa), "t_final": float(config.t_final), "task_config_sha256": config.digest(),
        "rtol": rtol, "warmup": int(protocol.warmup), "repeats": int(protocol.repeats),
        "timing_protocol_sha256": protocol.digest(), "clock": _clock_name(clock),
    }
    points = []
    for steps in grid:
        predictions = np.empty_like(inputs)
        iterations = np.empty(samples, dtype=np.float64)
        rows = np.empty((timed, protocol.repeats), dtype=np.float64)
        for i in range(samples):
            holder: Dict[str, Any] = {}

            def run(index: int = i, holder: Dict[str, Any] = holder) -> None:
                holder["out"] = mechanistic_baseline(config, inputs[index], lattice, steps=steps, rtol=rtol)

            if i < timed:
                rows[i] = measure_latency(run, protocol, clock=clock).times_s
            else:
                run()
            prediction, info = holder["out"]
            predictions[i] = prediction
            iterations[i] = float(info["cg_iterations"])
        metrics = evaluate_predictions(predictions, targets, lattice)
        rel_l2 = np.asarray(metrics["rel_l2"], dtype=np.float64)
        low, high = _bootstrap_median_rows(rows, protocol)
        points.append(CostAccuracyPoint(
            n=int(lattice.n), steps=int(steps), dt=settings["t_final"] / steps,
            sample_count=int(samples), timed_samples=int(timed),
            mean_rel_l2=float(np.mean(rel_l2)), median_rel_l2=float(np.median(rel_l2)),
            mean_rel_h1=float(np.mean(metrics["rel_h1"])), mean_rel_max=float(np.mean(metrics["rel_max"])),
            median_wall_time_s=float(np.median(rows)), wall_time_ci_s=(low, high),
            cg_iterations_mean=float(np.mean(iterations)), domain=str(lattice.domain), **settings,
        ))
    return tuple(points)


# ------------------------------------------------------------------
# Section 3: Pareto frontier and log-log interpolation (no extrapolation)
# ------------------------------------------------------------------


def _frontier_arrays(times: Any, errors: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = _positive_vector(times, name="times")
    e = _positive_vector(errors, name="errors")
    if t.shape != e.shape:
        raise ValueError("times and errors must have the same length")
    order = np.lexsort((e, t))  # by time, then by error
    keep = []
    best = math.inf
    for k in order:
        if e[k] < best:
            keep.append(int(k))
            best = e[k]
    index = np.asarray(keep, dtype=np.int64)
    return index, t[index], e[index]


def pareto_frontier(times: Any, errors: Any) -> np.ndarray:
    """Indices of the non-dominated points, sorted by time (error strictly decreasing)."""
    return _frontier_arrays(times, errors)[0]


def error_at_time(times: Any, errors: Any, t: Any) -> Optional[float]:
    """Frontier error at time ``t`` by log-log interpolation; ``None`` outside the measured range."""
    _, ft, fe = _frontier_arrays(times, errors)
    t = _positive_float(t, name="t")
    if t < ft[0] or t > ft[-1]:
        return None
    hit = np.flatnonzero(ft == t)
    if hit.size:  # a measured point answers exactly, without log/exp rounding
        return float(fe[hit[0]])
    return float(math.exp(np.interp(math.log(t), np.log(ft), np.log(fe))))


def time_at_error(times: Any, errors: Any, e: Any) -> Optional[float]:
    """Frontier time reaching error ``e`` by log-log interpolation; ``None`` outside the measured range."""
    _, ft, fe = _frontier_arrays(times, errors)
    e = _positive_float(e, name="e")
    if e < fe[-1] or e > fe[0]:
        return None
    hit = np.flatnonzero(fe == e)
    if hit.size:  # a measured point answers exactly, without log/exp rounding
        return float(ft[hit[0]])
    # np.interp needs increasing abscissae: the frontier error decreases with time.
    return float(math.exp(np.interp(math.log(e), np.log(fe[::-1]), np.log(ft[::-1]))))


# ------------------------------------------------------------------
# Section 4: IQM and the two-stage paired bootstrap
# ------------------------------------------------------------------


def _iqm_rows(rows: np.ndarray, proportion: float) -> np.ndarray:
    """Row-wise trimmed mean: drop int(proportion * n) values at each end."""
    n = rows.shape[1]
    cut = int(proportion * n)
    ordered = np.sort(rows, axis=1)
    return np.mean(ordered[:, cut:n - cut], axis=1)


def interquartile_mean(values: Any, proportion: float = PREREGISTERED_IQM_PROPORTION) -> float:
    """Trimmed mean (Agarwal et al. 2021): for five values the mean of the middle three."""
    arr = _finite_vector(values, name="values")
    return float(_iqm_rows(arr[None, :], _proportion(proportion))[0])


@dataclass(frozen=True)
class BootstrapSummary:
    """Point estimate and percentile interval of the IQM of the seed-level means."""

    point: float
    low: float
    high: float
    n_seeds: int
    n_samples: int
    n_resamples: int
    seed: int
    confidence: float
    proportion: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PairingError(ValueError):
    """Error matrices were to share bootstrap index sets without sharing their sample ids.

    The same resampled column index means the same sample only if column ``j``
    holds the same ``sample_id`` in every matrix. Equal array shapes do not
    make a physical pairing: the triangle test sets share ids 100000..100499,
    the torus OOD set uses 200000..200499 (GPT-6 R2, 2026-10-19).
    """


def _sample_id_list(value: Any, *, name: str, length: int) -> Tuple[int, ...]:
    out = _distinct_ints(value, name=name)
    if len(out) != length:
        raise ValueError(f"{name} names {len(out)} values for {length} rows or columns")
    return out


def _paired_sample_ids(table: Mapping[str, np.ndarray], sample_ids: Any) -> Optional[Tuple[int, ...]]:
    """Column ids shared by every matrix; ``PairingError`` unless the pairing is by id."""
    n_samples = next(iter(table.values())).shape[1]
    if sample_ids is None:
        if len(table) > 1:
            raise PairingError(
                f"{len(table)} error matrices would share one sample-id draw, but no sample_ids were "
                "given; equal shapes are not a pairing. Pass sample_ids per test set, or bootstrap "
                "the test sets separately.")
        return None
    if not isinstance(sample_ids, Mapping):
        if len(table) > 1:
            raise PairingError("several error matrices need sample_ids as a mapping test set -> ids")
        return _sample_id_list(sample_ids, name="sample_ids", length=n_samples)
    if set(sample_ids) != set(table):
        raise ValueError(f"sample_ids must name exactly the test sets {sorted(table)}, got {sorted(sample_ids)}")
    lists = {name: _sample_id_list(sample_ids[name], name=f"sample_ids[{name!r}]", length=n_samples)
             for name in table}
    first_name, first = next(iter(lists.items()))
    for name, ids in lists.items():
        if ids != first:
            same_set = set(ids) == set(first)
            raise PairingError(
                f"test sets {first_name!r} and {name!r} "
                + ("hold the same sample ids in a different column order; align the columns first"
                   if same_set else "do not share their sample ids")
                + " -- one sample-id draw for both would pair different samples")
    return first


def _error_table(errors: Any) -> Dict[str, np.ndarray]:
    if isinstance(errors, Mapping):
        items = list(errors.items())
        if not items:
            raise ValueError("errors must name at least one test set")
    else:
        items = [("", errors)]
    table: Dict[str, np.ndarray] = {}
    shape = None
    for name, value in items:
        if not isinstance(name, str):
            raise TypeError("test-set names must be strings")
        arr = np.asarray(value)
        if arr.dtype == np.bool_ or np.iscomplexobj(arr) or not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"errors[{name!r}] must be a real numeric array")
        # C order: numpy's mean along a row sums pairwise only over contiguous
        # memory, so a Fortran-ordered or index-permuted view of the same values
        # rounded the point estimate differently (found 2026-10-21, HEX-P1-01 fix).
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"errors[{name!r}] must be a non-empty [seeds, samples] matrix, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"errors[{name!r}] must be finite")
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError("every error matrix must share the seed axis and the sample-id axis (same shape)")
        table[name] = arr
    return table


def paired_bootstrap_iqm(
    errors: Any,
    *,
    sample_ids: Any = None,
    n_resamples: int = PREREGISTERED_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    confidence: float = PREREGISTERED_CONFIDENCE,
    proportion: float = PREREGISTERED_IQM_PROPORTION,
    chunk_elements: int = 4_000_000,
) -> Dict[str, BootstrapSummary]:
    """IQM of the seed-level means with a two-stage paired bootstrap (spec section 4).

    ``errors``: one ``[seeds, samples]`` matrix, or a mapping test set -> matrix
    (all the same shape). Seeds are resampled first, then sample ids; the same
    index sets are applied to every matrix. ``chunk_elements`` bounds memory
    only; the result depends on ``(seed, n_resamples, S, M)`` alone.

    ``sample_ids`` names the sample id of every column: one sequence for a
    single matrix, a mapping test set -> sequence for several. Several
    matrices are paired only when every one holds the same ids in the same
    column order; without ``sample_ids``, or with different ids, they raise
    ``PairingError`` (equal shape is no pairing).
    """
    table = _error_table(errors)
    _paired_sample_ids(table, sample_ids)
    n_resamples = _int_at_least(n_resamples, name="n_resamples", minimum=1)
    seed = _int_at_least(seed, name="seed", minimum=0)
    confidence = _confidence(confidence)
    proportion = _proportion(proportion)
    chunk_elements = _int_at_least(chunk_elements, name="chunk_elements", minimum=1)
    n_seeds, n_samples = next(iter(table.values())).shape
    rng = _rng(seed)
    seed_index = rng.integers(0, n_seeds, size=(n_resamples, n_seeds))
    sample_index = rng.integers(0, n_samples, size=(n_resamples, n_samples))
    stats = {name: np.empty(n_resamples, dtype=np.float64) for name in table}
    chunk = max(1, chunk_elements // (n_seeds * n_samples))
    for start in range(0, n_resamples, chunk):
        rows = seed_index[start:start + chunk]
        cols = sample_index[start:start + chunk]
        for name, arr in table.items():
            resampled = arr[rows[:, :, None], cols[:, None, :]]  # (b, S, M)
            stats[name][start:start + rows.shape[0]] = _iqm_rows(resampled.mean(axis=2), proportion)
    out: Dict[str, BootstrapSummary] = {}
    for name, arr in table.items():
        low, high = _percentile_interval(stats[name], confidence)
        out[name] = BootstrapSummary(
            point=interquartile_mean(arr.mean(axis=1), proportion), low=low, high=high,
            n_seeds=int(n_seeds), n_samples=int(n_samples), n_resamples=int(n_resamples),
            seed=int(seed), confidence=float(confidence), proportion=float(proportion),
        )
    return out


# ------------------------------------------------------------------
# Section 5: decision rules
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RuleVerdict:
    rule: str
    evaluable: bool
    claim: bool
    reason: str
    gino_upper_rel_l2: float
    gino_latency_s: float
    baseline_value: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionSummary:
    """DIAGNOSTIC verdict of both rules on a caller-supplied summary (``decide_test_set``).

    ``gino`` is the bootstrap summary the rules used; ``to_dict`` carries it so
    the confidence level, the resample count and the trimming proportion stay
    visible next to the verdict, and marks the record ``"grade": "diagnostic"``:
    nothing ties the summary to measured errors, so this is never a
    preregistered decision (HEX-P1-01). The decision-grade record is
    ``PreregisteredDecision`` from ``decide_preregistered``.
    """

    test_set: str
    n: int
    verdict: str
    rules: Tuple[RuleVerdict, ...]
    gino: BootstrapSummary

    def to_dict(self) -> Dict[str, Any]:
        return {"grade": "diagnostic", "test_set": self.test_set, "n": self.n, "verdict": self.verdict,
                "rules": [r.to_dict() for r in self.rules], "gino": self.gino.to_dict()}


class DecisionInputError(ValueError):
    """The GINO summary handed to the decision rule is not the preregistered statistic.

    Raised when the confidence level, the resample count, the trimming
    proportion, the seed count or the per-test-set sample count differ from the
    preregistered values, or when the summary is not a finite, non-negative,
    ordered interval (``0 <= low <= point <= high``). A rule
    evaluated on another statistic would be a different, unregistered test.
    ``decide_preregistered`` raises it too for raw data outside the registered
    seeds, test sets, sample ids or lattices.
    """


class DecisionProvenanceError(ValueError):
    """A decision record or run manifest does not follow from the data it names.

    Raised when a ``PreregisteredDecision`` changed after it was computed, when
    a serialised record's self-digest does not match its body, when the record
    differs from the one recomputed from the raw errors, or when the run
    manifest was edited after its digest was taken (HEX-P1-01).
    """


def _preregistered_summary(gino: BootstrapSummary) -> Tuple[float, float, float]:
    """``(low, point, high)`` of ``gino`` after binding it to the preregistered statistic."""
    binding = (
        ("confidence", gino.confidence, PREREGISTERED_CONFIDENCE, numbers.Real),
        ("n_resamples", gino.n_resamples, PREREGISTERED_BOOTSTRAP_RESAMPLES, numbers.Integral),
        ("proportion", gino.proportion, PREREGISTERED_IQM_PROPORTION, numbers.Real),
        ("n_seeds", gino.n_seeds, PREREGISTERED_SEEDS, numbers.Integral),
        ("n_samples", gino.n_samples, PREREGISTERED_TEST_SAMPLES, numbers.Integral),
    )
    for name, value, expected, kind in binding:
        # Compare as Python numbers: under NEP 50 np.float32(0.95) == 0.95 is True
        # although its value is 0.949999988 (Equalita R2 P3).
        if (isinstance(value, bool) or not isinstance(value, kind)
                or (float(value) if kind is numbers.Real else int(value)) != expected):
            raise DecisionInputError(
                f"gino.{name} = {value!r}, but the preregistered decision statistic uses "
                f"{name} = {expected!r} (spec/PREREGISTRATION_PHASE1_v1.md sections 3, 5-6)")
    bounds = []
    for name in ("low", "point", "high"):
        value = getattr(gino, name)
        if not _is_real(value) or not math.isfinite(float(value)):
            raise DecisionInputError(f"gino.{name} must be a finite real number, got {value!r}")
        bounds.append(float(value))
    low, point, high = bounds
    if not 0.0 <= low <= point <= high:
        raise DecisionInputError(
            f"gino must satisfy 0 <= low <= point <= high (rel_l2 is non-negative), "
            f"got low={low!r}, point={point!r}, high={high!r}")
    return low, point, high


def decide_test_set(
    *,
    test_set: str,
    n: int,
    gino: BootstrapSummary,
    gino_latency_s: Any,
    baseline_times_s: Any,
    baseline_errors: Any,
) -> DecisionSummary:
    """DIAGNOSTIC low-level helper: both rules on a caller-supplied summary (spec section 5).

    Not a preregistered decision (HEX-P1-01, GPT-6 2026-10-21): ``gino`` is a
    public frozen dataclass, and its labels are checked, not its origin. A
    ``dataclasses.replace(summary, low=..., point=..., high=...)`` with intact
    labels passes and can turn ``H0`` into ``advantage``. The result is marked
    ``"grade": "diagnostic"``; the decision-grade entry is
    ``decide_preregistered``, which takes the raw errors and computes the
    summary itself.

    ``gino`` must carry the preregistered labels (confidence
    ``PREREGISTERED_CONFIDENCE``, ``PREREGISTERED_BOOTSTRAP_RESAMPLES`` resamples,
    trimming ``PREREGISTERED_IQM_PROPORTION``, 5 seeds, 500 samples) with a finite
    ordered interval; anything else raises ``DecisionInputError`` before a rule
    is evaluated.
    """
    if not isinstance(test_set, str) or not test_set:
        raise TypeError("test_set must be a non-empty string")
    n = _int_at_least(n, name="n", minimum=1)
    if not isinstance(gino, BootstrapSummary):
        raise TypeError("gino must be a BootstrapSummary")
    # The rule uses the validated value, not a second read of gino.high (TOCTOU,
    # Equalita R2 P3).
    _, _, high = _preregistered_summary(gino)
    upper = _positive_float(high, name="gino.high")
    latency = _positive_float(gino_latency_s, name="gino_latency_s")
    runtime, accuracy, verdict = _rules(upper, latency, baseline_times_s, baseline_errors)
    return DecisionSummary(test_set=test_set, n=n, verdict=verdict, rules=(runtime, accuracy), gino=gino)


def _rules(
    upper: float, latency: float, baseline_times_s: Any, baseline_errors: Any,
) -> Tuple[RuleVerdict, RuleVerdict, str]:
    """Equal runtime, equal accuracy and the verdict on a validated upper bound and latency."""
    _, ft, fe = _frontier_arrays(baseline_times_s, baseline_errors)
    # "Measured range" of preregistration section 6 is read as the range of the
    # Pareto-efficient part (the interpolation never leaves it). A dominated
    # point can extend the measured times or errors beyond it; the reasons name
    # both ranges so the narrower, conservative one is visible (E3, 2026-10-20).
    measured_t = _positive_vector(baseline_times_s, name="baseline_times_s")
    measured_e = _positive_vector(baseline_errors, name="baseline_errors")

    e_base = error_at_time(ft, fe, latency)
    if e_base is None:
        runtime = RuleVerdict(
            "equal_runtime", False, False,
            f"GINO latency {latency:.6g} s lies outside the time range of the baseline's Pareto "
            f"frontier [{ft[0]:.6g}, {ft[-1]:.6g}] s (all measured points: "
            f"[{float(measured_t.min()):.6g}, {float(measured_t.max()):.6g}] s); "
            "no equal-runtime claim (no extrapolation)",
            upper, latency, None)
    else:
        claim = upper < e_base
        runtime = RuleVerdict(
            "equal_runtime", True, claim,
            (f"upper IQM bound {upper:.6g} {'<' if claim else '>='} baseline error "
             f"{e_base:.6g} at {latency:.6g} s"),
            upper, latency, e_base)

    floor, coarsest = float(fe[-1]), float(fe[0])
    if upper < floor:
        accuracy = RuleVerdict(
            "equal_accuracy", False, False,
            f"baseline floor: the measured baseline does not reach rel_l2 {upper:.6g} "
            f"(floor {floor:.6g}); no speed claim",
            upper, latency, None)
    elif upper > coarsest:
        accuracy = RuleVerdict(
            "equal_accuracy", False, False,
            f"rel_l2 {upper:.6g} lies above the coarsest error of the baseline's Pareto frontier "
            f"{coarsest:.6g} (all measured points up to {float(measured_e.max()):.6g}); "
            "no equal-accuracy claim (no extrapolation)",
            upper, latency, None)
    else:
        t_base = time_at_error(ft, fe, upper)
        assert t_base is not None  # inside [floor, coarsest] by construction
        claim = latency < t_base
        accuracy = RuleVerdict(
            "equal_accuracy", True, claim,
            (f"GINO latency {latency:.6g} s {'<' if claim else '>='} baseline time "
             f"{t_base:.6g} s at rel_l2 {upper:.6g}"),
            upper, latency, t_base)

    verdict = "advantage" if (runtime.claim or accuracy.claim) else "H0"
    return runtime, accuracy, verdict


# ------------------------------------------------------------------
# Section 5a: the decision-grade entry (HEX-P1-01)
# ------------------------------------------------------------------


def _canonical_json(body: Any) -> bytes:
    return json.dumps(body, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _array_sha256(arr: np.ndarray) -> str:
    """SHA-256 of a float array's canonical bytes: little-endian float64, C order, -0.0 as 0.0, shape header."""
    canon = np.ascontiguousarray((np.asarray(arr, dtype=np.float64) + 0.0).astype("<f8"))
    header = _canonical_json({"dtype": "<f8", "order": "C", "shape": list(canon.shape)})
    return hashlib.sha256(header + b"\n" + canon.tobytes()).hexdigest()


def _distinct_ints(value: Any, *, name: str) -> Tuple[int, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (Sequence, np.ndarray, range)):
        raise TypeError(f"{name} must be a sequence of non-negative ints")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, numbers.Integral) or item < 0:
            raise TypeError(f"{name} must contain non-negative ints, got {item!r}")
        out.append(int(item))
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must not repeat a value")
    return tuple(out)


def _verified_run_manifest(run_manifest: Any) -> Dict[str, Any]:
    """The manifest rebuilt from its own fields; ``DecisionProvenanceError`` if it was edited after digesting."""
    if not isinstance(run_manifest, Mapping):
        raise DecisionProvenanceError("run_manifest must be the mapping returned by build_run_manifest")
    fields = {k: v for k, v in run_manifest.items() if k not in ("schema", "protocol", "manifest_sha256")}
    try:
        rebuilt = build_run_manifest(fields)
    except RunManifestError as exc:
        raise DecisionProvenanceError(f"run_manifest is not a valid run manifest: {exc}") from exc
    if (run_manifest.get("schema") != _MANIFEST_SCHEMA or run_manifest.get("protocol") != PROTOCOL_ID
            or run_manifest.get("manifest_sha256") != rebuilt["manifest_sha256"]):
        raise DecisionProvenanceError(
            "run_manifest does not match its manifest_sha256 (edited after build_run_manifest, "
            "or not built by it)")
    return rebuilt


def _check_registered_manifest_content(manifest: Mapping[str, Any], test_set: str) -> None:
    """Equalita R3 F2: the manifest fields the preregistration fixes; ``DecisionProvenanceError`` otherwise.

    Only what spec/PREREGISTRATION_PHASE1_v1.md states is checked (file:line in each
    message). Not checked, because the preregistration does not fix it: the machine
    name (D5 still open, :263), the metric spec file behind ``metric_spec_sha256``
    (:230 "the instrument spec", PHASE1_PROTOCOL_v1.md section 7 names another file),
    the exact 1800/200 split (needs ``split_dataset_by_trajectory``), the torch version.
    """
    pool = _REGISTERED_TRAIN_POOL
    for part in ("train", "val"):
        outside = [i for i in manifest["split_ids"][part] if i not in pool]
        if outside:
            # :66 train pool 0..1999; :71-75 the split comes from the pool and the disjoint
            # id ranges keep every test set out of the split and out of model selection.
            raise DecisionProvenanceError(
                f"run_manifest split_ids[{part!r}] holds ids outside the registered train pool "
                f"{pool.start}..{pool.stop - 1} (preregistration :66, :71-75), e.g. {outside[:3]}")
    if test_set not in manifest["datasets"]:
        # :87-88 and :223: the dataset_sha256 of every set goes into the manifest.
        raise DecisionProvenanceError(
            f"run_manifest datasets name no digest for the decided test set {test_set!r} "
            "(preregistration :87-88, :223)")
    try:
        registered = file_sha256(_PREREGISTRATION_FILE)
    except OSError as exc:
        raise DecisionProvenanceError(f"the preregistration file cannot be read: {exc}") from exc
    if manifest["preregistration"]["sha256"] != registered:
        # :222 "this file's version and SHA-256".
        raise DecisionProvenanceError(
            f"run_manifest preregistration sha256 {manifest['preregistration']['sha256']} is not the "
            f"file_sha256 {registered} of {_PREREGISTRATION_FILE.name} (preregistration :222)")
    hardware = manifest["hardware"]
    if hardware["cuda"] is not False:
        # :140-143 same CPU machine for the algorithmic comparison; :262 D5 "compute = CPU, no GPU".
        raise DecisionProvenanceError(
            f"run_manifest hardware cuda = {hardware['cuda']!r}; the registered compute is CPU without "
            "GPU, recorded as cuda = False (preregistration :140-143, :262)")
    if hardware["dtype"] != "float32":
        # :103 "Precision: float32 for training".
        raise DecisionProvenanceError(
            f"run_manifest hardware dtype = {hardware['dtype']!r}; registered is float32 "
            "(preregistration :103)")
    for key in ("amp", "autocast"):
        # :103 float32 for training (no mixed precision); :122 and :228 dtype/AMP go into the
        # manifest, so `amp` must be recorded, as False (Equalita R4-F3). An `autocast` entry,
        # where present, must be False too. Other spellings are not recognised (owner, R4-F5).
        if (key == "amp" or key in hardware) and hardware.get(key) is not False:
            raise DecisionProvenanceError(
                f"run_manifest hardware {key} = {hardware.get(key, '<missing>')!r}; registered is float32 "
                f"without AMP, recorded as {key} = False (preregistration :103, :122, :228)")
    if manifest["backend"].get("neuraloperator") != _REGISTERED_NEURALOPERATOR:
        # :94 D7 neuraloperator==2.0.0.
        raise DecisionProvenanceError(
            f"run_manifest backend neuraloperator = {manifest['backend'].get('neuraloperator')!r}; "
            f"registered is {_REGISTERED_NEURALOPERATOR} (preregistration :94, D7)")


def _latency_times(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.bool_ or np.iscomplexobj(arr) or not np.issubdtype(arr.dtype, np.number):
        raise TypeError("gino_latency_times_s must be a real numeric array of timed calls")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim not in (1, 2) or arr.size == 0:
        raise ValueError(f"gino_latency_times_s must be a non-empty 1-D or 2-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("gino_latency_times_s must be finite and > 0")
    # HEXA-TIM-01: the preregistration measures `repeats` timed calls per sample (spec section 1);
    # 1-D is one sample, 2-D is samples x repeats. Checked last, so it masks no check above.
    repeats = TimingProtocol().repeats
    if arr.shape[-1] != repeats:
        raise DecisionInputError(
            f"gino_latency_times_s must hold the {repeats} preregistered timed repeats per sample, "
            f"shape ({repeats},) or (samples, {repeats}), got shape {arr.shape}")
    return arr


def _point_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise DecisionInputError(f"{name} must be an int, got {value!r}")
    return int(value)


def _point_float(value: Any, *, name: str, minimum: Optional[float] = None, strict: bool = False) -> float:
    """One conversion to a plain float; finite, and ``> minimum`` (strict) or ``>= minimum``."""
    if not _is_real(value):
        raise DecisionInputError(f"{name} must be a finite real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and (result <= minimum if strict else result < minimum)):
        bound = "" if minimum is None else f" {'>' if strict else '>='} {minimum:g}"
        raise DecisionInputError(f"{name} must be a finite real number{bound}, got {value!r}")
    return result


def _point_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise DecisionInputError(f"{name} must be a str, got {value!r}")
    # str.__str__ copies the characters into a plain str: a subclass's __eq__/__ne__/__str__ do not reach
    # the checks or the record (Equalita R5-F3).
    return str.__str__(value)


# Read order of one point: every field exactly once, converted to a plain value (Equalita R5-F3).
_POINT_INTS = ("n", "steps", "sample_count", "timed_samples", "warmup", "repeats")
_POINT_STRS = ("domain", "task_config_sha256", "timing_protocol_sha256", "clock")
# (field, lower bound, strict): R5-F4 plausibility of the fields that only enter the record.
_POINT_FLOATS = (
    ("dt", 0.0, True), ("kappa", 0.0, True), ("t_final", 0.0, True), ("rtol", 0.0, True),
    ("mean_rel_l2", 0.0, True), ("median_wall_time_s", 0.0, True),
    ("median_rel_l2", 0.0, False), ("mean_rel_h1", 0.0, False), ("mean_rel_max", 0.0, False),
    ("cg_iterations_mean", 0.0, False),
)


def _point_copy(point: "CostAccuracyPoint") -> Dict[str, Any]:
    """The point as plain ints, floats and strs, each field read once; checks and record use only this copy."""
    copy: Dict[str, Any] = {}
    for field in _POINT_INTS:
        copy[field] = _point_int(getattr(point, field), name=f"baseline_curve {field}")
    for field in _POINT_STRS:
        copy[field] = _point_str(getattr(point, field), name=f"baseline_curve {field}")
    for field, minimum, strict in _POINT_FLOATS:
        copy[field] = _point_float(getattr(point, field), name=f"baseline_curve {field}", minimum=minimum,
                                   strict=strict)
    ci = point.wall_time_ci_s
    if isinstance(ci, (str, bytes)) or not isinstance(ci, Sequence):
        raise DecisionInputError(f"baseline_curve wall_time_ci_s must be a (low, high) pair, got {ci!r}")
    ci = tuple(ci)
    if len(ci) != 2:
        raise DecisionInputError(f"baseline_curve wall_time_ci_s must be a (low, high) pair, got {ci!r}")
    low, high = (_point_float(c, name="baseline_curve wall_time_ci_s", minimum=0.0) for c in ci)
    if low > high:
        # R5-F4: an interval is ordered. Whether it contains the median is not required: the
        # percentile bootstrap does not guarantee it, and a genuine curve must not be refused.
        raise DecisionInputError(f"baseline_curve wall_time_ci_s ({low!r}, {high!r}) is not ordered low <= high")
    copy["wall_time_ci_s"] = [low, high]
    if set(copy) != {f.name for f in dataclasses.fields(CostAccuracyPoint)}:
        raise AssertionError("_point_copy does not cover every CostAccuracyPoint field")  # pragma: no cover
    return copy


def _registered_baseline_points(curve: Any, *, test_set: str, n: int) -> List[Dict[str, Any]]:
    """Equalita R4-F1: every baseline point checked by its own coordinates; the record's point list.

    R3 bound only the number of values, R4 found that a genuine curve measured on another
    steps grid (1, 8, 16, 24, 32, 48, 64) or on another lattice (n = 64 under n = 32) still
    reached "advantage" while the record named the registered steps and n. Here the
    coordinates come with each point (``baseline_cost_accuracy_curve`` writes them from the
    lattice and inputs it ran on) and are compared with the preregistration; the record is
    written from the checked points. R5 found the same class on the settings: a genuine curve
    solved with kappa = 2 or T = 0.02 reached "advantage" too; the point now also carries the
    task configuration, the tolerance and the timing protocol, checked here. Each point is read
    once into plain values (R5-F3); the checks and the record use only that copy. Types and
    plausibility come first (R5-F4), then the coordinates in a fixed order (steps, n, domain,
    kappa, t_final, dt, task configuration digest, rtol, warm-ups, repeats, timing protocol
    digest, clock, sample count, timed samples), so each refusal names the first one that differs.
    """
    if isinstance(curve, (str, bytes, Mapping, np.ndarray)) or not isinstance(curve, Sequence):
        curve = None
    else:
        # One snapshot for every check and for the record: a sequence that yields other points on
        # a second pass must not have one set checked and another recorded (self-check, TOCTOU).
        curve = tuple(curve)
    if curve is None or any(type(p) is not CostAccuracyPoint for p in curve):
        raise TypeError(
            "baseline_curve must be the tuple of CostAccuracyPoint returned by baseline_cost_accuracy_curve; "
            "bare time/error lists carry no steps, lattice or sample count and are not accepted "
            "(API change 2026-10-21, Equalita R4-F1)")
    grid = _DEFAULT_STEPS_GRID
    domain = _LATTICE_DOMAIN[PREREGISTERED_TEST_SETS[test_set][0]]
    # Equalita R5-F1: the settings a point must have been computed with. [D1] task configuration
    # of the test set's domain (preregistration :51-55), timing protocol 3 warm-ups / 10 measured
    # repeats with the registered bootstrap (:145, spec section 1), the baseline's CG tolerance.
    registered_config = HeatOperatorTaskConfig(domain=domain)
    registered_timing = TimingProtocol()
    # R5-F3: every field of every point read once into plain values; checks and record use only these.
    points = [_point_copy(p) for p in curve]
    steps = tuple(p["steps"] for p in points)
    if steps != grid:
        # :134-137 the baseline runs with steps 1, 2, 4, 8, 16, 32, 64. A subset, another grid
        # or another order is another frontier (a chord lies above a convex curve).
        raise DecisionInputError(
            f"baseline_curve steps {list(steps)} are not the registered steps {list(grid)}, one point "
            "each in that order (preregistration :134-137)")
    for p in points:
        at = f"baseline_curve point steps={p['steps']}"
        if p["n"] != n:
            # :134-135 the curve runs on the evaluation lattice N_eval itself.
            raise DecisionInputError(
                f"{at} was measured on lattice n = {p['n']}, the decision is "
                f"for n = {n} (preregistration :134-135, the evaluation lattice)")
        if p["domain"] != domain:
            raise DecisionInputError(
                f"{at} was measured on domain {p['domain']!r}, test set "
                f"{test_set!r} is registered on {domain!r} (preregistration :62-68 [D3])")
        for field in ("kappa", "t_final"):
            # R5-F1: a solver kappa = 2 or T = 0.02 against the registered truth reached "advantage".
            registered = float(getattr(registered_config, field))
            if p[field] != registered:
                raise DecisionInputError(
                    f"{at} was solved with {field} = {p[field]!r}; the preregistered task configuration "
                    f"has {field} = {registered!r} (preregistration :51-55 [D1])")
        if p["dt"] != p["t_final"] / p["steps"]:
            # mechanistic_baseline steps with dt = T / steps (spec/HEAT_OPERATOR_TASK_v1.md section 6).
            raise DecisionInputError(
                f"{at} has dt = {p['dt']!r}, not t_final / steps = {p['t_final'] / p['steps']!r}")
        if p["task_config_sha256"] != registered_config.digest():
            # The whole [D1] configuration, lambda_cutoff and spectral decay included.
            raise DecisionInputError(
                f"{at} carries task configuration digest {p['task_config_sha256']!r}, not the digest of the "
                f"preregistered [D1] configuration {registered_config.canonical()} (preregistration :51-55)")
        if p["rtol"] != _REGISTERED_BASELINE_RTOL:
            raise DecisionInputError(
                f"{at} was solved with rtol = {p['rtol']!r}; the baseline's tolerance is "
                f"{_REGISTERED_BASELINE_RTOL!r} (mechanistic_baseline as verified, preregistration :128-133)")
        for field in ("warmup", "repeats"):
            registered = int(getattr(registered_timing, field))
            if p[field] != registered:
                raise DecisionInputError(
                    f"{at} was timed with {field} = {p[field]}; the preregistered timing protocol has "
                    f"3 warm-up runs and 10 measured repeats, {field} = {registered} (preregistration :145)")
        if p["timing_protocol_sha256"] != registered_timing.digest():
            # Bootstrap resamples, seed and confidence of wall_time_ci_s (spec section 1).
            raise DecisionInputError(
                f"{at} carries timing protocol digest {p['timing_protocol_sha256']!r}, not the digest of "
                f"the preregistered protocol {registered_timing.canonical()} (preregistration :145)")
        if p["clock"] != _REGISTERED_CLOCK:
            raise DecisionInputError(
                f"{at} was timed with clock {p['clock']!r}; the latency is wall time read with "
                f"{_REGISTERED_CLOCK!r} (spec section 1, preregistration :145)")
        if p["sample_count"] != PREREGISTERED_TEST_SAMPLES:
            # :62-68 [D3] 500 ids per test set; :136 "on the same test inputs".
            raise DecisionInputError(
                f"{at} has sample_count {p['sample_count']}; the curve runs on "
                f"the {PREREGISTERED_TEST_SAMPLES} registered test inputs (preregistration :62-68, :136)")
        if not 1 <= p["timed_samples"] <= p["sample_count"]:
            raise DecisionInputError(
                f"{at} has timed_samples {p['timed_samples']}, outside 1..{p['sample_count']}")
    times = [p["median_wall_time_s"] for p in points]
    errors = [p["mean_rel_l2"] for p in points]
    # Self-check after R3 F1: two genuine points copied to seven pass a count check. This
    # catches only the literal repetition of a value; copies shifted by 1 ulp or 1e-12, or
    # chord points, pass (Equalita R4-F2): the values themselves cannot show where they were
    # measured, only a recomputation or an attestation can (claim boundary, authenticity).
    if len(set(times)) != len(times) or len(set(errors)) != len(errors):
        raise DecisionInputError(
            "baseline_curve must not repeat a time or an error: one measured point per registered "
            f"steps value {list(grid)} (preregistration :134-137)")
    return points


@dataclass(frozen=True)
class PreregisteredDecision:
    """Decision-grade verdict for one test set and lattice, computed from raw errors.

    Produced only by ``decide_preregistered``. ``record_sha256`` is the digest of
    the canonical record; ``to_dict`` recomputes it and raises
    ``DecisionProvenanceError`` if any field was replaced afterwards
    (``dataclasses.replace`` included). The digest protects against drift, not
    against a deliberate forger who also fabricates the raw bytes: that needs a
    signature or a pipeline attestation (``provenance["authenticity"]``).
    """

    test_set: str
    n: int
    verdict: str
    rules: Tuple[RuleVerdict, ...]
    gino: BootstrapSummary
    gino_latency_s: float
    provenance_json: str
    record_sha256: str

    @property
    def provenance(self) -> Dict[str, Any]:
        return json.loads(self.provenance_json)

    def _body(self) -> Dict[str, Any]:
        return {
            "schema": _DECISION_SCHEMA,
            "protocol": PROTOCOL_ID,
            "grade": "preregistered",
            "test_set": self.test_set,
            "domain": PREREGISTERED_TEST_SETS.get(self.test_set, (None,))[0],
            "n": self.n,
            "verdict": self.verdict,
            "rules": [r.to_dict() for r in self.rules],
            "gino": self.gino.to_dict(),
            "gino_latency_s": self.gino_latency_s,
            "provenance": self.provenance,
        }

    def to_dict(self) -> Dict[str, Any]:
        body = self._body()
        digest = hashlib.sha256(_canonical_json(body)).hexdigest()
        if digest != self.record_sha256:
            raise DecisionProvenanceError(
                "the decision changed after decide_preregistered computed it (record_sha256 mismatch)")
        return {**body, "record_sha256": digest}


def decide_preregistered(
    *,
    test_set: str,
    n: int,
    errors: Any,
    sample_ids: Any,
    seeds: Any,
    run_manifest: Mapping[str, Any],
    gino_latency_times_s: Any,
    baseline_curve: Any,
) -> PreregisteredDecision:
    """Both preregistered rules from GINO's raw per-sample errors (spec section 5a, HEX-P1-01).

    - ``errors``: the ``[5, 500]`` ``rel_l2`` table of one test set and lattice,
      row ``i`` = training seed ``seeds[i]``, column ``j`` = ``sample_ids[j]``;
    - ``seeds`` must be the registered training seeds, ``sample_ids`` the
      registered ids of ``test_set``, ``n`` one of its registered lattices;
    - ``run_manifest`` is the output of ``build_run_manifest`` (git SHA,
      dataset digests, split ids, backend, hardware, seed set) and must still
      match its digest; its seed set must equal ``seeds``, and the content the
      preregistration fixes must hold (``_check_registered_manifest_content``:
      split ids in the train pool, the test set's dataset digest, the
      preregistration file's digest, CPU without CUDA, float32 without AMP,
      neuraloperator 2.0.0), else ``DecisionProvenanceError``;
    - ``gino_latency_times_s``: the timed calls, exactly ``TimingProtocol().repeats``
      (10) per sample: shape (10,) for one sample or (samples, 10); any other count
      raises ``DecisionInputError``. The latency is their median, as for the
      baseline curve. Warm-ups are not visible in these numbers.
    - ``baseline_curve``: the tuple of ``CostAccuracyPoint`` returned by
      ``baseline_cost_accuracy_curve`` (Equalita R4-F1, API change 2026-10-21:
      the former two float lists ``baseline_times_s``/``baseline_errors`` are
      no longer parameters, and a sequence of anything but ``CostAccuracyPoint``
      raises ``TypeError``). Every point carries its measurement coordinates and
      each is checked against the preregistration, else ``DecisionInputError``:
      the ``steps`` of the points, in order, are exactly 1, 2, 4, ..., 64
      (preregistration :134-137); every point's ``n`` is the decided lattice
      and its ``domain`` the test set's domain; every ``sample_count`` is the
      500 registered test inputs (:62-68 [D3], :136 "on the same test
      inputs"); ``1 <= timed_samples <= sample_count``. Since Equalita R5-F1
      also the settings: ``kappa`` and ``t_final`` of [D1] (:51-55), ``dt ==
      t_final / steps``, ``task_config_sha256`` the digest of the [D1]
      configuration of the domain, ``rtol`` the baseline's 1e-12, ``warmup``
      3 and ``repeats`` 10 (:145), ``timing_protocol_sha256`` the digest of
      ``TimingProtocol()`` and ``clock`` ``"time.perf_counter"``. Every field is read once into a plain value
      (R5-F3) and checked for plausibility (R5-F4: finite, errors, times and
      iteration counts ``>= 0``, the interval ordered). The record's baseline
      fields are this checked copy, never constants or arguments.

    Rows and columns are put into ascending seed and sample-id order, so the
    result does not depend on the caller's order, and the bootstrap and the
    IQM are computed here with the preregistered settings. No parameter
    accepts a summary of GINO's errors; the baseline curve enters as its
    per-point summaries (mean ``rel_l2``, median wall time; spec section 2).
    The record binds SHA-256 digests of the canonical raw
    bytes, the sample ids, the latency calls and the run manifest, every
    baseline point with its own coordinates, and the seed list;
    ``verify_preregistered_decision`` recomputes it from the raw data.
    """
    if not isinstance(test_set, str) or test_set not in PREREGISTERED_TEST_SETS:
        raise DecisionInputError(
            f"test_set must be one of the registered test sets {sorted(PREREGISTERED_TEST_SETS)}, got {test_set!r}")
    _, lattices, registered_ids = PREREGISTERED_TEST_SETS[test_set]
    n = _int_at_least(n, name="n", minimum=1)
    if n not in lattices:
        raise DecisionInputError(f"test set {test_set!r} is registered on N in {list(lattices)}, got N = {n}")
    if isinstance(errors, Mapping):
        raise TypeError("errors must be the single [seeds, samples] table of this test set and lattice")
    raw = _error_table(errors)[""]
    if raw.shape != (PREREGISTERED_SEEDS, PREREGISTERED_TEST_SAMPLES):
        raise DecisionInputError(
            f"errors must have the registered shape ({PREREGISTERED_SEEDS}, {PREREGISTERED_TEST_SAMPLES}) "
            f"(seeds x sample ids), got {raw.shape}")
    if np.any(raw < 0.0):
        raise DecisionInputError("errors must be >= 0 (rel_l2 is non-negative)")
    seed_list = _distinct_ints(seeds, name="seeds")
    if sorted(seed_list) != list(PREREGISTERED_TRAINING_SEEDS) or len(seed_list) != raw.shape[0]:
        raise DecisionInputError(
            f"seeds must be the registered training seeds {list(PREREGISTERED_TRAINING_SEEDS)} "
            f"in row order, got {list(seed_list)}")
    ids = _sample_id_list(sample_ids, name="sample_ids", length=raw.shape[1])
    if sorted(ids) != list(registered_ids):
        raise DecisionInputError(
            f"sample_ids must be the registered ids {registered_ids.start}..{registered_ids.stop - 1} "
            f"of test set {test_set!r}")
    manifest = _verified_run_manifest(run_manifest)
    if sorted(manifest["seeds"]) != sorted(seed_list):
        raise DecisionProvenanceError(
            f"run_manifest seeds {manifest['seeds']} differ from the error table's seeds {list(seed_list)}")
    _check_registered_manifest_content(manifest, test_set)
    latency_calls = _latency_times(gino_latency_times_s)
    points = _registered_baseline_points(baseline_curve, test_set=test_set, n=n)
    times = np.array([p["median_wall_time_s"] for p in points], dtype=np.float64)
    curve_errors = np.array([p["mean_rel_l2"] for p in points], dtype=np.float64)

    rows = np.argsort(np.asarray(seed_list), kind="stable")
    cols = np.argsort(np.asarray(ids), kind="stable")
    canonical = np.ascontiguousarray(raw[rows][:, cols])
    canonical_ids = tuple(sorted(ids))
    summary = paired_bootstrap_iqm({test_set: canonical}, sample_ids={test_set: canonical_ids})[test_set]
    _, _, high = _preregistered_summary(summary)
    upper = _positive_float(high, name="gino.high")
    latency = float(np.median(latency_calls))
    runtime, accuracy, verdict = _rules(upper, latency, times, curve_errors)

    provenance = {
        "raw_errors_sha256": _array_sha256(canonical),
        "raw_errors_layout": "rel_l2 [seed, sample], rows by ascending seed, columns by ascending sample_id",
        "seeds": sorted(seed_list),
        "sample_ids_sha256": hashlib.sha256(_canonical_json(list(canonical_ids))).hexdigest(),
        "sample_id_range": [canonical_ids[0], canonical_ids[-1]],
        "gino_latency_times_sha256": _array_sha256(latency_calls),
        "gino_latency_times_shape": list(latency_calls.shape),
        # Written from the checked points (Equalita R4-F1), not from _DEFAULT_STEPS_GRID or n.
        "baseline_points": points,
        "baseline_times_s": [p["median_wall_time_s"] for p in points],
        "baseline_errors": [p["mean_rel_l2"] for p in points],
        "baseline_steps": [p["steps"] for p in points],
        "baseline_n": sorted({p["n"] for p in points}),
        "run_manifest_sha256": manifest["manifest_sha256"],
        "git_sha": manifest["git_sha"],
        "preregistration": manifest["preregistration"],
        "datasets": manifest["datasets"],
        "backend": manifest["backend"],
        "hardware": manifest["hardware"],
        "authenticity": _AUTHENTICITY,
    }
    provenance_json = _canonical_json(provenance).decode("utf-8")
    draft = PreregisteredDecision(
        test_set=test_set, n=n, verdict=verdict, rules=(runtime, accuracy), gino=summary,
        gino_latency_s=latency, provenance_json=provenance_json, record_sha256="")
    digest = hashlib.sha256(_canonical_json(draft._body())).hexdigest()
    return dataclasses.replace(draft, record_sha256=digest)


def verify_preregistered_decision(record: Any, **raw: Any) -> PreregisteredDecision:
    """Recompute a serialised decision record from its raw data; ``DecisionProvenanceError`` on any difference.

    ``record`` is ``PreregisteredDecision.to_dict()`` (also after a JSON round
    trip); ``raw`` are the keyword arguments of ``decide_preregistered``. First
    the self-digest is checked (drift), then the whole record is compared with
    the one recomputed from the raw data (a summary, verdict or digest that does
    not follow from them). A forger who also fabricates the raw bytes is not
    caught here (``provenance["authenticity"]``).
    """
    if not isinstance(record, Mapping) or not isinstance(record.get("record_sha256"), str):
        raise DecisionProvenanceError("record must be a mapping with a record_sha256")
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    try:
        payload = _canonical_json(body)
    except (TypeError, ValueError) as exc:
        raise DecisionProvenanceError(f"record is not canonical JSON: {exc}") from exc
    if hashlib.sha256(payload).hexdigest() != record["record_sha256"]:
        raise DecisionProvenanceError("record_sha256 does not match the record body (edited after it was computed)")
    recomputed = decide_preregistered(**raw)
    expected = recomputed.to_dict()
    if _canonical_json(expected) != _canonical_json(dict(record)):
        differing = sorted(k for k in set(expected) | set(record) if expected.get(k) != record.get(k))
        raise DecisionProvenanceError(f"record does not follow from the raw data; differing fields: {differing}")
    return recomputed


# ------------------------------------------------------------------
# Section 6: break-even
# ------------------------------------------------------------------


def break_even_queries(
    training_time_s: Any,
    data_generation_time_s: Any,
    gino_latency_s: Any,
    baseline_latency_s: Any,
) -> Optional[int]:
    """Smallest q with q * (baseline - gino) >= training + data; None if GINO is not faster."""
    training = _non_negative_float(training_time_s, name="training_time_s")
    data = _non_negative_float(data_generation_time_s, name="data_generation_time_s")
    gino = _positive_float(gino_latency_s, name="gino_latency_s")
    baseline = _positive_float(baseline_latency_s, name="baseline_latency_s")
    if baseline <= gino:
        return None
    quotient = (training + data) / (baseline - gino)
    if math.isfinite(quotient):
        return int(math.ceil(quotient))  # the v1 float evaluation, unchanged wherever it is finite
    # HNT-R86-04: finite inputs, yet the sum or the quotient overflowed and ceil(inf) raised a
    # raw OverflowError. Evaluate exactly on the given binary64 values instead (Python ints
    # are unbounded); the count is then astronomically large, never a practical budget.
    fixed = Fraction(training) + Fraction(data)
    saving = Fraction(baseline) - Fraction(gino)
    return -((-fixed) // saving)


# ------------------------------------------------------------------
# Section 7: run manifest
# ------------------------------------------------------------------


class RunManifestError(ValueError):
    """A run manifest field is missing, empty, mistyped or not representable."""


_REQUIRED_MANIFEST_KEYS = (
    "git_sha", "preregistration", "datasets", "split_ids", "backend", "config_digest",
    "seeds", "hardware", "reference_spec_ids", "metric_spec_sha256", "result_digests",
)
_OPTIONAL_MANIFEST_KEYS = ("extra",)
_REQUIRED_HARDWARE_KEYS = ("machine", "threads", "dtype", "cuda")


def file_sha256(path: Any) -> str:
    """Full SHA-256 (hex) of a file's LF-normalised bytes, as the provenance gate hashes."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def source_revision(root: Optional[Path] = None) -> Dict[str, Any]:
    """Git HEAD and whether the work tree is dirty; raises RunManifestError if git cannot answer."""
    root = _ROOT if root is None else Path(root)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True, stdin=subprocess.DEVNULL).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                text=True, check=True, stdin=subprocess.DEVNULL).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunManifestError(f"git did not answer in {root}: {exc}") from exc
    if not _HEX40.match(head):
        raise RunManifestError(f"git HEAD is not a 40-hex sha: {head!r}")
    return {"git_sha": head, "worktree_dirty": bool(status.strip())}


def _hex(value: Any, *, name: str, pattern: re.Pattern) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise RunManifestError(f"{name} must be a lowercase hex digest of the expected length, got {value!r}")
    return value


def _non_empty_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunManifestError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _str_mapping(value: Any, *, name: str, allow_empty: bool = False) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise RunManifestError(f"{name} must be a mapping, got {type(value).__name__}")
    if not value and not allow_empty:
        raise RunManifestError(f"{name} must not be empty")
    out = {}
    for key, item in value.items():
        out[_non_empty_str(key, name=f"{name} key")] = _non_empty_str(item, name=f"{name}[{key!r}]")
    return out


def _int_list(value: Any, *, name: str, distinct: bool) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) == 0:
        raise RunManifestError(f"{name} must be a non-empty list of non-negative ints")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, numbers.Integral) or item < 0:
            raise RunManifestError(f"{name} must contain non-negative ints, got {item!r}")
        out.append(int(item))
    if distinct and len(set(out)) != len(out):
        raise RunManifestError(f"{name} must not repeat a value")
    return out


def _json_safe(value: Any, *, name: str) -> Any:
    """Plain JSON data only (no NaN/inf, no foreign objects), returned as a copy."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise RunManifestError(f"{name} must be finite, got {value!r}")
        return result
    if isinstance(value, Mapping):
        return {_non_empty_str(k, name=f"{name} key"): _json_safe(v, name=f"{name}[{k!r}]")
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, name=f"{name}[{i}]") for i, v in enumerate(value)]
    raise RunManifestError(f"{name} holds a non-JSON value of type {type(value).__name__}")


def build_run_manifest(fields: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the preregistration's run record (spec section 7) and digest it."""
    if not isinstance(fields, Mapping):
        raise RunManifestError("fields must be a mapping")
    unknown = sorted(set(fields) - set(_REQUIRED_MANIFEST_KEYS) - set(_OPTIONAL_MANIFEST_KEYS))
    if unknown:
        raise RunManifestError(f"unknown manifest keys: {unknown}")
    missing = [k for k in _REQUIRED_MANIFEST_KEYS if k not in fields]
    if missing:
        raise RunManifestError(f"missing manifest keys: {missing}")

    prereg = fields["preregistration"]
    if not isinstance(prereg, Mapping) or set(prereg) != {"version", "sha256"}:
        raise RunManifestError("preregistration must be a mapping with exactly 'version' and 'sha256'")
    split = fields["split_ids"]
    if not isinstance(split, Mapping) or set(split) != {"train", "val"}:
        raise RunManifestError("split_ids must be a mapping with exactly 'train' and 'val'")
    train = _int_list(split["train"], name="split_ids['train']", distinct=True)
    val = _int_list(split["val"], name="split_ids['val']", distinct=True)
    if set(train) & set(val):
        raise RunManifestError("split_ids['train'] and split_ids['val'] must be disjoint")
    hardware = fields["hardware"]
    if not isinstance(hardware, Mapping) or any(k not in hardware for k in _REQUIRED_HARDWARE_KEYS):
        raise RunManifestError(f"hardware must contain {list(_REQUIRED_HARDWARE_KEYS)}")
    ids = fields["reference_spec_ids"]
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence) or len(ids) == 0:
        raise RunManifestError("reference_spec_ids must be a non-empty list of strings")

    body: Dict[str, Any] = {
        "schema": _MANIFEST_SCHEMA,
        "protocol": PROTOCOL_ID,
        "git_sha": _hex(fields["git_sha"], name="git_sha", pattern=_HEX40),
        "preregistration": {
            "version": _non_empty_str(prereg["version"], name="preregistration['version']"),
            "sha256": _hex(prereg["sha256"], name="preregistration['sha256']", pattern=_HEX64),
        },
        "datasets": {k: _hex(v, name=f"datasets[{k!r}]", pattern=_HEX64)
                     for k, v in _str_mapping(fields["datasets"], name="datasets").items()},
        "split_ids": {"train": train, "val": val},
        "backend": _str_mapping(fields["backend"], name="backend"),
        "config_digest": _hex(fields["config_digest"], name="config_digest", pattern=_HEX64),
        "seeds": _int_list(fields["seeds"], name="seeds", distinct=True),
        "hardware": _json_safe(hardware, name="hardware"),
        "reference_spec_ids": [_non_empty_str(i, name="reference_spec_ids item") for i in ids],
        "metric_spec_sha256": _hex(fields["metric_spec_sha256"], name="metric_spec_sha256", pattern=_HEX64),
        "result_digests": _str_mapping(fields["result_digests"], name="result_digests", allow_empty=True),
    }
    if "extra" in fields:
        extra = fields["extra"]
        if not isinstance(extra, Mapping):
            raise RunManifestError("extra must be a mapping")
        body["extra"] = _json_safe(extra, name="extra")
    payload = json.dumps(body, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    return body


# ------------------------------------------------------------------
# Section 11: D4 decision support -- float-order stability of a radius graph
# (GPT-6 HEX-P2-02). A tool for the owner; it decides nothing and changes
# neither the adapter's tie band (1e-6 r) nor any preregistered threshold.
# ------------------------------------------------------------------


FLOAT32_UNIT_ROUNDOFF = 2.0 ** -24
FLOAT64_UNIT_ROUNDOFF = 2.0 ** -53


def _gamma(k: int, u: float) -> float:
    """Higham's gamma_k = k u / (1 - k u)."""
    if k * u >= 1.0:
        raise ValueError(f"gamma_{k} is undefined for unit roundoff {u!r}")
    return k * u / (1.0 - k * u)


def cdist_rounding_margin(
    radius: Any,
    *,
    max_point_norm: Any,
    unit_roundoff: Any = FLOAT32_UNIT_ROUNDOFF,
    input_abs_error: Any = 0.0,
) -> float:
    """Relative margin above which a computed distance decides ``d <= radius`` exactly, in ANY summation order.

    Model (ATen ``_euclidean_dist``, the matmul path of ``torch.cdist`` that
    neuraloperator's native neighbour search uses for more than 25 points):
    ``s = [-2x, |x|^2, 1] . [y, 1, |y|^2]``, then ``sqrt``. With IEEE
    arithmetic of unit roundoff ``u``, Higham (2002, eq. 3.5 and lemma 3.3)
    bounds the computed square for every summation order (blocked, FMA or
    not) by ``|s_hat - s| <= gamma_6 (|x| + |y|)^2 <= gamma_6 (2 R)^2 = eps``,
    ``R = max_point_norm``; the square root adds a relative ``u``.
    ``input_abs_error`` bounds ``|x_hat - x| + |y_hat - y|`` (rounding of the
    coordinates themselves); ``max_point_norm`` bounds the norm of the
    coordinates as computed. The comparison ``d_hat <= r`` runs against the
    radius rounded to the working precision (torch casts the Python scalar to
    the tensor dtype), ``|r_hat - r| <= u r``.

    A pair with ``d >= r (1 + m)`` stays a non-edge if
    ``(1 - u) sqrt((r (1 + m) - e_in)^2 - eps) > r (1 + u)``; a pair with
    ``d <= r (1 - m)`` stays an edge if
    ``(1 + u) sqrt((r (1 - m) + e_in)^2 + eps) <= r (1 - u)``. Both are
    monotone in ``d``; the returned ``m`` is the larger of the two boundary
    values, so every pair with ``|d - r| > m r`` is decided as in exact
    arithmetic. ``ValueError`` if no margin below 1 can guarantee it.
    """
    r = _positive_float(radius, name="radius")
    big_r = _positive_float(max_point_norm, name="max_point_norm")
    u = _non_negative_float(unit_roundoff, name="unit_roundoff")
    e_in = _non_negative_float(input_abs_error, name="input_abs_error")
    eps = _gamma(6, u) * (2.0 * big_r) ** 2
    above = (e_in + math.sqrt((r * (1.0 + u) / (1.0 - u)) ** 2 + eps)) / r - 1.0
    inner = (r * (1.0 - u) / (1.0 + u)) ** 2 - eps
    if inner <= 0.0:
        raise ValueError(f"rounding of size {eps!r} on r^2 = {r * r!r} can turn any edge into a non-edge")
    below = 1.0 - (math.sqrt(inner) - e_in) / r
    margin = max(above, below, 0.0)
    if not margin < 1.0:  # NaN fails closed as well
        raise ValueError(f"no margin below 1 guarantees an exact decision (got {margin!r})")
    return margin


@dataclass(frozen=True)
class RadiusStabilityVerdict:
    """Whether a radius candidate may enter the D4 grid under a pre-specified float error budget.

    ``status``: ``"stable"`` (margin above the budget and no measured flip
    contradicting it: no flip for any summation
    order within the budget), ``"frozen_order_required"`` (not stable, but
    measured flip-free on one frozen hardware/kernel order, which must then be
    pinned in the run manifest), ``"refused"`` otherwise. Margin and budget are
    relative to the radius.
    """

    status: str
    min_relative_margin: float
    budget_relative: float
    margin_over_budget: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def radius_graph_stability(
    min_relative_margin: Any,
    *,
    budget_relative: Any,
    frozen_order_flips: Optional[int] = None,
) -> RadiusStabilityVerdict:
    """D4 criterion of HEX-P2-02: a candidate is stable only if ``min |d - r| / r`` exceeds the float budget.

    ``budget_relative`` is the float error budget relative to ``r``, fixed
    BEFORE the candidates are compared (for instance ``cdist_rounding_margin``
    for an order-independent bound, or an owner-specified measured budget).
    Equality is not stable, and neither is a margin above the budget when
    ``frozen_order_flips > 0`` (the measurement contradicts the budget:
    ``"refused"``). ``frozen_order_flips`` is the flip count measured on
    the one frozen hardware/kernel order the owner pins (``None``: no frozen
    order). Every argument is validated; NaN or inf raises ``ValueError``.
    This is not a tie band: the adapter's ``1e-6 r`` stays as it is and is
    never widened after the fact.
    """
    margin = _non_negative_float(min_relative_margin, name="min_relative_margin")
    budget = _positive_float(budget_relative, name="budget_relative")
    if frozen_order_flips is not None:
        frozen_order_flips = _int_at_least(frozen_order_flips, name="frozen_order_flips", minimum=0)
    ratio = margin / budget
    if margin > budget and frozen_order_flips:
        # Equalita R3 F4: flips measured on the frozen order contradict the budget the margin
        # clears (the budget or the measurement is wrong); "stable" would be fail-open.
        return RadiusStabilityVerdict(
            "refused", margin, budget, ratio,
            f"min |d - r| / r = {margin:.6g} exceeds the float budget {budget:.6g}, but "
            f"{frozen_order_flips} measured flips on the frozen order contradict that budget")
    if margin > budget:
        return RadiusStabilityVerdict(
            "stable", margin, budget, ratio,
            f"min |d - r| / r = {margin:.6g} exceeds the float budget {budget:.6g}: no flip within the budget")
    if frozen_order_flips == 0:
        return RadiusStabilityVerdict(
            "frozen_order_required", margin, budget, ratio,
            f"min |d - r| / r = {margin:.6g} does not exceed the float budget {budget:.6g}; flip-free only on "
            "the measured frozen hardware/kernel order, which the run manifest must pin")
    detail = ("no frozen order given" if frozen_order_flips is None
              else f"{frozen_order_flips} flips on the frozen order")
    return RadiusStabilityVerdict(
        "refused", margin, budget, ratio,
        f"min |d - r| / r = {margin:.6g} does not exceed the float budget {budget:.6g} ({detail})")
