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
    "paired_bootstrap_iqm",
    "RuleVerdict",
    "DecisionSummary",
    "DecisionInputError",
    "decide_test_set",
    "break_even_queries",
    "RunManifestError",
    "file_sha256",
    "source_revision",
    "build_run_manifest",
]

import hashlib
import json
import math
import numbers
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

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
# The preregistered statistic, single source for every default below and for
# the binding in ``decide_test_set``: spec/PREREGISTRATION_PHASE1_v1.md
# section 5 ("the IQM of the five seed-level mean errors; its 95 % interval
# uses a two-stage paired bootstrap with 10 000 resamples") and section 6 ("the
# upper 95 % CI bound"); spec/PHASE1_PROTOCOL_v1.md sections 1, 4 and 5.
PREREGISTERED_CONFIDENCE = 0.95
PREREGISTERED_BOOTSTRAP_RESAMPLES = 10_000
PREREGISTERED_IQM_PROPORTION = 0.25
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ROOT = Path(__file__).resolve().parents[2]


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
    """One ``(N, steps)`` point of the baseline's cost-accuracy curve (spec section 2)."""

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


def baseline_cost_accuracy_curve(
    config: HeatOperatorTaskConfig,
    u0: Any,
    truth: Any,
    lattice: HexLattice,
    *,
    steps_grid: Sequence[int] = _DEFAULT_STEPS_GRID,
    protocol: TimingProtocol = TimingProtocol(),
    timed_samples: Optional[int] = None,
    rtol: float = 1e-12,
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
            n=int(lattice.n), steps=int(steps), dt=float(config.t_final) / steps,
            sample_count=int(samples), timed_samples=int(timed),
            mean_rel_l2=float(np.mean(rel_l2)), median_rel_l2=float(np.median(rel_l2)),
            mean_rel_h1=float(np.mean(metrics["rel_h1"])), mean_rel_max=float(np.mean(metrics["rel_max"])),
            median_wall_time_s=float(np.median(rows)), wall_time_ci_s=(low, high),
            cg_iterations_mean=float(np.mean(iterations)),
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
        arr = np.asarray(arr, dtype=np.float64)
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
    """
    table = _error_table(errors)
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
    """Verdict of both rules for one test set and lattice, with the statistic it rests on.

    ``gino`` is the bootstrap summary the rules used; ``to_dict`` carries it so
    the confidence level, the resample count and the trimming proportion stay
    visible next to the verdict.
    """

    test_set: str
    n: int
    verdict: str
    rules: Tuple[RuleVerdict, ...]
    gino: BootstrapSummary

    def to_dict(self) -> Dict[str, Any]:
        return {"test_set": self.test_set, "n": self.n, "verdict": self.verdict,
                "rules": [r.to_dict() for r in self.rules], "gino": self.gino.to_dict()}


class DecisionInputError(ValueError):
    """The GINO summary handed to the decision rule is not the preregistered statistic.

    Raised when the confidence level, the resample count or the trimming
    proportion differ from the preregistered values, or when the summary is
    not a finite, ordered interval (``low <= point <= high``). A rule
    evaluated on another statistic would be a different, unregistered test.
    """


def _preregistered_summary(gino: BootstrapSummary) -> Tuple[float, float, float]:
    """``(low, point, high)`` of ``gino`` after binding it to the preregistered statistic."""
    binding = (
        ("confidence", gino.confidence, PREREGISTERED_CONFIDENCE, numbers.Real),
        ("n_resamples", gino.n_resamples, PREREGISTERED_BOOTSTRAP_RESAMPLES, numbers.Integral),
        ("proportion", gino.proportion, PREREGISTERED_IQM_PROPORTION, numbers.Real),
    )
    for name, value, expected, kind in binding:
        if isinstance(value, bool) or not isinstance(value, kind) or value != expected:
            raise DecisionInputError(
                f"gino.{name} = {value!r}, but the preregistered decision statistic uses "
                f"{name} = {expected!r} (spec/PREREGISTRATION_PHASE1_v1.md sections 5-6)")
    for name, value in (("n_seeds", gino.n_seeds), ("n_samples", gino.n_samples)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise DecisionInputError(f"gino.{name} must be an int >= 1, got {value!r}")
    bounds = []
    for name in ("low", "point", "high"):
        value = getattr(gino, name)
        if not _is_real(value) or not math.isfinite(float(value)):
            raise DecisionInputError(f"gino.{name} must be a finite real number, got {value!r}")
        bounds.append(float(value))
    low, point, high = bounds
    if not low <= point <= high:
        raise DecisionInputError(
            f"gino must satisfy low <= point <= high, got low={low!r}, point={point!r}, high={high!r}")
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
    """Both preregistered rules for one test set and lattice (spec section 5).

    ``gino`` must be the preregistered statistic (confidence
    ``PREREGISTERED_CONFIDENCE``, ``PREREGISTERED_BOOTSTRAP_RESAMPLES`` resamples,
    trimming ``PREREGISTERED_IQM_PROPORTION``) with a finite ordered interval;
    anything else raises ``DecisionInputError`` before a rule is evaluated.
    """
    if not isinstance(test_set, str) or not test_set:
        raise TypeError("test_set must be a non-empty string")
    n = _int_at_least(n, name="n", minimum=1)
    if not isinstance(gino, BootstrapSummary):
        raise TypeError("gino must be a BootstrapSummary")
    _preregistered_summary(gino)
    upper = _positive_float(gino.high, name="gino.high")
    latency = _positive_float(gino_latency_s, name="gino_latency_s")
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
    return DecisionSummary(test_set=test_set, n=n, verdict=verdict, rules=(runtime, accuracy), gino=gino)


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
    fixed = (_non_negative_float(training_time_s, name="training_time_s")
             + _non_negative_float(data_generation_time_s, name="data_generation_time_s"))
    gino = _positive_float(gino_latency_s, name="gino_latency_s")
    baseline = _positive_float(baseline_latency_s, name="baseline_latency_s")
    if baseline <= gino:
        return None
    return int(math.ceil(fixed / (baseline - gino)))


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
