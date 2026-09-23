
"""
stress_hex_v3.py
================

Model-agnostic stress/shift contract for HexaNTK operator evaluation.

Materially different generalization claims remain separate. Resolution,
boundary-condition, geometry and rollout robustness MUST NOT be collapsed into
one OOD score. This module does not decide pass/fail thresholds; Q-Forge policy
or an ExperimentSpec may impose claim-specific thresholds after raw evidence.
"""

from __future__ import annotations

__all__ = [
    "ShiftAxis",
    "ShiftRegime",
    "StressCase",
    "StressCaseResult",
    "StressReport",
    "compute_degradation_factor",
    "evaluate_stress_cases",
]

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


class ShiftAxis(str, Enum):
    """Independent axes along which a learned PDE operator can be stressed."""

    PARAMETER = "parameter"
    INITIAL_CONDITION = "initial_condition"
    FORCING = "forcing"
    BOUNDARY_CONDITION = "boundary_condition"
    GEOMETRY = "geometry"
    RESOLUTION = "resolution"
    TIMESTEP = "timestep"
    ROLLOUT_HORIZON = "rollout_horizon"
    NOISE = "noise"
    TOPOLOGY = "topology"


class ShiftRegime(str, Enum):
    """Relationship of a case to the training/evaluation distribution."""

    ID = "id"
    OOD = "ood"
    EXTRAPOLATION = "extrapolation"


def _finite_nonnegative(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _positive_finite(value: float, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result == 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def compute_degradation_factor(
    shifted_error: float,
    baseline_error: float,
    *,
    floor: float = 1e-12,
) -> float:
    """Return an error ratio relative to the in-distribution baseline.

    The denominator is floored only to make an exact or near-zero baseline
    numerically representable. No clipping of the numerator is performed.
    If both errors are exactly zero, the factor is defined as 1.0.
    """

    shifted = _finite_nonnegative(shifted_error, name="shifted_error")
    baseline = _finite_nonnegative(baseline_error, name="baseline_error")
    eps = _positive_finite(floor, name="floor")

    if shifted == 0.0 and baseline == 0.0:
        return 1.0
    return shifted / max(baseline, eps)


@dataclass(frozen=True)
class StressCase:
    """One pre-declared stress case with a stable identifier."""

    case_id: str
    axis: ShiftAxis
    regime: ShiftRegime
    metric_value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(self.axis, ShiftAxis):
            raise TypeError("axis must be a ShiftAxis")
        if not isinstance(self.regime, ShiftRegime):
            raise TypeError("regime must be a ShiftRegime")
        object.__setattr__(
            self,
            "metric_value",
            _finite_nonnegative(self.metric_value, name="metric_value"),
        )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")


@dataclass(frozen=True)
class StressCaseResult:
    case_id: str
    axis: ShiftAxis
    regime: ShiftRegime
    metric_value: float
    degradation_factor: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StressReport:
    """Axis-preserving stress evidence; deliberately no overall score."""

    metric_name: str
    baseline_case_id: str
    baseline_value: float
    cases: tuple[StressCaseResult, ...]
    degradation_floor: float

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise ValueError("metric_name must be a non-empty string")
        if (
            not isinstance(self.baseline_case_id, str)
            or not self.baseline_case_id.strip()
        ):
            raise ValueError("baseline_case_id must be a non-empty string")
        _finite_nonnegative(self.baseline_value, name="baseline_value")
        _positive_finite(self.degradation_floor, name="degradation_floor")

        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("StressReport contains duplicate case_id values")
        if self.baseline_case_id in case_ids:
            raise ValueError("baseline_case_id must not be repeated in shifted cases")

    def by_axis(self) -> dict[str, list[StressCaseResult]]:
        grouped: dict[str, list[StressCaseResult]] = {}
        for case in self.cases:
            grouped.setdefault(case.axis.value, []).append(case)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for case in payload["cases"]:
            case["axis"] = case["axis"].value
            case["regime"] = case["regime"].value
        return payload


def evaluate_stress_cases(
    *,
    metric_name: str,
    baseline_case_id: str,
    baseline_value: float,
    cases: Iterable[StressCase],
    degradation_floor: float = 1e-12,
) -> StressReport:
    """Build an axis-preserving report from precomputed error-like metrics."""

    if not isinstance(metric_name, str) or not metric_name.strip():
        raise ValueError("metric_name must be a non-empty string")
    if not isinstance(baseline_case_id, str) or not baseline_case_id.strip():
        raise ValueError("baseline_case_id must be a non-empty string")

    baseline = _finite_nonnegative(baseline_value, name="baseline_value")
    floor = _positive_finite(degradation_floor, name="degradation_floor")
    case_list = list(cases)

    if not all(isinstance(case, StressCase) for case in case_list):
        raise TypeError("cases must contain only StressCase values")

    ids = [case.case_id for case in case_list]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate stress case_id")
    if baseline_case_id in ids:
        raise ValueError("baseline_case_id must not be reused by a stress case")

    results = tuple(
        StressCaseResult(
            case_id=case.case_id,
            axis=case.axis,
            regime=case.regime,
            metric_value=case.metric_value,
            degradation_factor=compute_degradation_factor(
                case.metric_value,
                baseline,
                floor=floor,
            ),
            metadata=dict(case.metadata),
        )
        for case in case_list
    )

    return StressReport(
        metric_name=metric_name,
        baseline_case_id=baseline_case_id,
        baseline_value=baseline,
        cases=results,
        degradation_floor=floor,
    )
