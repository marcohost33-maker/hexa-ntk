
"""
stress_hex_v3.py
================

Model-agnostic stress/shift contract for HexaNTK operator evaluation.

Materially different generalization claims remain separate. Resolution,
boundary-condition, geometry and rollout robustness MUST NOT be collapsed into
one OOD score. Contract source: spec/STRESS_SHIFT_CONTRACT_v1.md.
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
import numbers
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


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


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _positive_finite(value: Any, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result == 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _freeze_metadata_value(value: Any, *, path: str) -> Any:
    """Deep-freeze JSON-like metadata and normalize real-number scalars."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{path} contains a non-finite numeric value")
        return result
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_metadata_value(
                item,
                path=f"{path}.{key}",
            )
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_metadata_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain only JSON-like immutable values, mappings, "
        "or sequences"
    )


def _freeze_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    frozen = _freeze_metadata_value(metadata, path="metadata")
    if not isinstance(frozen, Mapping):
        raise TypeError("metadata must be a mapping")
    return frozen


def _thaw_metadata_value(value: Any) -> Any:
    """Return a fresh JSON-serializable copy of frozen evidence metadata."""
    if isinstance(value, Mapping):
        return {
            key: _thaw_metadata_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_metadata_value(item) for item in value]
    return value


def _compute_degradation_details(
    shifted_error: Any,
    baseline_error: Any,
    *,
    floor: Any,
) -> tuple[float, bool, bool]:
    """Return finite ratio plus overflow/underflow clamp flags."""
    shifted = _finite_nonnegative(shifted_error, name="shifted_error")
    baseline = _finite_nonnegative(baseline_error, name="baseline_error")
    eps = _positive_finite(floor, name="floor")

    if shifted == 0.0 and baseline == 0.0:
        return 1.0, False, False
    if shifted == 0.0:
        return 0.0, False, False

    denominator = max(baseline, eps)
    ratio = shifted / denominator
    if math.isfinite(ratio) and ratio > 0.0:
        return ratio, False, False
    if ratio == 0.0:
        # Preserve the semantic distinction between an exact zero shifted error
        # and a positive mathematical ratio that underflows binary64.
        return math.nextafter(0.0, math.inf), False, True

    # Finite inputs can still overflow their ratio, e.g. baseline=0 and a
    # subnormal floor. Preserve a finite sortable value and expose saturation
    # explicitly rather than silently returning +inf.
    return sys.float_info.max, True, False


def compute_degradation_factor(
    shifted_error: Any,
    baseline_error: Any,
    *,
    floor: Any = 1e-12,
) -> float:
    """Return a finite lower-is-better error ratio.

    Ratios that exceed IEEE-754 binary64 range saturate at sys.float_info.max.
    StressCaseResult.degradation_saturated records that condition when reports
    are built through evaluate_stress_cases.
    """
    factor, _, _ = _compute_degradation_details(
        shifted_error,
        baseline_error,
        floor=floor,
    )
    return factor


@dataclass(frozen=True)
class StressCase:
    """One pre-declared stress case with a stable identifier."""

    case_id: str
    axis: ShiftAxis
    regime: ShiftRegime
    error_value: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(self.axis, ShiftAxis):
            raise TypeError("axis must be a ShiftAxis")
        if not isinstance(self.regime, ShiftRegime):
            raise TypeError("regime must be a ShiftRegime")
        object.__setattr__(
            self,
            "error_value",
            _finite_nonnegative(self.error_value, name="error_value"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )


@dataclass(frozen=True)
class StressCaseResult:
    """Immutable evidence for one evaluated shift case."""

    case_id: str
    axis: ShiftAxis
    regime: ShiftRegime
    error_value: float
    degradation_factor: float
    degradation_saturated: bool = False
    degradation_underflowed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(self.axis, ShiftAxis):
            raise TypeError("axis must be a ShiftAxis")
        if not isinstance(self.regime, ShiftRegime):
            raise TypeError("regime must be a ShiftRegime")
        object.__setattr__(
            self,
            "error_value",
            _finite_nonnegative(self.error_value, name="error_value"),
        )
        object.__setattr__(
            self,
            "degradation_factor",
            _finite_nonnegative(
                self.degradation_factor,
                name="degradation_factor",
            ),
        )
        if not isinstance(self.degradation_saturated, bool):
            raise TypeError("degradation_saturated must be bool")
        if not isinstance(self.degradation_underflowed, bool):
            raise TypeError("degradation_underflowed must be bool")
        if self.degradation_saturated and self.degradation_underflowed:
            raise ValueError(
                "degradation_saturated and degradation_underflowed "
                "cannot both be true"
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )


@dataclass(frozen=True)
class StressReport:
    """Axis-preserving stress evidence; deliberately no overall score."""

    error_metric_name: str
    baseline_case_id: str
    baseline_error: float
    cases: tuple[StressCaseResult, ...]
    degradation_floor: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.error_metric_name, str)
            or not self.error_metric_name.strip()
        ):
            raise ValueError("error_metric_name must be a non-empty string")
        if (
            not isinstance(self.baseline_case_id, str)
            or not self.baseline_case_id.strip()
        ):
            raise ValueError("baseline_case_id must be a non-empty string")

        baseline = _finite_nonnegative(
            self.baseline_error,
            name="baseline_error",
        )
        floor = _positive_finite(
            self.degradation_floor,
            name="degradation_floor",
        )
        object.__setattr__(self, "baseline_error", baseline)
        object.__setattr__(self, "degradation_floor", floor)

        if not isinstance(self.cases, tuple):
            raise TypeError("cases must be a tuple of StressCaseResult values")
        if not all(isinstance(case, StressCaseResult) for case in self.cases):
            raise TypeError("cases must contain only StressCaseResult values")

        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("StressReport contains duplicate case_id values")
        if self.baseline_case_id in case_ids:
            raise ValueError(
                "baseline_case_id must not be repeated in shifted cases"
            )

        for case in self.cases:
            expected, saturated, underflowed = _compute_degradation_details(
                case.error_value,
                baseline,
                floor=floor,
            )
            if case.degradation_saturated != saturated:
                raise ValueError(
                    f"case {case.case_id!r} has inconsistent "
                    "degradation_saturated"
                )
            if case.degradation_underflowed != underflowed:
                raise ValueError(
                    f"case {case.case_id!r} has inconsistent "
                    "degradation_underflowed"
                )
            if case.degradation_factor != expected and not math.isclose(
                case.degradation_factor,
                expected,
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                raise ValueError(
                    f"case {case.case_id!r} has inconsistent "
                    "degradation_factor"
                )

    def by_axis(self) -> dict[str, list[StressCaseResult]]:
        grouped: dict[str, list[StressCaseResult]] = {}
        for case in self.cases:
            grouped.setdefault(case.axis.value, []).append(case)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_metric_name": self.error_metric_name,
            "metric_semantics": "nonnegative_error_lower_is_better",
            "baseline_case_id": self.baseline_case_id,
            "baseline_error": self.baseline_error,
            "degradation_floor": self.degradation_floor,
            "cases": [
                {
                    "case_id": case.case_id,
                    "axis": case.axis.value,
                    "regime": case.regime.value,
                    "error_value": case.error_value,
                    "degradation_factor": case.degradation_factor,
                    "degradation_saturated": case.degradation_saturated,
                    "degradation_underflowed": case.degradation_underflowed,
                    "metadata": _thaw_metadata_value(case.metadata),
                }
                for case in self.cases
            ],
        }


def evaluate_stress_cases(
    *,
    error_metric_name: str,
    baseline_case_id: str,
    baseline_error: Any,
    cases: Any,
    degradation_floor: Any = 1e-12,
) -> StressReport:
    """Build an axis-preserving report from non-negative lower-is-better errors."""
    if not isinstance(error_metric_name, str) or not error_metric_name.strip():
        raise ValueError("error_metric_name must be a non-empty string")
    if not isinstance(baseline_case_id, str) or not baseline_case_id.strip():
        raise ValueError("baseline_case_id must be a non-empty string")

    baseline = _finite_nonnegative(baseline_error, name="baseline_error")
    floor = _positive_finite(degradation_floor, name="degradation_floor")
    try:
        case_list = list(cases)
    except TypeError as exc:
        raise TypeError("cases must be an iterable of StressCase values") from exc

    if not all(isinstance(case, StressCase) for case in case_list):
        raise TypeError("cases must contain only StressCase values")

    ids = [case.case_id for case in case_list]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate stress case_id")
    if baseline_case_id in ids:
        raise ValueError("baseline_case_id must not be reused by a stress case")

    results = []
    for case in case_list:
        factor, saturated, underflowed = _compute_degradation_details(
            case.error_value,
            baseline,
            floor=floor,
        )
        results.append(
            StressCaseResult(
                case_id=case.case_id,
                axis=case.axis,
                regime=case.regime,
                error_value=case.error_value,
                degradation_factor=factor,
                degradation_saturated=saturated,
                degradation_underflowed=underflowed,
                metadata=case.metadata,
            )
        )

    return StressReport(
        error_metric_name=error_metric_name,
        baseline_case_id=baseline_case_id,
        baseline_error=baseline,
        cases=tuple(results),
        degradation_floor=floor,
    )
