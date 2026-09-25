
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
    # Check the exact/extended-precision Real before binary64 conversion.
    # Otherwise Fraction(-1, 10**1000) becomes -0.0 and crosses the sign gate.
    if value < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    # Exact Reals beyond binary64 range (Fraction(10**400), 10**400) raise
    # OverflowError here instead of returning inf; keep the ValueError contract.
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and >= 0") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite and >= 0")
    if result == 0.0 and value != 0:
        raise ValueError(f"{name} is nonzero but underflows binary64")
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
        # Same binary64 overflow class as _finite_nonnegative (Fraction(10**400)).
        try:
            result = float(value)
        except OverflowError as exc:
            raise ValueError(
                f"{path} contains a non-finite numeric value"
            ) from exc
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


@dataclass(frozen=True)
class _DegradationDetails:
    factor: float
    saturated: bool
    underflowed: bool
    floored: bool


def _compute_degradation_details(
    shifted_error: Any,
    baseline_error: Any,
    *,
    floor: Any,
) -> _DegradationDetails:
    """Return the floor-bounded ratio plus floor/overflow/underflow flags.

    D = max(E_s, eps) / max(E_0, eps)   (spec/STRESS_SHIFT_CONTRACT_v1.md §4)

    Flooring BOTH errors keeps the direction of the comparison: the true
    quotient is >= 1 whenever E_s >= E_0 and <= 1 whenever E_s <= E_0, and
    IEEE-754 division is correctly rounded, so the stored factor never crosses
    1 against the direction of the raw errors. Flooring only the baseline
    (E_s / max(E_0, eps)) inverted it: D(1e-13, 0.0) was 0.1 although the
    shifted error is larger. ``floored`` marks every case in which at least
    one error lies below the floor, i.e. the factor is not the exact ratio
    E_s / E_0 and must not be read as a magnitude.
    """
    shifted = _finite_nonnegative(shifted_error, name="shifted_error")
    baseline = _finite_nonnegative(baseline_error, name="baseline_error")
    eps = _positive_finite(floor, name="floor")

    floored = shifted < eps or baseline < eps
    ratio = max(shifted, eps) / max(baseline, eps)
    if math.isfinite(ratio) and ratio > 0.0:
        return _DegradationDetails(ratio, False, False, floored)
    if ratio == 0.0:
        # The numerator is >= eps > 0, so the mathematical ratio is positive;
        # binary64 underflow must not be reported as an exact zero.
        return _DegradationDetails(
            math.nextafter(0.0, math.inf), False, True, floored
        )

    # Finite inputs can still overflow their ratio, e.g. baseline=0 and a
    # subnormal floor. Preserve a finite sortable value and expose saturation
    # explicitly rather than silently returning +inf.
    return _DegradationDetails(sys.float_info.max, True, False, floored)


def compute_degradation_factor(
    shifted_error: Any,
    baseline_error: Any,
    *,
    floor: Any = 1e-12,
) -> float:
    """Return the finite floor-bounded ratio max(E_s, eps) / max(E_0, eps).

    Direction-preserving: E_s >= E_0 implies a factor >= 1 and E_s <= E_0 a
    factor <= 1; both errors below the floor give exactly 1. Ratios that exceed
    IEEE-754 binary64 range saturate at sys.float_info.max. Positive ratios
    that underflow to zero clamp to the smallest positive representable float.
    StressCaseResult records floor use, saturation and underflow explicitly
    when reports are built through evaluate_stress_cases.
    """
    return _compute_degradation_details(
        shifted_error,
        baseline_error,
        floor=floor,
    ).factor


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
    # Appended after metadata so positional construction stays compatible.
    degradation_floored: bool = False

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
        if not isinstance(self.degradation_floored, bool):
            raise TypeError("degradation_floored must be bool")
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
            details = _compute_degradation_details(
                case.error_value,
                baseline,
                floor=floor,
            )
            if case.degradation_saturated != details.saturated:
                raise ValueError(
                    f"case {case.case_id!r} has inconsistent "
                    "degradation_saturated"
                )
            if case.degradation_underflowed != details.underflowed:
                raise ValueError(
                    f"case {case.case_id!r} has inconsistent "
                    "degradation_underflowed"
                )
            if case.degradation_floored != details.floored:
                raise ValueError(
                    f"case {case.case_id!r} has inconsistent "
                    "degradation_floored"
                )
            if case.degradation_factor != details.factor and not math.isclose(
                case.degradation_factor,
                details.factor,
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
                    "degradation_floored": case.degradation_floored,
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
        details = _compute_degradation_details(
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
                degradation_factor=details.factor,
                degradation_saturated=details.saturated,
                degradation_underflowed=details.underflowed,
                metadata=case.metadata,
                degradation_floored=details.floored,
            )
        )

    return StressReport(
        error_metric_name=error_metric_name,
        baseline_case_id=baseline_case_id,
        baseline_error=baseline,
        cases=tuple(results),
        degradation_floor=floor,
    )
