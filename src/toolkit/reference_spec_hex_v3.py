"""
reference_spec_hex_v3.py
========================

Fail-closed credibility contract for references used in HexaNTK/Q-Forge
evaluation. Contract source: spec/REFERENCE_SPEC_v1.md.

A numerical reference is never treated as generic ground truth. Credibility is
derived from verification, convergence, uncertainty, reference kind, and
context of use.
"""

from __future__ import annotations

__all__ = [
    "ReferenceKind",
    "EvidenceStatus",
    "ConvergenceAxis",
    "UncertaintyKind",
    "ClaimCeiling",
    "EvidenceCheck",
    "RefinementLevel",
    "ConvergenceStudy",
    "UncertaintyStatement",
    "ReferenceSpec",
]

import math
import numbers
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReferenceKind(str, Enum):
    ANALYTIC = "analytic"
    MANUFACTURED = "manufactured"
    NUMERICAL = "numerical"
    EXPERIMENTAL = "experimental"


class EvidenceStatus(str, Enum):
    NOT_ASSESSED = "not_assessed"
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class ConvergenceAxis(str, Enum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"


class UncertaintyKind(str, Enum):
    ABSOLUTE = "absolute"
    RELATIVE_FRACTION = "relative_fraction"


class ClaimCeiling(str, Enum):
    MATHEMATICAL_ORACLE = "mathematical_oracle"
    VERIFIED_NUMERICAL_REFERENCE = "verified_numerical_reference"
    VALIDATION_DATA_WITH_UNCERTAINTY = "validation_data_with_uncertainty"
    COMPARISON_ONLY = "comparison_only"


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: Any, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


@dataclass(frozen=True)
class EvidenceCheck:
    status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    evidence_ref: str | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, EvidenceStatus):
            raise TypeError("status must be an EvidenceStatus")

        evidence_ref = (
            None
            if self.evidence_ref is None
            else _nonempty_text(self.evidence_ref, name="evidence_ref")
        )
        rationale = (
            None
            if self.rationale is None
            else _nonempty_text(self.rationale, name="rationale")
        )

        if self.status is EvidenceStatus.PASS and evidence_ref is None:
            raise ValueError("PASS evidence requires evidence_ref")
        if self.status is EvidenceStatus.NOT_APPLICABLE and rationale is None:
            raise ValueError("NOT_APPLICABLE evidence requires rationale")
        if self.status is EvidenceStatus.FAIL and rationale is None:
            raise ValueError("FAIL evidence requires rationale")

        object.__setattr__(self, "evidence_ref", evidence_ref)
        object.__setattr__(self, "rationale", rationale)


@dataclass(frozen=True)
class RefinementLevel:
    level_id: str
    characteristic_size: float
    quantity_value: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "level_id",
            _nonempty_text(self.level_id, name="level_id"),
        )
        size = _finite_real(
            self.characteristic_size,
            name="characteristic_size",
        )
        if size <= 0.0:
            raise ValueError("characteristic_size must be > 0")
        object.__setattr__(self, "characteristic_size", size)
        object.__setattr__(
            self,
            "quantity_value",
            _finite_real(self.quantity_value, name="quantity_value"),
        )


@dataclass(frozen=True)
class ConvergenceStudy:
    axis: ConvergenceAxis
    quantity_of_interest: str
    levels: tuple[RefinementLevel, ...]
    evidence_ref: str
    observed_order: float | None = None
    fine_grid_uncertainty: float | None = None
    asymptotic_range_checked: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.axis, ConvergenceAxis):
            raise TypeError("axis must be a ConvergenceAxis")
        object.__setattr__(
            self,
            "quantity_of_interest",
            _nonempty_text(
                self.quantity_of_interest,
                name="quantity_of_interest",
            ),
        )
        object.__setattr__(
            self,
            "evidence_ref",
            _nonempty_text(self.evidence_ref, name="evidence_ref"),
        )
        if not isinstance(self.levels, tuple):
            raise TypeError("levels must be a tuple")
        if len(self.levels) < 2:
            raise ValueError("a convergence study requires at least two levels")
        if not all(isinstance(level, RefinementLevel) for level in self.levels):
            raise TypeError("levels must contain only RefinementLevel values")

        sizes = [level.characteristic_size for level in self.levels]
        if any(b <= a for a, b in zip(sizes, sizes[1:])):
            raise ValueError(
                "levels must be ordered finest-to-coarsest with strictly "
                "increasing characteristic_size"
            )

        if self.observed_order is not None:
            order = _finite_real(self.observed_order, name="observed_order")
            if order <= 0.0:
                raise ValueError("observed_order must be > 0")
            if len(self.levels) < 3:
                raise ValueError(
                    "observed_order requires at least three refinement levels"
                )
            object.__setattr__(self, "observed_order", order)

        if self.fine_grid_uncertainty is not None:
            object.__setattr__(
                self,
                "fine_grid_uncertainty",
                _finite_nonnegative(
                    self.fine_grid_uncertainty,
                    name="fine_grid_uncertainty",
                ),
            )

        if not isinstance(self.asymptotic_range_checked, bool):
            raise TypeError("asymptotic_range_checked must be bool")
        if self.asymptotic_range_checked and len(self.levels) < 3:
            raise ValueError(
                "asymptotic_range_checked requires at least three levels"
            )

    @property
    def level_count(self) -> int:
        return len(self.levels)


@dataclass(frozen=True)
class UncertaintyStatement:
    kind: UncertaintyKind
    value: float
    evidence_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, UncertaintyKind):
            raise TypeError("kind must be an UncertaintyKind")
        object.__setattr__(
            self,
            "value",
            _finite_nonnegative(self.value, name="uncertainty value"),
        )
        object.__setattr__(
            self,
            "evidence_ref",
            _nonempty_text(self.evidence_ref, name="evidence_ref"),
        )


def _default_check() -> EvidenceCheck:
    return EvidenceCheck()


@dataclass(frozen=True)
class ReferenceSpec:
    reference_id: str
    kind: ReferenceKind
    context_of_use: str
    quantity_of_interest: str

    code_verification: EvidenceCheck = field(default_factory=_default_check)
    spatial_verification: EvidenceCheck = field(default_factory=_default_check)
    temporal_verification: EvidenceCheck = field(default_factory=_default_check)
    iterative_verification: EvidenceCheck = field(default_factory=_default_check)
    oracle_verification: EvidenceCheck = field(default_factory=_default_check)

    spatial_study: ConvergenceStudy | None = None
    temporal_study: ConvergenceStudy | None = None
    numerical_uncertainty: UncertaintyStatement | None = None
    measurement_uncertainty: UncertaintyStatement | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_id",
            _nonempty_text(self.reference_id, name="reference_id"),
        )
        if not isinstance(self.kind, ReferenceKind):
            raise TypeError("kind must be a ReferenceKind")
        object.__setattr__(
            self,
            "context_of_use",
            _nonempty_text(self.context_of_use, name="context_of_use"),
        )
        object.__setattr__(
            self,
            "quantity_of_interest",
            _nonempty_text(
                self.quantity_of_interest,
                name="quantity_of_interest",
            ),
        )

        checks = (
            self.code_verification,
            self.spatial_verification,
            self.temporal_verification,
            self.iterative_verification,
            self.oracle_verification,
        )
        if not all(isinstance(check, EvidenceCheck) for check in checks):
            raise TypeError("verification fields must be EvidenceCheck values")

        if self.spatial_study is not None:
            if not isinstance(self.spatial_study, ConvergenceStudy):
                raise TypeError("spatial_study must be ConvergenceStudy")
            if self.spatial_study.axis is not ConvergenceAxis.SPATIAL:
                raise ValueError("spatial_study must use the spatial axis")
            if (
                self.spatial_study.quantity_of_interest
                != self.quantity_of_interest
            ):
                raise ValueError(
                    "spatial_study quantity_of_interest must match "
                    "ReferenceSpec"
                )

        if self.temporal_study is not None:
            if not isinstance(self.temporal_study, ConvergenceStudy):
                raise TypeError("temporal_study must be ConvergenceStudy")
            if self.temporal_study.axis is not ConvergenceAxis.TEMPORAL:
                raise ValueError("temporal_study must use the temporal axis")
            if (
                self.temporal_study.quantity_of_interest
                != self.quantity_of_interest
            ):
                raise ValueError(
                    "temporal_study quantity_of_interest must match "
                    "ReferenceSpec"
                )

        for name, uncertainty in (
            ("numerical_uncertainty", self.numerical_uncertainty),
            ("measurement_uncertainty", self.measurement_uncertainty),
        ):
            if uncertainty is not None and not isinstance(
                uncertainty,
                UncertaintyStatement,
            ):
                raise TypeError(f"{name} must be an UncertaintyStatement")

    @staticmethod
    def _solution_axis_ok(
        check: EvidenceCheck,
        study: ConvergenceStudy | None,
    ) -> bool:
        if check.status is EvidenceStatus.NOT_APPLICABLE:
            return study is None
        if check.status is not EvidenceStatus.PASS:
            return False
        return study is not None and study.level_count >= 3

    def claim_ceiling(self) -> ClaimCeiling:
        if self.kind in {
            ReferenceKind.ANALYTIC,
            ReferenceKind.MANUFACTURED,
        }:
            if self.oracle_verification.status is EvidenceStatus.PASS:
                return ClaimCeiling.MATHEMATICAL_ORACLE
            return ClaimCeiling.COMPARISON_ONLY

        if self.kind is ReferenceKind.EXPERIMENTAL:
            if self.measurement_uncertainty is not None:
                return ClaimCeiling.VALIDATION_DATA_WITH_UNCERTAINTY
            return ClaimCeiling.COMPARISON_ONLY

        if self.kind is ReferenceKind.NUMERICAL:
            if self.code_verification.status is not EvidenceStatus.PASS:
                return ClaimCeiling.COMPARISON_ONLY
            if not self._solution_axis_ok(
                self.spatial_verification,
                self.spatial_study,
            ):
                return ClaimCeiling.COMPARISON_ONLY
            if not self._solution_axis_ok(
                self.temporal_verification,
                self.temporal_study,
            ):
                return ClaimCeiling.COMPARISON_ONLY
            if self.iterative_verification.status not in {
                EvidenceStatus.PASS,
                EvidenceStatus.NOT_APPLICABLE,
            }:
                return ClaimCeiling.COMPARISON_ONLY
            if not any(
                check.status is EvidenceStatus.PASS
                for check in (
                    self.spatial_verification,
                    self.temporal_verification,
                    self.iterative_verification,
                )
            ):
                return ClaimCeiling.COMPARISON_ONLY
            if self.numerical_uncertainty is None:
                return ClaimCeiling.COMPARISON_ONLY
            return ClaimCeiling.VERIFIED_NUMERICAL_REFERENCE

        return ClaimCeiling.COMPARISON_ONLY

    def to_dict(self) -> dict[str, Any]:
        def check_dict(check: EvidenceCheck) -> dict[str, Any]:
            return {
                "status": check.status.value,
                "evidence_ref": check.evidence_ref,
                "rationale": check.rationale,
            }

        def study_dict(
            study: ConvergenceStudy | None,
        ) -> dict[str, Any] | None:
            if study is None:
                return None
            return {
                "axis": study.axis.value,
                "quantity_of_interest": study.quantity_of_interest,
                "levels": [
                    {
                        "level_id": level.level_id,
                        "characteristic_size": level.characteristic_size,
                        "quantity_value": level.quantity_value,
                    }
                    for level in study.levels
                ],
                "evidence_ref": study.evidence_ref,
                "observed_order": study.observed_order,
                "fine_grid_uncertainty": study.fine_grid_uncertainty,
                "asymptotic_range_checked": study.asymptotic_range_checked,
            }

        def uncertainty_dict(
            uncertainty: UncertaintyStatement | None,
        ) -> dict[str, Any] | None:
            if uncertainty is None:
                return None
            return {
                "kind": uncertainty.kind.value,
                "value": uncertainty.value,
                "evidence_ref": uncertainty.evidence_ref,
            }

        return {
            "reference_id": self.reference_id,
            "kind": self.kind.value,
            "context_of_use": self.context_of_use,
            "quantity_of_interest": self.quantity_of_interest,
            "claim_ceiling": self.claim_ceiling().value,
            "code_verification": check_dict(self.code_verification),
            "spatial_verification": check_dict(self.spatial_verification),
            "temporal_verification": check_dict(self.temporal_verification),
            "iterative_verification": check_dict(
                self.iterative_verification
            ),
            "oracle_verification": check_dict(self.oracle_verification),
            "spatial_study": study_dict(self.spatial_study),
            "temporal_study": study_dict(self.temporal_study),
            "numerical_uncertainty": uncertainty_dict(
                self.numerical_uncertainty
            ),
            "measurement_uncertainty": uncertainty_dict(
                self.measurement_uncertainty
            ),
        }
