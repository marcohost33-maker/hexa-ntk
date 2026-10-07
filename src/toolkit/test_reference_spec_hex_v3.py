"""
test_reference_spec_hex_v3.py
=============================
Contract tests for ReferenceSpec v1 credibility and claim ceilings.
"""

from __future__ import annotations

import unittest

from reference_spec_hex_v3 import (
    ClaimCeiling,
    ConvergenceAxis,
    ConvergenceStudy,
    EvidenceCheck,
    EvidenceStatus,
    ReferenceKind,
    ReferenceSpec,
    RefinementLevel,
    UncertaintyKind,
    UncertaintyStatement,
)


def pass_check(ref: str) -> EvidenceCheck:
    return EvidenceCheck(EvidenceStatus.PASS, evidence_ref=ref)


def na_check(reason: str) -> EvidenceCheck:
    return EvidenceCheck(
        EvidenceStatus.NOT_APPLICABLE,
        rationale=reason,
    )


def three_level_study(axis: ConvergenceAxis) -> ConvergenceStudy:
    return ConvergenceStudy(
        axis=axis,
        quantity_of_interest="drag_coefficient",
        levels=(
            RefinementLevel("fine", 0.25, 0.9710),
            RefinementLevel("medium", 0.5, 0.9670),
            RefinementLevel("coarse", 1.0, 0.9510),
        ),
        evidence_ref=f"results/{axis.value}_convergence.json",
        observed_order=2.0,
        fine_grid_relative_error_estimate=0.0012,
        asymptotic_range_checked=True,
    )


class TestEvidenceCheck(unittest.TestCase):
    def test_pass_requires_evidence_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "evidence_ref"):
            EvidenceCheck(EvidenceStatus.PASS)

    def test_not_applicable_requires_rationale(self) -> None:
        with self.assertRaisesRegex(ValueError, "rationale"):
            EvidenceCheck(EvidenceStatus.NOT_APPLICABLE)

    def test_fail_requires_rationale(self) -> None:
        with self.assertRaisesRegex(ValueError, "rationale"):
            EvidenceCheck(EvidenceStatus.FAIL)


class TestConvergenceStudy(unittest.TestCase):
    def test_observed_order_requires_three_levels(self) -> None:
        with self.assertRaisesRegex(ValueError, "three"):
            ConvergenceStudy(
                axis=ConvergenceAxis.SPATIAL,
                quantity_of_interest="drag_coefficient",
                levels=(
                    RefinementLevel("fine", 0.5, 1.0),
                    RefinementLevel("coarse", 1.0, 1.1),
                ),
                evidence_ref="results/two_grid.json",
                observed_order=2.0,
            )

    def test_observed_order_is_recomputed_from_levels(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            ConvergenceStudy(
                axis=ConvergenceAxis.SPATIAL,
                quantity_of_interest="drag_coefficient",
                levels=(
                    RefinementLevel("fine", 0.25, 0.9710),
                    RefinementLevel("medium", 0.5, 0.9670),
                    RefinementLevel("coarse", 1.0, 0.9510),
                ),
                evidence_ref="results/false_order.json",
                observed_order=1.0,
            )

    def test_observed_order_rejects_oscillatory_or_unequal_ratio_levels(self) -> None:
        with self.assertRaisesRegex(ValueError, "monotone"):
            ConvergenceStudy(
                axis=ConvergenceAxis.SPATIAL,
                quantity_of_interest="drag_coefficient",
                levels=(
                    RefinementLevel("fine", 0.25, 1.00),
                    RefinementLevel("medium", 0.5, 1.10),
                    RefinementLevel("coarse", 1.0, 1.05),
                ),
                evidence_ref="results/oscillatory.json",
                observed_order=2.0,
            )
        with self.assertRaisesRegex(ValueError, "equal refinement ratios"):
            ConvergenceStudy(
                axis=ConvergenceAxis.SPATIAL,
                quantity_of_interest="drag_coefficient",
                levels=(
                    RefinementLevel("fine", 0.2, 1.00),
                    RefinementLevel("medium", 0.5, 1.04),
                    RefinementLevel("coarse", 1.0, 1.20),
                ),
                evidence_ref="results/unequal_ratio.json",
                observed_order=2.0,
            )

    def test_asymptotic_claim_requires_three_levels(self) -> None:
        with self.assertRaisesRegex(ValueError, "three"):
            ConvergenceStudy(
                axis=ConvergenceAxis.SPATIAL,
                quantity_of_interest="drag_coefficient",
                levels=(
                    RefinementLevel("fine", 0.5, 1.0),
                    RefinementLevel("coarse", 1.0, 1.1),
                ),
                evidence_ref="results/two_grid.json",
                asymptotic_range_checked=True,
            )

    def test_level_ids_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            ConvergenceStudy(
                axis=ConvergenceAxis.SPATIAL,
                quantity_of_interest="drag_coefficient",
                levels=(
                    RefinementLevel("same", 0.5, 1.0),
                    RefinementLevel("same", 1.0, 1.1),
                ),
                evidence_ref="results/duplicate_ids.json",
            )

    def test_levels_must_be_finest_to_coarsest(self) -> None:
        with self.assertRaisesRegex(ValueError, "finest-to-coarsest"):
            ConvergenceStudy(
                axis=ConvergenceAxis.SPATIAL,
                quantity_of_interest="drag_coefficient",
                levels=(
                    RefinementLevel("coarse", 1.0, 1.1),
                    RefinementLevel("fine", 0.5, 1.0),
                ),
                evidence_ref="results/reversed.json",
            )


class TestReferenceClaimCeiling(unittest.TestCase):
    def test_numerical_reference_defaults_to_comparison_only(self) -> None:
        reference = ReferenceSpec(
            reference_id="solver/unverified",
            kind=ReferenceKind.NUMERICAL,
            context_of_use="steady external-flow regression",
            quantity_of_interest="drag_coefficient",
        )
        self.assertEqual(
            reference.claim_ceiling(),
            ClaimCeiling.COMPARISON_ONLY,
        )

    def test_verified_numerical_reference_requires_full_evidence(self) -> None:
        reference = ReferenceSpec(
            reference_id="solver/verified",
            kind=ReferenceKind.NUMERICAL,
            context_of_use="steady external-flow regression",
            quantity_of_interest="drag_coefficient",
            code_verification=pass_check("results/code_verification.log"),
            spatial_verification=pass_check(
                "results/spatial_convergence.json"
            ),
            temporal_verification=na_check("steady-state reference"),
            iterative_verification=pass_check(
                "results/iterative_convergence.log"
            ),
            spatial_study=three_level_study(ConvergenceAxis.SPATIAL),
            numerical_uncertainty=UncertaintyStatement(
                UncertaintyKind.ABSOLUTE,
                0.0012,
                "results/numerical_uncertainty.json",
            ),
        )
        self.assertEqual(
            reference.claim_ceiling(),
            ClaimCeiling.VERIFIED_NUMERICAL_REFERENCE,
        )

        payload = reference.to_dict()
        self.assertEqual(
            payload["claim_ceiling"],
            "verified_numerical_reference",
        )
        self.assertEqual(
            len(payload["spatial_study"]["levels"]),
            3,
        )

    def test_all_solution_axes_not_applicable_is_not_promoted(self) -> None:
        reference = ReferenceSpec(
            reference_id="solver/all-na",
            kind=ReferenceKind.NUMERICAL,
            context_of_use="algebraic reference comparison",
            quantity_of_interest="drag_coefficient",
            code_verification=pass_check("results/code.log"),
            spatial_verification=na_check("no spatial discretization"),
            temporal_verification=na_check("steady algebraic problem"),
            iterative_verification=na_check("closed-form linear solve"),
            numerical_uncertainty=UncertaintyStatement(
                UncertaintyKind.ABSOLUTE,
                0.001,
                "results/u.json",
            ),
        )
        self.assertEqual(
            reference.claim_ceiling(),
            ClaimCeiling.COMPARISON_ONLY,
        )

    def test_three_raw_levels_without_convergence_assessment_do_not_promote(self) -> None:
        raw_study = ConvergenceStudy(
            axis=ConvergenceAxis.SPATIAL,
            quantity_of_interest="drag_coefficient",
            levels=(
                RefinementLevel("fine", 0.25, 0.971),
                RefinementLevel("medium", 0.5, 0.969),
                RefinementLevel("coarse", 1.0, 0.963),
            ),
            evidence_ref="results/raw_three_level.json",
        )
        reference = ReferenceSpec(
            reference_id="solver/raw-three-level",
            kind=ReferenceKind.NUMERICAL,
            context_of_use="steady external-flow regression",
            quantity_of_interest="drag_coefficient",
            code_verification=pass_check("results/code.log"),
            spatial_verification=pass_check("results/spatial.log"),
            temporal_verification=na_check("steady"),
            iterative_verification=na_check("direct solve"),
            spatial_study=raw_study,
            numerical_uncertainty=UncertaintyStatement(
                UncertaintyKind.ABSOLUTE,
                0.01,
                "results/u.json",
            ),
        )
        self.assertEqual(
            reference.claim_ceiling(),
            ClaimCeiling.COMPARISON_ONLY,
        )

    def test_passed_spatial_check_without_study_is_not_promoted(self) -> None:
        reference = ReferenceSpec(
            reference_id="solver/no-study",
            kind=ReferenceKind.NUMERICAL,
            context_of_use="steady external-flow regression",
            quantity_of_interest="drag_coefficient",
            code_verification=pass_check("results/code.log"),
            spatial_verification=pass_check("results/spatial.log"),
            temporal_verification=na_check("steady"),
            iterative_verification=na_check("direct linear solve"),
            numerical_uncertainty=UncertaintyStatement(
                UncertaintyKind.ABSOLUTE,
                0.01,
                "results/u.json",
            ),
        )
        self.assertEqual(
            reference.claim_ceiling(),
            ClaimCeiling.COMPARISON_ONLY,
        )

    def test_two_level_study_does_not_support_verified_ceiling(self) -> None:
        two_level = ConvergenceStudy(
            axis=ConvergenceAxis.SPATIAL,
            quantity_of_interest="drag_coefficient",
            levels=(
                RefinementLevel("fine", 0.5, 1.0),
                RefinementLevel("coarse", 1.0, 1.1),
            ),
            evidence_ref="results/two_grid.json",
        )
        reference = ReferenceSpec(
            reference_id="solver/two-level",
            kind=ReferenceKind.NUMERICAL,
            context_of_use="steady external-flow regression",
            quantity_of_interest="drag_coefficient",
            code_verification=pass_check("results/code.log"),
            spatial_verification=pass_check("results/spatial.log"),
            temporal_verification=na_check("steady"),
            iterative_verification=na_check("direct solve"),
            spatial_study=two_level,
            numerical_uncertainty=UncertaintyStatement(
                UncertaintyKind.ABSOLUTE,
                0.02,
                "results/u.json",
            ),
        )
        self.assertEqual(
            reference.claim_ceiling(),
            ClaimCeiling.COMPARISON_ONLY,
        )

    def test_failed_verification_caps_numerical_reference(self) -> None:
        reference = ReferenceSpec(
            reference_id="solver/failed",
            kind=ReferenceKind.NUMERICAL,
            context_of_use="steady external-flow regression",
            quantity_of_interest="drag_coefficient",
            code_verification=EvidenceCheck(
                EvidenceStatus.FAIL,
                rationale="manufactured-solution convergence failed",
            ),
            spatial_verification=pass_check("results/spatial.log"),
            temporal_verification=na_check("steady"),
            iterative_verification=na_check("direct solve"),
            spatial_study=three_level_study(ConvergenceAxis.SPATIAL),
            numerical_uncertainty=UncertaintyStatement(
                UncertaintyKind.ABSOLUTE,
                0.01,
                "results/u.json",
            ),
        )
        self.assertEqual(
            reference.claim_ceiling(),
            ClaimCeiling.COMPARISON_ONLY,
        )

    def test_analytic_oracle_requires_oracle_verification(self) -> None:
        weak = ReferenceSpec(
            reference_id="analytic/unverified",
            kind=ReferenceKind.ANALYTIC,
            context_of_use="linear manufactured benchmark",
            quantity_of_interest="state",
        )
        self.assertEqual(
            weak.claim_ceiling(),
            ClaimCeiling.COMPARISON_ONLY,
        )

        strong = ReferenceSpec(
            reference_id="analytic/verified",
            kind=ReferenceKind.ANALYTIC,
            context_of_use="linear manufactured benchmark",
            quantity_of_interest="state",
            oracle_verification=pass_check(
                "spec/analytic_solution_derivation.md"
            ),
        )
        self.assertEqual(
            strong.claim_ceiling(),
            ClaimCeiling.MATHEMATICAL_ORACLE,
        )

    def test_experimental_data_requires_measurement_uncertainty(self) -> None:
        weak = ReferenceSpec(
            reference_id="experiment/no-u",
            kind=ReferenceKind.EXPERIMENTAL,
            context_of_use="wind-tunnel validation point",
            quantity_of_interest="drag_coefficient",
        )
        self.assertEqual(
            weak.claim_ceiling(),
            ClaimCeiling.COMPARISON_ONLY,
        )

        strong = ReferenceSpec(
            reference_id="experiment/with-u",
            kind=ReferenceKind.EXPERIMENTAL,
            context_of_use="wind-tunnel validation point",
            quantity_of_interest="drag_coefficient",
            measurement_uncertainty=UncertaintyStatement(
                UncertaintyKind.ABSOLUTE,
                0.003,
                "results/measurement_uncertainty.json",
            ),
        )
        self.assertEqual(
            strong.claim_ceiling(),
            ClaimCeiling.VALIDATION_DATA_WITH_UNCERTAINTY,
        )

    def test_wrong_study_axis_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "spatial axis"):
            ReferenceSpec(
                reference_id="solver/wrong-axis",
                kind=ReferenceKind.NUMERICAL,
                context_of_use="transient benchmark",
                quantity_of_interest="drag_coefficient",
                spatial_study=three_level_study(
                    ConvergenceAxis.TEMPORAL
                ),
            )

    def test_study_quantity_must_match_reference(self) -> None:
        study = ConvergenceStudy(
            axis=ConvergenceAxis.SPATIAL,
            quantity_of_interest="lift_coefficient",
            levels=(
                RefinementLevel("fine", 0.25, 0.8),
                RefinementLevel("medium", 0.5, 0.79),
                RefinementLevel("coarse", 1.0, 0.76),
            ),
            evidence_ref="results/lift_grid.json",
        )
        with self.assertRaisesRegex(ValueError, "quantity_of_interest"):
            ReferenceSpec(
                reference_id="solver/mismatch",
                kind=ReferenceKind.NUMERICAL,
                context_of_use="steady external-flow regression",
                quantity_of_interest="drag_coefficient",
                spatial_study=study,
            )


if __name__ == "__main__":
    unittest.main()
