
"""
test_stress_hex_v3.py
=====================
Contract tests for axis-preserving HexaNTK stress evaluation.
"""

from __future__ import annotations

import math
import sys
import unittest

from stress_hex_v3 import (
    ShiftAxis,
    ShiftRegime,
    StressCase,
    StressCaseResult,
    StressReport,
    compute_degradation_factor,
    evaluate_stress_cases,
)


class TestDegradationFactor(unittest.TestCase):
    def test_ratio(self) -> None:
        self.assertAlmostEqual(compute_degradation_factor(0.2, 0.1), 2.0)

    def test_both_zero_means_no_degradation(self) -> None:
        self.assertEqual(compute_degradation_factor(0.0, 0.0), 1.0)

    def test_zero_baseline_does_not_hide_nonzero_shift(self) -> None:
        value = compute_degradation_factor(0.1, 0.0, floor=1e-6)
        self.assertAlmostEqual(value, 100000.0)

    def test_overflow_saturates_instead_of_returning_inf(self) -> None:
        value = compute_degradation_factor(1.0, 0.0, floor=5e-324)
        self.assertEqual(value, sys.float_info.max)
        self.assertTrue(math.isfinite(value))

    def test_invalid_values_fail_closed(self) -> None:
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    compute_degradation_factor(value, 0.1)

        with self.assertRaises(ValueError):
            compute_degradation_factor(0.1, 0.1, floor=0.0)


class TestStressReport(unittest.TestCase):
    def test_axes_remain_separate(self) -> None:
        report = evaluate_stress_cases(
            error_metric_name="nrmse",
            baseline_case_id="id/base",
            baseline_error=0.05,
            cases=[
                StressCase(
                    "bc/dirichlet",
                    ShiftAxis.BOUNDARY_CONDITION,
                    ShiftRegime.OOD,
                    0.20,
                    {"family": "dirichlet"},
                ),
                StressCase(
                    "resolution/2x",
                    ShiftAxis.RESOLUTION,
                    ShiftRegime.OOD,
                    0.10,
                    {"scale": 2},
                ),
            ],
        )

        grouped = report.by_axis()
        self.assertEqual(set(grouped), {"boundary_condition", "resolution"})
        self.assertAlmostEqual(
            grouped["boundary_condition"][0].degradation_factor,
            4.0,
        )
        self.assertAlmostEqual(
            grouped["resolution"][0].degradation_factor,
            2.0,
        )

        payload = report.to_dict()
        self.assertNotIn("overall_score", payload)
        self.assertEqual(
            payload["metric_semantics"],
            "nonnegative_error_lower_is_better",
        )
        self.assertEqual(payload["cases"][0]["axis"], "boundary_condition")
        self.assertEqual(payload["cases"][0]["regime"], "ood")

    def test_duplicate_case_ids_fail_closed(self) -> None:
        cases = [
            StressCase("same", ShiftAxis.PARAMETER, ShiftRegime.OOD, 0.1),
            StressCase("same", ShiftAxis.GEOMETRY, ShiftRegime.OOD, 0.2),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_stress_cases(
                error_metric_name="nrmse",
                baseline_case_id="base",
                baseline_error=0.05,
                cases=cases,
            )

    def test_baseline_id_cannot_be_reused(self) -> None:
        with self.assertRaisesRegex(ValueError, "baseline_case_id"):
            evaluate_stress_cases(
                error_metric_name="nrmse",
                baseline_case_id="base",
                baseline_error=0.05,
                cases=[
                    StressCase(
                        "base",
                        ShiftAxis.PARAMETER,
                        ShiftRegime.ID,
                        0.05,
                    )
                ],
            )

    def test_case_validation_rejects_nonfinite_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            StressCase(
                "bad",
                ShiftAxis.PARAMETER,
                ShiftRegime.OOD,
                float("nan"),
            )

    def test_enum_contract_contains_claim_relevant_axes(self) -> None:
        required = {
            "parameter",
            "initial_condition",
            "forcing",
            "boundary_condition",
            "geometry",
            "resolution",
            "timestep",
            "rollout_horizon",
            "noise",
            "topology",
        }
        self.assertEqual({axis.value for axis in ShiftAxis}, required)

    def test_saturation_is_explicit_in_generated_report(self) -> None:
        report = evaluate_stress_cases(
            error_metric_name="nrmse",
            baseline_case_id="base",
            baseline_error=0.0,
            degradation_floor=5e-324,
            cases=[
                StressCase(
                    "severe",
                    ShiftAxis.ROLLOUT_HORIZON,
                    ShiftRegime.EXTRAPOLATION,
                    1.0,
                )
            ],
        )
        case = report.cases[0]
        self.assertEqual(case.degradation_factor, sys.float_info.max)
        self.assertTrue(case.degradation_saturated)
        self.assertTrue(report.to_dict()["cases"][0]["degradation_saturated"])

    def test_manual_report_rejects_inconsistent_factor(self) -> None:
        bad_case = StressCaseResult(
            case_id="shift",
            axis=ShiftAxis.PARAMETER,
            regime=ShiftRegime.OOD,
            error_value=0.2,
            degradation_factor=999.0,
        )
        with self.assertRaisesRegex(ValueError, "degradation_factor"):
            StressReport(
                error_metric_name="nrmse",
                baseline_case_id="base",
                baseline_error=0.1,
                cases=(bad_case,),
                degradation_floor=1e-12,
            )

    def test_manual_report_rejects_wrong_saturation_flag(self) -> None:
        bad_case = StressCaseResult(
            case_id="shift",
            axis=ShiftAxis.PARAMETER,
            regime=ShiftRegime.OOD,
            error_value=1.0,
            degradation_factor=sys.float_info.max,
            degradation_saturated=False,
        )
        with self.assertRaisesRegex(ValueError, "degradation_saturated"):
            StressReport(
                error_metric_name="nrmse",
                baseline_case_id="base",
                baseline_error=0.0,
                cases=(bad_case,),
                degradation_floor=5e-324,
            )

    def test_nested_metadata_is_snapshot_immutable(self) -> None:
        source = {
            "config": {
                "resolution": [64, 64],
                "family": "dirichlet",
            }
        }
        case = StressCase(
            "bc",
            ShiftAxis.BOUNDARY_CONDITION,
            ShiftRegime.OOD,
            0.2,
            source,
        )
        source["config"]["resolution"][0] = 999

        report = evaluate_stress_cases(
            error_metric_name="nrmse",
            baseline_case_id="base",
            baseline_error=0.1,
            cases=[case],
        )
        payload = report.to_dict()
        self.assertEqual(payload["cases"][0]["metadata"]["config"]["resolution"], [64, 64])

        with self.assertRaises(TypeError):
            report.cases[0].metadata["new"] = "mutate"

        nested = report.cases[0].metadata["config"]
        with self.assertRaises(TypeError):
            nested["family"] = "neumann"

    def test_non_case_input_fails_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "StressCase"):
            evaluate_stress_cases(
                error_metric_name="nrmse",
                baseline_case_id="base",
                baseline_error=0.1,
                cases=[object()],
            )


if __name__ == "__main__":
    unittest.main()
