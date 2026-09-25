
"""
test_stress_hex_v3.py
=====================
Contract tests for axis-preserving HexaNTK stress evaluation.
"""

from __future__ import annotations

import math
import sys
import unittest
from fractions import Fraction

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

    def test_positive_ratio_underflow_is_not_reported_as_zero(self) -> None:
        # Numerator is floored too, so underflow needs a subnormal floor.
        value = compute_degradation_factor(
            5e-324,
            sys.float_info.max,
            floor=5e-324,
        )
        self.assertGreater(value, 0.0)
        self.assertEqual(value, math.nextafter(0.0, math.inf))

    def test_floor_never_inverts_direction(self) -> None:
        """F32-1: E_s > E_0 must never yield D < 1 (and E_s < E_0 never D > 1).

        Under the old E_s / max(E_0, eps), D(1e-13, 0.0) = 0.1 with the
        default floor 1e-12 -- a larger shifted error read as a 10x
        improvement.
        """
        eps = 1e-12
        worse = [
            (1e-13, 0.0),
            (1e-13, 1e-14),
            (2e-12, 1e-13),
            (5e-13, 4e-13),
            (0.2, 0.1),
            (1e-12, 0.0),
        ]
        for shifted, baseline in worse:
            with self.subTest(shifted=shifted, baseline=baseline):
                self.assertGreaterEqual(
                    compute_degradation_factor(shifted, baseline, floor=eps),
                    1.0,
                )
                self.assertLessEqual(
                    compute_degradation_factor(baseline, shifted, floor=eps),
                    1.0,
                )

    def test_below_floor_on_both_sides_is_indistinguishable(self) -> None:
        for shifted, baseline in ((1e-13, 0.0), (1e-13, 1e-14), (0.0, 5e-13)):
            with self.subTest(shifted=shifted, baseline=baseline):
                self.assertEqual(
                    compute_degradation_factor(shifted, baseline, floor=1e-12),
                    1.0,
                )

    def test_floored_values_follow_spec_formula(self) -> None:
        eps = 1e-6
        cases = {
            (0.1, 0.0): 0.1 / eps,
            (0.0, 0.1): eps / 0.1,
            (2e-7, 0.5): eps / 0.5,
            (0.4, 0.2): 2.0,
        }
        for (shifted, baseline), expected in cases.items():
            with self.subTest(shifted=shifted, baseline=baseline):
                self.assertAlmostEqual(
                    compute_degradation_factor(shifted, baseline, floor=eps),
                    expected,
                    delta=1e-12 * expected,
                )

    def test_direction_property_randomized(self) -> None:
        """Seeded property check across magnitudes, incl. subnormal floors."""
        import random

        rng = random.Random(20260923)
        floors = (5e-324, 1e-300, 1e-12, 1e-6, 1.0)
        for _ in range(2000):
            eps = rng.choice(floors)
            a = rng.choice((0.0, 5e-324, 10.0 ** rng.uniform(-320, 300)))
            b = rng.choice((0.0, 5e-324, 10.0 ** rng.uniform(-320, 300)))
            d = compute_degradation_factor(a, b, floor=eps)
            self.assertTrue(math.isfinite(d) and d > 0.0, (a, b, eps, d))
            if a > b:
                self.assertGreaterEqual(d, 1.0, (a, b, eps, d))
            elif a < b:
                self.assertLessEqual(d, 1.0, (a, b, eps, d))
            else:
                self.assertEqual(d, 1.0, (a, b, eps, d))

    def test_invalid_values_fail_closed(self) -> None:
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    compute_degradation_factor(value, 0.1)

        with self.assertRaises(ValueError):
            compute_degradation_factor(0.1, 0.1, floor=0.0)
    def test_extended_real_underflow_cannot_cross_numeric_contract(self) -> None:
        tiny = Fraction(1, 10**1000)
        with self.assertRaisesRegex(ValueError, "finite and >= 0"):
            compute_degradation_factor(-tiny, Fraction(1, 10))
        with self.assertRaisesRegex(ValueError, "underflows binary64"):
            StressCase(
                "tiny",
                ShiftAxis.NOISE,
                ShiftRegime.OOD,
                tiny,
            )


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

    def test_underflow_is_explicit_in_generated_report(self) -> None:
        report = evaluate_stress_cases(
            error_metric_name="nrmse",
            baseline_case_id="base",
            baseline_error=sys.float_info.max,
            degradation_floor=5e-324,
            cases=[
                StressCase(
                    "tiny-positive",
                    ShiftAxis.NOISE,
                    ShiftRegime.OOD,
                    5e-324,
                )
            ],
        )
        case = report.cases[0]
        self.assertGreater(case.degradation_factor, 0.0)
        self.assertTrue(case.degradation_underflowed)
        self.assertFalse(case.degradation_saturated)
        self.assertTrue(
            report.to_dict()["cases"][0]["degradation_underflowed"]
        )

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

    def test_manual_report_rejects_wrong_underflow_flag(self) -> None:
        """S9: the underflow consistency check in StressReport must bite."""
        tiny = math.nextafter(0.0, math.inf)
        for claimed in (False, True):
            # claimed=False: case really underflows (5e-324 / float max,
            # floor 5e-324). claimed=True: plain ratio 0.2 / 0.1, no underflow.
            if claimed:
                error, baseline, factor, floor = 0.2, 0.1, 2.0, 1e-12
            else:
                error, baseline, factor, floor = (
                    5e-324,
                    sys.float_info.max,
                    tiny,
                    5e-324,
                )
            with self.subTest(claimed_underflow=claimed):
                bad_case = StressCaseResult(
                    case_id="shift",
                    axis=ShiftAxis.NOISE,
                    regime=ShiftRegime.OOD,
                    error_value=error,
                    degradation_factor=factor,
                    degradation_underflowed=claimed,
                )
                with self.assertRaisesRegex(
                    ValueError, "degradation_underflowed"
                ):
                    StressReport(
                        error_metric_name="nrmse",
                        baseline_case_id="base",
                        baseline_error=baseline,
                        cases=(bad_case,),
                        degradation_floor=floor,
                    )

    def test_floored_flag_is_explicit_in_generated_report(self) -> None:
        report = evaluate_stress_cases(
            error_metric_name="nrmse",
            baseline_case_id="base",
            baseline_error=0.0,
            cases=[
                StressCase(
                    "tiny-worse",
                    ShiftAxis.NOISE,
                    ShiftRegime.OOD,
                    1e-13,
                ),
                StressCase(
                    "clean",
                    ShiftAxis.RESOLUTION,
                    ShiftRegime.OOD,
                    0.2,
                ),
            ],
        )
        tiny, clean = report.cases
        self.assertTrue(tiny.degradation_floored)
        self.assertEqual(tiny.degradation_factor, 1.0)
        self.assertTrue(clean.degradation_floored)  # baseline 0 < floor
        payload = report.to_dict()
        self.assertIn("degradation_floored", payload["cases"][0])
        self.assertIs(payload["cases"][0]["degradation_floored"], True)

        exact = evaluate_stress_cases(
            error_metric_name="nrmse",
            baseline_case_id="base",
            baseline_error=0.1,
            cases=[StressCase("x", ShiftAxis.PARAMETER, ShiftRegime.OOD, 0.2)],
        )
        self.assertFalse(exact.cases[0].degradation_floored)
        self.assertIs(exact.to_dict()["cases"][0]["degradation_floored"], False)

    def test_manual_report_rejects_wrong_floored_flag(self) -> None:
        for claimed, baseline, factor in ((False, 0.0, 0.2 / 1e-12), (True, 0.1, 2.0)):
            with self.subTest(claimed_floored=claimed):
                bad_case = StressCaseResult(
                    case_id="shift",
                    axis=ShiftAxis.PARAMETER,
                    regime=ShiftRegime.OOD,
                    error_value=0.2,
                    degradation_factor=factor,
                    degradation_floored=claimed,
                )
                with self.assertRaisesRegex(ValueError, "degradation_floored"):
                    StressReport(
                        error_metric_name="nrmse",
                        baseline_case_id="base",
                        baseline_error=baseline,
                        cases=(bad_case,),
                        degradation_floor=1e-12,
                    )

    def test_floored_flag_must_be_bool(self) -> None:
        with self.assertRaisesRegex(TypeError, "degradation_floored"):
            StressCaseResult(
                case_id="shift",
                axis=ShiftAxis.PARAMETER,
                regime=ShiftRegime.OOD,
                error_value=0.2,
                degradation_factor=2.0,
                degradation_floored=1,
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


class TestBinary64OverflowContract(unittest.TestCase):
    """Exact Reals beyond binary64 range must fail with the contract ValueError.

    float(Fraction(10**400)) and float(10**400) raise OverflowError, which is
    not a ValueError subclass. The checks below capture any exception and
    assert its exact type, so a regression shows up as an assertion failure
    rather than as an unrelated error.
    """

    # Smallest integer that float() rounds past binary64 max: max + half an
    # ulp is a tie, and round-half-to-even goes to 2**1024, i.e. overflow.
    FIRST_OVERFLOWING_INT = 2**1024 - 2**970

    def assert_contract_value_error(self, call, pattern: str) -> None:
        with self.assertRaises(Exception) as ctx:
            call()
        exc = ctx.exception
        self.assertIs(
            type(exc),
            ValueError,
            f"expected contract ValueError, got {type(exc).__name__}: {exc}",
        )
        self.assertRegex(str(exc), pattern)

    def test_huge_exact_reals_raise_contract_value_error(self) -> None:
        huge_values = (
            Fraction(10**400),
            10**400,
            2**1024,
            self.FIRST_OVERFLOWING_INT,
        )
        for value in huge_values:
            with self.subTest(
                value_type=type(value).__name__,
                bits=int(value).bit_length(),
            ):
                self.assert_contract_value_error(
                    lambda: compute_degradation_factor(value, 0.1),
                    "shifted_error must be finite and >= 0",
                )
                self.assert_contract_value_error(
                    lambda: compute_degradation_factor(0.1, value),
                    "baseline_error must be finite and >= 0",
                )
                self.assert_contract_value_error(
                    lambda: compute_degradation_factor(0.1, 0.1, floor=value),
                    "floor must be finite and >= 0",
                )
                self.assert_contract_value_error(
                    lambda: StressCase(
                        "huge", ShiftAxis.NOISE, ShiftRegime.OOD, value
                    ),
                    "error_value must be finite and >= 0",
                )
                self.assert_contract_value_error(
                    lambda: evaluate_stress_cases(
                        error_metric_name="nrmse",
                        baseline_case_id="base",
                        baseline_error=value,
                        cases=[],
                    ),
                    "baseline_error must be finite and >= 0",
                )

    def test_overflow_keeps_original_exception_as_cause(self) -> None:
        with self.assertRaises(Exception) as ctx:
            compute_degradation_factor(Fraction(10**400), 0.1)
        self.assertIs(type(ctx.exception), ValueError)
        self.assertIsInstance(ctx.exception.__cause__, OverflowError)

    def test_huge_negative_exact_real_keeps_sign_error(self) -> None:
        for value in (-Fraction(10**400), -(10**400)):
            with self.subTest(value_type=type(value).__name__):
                self.assert_contract_value_error(
                    lambda: compute_degradation_factor(value, 0.1),
                    "shifted_error must be finite and >= 0",
                )
                with self.assertRaises(ValueError) as ctx:
                    compute_degradation_factor(value, 0.1)
                # The sign gate fires before any binary64 conversion.
                self.assertIsNone(ctx.exception.__cause__)

    def test_values_at_or_just_below_binary64_max_are_accepted(self) -> None:
        max_float = sys.float_info.max
        just_below_threshold = self.FIRST_OVERFLOWING_INT - 1
        for value in (
            max_float,
            Fraction(max_float),
            int(max_float),
            just_below_threshold,
            Fraction(just_below_threshold),
        ):
            with self.subTest(value_type=type(value).__name__):
                case = StressCase(
                    "edge", ShiftAxis.NOISE, ShiftRegime.OOD, value
                )
                self.assertEqual(case.error_value, max_float)
                self.assertIs(type(case.error_value), float)
                self.assertEqual(compute_degradation_factor(value, value), 1.0)
        self.assertEqual(
            compute_degradation_factor(Fraction(max_float), 1.0),
            max_float,
        )

    def test_huge_exact_metadata_value_raises_contract_value_error(self) -> None:
        self.assert_contract_value_error(
            lambda: StressCase(
                "meta",
                ShiftAxis.NOISE,
                ShiftRegime.OOD,
                0.2,
                {"scale": Fraction(10**400)},
            ),
            "metadata.scale contains a non-finite numeric value",
        )
        case = StressCase(
            "meta-edge",
            ShiftAxis.NOISE,
            ShiftRegime.OOD,
            0.2,
            {"scale": Fraction(sys.float_info.max)},
        )
        self.assertEqual(case.metadata["scale"], sys.float_info.max)


if __name__ == "__main__":
    unittest.main()
