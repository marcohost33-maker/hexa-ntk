"""No-training, stdlib-only adversarial and arithmetic oracles."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from phase1_envelope_hex_v3 import (
    Scenario,
    _passes,
    evaluate_envelope,
    parse_json_scenario,
    EvidenceError,
)

ROOT = Path(__file__).resolve().parents[2]
# Transcribed values, not the source record (the name says so). The evidence
# tier TRANSCRIBED_TEST_FIXTURE below is the only tier it may reach.
SOURCE = Path(__file__).resolve().parent / "TRANSCRIBED_VALUES_TEST_FIXTURE_NOT_SOURCE.json"


class EnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(SOURCE.read_text(encoding="utf-8"))

    def check(self, **config):
        return evaluate_envelope(self.data, Scenario.from_mapping(config), evidence_tier="TRANSCRIBED_TEST_FIXTURE")

    def test_default_matches_previous_candidate_arithmetic(self):
        result = self.check()
        self.assertAlmostEqual(result["nominal_median_scenario"]["total_hours"], 43.70609893005556)
        self.assertAlmostEqual(result["observed_sample_max_scenario"]["total_hours"], 45.433752111222226)
        self.assertEqual(result["validation_passes_budgeted"], 13)
        self.assertEqual(result["total_training_epochs_budgeted"], 650)
        self.assertEqual(result["owner_verdict"], "HOLD_PREREGISTRATION_DRAFT")
        self.assertEqual(result["cost_measurement_fidelity"], "PR91_THREE_EPOCH_SCHEDULER_NOT_PHASE1_FIDELITY")

    def test_validation_every_epoch_counts_650(self):
        baseline = self.check()
        extra = self.check(validation_interval_epochs=1)
        self.assertEqual(extra["validation_passes_per_fit"], 50)
        self.assertEqual(extra["validation_passes_budgeted"], 650)
        self.assertGreater(
            extra["nominal_median_scenario"]["total_hours"], baseline["nominal_median_scenario"]["total_hours"]
        )
        val = baseline["nominal_median_scenario"]["validation_hours"]
        self.assertAlmostEqual(extra["nominal_median_scenario"]["validation_hours"], 50 * val)

    def test_validation_includes_last_nonmultiple(self):
        self.assertEqual(_passes(50, 7), 8)
        self.assertEqual(_passes(50, 25), 2)
        self.assertEqual(_passes(50, 50), 1)
        self.assertEqual(_passes(50, 1), 50)

    def test_checkpoint_overhead_per_fit_per_checkpoint(self):
        result = self.check(checkpoint_interval_epochs=10, checkpoint_seconds_each=60)
        self.assertEqual(result["checkpoint_writes_budgeted"], 65)
        self.assertEqual(result["overhead_breakdown_assumed_seconds"]["checkpoint_seconds"], 3900)
        self.assertAlmostEqual(result["nominal_median_scenario"]["assumed_overhead_hours"], 3900 / 3600)

    def test_training_slowdown_exceeds_budget(self):
        result = self.check(training_multiplier=1.1)
        self.assertEqual(result["nominal_median_scenario"]["budget_fit_class"], "ESTIMATE_EXCEEDS_CAP")
        self.assertAlmostEqual(result["nominal_median_scenario"]["total_hours"], 48.07586784227779)

    def test_production_like_composite_case_exceeds_budget(self):
        r = self.check(
            validation_interval_epochs=1,
            checkpoint_interval_epochs=10,
            checkpoint_seconds_each=45,
            fit_setup_seconds_each=30,
            data_generation_seconds=3600,
            contingency_seconds=1800,
            training_multiplier=1.05,
        )
        self.assertGreater(r["nominal_median_scenario"]["total_hours"], 48.0)
        self.assertGreater(r["observed_sample_max_scenario"]["total_hours"], 48.0)
        self.assertAlmostEqual(
            r["nominal_median_scenario"]["maximum_training_multiplier_for_this_assumed_scenario"], 1.0334339682973028
        )
        self.assertGreater(r["nominal_median_scenario"]["minimum_speedup_from_assumed_training_time_to_fit"], 1.0)
        self.assertEqual(r["research_status"], "NONE_NO_T1_MODEL_FIT")

    def test_overheads_counted_once_correctly(self):
        r = self.check(
            fit_setup_seconds_each=10, data_generation_seconds=20, final_evaluation_seconds=30, contingency_seconds=40
        )
        self.assertEqual(r["overhead_breakdown_assumed_seconds"]["fit_setup_seconds"], 130)
        self.assertAlmostEqual(r["nominal_median_scenario"]["assumed_overhead_hours"], 220 / 3600)

    def test_unknown_field_rejected(self):
        with self.assertRaises(EvidenceError):
            Scenario.from_mapping({"allow_T1": True})

    def test_strings_bool_nonfinite_rejected(self):
        for name, bad in [
            ("training_multiplier", False),
            ("validation_multiplier", "1"),
            ("checkpoint_seconds_each", float("nan")),
            ("contingency_seconds", float("inf")),
            ("budget_hours", True),
            ("validation_interval_epochs", 1.5),
            ("checkpoint_interval_epochs", True),
            ("data_generation_seconds", -1),
        ]:
            with self.subTest(name=name, bad=str(bad)), self.assertRaises(EvidenceError):
                Scenario.from_mapping({name: bad})

    def test_sample_mutations_fail_on_R2_validator(self):
        for key in ("passed", "source", "configs"):
            d = copy.deepcopy(self.data)
            d.pop(key)
            with self.subTest(key=key), self.assertRaises(EvidenceError):
                evaluate_envelope(d, Scenario.from_mapping({}), evidence_tier="TRANSCRIBED_TEST_FIXTURE")

    def test_evidence_tier_cannot_make_approval(self):
        for tier in ("TRANSCRIBED_TEST_FIXTURE", "PINNED_TWO_SOURCE_CANDIDATE"):
            r = evaluate_envelope(self.data, Scenario.from_mapping({}), evidence_tier=tier)
            self.assertTrue(r["owner_verdict"].startswith("HOLD_"))
            self.assertFalse(r["statistical_upper_bound"])
        with self.assertRaises(EvidenceError):
            evaluate_envelope(self.data, Scenario.from_mapping({}), evidence_tier="APPROVED")

    def test_pinned_wrong_sha_cli_rejects_without_survey(self):
        p = subprocess.run(
            [sys.executable, str(ROOT / "src/toolkit/phase1_envelope_hex_v3.py"), str(SOURCE)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 2)
        self.assertEqual(json.loads(p.stdout)["owner_verdict"], "HOLD_INVALID_INPUT")

    def test_fixture_cli_explicit(self):
        p = subprocess.run(
            [sys.executable, str(ROOT / "src/toolkit/phase1_envelope_hex_v3.py"), str(SOURCE), "--fixture-only"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)["evidence_tier"], "TRANSCRIBED_TEST_FIXTURE")
        self.assertTrue(json.loads(p.stdout)["owner_verdict"].startswith("HOLD"))

    def test_strict_cannot_use_fixture_and_radius(self):
        p = subprocess.run(
            [
                sys.executable,
                str(ROOT / "src/toolkit/phase1_envelope_hex_v3.py"),
                str(SOURCE),
                "--fixture-only",
                "--radius-survey",
                str(SOURCE),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 2)

    def test_scenario_parser_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "scenario.json"
            p.write_text('{"budget_hours":48,"budget_hours":49}', encoding="utf-8")
            with self.assertRaises(EvidenceError):
                parse_json_scenario(p)

    def test_scenario_parser_rejects_oversized(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "scenario.json"
            p.write_bytes(b" " * 65537)
            with self.assertRaises(EvidenceError):
                parse_json_scenario(p)

    def test_intervals_enforced(self):
        with self.assertRaises(EvidenceError):
            Scenario.from_mapping({"validation_interval_epochs": 51})
        with self.assertRaises(EvidenceError):
            Scenario.from_mapping({"checkpoint_interval_epochs": 0})

    def test_negative_overheads_refused(self):
        with self.assertRaises(EvidenceError):
            Scenario.from_mapping({"fit_setup_seconds_each": -0.1})

    def test_very_high_budget_does_not_raise_owner_gate(self):
        r = self.check(budget_hours=100000)
        self.assertEqual(r["nominal_median_scenario"]["budget_fit_class"], "ESTIMATE_WITHIN_CAP_NOT_CERTIFIED")
        self.assertEqual(r["owner_verdict"], "HOLD_PREREGISTRATION_DRAFT")


if __name__ == "__main__":
    unittest.main()
