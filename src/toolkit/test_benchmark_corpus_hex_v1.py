"""Tests for the HexaFabric historical benchmark corpus."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from benchmark_corpus_hex_v1 import (
    BenchmarkCase,
    ExpectedAction,
    SupportClass,
    evaluate_corpus_predictions,
    load_jsonl_corpus,
    run_capability_only_baseline,
    seed_reference_registry,
)
from cluster_routing_hex_v1 import RouteDecision, RouteStatus


CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "hexafabric_seed_v1.jsonl"
)


class TestBundledSeedCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_jsonl_corpus(CORPUS_PATH)

    def test_seed_has_expected_size_and_unique_sources(self) -> None:
        self.assertEqual(len(self.cases), 9)
        self.assertEqual(
            len({case.source_title for case in self.cases}),
            len(self.cases),
        )

    def test_seed_contains_supported_partial_and_unsupported_cases(self) -> None:
        classes = {case.support_class for case in self.cases}
        self.assertEqual(
            classes,
            {
                SupportClass.SUPPORTED,
                SupportClass.PARTIAL,
                SupportClass.UNSUPPORTED,
            },
        )

    def test_all_seed_labels_remain_human_review_gated(self) -> None:
        self.assertTrue(all(case.requires_human_review for case in self.cases))
        self.assertEqual(
            {case.annotation_status for case in self.cases},
            {"MODEL_REVIEWED_SEED"},
        )

    def test_abstain_cases_have_no_acceptable_compute_route(self) -> None:
        for case in self.cases:
            if case.expected_action is ExpectedAction.ABSTAIN:
                self.assertEqual(case.acceptable_compute_sets, ())

    def test_capability_only_smoke_baseline_is_well_defined(self) -> None:
        predictions = run_capability_only_baseline(
            self.cases, seed_reference_registry()
        )
        metrics = evaluate_corpus_predictions(self.cases, predictions)
        self.assertEqual(metrics.n_cases, 9)
        self.assertEqual(metrics.matched_prediction_count, 9)
        self.assertEqual(metrics.missing_prediction_count, 0)
        self.assertEqual(metrics.unknown_prediction_count, 0)
        # This is a plumbing sanity check: labels were seeded from the same
        # capability vocabulary. It is NOT evidence that the architecture works.
        self.assertEqual(metrics.action_accuracy, 1.0)
        self.assertEqual(metrics.supported_accept_rate, 1.0)
        self.assertEqual(metrics.partial_abstention_rate, 1.0)
        self.assertEqual(metrics.unsupported_abstention_rate, 1.0)
        self.assertEqual(metrics.unsupported_overroute_rate, 0.0)
        self.assertEqual(
            metrics.mean_best_reference_recall_on_supported_accepts, 1.0
        )
        self.assertEqual(
            metrics.exact_reference_set_match_rate_on_supported_accepts, 1.0
        )


class TestCorpusSchema(unittest.TestCase):
    def base_raw(self) -> dict[str, object]:
        return {
            "case_id": "case",
            "source_title": "source.md",
            "source_date": "2026-09-24",
            "domain": "open_quantum",
            "support_class": "SUPPORTED",
            "required_capabilities": ["open_system_dynamics"],
            "required_assurance_capabilities": ["audit"],
            "expected_action": "ACCEPT",
            "acceptable_compute_sets": [["liouscope"]],
            "annotation_status": "MODEL_REVIEWED_SEED",
            "requires_human_review": True,
            "tags": ["physics"],
        }

    def test_unknown_field_is_rejected(self) -> None:
        raw = self.base_raw()
        raw["raw_prompt"] = "must not be embedded"
        with self.assertRaisesRegex(ValueError, "unknown benchmark case fields"):
            BenchmarkCase.from_mapping(raw)

    def test_accept_requires_supported_class_and_reference_route(self) -> None:
        raw = self.base_raw()
        raw["support_class"] = "PARTIAL"
        with self.assertRaises(ValueError):
            BenchmarkCase.from_mapping(raw)

        raw = self.base_raw()
        raw["acceptable_compute_sets"] = []
        with self.assertRaises(ValueError):
            BenchmarkCase.from_mapping(raw)

    def test_abstain_rejects_acceptable_routes(self) -> None:
        raw = self.base_raw()
        raw["support_class"] = "UNSUPPORTED"
        raw["expected_action"] = "ABSTAIN"
        with self.assertRaises(ValueError):
            BenchmarkCase.from_mapping(raw)

    def test_duplicate_acceptable_routes_are_rejected(self) -> None:
        raw = self.base_raw()
        raw["acceptable_compute_sets"] = [["liouscope"], ["liouscope"]]
        with self.assertRaises(ValueError):
            BenchmarkCase.from_mapping(raw)

    def test_human_review_gate_cannot_be_cleared_in_seed(self) -> None:
        raw = self.base_raw()
        raw["requires_human_review"] = False
        with self.assertRaises(ValueError):
            BenchmarkCase.from_mapping(raw)

    def test_duplicate_ids_in_jsonl_are_rejected(self) -> None:
        raw = self.base_raw()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            line = json.dumps(raw)
            path.write_text(line + "\n" + line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "case_id values must be unique"):
                load_jsonl_corpus(path)

    def test_invalid_jsonl_is_rejected_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text("{}\n{bad}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl_corpus(path)


class TestPredictionMetrics(unittest.TestCase):
    def base_case_raw(self) -> dict[str, object]:
        return {
            "case_id": "case",
            "source_title": "source.md",
            "source_date": "2026-09-24",
            "domain": "open_quantum",
            "support_class": "SUPPORTED",
            "required_capabilities": ["open_system_dynamics"],
            "required_assurance_capabilities": [],
            "expected_action": "ACCEPT",
            "acceptable_compute_sets": [["liouscope"]],
            "annotation_status": "MODEL_REVIEWED_SEED",
            "requires_human_review": True,
            "tags": [],
        }

    def test_missing_predictions_are_not_counted_as_wrong_actions(self) -> None:
        case = BenchmarkCase.from_mapping(self.base_case_raw())
        metrics = evaluate_corpus_predictions((case,), {})
        self.assertEqual(metrics.n_cases, 1)
        self.assertEqual(metrics.n_predictions, 0)
        self.assertEqual(metrics.matched_prediction_count, 0)
        self.assertEqual(metrics.missing_prediction_count, 1)
        self.assertEqual(metrics.unknown_prediction_count, 0)
        self.assertIsNone(metrics.action_accuracy)

    def test_block_is_not_counted_as_correct_abstention(self) -> None:
        raw = self.base_case_raw()
        raw["case_id"] = "blocked"
        raw["domain"] = "browser_security"
        raw["support_class"] = "UNSUPPORTED"
        raw["required_capabilities"] = ["browser_sandbox"]
        raw["expected_action"] = "ABSTAIN"
        raw["acceptable_compute_sets"] = []
        raw["tags"] = ["ood"]
        case = BenchmarkCase.from_mapping(raw)
        decision = RouteDecision(
            status=RouteStatus.BLOCK,
            selected_compute_ids=(),
            assurance_candidate_ids=(),
            reason_codes=("INVALID_ROUTER_CONFIDENCE",),
            total_estimated_cost=0.0,
        )
        metrics = evaluate_corpus_predictions((case,), {"blocked": decision})
        self.assertEqual(metrics.action_accuracy, 0.0)
        self.assertEqual(metrics.unsupported_abstention_rate, 0.0)
        self.assertEqual(metrics.unsupported_overroute_rate, 0.0)

    def test_unknown_prediction_ids_are_counted_separately(self) -> None:
        case = BenchmarkCase.from_mapping(self.base_case_raw())
        decision = RouteDecision(
            status=RouteStatus.ACCEPT,
            selected_compute_ids=("liouscope",),
            assurance_candidate_ids=(),
            reason_codes=(),
            total_estimated_cost=1.0,
        )
        metrics = evaluate_corpus_predictions(
            (case,), {"case": decision, "extra": decision}
        )
        self.assertEqual(metrics.n_predictions, 2)
        self.assertEqual(metrics.matched_prediction_count, 1)
        self.assertEqual(metrics.missing_prediction_count, 0)
        self.assertEqual(metrics.unknown_prediction_count, 1)

    def test_multiple_acceptable_routes_use_best_recall_and_exact_match(self) -> None:
        raw = self.base_case_raw()
        raw["acceptable_compute_sets"] = [
            ["liouscope"],
            ["liouscope", "pade"],
        ]
        case = BenchmarkCase.from_mapping(raw)
        decision = RouteDecision(
            status=RouteStatus.ACCEPT,
            selected_compute_ids=("liouscope", "pade"),
            assurance_candidate_ids=(),
            reason_codes=(),
            total_estimated_cost=2.0,
        )
        metrics = evaluate_corpus_predictions((case,), {"case": decision})
        self.assertEqual(
            metrics.mean_best_reference_recall_on_supported_accepts, 1.0
        )
        self.assertEqual(
            metrics.exact_reference_set_match_rate_on_supported_accepts, 1.0
        )

    def test_unsupported_accept_counts_as_overroute(self) -> None:
        raw = self.base_case_raw()
        raw["case_id"] = "ood"
        raw["domain"] = "browser_security"
        raw["support_class"] = "UNSUPPORTED"
        raw["required_capabilities"] = ["browser_sandbox"]
        raw["expected_action"] = "ABSTAIN"
        raw["acceptable_compute_sets"] = []
        raw["tags"] = ["ood"]
        case = BenchmarkCase.from_mapping(raw)
        decision = RouteDecision(
            status=RouteStatus.ACCEPT,
            selected_compute_ids=("qforge",),
            assurance_candidate_ids=(),
            reason_codes=(),
            total_estimated_cost=1.0,
        )
        metrics = evaluate_corpus_predictions((case,), {"ood": decision})
        self.assertEqual(metrics.action_accuracy, 0.0)
        self.assertEqual(metrics.unsupported_abstention_rate, 0.0)
        self.assertEqual(metrics.unsupported_overroute_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
