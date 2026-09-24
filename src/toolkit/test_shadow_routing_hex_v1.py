"""Tests for non-executing shadow-routing evaluation."""
from __future__ import annotations

import unittest

from cluster_routing_hex_v1 import (
    ExpertPlane,
    ExpertSpec,
    RouteStatus,
    RouterProposal,
    TaskSpec,
)
from shadow_routing_hex_v1 import ShadowCase, evaluate_shadow_cases


def expert(
    expert_id: str,
    capabilities: tuple[str, ...],
    *,
    family: str | None = None,
    plane: ExpertPlane = ExpertPlane.COMPUTE,
    cost: float = 1.0,
) -> ExpertSpec:
    return ExpertSpec(
        expert_id=expert_id,
        capabilities=frozenset(capabilities),
        domains=frozenset({"research"}),
        implementation_family=family or expert_id,
        plane=plane,
        estimated_cost=cost,
    )


def task(**kwargs: object) -> TaskSpec:
    base = dict(
        task_id="t",
        domain="research",
        required_capabilities=frozenset({"solve"}),
    )
    base.update(kwargs)
    return TaskSpec(**base)


class TestShadowRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.experts = (
            expert("pade", ("solve",), family="rational", cost=1.0),
            expert("liouscope", ("solve",), family="dynamics", cost=2.0),
            expert("mrt", ("inverse",), family="inverse", cost=3.0),
            expert("ark", ("audit",), plane=ExpertPlane.ASSURANCE, cost=0.5),
        )

    def test_empty_benchmark_is_well_defined(self) -> None:
        records, metrics = evaluate_shadow_cases((), self.experts)
        self.assertEqual(records, ())
        self.assertEqual(metrics.n_cases, 0)
        self.assertEqual(metrics.acceptance_rate, 0.0)
        self.assertIsNone(metrics.mean_accepted_cost)
        self.assertIsNone(metrics.utilization_concentration)

    def test_accept_abstain_block_are_counted(self) -> None:
        cases = (
            ShadowCase("ok", task(), RouterProposal(("pade",), 0.9)),
            ShadowCase(
                "abstain",
                task(min_router_confidence=0.8),
                RouterProposal(("pade",), 0.7),
            ),
            ShadowCase("block", task(), RouterProposal(("missing",), 0.9)),
        )
        records, metrics = evaluate_shadow_cases(cases, self.experts)
        self.assertEqual([r.decision.status for r in records], [
            RouteStatus.ACCEPT,
            RouteStatus.ABSTAIN,
            RouteStatus.BLOCK,
        ])
        self.assertEqual(metrics.n_accept, 1)
        self.assertEqual(metrics.n_abstain, 1)
        self.assertEqual(metrics.n_block, 1)
        self.assertAlmostEqual(metrics.acceptance_rate, 1 / 3)
        self.assertAlmostEqual(metrics.abstention_rate, 1 / 3)
        self.assertAlmostEqual(metrics.block_rate, 1 / 3)

    def test_reference_recall_is_agreement_not_required_for_execution(self) -> None:
        cases = (
            ShadowCase(
                "full",
                task(max_experts=2),
                RouterProposal(("pade", "liouscope"), 0.9),
                frozenset({"pade", "liouscope"}),
            ),
            ShadowCase(
                "partial",
                task(),
                RouterProposal(("pade",), 0.9),
                frozenset({"pade", "liouscope"}),
            ),
        )
        records, metrics = evaluate_shadow_cases(cases, self.experts)
        self.assertEqual(records[0].reference_recall, 1.0)
        self.assertEqual(records[1].reference_recall, 0.5)
        self.assertEqual(metrics.n_reference_cases, 2)
        self.assertEqual(metrics.n_reference_accepted_cases, 2)
        self.assertAlmostEqual(metrics.reference_acceptance_rate, 1.0)
        self.assertAlmostEqual(metrics.mean_reference_recall_on_accepts, 0.75)

    def test_utilization_and_cost_are_reported(self) -> None:
        cases = (
            ShadowCase(
                "a",
                task(max_experts=2),
                RouterProposal(("pade", "liouscope"), 0.9),
            ),
            ShadowCase("b", task(), RouterProposal(("pade",), 0.9)),
        )
        _, metrics = evaluate_shadow_cases(cases, self.experts)
        self.assertEqual(metrics.utilization, {"liouscope": 1, "pade": 2})
        self.assertAlmostEqual(metrics.mean_accepted_cost, 2.0)
        self.assertAlmostEqual(metrics.utilization_concentration, 5 / 9)

    def test_blocked_routes_do_not_contribute_utilization(self) -> None:
        cases = (
            ShadowCase("bad", task(), RouterProposal(("ark",), 0.9)),
        )
        _, metrics = evaluate_shadow_cases(cases, self.experts)
        self.assertEqual(metrics.n_block, 1)
        self.assertEqual(metrics.utilization, {})
        self.assertIsNone(metrics.utilization_concentration)

    def test_reference_agreement_does_not_penalize_safe_abstention(self) -> None:
        case = ShadowCase(
            "abstain-ref",
            task(min_router_confidence=0.8),
            RouterProposal(("pade",), 0.7),
            frozenset({"pade"}),
        )
        records, metrics = evaluate_shadow_cases((case,), self.experts)
        self.assertEqual(records[0].decision.status, RouteStatus.ABSTAIN)
        self.assertIsNone(records[0].reference_recall)
        self.assertEqual(metrics.n_reference_cases, 1)
        self.assertEqual(metrics.n_reference_accepted_cases, 0)
        self.assertEqual(metrics.reference_acceptance_rate, 0.0)
        self.assertIsNone(metrics.mean_reference_recall_on_accepts)

    def test_duplicate_case_ids_rejected(self) -> None:
        case = ShadowCase("same", task(), RouterProposal(("pade",), 0.9))
        with self.assertRaises(ValueError):
            evaluate_shadow_cases((case, case), self.experts)

    def test_reference_case_can_be_blocked_without_becoming_false_success(self) -> None:
        case = ShadowCase(
            "blocked-ref",
            task(),
            RouterProposal(("missing",), 0.9),
            frozenset({"pade"}),
        )
        records, metrics = evaluate_shadow_cases((case,), self.experts)
        self.assertEqual(records[0].decision.status, RouteStatus.BLOCK)
        self.assertIsNone(records[0].reference_recall)
        self.assertEqual(metrics.n_reference_cases, 1)
        self.assertEqual(metrics.n_reference_accepted_cases, 0)
        self.assertAlmostEqual(metrics.reference_acceptance_rate, 0.0)
        self.assertIsNone(metrics.mean_reference_recall_on_accepts)


if __name__ == "__main__":
    unittest.main()
