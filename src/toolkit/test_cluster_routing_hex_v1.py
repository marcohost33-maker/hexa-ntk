"""Tests for the deterministic HexaNTK cluster-routing safety spine."""
from __future__ import annotations

import math
import unittest

from cluster_routing_hex_v1 import (
    ExpertPlane,
    ExpertSpec,
    RouteStatus,
    RouterProposal,
    TaskSpec,
    adjudicate_router_proposal,
)


def expert(
    expert_id: str,
    *,
    capabilities: tuple[str, ...] = ("solve",),
    domains: tuple[str, ...] = ("physics",),
    family: str | None = None,
    risks: tuple[str, ...] = (),
    plane: ExpertPlane = ExpertPlane.COMPUTE,
    cost: float = 1.0,
    enabled: bool = True,
) -> ExpertSpec:
    return ExpertSpec(
        expert_id=expert_id,
        capabilities=frozenset(capabilities),
        domains=frozenset(domains),
        implementation_family=family or expert_id,
        shared_risk_groups=frozenset(risks),
        plane=plane,
        estimated_cost=cost,
        enabled=enabled,
    )


def task(**kwargs: object) -> TaskSpec:
    defaults = dict(
        task_id="T-1",
        domain="physics",
        required_capabilities=frozenset({"solve"}),
    )
    defaults.update(kwargs)
    return TaskSpec(**defaults)


class TestHardGates(unittest.TestCase):
    def test_accepts_eligible_compute_proposal(self) -> None:
        decision = adjudicate_router_proposal(
            task(),
            [expert("liouscope")],
            RouterProposal(("liouscope",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ACCEPT)
        self.assertEqual(decision.selected_compute_ids, ("liouscope",))

    def test_unknown_expert_blocks(self) -> None:
        decision = adjudicate_router_proposal(
            task(),
            [expert("known")],
            RouterProposal(("missing",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.BLOCK)
        self.assertEqual(decision.reason_codes, ("UNKNOWN_EXPERT",))

    def test_duplicate_router_target_blocks(self) -> None:
        decision = adjudicate_router_proposal(
            task(),
            [expert("pade")],
            RouterProposal(("pade", "pade"), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.BLOCK)
        self.assertEqual(decision.reason_codes, ("DUPLICATE_ROUTER_TARGET",))

    def test_assurance_expert_cannot_be_learned_compute_target(self) -> None:
        ark = expert(
            "ark",
            capabilities=("audit",),
            plane=ExpertPlane.ASSURANCE,
        )
        decision = adjudicate_router_proposal(
            task(),
            [ark],
            RouterProposal(("ark",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.BLOCK)
        self.assertEqual(decision.reason_codes, ("ASSURANCE_IN_COMPUTE_ROUTE",))

    def test_capability_mismatch_blocks(self) -> None:
        decision = adjudicate_router_proposal(
            task(required_capabilities=frozenset({"inverse"})),
            [expert("pade", capabilities=("approximate",))],
            RouterProposal(("pade",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.BLOCK)
        self.assertEqual(decision.reason_codes, ("INELIGIBLE_COMPUTE_EXPERT",))

    def test_domain_mismatch_blocks(self) -> None:
        decision = adjudicate_router_proposal(
            task(domain="mrt"),
            [expert("liouscope", domains=("open_quantum",))],
            RouterProposal(("liouscope",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.BLOCK)

    def test_wildcard_domain_is_allowed(self) -> None:
        decision = adjudicate_router_proposal(
            task(domain="mrt"),
            [expert("pade", domains=("*",))],
            RouterProposal(("pade",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ACCEPT)

    def test_invalid_router_confidence_blocks(self) -> None:
        for confidence in (-0.1, 1.1, math.nan, "not-a-number"):
            with self.subTest(confidence=confidence):
                decision = adjudicate_router_proposal(
                    task(),
                    [expert("pade")],
                    RouterProposal(("pade",), confidence=confidence),
                )
                self.assertEqual(decision.status, RouteStatus.BLOCK)
                self.assertEqual(
                    decision.reason_codes, ("INVALID_ROUTER_CONFIDENCE",)
                )

    def test_required_calibration_missing_abstains(self) -> None:
        decision = adjudicate_router_proposal(
            task(require_calibrated_confidence=True),
            [expert("pade")],
            RouterProposal(("pade",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ABSTAIN)
        self.assertEqual(
            decision.reason_codes, ("UNCALIBRATED_ROUTER_SCORE",)
        )

    def test_required_calibration_id_allows_routing(self) -> None:
        decision = adjudicate_router_proposal(
            task(require_calibrated_confidence=True),
            [expert("pade")],
            RouterProposal(
                ("pade",),
                confidence=0.9,
                calibration_id="cal-2026-09-shadow-v1",
            ),
        )
        self.assertEqual(decision.status, RouteStatus.ACCEPT)

    def test_blank_calibration_id_blocks(self) -> None:
        decision = adjudicate_router_proposal(
            task(),
            [expert("pade")],
            RouterProposal(("pade",), confidence=0.9, calibration_id="   "),
        )
        self.assertEqual(decision.status, RouteStatus.BLOCK)
        self.assertEqual(
            decision.reason_codes, ("INVALID_CALIBRATION_ID",)
        )

    def test_low_confidence_abstains(self) -> None:
        decision = adjudicate_router_proposal(
            task(min_router_confidence=0.8),
            [expert("pade")],
            RouterProposal(("pade",), confidence=0.79),
        )
        self.assertEqual(decision.status, RouteStatus.ABSTAIN)
        self.assertEqual(decision.reason_codes, ("LOW_ROUTER_CONFIDENCE",))

    def test_disabled_expert_blocks(self) -> None:
        decision = adjudicate_router_proposal(
            task(),
            [expert("pade", enabled=False)],
            RouterProposal(("pade",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.BLOCK)


class TestBudgetAndIndependence(unittest.TestCase):
    def test_budget_can_reduce_variable_expert_count(self) -> None:
        decision = adjudicate_router_proposal(
            task(max_experts=3, max_total_cost=2.1),
            [
                expert("a", cost=1.0),
                expert("b", cost=1.0),
                expert("c", cost=1.0),
            ],
            RouterProposal(("a", "b", "c"), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ACCEPT)
        self.assertEqual(decision.selected_compute_ids, ("a", "b"))
        self.assertAlmostEqual(decision.total_estimated_cost, 2.0)

    def test_budget_search_avoids_greedy_false_abstention(self) -> None:
        decision = adjudicate_router_proposal(
            task(min_experts=2, max_experts=2, max_total_cost=2.0),
            [
                expert("expensive", cost=2.0),
                expert("cheap-a", cost=1.0),
                expert("cheap-b", cost=1.0),
            ],
            RouterProposal(("expensive", "cheap-a", "cheap-b"), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ACCEPT)
        self.assertEqual(decision.selected_compute_ids, ("cheap-a", "cheap-b"))
        self.assertAlmostEqual(decision.total_estimated_cost, 2.0)

    def test_explicit_router_cardinality_prevents_unneeded_extra_experts(self) -> None:
        decision = adjudicate_router_proposal(
            task(min_experts=1, max_experts=3, max_total_cost=10.0),
            [expert("a"), expert("b"), expert("c")],
            RouterProposal(
                ("a", "b", "c"),
                confidence=0.9,
                requested_expert_count=1,
            ),
        )
        self.assertEqual(decision.status, RouteStatus.ACCEPT)
        self.assertEqual(decision.selected_compute_ids, ("a",))

    def test_invalid_requested_expert_count_blocks(self) -> None:
        for count in (0, 4, True, 1.5):
            with self.subTest(count=count):
                decision = adjudicate_router_proposal(
                    task(min_experts=1, max_experts=3),
                    [expert("a"), expert("b"), expert("c")],
                    RouterProposal(
                        ("a", "b", "c"),
                        confidence=0.9,
                        requested_expert_count=count,
                    ),
                )
                self.assertEqual(decision.status, RouteStatus.BLOCK)
                self.assertEqual(
                    decision.reason_codes, ("INVALID_EXPERT_COUNT",)
                )

    def test_budget_failure_abstains(self) -> None:
        decision = adjudicate_router_proposal(
            task(max_total_cost=0.5),
            [expert("a", cost=1.0)],
            RouterProposal(("a",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ABSTAIN)
        self.assertIn("INSUFFICIENT_BUDGET_OR_EXPERTS", decision.reason_codes)

    def test_independence_search_skips_same_family_second_choice(self) -> None:
        decision = adjudicate_router_proposal(
            task(
                min_experts=2,
                max_experts=2,
                require_independent_compute=True,
            ),
            [
                expert("l1", family="liou"),
                expert("l2", family="liou"),
                expert("pade", family="rational"),
            ],
            RouterProposal(("l1", "l2", "pade"), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ACCEPT)
        self.assertEqual(decision.selected_compute_ids, ("l1", "pade"))

    def test_shared_risk_group_blocks_false_independence(self) -> None:
        decision = adjudicate_router_proposal(
            task(
                min_experts=2,
                max_experts=2,
                require_independent_compute=True,
            ),
            [
                expert(
                    "solver-a",
                    family="family-a",
                    risks=("solver-core:shared",),
                ),
                expert(
                    "solver-b",
                    family="family-b",
                    risks=("solver-core:shared",),
                ),
            ],
            RouterProposal(("solver-a", "solver-b"), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ABSTAIN)
        self.assertIn("NO_INDEPENDENT_COMPUTE_PAIR", decision.reason_codes)

    def test_missing_independent_pair_abstains(self) -> None:
        decision = adjudicate_router_proposal(
            task(
                min_experts=2,
                max_experts=2,
                require_independent_compute=True,
            ),
            [
                expert("l1", family="same"),
                expert("l2", family="same"),
            ],
            RouterProposal(("l1", "l2"), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ABSTAIN)
        self.assertIn("NO_INDEPENDENT_COMPUTE_PAIR", decision.reason_codes)


class TestAssurancePlane(unittest.TestCase):
    def test_required_assurance_missing_blocks(self) -> None:
        decision = adjudicate_router_proposal(
            task(required_assurance_capabilities=frozenset({"audit"})),
            [expert("pade")],
            RouterProposal(("pade",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.BLOCK)
        self.assertEqual(decision.reason_codes, ("MISSING_REQUIRED_ASSURANCE",))

    def test_no_assurance_requirement_returns_no_assurance_selection(self) -> None:
        ark = expert(
            "ark",
            capabilities=("audit",),
            plane=ExpertPlane.ASSURANCE,
        )
        decision = adjudicate_router_proposal(
            task(),
            [expert("pade"), ark],
            RouterProposal(("pade",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ACCEPT)
        self.assertEqual(decision.assurance_candidate_ids, ())

    def test_assurance_capabilities_can_be_composed_across_experts(self) -> None:
        ark = expert(
            "ark",
            capabilities=("audit",),
            plane=ExpertPlane.ASSURANCE,
            cost=1.0,
        )
        usk = expert(
            "usk",
            capabilities=("stability",),
            plane=ExpertPlane.ASSURANCE,
            cost=1.0,
        )
        decision = adjudicate_router_proposal(
            task(
                required_assurance_capabilities=frozenset(
                    {"audit", "stability"}
                )
            ),
            [expert("pade"), ark, usk],
            RouterProposal(("pade",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ACCEPT)
        self.assertEqual(
            decision.assurance_candidate_ids, ("ark", "usk")
        )

    def test_assurance_set_prefers_minimum_cardinality_then_cost(self) -> None:
        combined = expert(
            "combined",
            capabilities=("audit", "stability"),
            plane=ExpertPlane.ASSURANCE,
            cost=5.0,
        )
        cheap_a = expert(
            "cheap-a",
            capabilities=("audit",),
            plane=ExpertPlane.ASSURANCE,
            cost=0.1,
        )
        cheap_b = expert(
            "cheap-b",
            capabilities=("stability",),
            plane=ExpertPlane.ASSURANCE,
            cost=0.1,
        )
        decision = adjudicate_router_proposal(
            task(
                required_assurance_capabilities=frozenset(
                    {"audit", "stability"}
                )
            ),
            [expert("pade"), combined, cheap_a, cheap_b],
            RouterProposal(("pade",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ACCEPT)
        self.assertEqual(
            decision.assurance_candidate_ids, ("combined",)
        )

    def test_assurance_candidates_returned_separately(self) -> None:
        ark = expert(
            "ark",
            capabilities=("audit", "provenance"),
            plane=ExpertPlane.ASSURANCE,
            cost=0.5,
        )
        decision = adjudicate_router_proposal(
            task(required_assurance_capabilities=frozenset({"audit"})),
            [expert("pade"), ark],
            RouterProposal(("pade",), confidence=0.9),
        )
        self.assertEqual(decision.status, RouteStatus.ACCEPT)
        self.assertEqual(decision.selected_compute_ids, ("pade",))
        self.assertEqual(decision.assurance_candidate_ids, ("ark",))

    def test_duplicate_registry_id_is_configuration_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate expert_id"):
            adjudicate_router_proposal(
                task(),
                [expert("dup"), expert("dup")],
                RouterProposal(("dup",), confidence=0.9),
            )


class TestSpecValidation(unittest.TestCase):
    def test_independence_requires_at_least_two_experts(self) -> None:
        with self.assertRaises(ValueError):
            task(require_independent_compute=True, min_experts=1)

    def test_empty_shared_risk_group_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            expert("bad-risk", risks=("",))

    def test_negative_expert_cost_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            expert("bad", cost=-1.0)


if __name__ == "__main__":
    unittest.main()
