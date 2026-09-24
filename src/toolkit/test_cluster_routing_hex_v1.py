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
    plane: ExpertPlane = ExpertPlane.COMPUTE,
    cost: float = 1.0,
    enabled: bool = True,
) -> ExpertSpec:
    return ExpertSpec(
        expert_id=expert_id,
        capabilities=frozenset(capabilities),
        domains=frozenset(domains),
        implementation_family=family or expert_id,
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

    def test_negative_expert_cost_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            expert("bad", cost=-1.0)


if __name__ == "__main__":
    unittest.main()
