"""
cluster_routing_hex_v1.py
=========================

Deterministic safety spine for a future learned HexaNTK expert router.

The learned component is intentionally NOT implemented here. A learned router may
only propose an ordered set of compute experts. This module enforces hard
capability/domain constraints, separates compute from assurance experts, applies
budget and independence rules, and can abstain or block fail-closed.

This is orchestration infrastructure, not evidence of model accuracy, scientific
validity, or physics correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from math import isfinite
from typing import Iterable, Sequence


class ExpertPlane(str, Enum):
    COMPUTE = "compute"
    ASSURANCE = "assurance"


class RouteStatus(str, Enum):
    ACCEPT = "accept"
    ABSTAIN = "abstain"
    BLOCK = "block"


@dataclass(frozen=True)
class ExpertSpec:
    expert_id: str
    capabilities: frozenset[str]
    domains: frozenset[str]
    implementation_family: str
    shared_risk_groups: frozenset[str] = frozenset()
    plane: ExpertPlane = ExpertPlane.COMPUTE
    estimated_cost: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.expert_id:
            raise ValueError("expert_id must be non-empty")
        if not self.domains:
            raise ValueError("domains must be non-empty; use {'*'} for a wildcard")
        if not self.implementation_family:
            raise ValueError("implementation_family must be non-empty")
        if any(not item for item in self.shared_risk_groups):
            raise ValueError("shared_risk_groups entries must be non-empty")
        if not isfinite(self.estimated_cost) or self.estimated_cost < 0:
            raise ValueError("estimated_cost must be finite and >= 0")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    domain: str
    required_capabilities: frozenset[str]
    min_experts: int = 1
    max_experts: int = 2
    max_total_cost: float | None = None
    min_router_confidence: float = 0.0
    require_independent_compute: bool = False
    required_assurance_capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not self.domain:
            raise ValueError("domain must be non-empty")
        if self.min_experts < 1:
            raise ValueError("min_experts must be >= 1")
        if self.max_experts < self.min_experts:
            raise ValueError("max_experts must be >= min_experts")
        if self.require_independent_compute and self.min_experts < 2:
            raise ValueError(
                "require_independent_compute requires min_experts >= 2"
            )
        if not 0.0 <= self.min_router_confidence <= 1.0:
            raise ValueError("min_router_confidence must be within [0, 1]")
        if self.max_total_cost is not None:
            if not isfinite(self.max_total_cost) or self.max_total_cost < 0:
                raise ValueError("max_total_cost must be finite and >= 0")


@dataclass(frozen=True)
class RouterProposal:
    ranked_expert_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class RouteDecision:
    status: RouteStatus
    selected_compute_ids: tuple[str, ...]
    assurance_candidate_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    total_estimated_cost: float


def _domain_matches(expert: ExpertSpec, task: TaskSpec) -> bool:
    return "*" in expert.domains or task.domain in expert.domains


def _supports_compute_task(expert: ExpertSpec, task: TaskSpec) -> bool:
    return (
        expert.enabled
        and expert.plane is ExpertPlane.COMPUTE
        and _domain_matches(expert, task)
        and task.required_capabilities.issubset(expert.capabilities)
    )


def _supports_assurance_task(expert: ExpertSpec, task: TaskSpec) -> bool:
    return (
        expert.enabled
        and expert.plane is ExpertPlane.ASSURANCE
        and _domain_matches(expert, task)
        and task.required_assurance_capabilities.issubset(expert.capabilities)
    )


def _registry(experts: Iterable[ExpertSpec]) -> dict[str, ExpertSpec]:
    registry: dict[str, ExpertSpec] = {}
    for expert in experts:
        if expert.expert_id in registry:
            raise ValueError(f"duplicate expert_id in registry: {expert.expert_id}")
        registry[expert.expert_id] = expert
    return registry


def _within_budget(
    chosen: Sequence[ExpertSpec],
    candidate: ExpertSpec,
    max_total_cost: float | None,
) -> bool:
    if max_total_cost is None:
        return True
    return (
        sum(item.estimated_cost for item in chosen) + candidate.estimated_cost
        <= max_total_cost
    )


def _pair_is_independent(left: ExpertSpec, right: ExpertSpec) -> bool:
    """Conservative pairwise independence check from governance metadata."""
    return (
        left.implementation_family != right.implementation_family
        and left.shared_risk_groups.isdisjoint(right.shared_risk_groups)
    )


def _has_independent_pair(chosen: Sequence[ExpertSpec]) -> bool:
    return any(
        _pair_is_independent(left, right)
        for left, right in combinations(chosen, 2)
    )


def _subset_within_budget(
    chosen: Sequence[ExpertSpec],
    max_total_cost: float | None,
) -> bool:
    if max_total_cost is None:
        return True
    return sum(item.estimated_cost for item in chosen) <= max_total_cost


def _select_ranked(
    candidates: Sequence[ExpertSpec],
    task: TaskSpec,
) -> tuple[ExpertSpec, ...]:
    """Choose the largest feasible ranked subset without greedy budget traps.

    Candidate order remains the router preference. We search subsets by decreasing
    cardinality and, within a cardinality, lexicographic index order. This preserves
    the earliest-ranked feasible combination while avoiding the classic greedy
    failure where an expensive first choice prevents a valid cheaper combination.
    """
    max_size = min(len(candidates), task.max_experts)
    for size in range(max_size, task.min_experts - 1, -1):
        for indices in combinations(range(len(candidates)), size):
            chosen = tuple(candidates[index] for index in indices)
            if not _subset_within_budget(chosen, task.max_total_cost):
                continue
            if task.require_independent_compute and not _has_independent_pair(chosen):
                continue
            return chosen
    return ()

def adjudicate_router_proposal(
    task: TaskSpec,
    experts: Iterable[ExpertSpec],
    proposal: RouterProposal,
) -> RouteDecision:
    """Apply deterministic hard gates to a learned/heuristic router proposal."""
    registry = _registry(experts)
    reasons: list[str] = []

    try:
        confidence_valid = (
            isfinite(proposal.confidence)
            and 0.0 <= proposal.confidence <= 1.0
        )
    except (TypeError, ValueError):
        confidence_valid = False
    if not confidence_valid:
        return RouteDecision(
            RouteStatus.BLOCK, (), (), ("INVALID_ROUTER_CONFIDENCE",), 0.0
        )

    assurance_candidates = tuple(
        expert
        for expert in sorted(
            registry.values(), key=lambda item: (item.estimated_cost, item.expert_id)
        )
        if _supports_assurance_task(expert, task)
    )
    assurance_ids = tuple(item.expert_id for item in assurance_candidates)

    if task.required_assurance_capabilities and not assurance_candidates:
        return RouteDecision(
            RouteStatus.BLOCK, (), (), ("MISSING_REQUIRED_ASSURANCE",), 0.0
        )

    if proposal.confidence < task.min_router_confidence:
        return RouteDecision(
            RouteStatus.ABSTAIN,
            (),
            assurance_ids,
            ("LOW_ROUTER_CONFIDENCE",),
            0.0,
        )

    if len(set(proposal.ranked_expert_ids)) != len(proposal.ranked_expert_ids):
        return RouteDecision(
            RouteStatus.BLOCK,
            (),
            assurance_ids,
            ("DUPLICATE_ROUTER_TARGET",),
            0.0,
        )

    proposed: list[ExpertSpec] = []
    for expert_id in proposal.ranked_expert_ids:
        expert = registry.get(expert_id)
        if expert is None:
            return RouteDecision(
                RouteStatus.BLOCK, (), assurance_ids, ("UNKNOWN_EXPERT",), 0.0
            )
        if expert.plane is ExpertPlane.ASSURANCE:
            return RouteDecision(
                RouteStatus.BLOCK,
                (),
                assurance_ids,
                ("ASSURANCE_IN_COMPUTE_ROUTE",),
                0.0,
            )
        if not _supports_compute_task(expert, task):
            return RouteDecision(
                RouteStatus.BLOCK,
                (),
                assurance_ids,
                ("INELIGIBLE_COMPUTE_EXPERT",),
                0.0,
            )
        proposed.append(expert)

    selected = _select_ranked(proposed, task)
    if len(selected) < task.min_experts:
        reasons.append("INSUFFICIENT_BUDGET_OR_EXPERTS")
    if task.require_independent_compute and not _has_independent_pair(selected):
        reasons.append("NO_INDEPENDENT_COMPUTE_PAIR")

    if reasons:
        return RouteDecision(
            RouteStatus.ABSTAIN, (), assurance_ids, tuple(reasons), 0.0
        )

    total_cost = sum(item.estimated_cost for item in selected)
    return RouteDecision(
        RouteStatus.ACCEPT,
        tuple(item.expert_id for item in selected),
        assurance_ids,
        (),
        total_cost,
    )
