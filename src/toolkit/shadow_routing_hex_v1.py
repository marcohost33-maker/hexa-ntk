"""
shadow_routing_hex_v1.py
========================

Evaluation harness for *shadow* expert routing. Proposed routes are adjudicated by
cluster_routing_hex_v1 but never executed here. The harness measures safety outcomes,
resource/use concentration and agreement with an optional human-reviewed reference set.

Reference routes are baselines, not scientific ground truth.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable

from cluster_routing_hex_v1 import (
    ExpertSpec,
    RouteDecision,
    RouteStatus,
    RouterProposal,
    TaskSpec,
    adjudicate_router_proposal,
)


@dataclass(frozen=True)
class ShadowCase:
    case_id: str
    task: TaskSpec
    proposal: RouterProposal
    reference_compute_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")


@dataclass(frozen=True)
class ShadowRecord:
    case_id: str
    decision: RouteDecision
    reference_compute_ids: frozenset[str]
    reference_recall: float | None


@dataclass(frozen=True)
class ShadowMetrics:
    n_cases: int
    n_accept: int
    n_abstain: int
    n_block: int
    acceptance_rate: float
    abstention_rate: float
    block_rate: float
    mean_accepted_cost: float | None
    utilization: dict[str, int]
    utilization_concentration: float | None
    n_reference_cases: int
    n_reference_accepted_cases: int
    reference_acceptance_rate: float | None
    mean_reference_recall_on_accepts: float | None


def _reference_recall(
    selected: tuple[str, ...], reference: frozenset[str]
) -> float | None:
    if not reference:
        return None
    return len(set(selected) & reference) / len(reference)


def _concentration(counts: Counter[str]) -> float | None:
    """Herfindahl concentration over accepted expert selections.

    1.0 means all selected slots went to one expert; lower values indicate more
    distributed use. It is a utilization diagnostic, not a quality score.
    """
    total = sum(counts.values())
    if total == 0:
        return None
    shares = (count / total for count in counts.values())
    return sum(share * share for share in shares)


def evaluate_shadow_cases(
    cases: Iterable[ShadowCase],
    experts: Iterable[ExpertSpec],
) -> tuple[tuple[ShadowRecord, ...], ShadowMetrics]:
    """Adjudicate proposals without execution and aggregate transparent metrics."""
    expert_tuple = tuple(experts)
    case_tuple = tuple(cases)
    if len({case.case_id for case in case_tuple}) != len(case_tuple):
        raise ValueError("case_id values must be unique")

    records: list[ShadowRecord] = []
    utilization: Counter[str] = Counter()
    accepted_costs: list[float] = []
    status_counts: Counter[RouteStatus] = Counter()
    reference_recalls: list[float] = []

    for case in case_tuple:
        decision = adjudicate_router_proposal(case.task, expert_tuple, case.proposal)
        status_counts[decision.status] += 1
        if decision.status is RouteStatus.ACCEPT:
            utilization.update(decision.selected_compute_ids)
            accepted_costs.append(decision.total_estimated_cost)
        recall = None
        if case.reference_compute_ids and decision.status is RouteStatus.ACCEPT:
            recall = _reference_recall(
                decision.selected_compute_ids, case.reference_compute_ids
            )
            if recall is not None:
                reference_recalls.append(recall)
        records.append(
            ShadowRecord(
                case_id=case.case_id,
                decision=decision,
                reference_compute_ids=case.reference_compute_ids,
                reference_recall=recall,
            )
        )

    n_cases = len(case_tuple)
    n_accept = status_counts[RouteStatus.ACCEPT]
    n_abstain = status_counts[RouteStatus.ABSTAIN]
    n_block = status_counts[RouteStatus.BLOCK]
    denominator = n_cases or 1
    n_reference_cases = sum(bool(case.reference_compute_ids) for case in case_tuple)
    n_reference_accepted_cases = len(reference_recalls)
    metrics = ShadowMetrics(
        n_cases=n_cases,
        n_accept=n_accept,
        n_abstain=n_abstain,
        n_block=n_block,
        acceptance_rate=n_accept / denominator if n_cases else 0.0,
        abstention_rate=n_abstain / denominator if n_cases else 0.0,
        block_rate=n_block / denominator if n_cases else 0.0,
        mean_accepted_cost=fmean(accepted_costs) if accepted_costs else None,
        utilization=dict(sorted(utilization.items())),
        utilization_concentration=_concentration(utilization),
        n_reference_cases=n_reference_cases,
        n_reference_accepted_cases=n_reference_accepted_cases,
        reference_acceptance_rate=(
            n_reference_accepted_cases / n_reference_cases
            if n_reference_cases
            else None
        ),
        mean_reference_recall_on_accepts=(
            fmean(reference_recalls) if reference_recalls else None
        ),
    )
    return tuple(records), metrics
