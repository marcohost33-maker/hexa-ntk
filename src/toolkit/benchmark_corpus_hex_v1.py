"""
benchmark_corpus_hex_v1.py
==========================

Schema, validation and deterministic smoke baselines for the HexaFabric historical
routing corpus.

The bundled seed cases are derived metadata from historical 007acc2 tasks. They are
NOT scientific ground truth and are explicitly marked for human review before they may
be used as training labels or activation evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from itertools import combinations
import json
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping

from cluster_routing_hex_v1 import (
    ExpertPlane,
    ExpertSpec,
    RouteDecision,
    RouteStatus,
    RouterProposal,
    TaskSpec,
    adjudicate_router_proposal,
)


class SupportClass(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


class ExpectedAction(str, Enum):
    ACCEPT = "ACCEPT"
    ABSTAIN = "ABSTAIN"


_ALLOWED_ANNOTATION_STATUS = {"MODEL_REVIEWED_SEED"}


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    source_title: str
    source_date: date
    domain: str
    support_class: SupportClass
    required_capabilities: frozenset[str]
    required_assurance_capabilities: frozenset[str]
    expected_action: ExpectedAction
    acceptable_compute_sets: tuple[frozenset[str], ...]
    annotation_status: str
    requires_human_review: bool
    tags: frozenset[str]

    def __post_init__(self) -> None:
        for name in ("case_id", "source_title", "domain", "annotation_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.annotation_status not in _ALLOWED_ANNOTATION_STATUS:
            raise ValueError("unsupported annotation_status")
        if self.requires_human_review is not True:
            raise ValueError("seed corpus cases must remain human-review-gated")
        if any(not item for item in self.required_capabilities):
            raise ValueError("required_capabilities entries must be non-empty")
        if any(not item for item in self.required_assurance_capabilities):
            raise ValueError(
                "required_assurance_capabilities entries must be non-empty"
            )
        if any(
            any(not item for item in route)
            for route in self.acceptable_compute_sets
        ):
            raise ValueError("acceptable_compute_sets entries must be non-empty")
        if len(set(self.acceptable_compute_sets)) != len(self.acceptable_compute_sets):
            raise ValueError("acceptable_compute_sets must not contain duplicates")
        if any(not item for item in self.tags):
            raise ValueError("tags entries must be non-empty")
        if self.expected_action is ExpectedAction.ACCEPT:
            if self.support_class is not SupportClass.SUPPORTED:
                raise ValueError("only SUPPORTED seed cases may expect ACCEPT")
            if not self.acceptable_compute_sets:
                raise ValueError("ACCEPT cases require acceptable_compute_sets")
            if any(not route for route in self.acceptable_compute_sets):
                raise ValueError("acceptable compute routes must be non-empty")
        elif self.acceptable_compute_sets:
            raise ValueError("ABSTAIN cases must not define acceptable_compute_sets")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "BenchmarkCase":
        required = {
            "case_id",
            "source_title",
            "source_date",
            "domain",
            "support_class",
            "required_capabilities",
            "required_assurance_capabilities",
            "expected_action",
            "acceptable_compute_sets",
            "annotation_status",
            "requires_human_review",
            "tags",
        }
        unknown = set(raw) - required
        missing = required - set(raw)
        if unknown:
            raise ValueError(f"unknown benchmark case fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing benchmark case fields: {sorted(missing)}")
        try:
            source_date = date.fromisoformat(str(raw["source_date"]))
            support_class = SupportClass(str(raw["support_class"]))
            expected_action = ExpectedAction(str(raw["expected_action"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid benchmark enum/date field") from exc

        def string_set(name: str) -> frozenset[str]:
            value = raw[name]
            if not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            ):
                raise ValueError(f"{name} must be a JSON string array")
            return frozenset(value)

        def _route_sets(value: object) -> tuple[frozenset[str], ...]:
            if not isinstance(value, list):
                raise ValueError("acceptable_compute_sets must be a JSON array")
            routes: list[frozenset[str]] = []
            for route in value:
                if not isinstance(route, list) or any(
                    not isinstance(item, str) for item in route
                ):
                    raise ValueError(
                        "acceptable_compute_sets must contain string arrays"
                    )
                routes.append(frozenset(route))
            return tuple(routes)

        human_review = raw["requires_human_review"]
        if not isinstance(human_review, bool):
            raise ValueError("requires_human_review must be boolean")

        return cls(
            case_id=str(raw["case_id"]),
            source_title=str(raw["source_title"]),
            source_date=source_date,
            domain=str(raw["domain"]),
            support_class=support_class,
            required_capabilities=string_set("required_capabilities"),
            required_assurance_capabilities=string_set(
                "required_assurance_capabilities"
            ),
            expected_action=expected_action,
            acceptable_compute_sets=_route_sets(raw["acceptable_compute_sets"]),
            annotation_status=str(raw["annotation_status"]),
            requires_human_review=human_review,
            tags=string_set("tags"),
        )

    def to_task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id=self.case_id,
            domain=self.domain,
            required_capabilities=self.required_capabilities,
            min_experts=1,
            max_experts=3,
            required_assurance_capabilities=self.required_assurance_capabilities,
        )


@dataclass(frozen=True)
class CorpusMetrics:
    n_cases: int
    n_predictions: int
    matched_prediction_count: int
    missing_prediction_count: int
    unknown_prediction_count: int
    action_accuracy: float | None
    supported_accept_rate: float | None
    partial_abstention_rate: float | None
    unsupported_abstention_rate: float | None
    unsupported_overroute_rate: float | None
    mean_best_reference_recall_on_supported_accepts: float | None
    exact_reference_set_match_rate_on_supported_accepts: float | None


def load_jsonl_corpus(path: str | Path) -> tuple[BenchmarkCase, ...]:
    path = Path(path)
    cases: list[BenchmarkCase] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_no}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"benchmark line {line_no} must be a JSON object")
        cases.append(BenchmarkCase.from_mapping(raw))
    if not cases:
        raise ValueError("benchmark corpus must be non-empty")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark case_id values must be unique")
    return tuple(cases)


def _domain_matches(expert: ExpertSpec, case: BenchmarkCase) -> bool:
    return "*" in expert.domains or case.domain in expert.domains


def _minimal_capability_cover(
    case: BenchmarkCase,
    experts: Iterable[ExpertSpec],
) -> tuple[ExpertSpec, ...]:
    """Deterministic capability-only smoke baseline.

    This intentionally ignores learned scores. It prefers minimum cardinality, then
    minimum estimated cost, then stable expert-id ordering.
    """
    compute = tuple(
        sorted(
            (
                expert
                for expert in experts
                if expert.enabled
                and expert.plane is ExpertPlane.COMPUTE
                and _domain_matches(expert, case)
                and (
                    not case.required_capabilities
                    or bool(expert.capabilities & case.required_capabilities)
                )
            ),
            key=lambda expert: expert.expert_id,
        )
    )
    required = case.required_capabilities
    if not required:
        return ()

    best: tuple[ExpertSpec, ...] | None = None
    best_key: tuple[int, float, tuple[str, ...]] | None = None
    max_size = min(3, len(compute))
    for size in range(1, max_size + 1):
        for subset in combinations(compute, size):
            covered = frozenset().union(
                *(expert.capabilities for expert in subset)
            )
            if not required.issubset(covered):
                continue
            key = (
                size,
                sum(expert.estimated_cost for expert in subset),
                tuple(expert.expert_id for expert in subset),
            )
            if best_key is None or key < best_key:
                best = subset
                best_key = key
        if best is not None:
            break
    return best or ()


def capability_only_proposal(
    case: BenchmarkCase,
    experts: Iterable[ExpertSpec],
) -> RouterProposal:
    expert_tuple = tuple(experts)
    cover = _minimal_capability_cover(case, expert_tuple)
    if cover:
        return RouterProposal(
            tuple(expert.expert_id for expert in cover),
            confidence=1.0,
            requested_expert_count=len(cover),
        )

    partial = tuple(
        expert.expert_id
        for expert in sorted(expert_tuple, key=lambda item: item.expert_id)
        if expert.enabled
        and expert.plane is ExpertPlane.COMPUTE
        and _domain_matches(expert, case)
        and bool(expert.capabilities & case.required_capabilities)
    )
    return RouterProposal(partial, confidence=1.0)


def run_capability_only_baseline(
    cases: Iterable[BenchmarkCase],
    experts: Iterable[ExpertSpec],
) -> dict[str, RouteDecision]:
    expert_tuple = tuple(experts)
    return {
        case.case_id: adjudicate_router_proposal(
            case.to_task_spec(),
            expert_tuple,
            capability_only_proposal(case, expert_tuple),
        )
        for case in cases
    }


def evaluate_corpus_predictions(
    cases: Iterable[BenchmarkCase],
    predictions: Mapping[str, RouteDecision],
) -> CorpusMetrics:
    case_tuple = tuple(cases)
    action_hits: list[float] = []
    supported_hits: list[float] = []
    partial_hits: list[float] = []
    unsupported_hits: list[float] = []
    unsupported_overroutes: list[float] = []
    reference_recalls: list[float] = []
    exact_reference_matches: list[float] = []

    for case in case_tuple:
        decision = predictions.get(case.case_id)
        if decision is None:
            continue
        expected_status = (
            RouteStatus.ACCEPT
            if case.expected_action is ExpectedAction.ACCEPT
            else RouteStatus.ABSTAIN
        )
        actual_accept = decision.status is RouteStatus.ACCEPT
        action_hits.append(float(decision.status is expected_status))

        if case.support_class is SupportClass.SUPPORTED:
            supported_hits.append(float(actual_accept))
            if actual_accept and case.acceptable_compute_sets:
                selected = frozenset(decision.selected_compute_ids)
                recalls = [
                    len(selected & route) / len(route)
                    for route in case.acceptable_compute_sets
                ]
                reference_recalls.append(max(recalls))
                exact_reference_matches.append(
                    float(selected in case.acceptable_compute_sets)
                )
        elif case.support_class is SupportClass.PARTIAL:
            partial_hits.append(float(decision.status is RouteStatus.ABSTAIN))
        else:
            abstained = decision.status is RouteStatus.ABSTAIN
            unsupported_hits.append(float(abstained))
            unsupported_overroutes.append(float(actual_accept))

    def mean_or_none(values: list[float]) -> float | None:
        return fmean(values) if values else None

    case_ids = {case.case_id for case in case_tuple}
    prediction_ids = set(predictions)
    return CorpusMetrics(
        n_cases=len(case_tuple),
        n_predictions=len(predictions),
        matched_prediction_count=len(case_ids & prediction_ids),
        missing_prediction_count=len(case_ids - prediction_ids),
        unknown_prediction_count=len(prediction_ids - case_ids),
        action_accuracy=mean_or_none(action_hits),
        supported_accept_rate=mean_or_none(supported_hits),
        partial_abstention_rate=mean_or_none(partial_hits),
        unsupported_abstention_rate=mean_or_none(unsupported_hits),
        unsupported_overroute_rate=mean_or_none(unsupported_overroutes),
        mean_best_reference_recall_on_supported_accepts=mean_or_none(
            reference_recalls
        ),
        exact_reference_set_match_rate_on_supported_accepts=mean_or_none(
            exact_reference_matches
        ),
    )


def seed_reference_registry() -> tuple[ExpertSpec, ...]:
    """Experimental registry for plumbing tests, not a capability truth source."""
    return (
        ExpertSpec(
            "ade",
            frozenset({"adaptive_decision"}),
            frozenset({"*"}),
            "ade",
        ),
        ExpertSpec(
            "sciforge",
            frozenset({"experiment_design", "workflow_compile"}),
            frozenset({"*"}),
            "sciforge",
        ),
        ExpertSpec(
            "qforge",
            frozenset({"controlled_execution", "evidence_contracts"}),
            frozenset({"research_engineering", "security_engineering"}),
            "qforge",
        ),
        ExpertSpec(
            "mrt",
            frozenset({"measurement", "inverse_problem", "reconstruction"}),
            frozenset({"computer_vision", "inverse_problems"}),
            "mrt",
        ),
        ExpertSpec(
            "pade",
            frozenset({"numerical_diagnostics", "rational_approximation"}),
            frozenset({"*"}),
            "pade",
        ),
        ExpertSpec(
            "liouscope",
            frozenset({"open_system_dynamics", "spectral_diagnostics"}),
            frozenset({"open_quantum"}),
            "liouscope",
        ),
        ExpertSpec(
            "ark",
            frozenset({"audit", "provenance", "contradiction"}),
            frozenset({"*"}),
            "ark",
            plane=ExpertPlane.ASSURANCE,
        ),
        ExpertSpec(
            "usk",
            frozenset({"stability", "robustness"}),
            frozenset({"*"}),
            "usk",
            plane=ExpertPlane.ASSURANCE,
        ),
        ExpertSpec(
            "qvar",
            frozenset({"uncertainty", "variance"}),
            frozenset({"*"}),
            "qvar",
            plane=ExpertPlane.ASSURANCE,
        ),
    )
