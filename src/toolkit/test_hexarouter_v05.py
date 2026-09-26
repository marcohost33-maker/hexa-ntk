"""Contract tests for HexaRouter/HCM v0.5 MVP."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from hexarouter_v05 import (
    AssuranceAttestation,
    CapabilitySpec,
    DeterministicPlanner,
    ExecutionProfile,
    ProviderSpec,
    ReliabilityEstimate,
    RuntimeState,
    SQLiteDecisionStore,
    TaskContract,
    full_edges,
    graph_metrics,
    hex_cell_patch,
    reference_topology_benchmark,
    ring_edges,
    star_edges,
)


CAP = CapabilitySpec(
    capability_id="science.resolvent.v1",
    functions=frozenset({"NONNORMALITY_ANALYSIS", "RESOLVENT_ANALYSIS"}),
    input_schemas=frozenset({"urn:test:matrix:v1"}),
    output_schemas=frozenset({"urn:test:resolvent:v1"}),
    dtypes=frozenset({"fp64"}),
    unit_systems=frozenset({"UCUM"}),
)


def provider(
    provider_id: str,
    *,
    code_sha: str,
    ram_gb: float = 8.0,
    gpu: bool = False,
    vram_gb: float = 0.0,
) -> ProviderSpec:
    return ProviderSpec(
        provider_id=provider_id,
        capability_id=CAP.capability_id,
        implementation_version="1.0.0",
        code_sha=code_sha,
        execution_profiles=(
            ExecutionProfile(
                profile_id="default",
                dtype="fp64",
                cpu_cores=2,
                ram_gb=ram_gb,
                gpu_required=gpu,
                vram_gb=vram_gb,
            ),
        ),
    )


def runtime(
    provider_id: str,
    *,
    latency: float,
    queue: int = 0,
    ram: float = 64.0,
    vram: float = 0.0,
    healthy: bool = True,
) -> RuntimeState:
    return RuntimeState(
        provider_id=provider_id,
        healthy=healthy,
        available_cpu_cores=8,
        available_ram_gb=ram,
        available_vram_gb=vram,
        estimated_latency_ms=latency,
        queue_depth=queue,
    )


def assurance(provider_id: str, code_sha: str, *, independent: bool = True) -> AssuranceAttestation:
    return AssuranceAttestation(
        provider_id=provider_id,
        code_sha=code_sha,
        numerical_validity="verified",
        independent_validator=independent,
        validator_id="validator-A",
    )


def reliability(provider_id: str, code_sha: str, lower: float) -> ReliabilityEstimate:
    return ReliabilityEstimate(
        provider_id=provider_id,
        code_sha=code_sha,
        lower_bound=lower,
        n_raw=20,
        n_independent=5,
    )


def task(**overrides: object) -> TaskContract:
    values = {
        "task_id": "task-1",
        "required_functions": frozenset({"RESOLVENT_ANALYSIS"}),
        "input_schema": "urn:test:matrix:v1",
        "output_schema": "urn:test:resolvent:v1",
        "dtype": "fp64",
        "unit_system": "UCUM",
        "min_numerical_validity": "verified",
        "independent_validation_required": True,
        "min_reliability_lower_bound": 0.8,
        "min_ram_gb": 4.0,
        "max_latency_ms": 1000.0,
        "max_queue_depth": 5,
    }
    values.update(overrides)
    return TaskContract(**values)


class PlannerTests(unittest.TestCase):
    def test_routes_best_eligible_provider(self) -> None:
        p1 = provider("a", code_sha="sha-a")
        p2 = provider("b", code_sha="sha-b")
        planner = DeterministicPlanner([CAP], [p1, p2])
        decision = planner.plan(
            task(),
            runtime={"a": runtime("a", latency=100), "b": runtime("b", latency=50)},
            assurance={
                "a": assurance("a", "sha-a"),
                "b": assurance("b", "sha-b"),
            },
            reliability={
                "a": reliability("a", "sha-a", 0.90),
                "b": reliability("b", "sha-b", 0.85),
            },
        )
        self.assertEqual(decision.status, "ROUTE")
        self.assertEqual(decision.provider_id, "a")
        self.assertEqual(decision.policy_version, "hexrouter.lexicographic.v1")
        self.assertEqual(decision.eligible_count, 2)

    def test_hard_gate_overrides_better_latency(self) -> None:
        good = provider("good", code_sha="sha-good")
        bad = provider("bad", code_sha="sha-bad")
        planner = DeterministicPlanner([CAP], [good, bad])
        decision = planner.plan(
            task(),
            runtime={
                "good": runtime("good", latency=100),
                "bad": runtime("bad", latency=1),
            },
            assurance={
                "good": assurance("good", "sha-good"),
                "bad": assurance("bad", "sha-bad", independent=False),
            },
            reliability={
                "good": reliability("good", "sha-good", 0.9),
                "bad": reliability("bad", "sha-bad", 0.99),
            },
        )
        self.assertEqual(decision.provider_id, "good")
        rejected = dict(decision.rejections)
        self.assertIn("INDEPENDENT_ASSURANCE_REQUIRED", rejected["bad"])

    def test_stale_code_sha_fails_closed(self) -> None:
        item = provider("p", code_sha="sha-new")
        planner = DeterministicPlanner([CAP], [item])
        decision = planner.plan(
            task(),
            runtime={"p": runtime("p", latency=10)},
            assurance={"p": assurance("p", "sha-old")},
            reliability={"p": reliability("p", "sha-new", 0.95)},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("ASSURANCE_MISSING_OR_STALE", dict(decision.rejections)["p"])

    def test_low_reliability_fails_closed(self) -> None:
        item = provider("p", code_sha="sha")
        planner = DeterministicPlanner([CAP], [item])
        decision = planner.plan(
            task(min_reliability_lower_bound=0.9),
            runtime={"p": runtime("p", latency=10)},
            assurance={"p": assurance("p", "sha")},
            reliability={"p": reliability("p", "sha", 0.89)},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("RELIABILITY_TOO_LOW", dict(decision.rejections)["p"])

    def test_candidate_invariance_under_dominated_addition(self) -> None:
        p1 = provider("a", code_sha="sha-a")
        p2 = provider("b", code_sha="sha-b")
        planner_two = DeterministicPlanner([CAP], [p1, p2])
        decision_two = planner_two.plan(
            task(),
            runtime={"a": runtime("a", latency=100), "b": runtime("b", latency=200)},
            assurance={"a": assurance("a", "sha-a"), "b": assurance("b", "sha-b")},
            reliability={
                "a": reliability("a", "sha-a", 0.90),
                "b": reliability("b", "sha-b", 0.85),
            },
        )

        p3 = provider("z-dominated", code_sha="sha-z")
        planner_three = DeterministicPlanner([CAP], [p1, p2, p3])
        decision_three = planner_three.plan(
            task(),
            runtime={
                "a": runtime("a", latency=100),
                "b": runtime("b", latency=200),
                "z-dominated": runtime("z-dominated", latency=900, queue=5),
            },
            assurance={
                "a": assurance("a", "sha-a"),
                "b": assurance("b", "sha-b"),
                "z-dominated": assurance("z-dominated", "sha-z"),
            },
            reliability={
                "a": reliability("a", "sha-a", 0.90),
                "b": reliability("b", "sha-b", 0.85),
                "z-dominated": reliability("z-dominated", "sha-z", 0.80),
            },
        )
        self.assertEqual(decision_two.provider_id, "a")
        self.assertEqual(decision_three.provider_id, "a")

    def test_resource_profile_is_hard_gate(self) -> None:
        item = provider("gpu", code_sha="sha", gpu=True, vram_gb=16.0)
        planner = DeterministicPlanner([CAP], [item])
        decision = planner.plan(
            task(require_gpu=True, min_vram_gb=16.0),
            runtime={"gpu": runtime("gpu", latency=10, vram=8.0)},
            assurance={"gpu": assurance("gpu", "sha")},
            reliability={"gpu": reliability("gpu", "sha", 0.95)},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("NO_RESOURCE_PROFILE", dict(decision.rejections)["gpu"])

    def test_deterministic_tie_break(self) -> None:
        p1 = provider("a", code_sha="sha-a")
        p2 = provider("b", code_sha="sha-b")
        planner = DeterministicPlanner([CAP], [p2, p1])
        decision = planner.plan(
            task(),
            runtime={"a": runtime("a", latency=10), "b": runtime("b", latency=10)},
            assurance={"a": assurance("a", "sha-a"), "b": assurance("b", "sha-b")},
            reliability={
                "a": reliability("a", "sha-a", 0.9),
                "b": reliability("b", "sha-b", 0.9),
            },
        )
        self.assertEqual(decision.provider_id, "a")


class StoreTests(unittest.TestCase):
    def _decision(self):
        item = provider("p", code_sha="sha")
        planner = DeterministicPlanner([CAP], [item])
        return planner.plan(
            task(),
            runtime={"p": runtime("p", latency=10)},
            assurance={"p": assurance("p", "sha")},
            reliability={"p": reliability("p", "sha", 0.95)},
        )

    def test_decision_and_outbox_recorded_atomically_and_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "router.sqlite3"
            with SQLiteDecisionStore(path) as store:
                decision = self._decision()
                event_id_1 = store.record_decision(decision)
                event_id_2 = store.record_decision(decision)
                self.assertEqual(event_id_1, event_id_2)
                self.assertEqual(store.counts(), {"decisions": 1, "outbox": 1, "inbox": 0})
                pending = store.pending_events()
                self.assertEqual(len(pending), 1)
                self.assertEqual(pending[0]["event_id"], event_id_1)

    def test_outbox_publish_result_and_retry_counter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteDecisionStore(Path(tmp) / "router.sqlite3") as store:
                event_id = store.record_decision(self._decision())
                store.mark_publish_result(event_id, success=False, error="offline")
                pending = store.pending_events()
                self.assertEqual(pending[0]["attempts"], 1)
                store.mark_publish_result(event_id, success=True)
                self.assertEqual(store.pending_events(), [])

    def test_inbox_claim_is_idempotent_per_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteDecisionStore(Path(tmp) / "router.sqlite3") as store:
                self.assertTrue(store.claim_inbox("consumer-a", "event-1"))
                self.assertFalse(store.claim_inbox("consumer-a", "event-1"))
                self.assertTrue(store.claim_inbox("consumer-b", "event-1"))
                self.assertEqual(store.counts()["inbox"], 2)


class TopologyTests(unittest.TestCase):
    def test_ring6_metrics(self) -> None:
        result = graph_metrics(6, ring_edges(6))
        self.assertEqual(result["edge_count"], 6)
        self.assertAlmostEqual(result["average_degree"], 2.0)
        self.assertEqual(result["diameter"], 3)
        self.assertEqual(result["articulation_nodes"], 0)
        self.assertAlmostEqual(result["mean_single_node_survival"], 1.0)

    def test_star6_metrics(self) -> None:
        result = graph_metrics(6, star_edges(6))
        self.assertEqual(result["edge_count"], 5)
        self.assertAlmostEqual(result["average_degree"], 10 / 6)
        self.assertEqual(result["diameter"], 2)
        self.assertEqual(result["articulation_nodes"], 1)
        self.assertLess(result["mean_single_node_survival"], 1.0)

    def test_full6_metrics(self) -> None:
        result = graph_metrics(6, full_edges(6))
        self.assertEqual(result["edge_count"], 15)
        self.assertAlmostEqual(result["average_degree"], 5.0)
        self.assertEqual(result["diameter"], 1)
        self.assertEqual(result["articulation_nodes"], 0)
        self.assertAlmostEqual(result["mean_shortest_path"], 1.0)

    def test_hex_patch_reference_counts(self) -> None:
        coords1, edges1 = hex_cell_patch(1)
        coords2, edges2 = hex_cell_patch(2)
        self.assertEqual((len(coords1), len(edges1)), (7, 12))
        self.assertEqual((len(coords2), len(edges2)), (19, 42))
        self.assertEqual(graph_metrics(7, edges1)["diameter"], 2)
        self.assertEqual(graph_metrics(19, edges2)["diameter"], 4)

    def test_reference_benchmark_is_reproducible(self) -> None:
        first = reference_topology_benchmark()
        second = reference_topology_benchmark()
        self.assertEqual(first, second)
        self.assertTrue(first["hex_cell_r1"]["connected"])
        self.assertTrue(math.isfinite(first["hex_cell_r2"]["mean_shortest_path"]))


class ContractValidationTests(unittest.TestCase):
    def test_reliability_independent_count_cannot_exceed_raw(self) -> None:
        with self.assertRaises(ValueError):
            ReliabilityEstimate("p", "sha", 0.9, n_raw=2, n_independent=3)

    def test_execution_profile_rejects_vram_without_gpu(self) -> None:
        with self.assertRaises(ValueError):
            ExecutionProfile("bad", "fp64", cpu_cores=1, ram_gb=1, vram_gb=1)

    def test_task_rejects_unknown_validity(self) -> None:
        with self.assertRaises(ValueError):
            task(min_numerical_validity="magic")


if __name__ == "__main__":
    unittest.main()
