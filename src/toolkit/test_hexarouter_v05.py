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
    PolicyDecision,
    ProviderSpec,
    ReliabilityEstimate,
    RuntimeState,
    SQLiteDecisionStore,
    TaskContract,
    bidirectional_edges,
    directed_chain_edges,
    directed_full_edges,
    directed_graph_metrics,
    directed_ring_edges,
    full_edges,
    graph_metrics,
    hex_cell_patch,
    policy_input_digest,
    reference_topology_benchmark,
    ring_edges,
    star_edges,
    workflow_topology_benchmark,
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
    code_sha: str,
    *,
    latency: float,
    age_ms: float = 10.0,
    queue: int = 0,
    ram: float = 64.0,
    vram: float = 0.0,
    healthy: bool = True,
) -> RuntimeState:
    return RuntimeState(
        provider_id=provider_id,
        code_sha=code_sha,
        healthy=healthy,
        age_ms=age_ms,
        available_cpu_cores=8,
        available_ram_gb=ram,
        available_vram_gb=vram,
        estimated_latency_ms=latency,
        queue_depth=queue,
    )


def assurance(
    provider_id: str,
    code_sha: str,
    *,
    independent: bool = True,
    task_class: str = "resolvent",
    regime: str = "default",
) -> AssuranceAttestation:
    return AssuranceAttestation(
        provider_id=provider_id,
        code_sha=code_sha,
        task_class=task_class,
        regime=regime,
        numerical_validity="verified",
        independent_validator=independent,
        validator_id="validator-a",
        attestation_id=f"attestation-{provider_id}",
        evidence_sha256="b" * 64,
    )


def reliability(
    provider_id: str,
    code_sha: str,
    success_rate: float,
    *,
    task_class: str = "resolvent",
    regime: str = "default",
    n_independent: int = 100,
) -> ReliabilityEstimate:
    successes = round(success_rate * n_independent)
    return ReliabilityEstimate(
        provider_id=provider_id,
        code_sha=code_sha,
        task_class=task_class,
        regime=regime,
        n_raw=n_independent,
        n_independent=n_independent,
        n_success_independent=successes,
        evidence_sha256="c" * 64,
    )


def policy(
    provider_id: str,
    *,
    allowed: bool = True,
    task_contract: TaskContract | None = None,
    input_sha256: str | None = None,
) -> PolicyDecision:
    bound_task = task() if task_contract is None else task_contract
    return PolicyDecision(
        provider_id=provider_id,
        task_id=bound_task.task_id,
        allowed=allowed,
        policy_version="test-policy.v1",
        policy_revision="rev-1",
        policy_bundle_sha256="a" * 64,
        decision_id=f"policy-{bound_task.task_id}-{provider_id}",
        input_sha256=(
            input_sha256
            if input_sha256 is not None
            else policy_input_digest(bound_task, provider_id)
        ),
    )


def task(**overrides: object) -> TaskContract:
    values = {
        "task_id": "task-1",
        "task_class": "resolvent",
        "regime": "default",
        "required_functions": frozenset({"RESOLVENT_ANALYSIS"}),
        "input_schema": "urn:test:matrix:v1",
        "output_schema": "urn:test:resolvent:v1",
        "dtype": "fp64",
        "unit_system": "UCUM",
        "min_numerical_validity": "verified",
        "independent_validation_required": True,
        "min_reliability_lower_bound": 0.70,
        "min_independent_samples": 20,
        "min_ram_gb": 4.0,
        "max_latency_ms": 1000.0,
        "max_queue_depth": 5,
        "max_runtime_age_ms": 5000.0,
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
            runtime={
                "a": runtime("a", "sha-a", latency=100),
                "b": runtime("b", "sha-b", latency=50),
            },
            assurance={
                "a": assurance("a", "sha-a"),
                "b": assurance("b", "sha-b"),
            },
            reliability={
                "a": reliability("a", "sha-a", 0.90),
                "b": reliability("b", "sha-b", 0.85),
            },
            policy={"a": policy("a"), "b": policy("b")},
        )
        self.assertEqual(decision.status, "ROUTE")
        self.assertEqual(decision.provider_id, "a")
        self.assertEqual(decision.policy_version, "hexrouter.lexicographic.v1")
        self.assertEqual(len(decision.registry_fingerprint), 64)
        self.assertEqual(len(decision.evaluation_fingerprint), 64)
        self.assertEqual(decision.eligible_count, 2)
        self.assertEqual(len(decision.eligible_snapshot), 2)

    def test_hard_gate_overrides_better_latency(self) -> None:
        good = provider("good", code_sha="sha-good")
        bad = provider("bad", code_sha="sha-bad")
        planner = DeterministicPlanner([CAP], [good, bad])
        decision = planner.plan(
            task(),
            runtime={
                "good": runtime("good", "sha-good", latency=100),
                "bad": runtime("bad", "sha-bad", latency=1),
            },
            assurance={
                "good": assurance("good", "sha-good"),
                "bad": assurance("bad", "sha-bad", independent=False),
            },
            reliability={
                "good": reliability("good", "sha-good", 0.9),
                "bad": reliability("bad", "sha-bad", 0.99),
            },
            policy={"good": policy("good"), "bad": policy("bad")},
        )
        self.assertEqual(decision.provider_id, "good")
        self.assertIn(
            "INDEPENDENT_ASSURANCE_REQUIRED",
            dict(decision.rejections)["bad"],
        )

    def test_stale_assurance_code_sha_fails_closed(self) -> None:
        item = provider("p", code_sha="sha-new")
        planner = DeterministicPlanner([CAP], [item])
        decision = planner.plan(
            task(),
            runtime={"p": runtime("p", "sha-new", latency=10)},
            assurance={"p": assurance("p", "sha-old")},
            reliability={"p": reliability("p", "sha-new", 0.95)},
            policy={"p": policy("p")},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn(
            "ASSURANCE_MISSING_OR_STALE",
            dict(decision.rejections)["p"],
        )

    def test_stale_runtime_code_sha_fails_closed(self) -> None:
        item = provider("p", code_sha="sha-new")
        planner = DeterministicPlanner([CAP], [item])
        decision = planner.plan(
            task(),
            runtime={"p": runtime("p", "sha-old", latency=10)},
            assurance={"p": assurance("p", "sha-new")},
            reliability={"p": reliability("p", "sha-new", 0.95)},
            policy={"p": policy("p")},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn(
            "RUNTIME_MISSING_OR_STALE",
            dict(decision.rejections)["p"],
        )

    def test_runtime_freshness_is_hard_gate(self) -> None:
        item = provider("p", code_sha="sha")
        planner = DeterministicPlanner([CAP], [item])
        contract = task(max_runtime_age_ms=100.0)
        decision = planner.plan(
            contract,
            runtime={"p": runtime("p", "sha", latency=10, age_ms=101.0)},
            assurance={"p": assurance("p", "sha")},
            reliability={"p": reliability("p", "sha", 0.95)},
            policy={"p": policy("p", task_contract=contract)},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("RUNTIME_TOO_OLD", dict(decision.rejections)["p"])

    def test_policy_is_separate_fail_closed_gate(self) -> None:
        item = provider("p", code_sha="sha")
        planner = DeterministicPlanner([CAP], [item])
        decision = planner.plan(
            task(),
            runtime={"p": runtime("p", "sha", latency=10)},
            assurance={"p": assurance("p", "sha")},
            reliability={"p": reliability("p", "sha", 0.95)},
            policy={},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("POLICY_DENY_OR_MISSING", dict(decision.rejections)["p"])

    def test_low_reliability_fails_closed(self) -> None:
        item = provider("p", code_sha="sha")
        planner = DeterministicPlanner([CAP], [item])
        contract = task(min_reliability_lower_bound=0.85)
        decision = planner.plan(
            contract,
            runtime={"p": runtime("p", "sha", latency=10)},
            assurance={"p": assurance("p", "sha")},
            reliability={"p": reliability("p", "sha", 0.89)},
            policy={"p": policy("p", task_contract=contract)},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("RELIABILITY_TOO_LOW", dict(decision.rejections)["p"])

    def test_reliability_is_bound_to_task_class_and_regime(self) -> None:
        item = provider("p", code_sha="sha")
        planner = DeterministicPlanner([CAP], [item])
        contract = task(regime="geometry-ood")
        decision = planner.plan(
            contract,
            runtime={"p": runtime("p", "sha", latency=10)},
            assurance={
                "p": assurance(
                    "p",
                    "sha",
                    regime="geometry-ood",
                )
            },
            reliability={"p": reliability("p", "sha", 0.95, regime="default")},
            policy={"p": policy("p", task_contract=contract)},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn(
            "RELIABILITY_MISSING_OR_STALE",
            dict(decision.rejections)["p"],
        )

    def test_independent_sample_floor_is_hard_gate(self) -> None:
        item = provider("p", code_sha="sha")
        planner = DeterministicPlanner([CAP], [item])
        contract = task(min_independent_samples=2)
        decision = planner.plan(
            contract,
            runtime={"p": runtime("p", "sha", latency=10)},
            assurance={"p": assurance("p", "sha")},
            reliability={
                "p": reliability(
                    "p",
                    "sha",
                    0.95,
                    n_independent=1,
                )
            },
            policy={"p": policy("p", task_contract=contract)},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn(
            "RELIABILITY_INDEPENDENCE_TOO_LOW",
            dict(decision.rejections)["p"],
        )

    def test_candidate_invariance_under_dominated_addition(self) -> None:
        p1 = provider("a", code_sha="sha-a")
        p2 = provider("b", code_sha="sha-b")
        planner_two = DeterministicPlanner([CAP], [p1, p2])
        decision_two = planner_two.plan(
            task(),
            runtime={
                "a": runtime("a", "sha-a", latency=100),
                "b": runtime("b", "sha-b", latency=200),
            },
            assurance={
                "a": assurance("a", "sha-a"),
                "b": assurance("b", "sha-b"),
            },
            reliability={
                "a": reliability("a", "sha-a", 0.90),
                "b": reliability("b", "sha-b", 0.85),
            },
            policy={"a": policy("a"), "b": policy("b")},
        )

        p3 = provider("z-dominated", code_sha="sha-z")
        planner_three = DeterministicPlanner([CAP], [p1, p2, p3])
        decision_three = planner_three.plan(
            task(),
            runtime={
                "a": runtime("a", "sha-a", latency=100),
                "b": runtime("b", "sha-b", latency=200),
                "z-dominated": runtime(
                    "z-dominated",
                    "sha-z",
                    latency=900,
                    queue=5,
                ),
            },
            assurance={
                "a": assurance("a", "sha-a"),
                "b": assurance("b", "sha-b"),
                "z-dominated": assurance("z-dominated", "sha-z"),
            },
            reliability={
                "a": reliability("a", "sha-a", 0.90),
                "b": reliability("b", "sha-b", 0.85),
                "z-dominated": reliability(
                    "z-dominated",
                    "sha-z",
                    0.80,
                ),
            },
            policy={
                "a": policy("a"),
                "b": policy("b"),
                "z-dominated": policy("z-dominated"),
            },
        )
        self.assertEqual(decision_two.provider_id, "a")
        self.assertEqual(decision_three.provider_id, "a")
        self.assertEqual(decision_two.eligible_count, 2)
        self.assertEqual(decision_three.eligible_count, 3)
        self.assertNotEqual(decision_two.decision_id, decision_three.decision_id)

    def test_resource_profile_is_hard_gate(self) -> None:
        item = provider("gpu", code_sha="sha", gpu=True, vram_gb=16.0)
        planner = DeterministicPlanner([CAP], [item])
        contract = task(require_gpu=True, min_vram_gb=16.0)
        decision = planner.plan(
            contract,
            runtime={"gpu": runtime("gpu", "sha", latency=10, vram=8.0)},
            assurance={"gpu": assurance("gpu", "sha")},
            reliability={"gpu": reliability("gpu", "sha", 0.95)},
            policy={"gpu": policy("gpu", task_contract=contract)},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn(
            "NO_RESOURCE_PROFILE",
            dict(decision.rejections)["gpu"],
        )

    def test_deterministic_tie_break(self) -> None:
        p1 = provider("a", code_sha="sha-a")
        p2 = provider("b", code_sha="sha-b")
        planner = DeterministicPlanner([CAP], [p2, p1])
        planner_reordered = DeterministicPlanner([CAP], [p1, p2])
        self.assertEqual(
            planner.registry_fingerprint,
            planner_reordered.registry_fingerprint,
        )
        decision = planner.plan(
            task(),
            runtime={
                "a": runtime("a", "sha-a", latency=10),
                "b": runtime("b", "sha-b", latency=10),
            },
            assurance={
                "a": assurance("a", "sha-a"),
                "b": assurance("b", "sha-b"),
            },
            reliability={
                "a": reliability("a", "sha-a", 0.9),
                "b": reliability("b", "sha-b", 0.9),
            },
            policy={"a": policy("a"), "b": policy("b")},
        )
        self.assertEqual(decision.provider_id, "a")


    def test_policy_input_digest_mismatch_fails_closed(self) -> None:
        item = provider("p", code_sha="sha")
        planner = DeterministicPlanner([CAP], [item])
        decision = planner.plan(
            task(),
            runtime={"p": runtime("p", "sha", latency=10)},
            assurance={"p": assurance("p", "sha")},
            reliability={"p": reliability("p", "sha", 0.95)},
            policy={"p": policy("p", input_sha256="d" * 64)},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn(
            "POLICY_INPUT_MISMATCH",
            dict(decision.rejections)["p"],
        )

    def test_policy_input_binds_complete_task_contract(self) -> None:
        item = provider("p", code_sha="sha")
        planner = DeterministicPlanner([CAP], [item])
        authorized_contract = task(max_latency_ms=1000.0)
        changed_contract = task(max_latency_ms=999.0)
        decision = planner.plan(
            changed_contract,
            runtime={"p": runtime("p", "sha", latency=10)},
            assurance={"p": assurance("p", "sha")},
            reliability={"p": reliability("p", "sha", 0.95)},
            policy={
                "p": policy(
                    "p",
                    task_contract=authorized_contract,
                )
            },
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn(
            "POLICY_INPUT_MISMATCH",
            dict(decision.rejections)["p"],
        )

    def test_rejected_evidence_changes_audit_identity(self) -> None:
        good = provider("good", code_sha="sha-good")
        bad = provider("bad", code_sha="sha-bad")
        planner = DeterministicPlanner([CAP], [good, bad])
        contract = task()

        common_runtime = {
            "good": runtime("good", "sha-good", latency=10),
            "bad": runtime("bad", "sha-bad", latency=20),
        }
        common_assurance = {
            "good": assurance("good", "sha-good"),
            "bad": assurance("bad", "sha-bad"),
        }
        common_policy = {
            "good": policy("good", task_contract=contract),
            "bad": policy("bad", task_contract=contract),
        }

        first = planner.plan(
            contract,
            runtime=common_runtime,
            assurance=common_assurance,
            reliability={
                "good": reliability("good", "sha-good", 0.95),
                "bad": reliability("bad", "sha-bad", 0.10),
            },
            policy=common_policy,
        )
        second = planner.plan(
            contract,
            runtime=common_runtime,
            assurance=common_assurance,
            reliability={
                "good": reliability("good", "sha-good", 0.95),
                "bad": reliability("bad", "sha-bad", 0.20),
            },
            policy=common_policy,
        )

        self.assertEqual(first.provider_id, "good")
        self.assertEqual(second.provider_id, "good")
        self.assertEqual(first.rejections, second.rejections)
        self.assertNotEqual(
            first.evaluation_fingerprint,
            second.evaluation_fingerprint,
        )
        self.assertNotEqual(first.decision_id, second.decision_id)

    def test_assurance_is_bound_to_task_regime(self) -> None:
        item = provider("p", code_sha="sha")
        planner = DeterministicPlanner([CAP], [item])
        contract = task(regime="geometry-ood")
        decision = planner.plan(
            contract,
            runtime={"p": runtime("p", "sha", latency=10)},
            assurance={"p": assurance("p", "sha", regime="default")},
            reliability={
                "p": reliability(
                    "p",
                    "sha",
                    0.95,
                    regime="geometry-ood",
                )
            },
            policy={"p": policy("p", task_contract=contract)},
        )
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn(
            "ASSURANCE_MISSING_OR_STALE",
            dict(decision.rejections)["p"],
        )

    def test_reliability_lower_bound_is_computed_not_declared(self) -> None:
        estimate = reliability("p", "sha", 0.95)
        self.assertLess(estimate.lower_bound, 0.95)
        self.assertGreater(estimate.lower_bound, 0.85)
        self.assertEqual(estimate.method, "wilson-95-two-sided")

    def test_reliability_rejects_successes_over_independent_trials(self) -> None:
        with self.assertRaises(ValueError):
            ReliabilityEstimate(
                provider_id="p",
                code_sha="sha",
                task_class="resolvent",
                regime="default",
                n_raw=10,
                n_independent=5,
                n_success_independent=6,
                evidence_sha256="c" * 64,
            )

    def test_policy_digest_requires_lowercase_sha256(self) -> None:
        with self.assertRaises(ValueError):
            PolicyDecision(
                provider_id="p",
                task_id="task-1",
                allowed=True,
                policy_version="v1",
                policy_revision="rev1",
                policy_bundle_sha256="not-a-digest",
                decision_id="decision-1",
                input_sha256=policy_input_digest(task(), "p"),
            )


class StoreTests(unittest.TestCase):
    def _decision(self):
        item = provider("p", code_sha="sha")
        planner = DeterministicPlanner([CAP], [item])
        return planner.plan(
            task(),
            runtime={"p": runtime("p", "sha", latency=10)},
            assurance={"p": assurance("p", "sha")},
            reliability={"p": reliability("p", "sha", 0.95)},
            policy={"p": policy("p")},
        )

    def test_decision_and_outbox_recorded_atomically_and_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "router.sqlite3"
            with SQLiteDecisionStore(path) as store:
                decision = self._decision()
                event_id_1 = store.record_decision(decision)
                event_id_2 = store.record_decision(decision)
                self.assertEqual(event_id_1, event_id_2)
                self.assertEqual(
                    store.counts(),
                    {"decisions": 1, "outbox": 1, "inbox": 0},
                )
                pending = store.pending_events()
                self.assertEqual(len(pending), 1)
                self.assertEqual(pending[0]["event_id"], event_id_1)
                envelope = __import__("json").loads(pending[0]["payload_json"])
                self.assertEqual(envelope["specversion"], "1.0")
                self.assertEqual(envelope["id"], event_id_1)
                self.assertEqual(
                    envelope["source"],
                    "urn:coworker:hexrouter:planner",
                )
                self.assertEqual(
                    envelope["type"],
                    "org.coworker.hexrouter.route.decision.v2",
                )
                self.assertEqual(envelope["subject"], "task-1")
                self.assertEqual(
                    envelope["dataschema"],
                    "urn:coworker:hexrouter:schema:route-decision:v2",
                )

    def test_outbox_publish_result_and_retry_counter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteDecisionStore(Path(tmp) / "router.sqlite3") as store:
                event_id = store.record_decision(self._decision())
                store.mark_publish_result(
                    event_id,
                    success=False,
                    error="offline",
                )
                pending = store.pending_events()
                self.assertEqual(pending[0]["attempts"], 1)
                store.mark_publish_result(event_id, success=True)
                self.assertEqual(store.pending_events(), [])

    def test_inbox_processing_uses_lease_then_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteDecisionStore(Path(tmp) / "router.sqlite3") as store:
                self.assertTrue(
                    store.begin_inbox(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-a",
                        now_ms=100,
                        lease_ms=50,
                    )
                )
                self.assertFalse(
                    store.begin_inbox(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-b",
                        now_ms=120,
                        lease_ms=50,
                    )
                )
                self.assertFalse(
                    store.is_inbox_processed("consumer-a", "event-1")
                )
                self.assertTrue(
                    store.complete_inbox(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-a",
                        now_ms=120,
                    )
                )
                self.assertTrue(
                    store.is_inbox_processed("consumer-a", "event-1")
                )
                self.assertFalse(
                    store.begin_inbox(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-c",
                        now_ms=200,
                        lease_ms=50,
                    )
                )
                self.assertEqual(store.counts()["inbox"], 1)

    def test_inbox_expired_lease_can_be_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteDecisionStore(Path(tmp) / "router.sqlite3") as store:
                self.assertTrue(
                    store.begin_inbox(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-a",
                        now_ms=100,
                        lease_ms=10,
                    )
                )
                self.assertTrue(
                    store.begin_inbox(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-b",
                        now_ms=111,
                        lease_ms=10,
                    )
                )
                with self.assertRaises(ValueError):
                    store.complete_inbox(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-a",
                        now_ms=112,
                    )
                self.assertTrue(
                    store.complete_inbox(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-b",
                        now_ms=115,
                    )
                )

    def test_inbox_expired_owner_cannot_complete_without_reclaim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteDecisionStore(Path(tmp) / "router.sqlite3") as store:
                self.assertTrue(
                    store.begin_inbox(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-a",
                        now_ms=100,
                        lease_ms=10,
                    )
                )
                with self.assertRaisesRegex(ValueError, "lease has expired"):
                    store.complete_inbox(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-a",
                        now_ms=110,
                    )
                self.assertFalse(
                    store.is_inbox_processed("consumer-a", "event-1")
                )

    def test_inbox_failed_attempt_can_release_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteDecisionStore(Path(tmp) / "router.sqlite3") as store:
                self.assertTrue(
                    store.begin_inbox(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-a",
                        now_ms=100,
                        lease_ms=1000,
                    )
                )
                self.assertTrue(
                    store.release_inbox_claim(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-a",
                    )
                )
                self.assertTrue(
                    store.begin_inbox(
                        "consumer-a",
                        "event-1",
                        claim_token="worker-b",
                        now_ms=101,
                        lease_ms=1000,
                    )
                )

    def test_storage_profile_is_durable_and_version_aware(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteDecisionStore(Path(tmp) / "router.sqlite3") as store:
                profile = store.storage_profile()
                self.assertIn(profile["journal_mode"], {"wal", "delete"})
                if profile["journal_mode"] == "wal":
                    self.assertEqual(profile["synchronous"], "FULL")
                else:
                    self.assertEqual(profile["synchronous"], "EXTRA")

    def test_publish_result_rejects_non_bool_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteDecisionStore(Path(tmp) / "router.sqlite3") as store:
                event_id = store.record_decision(self._decision())
                with self.assertRaises(ValueError):
                    store.mark_publish_result(event_id, success=1)


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
        self.assertTrue(
            math.isfinite(first["hex_cell_r2"]["mean_shortest_path"])
        )

    def test_directed_workflow_reference_metrics(self) -> None:
        cycle = directed_graph_metrics(6, directed_ring_edges(6))
        self.assertEqual(cycle["edge_count"], 6)
        self.assertAlmostEqual(cycle["average_out_degree"], 1.0)
        self.assertTrue(cycle["strongly_connected"])
        self.assertEqual(cycle["diameter"], 5)
        self.assertAlmostEqual(cycle["mean_reachable_shortest_path"], 3.0)

        chain = directed_graph_metrics(6, directed_chain_edges(6))
        self.assertEqual(chain["edge_count"], 5)
        self.assertFalse(chain["strongly_connected"])
        self.assertIsNone(chain["diameter"])
        self.assertAlmostEqual(chain["reachability_fraction"], 0.5)

        dense = directed_graph_metrics(6, directed_full_edges(6))
        self.assertEqual(dense["edge_count"], 30)
        self.assertAlmostEqual(dense["density"], 1.0)
        self.assertEqual(dense["diameter"], 1)

    def test_bidirectional_projection_doubles_undirected_edges(self) -> None:
        directed = bidirectional_edges(6, star_edges(6))
        self.assertEqual(len(directed), 10)
        metrics = directed_graph_metrics(6, directed)
        self.assertTrue(metrics["strongly_connected"])
        self.assertEqual(metrics["diameter"], 2)

    def test_workflow_reference_benchmark_is_reproducible(self) -> None:
        first = workflow_topology_benchmark()
        second = workflow_topology_benchmark()
        self.assertEqual(first, second)
        self.assertEqual(
            first["bidirectional_hex_cell_r1"]["edge_count"],
            24,
        )
        self.assertEqual(
            first["bidirectional_hex_cell_r2"]["edge_count"],
            84,
        )

    def test_topology_sizes_reject_bool(self) -> None:
        with self.assertRaises(ValueError):
            ring_edges(True)
        with self.assertRaises(ValueError):
            hex_cell_patch(False)


class IdentifierContractTests(unittest.TestCase):
    def test_confusable_or_uppercase_identity_tokens_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            provider("Provider", code_sha="sha")
        with self.assertRaises(ValueError):
            provider("p\u200b", code_sha="sha")
        with self.assertRaises(ValueError):
            PolicyDecision(
                provider_id="p",
                task_id="task-1",
                allowed=True,
                policy_version="v1",
                policy_revision="rev-1",
                policy_bundle_sha256="a" * 64,
                decision_id="decision-1",
                input_sha256=policy_input_digest(task(), "p"),
            ).__class__(
                provider_id="р",
                task_id="task-1",
                allowed=True,
                policy_version="v1",
                policy_revision="rev-1",
                policy_bundle_sha256="a" * 64,
                decision_id="decision-1",
                input_sha256=policy_input_digest(task(), "p"),
            )


class ContractValidationTests(unittest.TestCase):
    def test_reliability_independent_count_cannot_exceed_raw(self) -> None:
        with self.assertRaises(ValueError):
            ReliabilityEstimate(
                provider_id="p",
                code_sha="sha",
                task_class="resolvent",
                regime="default",
                n_raw=2,
                n_independent=3,
                n_success_independent=2,
                evidence_sha256="c" * 64,
            )

    def test_execution_profile_rejects_vram_without_gpu(self) -> None:
        with self.assertRaises(ValueError):
            ExecutionProfile(
                "bad",
                "fp64",
                cpu_cores=1,
                ram_gb=1,
                vram_gb=1,
            )

    def test_task_rejects_unknown_validity(self) -> None:
        with self.assertRaises(ValueError):
            task(min_numerical_validity="magic")

    def test_numeric_contracts_reject_text_coercion(self) -> None:
        with self.assertRaises(ValueError):
            task(min_reliability_lower_bound="0.8")


if __name__ == "__main__":
    unittest.main()
