"""Deterministic HexaRouter/HCM v0.5 MVP.

The module intentionally contains no learned routing and no external runtime
dependencies. It implements hard capability gates, deterministic ranking,
a local SQLite decision/outbox/inbox store, and structural topology metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import sqlite3
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


VALIDITY_RANK = {
    "unverified": 0,
    "checked": 1,
    "verified": 2,
}


def _require_nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_int_at_least(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real numeric scalar")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return out


def _finite_unit_interval(name: str, value: float) -> float:
    out = _finite_nonnegative(name, value)
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1")
    return out


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    functions: frozenset[str]
    input_schemas: frozenset[str]
    output_schemas: frozenset[str]
    dtypes: frozenset[str]
    unit_systems: frozenset[str]

    def __post_init__(self) -> None:
        _require_nonempty("capability_id", self.capability_id)
        if not self.functions:
            raise ValueError("functions must not be empty")
        if not self.input_schemas or not self.output_schemas:
            raise ValueError("input_schemas and output_schemas must not be empty")
        if not self.dtypes:
            raise ValueError("dtypes must not be empty")
        if not self.unit_systems:
            raise ValueError("unit_systems must not be empty")


@dataclass(frozen=True)
class ExecutionProfile:
    profile_id: str
    dtype: str
    cpu_cores: int
    ram_gb: float
    gpu_required: bool = False
    vram_gb: float = 0.0

    def __post_init__(self) -> None:
        _require_nonempty("profile_id", self.profile_id)
        _require_nonempty("dtype", self.dtype)
        if isinstance(self.cpu_cores, bool) or not isinstance(self.cpu_cores, int) or self.cpu_cores < 1:
            raise ValueError("cpu_cores must be an integer >= 1")
        object.__setattr__(self, "ram_gb", _finite_nonnegative("ram_gb", self.ram_gb))
        object.__setattr__(self, "vram_gb", _finite_nonnegative("vram_gb", self.vram_gb))
        _require_bool("gpu_required", self.gpu_required)
        if not self.gpu_required and self.vram_gb != 0.0:
            raise ValueError("vram_gb must be 0 when gpu_required is false")


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    capability_id: str
    implementation_version: str
    code_sha: str
    execution_profiles: tuple[ExecutionProfile, ...]

    def __post_init__(self) -> None:
        _require_nonempty("provider_id", self.provider_id)
        _require_nonempty("capability_id", self.capability_id)
        _require_nonempty("implementation_version", self.implementation_version)
        _require_nonempty("code_sha", self.code_sha)
        if not self.execution_profiles:
            raise ValueError("execution_profiles must not be empty")
        ids = [profile.profile_id for profile in self.execution_profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("execution profile ids must be unique per provider")


@dataclass(frozen=True)
class RuntimeState:
    provider_id: str
    code_sha: str
    healthy: bool
    age_ms: float
    available_cpu_cores: int
    available_ram_gb: float
    available_vram_gb: float
    estimated_latency_ms: float
    queue_depth: int

    def __post_init__(self) -> None:
        _require_nonempty("provider_id", self.provider_id)
        _require_nonempty("code_sha", self.code_sha)
        _require_bool("healthy", self.healthy)
        _require_int_at_least("available_cpu_cores", self.available_cpu_cores, 0)
        _require_int_at_least("queue_depth", self.queue_depth, 0)
        object.__setattr__(self, "age_ms", _finite_nonnegative("age_ms", self.age_ms))
        object.__setattr__(
            self,
            "available_ram_gb",
            _finite_nonnegative("available_ram_gb", self.available_ram_gb),
        )
        object.__setattr__(
            self,
            "available_vram_gb",
            _finite_nonnegative("available_vram_gb", self.available_vram_gb),
        )
        object.__setattr__(
            self,
            "estimated_latency_ms",
            _finite_nonnegative("estimated_latency_ms", self.estimated_latency_ms),
        )


@dataclass(frozen=True)
class AssuranceAttestation:
    provider_id: str
    code_sha: str
    numerical_validity: str
    independent_validator: bool
    validator_id: str

    def __post_init__(self) -> None:
        _require_nonempty("provider_id", self.provider_id)
        _require_nonempty("code_sha", self.code_sha)
        _require_nonempty("validator_id", self.validator_id)
        _require_bool("independent_validator", self.independent_validator)
        if self.numerical_validity not in VALIDITY_RANK:
            raise ValueError(f"unknown numerical_validity: {self.numerical_validity!r}")


@dataclass(frozen=True)
class ReliabilityEstimate:
    provider_id: str
    code_sha: str
    task_class: str
    regime: str
    lower_bound: float
    n_raw: int
    n_independent: int

    def __post_init__(self) -> None:
        _require_nonempty("provider_id", self.provider_id)
        _require_nonempty("code_sha", self.code_sha)
        _require_nonempty("task_class", self.task_class)
        _require_nonempty("regime", self.regime)
        object.__setattr__(
            self,
            "lower_bound",
            _finite_unit_interval("lower_bound", self.lower_bound),
        )
        _require_int_at_least("n_raw", self.n_raw, 0)
        _require_int_at_least("n_independent", self.n_independent, 0)
        if self.n_independent > self.n_raw:
            raise ValueError("n_independent must be <= n_raw")


@dataclass(frozen=True)
class PolicyDecision:
    provider_id: str
    task_id: str
    allowed: bool
    policy_version: str

    def __post_init__(self) -> None:
        _require_nonempty("provider_id", self.provider_id)
        _require_nonempty("task_id", self.task_id)
        _require_nonempty("policy_version", self.policy_version)
        _require_bool("allowed", self.allowed)


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    task_class: str
    regime: str
    required_functions: frozenset[str]
    input_schema: str
    output_schema: str
    dtype: str
    unit_system: str
    min_numerical_validity: str = "verified"
    independent_validation_required: bool = True
    min_reliability_lower_bound: float = 0.0
    min_independent_samples: int = 0
    require_gpu: bool = False
    min_ram_gb: float = 0.0
    min_vram_gb: float = 0.0
    max_latency_ms: float | None = None
    max_queue_depth: int | None = None
    max_runtime_age_ms: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty("task_id", self.task_id)
        _require_nonempty("task_class", self.task_class)
        _require_nonempty("regime", self.regime)
        if not self.required_functions:
            raise ValueError("required_functions must not be empty")
        _require_nonempty("input_schema", self.input_schema)
        _require_nonempty("output_schema", self.output_schema)
        _require_nonempty("dtype", self.dtype)
        _require_nonempty("unit_system", self.unit_system)
        _require_bool(
            "independent_validation_required",
            self.independent_validation_required,
        )
        _require_bool("require_gpu", self.require_gpu)
        if self.min_numerical_validity not in VALIDITY_RANK:
            raise ValueError(
                f"unknown min_numerical_validity: {self.min_numerical_validity!r}"
            )
        object.__setattr__(
            self,
            "min_reliability_lower_bound",
            _finite_unit_interval(
                "min_reliability_lower_bound",
                self.min_reliability_lower_bound,
            ),
        )
        _require_int_at_least(
            "min_independent_samples",
            self.min_independent_samples,
            0,
        )
        object.__setattr__(
            self,
            "min_ram_gb",
            _finite_nonnegative("min_ram_gb", self.min_ram_gb),
        )
        object.__setattr__(
            self,
            "min_vram_gb",
            _finite_nonnegative("min_vram_gb", self.min_vram_gb),
        )
        if self.max_latency_ms is not None:
            object.__setattr__(
                self,
                "max_latency_ms",
                _finite_nonnegative("max_latency_ms", self.max_latency_ms),
            )
        if self.max_queue_depth is not None:
            _require_int_at_least("max_queue_depth", self.max_queue_depth, 0)
        if self.max_runtime_age_ms is not None:
            object.__setattr__(
                self,
                "max_runtime_age_ms",
                _finite_nonnegative(
                    "max_runtime_age_ms",
                    self.max_runtime_age_ms,
                ),
            )


@dataclass(frozen=True)
class CandidateSnapshot:
    provider_id: str
    execution_profile_id: str
    policy_version: str
    validator_id: str
    numerical_validity: str
    reliability_lower_bound: float
    reliability_n_raw: int
    reliability_n_independent: int
    runtime_age_ms: float
    estimated_latency_ms: float
    queue_depth: int


@dataclass(frozen=True)
class PlanDecision:
    decision_id: str
    task_id: str
    status: str
    provider_id: str | None
    execution_profile_id: str | None
    policy_version: str
    registry_fingerprint: str
    eligible_count: int
    eligible_snapshot: tuple[CandidateSnapshot, ...]
    rejections: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        if self.status not in {"ROUTE", "ABSTAIN"}:
            raise ValueError("status must be ROUTE or ABSTAIN")
        if self.status == "ROUTE" and (
            self.provider_id is None or self.execution_profile_id is None
        ):
            raise ValueError("ROUTE requires provider_id and execution_profile_id")
        if self.status == "ABSTAIN" and (
            self.provider_id is not None or self.execution_profile_id is not None
        ):
            raise ValueError("ABSTAIN must not select a provider")
        if self.eligible_count != len(self.eligible_snapshot):
            raise ValueError("eligible_count must match eligible_snapshot")




class DeterministicPlanner:
    """Hard-gated deterministic planner.

    Policy v1 ranks only the already-eligible set and therefore cannot make a
    failed hard gate disappear.
    """

    policy_version = "hexrouter.lexicographic.v1"

    def __init__(
        self,
        capabilities: Sequence[CapabilitySpec],
        providers: Sequence[ProviderSpec],
    ) -> None:
        self._capabilities = {item.capability_id: item for item in capabilities}
        self._providers = {item.provider_id: item for item in providers}
        if len(self._capabilities) != len(capabilities):
            raise ValueError("duplicate capability_id")
        if len(self._providers) != len(providers):
            raise ValueError("duplicate provider_id")
        unknown = sorted(
            provider.provider_id
            for provider in providers
            if provider.capability_id not in self._capabilities
        )
        if unknown:
            raise ValueError(
                f"providers reference unknown capabilities: {unknown}"
            )
        registry_payload = {
            "capabilities": [
                {
                    "capability_id": item.capability_id,
                    "functions": sorted(item.functions),
                    "input_schemas": sorted(item.input_schemas),
                    "output_schemas": sorted(item.output_schemas),
                    "dtypes": sorted(item.dtypes),
                    "unit_systems": sorted(item.unit_systems),
                }
                for item in sorted(
                    capabilities,
                    key=lambda value: value.capability_id,
                )
            ],
            "providers": [
                {
                    "provider_id": item.provider_id,
                    "capability_id": item.capability_id,
                    "implementation_version": item.implementation_version,
                    "code_sha": item.code_sha,
                    "execution_profiles": [
                        {
                            "profile_id": profile.profile_id,
                            "dtype": profile.dtype,
                            "cpu_cores": profile.cpu_cores,
                            "ram_gb": profile.ram_gb,
                            "gpu_required": profile.gpu_required,
                            "vram_gb": profile.vram_gb,
                        }
                        for profile in sorted(
                            item.execution_profiles,
                            key=lambda value: value.profile_id,
                        )
                    ],
                }
                for item in sorted(
                    providers,
                    key=lambda value: value.provider_id,
                )
            ],
        }
        self.registry_fingerprint = hashlib.sha256(
            _canonical_json(registry_payload).encode("utf-8")
        ).hexdigest()

    def plan(
        self,
        task: TaskContract,
        runtime: Mapping[str, RuntimeState],
        assurance: Mapping[str, AssuranceAttestation],
        reliability: Mapping[str, ReliabilityEstimate],
        policy: Mapping[str, PolicyDecision],
    ) -> PlanDecision:
        eligible: list[
            tuple[ProviderSpec, ExecutionProfile, RuntimeState, ReliabilityEstimate]
        ] = []
        rejected: list[tuple[str, tuple[str, ...]]] = []

        for provider_id in sorted(self._providers):
            provider = self._providers[provider_id]
            capability = self._capabilities[provider.capability_id]
            reasons: list[str] = []

            if not task.required_functions.issubset(capability.functions):
                reasons.append("CAPABILITY_MISMATCH")
            if task.input_schema not in capability.input_schemas:
                reasons.append("INPUT_SCHEMA_MISMATCH")
            if task.output_schema not in capability.output_schemas:
                reasons.append("OUTPUT_SCHEMA_MISMATCH")
            if task.dtype not in capability.dtypes:
                reasons.append("DTYPE_MISMATCH")
            if task.unit_system not in capability.unit_systems:
                reasons.append("UNIT_SYSTEM_MISMATCH")

            policy_decision = policy.get(provider_id)
            if (
                policy_decision is None
                or policy_decision.provider_id != provider_id
                or policy_decision.task_id != task.task_id
                or not policy_decision.allowed
            ):
                reasons.append("POLICY_DENY_OR_MISSING")

            state = runtime.get(provider_id)
            if (
                state is None
                or state.provider_id != provider_id
                or state.code_sha != provider.code_sha
            ):
                reasons.append("RUNTIME_MISSING_OR_STALE")
            elif not state.healthy:
                reasons.append("UNHEALTHY")
            else:
                if (
                    task.max_runtime_age_ms is not None
                    and state.age_ms > task.max_runtime_age_ms
                ):
                    reasons.append("RUNTIME_TOO_OLD")
                if (
                    task.max_latency_ms is not None
                    and state.estimated_latency_ms > task.max_latency_ms
                ):
                    reasons.append("LATENCY_LIMIT")
                if (
                    task.max_queue_depth is not None
                    and state.queue_depth > task.max_queue_depth
                ):
                    reasons.append("QUEUE_LIMIT")

            att = assurance.get(provider_id)
            if (
                att is None
                or att.provider_id != provider_id
                or att.code_sha != provider.code_sha
            ):
                reasons.append("ASSURANCE_MISSING_OR_STALE")
            else:
                if (
                    VALIDITY_RANK[att.numerical_validity]
                    < VALIDITY_RANK[task.min_numerical_validity]
                ):
                    reasons.append("ASSURANCE_TOO_WEAK")
                if (
                    task.independent_validation_required
                    and not att.independent_validator
                ):
                    reasons.append("INDEPENDENT_ASSURANCE_REQUIRED")

            rel = reliability.get(provider_id)
            if (
                rel is None
                or rel.provider_id != provider_id
                or rel.code_sha != provider.code_sha
                or rel.task_class != task.task_class
                or rel.regime != task.regime
            ):
                reasons.append("RELIABILITY_MISSING_OR_STALE")
            else:
                if rel.lower_bound < task.min_reliability_lower_bound:
                    reasons.append("RELIABILITY_TOO_LOW")
                if rel.n_independent < task.min_independent_samples:
                    reasons.append("RELIABILITY_INDEPENDENCE_TOO_LOW")

            profile = None
            if (
                state is not None
                and state.provider_id == provider_id
                and state.code_sha == provider.code_sha
                and state.healthy
            ):
                profile = self._choose_profile(task, provider, state)
                if profile is None:
                    reasons.append("NO_RESOURCE_PROFILE")
            else:
                reasons.append("NO_RESOURCE_PROFILE")

            if reasons:
                rejected.append(
                    (provider_id, tuple(sorted(set(reasons))))
                )
                continue

            assert state is not None
            assert rel is not None
            assert profile is not None
            eligible.append((provider, profile, state, rel))

        eligible_snapshot = tuple(
            sorted(
                (
                    CandidateSnapshot(
                        provider_id=row[0].provider_id,
                        execution_profile_id=row[1].profile_id,
                        policy_version=policy[row[0].provider_id].policy_version,
                        validator_id=assurance[row[0].provider_id].validator_id,
                        numerical_validity=assurance[
                            row[0].provider_id
                        ].numerical_validity,
                        reliability_lower_bound=row[3].lower_bound,
                        reliability_n_raw=row[3].n_raw,
                        reliability_n_independent=row[3].n_independent,
                        runtime_age_ms=row[2].age_ms,
                        estimated_latency_ms=row[2].estimated_latency_ms,
                        queue_depth=row[2].queue_depth,
                    )
                    for row in eligible
                ),
                key=lambda item: item.provider_id,
            )
        )

        if not eligible:
            return self._decision(
                task,
                None,
                None,
                rejected,
                eligible_snapshot,
            )

        provider, profile, _, _ = min(
            eligible,
            key=lambda row: (
                -row[3].lower_bound,
                row[2].estimated_latency_ms,
                row[2].queue_depth,
                row[0].provider_id,
                row[1].profile_id,
            ),
        )
        return self._decision(
            task,
            provider.provider_id,
            profile.profile_id,
            rejected,
            eligible_snapshot,
        )

    @staticmethod
    def _choose_profile(
        task: TaskContract,
        provider: ProviderSpec,
        state: RuntimeState,
    ) -> ExecutionProfile | None:
        feasible: list[ExecutionProfile] = []
        for profile in provider.execution_profiles:
            if profile.dtype != task.dtype:
                continue
            if task.require_gpu and not profile.gpu_required:
                continue
            required_ram = max(task.min_ram_gb, profile.ram_gb)
            required_vram = max(task.min_vram_gb, profile.vram_gb)
            if profile.cpu_cores > state.available_cpu_cores:
                continue
            if required_ram > state.available_ram_gb:
                continue
            if profile.gpu_required:
                if required_vram > state.available_vram_gb:
                    continue
            elif task.min_vram_gb > 0.0:
                continue
            feasible.append(profile)

        if not feasible:
            return None
        return min(
            feasible,
            key=lambda item: (
                item.gpu_required,
                item.ram_gb,
                item.vram_gb,
                item.cpu_cores,
                item.profile_id,
            ),
        )

    def _decision(
        self,
        task: TaskContract,
        provider_id: str | None,
        execution_profile_id: str | None,
        rejections: Sequence[tuple[str, tuple[str, ...]]],
        eligible_snapshot: tuple[
            tuple[str, str, float, int, float, int],
            ...,
        ],
    ) -> PlanDecision:
        status = "ROUTE" if provider_id is not None else "ABSTAIN"
        identity = {
            "task": _task_payload(task),
            "status": status,
            "provider_id": provider_id,
            "execution_profile_id": execution_profile_id,
            "policy_version": self.policy_version,
            "registry_fingerprint": self.registry_fingerprint,
            "eligible_snapshot": [
                _candidate_payload(item)
                for item in eligible_snapshot
            ],
            "rejections": rejections,
        }
        decision_id = hashlib.sha256(
            _canonical_json(identity).encode("utf-8")
        ).hexdigest()
        return PlanDecision(
            decision_id=decision_id,
            task_id=task.task_id,
            status=status,
            provider_id=provider_id,
            execution_profile_id=execution_profile_id,
            policy_version=self.policy_version,
            registry_fingerprint=self.registry_fingerprint,
            eligible_count=len(eligible_snapshot),
            eligible_snapshot=eligible_snapshot,
            rejections=tuple(rejections),
        )



def _task_payload(task: TaskContract) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_class": task.task_class,
        "regime": task.regime,
        "required_functions": sorted(task.required_functions),
        "input_schema": task.input_schema,
        "output_schema": task.output_schema,
        "dtype": task.dtype,
        "unit_system": task.unit_system,
        "min_numerical_validity": task.min_numerical_validity,
        "independent_validation_required": task.independent_validation_required,
        "min_reliability_lower_bound": task.min_reliability_lower_bound,
        "min_independent_samples": task.min_independent_samples,
        "require_gpu": task.require_gpu,
        "min_ram_gb": task.min_ram_gb,
        "min_vram_gb": task.min_vram_gb,
        "max_latency_ms": task.max_latency_ms,
        "max_queue_depth": task.max_queue_depth,
        "max_runtime_age_ms": task.max_runtime_age_ms,
    }


def _candidate_payload(candidate: CandidateSnapshot) -> dict[str, Any]:
    return {
        "provider_id": candidate.provider_id,
        "execution_profile_id": candidate.execution_profile_id,
        "policy_version": candidate.policy_version,
        "validator_id": candidate.validator_id,
        "numerical_validity": candidate.numerical_validity,
        "reliability_lower_bound": candidate.reliability_lower_bound,
        "reliability_n_raw": candidate.reliability_n_raw,
        "reliability_n_independent": candidate.reliability_n_independent,
        "runtime_age_ms": candidate.runtime_age_ms,
        "estimated_latency_ms": candidate.estimated_latency_ms,
        "queue_depth": candidate.queue_depth,
    }


def _decision_payload(decision: PlanDecision) -> dict[str, Any]:
    return {
        "decision_id": decision.decision_id,
        "task_id": decision.task_id,
        "status": decision.status,
        "provider_id": decision.provider_id,
        "execution_profile_id": decision.execution_profile_id,
        "policy_version": decision.policy_version,
        "registry_fingerprint": decision.registry_fingerprint,
        "eligible_count": decision.eligible_count,
        "eligible_snapshot": [
            _candidate_payload(item)
            for item in decision.eligible_snapshot
        ],
        "rejections": [
            {"provider_id": provider_id, "reasons": list(reasons)}
            for provider_id, reasons in decision.rejections
        ],
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class SQLiteDecisionStore:
    """Local durable decision/inbox/outbox store.

    Uses explicit BEGIN IMMEDIATE transactions. WAL is enabled for file-backed
    databases; synchronous=FULL favors durability over write throughput.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, isolation_level=None, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=FULL")
        if self.path != ":memory:":
            mode = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise RuntimeError(f"failed to enable WAL mode, got {mode!r}")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS outbox (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                aggregate_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
                last_error TEXT,
                published_at TEXT
            );
            CREATE TABLE IF NOT EXISTS inbox (
                consumer_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (consumer_id, event_id)
            );
            """
        )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteDecisionStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def record_decision(self, decision: PlanDecision) -> str:
        payload = _canonical_json(_decision_payload(decision))
        payload_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        event_id = hashlib.sha256(
            f"{decision.decision_id}|route.decision.v1".encode("utf-8")
        ).hexdigest()

        with self._transaction():
            self._insert_or_verify(
                table="decisions",
                key_column="decision_id",
                key_value=decision.decision_id,
                payload_sha=payload_sha,
                insert_sql=(
                    "INSERT INTO decisions "
                    "(decision_id, task_id, payload_sha256, payload_json) VALUES (?, ?, ?, ?)"
                ),
                insert_args=(decision.decision_id, decision.task_id, payload_sha, payload),
            )
            self._insert_or_verify(
                table="outbox",
                key_column="event_id",
                key_value=event_id,
                payload_sha=payload_sha,
                insert_sql=(
                    "INSERT INTO outbox "
                    "(event_id, aggregate_id, event_type, payload_sha256, payload_json) "
                    "VALUES (?, ?, ?, ?, ?)"
                ),
                insert_args=(
                    event_id,
                    decision.task_id,
                    "hexrouter.route.decision.v1",
                    payload_sha,
                    payload,
                ),
            )
        return event_id

    def _insert_or_verify(
        self,
        *,
        table: str,
        key_column: str,
        key_value: str,
        payload_sha: str,
        insert_sql: str,
        insert_args: tuple[Any, ...],
    ) -> None:
        row = self._conn.execute(
            f"SELECT payload_sha256 FROM {table} WHERE {key_column} = ?",
            (key_value,),
        ).fetchone()
        if row is None:
            self._conn.execute(insert_sql, insert_args)
            return
        if row["payload_sha256"] != payload_sha:
            raise ValueError(f"identifier collision in {table}: {key_value}")

    def pending_events(self, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be an integer >= 1")
        rows = self._conn.execute(
            """
            SELECT sequence, event_id, aggregate_id, event_type, payload_json, attempts
            FROM outbox
            WHERE published_at IS NULL
            ORDER BY sequence
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def mark_publish_result(
        self,
        event_id: str,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        _require_nonempty("event_id", event_id)
        _require_bool("success", success)
        with self._transaction():
            row = self._conn.execute(
                "SELECT published_at FROM outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise KeyError(event_id)
            if row["published_at"] is not None:
                return
            if success:
                self._conn.execute(
                    """
                    UPDATE outbox
                    SET attempts = attempts + 1, last_error = NULL,
                        published_at = CURRENT_TIMESTAMP
                    WHERE event_id = ?
                    """,
                    (event_id,),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE outbox
                    SET attempts = attempts + 1, last_error = ?
                    WHERE event_id = ?
                    """,
                    (error or "publish failed", event_id),
                )

    def claim_inbox(self, consumer_id: str, event_id: str) -> bool:
        _require_nonempty("consumer_id", consumer_id)
        _require_nonempty("event_id", event_id)
        with self._transaction():
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO inbox (consumer_id, event_id) VALUES (?, ?)",
                (consumer_id, event_id),
            )
            return cursor.rowcount == 1

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for table in ("decisions", "outbox", "inbox"):
            out[table] = int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        return out


Edge = tuple[int, int]


def _canonical_edges(edges: Iterable[Edge], n_nodes: int) -> tuple[Edge, ...]:
    _require_int_at_least("n_nodes", n_nodes, 1)
    out: set[Edge] = set()
    for left, right in edges:
        if not (0 <= left < n_nodes and 0 <= right < n_nodes):
            raise ValueError("edge endpoint outside graph")
        if left == right:
            raise ValueError("self loops are not part of the topology benchmark")
        out.add((left, right) if left < right else (right, left))
    return tuple(sorted(out))


def ring_edges(n_nodes: int) -> tuple[Edge, ...]:
    _require_int_at_least("n_nodes", n_nodes, 1)
    if n_nodes == 1:
        return ()
    if n_nodes == 2:
        return ((0, 1),)
    return _canonical_edges(((idx, (idx + 1) % n_nodes) for idx in range(n_nodes)), n_nodes)


def star_edges(n_nodes: int) -> tuple[Edge, ...]:
    _require_int_at_least("n_nodes", n_nodes, 1)
    return tuple((0, idx) for idx in range(1, n_nodes))


def full_edges(n_nodes: int) -> tuple[Edge, ...]:
    _require_int_at_least("n_nodes", n_nodes, 1)
    return tuple(
        (left, right)
        for left in range(n_nodes)
        for right in range(left + 1, n_nodes)
    )


def hex_cell_patch(radius: int) -> tuple[tuple[tuple[int, int], ...], tuple[Edge, ...]]:
    """Return axial coordinates and six-neighbour edges for a compact hex patch."""

    _require_int_at_least("radius", radius, 0)
    coords = sorted(
        (q, r)
        for q in range(-radius, radius + 1)
        for r in range(-radius, radius + 1)
        if max(abs(q), abs(r), abs(q + r)) <= radius
    )
    index = {coord: idx for idx, coord in enumerate(coords)}
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
    edges: set[Edge] = set()
    for coord, left in index.items():
        q, r = coord
        for dq, dr in directions:
            other = (q + dq, r + dr)
            right = index.get(other)
            if right is not None and left != right:
                edges.add((left, right) if left < right else (right, left))
    return tuple(coords), tuple(sorted(edges))


def graph_metrics(n_nodes: int, edges: Iterable[Edge]) -> dict[str, Any]:
    canonical = _canonical_edges(edges, n_nodes)
    adjacency = [set() for _ in range(n_nodes)]
    for left, right in canonical:
        adjacency[left].add(right)
        adjacency[right].add(left)

    all_distances: list[int] = []
    connected = True
    diameter = 0
    for source in range(n_nodes):
        distances = _bfs_distances(adjacency, source)
        if len(distances) != n_nodes:
            connected = False
            continue
        diameter = max(diameter, max(distances.values(), default=0))
        all_distances.extend(
            distance for target, distance in distances.items() if target > source
        )

    articulation = 0
    survival_values: list[float] = []
    if n_nodes == 1:
        survival_values.append(1.0)
    else:
        for removed in range(n_nodes):
            components = _component_sizes(adjacency, removed)
            largest = max(components, default=0)
            survival_values.append(largest / (n_nodes - 1))
            if len(components) > 1:
                articulation += 1

    edge_count = len(canonical)
    pair_count = n_nodes * (n_nodes - 1) / 2
    return {
        "n_nodes": n_nodes,
        "edge_count": edge_count,
        "average_degree": 2.0 * edge_count / n_nodes,
        "density": (edge_count / pair_count) if pair_count else 0.0,
        "connected": connected,
        "diameter": diameter if connected else None,
        "mean_shortest_path": (
            sum(all_distances) / len(all_distances)
            if connected and all_distances
            else (0.0 if n_nodes == 1 else None)
        ),
        "articulation_nodes": articulation,
        "mean_single_node_survival": sum(survival_values) / len(survival_values),
    }


def _bfs_distances(adjacency: Sequence[set[int]], source: int, removed: int | None = None) -> dict[int, int]:
    if source == removed:
        return {}
    distances = {source: 0}
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour == removed or neighbour in distances:
                continue
            distances[neighbour] = distances[node] + 1
            queue.append(neighbour)
    return distances


def _component_sizes(adjacency: Sequence[set[int]], removed: int) -> list[int]:
    unseen = set(range(len(adjacency)))
    unseen.discard(removed)
    sizes: list[int] = []
    while unseen:
        source = min(unseen)
        component = set(_bfs_distances(adjacency, source, removed))
        sizes.append(len(component))
        unseen.difference_update(component)
    return sizes


def reference_topology_benchmark() -> dict[str, dict[str, Any]]:
    coords_r1, edges_r1 = hex_cell_patch(1)
    coords_r2, edges_r2 = hex_cell_patch(2)
    return {
        "ring6": graph_metrics(6, ring_edges(6)),
        "star6": graph_metrics(6, star_edges(6)),
        "full6": graph_metrics(6, full_edges(6)),
        "hex_cell_r1": graph_metrics(len(coords_r1), edges_r1),
        "hex_cell_r2": graph_metrics(len(coords_r2), edges_r2),
    }
