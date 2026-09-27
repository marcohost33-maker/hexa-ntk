# HexaRouter / HCM v0.5 — deterministic MVP contract

Status: EXPERIMENTAL / INTERNAL CONTRACT CANDIDATE
Date: 2026-09-27
Scope: deterministic capability planning, audit log, durable local inbox/outbox,
and reproducible topology baselines. No learned routing is enabled in v0.5.

## 1. Purpose

This contract turns the HexaRouter/HCM concept into a small falsifiable runtime core.
The MVP deliberately separates:

1. CapabilitySpec — what a capability can do.
2. ProviderSpec — which implementation provides it.
3. RuntimeState — current health, load, latency and available resources.
4. AssuranceAttestation — externally produced, evidence-bound validation for a provider build, task class and regime.
5. ReliabilityEstimate — a computed 95% Wilson lower bound from independent Bernoulli cases for a defined provider build, task class and regime.
6. PolicyDecision — task-scoped authorization bound to policy revision, policy-bundle digest, decision id and task/provider input digest.
7. TaskContract — what a task requires.

Runtime load and assurance MUST NOT be self-declared inside the static capability spec.

Security-relevant identity tokens (provider, capability, profile, task, regime,
validator, attestation and policy identifiers) are strict lowercase ASCII tokens matching
`[a-z0-9][a-z0-9._-]{0,63}`. They are rejected rather than Unicode-normalized,
case-folded or silently trimmed. Human-readable labels belong in separate metadata.

## 2. Hard routing invariants

A provider is eligible only if all applicable gates pass:

- required functions are a subset of the capability functions;
- input and output schema identifiers are compatible;
- requested numeric dtype and unit system are supported;
- provider identity links to the registered capability;
- an explicit task-scoped PolicyDecision allows the provider and its input digest matches the complete canonical TaskContract plus provider identity;
- runtime refers to the current provider code SHA, is healthy, fresh enough and resource-feasible;
- assurance refers to the provider's current code SHA, task class and regime, carries a content digest, and reaches the task threshold;
- an independent validator is present when required;
- reliability refers to the current provider code SHA, task class and regime and carries a content digest;
- the reliability lower bound is computed from independent Bernoulli outcomes rather than accepted as a producer-declared scalar;
- the computed reliability lower bound and independent-sample floor reach the task thresholds;
- latency and queue ceilings are respected when supplied.

If the feasible set is empty the planner MUST return ABSTAIN. A ranking policy MUST
NOT override a failed hard gate.

## 3. Deterministic v0.5 ranking

The first implementation intentionally does not use a learned router and does not use
candidate-set-dependent normalization.

Eligible providers are ranked lexicographically by:

1. higher reliability lower bound;
2. lower estimated latency;
3. lower queue depth;
4. provider_id as a deterministic tie-breaker.

The policy identifier is `hexrouter.lexicographic.v1`.

The planner computes two distinct deterministic SHA-256 fingerprints:

1. a static registry fingerprint over the complete capability/provider registry;
2. an evaluation fingerprint over every registered provider's dynamic runtime,
   assurance, reliability and policy input, including missing entries.

Every decision records both fingerprints plus an eligible-candidate snapshot containing
policy revision/bundle/decision identity, validator/attestation/evidence identity,
reliability method/evidence, runtime age, latency and queue depth. Rejected providers
are therefore also bound through the evaluation fingerprint: changing a rejected
candidate's evidence can leave the route unchanged while still changing the auditable
decision identity.

The prototype hash profile is explicitly named `python-json-sortkeys-v1`. It is
deterministic for this Python reference implementation, but it is NOT claimed to be
RFC 8785 JSON Canonicalization Scheme (JCS) compatible or cross-language signature
canonicalization. A cross-language Q-Forge port must either adopt a conforming JCS
implementation or use an equally explicit canonical binary/integer representation.

This avoids rank reversal caused solely by adding an irrelevant dominated candidate
while still making a changed candidate/registry state visible in the audit identity.
A future scoring policy may replace this ordering only behind an explicit policy version
and regression tests.

## 4. Resource model

A ProviderSpec advertises one or more immutable ExecutionProfile objects. Each profile
contains the static resources and dtype required by that execution mode.

RuntimeState contains the resources currently available.

A task is resource-feasible only if one complete execution profile can satisfy the task
and the current RuntimeState can satisfy that profile. The contract therefore models
alternative execution profiles instead of ambiguous booleans such as `gpu_optional`.

## 5. Assurance and reliability

AssuranceAttestation and ReliabilityEstimate MUST bind to the exact provider code SHA,
task class and regime, and carry a lowercase SHA-256 evidence digest. Evidence from an
ID regime therefore cannot silently satisfy a geometry-OOD task.

Assurance is a separate external statement. It is not a property a provider may grant
itself. The `independent_validator` flag is still a statement supplied to the prototype,
not cryptographic proof of organizational independence; the final Q-Forge owner must
derive that status from its trust/policy registry.

PolicyDecision is separate from capability. The prototype binds each decision to:
`policy_version`, `policy_revision`, `policy_bundle_sha256`, a policy `decision_id`,
and a digest of the exact task/provider identity pair. This follows the same audit
principle as external policy decision logs: policy identity and evaluated input must be
recoverable.

Reliability is not scientific truth. For v0.5 the only admitted statistical form is a
two-sided 95% Wilson lower confidence bound over independent Bernoulli pass/fail cases.
The lower bound is recomputed from `n_success_independent / n_independent`; it is not
accepted as a producer-declared number. `n_raw` remains lineage metadata. Continuous
errors, calibrated probabilities, survival times and correlated trajectories require
separate evidence models rather than being forced into this Bernoulli contract.

## 6. Durable local decision log

The reference store uses SQLite only as a local MVP persistence layer. It is deliberately
not a network-filesystem or multi-host database contract.

A 2026 SQLite WAL-reset race materially changes the safe default: upstream documents the
bug as affecting WAL-mode databases through SQLite 3.51.2, with the fix in 3.51.3 and
backports on the 3.50 and 3.44 release lines. Therefore the reference store enables
WAL+`synchronous=FULL` only on release lines explicitly known to contain that fix.
Otherwise it falls back to rollback-journal `DELETE` + `synchronous=EXTRA`.
This trades concurrency for a smaller correctness surface, which is appropriate for the
low-throughput local control-plane MVP. The active storage profile is inspectable at
runtime and must be captured with evidence if persistence semantics are claim-relevant.

A route decision and its outbox event are written in the same SQL transaction.
The producer operation is idempotent: replaying the same decision/event is a no-op,
while reusing an identifier with different payload bytes fails closed. The decision
identity includes the eligible-candidate snapshot, so adding an eligible but dominated
provider does not change the chosen provider but does produce a distinct auditable
decision record.

The outbox serializes the decision as a CloudEvents-compatible v1.0 JSON envelope with
stable `id`, `source`, `type`, `subject`, `dataschema`, content type and `data`.
The event id is deterministic from the decision id and event-contract version. The
absence of a wall-clock `time` attribute is deliberate because CloudEvents makes it
optional and the audit identity must not acquire accidental nondeterminism.

Inbox processing is NOT marked complete when a worker first sees an event. A worker
first acquires a bounded claim lease; successful processing is recorded separately.
Expired leases may be reclaimed and a stale claim token cannot mark an event complete.
For external side effects, the downstream operation must still use the event identity as
an idempotency key because a lease cannot make a remote effect exactly-once.

Transport semantics are intentionally not called exactly-once. Outbox delivery is
at-least-once compatible and consumers remain idempotent. The v0.5 local outbox has no
multi-publisher leasing protocol; run a single publisher or accept duplicate publish
attempts until a broker-backed transport contract is introduced.

## 7. Graph separation

Three graphs MUST remain distinct:

- physics graph — mathematical or spatial coupling, e.g. six-neighbour cell-centre grid;
- workflow graph — task dependencies between tools/services/agents;
- deployment graph — processes, hosts, GPUs, brokers and network placement.

A degree-six physics grid does not imply six neighbouring compute processes after domain
partitioning.

## 8. Topology benchmark

The MVP contains dependency-free graph metrics for reproducible structural baselines.
Undirected physics-style baselines cover ring, star, full graph, and compact
six-neighbour hex-cell patches. A separate directed workflow benchmark covers a
directed cycle, feed-forward chain, bidirectional orchestrator/star, directed full
graph, and bidirectional projections of compact hex-cell patches.

Undirected metrics include edge count, average degree, density, connectivity, diameter,
mean shortest path, articulation-node count, and mean largest-component survival after
one node failure. Directed workflow metrics separately include average in/out degree,
directed density, strong connectivity, directed diameter, mean reachable shortest path
and ordered-pair reachability fraction.

These are structural diagnostics only. They do not establish that any topology is best
for task quality, latency, cost, or scientific validity. Current multi-agent research
also provides evidence that task-adaptive and moderately sparse communication graphs can
outperform fixed dense or fixed hand-crafted structures on some workloads. Therefore the
hex topology remains a benchmark candidate, not an architectural theorem. Promotion
requires task-level comparisons of quality, communication volume, wall time, recovery,
error propagation and evidence integrity against star/router, ring, dense and dynamic
sparse baselines.

## 9. Non-goals for v0.5

The following remain deliberately out of scope:

- learned routing / contextual bandits;
- dynamic topology generation;
- broker deployment;
- A2A transport;
- SPIFFE/SPIRE;
- Kubernetes/DRA;
- UCUM conversion or dimensional-analysis engine;
- RO-Crate/in-toto/SLSA emission from this prototype (Q-Forge remains the evidence owner);
- cross-process locking beyond SQLite's local database guarantees.

Those are enabled only when an observed requirement justifies them.

## 10. Acceptance criteria

The v0.5 core is acceptable only when tests demonstrate at least:

- hard-gate fail-closed behavior, including task-scoped policy;
- deterministic and candidate-invariant provider choice with registry- and candidate-set-aware audit identity;
- code-SHA-bound runtime/assurance/reliability;
- runtime freshness enforcement;
- task-class/regime-bound assurance and reliability evidence;
- recomputed 95% Wilson reliability lower bound from independent Bernoulli cases;
- policy revision/bundle/full-TaskContract input binding and fail-closed mismatch;
- decision identity changes when any evaluated registered provider evidence changes, including rejected candidates;
- strict lowercase-ASCII identity tokens with confusable/Unicode rejection;
- resource-profile feasibility;
- atomic/idempotent decision+CloudEvents-outbox recording;
- lease/reclaim/complete inbox semantics that do not confuse receipt with successful processing;
- exact undirected reference topology metrics for C6, K1,5, K6 and compact hex patches;
- exact directed workflow metrics for cycle, chain, bidirectional-star, dense and hex projections;
- no third-party runtime dependency.

