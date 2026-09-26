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
4. AssuranceAttestation — what an external validator has established.
5. ReliabilityEstimate — empirical lower-bound evidence for a defined provider build, task class and regime.
6. PolicyDecision — task-scoped authorization for a provider.
7. TaskContract — what a task requires.

Runtime load and assurance MUST NOT be self-declared inside the static capability spec.

## 2. Hard routing invariants

A provider is eligible only if all applicable gates pass:

- required functions are a subset of the capability functions;
- input and output schema identifiers are compatible;
- requested numeric dtype and unit system are supported;
- provider identity links to the registered capability;
- an explicit task-scoped PolicyDecision allows the provider;
- runtime refers to the current provider code SHA, is healthy, fresh enough and resource-feasible;
- assurance refers to the provider's current code SHA and reaches the task threshold;
- an independent validator is present when required;
- reliability refers to the current provider code SHA, task class and regime;
- the reliability lower bound and independent-sample floor reach the task thresholds;
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

The planner also computes a deterministic SHA-256 fingerprint of the complete static
capability/provider registry. Every decision records this registry fingerprint plus an
eligible-candidate snapshot containing the authorization-policy version, validator id,
validity class, reliability evidence, runtime age, latency and queue depth.

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

AssuranceAttestation and ReliabilityEstimate MUST bind to the exact provider code SHA.
ReliabilityEstimate additionally binds to a task class and regime; evidence from an ID
regime therefore cannot silently satisfy a geometry-OOD task.

Assurance is a separate external statement. It is not a property a provider may grant
itself. PolicyDecision is also separate from capability: a provider can be technically
capable while still being unauthorized for a task.

Reliability is not scientific truth. The MVP stores a lower bound plus raw and
independent sample counts. A TaskContract can require a minimum independent-sample
count. No online learning or decay constant is defined in v0.5.

## 6. Durable local decision log

The reference store uses SQLite only as a local MVP persistence layer.

A route decision and its outbox event are written in the same SQL transaction.
The producer operation is idempotent: replaying the same decision/event is a no-op,
while reusing an identifier with different payload bytes fails closed. The decision
identity includes the eligible-candidate snapshot, so adding an eligible but dominated
provider does not change the chosen provider but does produce a distinct auditable
decision record.

Inbox processing is keyed by `(consumer_id, event_id)` and is idempotent.

Transport semantics are intentionally not called exactly-once. A future broker must
assume at-least-once delivery and retain idempotent consumers.

## 7. Graph separation

Three graphs MUST remain distinct:

- physics graph — mathematical or spatial coupling, e.g. six-neighbour cell-centre grid;
- workflow graph — task dependencies between tools/services/agents;
- deployment graph — processes, hosts, GPUs, brokers and network placement.

A degree-six physics grid does not imply six neighbouring compute processes after domain
partitioning.

## 8. Topology benchmark

The MVP contains dependency-free graph metrics for reproducible structural baselines:
ring, star, full graph, and compact six-neighbour hex-cell patches.

Metrics include edge count, average degree, density, connectivity, diameter, mean
shortest path, articulation-node count, and mean largest-component survival after one
node failure.

These are structural diagnostics only. They do not establish that any topology is best
for task quality, latency, cost, or scientific validity.

## 9. Non-goals for v0.5

The following remain deliberately out of scope:

- learned routing / contextual bandits;
- dynamic topology generation;
- broker deployment;
- A2A transport;
- SPIFFE/SPIRE;
- Kubernetes/DRA;
- UCUM conversion or dimensional-analysis engine;
- RO-Crate/in-toto emission;
- cross-process locking beyond SQLite's local database guarantees.

Those are enabled only when an observed requirement justifies them.

## 10. Acceptance criteria

The v0.5 core is acceptable only when tests demonstrate at least:

- hard-gate fail-closed behavior, including task-scoped policy;
- deterministic and candidate-invariant provider choice with registry- and candidate-set-aware audit identity;
- code-SHA-bound runtime/assurance/reliability;
- runtime freshness enforcement;
- task-class/regime-bound reliability and independent-sample floors;
- resource-profile feasibility;
- atomic/idempotent decision+outbox recording;
- idempotent inbox claim;
- exact reference topology metrics for C6, K1,5, K6 and compact hex patches;
- no third-party runtime dependency.

