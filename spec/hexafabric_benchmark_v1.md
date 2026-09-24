# HexaFabric Benchmark v1

**Status:** benchmark protocol / seed corpus / no model-performance claim  
**Scope:** routing and coordination evaluation for HexaFabric expert cells.

## 1. Purpose

The benchmark answers a narrower question than "is HexaNTK intelligent?":

> For a task with declared domain/capability requirements, does a routing/coordination
> architecture select an admissible expert set, abstain when support is missing, and do
> so with acceptable coordination cost and error amplification?

The benchmark MUST NOT be used to infer scientific truth from route agreement alone.

## 2. Historical seed corpus

`benchmarks/hexafabric_seed_v1.jsonl` contains derived metadata from historical
`007acc2/01_AUFTRAEGE` tasks. Full private task text is not copied into this repository.

Seed labels are `MODEL_REVIEWED_SEED` and `requires_human_review=true`.

They are therefore:
- suitable for schema/loader/abstention plumbing tests;
- suitable for planning the human annotation rubric;
- NOT training ground truth;
- NOT sufficient evidence for active learned routing.

The initial seed intentionally contains:
- clearly in-scope cases;
- near-OOD / partial-support cases;
- unsupported cases where correct behavior is ABSTAIN.

## 3. Annotation rubric for promotion to reviewed corpus

A case may be promoted only after independent review records:

1. source task identity/date and immutable source reference;
2. problem domain;
3. atomic required capabilities;
4. mandatory assurance capabilities;
5. expected action: ACCEPT or ABSTAIN;
6. for ACCEPT, one or more acceptable expert sets rather than a single forced ordering;
7. rationale for exclusions and unsupported capabilities;
8. reviewer identity/family and review timestamp;
9. conflicts/disagreements preserved rather than averaged away;
10. annotation version and supersession link.

Do not label a route as "correct" merely because that project originally owned the task.

## 4. Leakage controls

The benchmark must prevent project-name memorization.

Future model inputs SHOULD contain a normalized task representation and capability registry,
not the original filename/project token when evaluating routing generalization.

Required split families:
- time split: earlier tasks train/dev, later tasks test;
- project-held-out split;
- task-family-held-out split;
- OOD split with unsupported domains;
- adversarial metadata split with renamed/shuffled expert IDs.

Near-duplicates and later revisions of the same task MUST stay in the same split group.

## 5. Baselines

All architecture comparisons must use matched total model/tool/compute budgets where possible.

Required baselines:

### B0 — Single expert
No collaboration. Establish whether coordination is needed at all.

### B1 — Capability-only
Deterministic minimum-cardinality capability cover, then cost/id tie-breaks.
The bundled implementation is a **smoke baseline**, not an accuracy claim.

### B2 — Centralized
One orchestrator may query a bounded number of experts.

### B3 — Sparse decentralized
Peer communication on a task-conditioned sparse graph.

### B4 — Six-node Hex Ring
The current research hypothesis.

### B5 — All-to-all
Only where matched communication budgets make the comparison meaningful.

A random/round-robin routing negative control SHOULD also be included once executable
expert adapters exist.

## 6. Primary routing metrics

Report separately:
- expected-action accuracy;
- supported-case accept rate;
- partial-support abstention rate;
- unsupported abstention rate;
- unsupported over-route rate;
- reference expert-set recall on accepted reviewed cases;
- exact/subset/superset agreement where multiple acceptable routes exist;
- invalid-route BLOCK rate;
- calibrated abstention/coverage metrics when router uncertainty is used.

Never collapse ACCEPT/ABSTAIN/BLOCK into one undifferentiated accuracy value.

## 7. Coordination metrics

Following current controlled multi-agent evaluation practice, record:
- end-to-end task score;
- single-expert baseline score;
- total tool/model compute;
- wall time;
- message count;
- message bytes;
- message density;
- redundant work fraction;
- coordination overhead;
- trace-level error amplification;
- failed/contradictory handoffs;
- expert utilization concentration.

Architecture comparisons without matched compute are descriptive only.

## 8. Scientific task metrics

Routing success is not enough. Each domain keeps its own scientific metrics.

Examples:
- LiouScope/open-system cases: residuals, invariants, non-normality/conditioning checks;
- PADE: approximation error, pole/residue stability, Froissart diagnostics;
- MRT: reconstruction and measurement-domain metrics;
- physical HexOperator: field error, residual/invariants, boundary/interior slices,
  resolution transfer and geometry/coefficient OOD.

## 9. Physical HexOperator benchmark track

This track is separate from expert routing.

A valid operator-learning benchmark should test:
- train/test resolution transfer;
- unseen discretizations;
- coordinates + quadrature weights;
- query at new coordinates where applicable;
- mechanistic numerical baseline;
- geometry, coefficient, forcing and boundary-condition shifts separately;
- optional D6 equivariance only on problems whose full operator/boundary data possess
  that symmetry.

A fixed-resolution array benchmark alone is not sufficient to establish neural-operator
behavior.

## 10. Statistical protocol

For stochastic systems:
- predeclare seeds where feasible;
- report per-seed distributions and tail failures, not only means;
- bootstrap or otherwise quantify uncertainty on aggregate metrics;
- keep threshold tuning on dev/calibration partitions;
- freeze the held-out test set after first use;
- record every post-test design change as a new benchmark version.

The current nine-case seed is far too small for inferential performance claims.

## 11. Calibration/OOD

Raw router score != calibrated probability.

If calibrated confidence is required, record:
- calibration artefact/version;
- calibration split;
- target coverage/error definition;
- empirical coverage on held-out in-distribution data;
- OOD detector/abstention policy separately.

Conformal methods may provide coverage guarantees under their stated exchangeability or
distributional assumptions, but they do not by themselves solve arbitrary distribution
shift.

## 12. State-of-the-art design anchors

- Berner et al., Nature Machine Intelligence (2026): neural operators should be
  discretization-agnostic and coordinate/quadrature aware.
- Capable language models can outgrow the benefits of collaboration, Nature Machine
  Intelligence (2026): matched-budget coordination benefits are task/architecture
  dependent; measure overhead, redundancy and error amplification.
- RANGE, Nature Communications (2026): multiple global relay representations can reduce
  long-range GNN bottlenecks without requiring dense all-to-all communication.
- 2026 conformal-UQ work for neural operators: separate calibration/coverage guarantees
  from raw model scores.
- Dynamic-k MoE work: expert cardinality is a routing variable, not a universal fixed k.

## 13. Promotion gates

### Seed -> Reviewed
Requires independent annotation and conflict recording.

### Reviewed -> Shadow evaluation
Requires frozen splits and baseline definitions.

### Shadow -> Bounded active routing
Requires:
- no unacceptable unsupported over-routing;
- calibrated/OOD policy meeting predeclared criteria;
- demonstrated benefit over B0/B1 on the intended task family under matched budget;
- no regression in Assurance separation;
- rollback/replay evidence.

No global threshold is asserted in v1; criteria must be set before looking at the final
test results.
