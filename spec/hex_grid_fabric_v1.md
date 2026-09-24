
# Hex Grid Fabric v1 — topology, coupling and information flow

Status: research contract / deterministic reference layer  
Scope: logical expert-cell topology plus physical-hex geometry primitives. No learned
router, no GINO/GINOT/GAOT implementation and no scientific-validity claim.

## 1. Two hexagonal meanings must remain separate

### Physical hex-centre point lattice
For field/PDE/operator learning, axial coordinates (q,r) represent the regular point
lattice formed by hex-cell centres, equivalently a triangular Bravais lattice, with
implicit cube coordinate s=-q-r. Every interior site has six equidistant nearest
neighbours. This must not be confused with the degree-3 honeycomb vertex graph.

The canonical Cartesian embedding used by the toolkit is

    x = h (q + r/2)
    y = h (sqrt(3)/2) r

where h is the nearest-neighbour centre spacing. This is the same spacing convention used
by the existing 7-point stencil
    Delta_h u0 = 2/(3 h^2) sum_j (u_j - u0).

A future graph neural operator must treat coordinates and quadrature/area weights as part
of the operator discretization contract; index-only message passing is not sufficient for
mesh/discretization consistency.

### Six-project Expert Cell
A six-project expert cell places six compute experts on the radius-1 ring around the
origin. The origin is reserved for a virtual relay/control plane and is NOT a seventh
expert. Therefore each ring expert has two direct project neighbours, not six.

This distinction prevents a common category error: the expert graph is an orchestration
topology, while a physical hex grid is a spatial discretization.

## 2. Reference compute ring

A useful first profile is:

E   ADE
NE  SciForge
NW  Q-Forge
W   MRT
SW  PADE
SE  LiouScope

Clockwise semantic flow:

ADE -> SciForge -> Q-Forge -> MRT -> PADE -> LiouScope -> ADE

Interpretation:
- ADE: adaptive decision / candidate selection.
- SciForge: experiment and workflow compilation.
- Q-Forge: controlled execution/orchestration and run contracts.
- MRT: measurement, reconstruction and inverse-problem viewpoint.
- PADE: rational approximation, pole/residue and approximation diagnostics.
- LiouScope: dynamical/open-system/non-normality diagnostics.

This is a pilot data-flow layout, not a claim that these projects share one latent state
space. Node placement is configuration and must be ablated against alternative layouts.

ARK, USK and QVAR belong to the Assurance Plane and are deliberately external to this
compute ring.

## 3. Four information lanes

### LOCAL_STATE
- project-to-project only;
- exactly one semantic hop to a directly adjacent project;
- multi-hop influence requires explicit sequential update rounds, never transparent tunnelling;
- high-frequency, bounded delta/features;
- may mutate compute state;
- must have hop/size budgets before production use.

### GLOBAL_SUMMARY
- project -> virtual relay -> project;
- summary/embedding/metadata only;
- must not silently replace local raw state;
- intended for long-range context without deep local message passing.

### CONTROL
- separate virtual control endpoint;
- schema/task/budget/cancellation messages only;
- no scientific payload;
- control must not be inferred from evidence content.

### EVIDENCE
- one-way project -> virtual assurance boundary in v1;
- immutable references/checksums/claim identifiers only;
- peer-to-peer evidence masquerading as compute state is forbidden.

The separation is deliberate: mixing control, learned state and evidence into one message
channel would create circular trust and make replay/audit much harder.

## 4. Coupling model

For the logical expert ring, do NOT import the physical hex Laplacian coefficients as
expert weights. Expert coupling is semantic, not spatial physics.

A later learned local update may use a residual form such as:

h_i(next) = self_memory(h_i)
            + local_gate * aggregate(local messages)
            + global_gate * relay_context

but local_gate/global_gate and aggregation weights require measured routing data and
ablation. No numeric defaults are asserted here.

For a physical HexOperator, the rule is different: coordinates, cell/point weights and
geometry-aware kernels belong in the numerical operator contract. On irregular meshes,
quadrature-aware graph neural operator formulations should be preferred over assuming six
equal neighbours.

## 5. Physical-operator branch: numerical contract

The expert-routing graph and the physical neural-operator graph must remain separate
packages/contracts.

For a regular physical lattice, the existing 7-point Laplacian is a useful mechanistic
oracle/baseline. For irregular/adaptive meshes, do not reuse equal-neighbour weights:
supply geometry, cell/point measure, boundary-condition metadata and quadrature weights
to the operator layer.

A physical HexOperator benchmark SHOULD include:
- resolution transfer with fixed underlying continuum problem;
- rotated/reflected cases only when the PDE/boundary conditions possess the symmetry;
- boundary and interior error reported separately;
- mechanistic residual/invariant checks in addition to field RMSE;
- comparison against non-neural numerical baselines;
- OOD shifts in coefficients, forcing, geometry and resolution as separate slices.

D6 equivariance is therefore an optional inductive bias, not a universal requirement:
the regular hex neighbourhood has D6 symmetry, but boundary conditions, material tensors
or forcing may break it.

## 6. Why local-only propagation is insufficient

Modern GNN research consistently identifies long-range information bottlenecks
(oversquashing), oversmoothing and underreaching in deep local message passing. Current
mitigations include:
- local-global hybrids;
- hierarchical coarsening;
- multiple virtual/relay representations;
- adaptive/asynchronous message passing;
- topology rewiring when justified.

The v1 virtual relay therefore acts as a semantic placeholder for long-range exchange.
For larger systems it must NOT become a single finite-dimensional bottleneck.

## 7. Scaling beyond one six-project cell

Recommended hierarchy:

micro level:
  six compute experts + typed virtual relays + external assurance

macro level:
  six cluster cells arranged on an independent axial hex coordinate system

meta level:
  cluster summaries/evidence references -> Nigin/Q-Forge/Global-G/Global Archive

A macro cell exports only a ClusterEnvelope/summary, not every internal message.

For scaled systems, use multiple relay slots or hierarchical relay pools rather than one
global vector. The number of relay slots is an empirical capacity parameter.

## 8. Message envelope and replay contract

Every transportable message SHOULD use an immutable envelope containing at least:
- producer-scoped message ID;
- task ID and trace ID;
- source and target;
- FlowKind;
- schema version;
- per-producer sequence number;
- hop TTL;
- payload SHA-256 and payload byte length;
- optional causation ID.

The design follows the robust ideas behind CloudEvents event identity and W3C distributed
trace propagation without claiming protocol-level conformance.

Required invariants:
- source + message_id identifies a duplicate/retry of the same logical event;
- sequence_no orders messages only within a producer/task stream, not globally;
- TTL must cover the planned path before dispatch;
- payload bytes must match both digest and declared size;
- FlowKind/source/target must match the precomputed FlowPlan;
- transport integrity never implies scientific validity.

Payload content should remain out-of-band for large artefacts: send immutable references
plus digests rather than duplicating full result objects through every node.

## 9. Information-flow cadence

Do not synchronize every node after every action.

Preferred event classes:
- LOCAL_STATE: event-driven after meaningful state change;
- GLOBAL_SUMMARY: periodic or threshold-triggered compressed update;
- CONTROL: immediate and ordered;
- EVIDENCE: append-only at artifact/claim boundaries.

Future scheduling should include sequence numbers, TTL/hop budget, payload byte limits,
deduplication IDs and backpressure.

## 10. Opposite and adjacent semantics

Adjacency should encode frequent collaboration, not similarity alone. The proposed ring
creates the high-frequency workflow chain above.

Opposite nodes are not automatically independent validators. Independence requires
separate lineage/solver/training/backend metadata from the cluster-fabric contract.

## 11. Shadow-routing development protocol

Before a learned HexaRouter can affect execution:
1. collect historical task -> reviewed-route cases;
2. evaluate in shadow mode only;
3. report ACCEPT/ABSTAIN/BLOCK separately;
4. report expert utilization and concentration;
5. report cost and latency;
6. separate reference-case acceptance coverage from agreement among accepted routes;
7. report reviewed-route recall/agreement as agreement, not truth;
8. add OOD/calibration tests;
9. activate only bounded in-distribution task classes after review.

## 12. Topology ablation requirement

Do not promote the ring topology merely because it is geometrically elegant. Under the
same total compute/tool budget, compare at least:
- single expert / no collaboration;
- centralized orchestration;
- sparse decentralized graph;
- six-node ring;
- all-to-all where feasible.

Report message count, bytes, wall time, redundant work and error propagation. Architecture
selection is part of the experiment.

## 13. Research anchors

- Li et al., Geometry-Informed Neural Operator, NeurIPS 2023.
- Berner et al., Principled approaches for extending neural architectures to function
  spaces for operator learning, Nature Machine Intelligence 2026.
- Attali et al., Graph Rewiring in GNNs to Mitigate Over-Squashing and Over-Smoothing,
  IJCAI 2026.
- Errica et al., Adaptive Message Passing, ICML 2025.
- Finder et al., Improving the Effective Receptive Field of Message-Passing Neural
  Networks, ICML 2025.
- RANGE, Extending the range of graph neural networks with global encodings,
  Nature Communications 2026.
- Local-global Graph Transformer for fluid-dynamics simulations, Journal of
  Computational Science 2026.
- Expert Choice Routing / Switch Transformers for sparse expert-routing baselines.

## 14. Non-goals

- no claim that an expert ring is a biological neural network;
- no claim that physical-neighbour weights apply to project experts;
- no direct learned path into Assurance;
- no automatic evidence promotion;
- no single global relay as a production scalability assumption;
- no claim that the proposed node order is optimal before ablation.
