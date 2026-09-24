# Cluster Fabric v1 — deterministic safety spine for future HexaNTK routing

**Status:** research contract / no model claim  
**Scope:** heterogeneous expert orchestration; not a GINO/GINOT/GAOT implementation and not evidence of scientific correctness.

## 1. Decision

A future learned HexaNTK router may **propose** compute experts, but it does not authorize
scientific execution or evidence promotion. The architecture is split into two planes:

1. **Compute plane** — domain/method experts such as ADE, SciForge, PADE, MRT, LiouScope.
2. **Assurance plane** — independent audit/validation experts such as ARK/USK/QVAR.

Assurance experts are never selectable by the learned compute router. Hard contracts are
evaluated before execution and can BLOCK or ABSTAIN.

The six-project reference cell is therefore a *logical expert graph*, not a claim that the
current HexaNTK repository implements six neural nodes or an NTK over six projects.

## 2. Why the previous concept needed hardening

A fixed "Top-2" policy is too strong. Modern Mixture-of-Experts systems show routing/load
trade-offs, expert collapse, capacity effects and benefits from variable routing. In
scientific workflows the correct number of experts also depends on cost, uncertainty and
the required independence of cross-checks.

Likewise, a learned score cannot replace:
- declared capability/domain applicability;
- budget/resource admission;
- implementation-family independence for cross-checks;
- OOD/uncertainty handling;
- independent evidence/adjudication.

## 3. Contract objects

### ExpertSpec
- expert identity
- capabilities
- supported domains (wildcard allowed explicitly)
- compute vs assurance plane
- implementation family / independence group
- estimated cost
- enabled state

The independence metadata is governance-owned registry data, not a self-assertion by the
expert being routed. Two experts sharing the same solver core, training labels, checkpoint
lineage or dominant backend must not be treated as independent merely because their
project names differ.

### TaskSpec
- task/domain
- required compute capabilities
- minimum/maximum expert count
- maximum total estimated cost
- minimum acceptable router confidence
- optional requirement for independent implementation families
- mandatory assurance capabilities

### RouterProposal
- ordered compute-expert IDs
- router confidence

### RouteDecision
- ACCEPT / ABSTAIN / BLOCK
- selected compute experts
- separate assurance candidates
- machine-readable reason codes
- estimated selected cost

## 4. Fail-closed invariants

The deterministic adjudicator MUST:
1. BLOCK unknown or duplicated routed expert IDs.
2. BLOCK any attempt to route an assurance expert through the learned compute path.
3. BLOCK disabled, domain-invalid or capability-incompatible compute experts.
4. BLOCK invalid/non-finite router confidence.
5. ABSTAIN below the task's confidence floor.
6. Enforce budget and requested expert-count bounds.
7. For independent cross-checks, require at least two distinct implementation families.
8. BLOCK when mandatory assurance capabilities have no eligible assurance expert.
9. Never convert a BLOCK/ABSTAIN into a scientific PASS.

## 5. Reference six-project cell

A sensible **pilot profile**, not a hard-coded product topology:

- ADE — adaptive candidate generation / decision support
- SciForge — workflow/experiment compilation
- LiouScope — open-system/dynamical diagnostics
- PADE — rational approximation / pole-residue diagnostics
- MRT — inverse-problem / measurement-domain expert
- ARK — assurance/provenance plane

Important: ADE/SciForge/LiouScope/PADE/MRT are not assumed to share one mathematical state
space. Cross-domain transfer requires an explicit adapter and held-out evidence. ARK stays
outside the learned reward loop.

## 6. Future learned router objective — not implemented in v1

A later router should optimize a constrained multi-objective target rather than raw task
loss alone, for example:

    J = task_loss
        + lambda_cost * cost
        + lambda_balance * load_penalty
        + lambda_ood * ood_risk
        + lambda_tail * tail_risk

subject to the deterministic constraints above.

No lambda value is specified by this contract. These weights require measured pilot data
and ablation studies.

## 7. Evaluation protocol before learned routing may affect execution

### Phase A — deterministic baseline
- capability-only routing;
- manual/heuristic ordered proposals;
- synthetic negative tests;
- compute/assurance separation.

### Phase B — shadow learned router
- learned proposals are logged but do not affect execution;
- compare against deterministic/manual routes;
- report route confusion, cost, failure rate, expert utilization and abstention.

### Phase C — bounded activation
- only in-distribution task classes with calibrated uncertainty;
- route changes remain reversible and fully logged;
- assurance plane unchanged and excluded from training labels.

### Phase D — OOD / cluster-of-clusters
- only after explicit distribution-shift and transfer experiments;
- no cross-domain latent-space claim without held-out evidence.

## 8. Required evaluation metrics

Do not reduce evaluation to mean task score. Record:
- task error / domain-specific metrics;
- physical or mathematical residual/consistency metrics where applicable;
- per-seed distribution and tail failures;
- OOD slice performance;
- router abstention rate;
- invalid-route rejection rate;
- expert utilization and concentration;
- end-to-end compute/latency cost;
- disagreement rate between independent implementation families;
- calibration/coverage metrics when uncertainty is emitted.

## 9. External design anchors

- Fedus, Zoph, Shazeer (2022), Switch Transformers, JMLR — sparse routing,
  load/capacity and routing stability.
- Zhou et al. (2022), Mixture-of-Experts with Expert Choice Routing — fixed top-k is
  not the only viable routing policy.
- Li et al. (2023), Geometry-Informed Neural Operator — irregular geometry to regular
  latent operator processing; relevant to future operator backends, not implemented here.
- Berner et al. (2026), Principled approaches for extending neural architectures to
  function spaces for operator learning, Nature Machine Intelligence — discretization
  agnosticism and function-space consistency as neural-operator design requirements.
- 2025–2026 SciML benchmark/UQ literature — evaluate distribution shift, physical
  consistency and uncertainty separately; do not infer reliability from in-distribution
  mean error alone.

## 10. Non-goals

- no claim that six project nodes form a biological/neural hexagon;
- no learned router implementation in v1;
- no cross-domain shared latent representation claim;
- no automatic scientific claim promotion;
- no replacement of domain solvers/validators;
- no merge of PR #31 or other unrelated open work.
