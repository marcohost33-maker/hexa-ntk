# HexaFabric seed v1 — independent annotation queue

**Purpose:** promote selected historical routing cases from `MODEL_REVIEWED_SEED` to a
reviewed benchmark without treating the current model-authored labels as ground truth.

## Reviewer rules

- Review the original 007acc2 source task, not only this candidate label.
- Do not use project ownership as a shortcut for route correctness.
- Judge atomic capabilities first, then expert mapping.
- ACCEPT may have multiple acceptable expert sets.
- Prefer ABSTAIN when a required capability is absent or only weakly transferable.
- Record disagreements verbatim; do not average them into a fake consensus.
- Assurance experts are reviewed separately from compute experts.

## Cases

| Case | Historical source | Seed class | Seed action | Candidate compute route | Main review question |
|---|---|---|---|---|---|
| hxbench-001 | LiouScope #127 dissipation/tolerance design | SUPPORTED | ACCEPT | LiouScope + PADE | Is PADE genuinely useful for the numerical tolerance/norm-design part, or should LiouScope alone own it? |
| hxbench-002 | Q-Forge cross-backend P1 landing path | SUPPORTED | ACCEPT | Q-Forge | Does this need a second compute expert, or is independent assurance sufficient? |
| hxbench-003 | LiouScope #78 maximally-mixed physics review | SUPPORTED | ACCEPT | LiouScope | Is a second numerical/decision expert useful or only coordination overhead? |
| hxbench-004 | AllScan v7 detector/tracker/autocrop | PARTIAL | ABSTAIN | none | Are MRT measurement/inverse-problem skills transferable enough, or is computer-vision expertise mandatory? |
| hxbench-005 | DDE #49 fail-closed/security hardening | PARTIAL | ABSTAIN | none | Can Q-Forge evidence/contract skills contribute safely without a dedicated security expert? |
| hxbench-006 | Hex-HQST #22 absolute topology sign | UNSUPPORTED | ABSTAIN | none | Does the portfolio contain a true band-topology expert not represented in the six-node cell? |
| hxbench-007 | Wasser-Matrix #16 IAPWS R14 review | UNSUPPORTED | ABSTAIN | none | Is a thermodynamics/reference-data expert required before any compute route is admissible? |
| hxbench-008 | Browser hostile-content runtime review | UNSUPPORTED | ABSTAIN | none | Should Browser/Insandox become a dedicated security/runtime expert in a different cell? |
| hxbench-009 | Tool3D PR19/20 trust-boundary review | UNSUPPORTED | ABSTAIN | none | Should Tool3D be routed through a manufacturing/security cell rather than the research hex? |

## Promotion record per case

Record:

1. reviewer identifier/family;
2. review date;
3. ACCEPT/ABSTAIN;
4. acceptable compute expert sets;
5. required assurance capabilities;
6. excluded experts and reason;
7. confidence category: high / medium / low (descriptive, not calibrated);
8. disagreements with prior reviewers;
9. source revision/date;
10. supersedes annotation version.

Two independent reviews are recommended before a case loses `requires_human_review=true`.
