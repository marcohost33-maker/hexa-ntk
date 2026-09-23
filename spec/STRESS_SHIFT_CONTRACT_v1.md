# Stress / Distribution-Shift Contract v1

**Version:** 1.0  
**Status:** PROPOSED — contractual source for PR #32; active only after merge.  
**Scope:** model-agnostic evaluation evidence for HexaNTK operator experiments.

## 1. Purpose

Distribution-shift claims MUST remain axis-specific. A result that is robust to
resolution changes does not imply robustness to new boundary-condition families,
geometries, forcing distributions, rollout horizons, or topology changes.

The implementation in `src/toolkit/stress_hex_v3.py` follows this specification.
No implementation-only enum, ratio rule, or aggregation rule is authoritative.

## 2. Shift axes

The v1 closed enum is:

- `parameter`
- `initial_condition`
- `forcing`
- `boundary_condition`
- `geometry`
- `resolution`
- `timestep`
- `rollout_horizon`
- `noise`
- `topology`

Adding, removing, or renaming an axis is a contract change and requires a spec
version change plus regression tests.

## 3. Regimes

Each stress case MUST declare exactly one regime:

- `id` — in-distribution relative to the declared training/evaluation domain.
- `ood` — outside the declared distribution while remaining inside the intended
  application family.
- `extrapolation` — outside the declared training support in a direction or
  range that requires extrapolation.

The regime label is evidence context, not a pass/fail verdict.

## 4. Error-metric semantics

The v1 degradation ratio accepts only finite, non-negative,
**lower-is-better error metrics**. Accuracy-like, signed, reward-like, or
higher-is-better scores MUST first use a separately specified transformation.

For baseline error E_0, shifted error E_s, and positive numerical floor epsilon:

[
D = E_s / max(E_0, epsilon)
]

Special cases:

- E_s = 0 and E_0 = 0 -> D = 1 (no degradation).
- E_s = 0 and E_0 > 0 -> D = 0.
- finite inputs whose ratio exceeds IEEE-754 binary64 range -> the stored factor
  saturates at `sys.float_info.max` and the case MUST set
  `degradation_saturated = true`;
- finite positive inputs whose positive ratio underflows to IEEE-754 zero -> the
  stored factor is clamped to the smallest positive representable float and the
  case MUST set `degradation_underflowed = true`.

Saturation/underflow flags are explicit numerical evidence. A clamped factor MUST
NOT be interpreted as the exact mathematical ratio magnitude.

## 5. Evidence integrity

A `StressCase` MUST have:

- stable non-empty `case_id`;
- one `ShiftAxis`;
- one `ShiftRegime`;
- finite non-negative `error_value`;
- JSON-like metadata with string keys and finite numeric values.

Metadata is snapshot evidence. Implementations MUST deep-copy/freeze nested
metadata so later mutation of the source configuration cannot alter a completed
report.

A `StressReport` MUST validate every stored `degradation_factor` and
`degradation_saturated` value against its own baseline, floor, and case error.
Manually constructed or deserialized reports that violate this relation are
invalid.

## 6. Aggregation prohibition

The contract intentionally defines **no overall stress score**.

Axis-specific evidence MAY be grouped for presentation, but heterogeneous shift
axes MUST NOT be averaged, weighted, or otherwise collapsed into a single
generalization verdict unless a later claim-specific specification explicitly
defines and justifies that decision rule.

## 7. Claim boundary

This contract structures evidence only. It does not:

- define pass/fail thresholds;
- prove neural-operator discretization invariance;
- prove boundary-condition generalization;
- define physics conservation metrics;
- define numerical-reference credibility;
- replace a ReferenceSpec / V&V assessment.

Thresholds belong to an experiment or Q-Forge policy with an explicit context of
use.

## 8. Research rationale

- APEBench (NeurIPS 2024 Datasets & Benchmarks) treats autoregressive rollout
  evaluation as a first-class PDE-emulator concern.
- PDEBench (NeurIPS 2022) supplements global normalized errors with additional
  physics- and structure-sensitive metrics.
- Boundary-condition generalization work at ICLR 2026 demonstrates that good
  in-distribution, forcing, or resolution performance does not by itself
  establish robustness to unseen boundary-condition families.

These references motivate the separation of evidence axes; they are not results
of this repository.

## 9. Compatibility

v1 serialization includes:

- `error_metric_name`
- `metric_semantics = nonnegative_error_lower_is_better`
- `baseline_case_id`
- `baseline_error`
- `degradation_floor`
- per case:
  - `case_id`
  - `axis`
  - `regime`
  - `error_value`
  - `degradation_factor`
  - `degradation_saturated`
  - `degradation_underflowed`
  - `metadata`

Consumers MUST ignore unknown future fields but MUST NOT silently reinterpret
known v1 fields.
