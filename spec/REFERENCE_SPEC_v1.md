# ReferenceSpec v1 — Numerical Reference Credibility Contract

**Version:** 1.0  
**Status:** PROPOSED — becomes contractual after merge.  
**Scope:** references used to evaluate HexaNTK / Q-Forge simulation and operator results.

## 1. Purpose

A numerical solver output MUST NOT be labelled or treated as generic "ground truth"
merely because it is higher fidelity than a learned surrogate.

This contract separates:

1. **code verification** — evidence that the implementation solves the intended
   mathematical equations correctly;
2. **solution verification** — evidence about numerical error in the particular
   computed solution, including discretization and iterative error;
3. **model validation** — comparison of a verified simulation with physical
   observations in the relevant context of use;
4. **uncertainty** — quantified uncertainty attached to numerical or experimental
   evidence.

The contract is intentionally conservative: incomplete evidence lowers the claim
ceiling instead of being silently promoted.

## 2. Reference kinds

`ReferenceKind` is one of:

- `analytic` — closed-form/exact mathematical solution where applicable;
- `manufactured` — method-of-manufactured-solutions oracle for code verification;
- `numerical` — result produced by a numerical solver;
- `experimental` — physical measurement/experiment data.

A kind describes provenance, not credibility.

## 3. Evidence status

Every verification item uses:

- `not_assessed`
- `pass`
- `fail`
- `not_applicable`

A `pass` MUST cite a non-empty evidence reference.
A `not_applicable` MUST carry a rationale.

## 4. Context of use

Every `ReferenceSpec` MUST declare a non-empty `context_of_use` and
`quantity_of_interest`.

Reference credibility is bounded to that context. A reference verified for one
observable, parameter region, boundary-condition family, geometry, or numerical
regime MUST NOT automatically support a broader claim.

## 5. Refinement studies

Spatial and temporal solution-verification studies use ordered refinement levels:

[
(h_i, f_i), quad i=1,ldots,n
]

where `characteristic_size` is positive and finite and `quantity_value` is
finite.

The contract permits a study with two or more levels for descriptive refinement
evidence. However:

- an **observed order of convergence** requires at least three levels;
- an **asymptotic-range claim** requires at least three levels;
- a `ReferenceSpec` promoted to `verified_numerical_reference` requires at
  least three levels for every applicable spatial/temporal refinement axis.

This is a project-level conservative rule. NASA guidance notes that GCI can be
computed from two grids, but recommends three levels to estimate observed order
accurately and assess the asymptotic range.

If an observed order is recorded, it MUST be finite and positive and it is
**not trusted as caller metadata**. ReferenceSpec v1 recomputes the order from
the three finest levels using the constant-refinement-ratio Richardson form:

`p = ln(|(f3 - f2)/(f2 - f1)|) / ln(r)`.

The three finest levels MUST therefore have the same refinement ratio (within
floating-point tolerance) and monotone, non-degenerate successive
differences. A supplied `observed_order` inconsistent with that recomputation
is rejected. Unequal-r studies remain representable as raw refinement evidence,
but v1 does not permit them to claim a verified observed order; a generalized
Richardson treatment belongs in a future contract version.

A reported `fine_grid_relative_error_estimate` is a non-negative **fraction**
(not percent) associated with the finest level.

For an applicable spatial/temporal axis to support the
`verified_numerical_reference` ceiling, the matching refinement study MUST
contain all of:

- at least three levels;
- a finite positive observed order;
- `asymptotic_range_checked = true`;
- a finite non-negative `fine_grid_relative_error_estimate`.

Three raw levels without these assessments remain useful refinement evidence but
do not by themselves constitute high-credibility solution verification.

## 6. Iterative convergence

Iterative convergence is represented as an evidence check rather than a spatial
or temporal refinement study. A solver/reference may mark it
`not_applicable` only with a rationale explaining why no iterative solve exists
or why iterative error is not a relevant component.

For a verified numerical reference, iterative convergence MUST be either
`pass` or justified `not_applicable`.

## 7. Numerical uncertainty

A numerical reference promoted to `verified_numerical_reference` MUST include a
finite non-negative numerical-uncertainty statement.

`UncertaintyKind` is:

- `absolute`
- `relative_fraction`

The uncertainty includes an evidence reference and refers to the declared
quantity of interest and context of use.

The presence of an uncertainty value does not by itself prove verification.

## 8. Experimental uncertainty

Experimental data is validation evidence, not mathematical truth.

A physical dataset may reach
`validation_data_with_uncertainty` only when a finite non-negative measurement
uncertainty is supplied with an evidence reference. That classification does not
mean the experiment is error-free and does not turn it into a numerical oracle.

## 9. Oracle evidence

Analytic and manufactured references may reach
`mathematical_oracle` only when `oracle_verification` is `pass`.

For manufactured solutions this evidence should establish the substituted
solution, source/residual construction, and applicable initial/boundary
conditions. For analytic references it should document the equation/domain/
condition assumptions under which the solution is exact.

## 10. Derived claim ceiling

The implementation derives one of four claim ceilings:

- `mathematical_oracle`
- `verified_numerical_reference`
- `validation_data_with_uncertainty`
- `comparison_only`

No user-settable boolean such as `is_ground_truth` exists.

### 10.1 Mathematical oracle

Requires:

- kind is `analytic` or `manufactured`;
- `oracle_verification.status == pass`.

### 10.2 Verified numerical reference

Requires all of:

- kind is `numerical`;
- code verification is `pass`;
- spatial, temporal, and iterative solution-verification checks are each either
  `pass` or justified `not_applicable`;
- **at least one** of those solution-verification axes is an actual `pass`;
- each applicable spatial/temporal `pass` has a matching refinement study that
  satisfies the full high-credibility criteria in Section 5;
- numerical uncertainty is present.

A failed or not-assessed required check keeps the reference at
`comparison_only`.

### 10.3 Validation data with uncertainty

Requires:

- kind is `experimental`;
- measurement uncertainty is present.

This classification is for validation comparisons only.

## 11. Verification versus validation

Verification and validation MUST remain distinct:

- verification asks whether the mathematical model/equations are being solved
  correctly and how accurate the numerical solution is;
- validation asks how well the model represents the physical system for an
  intended context of use.

A numerically verified solver is not automatically physically validated.
Experimental agreement does not replace numerical solution verification.

## 12. Research basis

This contract is grounded in current VVUQ practice:

- NISTIR 8298 distinguishes code verification, solution verification,
  uncertainty quantification, and model validation.
- NASA grid-convergence guidance recommends three refinement levels for reliable
  observed-order estimation and asymptotic-range checking.
- ASME VVUQ 1 frames credibility evidence relative to a context of use.
- ASME V&V 20 compares simulation and data while accounting for errors and
  uncertainties in both.

References:
- https://doi.org/10.6028/NIST.IR.8298
- https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html
- https://www.asme.org/codes-standards/find-codes-standards/verification-validation-and-uncertainty-quantification-terminology-in-computational-modeling-and-simulation
- https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer

## 13. Claim boundary

ReferenceSpec v1 is an evidence-structure and claim-ceiling contract. It does not:

- generate refinement data or independently establish the asymptotic range;
- prove a code is correct;
- decide whether the underlying physical model is adequate;
- validate a trained neural operator;
- replace domain-specific V&V judgment or standards;
- permit a numerical result to be called exact truth.

Actual verification calculations and raw evidence MUST remain separately
auditable.
