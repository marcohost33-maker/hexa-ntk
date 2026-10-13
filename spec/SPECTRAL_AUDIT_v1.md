# Spectral audit v1 — input Jacobian on the T1 modes, and a linear POD-ridge baseline

**Version:** 1.0
**Status:** PROPOSED — a secondary instrument for Phase 1. It is NOT part of
the preregistration (`spec/PREREGISTRATION_PHASE1_v1.md`, DRAFT) and changes
none of its metrics or decision rules. Whether it enters a preregistration
version as a secondary measurement is the owner's decision.
**Implementation:** `src/toolkit/spectral_audit_hex_v3.py` (audit),
`src/toolkit/linear_baseline_hex_v3.py` (baseline).
**Tests:** `test_spectral_audit_hex_v3.py` (contract, torch, fail-closed),
`test_spectral_audit_oracles_hex_v3.py` (oracles, negative controls,
metamorphic relations), `test_linear_baseline_hex_v3.py`.

## 1. Purpose

The relative errors of task T1 (`spec/HEAT_OPERATOR_TASK_v1.md` section 3)
say how far a prediction is from the truth, not why. For the heat semigroup
the truth is diagonal in a known basis, so the error can be split by
frequency:
- a **frequency-response error**: mode `k` decays at the wrong rate;
- **mode coupling**: input mode `j` produces output in mode `k != j`;
- **out-of-span response**: input mode `j` produces output outside the basis.

The audit measures these three for any operator `u0 -> uT`: the mechanistic
baseline, a learned model, or the linear baseline of section 6.

## 2. Definition

For an operator `F` on lattice fields of length `M`, a base point `u0`
(default `0`), the cell weights `W = diag(w)` and a basis `Q` (`M x K`, columns
W-orthonormal, `Q^T W Q = I`):

    J = dF/du (u0),    S = Q^T W J Q    (K x K).

Column `j` of `S` holds the basis coordinates of `J q_j`, the response to input
mode `j`. Reported per audit:

| Quantity | Definition |
|---|---|
| `diagonal` | `S_kk`, the frequency response |
| `diag_abs_error`, `diag_rel_error` | `|S_kk - o_k|` and that divided by `|o_k|` for an oracle `o` (0 if both are 0, `inf` if only `o_k` is 0) |
| `coupling_fraction` | `||S - diag(S)||_F / ||S||_F`; `None` if `S = 0` |
| `out_of_span` | `||J q_k - Q S[:, k]||_W / ||J q_k||_W` per mode (0 if `J q_k = 0`); not available in torch reverse mode |
| `spectral_rel_error` | `||S - diag(o)||_F / ||o||_2` |

For a linear operator `S` does not depend on `u0`. For a nonlinear one it
does; the base point is part of the measurement and must be reported.

## 3. The T1 basis on a lattice

Measured on 2026-10-13 (`N = 16, 32, 64, 128`, both domains): the basis
functions of `mode_basis` are continuous eigenfunctions sampled at the nodes,
and they are
- exact eigenvectors of the discrete hex Laplacian `L_h` with eigenvalue
  `lambda_h = discrete_eigenvalue(h)` (relative residual <= 1.1e-12; Phase 0
  C1);
- W-orthogonal to rounding (normalised Gram off-diagonal <= 9.7e-16);
- **not** W-normalised: `||phi||_W` is 1.140 to 1.612 on the triangle and
  0.658 on the torus.

`heat_mode_matrix` therefore divides each function by its W-norm. It raises
`ValueError` when the lattice cannot carry the basis:
- more basis functions than nodes (triangle `N <= 6`, torus `N <= 6` for the
  T1 defaults);
- aliasing: max `|Q^T W Q - I|` above the rounding bound (example: torus,
  `lambda_cutoff = 2000`, `N = 12`, defect 1.0).

**Orthonormality bound.** Each entry of `Q^T W Q` is a W-weighted dot product
of two W-unit vectors. Its rounding error is at most `gamma_M` times
`sum_i w_i |q_ik q_il|`, which is at most 1 by Cauchy-Schwarz (Higham 2002,
section 3.1, inner products). The reference size is therefore 1; the default tolerance is
`4 M eps`. A caller may pass a looser `orthonormality_tol`.

## 4. Oracles

All three are diagonal in the basis.

| Operator | Diagonal of `S` |
|---|---|
| (i) semidiscrete semigroup `exp(kappa T L_h)` | `exp(-kappa lambda_h T)` |
| (ii) Crank-Nicolson, `steps` steps of `dt` (`mechanistic_baseline`) | `((1 - z/2) / (1 + z/2))^steps`, `z = kappa lambda_h dt` |
| (iii) continuous T1 truth | `exp(-kappa lambda T)` |

(i) and (ii) follow from `L_h q_k = -lambda_h q_k`. (iii) is what a model
trained on T1 data should learn. Against (iii), (i) carries the spatial
discretisation error: on the triangle the highest T1 mode differs by 97 %
at `N = 16` and by 19 % at `N = 32`; on the torus by 109 % and 21 %
(measured 2026-10-13).

**Measured accuracy of the audit** (2026-10-13, Windows, numpy, triangle
`N = 16, 32`, torus `N = 16, 32`):
- (i), operator built from the dense eigendecomposition of the assembled
  `L_h`: max relative diagonal error 3.2e-14, coupling <= 3.8e-14;
- (ii), `mechanistic_baseline` with `steps` 4 and 32: max relative error
  9.4e-15; the out-of-span residual sits at the CG stopping level
  (<= 2.2e-12);
- a torch linear model with known weights: `S = Q^T W M Q` to 1.4e-15
  relative to `max |S|`, forward and reverse mode;
- the closed form (ii) against a dense matrix power, independent of the
  basis: within the test tolerance `1e-12`.

The tests use `1e-12`. A rate error of `1e-10` and a coupling admixture of
`1e-9` are detected at that tolerance (the tolerance is not blind).

## 5. Backends, budget, verdict

**numpy callables:** central differences along each mode, `2K + 1` calls. The
first call evaluates `F(u0)`; its output dtype sets the step

    h = eps_out^(1/3) * max(1, ||u0||_W)

(Nocedal & Wright 2006, section 8.1). A float32 operator thus gets
`h ~ 4.9e-3`, not `6.1e-6`. Measured on a float32 MLP: 1.9e-5 relative error
with that step, 2.7e-3 with the float64 step. An operator that computes in
float32 but returns float64 hides its precision; pass `fd_step` then.

**torch callables and `nn.Module`:**
- forward mode (default): `torch.func.jvp`, `K` calls, gives `J q_k`;
- reverse mode: `torch.autograd.grad` with cotangents `W q_k`, one forward
  and `K` backward passes, gives the rows of `Q^T W J`. There is no
  out-of-span residual in this mode.
- The dtype follows the module's parameters (float64 without parameters).
- A module in training mode raises unless `allow_train_mode=True`: dropout and
  batch statistics would make `J` random and change buffers. The audit leaves
  no `.grad` behind.

**Fail-closed:**
- invalid arguments raise `TypeError` or `ValueError`;
- a non-orthonormal `Q` raises `ValueError`;
- a non-finite operator output (including at `u0` alone), Jacobian product or
  `S` raises `NonFiniteOutputError`;
- more than `max_evaluations` calls raise `BudgetExceededError` before the
  first call; exceeding `max_seconds` raises it during the run.

**Verdict** (`audit_verdict`). The result is `PASS`, `FAIL` or `ABSTAIN`.
- `PASS` needs all three:
  - every mode within `|S_kk - o_k| <= atol + rtol |o_k|`;
  - `coupling_fraction <= coupling_tol`;
  - if `out_of_span_tol` is given, every residual within it.
- Each acceptance is a comparison that must be true, so NaN fails.
- `ABSTAIN` in any of these cases:
  - there is no oracle;
  - a tolerance is missing, NaN, infinite, negative or not a real number;
  - an out-of-span tolerance is given, but reverse mode measured no residual.
- A zero `S` has no coupling fraction and fails.

## 6. Linear POD-ridge baseline

`PODRidgeBaseline` is the linear case of PCA-Net (Bhattacharya, Hosseini,
Kovachki & Stuart 2021):

    a = P_in^T W u0,   b = B^T a,   uT = P_out b,
    B = argmin ||A B - C||_F^2 + alpha ||B||_F^2.

`P_in` and `P_out` are W-orthonormal POD bases of the training inputs and
outputs. `A` and `C` are the training coordinates. The data are not centred,
because the target operators are linear.

**Rules fixed before any test data** (`LinearBaselineConfig`):
- POD rank: the singular values above `rank_rtol * s_max`, by default
  `max(N, M) eps`, the numerical-rank rule of `numpy.linalg.matrix_rank`;
- an optional `max_rank` caps the rank;
- `alpha = alpha_rel * s_max(A)^2`, with default `alpha_rel = 1e-10`, so the
  fit is equivariant under scaling of the data.

**Leakage guard.**
- `fit` records the training sample ids.
- `predict(u0, sample_ids)` raises `LeakageError` (a `ValueError`) if any
  scored id was trained on.
- The id-free operator view (`__call__`, `matrix`) is for instruments such as
  this audit. It must not be used to score samples.

**Checked by the audit** (synthetic linear data: `t_final = 0.005`, not the
T1 value, coefficients from a private RNG):
- with `2K` to `4K` samples, the fit reproduces `exp(-kappa lambda_h T)` to
  <= 9.3e-10 with the default `alpha_rel`, and to <= 8.5e-14 with
  `alpha_rel = 0`;
- the ridge bias grows linearly with `alpha_rel`;
- fewer samples than modes, or a capped rank, fail the audit.

## 7. Limits

- **Input Jacobian, not the parameter NTK.** `S` is the derivative with
  respect to the INPUT at one base point. The neural tangent kernel
  `J_theta J_theta^T` (derivative with respect to the parameters) describes
  training dynamics. It is not measured here and stays open in issue #60.
- **Local.** For a nonlinear model `S` holds at its base point only. One
  audit says nothing about other inputs.
- **Span.** `S` sees the basis only. Response outside it is visible only
  through the out-of-span residual. Inputs outside the basis are not probed at
  all.
- **Basis dependence.** Inside a degenerate eigenspace (the sin/cos pairs) the
  truth stays diagonal under any rotation. A model's coupling fraction can
  change under such a rotation.
- **Finite differences.** For a nonlinear numpy operator the result carries
  the truncation error `O(h^2)`. Prefer the torch backend for models.
- **No T1 fit.** Nothing here is fitted to or evaluated on T1 data. The
  preregistration forbids that before approval.

## 8. Use as a secondary measurement (adopted 2026-10-13 in PREREGISTRATION_PHASE1_v1 section 5)

If the owner adopts it in a preregistration version:
- audit every GINO seed at base point `0`, and at a fixed set of test-sample
  base points named in that version, against oracle (iii);
- report per mode `diag_rel_error`, `coupling_fraction` and `out_of_span`;
- report the mechanistic baseline at the same `(N, steps)` next to it;
- fit the POD-ridge baseline on the training split only and report it as an
  additional baseline, not a decision baseline.

No decision rule of the preregistration would depend on these numbers.

## 9. Claim boundary

Instruments only:
- no trained model;
- no statement that any learner is better or worse;
- the numbers above are measurements of the instruments against their
  oracles, on the stated lattices and machine.
