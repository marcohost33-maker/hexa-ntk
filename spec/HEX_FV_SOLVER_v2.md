# Hex FV solver v2 — CG stopping rule V2 (rounding floor)

**Version:** 2.0
**Status:** PROPOSED — becomes contractual after merge.
**Decision:** E1012-07 = A (Marco, 2026-10-12): Phase 0b I1 gets a versioned V2
with an attainable threshold; V1 stays as the historical, honestly red contract.
**Implementation:** `src/toolkit/hex_fv_solver_hex_v3.py`
(`conjugate_gradient(..., stopping="v2")`, `solve_steady_diffusion(..., stopping="v2")`).
**Scope:** this file changes ONLY the stopping rule of section 5 of
`spec/HEX_FV_SOLVER_v1.md`. Every other section of v1 (lattices, operator,
positivity gate, Crank-Nicolson, claim boundary) stays contractual unchanged.
The V1 stopping rule stays the default and callable (`stopping="v1"`); it is
not reinterpreted.

## 1. Why a V2

V1 decides `||b - A x||_2 <= rtol * ||b||_2` on the true residual. In binary64
the computed residual `fl(b - A x)` cannot resolve a residual below the error
of its own evaluation. On the Phase 0b MMS level N=192 (measured 2026-10-12,
see the gate log of that day) the true relative residual stagnates at
`2.344e-13 > rtol = 1e-13`, while the rounding scale
`u * (|| |A| |x| ||_2 + ||b||_2) / ||b||_2` is `5.546e-13` there and the worst-
case error bound of the residual evaluation (section 3) is `4.437e-12`. V1
asks for a computed residual more than an order of magnitude below the error
of its own evaluation; a value at that level is not distinguishable from
rounding. Both CG variants measured for v1 (restart from the true residual;
residual replacement keeping `p`, v1 section 5) miss it. V1 therefore stays
BLOCKED on that level, as a historical result.

## 2. Rule V2

With `u = 2^-53` (unit roundoff of binary64) and a factor `c` (section 3):

    rounding_floor(x) = c * u * (|| |A| |x| ||_2 + ||b||_2) / ||b||_2
    tol_eff(x)        = max(rtol, rounding_floor(x))

    accept x  iff  ||b - A x||_2 <= tol_eff(x) * ||b||_2

`|A|` and `|x|` are taken entrywise. The residual is the TRUE residual, as in
V1. The normwise backward error (Rigal & Gaches; Higham, *Accuracy and
Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, ch. 7)

    eta(x) = ||b - A x||_2 / (|| |A| |x| ||_2 + ||b||_2)

is reported in addition. It uses the same scale as the floor, so every
accepted iterate has `eta <= max(rtol * ||b|| / (|| |A| |x| || + ||b||), c u)`;
in particular `at_rounding_floor` implies `eta <= c u`.

**When the floor is consulted.** V2 iterates exactly like V1 (same pre-check,
same residual replacement, same stagnation counter). It returns:

- `status = "converged"` as soon as a true residual satisfies
  `||b - A x|| <= rtol * ||b||` (the V1 condition; then the iterate and its
  residual are bit-identical to V1);
- `status = "at_rounding_floor"` when V1 would give up (stagnation: no new
  minimum in 50 true-residual checks; `maxiter`; `r . r` underflow) AND the
  best iterate satisfies `rounding_floor < 1` and
  `||b - A x|| <= tol_eff * ||b||` for that iterate; `stop_reason` names the
  give-up;
- otherwise the V1 `ConvergenceError`, unchanged.

The floor is consulted only once rtol is shown unreachable, not at the first
true-residual check below `tol_eff`. Measured 2026-10-12 on the MMS levels: the
first true-residual check lies below `tol_eff` but above `1e-13` on N=48
(`1.244e-13`, floor `2.78e-13`) and N=96 (`5.727e-13`, floor `1.110e-12`),
and the restarts then reach `5.19e-14` and `9.29e-14`. Stopping at the first
check would label two levels `at_rounding_floor` on which rtol is attainable.
This follows the "stopping when progress is no longer being made" criterion of
the Templates (Barrett et al., *Templates for the Solution of Linear Systems*,
SIAM 1994, section 4.2).

**Fail-closed.**
- Acceptance hangs on comparisons that must be TRUE; a NaN in any term
  rejects.
- `|A| |x|` not finite, or `|| |A| |x| || + ||b||` overflowing: the floor is not
  computable -> `ConvergenceError` (no floor, no acceptance), also on the
  `converged` path, because V2 always reports `tol_eff` and `eta`.
- `apply_abs_a` returning a wrong shape or a negative entry -> `ValueError`.
- `rounding_floor >= 1` never accepts (the floor carries no information).
- Non-finite `b`, `x0` or iterate: the V1 errors, unchanged.
- `b == 0` returns `x = 0` with `status = "converged"`, `eta = 0` (the residual
  is exactly zero).

**Scale invariance.** CG runs on the system scaled by a power of two (v1
section 5). Every V2 term is a ratio of quantities that scale together, so
multiplying `A` by `2^k` and `b` by `2^j` (no overflow or underflow) leaves
the status, the relative residual, `tol_eff` and `eta` bit-identical, and the
solution scales by exactly `2^(j-k)`.

## 3. The factor `c`

The computed residual satisfies, componentwise,

    |fl(b - A x) - (b - A x)| <= gamma_(m+1) * (|b| + |A| |x|),
    gamma_k = k u / (1 - k u),

where `m` is the number of terms in a row of `A x` (the inner-product error
bound, Higham 2002, applied to `b - A x`; Templates section 4.2, "Accounting
for floating point errors", states the same kind of bound for the computed
residual). Taking 2-norms and the triangle inequality gives
`||error||_2 <= gamma_(m+1) (|| |A| |x| ||_2 + ||b||_2)`.

For the hex operator a row of `A x = -(diag x + sum_s off x_n(s))` has at most
`m = 7` terms (diagonal plus six slots; Dirichlet slots contribute an exact
zero), so `c = m + 1 = 8` (`HEX_RESIDUAL_FLOOR_FACTOR`). `c u` equals
`gamma_8` up to a relative `8u`; the threshold is a rounding-level scale, not
a certificate to the last ulp. The bound is a worst case; the typical error is
smaller, which is why a measured stagnation level (N=192: `2.344e-13`) lies
below `c u (...)` (`4.437e-12`).

Other operators passed to `conjugate_gradient(stopping="v2")` must supply
their own `c` (`floor_factor >= 1`) from the same argument.

## 4. What V2 claims and what it does not

- `converged`: the true relative residual is `<= rtol` (the V1 claim).
- `at_rounding_floor`: the true relative residual is `<= tol_eff` and `> rtol`;
  the iterate is backward stable to `eta`; rtol was NOT reached. A claim
  "reaches rtol" must not be made for such a solve.
- Neither status says anything about the forward error `||x - x_exact||`; that
  is bounded by the condition number times `eta` and is checked separately
  where it matters (Phase 0b I1 iterative-error criterion).
