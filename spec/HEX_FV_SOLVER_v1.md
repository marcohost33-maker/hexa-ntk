# Hex FV solver v1 — mechanistic baseline layer

**Version:** 1.0
**Status:** PROPOSED — becomes contractual after merge.
**Implementation:** `src/toolkit/hex_fv_solver_hex_v3.py`
**Consumers:**
- the Reality Anchor reference cases (`spec/REFERENCE_CASE_HEX_HEAT_TRI_v1.md`,
  `spec/REFERENCE_CASES_HEX_PHASE0B_v1.md`);
- the mechanistic baseline of Phase 1 (#60).

## 1. Purpose

This is one small, verifiable finite-volume layer on regular hex lattices. The
reference cases verify it against exact or manufactured solutions. Phase 1 can
then use the same code as the "efficient numerical method" baseline (McGreivy
& Hakim, rule 2), so no second, unverified solver is needed.

## 2. Lattices

Every lattice uses spacing `h = 1/N` and the six neighbour slots,
counter-clockwise from east, as index offsets `(di, dj)` of the node
`x_ij = h * (i + j/2, j * sqrt(3)/2)`:

| slot | 0 E | 1 NE | 2 NW | 3 W | 4 SW | 5 SE |
|---|---|---|---|---|---|---|
| `(di, dj)` | `(1, 0)` | `(0, 1)` | `(-1, 1)` | `(-1, 0)` | `(0, -1)` | `(1, -1)` |

Slot `s` and slot `s + 3 (mod 6)` are opposite. Each Voronoi cell is a regular
hexagon of area `(sqrt(3)/2) h^2`.

- **`triangle_dirichlet`** (`N >= 3`):
  - Unknowns: interior nodes of the unit equilateral triangle,
    `i, j >= 1`, `i + j <= N - 1`.
  - Slot value `-1`: the neighbour is a boundary node with value 0
    (homogeneous Dirichlet).
- **`torus_periodic`** (`N >= 4`, even):
  - Unknowns: the `N^2` nodes with `0 <= i, j < N`.
  - The lattice is periodic under `P1 = (1, 0)` and `P2 = (0, sqrt(3)/2)`.
    In index terms these are `(i, j) ~ (i + N, j) ~ (i - N/2, j + N)`, which is
    why `N` must be even.
  - Coordinates are stored reduced into `[0, 1) x [0, sqrt(3)/2)`.
  - Every slot is valid.

Arrays are read-only. `-1` never means "missing neighbour" here. That is a
different convention from the H1 proxy in `benchmark_hex_v3`.

**Lattice contract (added 2026-10-12, GPT6 W9-F01).** Every function that
takes a lattice accepts only one whose `n`, `h`, `coords` and `neighbors`
(shape, dtype and values) equal the factory lattice of its domain and `n`;
anything else raises `ValueError` (a wrong type or domain stays `TypeError`).
The computation then runs on a private read-only copy of the factory lattice.
The read-only flag alone is advisory, so a lattice changed after construction
is rejected as well.

## 3. Operator

Two-point-flux FV Laplacian with symmetric edge conductances `c`:

    (L_c u)_i = w * sum_s c_{i,s} (u_{n(i,s)} - u_i),   u_{-1} = 0,
    w = hex_laplacian_stencil_weight(h) = 2 / (3 h^2)   (engine, src/hexantk_v3.py)

**Coefficients:**
- **Constant coefficient:** `c = kappa` (a positive scalar).
- **Variable coefficient:** `c_{i,s} = kappa(x_i + (h/2) e_s)`, kappa evaluated at the
  edge midpoint, where `e_s` is the unit vector of slot `s`.
  - The value of an interior edge is computed once, from slots 0-2, and copied
    to the partner slot. This makes `c_{i,s} == c_{n(i,s), s+3}` exactly in
    binary64.
  - `kappa` must be finite and `> 0` at every midpoint.

**Consistency.** The lattice has the isotropic second moment
`sum_m a_m a_m^T = (3/2) h^2 I` over its three directions, so
`L_c u = div(kappa grad u) + O(h^2)` for smooth `kappa` and `u`.

**Contract checks on a conductance array:** shape `(M, 6)`, finite, `> 0` on
every slot (as for a scalar or callable `kappa`; added 2026-10-12) and
symmetric as above. A violation raises `ValueError`.

**Conservation.** On the torus, `sum_i (L_c u)_i = 0` for every `u`, because
every edge flux appears twice with opposite signs.

## 4. Positivity gate (issue #60, item 1)

`assemble_hex_operator` returns `L_c` as `(diag, off)`.

`positivity_max_dt(diag, off)` requires:
- a negative diagonal;
- non-negative off-diagonals (`-L_c` is then an M-matrix).

It returns `dt = min_i 1 / |diag_i|`, rounded so `1 + dt * diag_i >= 0` holds in
binary64. A violation raises `ValueError`. Under exactly these conditions a
forward-Euler step maps non-negative states to non-negative states.

**Explicit Euler** (`explicit_euler_heat`):
- **Scalar `kappa`:** `dt` must satisfy the gate bound and the engine's
  `hex_positivity_max_dt(h, kappa)`. If positivity is not required, only the
  engine's sharp L2 bound `hex_cfl_max_dt(h, kappa, safety=1)` applies.
- **Conductance array:** only the positivity bound of the assembled operator
  applies. The engine's L2 bound covers the constant-coefficient stencil only.

**Crank-Nicolson** (`crank_nicolson_heat`, added for the Phase 1 baseline):

    (I - dt/2 L_c) u^{n+1} = (I + dt/2 L_c) u^n

- Each step is solved with CG (`rtol`, default `1e-12`), warm-started from `u^n`.
- `I - dt/2 L_c` is SPD for every `dt > 0` on both lattices, so the scheme is
  unconditionally L2 stable and second order in time.
- It is NOT positivity preserving. Use explicit Euler with the gate when
  positivity is required.
- On an exact discrete eigenvector with eigenvalue `-lambda_h`, one step
  multiplies by `(1 - dt lambda_h / 2) / (1 + dt lambda_h / 2)`. The tests
  check this factor.
- This is the efficient implicit method that the Phase 1 baseline needs
  (McGreivy & Hakim, rule 2). Explicit Euler, with `dt = O(h^2)`, would be a
  weak baseline.

## 5. Linear solves

`conjugate_gradient` is used for symmetric positive definite operators only.

- Its stopping rule is relative to the right-hand side:
  `||b - A x||_2 <= rtol * ||b||_2`.
  - The rule is decided on the TRUE residual `b - A x`. The recursively
    updated residual is only a cheap pre-check; when the true residual then
    fails the rule, CG restarts from it (residual replacement). (Added
    2026-10-12: the recursive residual drifts from the true one in binary64;
    measured on the Phase 0b MMS levels, it passed `1e-13` while the true
    residual was `1.24e-13` (N=48) to `3.20e-12` (N=192).)
    (Measured 2026-10-12 and rejected: keeping the search direction at this
    point, `p = r_true + (r_true . r_true / r_old . r_old) p` as in residual
    replacement after van der Vorst & Ye, made the MMS solves at `rtol = 1e-13`
    worse: N=96 no longer converged (44750 iterations = maxiter, best
    `4.26e-13`; with the restart 302 iterations, `9.29e-14`) and N=192 ran to
    maxiter at `3.20e-12` instead of stagnating at `2.34e-13`. The
    replacement happens only once the pre-check passes, i.e. at the
    attainable-accuracy floor, which is not where that scheme replaces.)
- It fails closed in these cases:
  - a non-positive curvature `p . A p <= 0` raises `ValueError` (the operator
    is not SPD);
  - exceeding `maxiter` raises `ConvergenceError`;
  - a non-finite iterate raises `ConvergenceError`;
  - a stagnating true residual (no new minimum in 50 consecutive true-residual
    checks; the binary64 floor of `||b - A x||` lies above `rtol * ||b||`)
    raises `ConvergenceError`.
  - (Added 2026-10-12.) Finite entries do not make `||b||_2` representable.
    CG therefore runs on the system scaled by a power of two,
    `max|b / 2^e|` in `[0.5, 1)`; this is exact in binary64 and leaves the
    relative residual unchanged. A solution that is not exactly representable
    at the scale of `b` (overflow or subnormal) raises `ConvergenceError`; an
    `x0` that overflows at that scale raises `ValueError`. Only `b == 0`
    returns the zero vector without iterating.
  - (Added 2026-10-12.) Every stopping decision uses `||r||_2` without
    underflow: `sqrt(r . r)` while `r . r >= 2^-900` (bit-identical there),
    otherwise the norm of `r` scaled by a power of two. If `r . r` underflows
    to `0` while `||r|| > rtol * ||b||`, the recurrence cannot continue and
    `ConvergenceError` is raised; it is never reported as converged.
  - Every `ConvergenceError` from the iteration carries the iterate with the
    smallest true residual seen, that residual and the iteration count as
    diagnostics. They are never a converged result.
- It returns the iteration count and the achieved TRUE relative residual.
- (Added 2026-10-12, E1012-07.) This rule is stopping rule **V1**; it stays the
  default (`stopping="v1"`) and is not reinterpreted. Stopping rule **V2**,
  which also accepts a residual at the rounding floor of its own evaluation
  and then reports `status="at_rounding_floor"`, is specified in
  `spec/HEX_FV_SOLVER_v2.md`.
- An optional initial guess `x0` (warm start) does not change the stopping
  rule. If `x0` already satisfies the rule, CG returns after 0 iterations.

`solve_steady_diffusion` solves `-L_c u = f`:
- **Triangle:** solved with CG.
- **Torus:** refused with `ValueError`. There `-L_c` has the constants as its
  null space, so the problem is singular without an extra constraint (out of
  scope for v1).

## 6. Claim boundary

What this layer is:
- regular lattices only;
- isotropic, scalar `kappa`;
- the two-point flux.

What it is not:
- no irregular or anisotropic meshes, no MPFA/NLTPFA;
- no claim of a discrete maximum principle beyond the positivity statement in
  section 4;
- no performance claim.

Correctness evidence lives in the reference cases, not in this file.
