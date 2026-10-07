# Reference cases `hex-heat-torus` and `hex-mms-tri` v1 — Reality Anchor Phase 0b

**Version:** 1.0
**Status:** PROPOSED — becomes contractual after merge.

**Decision:** issue #60, comment of 2026-10-06:
- E4 = A: Phase 0 = `hex-heat-tri` + MMS + periodic torus;
- wave plan W1-K08.

**Builds on:**
- `spec/REFERENCE_CASE_HEX_HEAT_TRI_v1.md` (Phase 0a: same lattice, study structure and Celik method);
- `spec/HEX_FV_SOLVER_v1.md` (the code under test);
- `spec/REFERENCE_SPEC_v1.md` (claim ceilings).

**Implementation:**
- `src/toolkit/reference_case_hex_v3.py`;
- evidence writer `scripts/oracles/hex_phase0_reference_oracle.py`.

## 1. Purpose

Phase 0a verified the constant-coefficient operator on one boundary condition
family. Phase 0b adds the two missing axes named in #60:

- **`hex-heat-torus`**: a second boundary-condition family, periodic. It also
  checks:
  - the wrap-around neighbour assembly;
  - the isotropy of the leading error term in arbitrary lattice-compatible
    directions;
  - discrete conservation.
- **`hex-mms-tri`**: the general code paths. These are a variable
  coefficient, a source term and an iterative linear solve, checked by the
  method of manufactured solutions (Salari & Knupp, SAND2000-1444). Unlike
  Phase 0a, the discrete solution is not exact here, so this case exercises
  the full error behaviour of the scheme.

The gate thresholds are the same contract constants as in Phase 0a:
- observed order band `[1.95, 2.05]`;
- coefficient band `[0.99, 1.01]`;
- asymptotic tolerance `0.05`;
- GCI safety factor `1.25`;
- refinement ratio `r > 1.3`;
- at least four levels.

## 2. `hex-heat-torus`

**Problem**
- Equation: `u_t = kappa * Lap(u)` on the hex torus with periods `(1, 0)` and
  `(0, sqrt(3)/2)` (`torus_periodic` lattice, `N` even).
- Modes: `phi = cos(<k, x>)` or `sin(<k, x>)` with `k = 2*pi * (p, 2q/sqrt(3))`,
  for integers `p, q`.
- Eigenvalues:
  - continuous: `lambda = |k|^2 = 4 pi^2 (p^2 + 4 q^2 / 3)`;
  - discrete: `lambda_h`, the closed form of Phase 0a section 4.
  Every such mode is an exact discrete eigenvector, because the lattice
  symbol is the same plane-wave symbol.
- The constant mode `(0, 0, cos)` has `lambda = 0`. The mode `(0, 0, sin)`
  vanishes identically and is rejected.

**Default data**
- `kappa = 1`, `T_final = 0.01`;
- modes, with amplitudes `0.3`, `1`, `1/2`, `1/4`:
  - `(0, 0, cos)` (the mean);
  - `(1, 1, cos)`;
  - `(2, -1, sin)`;
  - `(1, 0, sin)`.
- The wave vectors point at 49.1, -30 and 0 degrees. Reduced by the lattice
  symmetry (rotations by 60 degrees, reflections) into the fundamental
  sector `[0, 30]` degrees, these are 10.9, 30 and 0 degrees: three
  non-equivalent directions, so C3 checks the isotropy of the leading error
  term across the sector. Two modes whose directions differ by a multiple of
  60 degrees would only repeat one check.
- QoI mode: `(1, 1, cos)`, with `lambda = 28 pi^2 / 3`.
- Levels: `N = 16, 32, 64, 128`.

**Criteria**

C1-C9 are those of Phase 0a section 5, with two changes:
- the eigen-residual of a `lambda = 0` mode is normalised by `w * max|phi|`
  instead of `lambda_h * max|phi|`;
- the coefficient ratio skips `lambda = 0` modes.

**C10 (code verification): discrete conservation**
- For every level, `|sum_i (L_h v)_i| <= 1e-12 * w * sum_i |v_i|`.
- `v` is a seeded pseudo-random field.

**ReferenceSpec mapping**

The mapping is that of Phase 0a section 6, except:
- `reference_id = hex-heat-torus/v1`;
- the context of use is the periodic hex torus;
- code verification requires C1-C4 and C10.

## 3. `hex-mms-tri`

**Problem**
- Equation: `-div(kappa grad u) = f` on the unit equilateral triangle, with
  homogeneous Dirichlet BC.
- Manufactured solution: `u* = -1/2 * phi_(1,1,sin) + 1/5 * phi_(1,2,cos)`.
  These are Lamé modes, so `u* = 0` on the boundary and its gradient and
  Laplacian are in closed form.
- Coefficient:
  - `kappa(x, y) = 1 + 0.4 * sin(2 pi x + 0.3) * cos(3 pi y)`;
  - range `[0.6, 1.4]`;
  - its gradient is in closed form.
- Source: `f = -div(kappa grad u*) = kappa * sum_i a_i lambda_i phi_i - grad(kappa) . grad(u*)`.

**Discretisation (`spec/HEX_FV_SOLVER_v1.md`)**
- `-L_c u_h = f` at the nodes, with `c = kappa` at the edge midpoints.
- The system is solved with CG to `rtol = 1e-13`.

**Levels**
- `N = 24, 48, 96, 192`.
- `N` must be divisible by 3, so that the centroid `(1/2, sqrt(3)/6)` is the
  lattice node `(N/3, N/3)` on every level.

**Oracle verification** (all must hold, else the oracle stays at `comparison_only`):

- **O1.** `u*` vanishes on the boundary nodes of the finest lattice:
  `max|u*_boundary| <= 1e-12 * max|u*|`.
- **O2.** The closed-form source agrees with an independent fourth-order
  central difference of the flux, `-div(kappa grad u*)`, at 64 seeded
  interior points. The step is `1e-4` and the agreement threshold is a
  maximum relative error of `1e-8`. This check catches derivation errors in
  `f`.
- **O3.** `kappa > 0` at every edge midpoint of every level. The minimum is
  reported.

**Code verification of the solver** (requires O1-O3):

- **M1.** The observed orders of the relative L2 error and the max-norm error
  against `u*` are in the band for both pairs among the three finest levels.
- **M2.** The edge conductances are exactly symmetric (solver contract check).
- **M3.** The assembled operator passes the positivity gate (M-matrix sign
  structure) on every level.

**Solution verification**
- QoI: `u_h` at the centroid.
- Criteria C5-C9 as in Phase 0a.
- For C9, the exact value is `u*(centroid)`.

**Iterative verification**
- **I1.** On every level CG converges at `rtol = 1e-13`.
- In addition, the iterative-error estimate must be at most `1e-3` times the
  discretisation error:
  `||u_h(rtol=1e-10) - u_h(rtol=1e-13)||_inf <= 1e-3 * ||u_h(rtol=1e-13) - u*||_inf`.

**ReferenceSpec mapping** (two specs):

| Field | `hex-mms-tri/oracle` | `hex-mms-tri/v1` |
|---|---|---|
| `kind` | `manufactured` | `numerical` |
| QoI | manufactured solution and source | `u_h` at the centroid |
| `oracle_verification` | `pass` iff O1-O3 | — |
| `code_verification` | — | `pass` iff O1-O3 and M1-M3 |
| `spatial_verification` + study | — | `pass` iff C5-C9; levels finest to coarsest, `quantity_value = u_h(centroid)` |
| `temporal_verification` | — | `not_applicable`: steady problem |
| `iterative_verification` | — | `pass` iff I1 |
| `numerical_uncertainty` | — | `relative_fraction`, `GCI_fine` |
| derived ceiling when all pass | `mathematical_oracle` | `verified_numerical_reference` |

## 4. Claim boundary

Phase 0b verifies, for the regular hex lattice:
- the periodic assembly and conservation;
- the variable-coefficient two-point flux with midpoint conductances;
- the source path;
- the CG solve;

each for smooth data and an isotropic scalar `kappa`.

It does not verify:
- irregular or anisotropic meshes;
- discontinuous coefficients;
- Neumann or Robin boundaries;
- nonlinear problems.

It validates no physics and assesses no neural operator. Every
`verified_numerical_reference` ceiling holds only for the stated context of
use.
