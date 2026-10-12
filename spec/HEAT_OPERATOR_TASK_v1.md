# Task T1 `hex-heat-operator` v1 — Phase 1 instrument (data, truth, metrics, baseline)

**Version:** 1.0
**Status:** PROPOSED — an instrument for the Phase 1 preregistration
(`spec/PREREGISTRATION_PHASE1_v1.md`, DRAFT). This file fixes how the data,
truth, metrics and mechanistic baseline are computed. It trains no model and
makes no learning claim.
**Implementation:** `src/toolkit/heat_operator_task_hex_v3.py`, built on
`src/toolkit/hex_fv_solver_hex_v3.py` and the Phase 0 modes in
`src/toolkit/reference_case_hex_v3.py`.

## 1. Task

The operator to learn is the heat semigroup `G: u0 -> u(., T)` for
`u_t = kappa * Lap(u)` with `kappa = 1` and `T = 0.01`.

- **In distribution (ID):** unit equilateral triangle, homogeneous Dirichlet
  (`triangle_dirichlet`).
- **Boundary-condition-family OOD:** periodic hex torus (`torus_periodic`).

Inputs and outputs are the values at the lattice nodes, `h = 1/N`.

## 2. Basis, prior and exact truth

**Basis.** All exact eigenfunctions with `lambda <= lambda_cutoff`
(default `600`), sorted by `lambda` and then by key.

- **Triangle:** Lamé modes `(m, n)` with `1 <= m <= n`; `sin` always, `cos`
  only for `m < n`.
  - `(m, n)` and `(n, m)` span the same two-dimensional eigenspace. This was
    checked in-repo: the Gram rank of the four functions is 2.
- **Torus:** plane waves `(p, q)` in the half plane `p > 0`, or `p = 0` and
  `q > 0`, with both parities.
  - The constant mode is excluded.
- Every basis function is an exact eigenvector of the Phase 0 hex operator
  (Phase 0 C1).

**Prior.**
- `a_i = sigma_i * xi_i`, with `xi_i` i.i.d. `N(0, 1)`.
- `sigma_i = (lambda_min / lambda_i)^s`, where `lambda_min` is the smallest
  basis eigenvalue and `s` defaults to `1.0`.

**Randomness.**
- `xi` comes from `numpy.random.default_rng(SeedSequence([dataset_seed, sample_id]))`.
- The coefficients of a sample therefore do not depend on the sample order,
  the other samples or the lattice resolution.

**Truth** (exact continuous solution, sampled at the nodes):

    u0(x)   = sum_i a_i phi_i(x)
    u(x, T) = sum_i a_i exp(-kappa * lambda_i * T) phi_i(x)

**Resolution axis.** The same `sample_id` evaluated at `N = 16, 32, 64, 128`
is the same continuous function.
- Modes with large `|k| h` are resolved poorly on coarse lattices. This is
  intended: it is what the resolution OOD axis measures (ReNO,
  arXiv 2305.19913).
- The truth stays exact on every lattice.

## 3. Metrics (per sample, relative to the truth)

| Metric | Definition |
|---|---|
| `rel_l2` | `||e||_{A} / ||u||_{A}`, with cell-area weights `A = (sqrt(3)/2) h^2`. All cells have equal area, so this is a plain RMS ratio |
| `rel_h1` | `|e|_{1,h} / |u|_{1,h}` |
| `rel_max` | `max|e| / max|u|` |

**The H1 seminorm** is defined through the energy form of the verified
operator, `|v|_{1,h}^2 = -A * v . (L_h v)`.
- On the triangle, the edges to Dirichlet nodes are included.
- Equivalently, it is `sum over edges (v_i - v_j)^2 / sqrt(3)`.
- For an exact eigenvector, `|phi|_{1,h}^2 = lambda_h * ||phi||_A^2`.
- This is not the H1 proxy in `benchmark_hex_v3`, which drops `-1` slots.

**Fail-closed rules:**
- shapes must match and be exactly `[S, M]`; a 1-D field counts as one
  sample, any further axis raises `ValueError` (added 2026-10-12);
- values must be finite;
- a zero truth norm raises `ValueError`;
- `rel_l2` and `rel_max` divide both fields per sample by one power of two
  before differencing and taking norms, which is exact and leaves the ratio
  unchanged, so finite inputs near the binary64 limits neither over- nor
  underflow; a ratio that binary64 cannot represent raises `ValueError`
  instead of returning `inf` or `NaN` (added 2026-10-12);
- materially negative discrete energy raises `ValueError`; only a
  floating-point roundoff band of `64 * eps * max(1, |energy|)` may be
  clamped to zero.

## 4. Manifest and digests

`dataset_manifest` records:
- the schema and task id;
- the domain and `N`;
- `lattice_sha256`, over the domain, `N`, `h`, coordinates and neighbour
  topology in canonical little-endian encodings;
- the canonical configuration and its SHA-256;
- `dataset_seed` and the ordered `sample_ids`;
- the basis (keys and eigenvalues);
- one SHA-256 per sample, over the little-endian float64 bytes of the sampled
  coefficients, `u0`, and `uT`;
- `dataset_sha256`, the SHA-256 of the canonical JSON binding
  `config_sha256`, seed, ordered sample ids, lattice digest and sample
  digests;
- the SHA-256 of the generator module's LF-normalised bytes;
- the numpy version.

**Integrity vs. reproducibility.**
- **Integrity:** `verify_dataset` recomputes the digests of the lattice,
  coefficients and solution arrays. Any content change fails.
- **Reproducibility:** `verify_dataset` regenerates the lattice and samples;
  neighbour topology must match exactly and floating arrays must agree within
  `1e-12` relative max-norm.
- Bitwise equality across machines is NOT promised. numpy's `sin`/`cos` are
  not guaranteed to be correctly rounded and may differ in the last bit
  between CPU dispatch paths.

## 5. Split

Samples carry `trajectory_id = sample_id`. `HeatOperatorDataset` also provides the
`samples` metadata view that the splitter groups on. Splits use
`dataset_split_hex_v3.split_dataset_by_trajectory`, which is leakage-safe by
construction. The preregistration fixes the split seed.

## 6. Mechanistic baseline

- `mechanistic_baseline` maps `u0` at the nodes to `u_h(T)` with
  Crank-Nicolson and warm-started CG (`spec/HEX_FV_SOLVER_v1.md` section 4).
- It runs on the input lattice with `dt = T / steps`.
- It returns the prediction, the total number of CG iterations and the wall
  time (`time.perf_counter`).
- Its error against the truth is second order in `h` and in `dt`. The tests
  check this.
- The cost-accuracy sweep, i.e. the `(N, steps)` grid and the hardware
  record, is fixed by the preregistration (McGreivy & Hakim, rules 1-2).

## 7. Claim boundary

This file defines instruments only:
- no model and no training;
- no statement that any learner is better or worse;
- wall times are hardware-specific measurements, not properties of the
  method.
