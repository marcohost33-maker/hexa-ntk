# GINO Adapter Phase 1 v2 — operational resource and provenance envelope

**Status:** IMPLEMENTED CANDIDATE in Wave20 PR; owner and independent science review pending.
**Lineage:** v1 is preserved unchanged. This v2 narrows the operationally accepted configuration set; it is **not** a change to scientific data, metrics, training hyperparameters, acceptance thresholds or preregistration D1–D8.

## Hardening contract (fail-closed before allocation)

- Training construction requires the canonical Dirichlet triangle, `N=32`.
- A supplied evaluation lattice must be canonical, with exact shape/dtype/h/topology and `3 <= N <= 512` (periodic torus needs even `N>=4`). Reject invalid arrays and unbounded sizes before any O(N²) reference assembly.
- Architecture budgets: `8 <= latent_resolution <= 128`, `1 <= hidden_channels <= 512`, `1 <= n_layers <= 16`, `0 < radius_factor <= 16`, `n_modes < latent_resolution/2`.
- **Joint spectral budget:** estimate complex64, dense unfactorized spectral weights with the pinned neuraloperator 2.0.0 real-FFT last-axis reduction. Conservatively round an odd `n_modes` upward to an even count `m`, and require:

  `B_spectral = 8 * n_layers * hidden_channels² * m * (m/2 + 1) <= 268435456 bytes`.

  This is a conservative allocation *gate* for the FNO spectral weights, **not** an upper bound for the whole GINO model or training peak RSS/VRAM: GNO/MLP weights, optimizer states, activations, input grids, workspace buffers and framework allocator overhead are not counted. Run-time OOM handling, device memory monitors and D5 profiling remain mandatory.
- Resource-policy identity `GINO_ADAPTER_PHASE1_v2` enters `GinoPhase1Config.canonical()` and its SHA-256. Existing historical v1 digests remain archived, not silently rewritten. No prior manifests can be treated as v2-identical.
- Exact unchanged geometry frame, GNO radius fixed from N=32 train h, pinned backend version and actual model dtype/device validation are retained.
- dtype contract (2026-10-13, measured with the real neuraloperator 2.0.0 wheel): a neuraloperator GINO runs in float32/complex64 only. The backend allocates `segment_csr` output in the default dtype (`neuralop/layers/segment_csr.py:76`) and the spectral buffer as complex64 (`spectral_convolution.py:444`), so a float64 GINO fails inside the backend and would be float32-accurate in the spectral path anyway. The adapter therefore rejects float64 for a `neuralop.*` model, and for any model complex weights must match the real dtype (float32 with complex64, float64 with complex128); `model.double()` leaves complex64 spectral weights and is rejected.
- Radius ties are a separate error class, not rounding: at `radius_factor=5.0` eight node distances lie exactly on the GNO radius, so float32 and float64 build different neighbour graphs (1984 vs 1985 pairs) and outputs differ by about 1e-2 relative. The Phase-1 default (2.5 h) has no ties; float32 vs float64 output differs there by 3.3e-7 relative. Tolerances must not be widened to absorb a tie.
- **Amendment 2026-10-15 (runtime enforcement).**
  - float64 refusal: the check covers every submodule (`model.modules()`, which includes the model itself), so a caller-side wrapper outside `neuralop.*` around a float64 neuraloperator module is rejected too. Before, only `type(model).__module__` was checked; with the real 2.0.0 backend such a wrapper failed deep in the backend with a RuntimeError.
  - Radius ties: `build_gino` (training lattice) and `prepare_gino_batch` (every lattice it is given) count node-latent distances `d` with `|d - r| <= 1e-6 r` in float64, `r = radius_factor / 32`, and raise ValueError on any tie. Measured with this rule (latent grid as in `prepare_gino_batch`): N=32, latent 8 and latent 32: 8 ties at `radius_factor=5.0`, 0 at 2.5. **With the default configuration (latent 32, 2.5) the resolution-axis lattices N=64 and N=128 have 4 ties each** (N=16 has 0); those evaluations are refused until the owner fixes the radius or latent grid for them. This is an open owner decision, not a tolerance to widen.

- **Amendment 2026-10-19 (radius-tie survey, decision support only).** `scripts/oracles/gino_radius_tie_survey.py`
  applies the tie rule above to 6 latent resolutions x 8 radius factors on the five preregistered lattices
  (triangle `N = 16, 32, 64, 128`, torus `N = 32`). Measured (`results/2026-10-19_gino_radius_tie_survey.json`):
  the default (latent 32, 2.5) has 4 ties on `N = 64` and on `N = 128`; latent 8, 32 and 64 tie on at least one
  lattice for every radius factor tried; latent 16, 24 and 48 with `radius_factor` 1.5, 2.25, 2.5 or 2.75 are
  tie-free on all five lattices (12 of 48 candidates); `radius_factor` 2.0, 3.0 and 4.0 tie on the torus for
  every latent resolution. The default stays as it is; choosing a pair is the owner's D4 decision.
  *(As a recommendation superseded by the amendment of 2026-10-20: three of these twelve build different
  float32 and float64 graphs.)*

- **Amendment 2026-10-20 (float32 graph check, review finding F-S4; decision support only).** "Tie-free"
  under the rule above is not "float32 and float64 build the same neighbour graph". The survey now rebuilds
  both GNO graphs as the pinned backend builds them (neuraloperator 2.0.0 `native_neighbor_search`, used
  because `build_gino` sets `gno_use_open3d=False`: `torch.cdist(queries, data)`, edge iff `<= radius`;
  input GNO latent grid against nodes, output GNO nodes against latent grid; geometry of
  `prepare_gino_batch`) in float32 and in float64, and checks every graph against the backend's own search
  (960 graphs, 0 mismatches). Measured (`results/2026-10-20_gino_radius_tie_survey_float32-graphs.json`,
  Windows, Python 3.12.10, torch 2.6.0+cpu, neuraloperator 2.0.0, source 69740d1):
  - latent 48 with `radius_factor` 1.5, 2.25 and 2.5 is tie-free but its float32 and float64 graphs differ
    by 8, 8 and 3 edges (both directions; on `N = 128`, for 2.25 also on `N = 64`; minimal margin
    2.3e-6 to 6.3e-6 r). These three are **no longer recommended**. Positive control: the default
    (32, 2.5) differs by 4 edges per direction on `N = 64` and on `N = 128`.
  - Tie-free **and** flip-free on all five lattices here: latent 16 and 24 with `radius_factor` 1.5, 2.25,
    2.5 or 2.75, and latent 48 with 2.75 (9 of 48 candidates).
  - **Amendment 2026-10-20 (second review, Equalita R2 @35a5a45):** "flip-free here" is a statement about
    this machine's summation order. Emulating the float32 path of `torch.cdist` under 864 summation orders,
    (48, 2.75) flips under 248 of them (pairwise summation in natural order already flips 8 edges); the other
    eight flip under none, the default (32, 2.5) under all 864 (positive control). Linux/CUDA/OpenBLAS
    orders are not measured. **(48, 2.75) is therefore no longer recommended; only the remaining 8 (latent 16
    and 24 with 1.5, 2.25, 2.5 or 2.75) may be named in the D4 grid.** The probe is a review artifact, not a
    gate result; the owner questions below stay open.
  - The float32 error of `torch.cdist` near the radius (largest `|d32 - d64| / r` over pairs with
    `|d - r| <= 0.1 r`) is 1.7e-5 to 6.9e-5 r on these nine, 17 to 69 times the tie band of 1e-6 r.
    Minimal margin over that error: (16, 1.5) 3.53, (16, 2.75) 2.20, (16, 2.25) 2.10, (24, 2.5) 1.32,
    (24, 1.5) 1.14, (16, 2.5) 1.11, (24, 2.25) 0.82, (24, 2.75) 0.80, (48, 2.75) 0.32. Below 1 the
    candidate is flip-free on this machine only through the rounding of this torch/BLAS; flips on Linux/CI
    were not measured.
  - **Open owner questions, not decided here:** which margin-over-error factor the D4 choice must meet, and
    whether the adapter's tie band (1e-6 r) should be widened towards the measured float32 error so that the
    adapter refuses every pair that can flip. The default and the adapter are unchanged.

## Testing and falsification

- Construct a config with `(R=128, m=63, C=512, L=16)` without allocating neuraloperator; it MUST be rejected due to the combined budget even though every *individual* ceiling passes.
- Accept the default architecture and a documented normal-range configuration. Test digest policy identity.
- Run the entire repository verifier, toolkit coverage, engine gate and pinned real-backend forward/backward/optimizer synthetic smoke on the exact candidate commit.
- The CI smoke is compatibility evidence, not a proof of GPU/D5 safety or scientific generalization.

## Governance

Preregistration D1–D8 remain DRAFT. The existing contradictory required-GINO check versus torch-only CI contract is an unresolved owner choice captured in ADR-002; this v2 resource contract does not authorise either option. No merger or production release without independent review and documented decisions.

Sources: https://neuraloperator.github.io/dev/modules/generated/neuralop.layers.spectral_convolution.SpectralConv.html ; https://github.com/neuraloperator/neuraloperator/blob/2.0.0/neuralop/layers/spectral_convolution.py
