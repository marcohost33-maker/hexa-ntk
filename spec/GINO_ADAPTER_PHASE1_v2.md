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

## Testing and falsification

- Construct a config with `(R=128, m=63, C=512, L=16)` without allocating neuraloperator; it MUST be rejected due to the combined budget even though every *individual* ceiling passes.
- Accept the default architecture and a documented normal-range configuration. Test digest policy identity.
- Run the entire repository verifier, toolkit coverage, engine gate and pinned real-backend forward/backward/optimizer synthetic smoke on the exact candidate commit.
- The CI smoke is compatibility evidence, not a proof of GPU/D5 safety or scientific generalization.

## Governance

Preregistration D1–D8 remain DRAFT. The existing contradictory required-GINO check versus torch-only CI contract is an unresolved owner choice captured in ADR-002; this v2 resource contract does not authorise either option. No merger or production release without independent review and documented decisions.

Sources: https://neuraloperator.github.io/dev/modules/generated/neuralop.layers.spectral_convolution.SpectralConv.html ; https://github.com/neuraloperator/neuraloperator/blob/2.0.0/neuralop/layers/spectral_convolution.py
