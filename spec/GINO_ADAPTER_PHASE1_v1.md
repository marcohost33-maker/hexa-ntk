# GINO Adapter Phase 1 v1 — Backend/Geometry Contract

**Status:** implementation-ready adapter contract; no model fit is authorised by this file.
**Backend pin:** `neuraloperator==2.0.0`.

## Purpose

HexaNTK does not reimplement GINO. The adapter binds the verified T1
`hex-heat-operator` arrays and `HexLattice` geometry to the external
NeuralOperator GINO API. Training remains behind
`PREREGISTRATION_PHASE1_v1.md`.

## Tensor contract

For a lattice with `M` nodes and batch size `B`:

- `input_geom`: `(1, M, 2)`;
- `output_queries`: `(1, M, 2)`;
- `x`: `(B, M, 1)`;
- `latent_queries`: `(1, R, R, 2)`.

GINO 2.0.0 supports batching when input/output geometry is shared across the
batch; T1 satisfies that condition within each resolution/domain batch.

The latent rectangle uses the physical coordinate frame
`x in [0,1]`, `y in [0,sqrt(3)/2]`. The same frame is used for ID,
resolution OOD and torus evaluation; no data-dependent rescaling is allowed.

## Fixed geometric choices

- `gno_coord_dim = 2`;
- Open3D neighbour search is disabled because the task is genuinely 2-D;
- torch-scatter is not required by the adapter;
- GNO radius is `radius_factor * h_train` and is fixed from the N=32
  triangle training lattice. It MUST NOT be retuned on an OOD test set;
- the v1 adapter does not add periodic image copies. Therefore torus results
  remain the preregistered **composite domain/topology/BC OOD** stress test,
  not evidence for a pure periodic-boundary or periodic-equivariance effect.

## Spectral/Nyquist guard

For latent resolution `R`, each retained mode count MUST satisfy
`n_modes < R/2`. Invalid configurations fail before model construction.

## Evidence and reproducibility

`GinoPhase1Config.canonical()` records the backend pin, geometry semantics and
architecture-side choices. Its SHA-256 digest belongs in every Phase-1 run
manifest. Optimizer, learning-rate, epoch and seed policy remain separate
preregistered training metadata.

The ordinary CPU CI tests the adapter contract without installing the optional
backend. `.github/workflows/gino-smoke.yml` installs the pinned backend and
executes a real forward smoke test on synthetic data. Neither path trains,
tunes, or opens held-out T1 test data.

## Claim boundary

A successful adapter smoke test proves only API/shape compatibility. It does
not establish accuracy, generalisation, speed advantage, physical validation,
or a trained model.
