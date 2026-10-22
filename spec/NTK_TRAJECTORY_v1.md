# NTK_TRAJECTORY_v1 — parameter NTK over declared training checkpoints

**Version:** 1.0
**Status:** PROPOSED — an instrument. Not part of the Phase 1 preregistration
(`spec/PREREGISTRATION_PHASE1_v1.md`, DRAFT) unless the owner adopts it there,
as was done for the spectral audit. It trains no model and makes no learning claim.
**Implementation:** `src/toolkit/ntk_trajectory_hex_v3.py`, built on
`src/toolkit/param_ntk_hex_v3.py` (`spec/PARAM_NTK_v1.md`).
**Tests:** `src/toolkit/test_ntk_trajectory_hex_v3.py`; real backend
`test_gino_adapter_hex_v3.py::test_parameter_ntk_trajectory_of_the_real_gino`
(neuraloperator 2.0.0, runs in the CI `gate` job).
**Issue:** #60 item 2 (NTK mode coupling). `PARAM_NTK_v1` measures the kernel at
one parameter point. This contract measures how that kernel moves during one
training run.

## 1. Definition

For one run, a fixed input `x`, the W-orthonormal basis `phi` (`M x K`,
`phi^T W phi = I`) and declared checkpoints `theta_0, ..., theta_T` at steps
`s_0 < ... < s_T`, the instrument measures at every checkpoint

    Theta_t = PARAM_NTK_v1(model(theta_t), x)        (K x K, symmetric PSD)

and reports:

| Quantity | Definition | Undefined (`null`) when |
|---|---|---|
| `relative_drift` | `||Theta_t - Theta_0||_F / ||Theta_0||_F` | `Theta_0 = 0` |
| `distance_from_initial` | `S(Theta_0, Theta_t)` | either kernel is 0 |
| `distance_from_previous` | `S(Theta_{t-1}, Theta_t)` | first checkpoint, or either kernel is 0 |
| `velocity` | `distance_from_previous / (s_t - s_{t-1})` | as above |
| `offdiag_energy_fraction` | `||Theta - diag(Theta)||_F^2 / ||Theta||_F^2` | `Theta = 0` |
| `effective_rank` | `(tr Theta)^2 / ||Theta||_F^2` (participation ratio, in `[1, K]`) | `Theta = 0` |
| `target_alignment` | `c^T Theta c / (||Theta||_F ||c||^2)` for target coordinates `c = phi^T W y*` | no target, or `Theta = 0` |
| `trace`, eigenvalue min/max, kernel diagonal | from `Theta_t` | never |

`S(A, B) = 1 - <A, B>_F / (||A||_F ||B||_F)` is the kernel distance used by
Fort et al. (NeurIPS 2020, "Deep learning versus kernel learning"); the velocity
divides it by the step difference. The formula is fixed here, not taken on
trust from the paper (the primary source was not retrievable from the agent
session on 2026-10-22). Kernel-target alignment follows Cristianini et al.
(NIPS 2001), normalised by Frobenius norms.

Properties the tests pin:
- For PSD kernels `<A, B>_F = tr(AB) >= 0`, so `S` lies in `[0, 1]`; values a
  rounding error outside are clipped into it. Bitwise equal kernels give exactly 0.
- `S`, `relative_drift`, `effective_rank`, `offdiag_energy_fraction` and
  `target_alignment` are invariant under scaling the output by a constant
  (`Theta -> a^2 Theta`).
- `S`, `relative_drift`, `effective_rank` and the spectrum are invariant under an
  orthogonal change of the basis (`phi -> phi R`, `Theta -> R^T Theta R`).
  `offdiag_energy_fraction` is NOT: mode coupling is a property of the chosen
  basis (for T1 the Laplace eigenbasis), which is why it is reported in it.
- `target_alignment` lies in `[0, 1]` (`c^T Theta c <= lambda_max ||c||^2 <= ||Theta||_F ||c||^2`).
- Every norm is taken after dividing by `max|Theta|` (underflow/overflow of valid
  tiny or huge kernels, as in PARAM_NTK_v1).

## 2. Declared checkpoints

- `declared_steps` is a non-empty, strictly increasing list of non-negative ints
  (bool refused), at most `MAX_CHECKPOINTS = 1000`. The checkpoint steps must
  equal it exactly, in order: a post-hoc subset, superset or reordering is refused.
  Declaring the steps before training is the caller's duty; the instrument
  enforces that the measured sequence is the declared one.
- Each checkpoint state dict must have exactly the model's keys. Tensor entries
  must match dtype and shape (a `float16` checkpoint of a `float32` run is a
  different measurement subject; `load_state_dict` would cast it silently) and be
  finite.
- Non-tensor entries (neuraloperator 2.0.0 stores its constructor kwargs as
  `_metadata`, including `F.gelu` and the `SpectralConv` class) must equal the
  model's own entry in canonical form (section 3); otherwise the checkpoint is of
  another architecture and is refused.
- Two checkpoints with identical bytes are refused: a mislabelled copy (the same
  file saved twice) would fake a frozen kernel.
- Each checkpoint is loaded into a copy of the model, from a shallow copy of the
  state dict (neuraloperator's `load_state_dict` pops `_metadata` from the dict it
  is given). The caller's model and checkpoints are never changed.
- The model must be in eval mode (as in PARAM_NTK_v1); `apply(model, x)` must
  not write into `x` (its digest is recomputed after every checkpoint).

## 3. Binding (digests)

All digests are SHA-256, computed by the instrument from the bytes it measured;
no caller-supplied hash is accepted. Bytes are little-endian (a big-endian host
is refused).

| Field | Covers |
|---|---|
| `state_sha256` (per entry) | every state entry in sorted key order: key (length-prefixed UTF-8), then for a tensor `tensor|dtype|shape|` and the raw element bytes (complex as interleaved real/imag), for other entries `json|` and canonical JSON (sorted keys, no NaN, callables as `{"__qualified_name__": module.qualname}`) |
| `kernel_sha256` (per entry) | the float64 kernel |
| `inputs_sha256` | dtype, shape and bytes of `x` |
| `basis_sha256` | `phi` and `w` as float64 |
| `parameter_signature_sha256` | names, dtypes, shapes, `requires_grad` of parameters; names, dtypes, shapes of buffers |
| `target_sha256` | `c` as given (float64), or `null` |
| `record_sha256` | canonical JSON of the record without this field |

A non-tensor entry without a canonical form is refused: lambdas, local
functions, bound methods (their instance is not covered by a name), arbitrary
objects, NaN.

**Claim boundary.** The digests show WHICH bytes were measured and make the
record tamper-evident. They do not show that the checkpoints came from a real
training run; the record says `"authenticity": "unattested"`.

## 4. Fail-closed guards

Inherited from PARAM_NTK_v1 at every checkpoint: complete parameter set, autograd
against a central difference of all K projected modes, finite 1-D output,
W-orthonormal basis, eval mode. Added here: section 2, the target vector (finite,
length K, not zero), `MAX_MODES = 256`, `run_id` a non-empty string, finite
input. `MAX_CHECKPOINTS` and `MAX_MODES` are operational budget limits (one
checkpoint costs K reverse passes plus the float32 cross-check), not scientific
bounds.

## 5. Evidence (tests)

- **Closed forms.** Linear model `y = A theta + x`: the kernel is constant,
  drift and distance exactly 0. Product `y = a b x`: `Theta = (a^2+b^2) c c^T`,
  drift `|s_t/s_0 - 1|`, distance 0, rank 1, alignment 1. `y = a^2 phi_1 + b phi_2`:
  `Theta = diag(4a^2, 1, 0)`, distance, rank, alignment, drift and velocity in
  closed form. Zero kernels and a zero anchor give `null`, never `inf`.
- **Metamorphic.** Output scale and orthogonal basis change (section 1).
- **Theory oracle (lazy regime).** A two-layer tanh network in NTK
  parameterisation, same task and optimiser: width 1024 shows less than a quarter
  of the drift and a tenth of the distance of width 16 (Jacot et al. 2018,
  Lee et al. 2019: the kernel moves `O(1/sqrt(n))`).
- **Binding.** Digests equal the module's own recomputation; key names, key
  split, dtype, shape and a one-ulp change alter `state_sha256`; tampering alters
  `record_sha256`.
- **Fail-closed.** Every guard of sections 2 and 4 has a negative test.
- **Real backend.** neuraloperator 2.0.0 GINO in float32 mode, two checkpoints
  (initial and after one synthetic Adam step, no T1 data): positive drift,
  distance in `(0, 1]`, `_metadata` bound and the caller's dict kept.
- **Mutation probe** (`scripts/mutation_probe.py`, module `ntk_trajectory_hex_v3.py`):
  every guard and closed-form metric has a mutant that its test class must kill.

## 6. Not claimed

An empirical kernel at one input and a handful of checkpoints is a diagnostic. It
is not the infinite-width NTK, says nothing about generalisation, and the
mode-coupling and alignment numbers are no causal statement about why a model
learns. Which runs, inputs, checkpoints and thresholds (if any) enter Phase 1 is
an owner decision and belongs in the preregistration.
