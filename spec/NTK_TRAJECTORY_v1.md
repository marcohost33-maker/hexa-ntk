# NTK_TRAJECTORY_v1 — parameter NTK over declared training checkpoints

**Version:** 1.0 (draft, amended before merge on 2026-10-23 for HNT-R25-01:
identical checkpoint bytes are measured and marked instead of refused, section 2.1)
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
- `relative_drift` (HNT-R86-02, cross-family GPT-6 R86): where the formula above
  overflows on finite kernels (`Theta_t - Theta_0`, its division by a tiny anchor
  scale, or the squares inside the norm), the magnitude is factored out
  (`d = Theta_t/2 - Theta_0/2`, `m = max|d|`, drift `= (m / max|Theta_0|) * 2 ||d/m||_F
  / ||Theta_0 / max|Theta_0| ||_F`). A drift beyond the binary64 range is refused with
  `NTKTrajectoryError`, never written as `inf`. Wherever the formula above was finite
  its value is unchanged, bit for bit.

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
- Two checkpoints with identical bytes are measured, not refused (section 2.1).
- Each checkpoint is loaded into a copy of the model, from a private snapshot of
  the state dict (neuraloperator's `load_state_dict` pops `_metadata` from the dict it
  is given). The caller's model and checkpoints are never changed.
- Preflight binding (HNT-R85-01, cross-family GPT-6 R85-R87): all digests are taken
  before the first measurement. Each state is then read once into a snapshot
  (tensors detached and cloned, other entries deep-copied), the snapshot's digest must
  equal the preflight digest, and only the snapshot is loaded. A checkpoint changed in
  between (by `apply` at an earlier step, or a `Mapping` whose reads differ) is
  refused. This closes a change between preflight and load; it does not make the copy
  atomic against a writer running concurrently with the call, so exclusive ownership
  of the checkpoints during the call is the caller's duty.
- The model must be in eval mode (as in PARAM_NTK_v1); `apply(model, x)` must
  not write into `x` (its digest is recomputed after every checkpoint).

### 2.1 Repeated states (HNT-R25-01)

Byte equality of two checkpoints proves neither a mislabelled copy nor a
genuinely stationary run: a learning rate of 0, frozen parameters or a plateau
in a real run produce the same bytes as a file saved twice. Refusing equal bytes
excluded valid zero-drift trajectories (cross-family GPT-6 R25, HNT-R25-01, P2);
the first draft of this contract did that and was amended before merge.

- Identical state digests are allowed. Every checkpoint, also a repeated one, is
  loaded and measured again under the full PARAM_NTK_v1 oracle (autograd against
  central differences); its kernel is never copied from an earlier entry. An
  oracle failure at any checkpoint aborts the trajectory, so no record exists.
- Each entry carries three markers, computed from the digests:

| Field | Value |
|---|---|
| `state_unchanged_since_previous` | `true` iff `state_sha256` equals that of the previous entry (`false` for the first) |
| `repeated_state_digest_of` | the earliest earlier step with the same `state_sha256`, else `null` |
| `stationary_drift` | `null` if the state changed since the previous entry; otherwise `"measured_zero"` if the re-measured kernel equals the previous one bit for bit (`kernel_sha256` equal, so `‖Theta_t - Theta_{t-1}‖_F = 0` exactly and `distance_from_previous` is 0, or `null` for the zero kernel), else `"not_measured"` |

- `"not_measured"` means identical bytes gave a different kernel: the backend did
  not reproduce, and the zero is not established. The numeric drift fields still
  report what was measured.
- A state that returns to an earlier one (A, B, A) is not unchanged since the
  previous checkpoint; `repeated_state_digest_of` names the earlier step.
- The step guards are unchanged: declared, strictly increasing, measured in order.
- **A copy under a wrong step number is a provenance defect.** It is refused at
  the level of an authenticated checkpoint history (signed or otherwise attested
  save events per step, future work), not by hashing; this instrument attests no
  provenance (`"authenticity": "unattested"`, section 3).
- `check_trajectory_record(record)` refuses a record whose markers disagree with
  its own `state_sha256`/`kernel_sha256` sequence (recomputed independently of
  the producer), a missing or mistyped marker (`1` is not `true`), a
  `"measured_zero"` with nonzero `distance_from_previous`, another policy, an
  authenticity other than `"unattested"`, steps that differ from `declared_steps`,
  or a `record_sha256` that does not match. It checks internal consistency only:
  `record_sha256` is no signature, anyone can recompute it after an edit.

## 3. Binding (digests)

All digests are SHA-256, computed by the instrument from the bytes it measured;
no caller-supplied hash is accepted. Bytes are little-endian (a big-endian host
is refused).

| Field | Covers |
|---|---|
| `state_sha256` (per entry) | every state entry in sorted key order: key (length-prefixed UTF-8), then for a tensor `tensor|dtype|shape|` and the raw element bytes (complex as interleaved real/imag), for other entries `json|` and canonical JSON (sorted keys, no NaN, callables as `{"__qualified_name__": module.qualname}`; a data dict carrying the key `__qualified_name__`, a non-`str` dict key and nesting deeper than 64 levels are refused, HNT-R86-01; an entry whose canonical JSON exceeds 1 MiB (1 048 576 bytes), that holds more than 100 000 values (dict keys count) or whose str characters exceed 1 MiB in total is refused, the last two before encoding, R87-P2) |
| `kernel_sha256` (per entry) | the float64 kernel |
| `inputs_sha256` | dtype, shape and bytes of `x` |
| `basis_sha256` | `phi` and `w` as float64 |
| `parameter_signature_sha256` | names, dtypes, shapes, `requires_grad` of parameters; names, dtypes, shapes of buffers |
| `target_sha256` | `c` as given (float64), or `null` |
| `record_sha256` | canonical JSON of the record without this field |

A non-tensor entry without a canonical form is refused: lambdas, local
functions, bound methods (their instance is not covered by a name), arbitrary
objects, NaN. Refused as well since HNT-R86-01 (cross-family GPT-6 R86): entries
JSON would encode like a different subject, namely a data dict
`{"__qualified_name__": name}` (the bytes of the callable `name`) and dict keys that
are not `str` (`1`, `True`, `None`, `2.5` encode like the strings `"1"`, `"true"`,
`"null"`, `"2.5"`). Every entry that was bindable before keeps its bytes, so v1
digests already written stay valid. Not distinguished, deliberately: a list and a
tuple are the same JSON array (they configure the same constructor). A qualified
name does not attest the callable's implementation.

**Claim boundary.** The digests show WHICH bytes were measured and make the
record tamper-evident. They do not show that the checkpoints came from a real
training run; the record says `"authenticity": "unattested"`. Equal digests show
equal bytes, nothing about how the bytes arose (section 2.1).

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
- **Repeated states (section 2.1).** Positive control: a real `torch.optim.SGD`
  step with `lr = 0` gives identical bytes, both checkpoints are measured, the
  second is marked and its drift, distances and velocity are exactly 0
  (`"measured_zero"`). Control: the same step with `lr > 0` changes the bytes,
  is not marked, and drifts by the closed form of the product model. Markers for
  A-B-B, A-A-A, A-B-A and the zero kernel; an injected last-bit kernel difference
  gives `"not_measured"`; an injected oracle failure at a repeat gives no record.
  `check_trajectory_record` refuses resealed records with a removed, wrong or
  mistyped marker.
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
