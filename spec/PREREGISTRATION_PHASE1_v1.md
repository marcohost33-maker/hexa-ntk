# Preregistration Phase 1 v1 — GINO vs. the verified mechanistic hex solver on T1

**Status:** DRAFT — NOT binding.

**Approval.** This file becomes binding only after the owner `[M]` approves it.
On approval:
- the status line changes to `APPROVED <date>`;
- the SOURCES.md pin of that exact file version is the preregistration hash
  (16-hex prefix), and the full SHA-256 goes into the approval commit and the
  first run manifest;
- any later change is a new version with a deviation log;
- every result cites the version it follows.

**Until approval, no model may be fit, tuned or evaluated on T1 data.**
Writing training code and running smoke tests on synthetic tensors is allowed.

**Decision basis:** issue #60, comment of 2026-10-06. That comment records the
Reality Anchor, preregistration/OOD protocol, backend and baseline rules
(E4 = A, E5 = A with C second, E6 = A); see also wave plan W1-K08.

**Instruments:**
- `spec/HEAT_OPERATOR_TASK_v1.md`: data, truth, metrics, baseline;
- `spec/HEX_FV_SOLVER_v1.md`: the mechanistic solver;
- `spec/REFERENCE_CASE_HEX_HEAT_TRI_v1.md` and
  `spec/REFERENCE_CASES_HEX_PHASE0B_v1.md`: Phase 0 verification, all green;
- `spec/PHASE1_PROTOCOL_v1.md` (added 2026-10-19): how the numbers of
  sections 4-7 are computed (timing, cost-accuracy curve, Pareto log-log
  interpolation without extrapolation, IQM with the two-stage paired
  bootstrap, decision rules, break-even, run manifest). It fixes
  computation only; the rules below are unchanged.

## 1. Question and hypotheses

**Question.** On task T1 (the hex heat semigroup `u0 -> u(T)`), does GINO from
`neuraloperator` beat the verified mechanistic baseline? Concretely, does it
- reach the baseline's accuracy at lower cost, or
- reach higher accuracy at equal cost,

in distribution and under the two preregistered shifts (resolution and a
composite domain/topology/boundary-condition shift)?

**H0 (default, kept unless rejected by section 6):** no advantage, at equal
accuracy or at equal runtime (McGreivy & Hakim, Nat. Mach. Intell. 6:1256
(2024), rule 1).

**Secondary:** how much accuracy is lost on each OOD axis. Losses are reported
whatever their sign.

## 2. Data (fixed in advance)

**Task configuration.** `HeatOperatorTaskConfig` defaults [D1]:
- `kappa = 1`;
- `T = 0.01`;
- `lambda_cutoff = 600`, which gives 15 triangle modes and 40 torus modes;
- spectral decay `s = 1`.

**Seeds** [D2]:
- `dataset_seed = 1`;
- split seed `0`;
- training seeds `0, 1, 2, 3, 4` (#60: at least 5).

**Samples** [D3]:

| Set | `sample_id`s | Domain | N |
|---|---|---|---|
| train pool | `0..1999` | triangle | 32 |
| ID test | `100000..100499` | triangle | 32 |
| resolution OOD | `100000..100499` (same functions as ID test) | triangle | 16, 64, 128 |
| composite domain/topology/BC OOD | `200000..200499` | torus | 32 |

- The train pool is split 90/10 into train and validation with
  `split_dataset_by_trajectory(val_fraction=0.1, seed=0)`. This split is
  leakage-safe by construction.
- The disjoint ID ranges keep every test set out of the split and out of
  model selection.
- The train resolution `N = 32` and the test resolutions `16, 64, 128` come
  from #60 (ReNO, arXiv 2305.19913).

**Deviation from #60, stated now [D6].** #60 names periodic **and Neumann** as
a BC-family OOD. The v1 torus changes topology/domain geometry as well as the
boundary condition, so it MUST NOT be reported as a pure BC-family effect.
v1 therefore names this axis a **composite domain/topology/BC OOD**. A pure
BC-family test requires the same domain and mesh geometry with an alternate
boundary condition (for example triangle Neumann/Robin); that is deferred to a
v2 written before any such run.

**Recording.** Before training, every `dataset_sha256` goes into the run
manifest (section 7), and `verify_dataset` must report `integrity` and
`reproducible` as true.

## 3. Model and training

**Backend** [D7]:
- `neuraloperator==2.0.0` (MIT, verified 2026-10-07), GINO as shipped.
- Second baseline, not before the GINO results: Transolver (MIT) (#60, E5 = C).
- GAOT is a clean-room build from the paper only, in Phase 2 (#60, E6 = A).
- The dependency goes into a separate optional requirements file, not into
  `requirements-dev.txt`. CI stays torch-only.

**Inputs and outputs.**
- Input: node coordinates (`HexLattice.coords`) and `u0` at the nodes.
- Output: `u(T)` at the same nodes.
- Precision: float32 for training; evaluation metrics in float64.

**Training.**
- Loss: relative L2, the primary metric.
- Optimiser: Adam.
- Schedule: cosine.
- Epoch budget: fixed per configuration [D4].

**Model selection** [D4]. A grid of at most 8 configurations is fixed in the
approved version, over width, latent grid resolution and learning rate.
- Each configuration is trained with seed 0.
- The configuration with the lowest mean **validation** `rel_l2` is selected.
- The selected configuration is then trained with seeds 0-4.
- No other configuration, schedule or early-stopping rule may be tried after
  approval.

**Test sets.** Each test set is evaluated exactly once per seed, after
selection. Nothing is tuned after a test set has been seen.

**Compute** [D5]. Hardware, framework versions, dtype/AMP and CUDA go into the
run manifest. If the agreed budget runs out, the runs are stopped and reported
as incomplete. They are not extended selectively.

## 4. Baseline (McGreivy & Hakim, rules 1 and 2)

**Method.** The mechanistic baseline is `mechanistic_baseline`:
Crank-Nicolson with warm-started CG on the evaluation lattice. It was verified
in Phase 0 and in the instrument tests (second order in `h` and `dt`). It is
the efficient implicit method. Explicit Euler, with `dt = O(h^2)`, would be a
weak baseline (rule 2).

**Cost-accuracy curve.** For each evaluation lattice `N_eval` in
`16, 32, 64, 128`, the baseline runs with `steps` in `1, 2, 4, 8, 16, 32, 64`
on the same test inputs. Per point it records the mean `rel_l2` and the median
wall time per sample.

**Timing protocol.**
- Primary algorithmic comparison: GINO and the mechanistic baseline run on the
  **same CPU machine and thread policy**. GPU-vs-CPU numbers may be reported
  only as a separate deployment/system comparison, never as an algorithmic
  speed claim.
- single process; fixed thread counts recorded in the manifest;
- 3 warm-up runs, then 10 measured repeats; report median and a 95 % bootstrap
  interval for latency;
- CUDA timing, when reported secondarily, synchronizes immediately before and
  after the timed region (or uses CUDA events) so asynchronous execution cannot
  under-report latency;
- batch size 1 is the latency comparison. Batch 500 is reported separately as
  throughput and is not substituted for latency.

**GINO cost.** Recorded separately:
- inference time per sample, at batch size 1 and at batch size 500;
- training time;
- data-generation time.

The total-cost view reports the break-even number of queries at which training
plus data cost is amortised.

## 5. Metrics and aggregation

**Per sample:**
- primary: `rel_l2`;
- secondary: `rel_h1` (energy form) and `rel_max` (`spec/HEAT_OPERATOR_TASK_v1.md`
  section 3).

**Per seed and test set:** the mean and the median over the samples.

**Across seeds and samples:**
- the decision statistic is the IQM of the five seed-level mean errors;
- its 95 % interval uses a two-stage paired bootstrap with 10 000 resamples:
  resample training seeds, then sample ids; for the resolution axis the same
  resampled sample ids are used jointly at N=16/32/64/128 so the known pairing
  is preserved;
- no CI is pooled across different test sets for a per-test decision;
- every individual seed and every test set is reported. This adapts the
  IQM/bootstrap principle to the clustered PDE design instead of treating
  correlated seed-by-sample cells as independent.

**Negative results** (no advantage, OOD failure) are reported with the same
prominence as positive ones (AGENTS.md: first-class).

**Secondary measurement: spectral audit** (owner decision 2026-10-13, `spec/SPECTRAL_AUDIT_v1.md`).
- Every trained GINO seed is audited at base point `0` and at the first 8 sample ids
  of each test set, against oracle (iii) (`exp(-kappa lambda T)`, T1 truth).
- Reported per mode: `diag_rel_error`, `coupling_fraction`, `out_of_span`; the
  mechanistic baseline is audited at the same `(N, steps)` next to it.
- The POD-ridge baseline (`spec/SPECTRAL_AUDIT_v1.md` section 6) is fit on the training
  split only and reported as an additional baseline, not a decision baseline.
- No decision rule in section 6 depends on these numbers.

## 6. Decision rules (per test set)

**Equal runtime.**
- `t_NO` is GINO's median inference time per sample at batch size 1, with
  batch 500 reported as well.
- `e_base(t_NO)` is the baseline error at `t_NO`, interpolated log-log on
  the **Pareto-efficient** part of its cost-accuracy curve. No extrapolation is
  allowed; if `t_NO` lies outside the measured range, no equal-runtime claim
  is made.
- GINO is "more accurate at equal runtime" only if the upper 95 % CI bound of
  its IQM `rel_l2` is below `e_base(t_NO)`.

**Equal accuracy.**
- `t_base(e)` is the baseline time needed to reach error `e`, from the
  Pareto-efficient measured curve only; no extrapolation is allowed.
- GINO is "faster at equal accuracy" only if `t_NO < t_base(e)`, where `e` is
  the upper 95 % CI bound of GINO's IQM `rel_l2`.
- If the baseline cannot reach `e` on that lattice, the comparison is reported
  as "baseline floor" and no speed claim is made.

**Otherwise** H0 stands: "no advantage". Every claim names its test set and
its N. No claim extends beyond T1. No physics claim is made.

## 7. Run manifest and deviations

Every run writes a manifest with:
- the git SHA;
- this file's version and SHA-256;
- the `dataset_sha256` of every set;
- the split ids;
- the backend and all versions;
- the config digest;
- the seed set;
- hardware, dtype/AMP and CUDA;
- the Phase 0 ReferenceSpec IDs;
- the metric version (the instrument spec SHA-256);
- the result digests.

This follows the HexaNTK Technischer Gesamtbericht of 2026-10-04.

Deviations go into `results/` with a reason, before the affected numbers are
reported.

## 8. Owner decisions before approval

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Task configuration (section 2) | keep the defaults; T1 is non-trivial (`mean ||uT - u0|| / ||u0|| = 0.64`) yet cheap |
| D2 | Seeds | data 1, split 0, training 0-4 |
| D3 | Sample counts | train pool 2000 (1800/200), 500 per test set |
| D4 | GINO grid and epoch budget | grid of 8 configurations, fixed in the approved version |
| D5 | Hardware and compute budget | name the machine; GPU if available, otherwise CPU (`N = 32`, 465 nodes) |
| D6 | Torus shift naming | treat it as composite domain/topology/BC OOD in v1; reserve "BC-family" for same-domain Neumann/Robin in v2 |
| D7 | Add `neuraloperator==2.0.0` as an optional dependency | accept (MIT) |
| D8 | Optional licence request to the GAOT/GINOT authors (#60) | external action, Marco's call |

Owner `[M]`, 2026-10-13: D1-D8 accepted as recommended; T1 is a calibration study (a negative
result is an accepted outcome); the spectral audit is adopted as a secondary measurement
(section 5); the licence request (D8) is drafted for the owner to send. Still open before the
status line can change: the concrete D4 grid (at most 8 configurations) and the named D5 machine.

2026-10-15 (owner decision E1015M-01: enter D4/D5 now, only with content from the approved
source). The approval of 2026-10-13 (E1013M-01) rests on the research synthesis of 2026-10-13,
question 1 (Vero research archive, not in this repo:
`Recherche/2026-10-13_neuronale-operatoren-und-hexa-ntk/01_SYNTHESE.md`, line 33, recommendation
1(d) "D5 = CPU"; rationale in `04_T2_hexa-ntk.md`, line 118: the default GINO spectral weights
need about 11 MB, `N = 32` has 465 nodes).
- **D5, filled:** compute = CPU, no GPU.
- **D5, still open:** the named machine. The source names no machine.
- **D4, still open:** the source contains no grid and no epoch budget.

2026-10-19 (decision support for D4, no decision): `scripts/oracles/gino_radius_tie_survey.py`
measured with the adapter's tie rule (`spec/GINO_ADAPTER_PHASE1_v2.md`, amendment 2026-10-15)
that the adapter default (`latent_resolution = 32`, `radius_factor = 2.5`) carries 4 radius
ties on each of the resolution-axis lattices `N = 64` and `N = 128`, so those evaluations
would be refused. Tie-free on all five lattices of section 2 are `latent_resolution` 16, 24
or 48 with `radius_factor` 1.5, 2.25, 2.5 or 2.75 (12 of 48 candidates;
`results/2026-10-19_gino_radius_tie_survey.json`). The D4 grid must name a tie-free pair for
every evaluation lattice; the adapter default is unchanged until the owner decides.

The status line therefore stays DRAFT; nothing here allows a fit on T1 data.

## 9. Claim boundary

- This draft fixes the protocol only. It produces no result.
- Every later result holds for T1, its preregistered axes and the recorded
  hardware only.
