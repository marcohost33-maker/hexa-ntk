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
  `spec/REFERENCE_CASES_HEX_PHASE0B_v1.md`: Phase 0 verification, all green.

## 1. Question and hypotheses

**Question.** On task T1 (the hex heat semigroup `u0 -> u(T)`), does GINO from
`neuraloperator` beat the verified mechanistic baseline? Concretely, does it
- reach the baseline's accuracy at lower cost, or
- reach higher accuracy at equal cost,

in distribution and under the two preregistered shifts (resolution and
boundary-condition family)?

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
| BC-family OOD | `200000..200499` | torus | 32 |

- The train pool is split 90/10 into train and validation with
  `split_dataset_by_trajectory(val_fraction=0.1, seed=0)`. This split is
  leakage-safe by construction.
- The disjoint ID ranges keep every test set out of the split and out of
  model selection.
- The train resolution `N = 32` and the test resolutions `16, 64, 128` come
  from #60 (ReNO, arXiv 2305.19913).

**Deviation from #60, stated now [D6].** #60 names periodic **and Neumann** as
the BC-family OOD. The v1 solver layer has no Neumann lattice, so v1 tests
periodic only. Adding Neumann means a v2 of this file, written before any run.

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
- same machine;
- single process;
- 3 warm-up runs, then the median of 5 repeats.

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

**Across seeds:**
- the IQM, plus a 95 % stratified bootstrap confidence interval (strata = test
  set; 10 000 resamples; Agarwal et al., NeurIPS 2021);
- every individual seed and every test set is reported.

**Negative results** (no advantage, OOD failure) are reported with the same
prominence as positive ones (AGENTS.md: first-class).

## 6. Decision rules (per test set)

**Equal runtime.**
- `t_NO` is GINO's median inference time per sample at batch size 1, with
  batch 500 reported as well.
- `e_base(t_NO)` is the baseline error at `t_NO`, interpolated log-log on its
  cost-accuracy curve.
- GINO is "more accurate at equal runtime" only if the upper 95 % CI bound of
  its IQM `rel_l2` is below `e_base(t_NO)`.

**Equal accuracy.**
- `t_base(e)` is the baseline time needed to reach error `e`.
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
| D6 | BC-family axis periodic only in v1 | accept (Neumann in v2), or build a Neumann lattice first |
| D7 | Add `neuraloperator==2.0.0` as an optional dependency | accept (MIT) |
| D8 | Optional licence request to the GAOT/GINOT authors (#60) | external action, Marco's call |

## 9. Claim boundary

- This draft fixes the protocol only. It produces no result.
- Every later result holds for T1, its preregistered axes and the recorded
  hardware only.
