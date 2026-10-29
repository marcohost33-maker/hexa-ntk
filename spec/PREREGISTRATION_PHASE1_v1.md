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
  is made. (Clarified 2026-10-20: "measured range" here and in the
  equal-accuracy rule is the range of the Pareto-efficient part;
  `spec/PHASE1_PROTOCOL_v1.md` section 5.)
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
*(Candidate list superseded on 2026-10-20, next paragraph.)*

2026-10-20 (review finding F-S4, decision support, no decision): tie-free is not enough. The same
survey now compares the float32 and float64 neighbour graphs that the pinned backend
(neuraloperator 2.0.0, native neighbour search) builds; latent 48 with `radius_factor` 1.5, 2.25 or
2.5 is tie-free but builds different graphs on `N = 128` (2.25 also on `N = 64`). Tie-free and
flip-free on all five lattices are latent 16 and 24 with `radius_factor` 1.5, 2.25, 2.5 or 2.75, and
latent 48 with 2.75 (9 of 48; `results/2026-10-20_gino_radius_tie_survey_float32-graphs.json`,
`spec/GINO_ADAPTER_PHASE1_v2.md` amendment 2026-10-20, which also lists each candidate's margin
over the measured float32 distance error). Flip-free means flip-free on the measuring machine: a
second review (Equalita R2, 2026-10-20) found that (48, 2.75) flips under 248 of 864 emulated float32
summation orders, the other eight under none; it is therefore not recommended, leaving 8 candidates
(latent 16 and 24). The D4 grid must name a tie-free and flip-free pair for every evaluation lattice. The required margin and a wider tie band in the adapter are open owner
questions.

2026-10-29 (approval candidate, no decision). Prepared so that the owner can approve D4 and D5 in
one step; this paragraph decides nothing and the status line stays DRAFT. No model was fit, tuned or
evaluated on T1 data: the cost numbers come from synthetic standard-normal tensors on the `N = 32`
lattice geometry (line 14 allows this), measured by `scripts/oracles/gino_epoch_cost_synthetic.py`
(`results/2026-10-29_gino_epoch_cost_synthetic.json`, source `35f82dc`, command
`python scripts/oracles/gino_epoch_cost_synthetic.py --threads 4 --output results/2026-10-29_gino_epoch_cost_synthetic.json`).
`src/toolkit/test_preregistration_candidate_hex_v3.py` recomputes every number of this paragraph from
the two JSON files it cites.

*Radius.* `radius_factor` is not a grid axis (lines 111-114), so one value serves both latent
resolutions and counts with the smaller of its two margins. Values: field `margin_over_float32_error`
of `results/2026-10-20_gino_radius_tie_survey_float32-graphs.json` (minimum over the five lattices),
read from the file, not recomputed:

| `radius_factor` | latent 16 | latent 24 | minimum |
|---|---|---|---|
| `1.5` | 3.53 | 1.14 | 1.14 |
| `2.25` | 2.10 | 0.82 | 0.82 |
| `2.5` | 1.11 | 1.32 | 1.11 |
| `2.75` | 2.20 | 0.80 | 0.80 |

Proposed: `radius_factor = 1.5` (largest minimum). For the owner: `2.5` is close (1.11);
with `1.5` and latent 16 some node has only 1 latent point(s) within the radius (`2.5`: 3;
`neighbourhoods` in the cost JSON, all five lattices); the cost numbers below hold for this radius only. Under
the order-independent budget of `spec/PHASE1_PROTOCOL_v1.md` section 11 both pairs are
`frozen_order_required` (margin over budget 0.32 for latent 16 and 0.14 for latent 24), so the D5
machine below is the frozen hardware/kernel order the run manifest must pin.

*D4 grid (proposal).* Width {32, 64} x latent resolution {16, 24} x learning rate {1e-3, 3e-4} = 8
configurations, `n_layers = 4`, `n_modes` = 12/32 of the latent resolution as in the adapter default
(6 and 9; the default 12 is refused for both, `n_modes < latent_resolution / 2`). Sources: width 64 is
the adapter default and `fno_hidden_channels` of every GINO config of neuraloperator 2.0.0
(`config/models.py`), 32 halves it (T1 spans 15 triangle modes, section 2); lr 1e-3 is
`CarCFDOptConfig.learning_rate` of `config/gino_carcfd_config.py`, 3e-4 the library default
`OptimizationConfig.learning_rate` of `config/opt.py` (both at tag 2.0.0). Fixed for all eight: Adam
without weight decay, cosine schedule (`CosineAnnealingLR`, `T_max = E`, stepped once per epoch),
relative-L2 loss, batch size `20`, `1800` training samples (the 90 % split of D3), seed 0, no early
stopping; selected is the lowest mean validation `rel_l2` after the last epoch (lines 111-116).
Measured cost (median of 3 timed epochs after 10 untimed warm-up steps; validation pass = 200 samples,
no grad):

| config | width | latent | n_modes | lr | s/epoch | s/val pass |
|---|---|---|---|---|---|---|
| G1 | 32 | 16 | 6 | 1e-3 | 116.7 | 2.1 |
| G2 | 32 | 16 | 6 | 3e-4 | 121.7 | 2.3 |
| G3 | 32 | 24 | 9 | 1e-3 | 176.8 | 2.1 |
| G4 | 32 | 24 | 9 | 3e-4 | 143.6 | 2.1 |
| G5 | 64 | 16 | 6 | 1e-3 | 152.1 | 1.3 |
| G6 | 64 | 16 | 6 | 3e-4 | 133.1 | 1.3 |
| G7 | 64 | 24 | 9 | 1e-3 | 314.5 | 2.7 |
| G8 | 64 | 24 | 9 | 3e-4 | 331.3 | 2.7 |

*Epoch budget (proposal).* `E = 50` epochs per configuration, the same for all eight. Cost:
selection (8 runs, seed 0) plus the final 5 seeds of the selected configuration; that one is not known
in advance, so the most expensive is counted: `E (sum t + 5 max t) + sum v + 5 max v` with `t` the
seconds per epoch and `v` the validation pass above, worst case **43.7 h**, within the proposed D5
budget of **48 h** wall-clock on the machine below. If the budget runs out, section 3 applies (stop, report
incomplete).

*D5 machine (proposal).* The machine of the measurement:

| Field | Value |
|---|---|
| D5 cpu | Intel(R) Core(TM) i5-1035G1 CPU @ 1.00GHz |
| D5 cores | 4 physical, 8 logical |
| D5 ram | 7.69 GiB (8260374528 bytes) |
| D5 os | Microsoft Windows 11 Home 10.0.26200 build 26200 |
| D5 python | 3.12.10 |
| D5 torch | 2.6.0+cpu |
| D5 neuraloperator | 2.0.0 |
| D5 numpy | 2.5.3 |
| D5 threads | 4 (torch.set_num_threads) |
| D5 gpu | none (CUDA available: False) |

The machine is shared: whole-machine CPU load was 32 % at the start and 25 % at the end of the
measurement, and single timed epochs ranged from 82.1 to 335.0 s; the seconds
above are estimates under load, not latency results (section 4). The adapter builds GINO with
`gno_use_torch_scatter=False`, so the neighbour reduction runs in the backend's pure-torch
`segment_csr` (`neuralop/layers/segment_csr.py`); a faster scatter backend would change these numbers.

*Margin and tie band (proposal; the open owner questions of 2026-10-20).* Margin: the chosen pair must
clear the measured float32 distance error strictly, `margin_over_float32_error > 1`; with
`radius_graph_stability(min_relative_margin, budget_relative = float32_error_near_radius,
frozen_order_flips = 0)` both pairs are `stable`, on the frozen order of D5, which the run manifest
pins. The survey records Windows 11 build 26200, Python 3.12.10 and torch 2.6.0+cpu but no CPU model, so
that it ran on the D5 machine is not recorded: re-running `scripts/oracles/gino_radius_tie_survey.py` on
the D5 machine before the first fit is proposed as a precondition (0 flips required for the chosen
pairs). That error was measured on 2026-10-20, after the margins
were visible; the owner should weigh this. A stricter factor of 2 leaves no `radius_factor` for both
latent resolutions (largest minimum 1.14). Tie band: keep the adapter's `1e-6 r` unchanged; it is a
refusal rule, not the D4 criterion, and is not widened after the fact
(`spec/GINO_ADAPTER_PHASE1_v2.md`, amendment 2026-10-21).

*Approval sentence.* The owner approves this candidate by saying, with the commit filled in:
"Ich gebe spec/PREREGISTRATION_PHASE1_v1.md in der Fassung von Commit <SHA> frei, mit D4, D5, Marge
und Tie-Band wie im Absatz 2026-10-29 vorgeschlagen." Then the status line changes as in the
approval rules at the top of this file.
The status line therefore stays DRAFT; nothing here allows a fit on T1 data.

## 9. Claim boundary

- This draft fixes the protocol only. It produces no result.
- Every later result holds for T1, its preregistered axes and the recorded
  hardware only.
