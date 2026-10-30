# Phase 1 evaluation protocol v1 — instruments for sections 4-7 of the preregistration

**Version:** 1.0
**Status:** PROPOSED — an instrument for the Phase 1 preregistration
(`spec/PREREGISTRATION_PHASE1_v1.md`, DRAFT). This file fixes HOW the
cost-accuracy curve, the latency, the aggregation statistic, its confidence
interval, the decision rules, the break-even count and the run manifest are
computed. It changes no decision rule, no metric and no owner decision
D1-D8. It trains no model and scores no model on T1 data.
**Implementation:** `src/toolkit/phase1_protocol_hex_v3.py` (numpy only).
**Tests:** `src/toolkit/test_phase1_protocol_hex_v3.py` (closed-form oracles,
negative controls, metamorphic relations, fail-closed paths).

## 0. Purpose

The preregistration says WHAT is decided (section 6) and WHICH numbers enter
the decision (sections 4, 5 and 7). Before any run exists, the computation of
those numbers must be fixed in code that is tested against closed-form
oracles, so that a later result cannot be shaped by an ad-hoc choice of
interpolation, trimming or resampling. This file is that fixation. Every
quantity below is a measurement of the instruments; none is a result.

## 1. Timing protocol (preregistration section 4)

`TimingProtocol(warmup=3, repeats=10, bootstrap_resamples=10000, seed=0,
confidence=0.95)`:

- `measure_latency(fn, protocol)` calls `fn` `warmup` times untimed, then
  `repeats` times timed with `time.perf_counter` (injectable for tests);
- the latency is the **median** of the timed calls;
- its interval is the percentile bootstrap of the median: `bootstrap_resamples`
  resamples of the timed calls with replacement, drawn from
  `numpy.random.default_rng(SeedSequence(seed))`, interval at the
  `(1 - confidence) / 2` and `1 - (1 - confidence) / 2` quantiles
  (`numpy.quantile`, linear method);
- a timed call with a negative or non-finite duration raises `ValueError`
  (a clock that went backwards is not a measurement).

Single process, batch size 1. The thread policy, the machine and the dtype
are not measured here; they go into the run manifest (section 7).
CUDA synchronisation, when a GPU is ever timed, is the caller's duty around
`fn` (preregistration section 4) and is not part of this instrument.

## 2. Mechanistic cost-accuracy curve (preregistration section 4)

`baseline_cost_accuracy_curve(config, u0, truth, lattice, steps_grid,
protocol, timed_samples, rtol)` runs `mechanistic_baseline`
(`spec/HEAT_OPERATOR_TASK_v1.md` section 6: Crank-Nicolson, warm-started CG,
`dt = T / steps`) on every input of the given set for every `steps` in
`steps_grid` (default `1, 2, 4, 8, 16, 32, 64`) and records per point:

| Field | Definition |
|---|---|
| `n`, `steps`, `dt` | the lattice and the time step |
| `sample_count`, `timed_samples` | the number of inputs solved and the number of them timed |
| `mean_rel_l2`, `median_rel_l2` | over the samples, `relative_l2` of the task spec |
| `mean_rel_h1`, `mean_rel_max` | the secondary metrics, mean over the samples |
| `median_wall_time_s` | median over ALL timed calls (`timed_samples` samples x `repeats`) |
| `wall_time_ci_s` | cluster bootstrap of that median: samples are resampled with replacement, each drawn sample contributes all its timed calls |
| `cg_iterations_mean` | mean total CG iterations per sample |
| `domain` | `lattice.domain` (`triangle_dirichlet`, `torus_periodic`); added 2026-10-21 (Equalita R4-F1), no default |
| `kappa`, `t_final` | the diffusivity and the horizon `T` of the `config` the solver ran with (`dt = t_final / steps`); added 2026-10-21 (Equalita R5-F1), no default |
| `task_config_sha256` | `config.digest()`, the whole task configuration (`lambda_cutoff`, spectral decay and domain included) |
| `rtol` | the CG tolerance the solver ran with |
| `warmup`, `repeats` | of the `protocol` the timed samples ran with |
| `timing_protocol_sha256` | `protocol.digest()` (`TimingProtocol.digest`: SHA-256 of `canonical()`, bootstrap resamples, seed and confidence of `wall_time_ci_s` included) |
| `clock` | `"time.perf_counter"` when the timed calls were read with the default wall clock, else `"custom:<qualname>"` (tests pass a fake clock) |

The first `timed_samples` inputs are timed with the protocol of section 1;
the others are solved once (the solver is deterministic, the prediction does
not depend on timing). The default times every sample. The solver runs on
the input lattice: the preregistration's evaluation lattices `N = 16, 32,
64, 128` are four separate curves.

## 3. Pareto-efficient frontier and log-log interpolation (section 6)

`pareto_frontier(times, errors)` returns the indices of the points that no
other point dominates (`t' <= t` and `e' <= e` with at least one strict),
sorted by time. On the frontier the error is strictly decreasing in the
time. Equal times keep the lower error; equal errors keep the faster point.
Times and errors must be finite and > 0 (they are logged).

`error_at_time(times, errors, t)` and `time_at_error(times, errors, e)`
interpolate **linearly in log-log** between neighbouring frontier points.
Both return `None` outside the frontier's measured range: **no
extrapolation** (preregistration section 6). A frontier of one point answers
only its own time or error.

## 4. Aggregation (preregistration section 5)

**IQM.** `interquartile_mean(values, proportion=0.25)` sorts the values,
drops `int(proportion * n)` from each end and averages the rest. This is the
trimmed mean of Agarwal et al. (NeurIPS 2021, `rliable`): for the five
seed-level means of the preregistration it is the mean of the middle three.
`proportion` must lie in `[0, 0.5)`; the values must be finite.

**Two-stage paired bootstrap.** `paired_bootstrap_iqm(errors, n_resamples=
10000, seed, confidence=0.95, proportion=0.25)` takes one `(S x M)` error
matrix per test set (seeds x sample ids; every matrix the same shape, the
sample-id axis shared), and:

1. draws, once, `n_resamples` index sets: `S` training seeds with replacement,
   then `M` sample ids with replacement, from
   `numpy.random.default_rng(SeedSequence(seed))`;
2. applies the SAME index sets to every matrix (the pairing across the
   resolution axis `N = 16, 32, 64, 128` and across test sets);
3. for each resample and matrix: mean over the resampled sample ids per
   resampled seed, then the IQM over the seeds;
4. reports the point estimate (IQM of the seed-level means of the original
   matrix) and the percentile interval of the resampled statistic.

**Pairing is by sample id, not by shape (amendment 2026-10-21, GPT-6 R2).**
`paired_bootstrap_iqm(..., sample_ids=...)` names the `sample_id` of every
column: one sequence for a single matrix, a mapping test set -> sequence for
several. Several matrices share one index draw only if every matrix holds the
same ids in the same column order; with no ids, different ids or a different
column order the call raises `PairingError` (a `ValueError`) before any
resample. The triangle sets (ID test and resolution OOD, `N = 16, 32, 64, 128`)
share ids `100000..100499` and are paired; the torus OOD set uses
`200000..200499` and is bootstrapped on its own. The check changes no number:
with equal ids the result is bit-identical to the unchecked computation.
**API change:** a mapping of two or more matrices without `sample_ids`, accepted
before, is now refused.

Every matrix is copied to C order first (amendment 2026-10-21): numpy sums a
row pairwise only over contiguous memory, and at 3b041db a Fortran-ordered or
column-indexed copy of the same values moved the point estimate by one ulp.

The index sets depend only on `(seed, n_resamples, S, M)`; the evaluation is
chunked for memory and the chunk size does not change the result.

*Design and citation (corrected 2026-10-30, issue #82).* Seeds and sample ids
are **crossed**, not nested: every sample id is evaluated under every seed, and
one sample-id draw is applied to all drawn seeds. Resampling the rows (seeds)
and the columns (sample ids) independently with replacement is Owen's
*pigeonhole bootstrap* for crossed random-effects data (A. B. Owen, Ann. Appl.
Stat. 1(2):386-411, 2007, doi:10.1214/07-AOAS122; extended to more factors by Owen and
Eckles 2012, doi:10.1214/12-AOAS547). It is mildly conservative: its variance estimate
is biased upward (Owen 2007); no bootstrap is exact for crossed designs
(McCullagh, "Resampling and exchangeable arrays", Bernoulli 6(2), 2000). The earlier text cited the two-stage bootstrap
of Davison and Hinkley (1997, section 3.8), which resamples sample ids *within*
each drawn cluster (nested design) -- that is not what the code does, and the
code is not changed: the preregistration's wording "resample training seeds,
then sample ids" with "the same resampled sample ids ... jointly" describes the
crossed scheme. Percentile intervals carry no coverage guarantee for five
seeds; the preregistration reports every seed and every test set next to them.

## 5. Decision rules (preregistration section 6)

`decide_test_set(test_set, n, gino, gino_latency_s, baseline_times_s,
baseline_errors)` evaluates both rules for one test set and one lattice:

| Rule | Evaluable | Claim |
|---|---|---|
| equal runtime | `gino_latency_s` inside the frontier's time range | `gino.high < error_at_time(gino_latency_s)` |
| equal accuracy | `gino.high` inside the frontier's error range | `gino_latency_s < time_at_error(gino.high)` |

`gino` is the bootstrap summary of GINO's IQM `rel_l2` on that test set;
`gino.high` is its upper interval bound. `gino_latency_s` is GINO's median
latency per sample at batch size 1 (section 1).

**Binding to the preregistered statistic (amendment 2026-10-20, review
finding F-E1).** The rules are defined on the upper **95 %** bound of the IQM
from the **10 000**-resample two-stage bootstrap (preregistration sections 5
and 6). `decide_test_set` therefore accepts `gino` only with
`confidence == 0.95`, `n_resamples == 10000` and `proportion == 0.25`
(module constants `PREREGISTERED_CONFIDENCE`,
`PREREGISTERED_BOOTSTRAP_RESAMPLES`, `PREREGISTERED_IQM_PROPORTION`, which are
also the defaults of sections 1 and 4), over all `n_seeds == 5` seeds and
`n_samples == 500` samples of the test set (`PREREGISTERED_SEEDS`,
`PREREGISTERED_TEST_SAMPLES`; preregistration [D2], [D3]), with finite
`0 <= low <= point <= high`; the values are compared as Python numbers (so
`np.float32(0.95)` is not 0.95) and the rule uses the validated upper bound,
not a second read. Anything else raises `DecisionInputError` (a `ValueError`)
before a rule is evaluated. **Amendment 2026-10-20 (second review, Equalita
R2):** with honest labels, a summary over 3 of 5 seeds or 20 of 40 samples had
turned `H0` into `advantage`; the seed and sample counts are now bound. Not
bound by `decide_test_set`: provenance — a summary computed at another level and
relabelled is still accepted (issue #82). **Amendment 2026-10-21 (GPT-6
HEX-P1-01):** validated labels do not replace measured provenance:
`dataclasses.replace(summary, low=0.001, point=0.001, high=0.001)` keeps every
label and turned `H0` into `advantage` at 3b041db. `decide_test_set` is
therefore a **diagnostic low-level helper**; its `to_dict()` carries
`"grade": "diagnostic"` and is never a preregistered decision. The
decision-grade entry is `decide_preregistered` (section 5a). A percentile interval is not guaranteed to contain
the point estimate; a run where it does not is refused rather than decided
(fail-closed; the review searched 6 800 designs and found no case). Before
this amendment a summary at confidence 0.5 or with one
resample was evaluated and could turn `H0` into `advantage` on the same data.
`DecisionSummary.to_dict()` carries the summary under `gino`, so the
confidence level, the resample count and the trimming proportion stand next to
the verdict.

Not evaluable, with the reason recorded and **no claim**:
- the latency lies outside the frontier's time range (no extrapolation);
- the error lies below the frontier's smallest error: **baseline floor**, no
  speed claim;
- the error lies above the frontier's largest error: no extrapolation.

**Which range counts (amendment 2026-10-20, review finding E3).** The
preregistration's "measured range" (section 6) is read as the range of the
**Pareto-efficient part** of the measured curve, in time for the
equal-runtime rule and in error for the equal-accuracy rule. A dominated point
can lie outside that range (a slow point with a large error); it does not
widen the range in which a rule is evaluable. This is the narrower and
therefore conservative reading; the behaviour is unchanged. The reasons name
the frontier range and, next to it, the range of all measured points.

The verdict is `advantage` if at least one rule claims, otherwise `H0` (no
advantage). Each rule's verdict, reason and the numbers it used are returned.
A verdict names its test set and its `N` and extends to nothing else.

## 5a. Decision-grade entry from raw errors (amendment 2026-10-21, GPT-6 HEX-P1-01)

`decide_preregistered(test_set, n, errors, sample_ids, seeds, run_manifest,
gino_latency_times_s, baseline_curve)` is the only entry
whose result is a preregistered decision. No parameter accepts a summary of
GINO's errors. The baseline curve does enter as summaries, one
`CostAccuracyPoint` per `steps` (section 2). Each point carries its own
measurement coordinates, which are checked below; its values are not
recomputed (claim boundary).

- `test_set` is one of the registered sets of preregistration section 2
  (`PREREGISTERED_TEST_SETS`): `id` (triangle, `N = 32`, ids
  `100000..100499`), `resolution_ood` (triangle, `N = 16, 64, 128`, same ids),
  `composite_ood` (torus, `N = 32`, ids `200000..200499`); `n` must be one of
  its lattices.
- `errors` is GINO's raw `rel_l2` table `[5, 500]` of that set and lattice,
  finite and `>= 0`; row `i` belongs to `seeds[i]`, column `j` to
  `sample_ids[j]`. `seeds` must be the registered training seeds `0..4`, the
  ids exactly the registered ids of the set.
- `run_manifest` is the output of `build_run_manifest` (section 7); it is
  rebuilt from its own fields and must match its `manifest_sha256`, and its
  seed set must equal `seeds`. Its content must hold what the preregistration
  fixes (Equalita R3 F2, amendment 2026-10-21), else `DecisionProvenanceError`:
  every `split_ids` id in the train pool `0..1999` (preregistration section 2,
  which keeps the test ids out of the split); a `datasets` digest under the
  name of the decided test set (section 2 "Recording", section 7); the
  preregistration `sha256` equal to `file_sha256` of
  `spec/PREREGISTRATION_PHASE1_v1.md` in the repository (section 7); hardware
  `cuda` = `False` and `dtype` = `"float32"` (section 4 same CPU machine,
  section 3 float32, D5 CPU without GPU); hardware `amp` present and `False`,
  and `autocast`, where present, `False` (Equalita R4-F3, amendment
  2026-10-21: preregistration :103 float32 for training, :122 and :228
  dtype/AMP go into the manifest; other spellings of mixed precision are not
  recognised, owner decision R4-F5); backend `neuraloperator` = `"2.0.0"`
  (D7). Not checked, because the preregistration does not fix it: the machine
  name (D5 open), the file behind `metric_spec_sha256` (preregistration "the
  instrument spec" vs. section 7 here), the exact 1800/200 split, the torch
  version.
- `baseline_curve` is the tuple (or list) of `CostAccuracyPoint` returned by
  `baseline_cost_accuracy_curve`; anything else, including the former two
  float lists, a subclass of the point or its `to_dict`, raises `TypeError`.
  Every field of every point is read once into a plain `int`, `float` or
  `str` (Equalita R5-F3: a coordinate read twice let an `int()` that changes
  between the reads be checked as 32 and recorded as 64); the checks and the
  record use only this copy. Types and plausibility first, else
  `DecisionInputError` (R5-F4): every number finite; `dt`, `kappa`,
  `t_final`, `rtol`, mean `rel_l2` and median time `> 0`; median `rel_l2`,
  `rel_h1`, `rel_max` and `cg_iterations_mean` `>= 0`; `wall_time_ci_s` a pair
  with `0 <= low <= high` (whether it contains the median is not required:
  the percentile bootstrap does not guarantee it). Then every point is checked
  by its own coordinates, in this order, else
  `DecisionInputError` naming the first deviating coordinate (Equalita R4-F1):
  the `steps` of the points, in order, are exactly `1, 2, 4, 8, 16, 32, 64`
  (preregistration :134-137; a subset, another grid or another order is
  another frontier: R3 F1 found the genuine points for steps `{1, 64}`, R4-F1
  the genuine curve on steps `(1, 8, 16, 24, 32, 48, 64)` reaching
  `advantage`); every point's `n` is the decided lattice (:134-135; R4-F1: the
  genuine `n = 64` curve under `n = 32` reached `advantage`); its `domain` is
  the test set's domain in the lattice's spelling; the settings it was
  computed with are the registered ones (Equalita R5-F1: the genuine curve
  solved with `kappa = 2` or `T = 0.02` against the registered truth reached
  `advantage`, and for `kappa` its record was indistinguishable from a
  registered run): `kappa = 1` and `t_final = 0.01` ([D1], preregistration
  :51-55), `dt == t_final / steps` exactly (the producer's own division),
  `task_config_sha256` the digest of `HeatOperatorTaskConfig(domain=...)` of
  the test set's domain (the whole [D1] configuration), `rtol = 1e-12` (the
  tolerance of `mechanistic_baseline` as verified; the preregistration names
  the method at :128-133 but writes out no tolerance, owner question),
  `warmup = 3` and `repeats = 10` (:145), `timing_protocol_sha256` the
  digest of `TimingProtocol()` and `clock = "time.perf_counter"` (the wall
  clock of section 1; a slow fake clock would make the baseline look slow);
  its `sample_count` is 500
  (:62-68 [D3], :136 "on the same test inputs"); `1 <= timed_samples <=
  sample_count`. No time and no error may repeat: the same two points
  copied to seven would keep the chord. This catches only the literal
  repetition; copies shifted by one ulp or `1e-12` and chord points pass
  (Equalita R4-F2, see the claim boundary). `provenance` is written from the
  checked points, never from the registered grid or the `n` argument:
  `baseline_points` (every point with all its fields), and derived from them
  `baseline_times_s`, `baseline_errors`, `baseline_steps` and `baseline_n`
  (the sorted list of the points' lattices).
- `gino_latency_times_s` are the timed calls, exactly the preregistered 10
  measured repeats per sample (`TimingProtocol().repeats`, section 1): shape
  `(10,)` for one sample or `(samples, 10)`; any other count, including a flat
  1-D vector of several samples, raises `DecisionInputError` (HEXA-TIM-01). The
  latency is their median, as for `median_wall_time_s` of the baseline.
  Limit: the 3 warm-up runs are untimed and leave no trace in these numbers, so
  they are not checked here; binding them needs a pipeline attestation of the
  timing run (open, see `provenance["authenticity"]`).

Rows and columns are sorted by seed and sample id (C order), the bootstrap and
the IQM are computed with the preregistered settings of section 4, and both
rules of section 5 are applied. A violation of any check listed in this
section raises `DecisionInputError` (registration) or
`DecisionProvenanceError` (manifest) before a rule runs. Only these checks
are made: what the list names as not checked, the latency's sample ids and
batch size, the baseline curve's input identity, its machine and thread
policy, which `timed_samples` inputs were timed (claim boundary below), and
the authenticity of all numbers are not. The baseline's horizon and timing
protocol are checked since R5-F1 (list above).

The result `PreregisteredDecision` serialises (`to_dict`) to a record with
`"grade": "preregistered"`, the verdict, both rules, the summary, and
`provenance`: SHA-256 of the canonical raw bytes (little-endian float64, C
order, `-0.0` as `0.0`, shape header), of the sample ids and of the latency
calls; the baseline curve values; the seed list; the run manifest digest with
the git SHA, preregistration, dataset digests, backend and hardware. The
record carries `record_sha256` over its canonical JSON; `to_dict` recomputes it
and refuses a decision whose fields were replaced afterwards.
`verify_preregistered_decision(record, **raw)` checks the self-digest, then
recomputes the record from the raw data and refuses any difference.

**Claim boundary.** The self-digest protects against drift and against an
edited summary, verdict or digest. It does not protect against a forger who
fabricates the raw table itself: such a record is consistent with the
fabricated bytes and only its `raw_errors_sha256` differs. `provenance`
therefore states `"authenticity": "unattested: ..."`. An independent review
needs, in addition, a signature over the raw bytes or an authentic pipeline
attestation (GPT-6 fix step 4; not implemented, owner decision). The same
holds for the baseline curve: its points enter as values with coordinates and
are bound in the record, but `verify_preregistered_decision` does not rerun
`baseline_cost_accuracy_curve`; a reviewer reproduces the errors from the
datasets named in the manifest (the solver is deterministic) and the times
only on the recorded machine. In particular (Equalita R4, self-check of the
R4-F1 fix):
- The coordinates are those the producer wrote. A caller who builds
  `CostAccuracyPoint`s by hand can put the registered coordinates on values
  measured elsewhere (a foreign-grid curve relabelled with the registered
  steps reaches `advantage`); this is a fabrication and falls under
  `authenticity`, like a fabricated raw table.
- Which 500 inputs the curve ran on is not bound: a point carries their
  count, not their sample ids or a dataset digest, so a curve on 500 other
  inputs of the same domain and lattice passes. Binding it needs an input
  digest in the point compared with the manifest's dataset digest per
  `(test_set, n)` (R4-F4, owner).
- Corrected 2026-10-21 (Equalita R5-F2): this list said that `dt`, the
  solver tolerance and the baseline's warm-ups and repeats were "recorded,
  not checked". `dt` was recorded but not compared with the preregistration;
  the tolerance, the warm-ups, the repeats and `kappa` were not in the record
  at all. Since R5-F1 they are carried, checked and recorded (see the list of
  checks above).
- The machine and thread policy are not checked. The preregistration requires
  GINO and the baseline on the **same CPU machine and thread policy** (:141)
  with the thread counts in the manifest (:144). Neither side carries a
  machine identity the function could compare: the baseline point has none,
  the GINO latency calls are bare numbers, and the manifest's `hardware`
  block is the GINO run's, of which only `cuda`, `dtype` and `amp` are fixed
  (D5 open). A stamp written by the caller would be self-declared, i.e. the
  `authenticity` class. Equalita R5 saw the baseline times of two curves in
  one process differ by a factor 2 to 4 under load, enough to flip a verdict:
  the time axis holds only for the recorded run conditions (owner question).
- `timed_samples` is checked only for `1 <= timed_samples <= sample_count`.
  It decides which inputs are timed (the first k, :69) and so
  `median_wall_time_s`; the preregistration fixes no value, so a caller may
  time a cheaper subset (Equalita R6-F2, owner question).
- Copies shifted by one ulp or `1e-12` and chord points pass the repeat check
  (R4-F2); only a recomputation of the errors or an attestation of the times
  closes that.

**API change (2026-10-21):** callers that took `decide_test_set(...).to_dict()`
as the decision must call `decide_preregistered` with the raw table instead;
`decide_test_set` keeps its signature and verdicts, and its record gains
`"grade": "diagnostic"`.

**API change (2026-10-21, Equalita R4-F1, breaking):** `decide_preregistered`
and `verify_preregistered_decision` take `baseline_curve` (the
`CostAccuracyPoint` tuple) instead of `baseline_times_s` and
`baseline_errors`; the old keywords raise `TypeError` (unexpected keyword),
bare lists raise `TypeError`. `CostAccuracyPoint` gains the required last
field `domain`. Run manifests for a preregistered decision need hardware
`amp = False` (R4-F3). The record gains `provenance["baseline_points"]`;
`baseline_n` is now a list (the points' lattices). `decide_test_set` (the
diagnostic helper) keeps the float lists. Whether the API stays so is an
owner decision (Equalita R4 owner question 2); this form fails closed.

**API and format change (2026-10-21, Equalita R5-F1/F3/F4, breaking):**
`CostAccuracyPoint` gains eight required fields after `domain`: `kappa`,
`t_final`, `task_config_sha256`, `rtol`, `warmup`, `repeats`,
`timing_protocol_sha256`, `clock` (a construction without them raises `TypeError`);
`TimingProtocol` gains `digest()`. `baseline_cost_accuracy_curve` keeps its
signature and writes them. `decide_preregistered` refuses curves computed
with other settings than the registered ones (other `kappa`/`T`/task
configuration, `rtol != 1e-12`, a timing protocol other than
`TimingProtocol()`, a clock other than `time.perf_counter`), a `dt` that is
not `t_final / steps`, negative record fields and an inverted interval. The
record's `baseline_points` entries gain the eight fields, so a record written before this change does not verify
against the new code. `decide_test_set` is unchanged.

## 6. Break-even (preregistration section 4)

`break_even_queries(training_time_s, data_generation_time_s, gino_latency_s,
baseline_latency_s)` is the smallest integer `q` with
`q * (baseline_latency_s - gino_latency_s) >= training + data`, i.e.
`ceil((training + data) / (baseline - gino))`; `None` when GINO is not faster
per query (the fixed cost is never amortised). The quotient is evaluated in
binary64 as written; only where that evaluation overflows on finite inputs (sum
or quotient beyond the float range; `ceil(inf)` raised a raw `OverflowError`)
is the count computed exactly on the given binary64 values, as an unbounded int
(HNT-R86-04, issue #82, cross-family GPT-6 R86). Wherever the float evaluation
is finite its result is unchanged. The baseline latency is the
`time_at_error` of the equal-accuracy rule, so a break-even count exists only
where that rule is evaluable. `break_even_queries` itself takes any baseline
latency (unchanged); `break_even_at_equal_accuracy(training_time_s,
data_generation_time_s, gino_upper_rel_l2, gino_latency_s, baseline_times_s,
baseline_errors)` (added 2026-10-30, issue #82) recomputes that latency from the
baseline frontier at the GINO upper error bound instead of accepting it, and
returns `None` where the equal-accuracy rule is not evaluable.

## 7. Run manifest (preregistration section 7)

`build_run_manifest(fields)` validates and digests the record of a run.
Required, each non-empty and typed:

| Key | Content |
|---|---|
| `git_sha` | 40-hex commit of the code that ran |
| `preregistration` | `{"version": str, "sha256": 64-hex}` of the preregistration file (`file_sha256`, LF-normalised) |
| `datasets` | set name -> `dataset_sha256` (64-hex) from `dataset_manifest` |
| `split_ids` | `{"train": [...], "val": [...]}`, disjoint non-negative ints |
| `backend` | name -> version string (torch, numpy, neuraloperator, python) |
| `config_digest` | `GinoPhase1Config.digest()` (64-hex) |
| `seeds` | distinct non-negative training seeds |
| `hardware` | at least `machine`, `threads`, `dtype`, `cuda` |
| `reference_spec_ids` | the Phase 0 ReferenceSpec ids |
| `metric_spec_sha256` | `file_sha256` of `spec/HEAT_OPERATOR_TASK_v1.md` |
| `result_digests` | name -> SHA-256 (64 lowercase hex) of every result file (may be empty before results exist; each value is checked, Codex #80 2026-10-23) |
| `worktree_dirty` (optional) | bool from `source_revision()`; recorded as given (added 2026-10-30, issue #82 F-E3: `{**fields, **source_revision()}` raised "unknown manifest keys", so a dirty tree had no slot) |
| `extra` (optional) | free-form mapping |

Any other key, a missing key, a wrong type, a NaN or an empty required value
raises `RunManifestError`. Every hex digest must be the whole string: the
patterns end in `\Z`, not `$`, which also matched before a final newline
(HNT-R86-03, issue #82). The manifest carries `schema`
`hexantk_phase1_run_manifest_v1`, `protocol` `PHASE1_PROTOCOL_v1` and
`manifest_sha256` over the canonical JSON (sorted keys, no NaN) of
everything else. `source_revision()` reads the git HEAD and whether the work
tree is dirty; a dirty tree is recorded, not hidden.

## 8. Fail-closed

- every argument is type- and range-checked; invalid input raises
  `TypeError` or `ValueError` before any computation;
- non-finite inputs are rejected everywhere; a non-finite derived statistic
  cannot occur for finite inputs (means, medians, quantiles of finite data);
- no function returns `inf` or `NaN`; "not evaluable" is `None` with a
  reason. Means, medians and quantiles of finite data near the float limit
  overflowed in their intermediate sums (issue #82 F-E5: `IQM([1e308] * 3)` was
  `inf`, the median of two durations of 1.5e308 s too); since 2026-10-30 such a
  statistic is recomputed on a copy scaled by an exact power of two and scaled
  back, and only where the plain evaluation was non-finite, so every finite
  result is unchanged bit for bit;
- all randomness comes from an explicit seed; two calls with the same
  arguments return identical numbers.

## 9. Measured accuracy of the instruments (2026-10-19, Linux container)

- IQM equals the closed form (`[1, 2, 3, 4, 100] -> 3`, middle three);
- paired bootstrap: a degenerate design (all errors equal) gives an interval
  of zero width; a second matrix equal to twice the first gives exactly twice
  its bounds (pairing);
- log-log interpolation on `e = C t^-p` reproduces `e(t)` and `t(e)` to
  binary64 rounding at every interior point; outside the range both return
  `None`;
- the cost-accuracy sweep on a 3-sample triangle `N = 12` set reproduces
  second-order convergence in `dt` of the Crank-Nicolson baseline against
  the exact semidiscrete solution.

## 10. Limits and claim boundary

- Latency is a property of the machine, the thread policy and the dtype
  recorded in the manifest, never of the method.
- The frontier interpolation is a convention (log-log linear); the
  preregistration fixes it before any run so it cannot be chosen after.
- The percentile bootstrap is a description of the five seeds' variability,
  not a frequentist guarantee.
- Nothing here is fitted to, tuned on, or scored against T1 data; the tests
  use synthetic numbers and the verified mechanistic baseline.

## 11. D4 decision support: float-order stability of a radius graph (amendment 2026-10-21, GPT-6 HEX-P2-02)

Two instruments for the owner's D4 choice; they decide nothing and change no
preregistered threshold and not the adapter's tie band (`1e-6 r`).

- `cdist_rounding_margin(radius, *, max_point_norm, unit_roundoff=2**-24,
  input_abs_error=0)` returns the relative margin `m` such that every pair with
  `|d - r| > m r` is decided `d <= r` as in exact arithmetic by the matmul path
  of `torch.cdist` (`s = [-2x, |x|^2, 1] . [y, 1, |y|^2]`, then `sqrt`), for
  every summation order: `|s_hat - s| <= gamma_6 (2R)^2` (Higham 2002, eq. 3.5
  and lemma 3.3), a relative `u` for the square root and for the radius cast
  to the working dtype, and `input_abs_error` for the rounded coordinates.
- `radius_graph_stability(min_relative_margin, *, budget_relative,
  frozen_order_flips=None)` returns `stable` only if the margin strictly
  exceeds the budget and no flip was measured on the frozen order (flips > 0
  contradict the budget: `refused`, Equalita R3 F4), `frozen_order_required` if
  not but the candidate was
  measured flip-free (`frozen_order_flips == 0`) on the one frozen
  hardware/kernel order the run manifest must then pin, and `refused`
  otherwise. Equality is not stable; NaN or inf raise `ValueError`.

The budget is fixed before candidates are compared; choosing it after seeing
the margins would be the post-hoc widening the review excludes. Tests: closed
forms, an emulation of 48 summation orders as an independent oracle (positive
and negative control), and the committed survey
`results/2026-10-20_gino_radius_tie_survey_float32-graphs.json` (measured errors
below the bound; no tie-free candidate clears it, `spec/GINO_ADAPTER_PHASE1_v2.md`
amendment 2026-10-21).

References: McGreivy and Hakim, Nat. Mach. Intell. 6:1256 (2024), rules 1-2
(equal accuracy, equal runtime, no weak baseline); Agarwal, Schwarzer,
Castro, Courville and Bellemare, NeurIPS 2021 (IQM, stratified bootstrap);
Owen, The pigeonhole bootstrap, Ann. Appl. Stat. 1(2):386-411 (2007)
(crossed designs; replaces the earlier citation of Davison and Hinkley 1997,
section 3.8, nested two-stage resampling -- issue #82); Efron and Tibshirani 1993,
chapter 13 (percentile interval).
