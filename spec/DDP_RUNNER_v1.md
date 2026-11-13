# DDP_RUNNER_v1 — the training runner under DistributedDataParallel

**Version:** 1.0 (2026-10-30; hardened before merge the same day after a critical
self-review: agreed construction, adapter rule, agreed forward errors, deterministic
validation sampler, rank-0-bound decision sums, per-rank RNG streams on resume;
second round after an independent review of head a875020: errors before the forward
agreed, step count agreed per step, buffers synchronised from rank 0, config digest,
per-rank loader generators; third round after the review of 205f293: buffer
signatures include dtype and shape, OOM rescue while preparing, effective AMP mode
agreed, RNG/generator states moved to the CPU before restore)
**Version 1.1** (2026-11-02): errors after the gradient all-reduce (clipping,
optimizer/scaler step, EMA update) are held and agreed at the next agreement point
(section 3.6); evidence extended to a 3-rank gloo job with an uneven validation
partition (section 8). Review round the same day: `GradScaler.unscale_` belongs to
the held region too; every agreed failure names the failing ranks and is logged
on them; a failed agreement collective no longer hides the rank's own exception
(section 5). No change to what a healthy run computes.
**Version 1.2** (2026-11-09): a non-finite total gradient norm (finite loss) skips the
update instead of writing NaN into every parameter, the EMA copy and the checkpoints;
a parameter that is non-finite after the optimizer step stops the run; distributed,
all ranks or none must skip (section 3.7). Applies with and without DDP. No change to
what a healthy run computes (every finite-gradient step is taken as before).
Third round the same day (independent review of cc9f73a): the record of a skipped
update is held under DDP like the step itself (section 3.6, HX102-01); with an
active `GradScaler` a non-finite norm that `unscale_` did not see (fp32 overflow of
finite gradients, a zero scale) is skipped too (section 3.7, HX102-02/-04);
`RunnerConfig.max_consecutive_skips` (default 50, 0 = off) stops a run at the N-th
skipped step in a row, agreed under DDP and carried across resume (section 3.7,
HX102-03, owner decision E1109-06a).
**Version 1.3** (2026-11-10): the post-resume agreement covers every state the ranks
must share, not only the weights: optimizer, scheduler, `GradScaler` and EMA state
(section 6). Measured red on main 22e6308: rank 1 resuming from a copy with equal
weights but other momentum buffers (or another loss scale, or another EMA average)
ran to the end on both ranks with different weights. First AMP-under-DDP evidence:
a CPU `GradScaler` in the 2-rank gloo job (section 8). No change to what a healthy
run computes.
Review round the same day (self-review of 5c9d753): the same agreement now also runs
at construction (section 2: ranks built with another learning rate or scheduler, red
on 5c9d753 -- both ranks trained to the end with diverging weights); the digest no
longer loses values of NumPy scalars/arrays, bytes, sets and complex numbers (they
fell into the type-name branch: different values, equal digest); set elements are
sorted because string hashing, and with it set order, differs between processes.
Third round the same day (self-review of f173c56): computing the digests is agreed
(`run_agreed`): a `state_dict()` that raised on one rank left it alone while its peers
waited in the digest broadcast until the timeout (red: 30.1 s, gloo `RuntimeError`
instead of `PeerRankError`); sparse tensors are digested by value instead of crashing
(also in the weights digest).
**Version 1.4** (2026-11-12, R12 phase 2; 007acc2 R11 HX-R11-DISTRIBUTED/-NUMPY): the
optimizer/GradScaler behaviour check of section 6 is agreed BEFORE the step (one
1-element all-reduce per update, section 3 item 6): a refusal on one rank stops every
rank before any rank steps. Red on 4e6bcf0 with a 2-rank gloo job: rank 1's
`SGD._init_group` patched after the initial agreement, rank 1 refused and held its
error, rank 0 stepped (its weights changed, rank 1's did not) until the next agreement
point stopped both. `unscale_` and clipping errors are agreed at the same point (raised
at once, no longer held). `scheduler.step()` gets the same check, agreed once per epoch.
The state digest encodes NumPy dtypes by their full schema (structured field names,
order, offsets, titles, subarrays; red on 4e6bcf0: `[('x',i4),('y',i4)]` and
`[('u',i4),('v',i4)]` digested alike, both `|V8`), refuses dtype metadata and carries a
schema id (`hexantk/shared-state-digest/v2`). No change to what a healthy run computes.
Self-review the same day (of 3b11876): the pre-step all-reduce now also carries the
skip decision (2 elements, same single collective): a split of the norm-based skip
is refused before any rank steps (red on 3b11876: rank 0 stepped, the split was
caught one step later with differing weights); a failure of the pre-step collective
itself (a dead peer, the timeout) is raised at once instead of held into the next
collective on a broken group (red on 3b11876: held); the `GradScaler` check on a
skipped step is pinned by a test; a 3-rank job covers a refusal on the middle rank;
CI runs the multi-rank jobs under `TORCH_DISTRIBUTED_DEBUG=DETAIL` (every collective's
sequence number, type, shapes and dtypes compared across ranks).
**Version 1.5** (2026-11-13, WP1 of the A7 round-2 follow-up on #107): the shared-state
agreement of section 6 also runs at the end of every epoch (after validation, before any
checkpoint of that epoch) and before every mid-epoch save. DDP all-reduces the gradients,
but nothing checked what each rank then does with them: a `clip_grad_norm_` that differs
on one rank, or a DDP comm hook that scales one rank's gradients, trained the ranks apart
to the end of the run, and rank 0 wrote its own weights as the run's (red on 47cda7f).
Now every rank stops with `DistributedContractError` naming the component
(`end-of-epoch-<e> state differs from rank 0 on at least one rank: model weights`) and no
checkpoint of the differing state is written. At a mid-epoch save the model's and the EMA
module's buffers are left out (rank-local between two validations, section 5); they are
digested as skipped names, so the key set must still agree, and the EMA's `n_averaged`
stays in. No change to what a healthy run computes.
**Status:** ACTIVE for `HexTrainingRunner` with `RunnerConfig(distributed=True)`.
Infrastructure contract; it trains no model for the Phase 1 study and makes no
learning claim (`spec/PREREGISTRATION_PHASE1_v1.md` stays DRAFT).
**Implementation:** `src/toolkit/runner_hex_v3.py` (runner),
`src/toolkit/ddp_hex_v3.py` (`ExactDistributedEvalSampler`, `DistributedRunContext`).
**Tests:** `src/toolkit/test_ddp_hex_v3.py` — no process group, an in-process gloo
group of world size 1, a real 2-rank gloo job and a 3-rank gloo job; mutants in `scripts/mutation_probe.py`.
**Origin:** ROADMAP "DDP-Integration des Runners" (only an `is_main_process` guard,
no `set_epoch`, no barrier after the checkpoint write, metrics not reduced);
cross-family GPT-6 R89 (2026-10-24, sidecar prototype, counterexamples D1-D4) and
the cross-family answer of 2026-10-28 (item 7 / A6).

## 1. Why a contract

DDP all-reduces gradients, nothing else. Every decision the runner takes outside
that all-reduce must come out the same on every rank, otherwise one rank enters a
collective its peers never reach (the job hangs until the process-group timeout)
or the ranks silently disagree (different best model, different metric). The
contract lists those decisions and how each is agreed.

## 2. Preconditions (checked at construction, fail-closed)

| Condition | Violation |
|---|---|
| `torch.distributed` initialized | `DistributedContractError` |
| model is `DistributedDataParallel` | `ValueError` (an unwrapped model trains independently per rank) |
| `config.is_main_process == (rank == 0)` | `ValueError` |
| an `adapter` wraps the DDP model itself (`adapter.model is model`) | `ValueError`: the adapter runs the training forward; around `model.module` it bypasses DDP and no gradient is ever all-reduced (silent divergence) |
| `config.distributed=False` with a DDP model | `ValueError` (validation and selection would be rank-local) |

The checks and every local construction step (directories, checkpoint manager, EMA
copy) run inside one `run_agreed` (section 5) before the first other collective: a
check that fails on one rank only (e.g. `is_main_process=True` on rank 1) raises
there and `PeerRankError` on the others, instead of leaving them in the `run_id`
broadcast until the timeout.

**Same configuration.** Next, a SHA-256 over all `RunnerConfig` fields except
`is_main_process`, `output_dir` and the `resume_from` path (mount points may differ
between hosts; *whether* `resume_from` is set is included) must equal rank 0's,
else `DistributedContractError` on every rank. The digest also covers the AMP mode the
host actually resolves (autocast on/off, dtype, GradScaler present): `amp_dtype="bfloat16"`
falls back to float16 + GradScaler on a GPU without bf16 support, and ranks on mixed GPUs
would all-reduce scaled with unscaled gradients (review round 2). A different `num_epochs` or save
cadence would issue a different sequence of collectives (hang), a different best
policy or `use_ema` would silently select differently, and `resume_from` on one
rank only left the others in the first training collective (review finding 3).

**Same state (v1.3).** Then the digests of the model state, optimizer, scheduler,
`GradScaler` and EMA must equal rank 0's (section 6, the same check as after a
resume), else `DistributedContractError` naming the components on every rank. An
equal config does not imply equal state: an optimizer built with another learning
rate on one rank passed every check before and trained the ranks apart. Since R8
(2026-11-11) the optimizer, scheduler and `GradScaler` must also be built-in classes
whose state determines their behaviour (section 6, "Which states are comparable"):
a `LambdaLR`, a subclass or a look-alike class is refused here, on every rank.

`distributed` and `is_main_process` must be `bool`. All checkpoints, the EMA copy,
validation and replay capsules use the unwrapped module (`runner.raw_model`):
checkpoint keys carry no `module.` prefix, and validation runs no DDP collective.

## 3. Per-epoch agreement

1. **Sampler epoch.** Before each epoch the runner calls `set_epoch(epoch)` on the
   train and validation loaders' sampler (and batch sampler's sampler) if it has
   one — also without DDP. `DistributedSampler` shuffles with `seed + epoch`;
   without `set_epoch` every epoch repeats the first permutation.
2. **Equal step count.** Each step is a DDP gradient all-reduce; a rank with an
   extra batch would wait forever. `len(train_loader)` must be equal on every rank
   (fast fail before the first step; unsized loaders count as -1), but `len` can
   be wrong (a sized `IterableDataset` declaring 4 and yielding 3), so the binding
   check is per step (next item). `torch.distributed.algorithms.Join` is not used.
3. **Before the forward.** Each rank takes its next batch and prepares it
   (`zero_grad`, warm-up LR, `.to(device)`, the adapter's context validation);
   then one int64 all-reduce (SUM) of [ranks that raised, ranks out of data]:
   an error raises on every rank (the failing rank its own exception, the others
   `PeerRankError`); all ranks out of data ends the epoch; some out of data raises
   `DistributedContractError` on every rank. An OOM while preparing gets the same
   rescue (replay capsule, FailureSignature) as one inside the step. This agreement must precede the DDP
   forward: with buffers (BatchNorm) and the default `broadcast_buffers=True`, DDP
   starts every forward with a buffer broadcast, which a rank that failed before
   it never reaches (review finding 1: a missing `target` key on one rank hung
   both). The train iterator itself is created inside `run_agreed`.
4. **Step status.** Forward, loss and the non-finite-loss check run per rank
   (the detecting rank writes its FailureSignature and replay capsule); then one
   int64 all-reduce (MAX) agrees on a status before any backward:
   0 = step, 1 = skip (a non-finite loss on some rank), 2 = a rank raised in its
   forward/loss. On 2 the failing rank re-raises its exception (an OOM still
   reaches the rescue path), the others raise `PeerRankError`; on 1 no rank runs
   backward. A rank that skipped backward or left the step while its peers entered
   the gradient all-reduce would deadlock them ("DDP expects all processes to launch
   the same number of AllReduce comm ops in every iteration"). Forward without
   backward on every rank was checked with default DDP, `find_unused_parameters=True`
   and `static_graph=True` (torch 2.6.0, 2 gloo ranks; also a test scenario).
   Loss-spike detection only logs and needs no agreement. An exception inside
   backward (e.g. an OOM there) is not covered: the peers are already inside the
   gradient all-reduce (section 8). An exception after it is agreed (item 6).
   A step therefore costs two small all-reduces besides DDP's own.
5. **Training metrics.** `mean_loss` and `mean_grad_norm` are sums over all
   ranks' healthy steps divided by their count. `n_batches` (healthy steps) and
   `n_anomalies` (skipped steps) are per rank and equal on every rank because the
   skips are agreed.
6. **After backward (v1.1; v1.4 pre-step agreement).** Since v1.4 one agreement sits
   between clipping and the step (`_agree_before_step`, one 1-element int64
   all-reduce per update): every rank reaches it after DDP's gradient all-reduce, with
   the errors of `unscale_` and clipping caught on the way, then runs the behaviour
   check of section 6 (`_admit_step_behaviour`: the optimizer if the step will be
   taken, the `GradScaler` whenever there is one). The all-reduce carries two
   counts, ranks that failed or refused and ranks that would take the step. If any
   rank failed or refused, every rank raises there -- the failing rank its own
   exception with a note, the others `PeerRankError` -- and no rank has stepped. If
   only some ranks would take the step (a split of the skip decision of item 7),
   every rank raises `DistributedContractError` there, again before any step.
   Whatever leaves this agreement, also a failure of its own collective, is raised
   at once, never held (holding it would send the rank into the next collective on
   a broken group). What follows (the step, the
   EMA update, the skip record) is held as described next. `GradScaler.unscale_` (AMP), clipping, the optimizer
   (or `GradScaler`) step, the EMA update and, for a skipped update (item 7), its
   replay capsule and `FailureSignature` run per rank after DDP's gradient
   all-reduce has completed; none of them is a collective. (The skip record was
   outside the held region until v1.2 third round, HX102-01: an `OSError` there on
   one rank only, a full disk, left that rank alone, measured with a real 2-rank gloo
   job: the peer waited in the next agreement until the collective timeout.) A rank that raised there used to leave alone: its peers
   waited in the next step's agreement until the process-group timeout (or, with
   `save_every_n_steps`, a peer could start the mid-epoch save's gathers without
   it). Now the failing rank holds the exception (an OOM gets the rescue first),
   logs one line naming it (if the agreement never completes, the cause is still in
   that rank's log) and returns from the step; the next agreement point raises it
   on every rank:
   - `_agreed_next` (item 3) carries a third count, ranks holding a step error,
     in the same all-reduce, so no extra collective per step. A held error is
     raised before the batch-preparation errors and before the end of the epoch
     is accepted (a failure in the epoch's last step does not end the epoch
     cleanly). The holding rank loads no further batch.
   - `_save_mid_epoch` (`spec/MID_EPOCH_RESUME_v1.md`) starts with one int64
     all-reduce of the same flag, before any gather or write: no rank writes a
     checkpoint of a step that failed on a peer.
   The holding rank re-raises its own exception, the others raise `PeerRankError`
   naming the failed step. Between the failure and the agreement the holding
   rank's parameters may be partially updated; nothing reads them before the
   raise (no collective, no save). Every step is followed by one of these two
   points, so a held error never outlives its epoch. Without DDP the exception is
   raised where it happens, as before.
7. **Non-finite gradient (v1.2; with and without DDP).** The loss check (item 4)
   sees only the loss value. A finite loss can still give a NaN/Inf gradient (an
   operation whose derivative is singular at the point, an overflow in backward);
   `clip_grad_norm_` then returns a non-finite total norm and scales every gradient
   by it, and the optimizer step wrote NaN into all parameters, from there into the
   EMA copy, `last.pt`, `resume_last.pt` and every later checkpoint, while the run
   went on and reported a NaN *loss* one step late (measured on main cd5f6dc,
   NTK-R5-NR03 of the cross-family R5 round of 2026-11-04). Now, without a
   `GradScaler`, a non-finite norm skips the optimizer step and the EMA update,
   clears the gradients and counts the step as skipped (`n_anomalies`, no
   `grad_norm` entry), with a replay capsule (weights *before* the skipped update)
   and a `FailureSignature` of kind `NaN_or_Inf_Grad`, locus `train_step_grad_check`.
   This includes an fp32 overflow of the norm of *finite* gradients (elements around
   1e19 and up): clipping by an inf norm zeroes every gradient and the step would
   apply only momentum and weight decay; it is skipped too.
   With a `GradScaler` the scaler decides, as in PyTorch AMP: `scaler.step` skips
   when `unscale_` found inf/NaN, and `update` then lowers the scale. A scale lower
   after `update` than before is that skip (the scale drops only on a found inf;
   growth is no skip). v1.2 first round left this path unchanged and said so; the
   second round (same day) measured it with `torch.amp.GradScaler("cpu")` (torch
   2.6) and found two defects: the EMA was still updated after the skipped step (an
   EMA update on unchanged weights still moves the average and counts in
   `n_averaged`), and the step counted as healthy with `grad_norm` NaN/inf, which
   made the epoch's `mean_grad_norm` non-finite. Now such a step skips the EMA
   update and counts as skipped like the others; it writes no capsule and no
   FailureSignature (loss-scale backoff is routine in AMP, a capsule per overflow
   would fill the disk), only a warning. A disabled scaler (`enabled=False`: scale
   fixed at 1.0, `step` is a plain `optimizer.step()` that never skips) takes the
   no-scaler path; without that guard its NaN update reached the weights. (Since
   the third round the guard is redundant and removed: every non-finite norm skips
   before `scaler.step`, `unscale_`/`update` of a disabled scaler do nothing and its
   scale never drops, so it behaves like no scaler. Its mutant survived the full
   probe on 3ba3835 as an equivalent mutant; the test of the behaviour stays.)
   Third round (same day, independent review of cc9f73a): `unscale_` checks the
   gradients *before* it unscales them, so two non-finite norms passed its inf check
   and the scaler stepped. HX102-02: finite gradients whose fp32 norm overflows
   (elements around 1e25): clipping zeroed them, the step counted as healthy with
   `grad_norm` inf and `mean_grad_norm` became inf. HX102-04: a scale of 0 (the scale
   cannot drop below 0, so the backoff test never fires): `unscale_` multiplies the
   zero gradients by 1/0, the NaN reached the weights and `NonFiniteParameterError`
   stopped the run with a text that claims finite gradients (both measured with the
   `_apply_update` of cc9f73a and `torch.amp.GradScaler("cpu")`, torch 2.12). Now, with an active
   scaler too, a non-finite norm skips `scaler.step`; `update` still runs (a found
   inf still lowers the scale). Only a skip with a lowered scale is the routine
   backoff (warning, no capsule); a skip without one (overflowing norm, zero scale)
   is a failure like the no-scaler case: replay capsule and `FailureSignature`
   `NaN_or_Inf_Grad`. A zero scale stays 0 (growth multiplies it), so from then on
   every step is skipped, each with a capsule, until the limit below stops the run.
   **Limit on skips in a row (v1.2 third round, HX102-03, owner decision
   E1109-06a).** A run that skips every step protects its weights but trains
   nothing and used to end with untrained weights as if it had succeeded (a NaN
   that rescaling cannot remove, a scale of 0, a loss that stays NaN).
   `RunnerConfig.max_consecutive_skips` (int >= 0, default 50; 0 turns the limit off)
   counts skipped training steps in a row: a non-finite loss (item 4), a non-finite
   gradient norm, a GradScaler skip without a backoff. A routine loss-scale backoff
   to a still positive scale does *not* count and does not reset the count (early
   AMP training backs off many times in a row and recovers); a backoff to 0 counts
   (a zero scale never grows again). A healthy step resets the count. At the N-th
   skip in a row the run writes a replay capsule and a `FailureSignature` of kind
   `Consecutive_Skips_Exceeded`, locus `train_step_skip_limit` (a failed write is only
   logged), and raises `ConsecutiveSkipsError`. Distributed it is a held step error
   (item 6), agreed at the existing agreement points with no extra collective; the
   count is equal on every rank because every skip is agreed (items 3, 4 and the
   check above), so every rank raises its own error, and a rank whose count differed
   would still stop every rank (its peers raise `PeerRankError`), never leave alone.
   The count is part of both resume checkpoints (`extra["consecutive_skips"]` at the
   epoch end and mid-epoch, `spec/MID_EPOCH_RESUME_v1.md`) and restored on resume,
   so a resume does not step around the limit; a checkpoint without the field
   (written before the limit existed) resumes with 0, a value that is not an int
   >= 0 is refused. The field is part of the configuration digest (section 2).
   A GradScaler that keeps finding inf halves the scale on every step without
   counting; from 65536 it reaches 0 after about 165 backoffs (the smallest float32
   is 2^-149), and then the limit stops the run within N more steps.
   Verified on 3ba3835 (Python 3.12.3, torch 2.6.0): with a zero scale and with
   gradients of 3e19, `unscale_` reports `found_inf` 0 while the norm is NaN resp.
   inf -- the premise of HX102-02/-04 holds on 2.6.0 too.
   After every applied step all parameters must be finite, else
   `NonFiniteParameterError` (the step already wrote them: no rollback) before the
   EMA update and before any checkpoint; distributed it is a held error (item 6).
   The check is exact and cannot overflow: the max-abs norm (`ord=inf`) of a tensor
   is finite iff every element is (3e38 in float32 stays finite, NaN and +-inf
   propagate; checked for fp16, bf16, fp32, fp64), one `torch._foreach_norm` per
   (device, dtype) group as in `clip_grad_norm_`, empty tensors left out. Cost: one
   read of all weights per step, measured on the CPU at about 20 % of a small-batch
   step (batch 64) of a 10.5M-parameter MLP (a first version with `isfinite().all()`
   per tensor: 23-29 %); one fused kernel per group on CUDA, not measured here.
   `RunnerConfig(check_finite_params=False)` turns it off (a non-finite weight is
   then only seen as a NaN loss on the next step, which is skipped, not stopped);
   the field is part of the configuration digest (section 2).
   Distributed, the decision needs no extra collective: it is taken after DDP's
   gradient all-reduce, and every rank reads the same reduced gradients, so a
   NaN/Inf on one rank's local gradient is in every rank's reduced gradient and
   every rank skips. That claim is checked, not assumed: the agreement points of
   item 6 carry one more count (ranks that skipped the previous step's update), and
   a count strictly between 0 and `world_size` raises `DistributedContractError` on
   every rank before anything reads the now differing weights (no further batch,
   no mid-epoch save; the epoch-end save comes after `_agreed_next` returned None).

## 4. Validation: exact partition, sufficient statistics

1. **Exact partition.** The indices the validation loader will visit on each rank
   (from its batch sampler, else its sampler) must together cover
   `range(len(dataset))` exactly once: one int64 all-reduce of the per-index
   counts; any count != 1 or an index outside the range raises
   `DistributedContractError` on every rank. An `IterableDataset` is refused.
   `DistributedSampler(drop_last=False)` pads with repeated samples when the
   length is not a multiple of the world size (R89 D1: n=5, world 2 gives
   `[[0, 2, 4], [1, 3, 0]]`) and is refused then; `drop_last=True` drops samples
   and is refused as well. `ExactDistributedEvalSampler` gives rank r the indices
   `r, r + W, ...` (no padding, no shuffle; ranks may differ by one sample) -- the
   same pattern as Lightning's `UnrepeatedDistributedSampler` and
   pytorch-pfn-extras' `DistributedValidationSampler`, which also exist because
   the stock sampler duplicates evaluation samples. The index sequence is read
   twice and must be equal: the check sees one iteration of the sampler, the loop
   runs another, so a sampler that changes between iterations is refused.
2. **Sufficient statistics.** Each rank sums `[loss_sum, n_batches, sq_err, sq_ref]`
   in float64; one all-reduce adds them and rank 0's result is broadcast (NCCL
   describes all-reduce results as identical on every rank, but no backend
   documents that bit for bit for every algorithm; the best-checkpoint decision
   must see the same bytes everywhere, by construction); then
   `val_nrmse = sqrt(sq_err / sq_ref)` and `val_loss = loss_sum / n_batches`.
   The pooled nRMSE does not depend on how the samples were split; a mean of rank
   nRMSEs does (R89 D2: `(9, 4)` and `(1, 96)` give 0.801 vs. pooled 0.316).
   `val_loss` is a mean of batch means and depends on batching, as without DDP.
   NaN/inf propagate through the sum, so every rank sees the same non-finite
   value and takes the same skip path (section 5). `sq_ref = 0` keeps its
   single-process meaning (NaN for 0/0, inf for e/0).
3. **No collective inside the loop.** Validation runs on the unwrapped module or
   its EMA copy; ranks may have different batch counts. A local exception during
   validation is agreed (section 5) before the reduction.
4. **One model.** Before validation every buffer of the unwrapped module and of
   the EMA copy is broadcast from rank 0, after checking that the buffers' names,
   dtypes and shapes agree (a broadcast of differently sized buffers is no exception
   but a process abort -- gloo `EnforceNotMet`, SIGABRT -- e.g. a non-persistent cache
   following the last batch under `broadcast_buffers=False`; review round 2).
   *Behaviour change, deliberate:* such a setup (rank-dependent buffer shapes) ran
   before this contract; it is now refused at the first validation, with a
   `DistributedContractError` on every rank -- a pooled metric over different
   buffers would belong to no single model, and the broadcast would abort. DDP broadcasts rank 0's buffers at the start of each forward, but the
   training forward then updates them with the local batch (BatchNorm running
   statistics) and the EMA copy averages those local buffers: without the sync the
   ranks evaluated different models, the pooled metric belonged to none of them,
   and `best.pt` (rank 0's) was not the model the metric scored (review finding 2,
   2 ranks / 3 epochs: 0.91806 pooled vs. 0.92015 for rank 0's EMA). The
   following checkpoint save therefore also writes the model every rank holds.

## 5. Checkpoints and errors

`DistributedRunContext.run_agreed(fn)` runs `fn` on every rank, then all-reduces
a failure flag: the failing rank re-raises its own exception, the others raise
`PeerRankError`. The closing all-reduce is also the barrier.

**Every agreed failure** (construction, batch preparation, forward/loss, a held
step error, `run_agreed`) goes through `raise_agreed_failure` (v1.1 review round):
once an agreement found a failure, the failing rank logs its exception with
traceback, then every rank takes part in one more int64 all-reduce of
`world_size` one-hot entries (`failed_ranks`, error path only, itself under
`keeping`: a peer lost during this diagnostic step must not replace the root cause,
Codex P2 on #97), so every rank knows *which* ranks failed. The failing rank adds a note (`[rank r] <what>: agreed failure on rank(s) [..]`) and
re-raises it; the others raise `PeerRankError("<what> failed on rank(s) [..] (see
their logs)")`. Before, the message said "see that rank's log" without naming the
rank, and nothing had been logged there. If the agreement collective itself fails
(a peer process died, the process-group timeout), `keeping` re-raises this rank's
own exception with a note naming the failed agreement (the transport error stays
attached as `__context__`): the root cause is no longer replaced by
"Connection closed by peer".

- **Saves.** Rank 0 writes `best.pt`, `last.pt`, `resume_last.pt`
  (`CheckpointIntegrationManager(is_main_process=...)`); every rank runs the same
  best-policy update. The whole save step is one `run_agreed`: no rank continues
  before rank 0's write returned, and a failed write (R89 D4) fails every rank.
- **Run identity.** Rank 0's `run_id` is broadcast at construction (16 bytes), so
  `best.pt` and the resume record name one run.
- **Failure records.** Each rank writes FailureSignatures and replay capsules to
  `failures/rank{r}/` (the file name is step + kind; ranks on a shared file system
  would overwrite each other).

## 6. Resume

Every rank loads `config.resume_from` itself (shared file system assumed). The
load is one `run_agreed` (a missing or unreadable file on one rank fails all).
**Per-rank RNG streams.** The resume payload records the RNG of the process that
writes it (rank 0). Before each resume save, every rank's torch-CPU, Python and
NumPy (legacy MT19937) RNG states and the states of its loaders' explicit
`torch.Generator`s (names checked equal first) are all-gathered as fixed-length
tensors (`rng_state_tensors`; floats as IEEE-754 bits, exact) into
`extra["rank_rng_states"]`; on resume each rank restores its own after the
manager restored rank 0's (review finding 5: per-rank shards shuffled with
per-rank generator seeds continued rank 0's permutation). The same field carries
every rank's anomaly-detector loss history (`anomaly_history`, encoded as
`[count, values..., NaN padding]` of length `spike_window_size + 1`, since a spike
kept out of the window on one rank makes the counts differ): each rank checks its
own loss, and a resumed rank would otherwise start the spike median from an empty
window (review of #94). With another world size, or a checkpoint without the
field, the rank-0 states stay, the history window starts empty,
`rank_rng_restored = False` and a warning is logged.
Afterwards the resumed epoch, global step and best metric value (bitwise,
NaN for `None`), the mid-epoch position (`spec/MID_EPOCH_RESUME_v1.md`, -1 at an
epoch boundary), the SHA-256 of the adopted `run_id`, and a SHA-256 over the
names, dtypes, shapes and bytes of the loaded model state must equal rank 0's,
else `DistributedContractError` on every rank (same path, different bytes: no
shared file system, a stale copy).
**v1.3: every shared state, named (at construction and after a resume).** The weights
are not the whole state the ranks share. Equal weights with another optimizer state (momentum buffers), learning-rate
schedule, EMA average or `GradScaler` state pass a weights-only check and diverge the
ranks silently from the first step: DDP averages the gradients, each rank then
applies them with its own momentum or rate; with unequal loss scales DDP even
averages gradients scaled differently and each rank unscales them with its own
scale. Five SHA-256 digests are now computed in one `run_agreed` (a failure on one rank --
a `state_dict()` that raises, an OOM while copying a GPU state -- raises there and
`PeerRankError` on the peers, instead of leaving them in the broadcast) and compared
with rank 0's in one broadcast
(160 bytes) and one 5-element int64 all-reduce (the count of ranks that differ, per
component). Collectives per check: the 1-element all-reduce of `run_agreed`, the
broadcast with its own 1-element flag all-reduce, the 5-element all-reduce (v1.2: the
same without the `run_agreed` step; the first v1.3 text said "as many as before",
which held until the third round added the agreement of the digest computation): the model state
(as before), `optimizer.state_dict()`, `scheduler.state_dict()`,
`grad_scaler.state_dict()` and the EMA module's state (`n_averaged` included). A
component the run does not use is digested as `None` on every rank. The error
names the components that differ (`resumed state differs from rank 0 on at least
one rank: optimizer state`). The same agreement runs once at construction, after the
config digest (section 2): the same `RunnerConfig` does not make the same state, and
an optimizer built with another learning rate or momentum, another scheduler, or a
pre-loaded optimizer state or loss scale on one rank would diverge the ranks from the
first step (`initial state differs from rank 0 on at least one rank: optimizer state,
scheduler state` -- a learning rate also shows in the scheduler's `base_lrs`). Since
v1.5 it also runs at the end of every epoch and before every mid-epoch save (buffers
left out there, see the version note): every rank reaches both points, because the epoch
ends and the step count are agreed, so the collectives keep one order on every rank. The
optimizer/scheduler/scaler digest is canonical and exact and every value is typed:
tensors (any device, any dtype incl. bf16/fp8/complex/bool) by dtype, shape and bytes,
NumPy arrays by their dtype schema (v1.4, `_dtype_token`: byte order, kind and size;
for a structured dtype the itemsize, alignment and every field in order with name,
title, offset and dtype; for a subarray its shape and base -- `dtype.str` alone is
`|V<n>` for every structured dtype of one size), shape and bytes; a dtype carrying
`metadata` raises `TypeError` (arbitrary Python state outside the bytes), sparse tensors (COO, CSR, CSC, BSR, BSC) by their coalesced COO indices
and values (an uncoalesced tensor equals its coalesced form, CSR equals COO of the same
value) (object arrays by their elements),
NumPy scalars by dtype schema and bytes, floats by `float.hex` (one ulp and -0.0 count),
complex numbers by both parts, ints, bools, strings, bytes and `None`; dict keys and
set elements sorted by their encoding (string hashing and with it set order differ
between processes: `PYTHONHASHSEED`). Values are admitted by the identity of their
exact type (R8, 2026-11-11): any other object -- also a subclass of an admitted type,
whose `hex()`, `__format__` or `items()` may write other bytes -- raises `TypeError`
(agreed like any digest failure). Before R8 it counted by its type name only (never by
`repr`, which may carry a memory address): two classes of one name, or one class with
other hidden attributes, agreed (cross-family GPT-6 A7-R2, "opaque fallback").
`collections.Counter` is admitted (MultiStepLR keeps its milestones in one). The
weights and EMA digests include a module's non-tensor `get_extra_state()` entries the
same way (before R8 they were skipped). Measured without a false refusal: 6
optimizers (SGD with Nesterov, Adam with AMSGrad, AdamW, centred RMSprop, Adagrad,
LBFGS) x 11 schedulers (none, StepLR, MultiStepLR, LambdaLR, OneCycleLR,
CosineAnnealingWarmRestarts, ReduceLROnPlateau, SequentialLR, ChainedScheduler,
CyclicLR, PolynomialLR): equal state built twice or reloaded digests equally, one
more step digests differently (a 6 x 6 subset is a unit test).

**Which states are comparable (R8, 2026-11-11, cross-family GPT-6 R8-01).** Equal
`state_dict()` bytes mean equal future training only if the state holds everything
`step()` reads. Before the digest, `_comparable_state` therefore admits the optimizer,
scheduler and `GradScaler` only if `type(obj)` *is* one of the built-in classes held as
objects at import -- not by `__module__`/`__qualname__` (writable: a class built as
`type('StepLR', ..., {'__module__': 'torch.optim.lr_scheduler'})` carries both names),
not by `isinstance` (a subclass, or a `__class__` property), not by set membership (a
metaclass `__eq__`/`__hash__`). Admitted: the optimizers Adadelta, Adafactor, Adagrad,
Adam, Adamax, AdamW, ASGD, LBFGS, NAdam, RAdam, RMSprop, Rprop, SGD, SparseAdam; the
schedulers StepLR, MultiStepLR, ConstantLR, LinearLR, ExponentialLR, PolynomialLR,
CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR,
SequentialLR, ChainedScheduler (the inner schedulers of the last two pass the same check,
in a plain list or tuple); `torch.amp.GradScaler`. Oracle (unit tests, torch 2.6.0): a
state saved and loaded into an instance built with other arguments continues with the
original's learning rates (every admitted scheduler) or updates (every admitted
optimizer). Refused: `LambdaLR` and `MultiplicativeLR` (a function is not in their
state: 0.0531 vs 0.0262 after 3 steps with two lambdas), `CyclicLR` (a custom
`scale_fn` is not in its state), every subclass and look-alike; an instance attribute
that shadows a method (`state_dict`, `get_lr`, `step`; for the optimizer the
instance-level `step` that torch's own schedulers install is tolerated); an optimizer
with a `state_dict` or `load_state_dict` pre/post hook (it rewrites the exported or
loaded state); a class whose behaviour-carrying methods (`step`, `state_dict`,
`load_state_dict`, `get_lr`, `_get_closed_form_lr`, `get_last_lr`, `zero_grad`,
`update`, `scale`, `unscale_`, resolved through the MRO) are no longer the objects
held at import (R8.1, Quella F2-A 2026-11-11: `type(obj) is SGD` still holds after
`SGD.step = ...`). torch's own `profile_hook_step` wrapper around `step` is admitted
only if its code object is torch's wrapper and its closure holds the function held at
import. **R12 (007acc2 R10.1/R11, issue #106) replaces the method list by a behaviour
graph** (`_HELD`, `_behaviour_changed`): every callable or descriptor of every class of
the MRO (except `object`), the MRO and the metaclass; for every function reached, its
code object, globals dict, defaults, kwdefaults, closure cells and the value of every
name its code may load as a global (`co_names`, also names absent at import), followed
without a depth limit; and torch's lazy `fn.__dynamo_disable` cache. A function that
differs from the held one passes only as torch's own wrapper around it, by code object
AND globals dict AND the wrapped cell (`profile_hook_step`; dynamo's `DisableContext`
wrapper with `callback=None`, installed by `torch.compile`); the instance-level `step`
only as torch's LR-scheduler wrapper of the held class step for this optimizer. A
registered optimizer step pre/post hook, global or on the instance, is refused. The
graph must hold `SGD._init_group`, `Optimizer._optimizer_step_code`,
`Optimizer.profile_hook_step`, `GradScaler._maybe_opt_step` and
`torch.optim.sgd.{sgd,_single_tensor_sgd,_multi_tensor_sgd,_fused_sgd}` (`_REQUIRED_PINS`),
else every admission is refused. The check runs at the shared-state agreement AND
immediately before every optimizer step of a distributed run (`_admit_step_behaviour`).
The refusal is a
`DistributedContractError` inside the agreed digest computation, so a refusal on one
rank stops every rank (2-rank job: `lambda_lr_other_function_on_rank1`,
`forged_step_lr_on_rank1`, `state_dict_error_rank1`); before a step it is agreed
before any rank steps (v1.4, section 3 item 6; until v1.4 it was held, so the other
ranks had taken that step when the run stopped -- 2-rank job `optimizer_patch_rank1`),
and before `scheduler.step()` at every epoch's end it is agreed the same way
(`run_agreed`; 2-rank job `scheduler_patch_rank1`). Not covered: a class patched before the runner module was
imported, torch itself (in-process identity is no supply-chain attestation: pin and
verify the wheel), C/ATen kernels and builtins (except module globals shadowing them),
attributes reached through a module or class object that is not held (`torch.foo`),
mutable data (state, not behaviour), hostile code inside the rank process (no
isolation), and the model class (user code; its weights are compared).
Rank-local state is
excluded by design: RNG streams and loader generators (restored per rank above), the
anomaly history, and mid-epoch rank buffers (copied in after this check,
`spec/MID_EPOCH_RESUME_v1.md`).

## 7. Collectives used

Only tensor collectives (`all_reduce`, `broadcast`, `all_gather`), never the
`*_object` variants, which unpickle peer data. Tensors live on the CPU; on the
NCCL backend on the runner's CUDA device (`device="cuda:k"`), else on the current
CUDA device (`torch.cuda.set_device(local_rank)` first). Only the default process
group is used. Cost per training step: a 4-element and a 1-element all-reduce (v1.0: 2 + 1;
the third element is the held-step-error count, section 3.6, the fourth the
skipped-update count, section 3.7), plus one 2-element all-reduce before the update
(v1.4, section 3 item 6; with the behaviour check about 0.6 ms on CPU for SGD, the
check measured with torch 2.6.0 on the container CPU). Per epoch with a scheduler:
one 1-element all-reduce before `scheduler.step()` (v1.4). Per mid-epoch save: one more 2-element all-reduce
before the gathers. Per agreed failure (error path only): one
all-reduce of `world_size` int64 to name the failing ranks. Per validation:
broadcast of every buffer of the model and the EMA copy;
one all-reduce of `len(dataset)` int64 counts (8 bytes per sample), a few
1-element flags, one all-reduce + broadcast of 4 float64. Once at construction, once
per resume and (v1.5) once per epoch end and per mid-epoch save: a 1-element all-reduce (digest computation agreed), a 160-byte
broadcast and a 5-element int64 all-reduce (the shared state, section 6). Their local cost is hashing: every rank copies its weights, EMA,
optimizer and scheduler state to the CPU one tensor at a time and runs SHA-256 over
it -- measured 2026-11-10 on the container CPU (torch 2.6.0, fp32, 50.4M parameters,
Adam): weights 0.17 s, Adam state 0.34 s (about 1.1 GB/s), linear in the bytes, plus the
device-to-host copy on a GPU. Per resume save: three
all-gathers of ~10 KB per rank. Rank 0's checkpoint write happens while the
peers wait in the closing all-reduce: a write longer than the process-group
timeout (gloo default 30 min, NCCL 10 min) ends the job.

## 8. Evidence and limits

Evidence v1.2 (2026-11-09, Linux container, Python 3.13, torch 2.6.0, CPU, gate log
`results/2026-11-09_non-finite-gradient_local-gate.log`): without DDP a NaN and an
Inf gradient (finite loss) leave weights and EMA bitwise at their start values, no
saved checkpoint holds a non-finite tensor, the FailureSignature names step 0 and
`NaN_or_Inf_Grad`, the capsule holds the pre-update weights; a single bad step is
skipped and the others train; an optimizer that overflows a weight raises
`NonFiniteParameterError` with no checkpoint written and the EMA untouched. With 2
and 3 gloo ranks a NaN gradient from rank 1's batch is skipped on every rank
(weights finite and bitwise equal, `n_anomalies` `[1, 0]`, mid-epoch saves every
step); an injected split (rank 1 skips, the others step) raises
`DistributedContractError` on every rank, with no checkpoint, also right before a
mid-epoch save. All of these tests were red against cd5f6dc (5 single-process and 11 multi-rank
failures; the job desynchronised and the 2-rank mid-save case had no result, which the
test now reports instead of skipping); 7 new mutants, all killed (full probe 167/167).
Second round (same day, self-review of PR head 1f06120): with a CPU `GradScaler` a
NaN gradient skips step and EMA (`n_averaged` 1 after one skipped and one applied
step, scale 1024 -> 512, no FailureSignature), red against 1f06120; a scale that grows
after every step (`growth_interval=1`) is no skip and a power-of-two scale reproduces
the unscaled run bitwise (positive control, green on both heads); an overflowing norm
of finite gradients (1e25) is skipped; a mid-epoch resume across a gradient skip
equals the uninterrupted run bitwise (red against cd5f6dc); the parameter check has no
false alarm at 3e38 (float32) / 1e308 (float64), ignores an empty parameter and names
the NaN parameter in a mixed-dtype model; `check_finite_params=False` opts out; a
disabled `GradScaler` skips like no scaler (without the guard: `NonFiniteParameterError`
with NaN weights).

Evidence v1.1 (2026-11-02, Linux container, Python 3.12, torch 2.6.0, CPU): an
optimizer step that raises on rank 1 in the last step of an epoch, and one that
raises right before a mid-epoch save, end on both ranks with the injected exception
(rank 1) and `PeerRankError` (rank 0), and rank 0 writes no checkpoint of the failed
step; an EMA update that raises on rank 0 ends the same way. Before the fix the same
scenarios desynchronised the ranks (test red against main f414a98, gate log
`results/2026-11-02_ddp-post-step_local-gate.log`). A **3-rank** gloo job (train
12 samples, global batch 3 x 2; validation 5 samples split 2/2/1 by
`ExactDistributedEvalSampler`) reproduces a single-process run with batch 6 within
1e-12 (weights, EMA, `val_nrmse`, `mean_loss`), all three ranks agree bitwise, and
the forward-error, NaN-skip and post-step-error agreements end on all three ranks.
Review round (same day): with 2 and 3 ranks every peer's `PeerRankError` names the
failing rank (`[1]`, also the middle rank of three, which pins the one-hot index),
the failing rank's own exception carries the note; with world size 1 an injected
`unscale_` error is held (since v1.4 agreed before the step and raised at once), and an agreement collective that raises leaves the own
exception (with note) as what the caller sees. These tests were red against PR
head 8448003 (22 failures, 5 errors) and are green now. Codex P2 on d2e7e9e: the
rank-identification all-reduce ran outside `keeping`; a test injecting a failure
there was red (the transport error replaced the `KeyError`) and is green now.

Evidence v1.0 (2-rank gloo, CPU, float64; `src/toolkit/test_ddp_hex_v3.py`): the DDP run (global batch 2 x 2,
`ExactDistributedEvalSampler` on 5 validation samples) reproduces a single-process
run with batch 4 for weights, EMA, `val_nrmse` and `mean_loss` within 1e-12, and
the two ranks agree bitwise; a 2 + 2 epoch resume with a shuffling
`DistributedSampler` equals the 4-epoch run bitwise, also with dropout and
rank-specific seeds and loss weights drawn from the Python and NumPy streams,
with per-rank loader generators on per-rank shards, and with a BatchNorm model;
with a BatchNorm model both ranks hold bitwise equal buffers (model and EMA) after
training and the pooled `val_nrmse` equals rank 0's EMA model on the whole
validation set (<= 1e-12); under fault injection (rank 1's float64 all-reduce
results moved by one ulp) both ranks still decide on identical validation values;
each agreed failure (also: an error before the forward of a BatchNorm model,
`resume_from` on one rank only, a `len()` that lies) ends on both ranks with the
expected exception (none hangs). World size 1 reproduces the
non-distributed runner bitwise.
Mutation probe: 35 mutants, each switching one guard of sections 2-6 off, all
killed (`scripts/mutation_probe.py`, full probe 120/120 on 2026-10-30, exit 0; the
first full probe after the review round left the two `len()` fast-fail mutants alive,
the test now checks the fast fail's own message).

v1.4 self-review (2026-11-12, same environment): `pre_step_collective failure` raised,
not held, and the norm-split refused before any step (both red on 3b11876); the
GradScaler check on a skipped step and the 3-rank middle-rank refusal (both red on
4e6bcf0, green on 3b11876: they pin what 3b11876 introduced). The GradScaler's own
skip inside `step()` on one rank (`scaler_skip_split_*`, 2 and 3 ranks) is only known
after the step and is still caught by section 3.7's agreement before the next batch
or the mid-epoch save. All multi-rank jobs also pass under
`TORCH_DISTRIBUTED_DEBUG=DETAIL` (82 tests, no mismatch); a positive control (a
1- vs 2-element all-reduce) is reported there at once as "Detected mismatch between
collectives", where plain gloo waits for the timeout.

v1.5 (2026-11-13, Windows, Python 3.12.10, torch 2.6.0+cpu, 2- and 3-rank gloo): WP1 red
on 47cda7f (rank 1 halves its gradients after the all-reduce: clip_grad_norm_ patch, a DDP
comm hook on one rank, before a mid-epoch save, the middle rank of 3; the ranks trained
apart to the end and rank 0 saved), green on 7d8875e with unpatched positive controls. WP2:
a profile_hook_step wrapper around the held `SGD.step`, made in the same globals while
`__builtins__` pointed at a forged mapping, was admitted; now refused, torch's own wrapper
still passes. Discrimination DK-20261113T103102-a4e0c31550f0 at c06834b: 6/6 (the
buffers-included mutant fails the mid-epoch resume test before its 4 subtests, by the
intended false refusal). `test_ddp_hex_v3.py`, `test_runner_hex_v3.py`,
`test_mid_epoch_resume_hex_v3.py` at c06834b: 233 passed, 298 subtests passed. The full
mutation probe (253) has not been run for v1.5.

v1.4 (R12 phase 2, 2026-11-12, Linux container, Python 3.12, torch 2.6.0): six new or
changed tests, all red against 4e6bcf0 (run there with the new test file and a
two-name import shim, the shim only in a scratch worktree): the structured-dtype
schema (8 collisions, all with equal bytes), dtype metadata refused, the schema id,
the 2-rank `optimizer_patch_rank1` (red: rank 0 stepped, weights changed, rank 1
did not; now neither steps and both report `global_step` 0), the 2-rank
`scheduler_patch_rank1` (red: both ranks ran to the end, the learning rates parted),
and the world-1 `unscale_` error agreed before the step (red: held). The 2-rank
split-skip injection (section 3.7) now makes rank 1's gradient norm read NaN through
the real update path; replacing `_apply_update` on one rank bypassed the new
collective, a schedule no real run has. The failing-step harness injects at the step
call (`_step_optimizer`) after the agreement.

Not covered (open):
- **Pre-step agreement window:** between the agreement and the step nothing runs but
  the local return from the collective; a change made by another thread in that
  window is not seen (in-process hostile code is out of scope, section 6).
- **CUDA RNG per rank** (`torch_cuda_random_state_all` of rank 0 is restored on
  every rank's devices); DataLoader worker RNG; a custom (non-legacy) NumPy
  `Generator` held by user code.
- **Endless skips** (closed in v1.2 third round): `max_consecutive_skips` (section 3.7)
  stops a run that skips every step. Still open: a GradScaler backoff to a positive
  scale never counts, so a run whose scale keeps dropping skips about 165 steps from
  65536 before the count starts; with `max_consecutive_skips=0` nothing stops it.
- **NCCL/CUDA, more than 3 ranks, more than one host**: not run (2 and 3 gloo ranks
  on one host are). **AMP under DDP** (v1.3, partly): the runner builds a
  `GradScaler` and autocast only on CUDA; the 2-rank gloo job injects
  `torch.amp.GradScaler("cpu", init_scale=1024, growth_interval=2)` on every rank.
  Measured: both ranks end with bitwise equal weights, EMA and scale, equal to the
  single-process run with the same scaler (<= 1e-12; the scale grew from 1024 to
  8192); a NaN gradient on rank 1 only is spread by the all-reduce, both ranks
  back off once (no capsule, counted as one skip) and stay bitwise equal; a resume
  with a halved scale on rank 1 is refused on both ranks (section 6). Not covered:
  autocast (fp16/bf16 forward) under DDP, a CUDA `GradScaler`, NCCL.
- ~~**Mid-epoch resume**: resume stays at epoch boundaries.~~ Done 2026-10-30,
  `spec/MID_EPOCH_RESUME_v1.md` (this line had not been updated, v1.1).
- **Errors before the forward are agreed (section 3.3), errors inside DDP's forward
  before the module runs are not** (none known besides the buffer broadcast itself).
- **SyncBatchNorm / models with collectives of their own in forward** are untested;
  a rank that fails inside such a forward hangs its peers like an error in backward.
- **A rank that raises inside backward** (e.g. an OOM in backward) still leaves its
  peers in the gradient all-reduce until the process-group timeout; the NCCL
  watchdog (`TORCH_NCCL_ASYNC_ERROR_HANDLING`) / the gloo timeout govern that case.
  Forward and loss errors (section 3.4) and errors after backward (section 3.6,
  v1.1) are agreed. A `KeyboardInterrupt` or other non-`Exception` is never held.
- **Old DDP checkpoints** written by the runner before this contract with a
  DDP-wrapped model carry `module.`-prefixed keys and do not load strictly into the
  unwrapped module (no such run exists in this repo).
- Training batches follow `DistributedSampler` semantics: with `drop_last=False`
  a few samples are seen twice per epoch (gradient weight), which the contract
  accepts for training and refuses for validation.
