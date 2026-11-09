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
6. **After backward (v1.1).** `GradScaler.unscale_` (AMP), clipping, the optimizer
   (or `GradScaler`) step and the EMA update run per rank after DDP's gradient
   all-reduce has completed; none of them is a collective. A rank that raised there used to leave alone: its peers
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
   no-scaler path; without that guard its NaN update reached the weights.
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

## 7. Collectives used

Only tensor collectives (`all_reduce`, `broadcast`, `all_gather`), never the
`*_object` variants, which unpickle peer data. Tensors live on the CPU; on the
NCCL backend on the runner's CUDA device (`device="cuda:k"`), else on the current
CUDA device (`torch.cuda.set_device(local_rank)` first). Only the default process
group is used. Cost per training step: a 4-element and a 1-element all-reduce (v1.0: 2 + 1;
the third element is the held-step-error count, section 3.6, the fourth the
skipped-update count, section 3.7). Per mid-epoch save: one more 2-element all-reduce
before the gathers. Per agreed failure (error path only): one
all-reduce of `world_size` int64 to name the failing ranks. Per validation:
broadcast of every buffer of the model and the EMA copy;
one all-reduce of `len(dataset)` int64 counts (8 bytes per sample), a few
1-element flags, one all-reduce + broadcast of 4 float64. Per resume save: three
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
`unscale_` error is held, and an agreement collective that raises leaves the own
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

Not covered (open):
- **CUDA RNG per rank** (`torch_cuda_random_state_all` of rank 0 is restored on
  every rank's devices); DataLoader worker RNG; a custom (non-legacy) NumPy
  `Generator` held by user code.
- **A GradScaler that skips forever** (a NaN that rescaling cannot remove) shrinks the
  scale towards 0 and skips every step without stopping; the NaN-loss path has the same
  shape (every step skipped, the run ends with untrained weights). A bound on consecutive
  skips is an owner decision (default and whether it stops the run), not part of v1.2.
- **NCCL/CUDA, more than 3 ranks, more than one host, AMP under DDP**: not run
  (2 and 3 gloo ranks on one host are).
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
