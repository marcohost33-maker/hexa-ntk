# DDP_RUNNER_v1 — the training runner under DistributedDataParallel

**Version:** 1.0 (2026-10-30; hardened before merge the same day after a critical
self-review: agreed construction, adapter rule, agreed forward errors, deterministic
validation sampler, rank-0-bound decision sums, per-rank RNG streams on resume;
second round after an independent review of head a875020: errors before the forward
agreed, step count agreed per step, buffers synchronised from rank 0, config digest,
per-rank loader generators; third round after the review of 205f293: buffer
signatures include dtype and shape, OOM rescue while preparing, effective AMP mode
agreed, RNG/generator states moved to the CPU before restore)
**Status:** ACTIVE for `HexTrainingRunner` with `RunnerConfig(distributed=True)`.
Infrastructure contract; it trains no model for the Phase 1 study and makes no
learning claim (`spec/PREREGISTRATION_PHASE1_v1.md` stays DRAFT).
**Implementation:** `src/toolkit/runner_hex_v3.py` (runner),
`src/toolkit/ddp_hex_v3.py` (`ExactDistributedEvalSampler`, `DistributedRunContext`).
**Tests:** `src/toolkit/test_ddp_hex_v3.py` — no process group, an in-process gloo
group of world size 1, and a real 2-rank gloo job; mutants in `scripts/mutation_probe.py`.
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
   backward or the optimizer step (e.g. an OOM there) is not covered: the peers are
   already inside the gradient all-reduce (section 8).
   A step therefore costs two 1-element all-reduces besides DDP's own.
5. **Training metrics.** `mean_loss` and `mean_grad_norm` are sums over all
   ranks' healthy steps divided by their count. `n_batches` (healthy steps) and
   `n_anomalies` (skipped steps) are per rank and equal on every rank because the
   skips are agreed.

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
   following the last batch under `broadcast_buffers=False`; review round 2). DDP broadcasts rank 0's buffers at the start of each forward, but the
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
per-rank generator seeds continued rank 0's permutation). With another world size, or a checkpoint without the
field, the rank-0 states stay, `rank_rng_restored = False` and a warning is logged.
Afterwards the resumed epoch, global step and best metric value (bitwise,
NaN for `None`), the SHA-256 of the adopted `run_id`, and a SHA-256 over the
names, dtypes, shapes and bytes of the loaded model state must equal rank 0's,
else `DistributedContractError` on every rank (same path, different bytes: no
shared file system, a stale copy).

## 7. Collectives used

Only tensor collectives (`all_reduce`, `broadcast`, `all_gather`), never the
`*_object` variants, which unpickle peer data. Tensors live on the CPU; on the
NCCL backend on the runner's CUDA device (`device="cuda:k"`), else on the current
CUDA device (`torch.cuda.set_device(local_rank)` first). Only the default process
group is used. Cost per training step: a 2-element and a 1-element all-reduce. Per validation:
broadcast of every buffer of the model and the EMA copy;
one all-reduce of `len(dataset)` int64 counts (8 bytes per sample), a few
1-element flags, one all-reduce + broadcast of 4 float64. Per resume save: three
all-gathers of ~10 KB per rank. Rank 0's checkpoint write happens while the
peers wait in the closing all-reduce: a write longer than the process-group
timeout (gloo default 30 min, NCCL 10 min) ends the job.

## 8. Evidence and limits

Evidence (2-rank gloo, CPU, float64; `src/toolkit/test_ddp_hex_v3.py`): the DDP run (global batch 2 x 2,
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
Mutation probe: 34 mutants, each switching one guard of sections 2-6 off, all
killed (`scripts/mutation_probe.py`, full probe 119/119 on 2026-10-30, exit 0; the
first full probe after the review round left the two `len()` fast-fail mutants alive,
the test now checks the fast fail's own message).

Not covered (open):
- **CUDA RNG per rank** (`torch_cuda_random_state_all` of rank 0 is restored on
  every rank's devices); DataLoader worker RNG; a custom (non-legacy) NumPy
  `Generator` held by user code.
- **NCCL/CUDA, more than 2 ranks, more than one host, AMP under DDP**: not run.
- **Mid-epoch resume** (separate ROADMAP item): resume stays at epoch boundaries.
- **Errors before the forward are agreed (section 3.3), errors inside DDP's forward
  before the module runs are not** (none known besides the buffer broadcast itself).
- **SyncBatchNorm / models with collectives of their own in forward** are untested;
  a rank that fails inside such a forward hangs its peers like an error in backward.
- **A rank that raises inside backward or the optimizer step** (e.g. an OOM in
  backward) still leaves its peers in the gradient all-reduce until the
  process-group timeout; the NCCL watchdog / the gloo timeout govern that case.
  Forward and loss errors are agreed (section 3).
- **Old DDP checkpoints** written by the runner before this contract with a
  DDP-wrapped model carry `module.`-prefixed keys and do not load strictly into the
  unwrapped module (no such run exists in this repo).
- Training batches follow `DistributedSampler` semantics: with `drop_last=False`
  a few samples are seen twice per epoch (gradient weight), which the contract
  accepts for training and refuses for validation.
