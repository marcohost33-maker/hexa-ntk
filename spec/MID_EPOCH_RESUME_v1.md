# MID_EPOCH_RESUME_v1 — resume the training runner inside an epoch

**Version:** 1.0 (2026-10-30)
**Status:** ACTIVE for `HexTrainingRunner` with `RunnerConfig(save_every_n_steps=n > 0)`.
Infrastructure contract; it trains no model for the Phase 1 study and makes no learning
claim. Single process and distributed (`spec/DDP_RUNNER_v1.md`).
**Implementation:** `src/toolkit/runner_hex_v3.py` (`_save_mid_epoch`,
`_check_mid_epoch_record`, `_train_one_epoch`).
**Tests:** `src/toolkit/test_mid_epoch_resume_hex_v3.py`; 2-rank DDP scenarios
`mid_epoch_resume`, `mid_epoch_rank_buffers` and `resume_other_position_on_rank1` and the
world-1 rank-buffer check in `src/toolkit/test_ddp_hex_v3.py`; mutants in
`scripts/mutation_probe.py`.
**Origin:** ROADMAP "Mid-Epoch-Resume" (the loader state was saved but resume worked only at
epoch boundaries); cross-family GPT-6 R89/R90 P1 ("rank-local mid-epoch state coupled with
model/optimizer/RNG").

## 1. Claim

A run interrupted inside an epoch and resumed from its last mid-epoch checkpoint ends with
the same weights, EMA, epoch reports (`mean_loss`, `mean_grad_norm`, `n_batches`,
`n_anomalies`, `lr`), validation reports and anomaly-detector history as the run that was
never interrupted, bit for bit, on the same machine and software (the tests compare with
`==`). Shown for: shuffling through an explicit `torch.Generator` and through the global
torch RNG (no generator), dropout, loss weights drawn from the Python and NumPy streams,
a step skipped (NaN) before the save, `num_workers=2` with a dataset transform that draws
from the worker RNG, and 2 DDP ranks with BatchNorm, dropout and rank-specific seeds, and
with `broadcast_buffers=False` and a buffer the training forward reads and updates from
the local batch. Distributed, "the same" holds per rank: each rank's own parameters,
buffers, EMA and anomaly history.

## 2. What a mid-epoch save records

Every `save_every_n_steps` global steps the resume checkpoint (`resume_last.pt`, the same
file as the epoch-boundary save, the newest wins) is rewritten with the full resume state
(model, optimizer, scheduler, scaler, EMA, RNG, loader generators, best-metric record) and
`extra["mid_epoch"]`, version 1:

| Field | Content |
|---|---|
| `epoch`, `batches_done` | the running epoch and how many of its batches were consumed |
| `world_size` | ranks that wrote it |
| `losses`, `grad_norms` | per rank: the healthy losses and gradient norms so far (the epoch means are then summed in the same order as without the interruption) |
| `n_anomalies` | skipped steps so far (agreed across ranks, one value) |
| `epoch_start` | per rank: the global torch RNG state and every explicit loader generator state *at the start of the epoch* |
| `train_loader` | per rank: `[batches per epoch, batch size, dataset length]` of the train loader (-1 where undefined, e.g. an unsized stream) |
| `rank_buffers` | distributed with `broadcast_buffers=False` only: per rank, every persistent buffer of the model |

Every resume save (mid-epoch and at epoch boundaries) also records the anomaly detector's
loss history: single process in `extra["anomaly_history"]`, distributed every rank's in
`extra["rank_rng_states"]` together with every rank's current RNG and generator states
(`spec/DDP_RUNNER_v1.md` section 6). Distributed, the per-rank fields are all-gathered
(fixed-length tensors, names checked equal first) and the save is one `run_agreed`.

**Buffers.** They are *not* synchronised before a mid-epoch save: where a training forward
reads a buffer, a sync would make the save cadence change the run. EMA buffers never feed
back into training and the sync before the next validation (DDP_RUNNER section 4.4)
equalises them. The model's buffers: with `broadcast_buffers=True` (the DDP default) DDP
overwrites them with rank 0's at the next forward, so rank 0's in the payload are every
rank's. With `broadcast_buffers=False` each rank keeps its own; the payload holds only rank
0's, so every rank's persistent buffers go into `rank_buffers` (names, dtypes and shapes
checked equal first) and each rank copies its own back in at the start of the resumed epoch.
Without that, a rank-local buffer read by the training forward changed the weights and both
reports (review of #94, 2-rank reproduction).

## 3. Resume: rebuild the iterator, fast-forward, continue

1. The resume restores the state as before (and, distributed, each rank's own RNG and
   generators); a mid-epoch record makes `start_epoch` the interrupted epoch instead of the
   next one, and the RNG state at the interruption is kept aside.
2. At the start of that epoch the runner sets the global torch RNG and the loader generators
   back to their recorded epoch-start states and creates the iterator: it draws the same
   sampler permutation and the same worker base seed (`_BaseDataLoaderIter` draws
   `_base_seed` from `loader.generator`, or from the global RNG without one; `RandomSampler`
   draws its seed lazily at the first `next()`; torch 2.6 source).
3. **Fast-forward:** the first `batches_done` batches are loaded and dropped. Loading, not only
   advancing the sampler, replays every worker's RNG consumption, so the later batches come out
   identical. (torchdata's `StatefulDataLoader` instead snapshots sampler and worker state;
   it is an optional differential lane here, `scripts/oracles/stateful_loader_resume_oracle.py`.)
   Cost: `batches_done` batch loads. A loader that yields fewer batches raises `ValueError`
   (distributed: inside `run_agreed`, so on every rank).
4. Then the RNG state of the interruption is restored and training continues with the
   recorded losses, gradient norms and skip count; the per-epoch scheduler steps at the end of
   the epoch as before (a mid-epoch save precedes it).

The record is validated (`_check_mid_epoch_record`): version, epoch equal to the checkpoint's,
`batches_done >= 0`, `0 <= n_anomalies <= batches_done`, healthy losses + skips =
`batches_done`, uint8 RNG states, rank buffers that are tensors with the names, dtypes and
shapes of the model's persistent buffers. A record written by another world size or for a
train loader of another shape is refused: the ranks' sample partitions, and with them the
batches already done, would not correspond (resume from an epoch-boundary checkpoint then).
Before the shape was recorded, a resume with batch size 4 instead of 2 ran without an error
and its fast-forward consumed the whole epoch, so those samples never trained (review of
#94). An unsized loader records -1; only the fast-forward then sees a shorter stream.
Distributed, the resumed position is part of the rank-0 agreement after the load.

## 4. Limits

- **`persistent_workers=True` is refused** (`ValueError`, at construction with
  `save_every_n_steps > 0` and at the resume of a mid-epoch checkpoint): the iterator is created
  once per run and draws `_base_seed` only then, later epochs reuse it. The resumed run's fresh
  iterator draws that seed again from the restored epoch-start state, so not only the worker
  RNG but the sampler permutation differs, with or without an explicit loader generator
  (`test_persistent_workers_break_the_replay` pins this for torch 2.6). The same mechanism makes
  an *epoch-boundary* resume with persistent workers inexact; that path is not refused and
  carries no equality claim.
- **CUDA RNG** per device is not part of the epoch-start record; **custom samplers** must be
  deterministic given the RNG/generator state (stock samplers are).
- **Non-persistent buffers** are not state (`state_dict` leaves them out) at any resume; a
  model whose training forward reads a non-persistent buffer it also updates is not covered.
- **The loader shape** is `[len(loader), batch_size, len(dataset)]`; a changed sampler or
  transform with the same shape is not detected.
- **Gradient accumulation** does not exist in the runner; a save always falls on an optimizer
  step.
- **Equality is bitwise only on the same machine and software** (PyTorch documents no
  cross-version or cross-hardware reproducibility); the fast-forward costs `batches_done`
  batch loads.
- **Torn writes**: the mid-epoch save uses the same atomic writer as every resume save.
