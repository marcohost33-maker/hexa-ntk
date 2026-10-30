# MID_EPOCH_RESUME_v1 — resume the training runner inside an epoch

**Version:** 1.0 (2026-10-30)
**Status:** ACTIVE for `HexTrainingRunner` with `RunnerConfig(save_every_n_steps=n > 0)`.
Infrastructure contract; it trains no model for the Phase 1 study and makes no learning
claim. Single process and distributed (`spec/DDP_RUNNER_v1.md`).
**Implementation:** `src/toolkit/runner_hex_v3.py` (`_save_mid_epoch`,
`_check_mid_epoch_record`, `_train_one_epoch`).
**Tests:** `src/toolkit/test_mid_epoch_resume_hex_v3.py`; 2-rank DDP scenario
`mid_epoch_resume` in `src/toolkit/test_ddp_hex_v3.py`; mutants in `scripts/mutation_probe.py`.
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
`num_workers=2` with a dataset transform that draws from the worker RNG, and 2 DDP ranks
with BatchNorm, dropout and rank-specific seeds.

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

Single process it also records the anomaly detector's loss history (also at epoch
boundaries now). Distributed, the per-rank fields are all-gathered (fixed-length tensors,
names checked equal first), every rank's current RNG and generator states go into
`extra["rank_rng_states"]` (`spec/DDP_RUNNER_v1.md` section 6) and the save is one
`run_agreed`. The buffers are *not* synchronised before a mid-epoch save: EMA buffers never
feed back into training, DDP overwrites the model's buffers at every forward, and the sync
before the next validation (DDP_RUNNER section 4.4) equalises them; a sync here survived as an
equivalent mutant and was removed.

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
`batches_done`, uint8 RNG states. A record written by another world size is refused: the
ranks' sample partitions, and with them the batches already done, would not correspond
(resume from an epoch-boundary checkpoint then). Distributed, the resumed position is part of
the rank-0 agreement after the load.

## 4. Limits

- **`persistent_workers=True`**: the iterator, and with it `_base_seed`, is created once per
  run, so worker RNG streams after a resume differ from the uninterrupted run's (the order
  of samples is still reproduced). Not tested.
- **CUDA RNG** per device is not part of the epoch-start record; **custom samplers** must be
  deterministic given the RNG/generator state (stock samplers are).
- **Gradient accumulation** does not exist in the runner; a save always falls on an optimizer
  step.
- **Equality is bitwise only on the same machine and software** (PyTorch documents no
  cross-version or cross-hardware reproducibility); the fast-forward costs `batches_done`
  batch loads.
- **Torn writes**: the mid-epoch save uses the same atomic writer as every resume save.
