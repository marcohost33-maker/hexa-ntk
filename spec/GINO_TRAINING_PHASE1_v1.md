# GINO_TRAINING_PHASE1_v1 — the pinned GINO through the training runner

**Version:** 1.0 (2026-11-19)
**Status:** PROPOSED. Training code only. No model is fit, tuned or evaluated on T1 data
under this contract; `spec/PREREGISTRATION_PHASE1_v1.md` (lines 14-15) forbids that until
the owner approves it and allows "writing training code and running smoke tests on
synthetic tensors". Every test here uses standard-normal tensors on lattice coordinates.
**Implementation:** `src/toolkit/gino_training_hex_v3.py`.
**Tests:** `src/toolkit/test_gino_training_hex_v3.py` (contract tests with a stand-in for
the backend; `TestRealBackend` with neuraloperator 2.0.0); 22 mutants in
`scripts/mutation_probe.py`.
**Evidence:** `scripts/oracles/gino_runner_integration_oracle.py` runs the real backend
through the runner without this module (the failures of section 1) and with it, and writes
`results/2026-11-19_gino_runner_integration.json`, bound to the source revision.
**Depends on:** `spec/GINO_ADAPTER_PHASE1_v2.md` (model, geometry, config digest),
`spec/DDP_RUNNER_v1.md` and `spec/MID_EPOCH_RESUME_v1.md` (runner), `spec/HEAT_OPERATOR_TASK_v1.md`
(metric `rel_l2`).

## 1. Why

The adapter builds the pinned GINO and the runner trains models, but nothing connected
them. Measured on main 18ab389 (Linux, Python 3.12.3, torch 2.6.0, neuraloperator 2.0.0),
with the smallest wrapper that makes the call possible (`_PlainWrapper` in the oracle, so
the table can be re-measured from the repo):

| # | Finding | Measured |
|---|---|---|
| 1 | No call path. The runner calls `model(x)`; `OperatorAdapterV3` calls `model(x, coords_cart=...)` for GINO. The backend takes `input_geom, latent_queries, output_queries, x`. | `GINO.forward` signature |
| 2 | With EMA (the runner default) the run stops at the first checkpoint. | `AttributeError: 'dict' object has no attribute 'detach'` in `_ema_state_snapshot` |
| 3 | Without EMA the run ends and writes `best.pt` and `last.pt`, and neither can be read with `weights_only=True`. | `Unsupported global: GLOBAL torch._C._nn.gelu` |
| 4 | A trusted resume (pickle) reads the file and then fails to load the model. | `Unexpected key(s) in state_dict: "_metadata"` |
| 5 | The task dataset yields `{"trajectory_id", "u0", "uT"}` in float64; the runner reads `{"input", "target"}`. | `HeatOperatorDataset.__getitem__` |
| 6 | The runner defaults are gradient clipping at 1.0, EMA with decay 0.9999 and best model by the pooled `val_nrmse`. The preregistration names none of them. | `RunnerConfig()` |

Findings 2-4 have one cause: `neuralop.models.base_model.BaseModel.state_dict()` adds a
`"_metadata"` entry holding the constructor arguments, among them the function
`torch.nn.functional.gelu`. It writes the key without the module prefix, so it also
appears in the state of any module that contains a GINO.

A Phase 1 run started without this layer would have stopped at the first checkpoint, or,
with EMA off, trained to the end and left weights no safe loader reads.

## 2. `GinoRunnerModule`

`GinoRunnerModule(config, lattice)` holds the model of `build_gino(config, N = 32 triangle)`
and the geometry of `lattice` as prepared by `prepare_gino_batch`.

- **Call.** `forward(x)` takes float32 `[batch, nodes]` and returns `[batch, nodes]`. It is
  bit-identical to `predict_gino(module.gino, lattice, x, config)` (tested with the real
  backend). Other shapes raise `ValueError`, another dtype raises `TypeError` (the backend
  computes in float32 whatever it is given; a silent cast would hide a float64 pipeline).
- **Radius.** Always fixed from the N = 32 triangle training lattice, as in the adapter,
  whatever lattice the module evaluates on.
- **Geometry.** `input_geom` and `latent_queries` are non-persistent buffers. They follow
  `.to(device)` and are not saved, so one set of weights loads into a module on any
  lattice. `predict(lattice, u0)` runs `predict_gino` on another lattice in eval mode
  without gradients and restores the mode.
- **State holds tensors only.** `state_dict()` removes the backend's `"_metadata"` entry
  (also when the module is nested, e.g. inside the EMA `AveragedModel`) and raises
  `TypeError` on any other non-tensor entry under its prefix. A `"_metadata"` entry that
  was in the destination before the call belongs to another module and is kept.
- **State is bound to the config.** The state carries `hexantk_gino_config_sha256`, the
  SHA-256 of `GinoPhase1Config.canonical()` as a `uint8[32]` tensor. `load_state_dict`
  compares it with the module's config and raises `RuntimeError` on a mismatch, under
  `strict=False` too. A state without the entry fails a strict load (missing key); a
  non-strict load reports the key as missing.
  Reason: `radius_factor` changes no tensor shape. Weights trained at 2.5 load into a model
  built at 2.25 without any error and compute another function.

Not carried over from `"_metadata"`: the backend's constructor arguments and version string.
The config digest already covers the arguments the adapter sets, and `build_gino` checks
the installed backend version. `ntk_trajectory_hex_v3` reads the backend's own state
(`module.gino.state_dict()`), which is unchanged.

## 3. `RelativeL2Loss`

Mean over the samples of `||pred - target||_2 / ||target||_2` on `[batch, nodes]`. With
equal cell areas this is the per-sample `rel_l2` of `heat_operator_task_hex_v3.relative_l2`
(tested to 1e-15 in float64). It is a mean of ratios: scaling one sample by 1000 leaves it
unchanged. A zero target raises `ValueError`, as the metric does.

## 4. `RunnerPairDataset`

Wraps a map-style dataset of `{"u0", "uT"}` samples (`HeatOperatorDataset`, or a `Subset`
after the split) and yields `{"input", "target"}` as float32 `[nodes]`, plus `sample_id`
when the source has a `trajectory_id`. Refused: a wrong shape or a non-real dtype, a
non-finite value, an overflow to inf in float32, and a nonzero value that becomes zero.

## 5. `Phase1TrainingRecipe` and `build_phase1_runner`

The recipe is a frozen dataclass with one digest (SHA-256 of its canonical JSON, floats as
`float.hex`).

| Field | Default | Source |
|---|---|---|
| `epochs`, `learning_rate`, `batch_size`, `seed` | none, must be given | preregistration D2/D4 |
| optimizer Adam, `CosineAnnealingLR` with `T_max = epochs` stepped once per epoch, relative L2 loss, float32, no AMP | fixed | preregistration section 3, D5 |
| `weight_decay` | 0 | not named |
| `adam_betas`, `adam_eps` | (0.9, 0.999), 1e-8 | not named |
| `cosine_eta_min` | 0 | not named |
| `grad_clip_max_norm` | inf (no clipping) | not named |
| `use_ema`, `ema_decay` | False, 0.9999 | not named |
| `warmup_steps` | 0 | not named |
| `shuffle` | True (`drop_last` is always False) | not named |

`build_phase1_runner(recipe, gino_config, train_data, val_data, output_dir=...)`:

1. checks that both datasets are `RunnerPairDataset`s on the N = 32 triangle lattice and not empty;
2. seeds torch with `recipe.seed`, then builds the model (the seed fixes the initial weights);
3. builds Adam, the cosine schedule, the training loader with its own generator seeded
   from `recipe.seed`, and the validation loader in order;
4. builds the runner with these settings and nothing from `RunnerConfig`'s defaults for
   clipping, EMA, AMP or warm-up.

It returns the runner and the module. After `runner.run()` the module holds the weights of
the **last epoch**. The runner still writes `best.pt` (by `val_loss`); that file is
information only. Choosing it would be an early-stopping rule, which the preregistration
excludes.

Shown by the tests, with the stand-in and with the real backend on synthetic tensors:

- the learning rate after each epoch is the closed-form cosine value;
- two builds with one seed train to the same bits; another seed gives other initial and final weights;
- `best.pt` and `last.pt` load with `weights_only=True`, with and without EMA;
- a 4-epoch run interrupted after 2 epochs, or inside the second epoch
  (`save_every_n_steps=1`), and resumed ends with the bits of the uninterrupted run
  (weights, and EMA where used). The resume file holds Python and NumPy RNG states and is
  read with pickle (`resume_trusted=True`), as for every runner resume.

## 6. Selection statistic

The preregistration selects "the lowest mean validation `rel_l2`". The runner reports two
other numbers:

- `val_loss` is the mean of the batch means. It equals the mean over samples only when
  every validation batch has the same size (200 validation samples at batch size 20 do;
  6 samples at batch size 4 do not, as the test shows);
- `val_nrmse` is the pooled `sqrt(sum e^2 / sum y^2)`, in which large samples weigh more.

`mean_rel_l2(model, dataset)` computes the registered statistic: the per-sample `rel_l2`
through `relative_l2` in float64, then the mean. It does not depend on the batch size.

## 7. Open owner questions

The preregistration (and the D4 candidate in PR #91) does not name the settings marked
"not named" in section 5. The recipe makes them explicit and puts them in its digest, with
the plain reading as default. The approved preregistration should state:

1. gradient clipping (recipe default: none);
2. EMA (default: off) and, if on, whether the EMA or the raw weights are evaluated;
3. weight decay, Adam betas and eps, `eta_min` (defaults: torch's, 0);
4. that the evaluated model is the last epoch, not `best.pt`;
5. that selection uses `mean_rel_l2`, not the runner's `val_loss` or `val_nrmse`;
6. the run manifest should carry `Phase1TrainingRecipe.digest()` and `GinoPhase1Config.digest()`.

## 8. Limits

- Not run: CUDA, DDP with the real GINO (the module adds no collective; the constant
  geometry buffers are equal on every rank), AMP (excluded by the recipe).
- "The same bits" holds on one machine, one software stack and one thread setting, as in
  `spec/MID_EPOCH_RESUME_v1.md`.
- The stand-in for the backend has the backend's call signature and its `"_metadata"`
  entry; everything else about the real model is covered only by `TestRealBackend`, which
  runs where neuraloperator 2.0.0 is installed (the `gate` job and the GINO smoke job).
- Cost of a real training run is not measured here (decision support for that:
  `scripts/oracles/gino_epoch_cost_synthetic.py` in PR #91).
- A checkpoint written by a raw GINO or by another wrapper carries no config binding and
  is refused by a strict load.

## 9. Sources

- PyTorch, "Serialization semantics" (`torch.load` uses `weights_only=True` by default
  since 2.6 and then refuses globals outside its allowlist):
  https://pytorch.org/docs/2.6/notes/serialization.html
- neuraloperator 2.0.0, `neuralop/models/base_model.py`, `BaseModel.state_dict` and
  `BaseModel.load_state_dict` (read from the installed package).
