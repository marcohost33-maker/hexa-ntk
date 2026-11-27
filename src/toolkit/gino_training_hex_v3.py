"""
gino_training_hex_v3.py
=======================
Runs the pinned GINO (``gino_adapter_hex_v3``) through ``HexTrainingRunner``.

Contract: ``spec/GINO_TRAINING_PHASE1_v1.md``. No model is fit on task data
here; the module is training code, its tests use synthetic tensors only
(``spec/PREREGISTRATION_PHASE1_v1.md``, lines 14-15).

Why it exists (measured on main 18ab389, torch 2.6.0, neuraloperator 2.0.0):

- the runner calls ``model(x)`` (or ``OperatorAdapterV3``, whose GINO branch
  calls ``model(x, coords_cart=...)``); the pinned backend takes
  ``input_geom, latent_queries, output_queries, x``. No call path existed;
- neuraloperator puts its constructor arguments into ``state_dict()`` as a
  ``"_metadata"`` dict that holds ``torch.nn.functional.gelu``. With EMA the
  runner failed at the first checkpoint (``'dict' object has no attribute
  'detach'``); without EMA every checkpoint was written but none could be
  read back (``weights_only=True`` refuses the ``gelu`` global), so a run
  could not resume;
- the task dataset yields ``{"u0", "uT"}`` in float64, the runner reads
  ``{"input", "target"}``;
- the runner's defaults (gradient clipping at 1.0, EMA 0.9999, best model by
  the pooled ``val_nrmse``) are not what the preregistration names.

Parts:

- ``GinoRunnerModule``: ``forward(x[B, M]) -> [B, M]`` on one lattice, the
  same numbers as ``predict_gino``; ``state_dict()`` holds tensors only and
  carries the config digest, which ``load_state_dict`` checks;
- ``RelativeL2Loss``: mean over samples of ``||p - t|| / ||t||``;
- ``RunnerPairDataset``: ``{"u0", "uT"}`` samples as runner batches;
- ``Phase1TrainingRecipe`` / ``build_phase1_runner``: every training setting
  explicit and in one digest;
- ``mean_rel_l2``: the selection statistic of the preregistration (mean of
  the per-sample ``rel_l2``), which the runner's ``val_nrmse`` is not.
"""
from __future__ import annotations

__all__ = [
    "CONFIG_DIGEST_KEY",
    "GinoRunnerModule",
    "Phase1TrainingRecipe",
    "RelativeL2Loss",
    "RunnerPairDataset",
    "build_phase1_runner",
    "mean_rel_l2",
]

import hashlib
import json
import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gino_adapter_hex_v3 import (
    _TRAINING_N,
    GINO_DATA_PARALLEL_REFUSAL,
    GinoPhase1Config,
    _check_lattice,
    build_gino,
    predict_gino,
    prepare_gino_batch,
)
from heat_operator_task_hex_v3 import relative_l2
from hex_fv_solver_hex_v3 import HexLattice, triangle_hex_lattice
from runner_hex_v3 import HexTrainingRunner, RunnerConfig

# State-dict entry (uint8[32]) that binds saved weights to their GinoPhase1Config.
CONFIG_DIGEST_KEY = "hexantk_gino_config_sha256"
# The backend's non-tensor state entry (constructor arguments, incl. functions).
_BACKEND_METADATA_KEY = "_metadata"


def _digest_tensor(hex_digest: str) -> torch.Tensor:
    return torch.tensor(list(bytes.fromhex(hex_digest)), dtype=torch.uint8)


class GinoRunnerModule(nn.Module):
    """The pinned GINO on one lattice behind the runner's ``model(x)`` call.

    ``lattice`` is the lattice the fields live on (any canonical lattice the
    adapter accepts); the GNO radius is always fixed from the N = 32 triangle
    training lattice (``build_gino``). The geometry tensors are non-persistent
    buffers: they follow ``.to(device)`` and are not part of the saved weights,
    so one set of weights loads into a module on any lattice.

    Never trained data-parallel (owner decision 2026-11-27, spec section 8a): the
    class carries ``DATA_PARALLEL_REFUSAL_ATTR`` itself, so the refusal holds with
    any backend, also a stand-in that build_gino did not mark.
    """

    hexantk_data_parallel_refusal = GINO_DATA_PARALLEL_REFUSAL

    def __init__(self, config: GinoPhase1Config, lattice: HexLattice) -> None:
        super().__init__()
        if not isinstance(config, GinoPhase1Config):
            raise TypeError("config must be a GinoPhase1Config")
        self.lattice = _check_lattice(lattice)
        self.config = config
        self.gino = build_gino(config, triangle_hex_lattice(_TRAINING_N))
        geometry = prepare_gino_batch(self.lattice, np.zeros(self.lattice.num_nodes), config)
        self.register_buffer("input_geom", geometry["input_geom"], persistent=False)
        self.register_buffer("latent_queries", geometry["latent_queries"], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.lattice.num_nodes:
            raise ValueError(f"x must be a tensor shaped [batch, {self.lattice.num_nodes}]")
        if x.dtype != self.input_geom.dtype:
            # The backend allocates float32 whatever the model holds (adapter spec);
            # a silent cast here would hide a float64 pipeline.
            raise TypeError(f"x must be {self.input_geom.dtype}, got {x.dtype}")
        out = self.gino(
            input_geom=self.input_geom,
            latent_queries=self.latent_queries,
            output_queries=self.input_geom,
            x=x.unsqueeze(-1),
        )
        if not isinstance(out, torch.Tensor) or tuple(out.shape) != (x.shape[0], x.shape[1], 1):
            raise ValueError("GINO backend returned an unexpected output shape")
        return out[..., 0]

    def predict(self, lattice: HexLattice, u0: Any) -> torch.Tensor:
        """``predict_gino`` with these weights on any lattice, in eval mode without gradients."""
        was_training = self.gino.training
        self.gino.eval()
        try:
            with torch.no_grad():
                return predict_gino(self.gino, lattice, u0, self.config)
        finally:
            self.gino.train(was_training)

    # -- state: tensors only, bound to the config -------------------------

    def state_dict(self, *args: Any, destination: Any = None, prefix: str = "", keep_vars: bool = False) -> Any:
        if args:
            raise TypeError("state_dict takes keyword arguments only")
        had_metadata = destination is not None and _BACKEND_METADATA_KEY in destination
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # neuraloperator writes "_metadata" without the prefix, whatever the nesting.
        if (not had_metadata and _BACKEND_METADATA_KEY in state
                and not isinstance(state[_BACKEND_METADATA_KEY], torch.Tensor)):
            del state[_BACKEND_METADATA_KEY]
        foreign = [key for key, value in state.items()
                   if key.startswith(prefix) and not isinstance(value, torch.Tensor)]
        if foreign:
            raise TypeError(f"non-tensor state entries would make the checkpoint unreadable: {foreign}")
        return state

    def _save_to_state_dict(self, destination: Any, prefix: str, keep_vars: bool) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + CONFIG_DIGEST_KEY] = _digest_tensor(self.config.digest())

    def _load_from_state_dict(self, state_dict: Any, prefix: str, local_metadata: Any, strict: bool,
                              missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]) -> None:
        key = prefix + CONFIG_DIGEST_KEY
        saved = state_dict.pop(key, None)
        expected = _digest_tensor(self.config.digest())
        if saved is None:
            # Weights without a binding: a strict load fails on the missing key, a
            # non-strict load reports it (torch applies ``strict`` after this hook).
            missing_keys.append(key)
        elif (not isinstance(saved, torch.Tensor) or saved.dtype != torch.uint8
              or saved.shape != expected.shape or not torch.equal(saved.cpu(), expected)):
            error_msgs.append(
                f"{key}: the weights were saved for another GinoPhase1Config than "
                f"{self.config.digest()[:16]}... (modes, channels, layers, latent grid or radius differ)"
            )
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class RelativeL2Loss(nn.Module):
    """Mean over the samples of ``||pred - target||_2 / ||target||_2`` on ``[batch, nodes]``.

    Equal cell areas make this the per-sample ``rel_l2`` of
    ``heat_operator_task_hex_v3.relative_l2`` (the primary metric). A zero
    target has no relative error and is refused, as in the metric.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 2 or pred.shape != target.shape:
            raise ValueError("pred and target must both be shaped [batch, nodes]")
        reference = torch.linalg.vector_norm(target, dim=1)
        if bool((reference == 0).any()):
            raise ValueError("relative L2 is undefined for a zero target")
        return (torch.linalg.vector_norm(pred - target, dim=1) / reference).mean()


class RunnerPairDataset(Dataset):
    """Samples ``{"u0", "uT"}`` (``HeatOperatorDataset`` or a ``Subset`` of it) as runner batches.

    Yields ``{"input", "target"}`` as float32 tensors shaped ``[nodes]`` plus the
    integer ``sample_id`` when the source carries a ``trajectory_id``. The
    float64 -> float32 conversion is checked: non-finite values, an overflow to
    inf and a nonzero entry that becomes zero are refused.
    """

    def __init__(self, source: Any, num_nodes: int) -> None:
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, numbers.Integral) or num_nodes < 1:
            raise ValueError("num_nodes must be a positive int")
        if not hasattr(source, "__len__") or not hasattr(source, "__getitem__"):
            raise TypeError("source must be a map-style dataset")
        self.source = source
        self.num_nodes = int(num_nodes)

    def __len__(self) -> int:
        return len(self.source)

    def _field(self, value: Any, name: str) -> torch.Tensor:
        field = np.asarray(value)
        if field.dtype.kind not in "fiu" or field.shape != (self.num_nodes,):
            raise ValueError(f"{name} must be a real field shaped [{self.num_nodes}]")
        field = field.astype(np.float64)
        if not np.all(np.isfinite(field)):
            raise ValueError(f"{name} must be finite")
        with np.errstate(over="ignore"):
            narrow = field.astype(np.float32)
        if not np.all(np.isfinite(narrow)):
            raise ValueError(f"{name} overflows float32")
        if np.any((field != 0.0) & (narrow == 0.0)):
            raise ValueError(f"{name} conversion to float32 underflows a nonzero entry")
        return torch.from_numpy(narrow)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.source[index]
        item: Dict[str, Any] = {"input": self._field(sample["u0"], "u0"),
                                "target": self._field(sample["uT"], "uT")}
        if "trajectory_id" in sample:
            item["sample_id"] = int(sample["trajectory_id"])
        return item


def mean_rel_l2(model: GinoRunnerModule, dataset: RunnerPairDataset, *, batch_size: int = 100) -> float:
    """Mean of the per-sample ``rel_l2`` of ``model`` on ``dataset``, metric in float64.

    This is the selection statistic of the preregistration ("lowest mean
    validation rel_l2"). It does not depend on ``batch_size``; the runner's
    ``val_loss`` (mean of batch means) does when the last batch is smaller, and
    its ``val_nrmse`` is the pooled ``sqrt(sum e^2 / sum y^2)``, another number.
    """
    if not isinstance(model, GinoRunnerModule) or not isinstance(dataset, RunnerPairDataset):
        raise TypeError("mean_rel_l2 takes a GinoRunnerModule and a RunnerPairDataset")
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    device = model.input_geom.device
    values: List[float] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                pred = model(batch["input"].to(device))
                values.extend(relative_l2(pred.cpu().numpy(), batch["target"].numpy(), model.lattice).tolist())
    finally:
        model.train(was_training)
    return math.fsum(values) / len(values)


# ------------------------------------------------------------------
# Training recipe
# ------------------------------------------------------------------


def _int_field(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return int(value)


def _real_field(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    return float(value)


@dataclass(frozen=True)
class Phase1TrainingRecipe:
    """Every training setting of one run. Nothing here is a default of a library.

    The first four fields have no default: the preregistration fixes them
    (epoch budget and learning rate per grid configuration, batch size, seed).
    The others are settings the preregistration text does not name. Their
    defaults are the plain reading of "Adam, cosine schedule, relative L2, no
    early stopping" and they enter the digest, so a run that changes one is a
    different recipe. The approved preregistration has to name them
    (spec section 6).
    """

    epochs: int
    learning_rate: float
    batch_size: int
    seed: int
    weight_decay: float = 0.0
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    cosine_eta_min: float = 0.0
    grad_clip_max_norm: float = math.inf
    use_ema: bool = False
    ema_decay: float = 0.9999
    warmup_steps: int = 0
    shuffle: bool = True

    def __post_init__(self) -> None:
        _int_field(self.epochs, "epochs", 1)
        _int_field(self.batch_size, "batch_size", 1)
        _int_field(self.seed, "seed", 0)
        _int_field(self.warmup_steps, "warmup_steps", 0)
        lr = _real_field(self.learning_rate, "learning_rate")
        if not (math.isfinite(lr) and lr > 0.0):
            raise ValueError("learning_rate must be finite and > 0")
        for name in ("weight_decay", "cosine_eta_min"):
            value = _real_field(getattr(self, name), name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0")
        eps = _real_field(self.adam_eps, "adam_eps")
        if not (math.isfinite(eps) and eps > 0.0):
            raise ValueError("adam_eps must be finite and > 0")
        if (not isinstance(self.adam_betas, tuple) or len(self.adam_betas) != 2
                or not all(0.0 <= _real_field(b, "adam_betas") < 1.0 for b in self.adam_betas)):
            raise ValueError("adam_betas must be a tuple of two numbers in [0, 1)")
        clip = _real_field(self.grad_clip_max_norm, "grad_clip_max_norm")
        if math.isnan(clip) or clip <= 0.0:
            raise ValueError("grad_clip_max_norm must be > 0 (inf = no clipping)")
        decay = _real_field(self.ema_decay, "ema_decay")
        if not (math.isfinite(decay) and 0.0 <= decay <= 1.0):
            raise ValueError("ema_decay must be in [0, 1]")
        for name in ("use_ema", "shuffle"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")

    def canonical(self) -> Dict[str, Any]:
        clip = float(self.grad_clip_max_norm)
        return {
            "recipe": "gino-training-phase1/v1",
            "optimizer": "torch.optim.Adam",
            "schedule": "torch.optim.lr_scheduler.CosineAnnealingLR, T_max = epochs, one step per epoch",
            "loss": "relative L2, mean over the samples of a batch",
            "dtype": "float32",
            "amp": False,
            "registered_weights": "last epoch (best.pt is not the registered model)",
            "epochs": int(self.epochs),
            "learning_rate": float(self.learning_rate).hex(),
            "batch_size": int(self.batch_size),
            "seed": int(self.seed),
            "weight_decay": float(self.weight_decay).hex(),
            "adam_betas": [float(b).hex() for b in self.adam_betas],
            "adam_eps": float(self.adam_eps).hex(),
            "cosine_eta_min": float(self.cosine_eta_min).hex(),
            "grad_clip_max_norm": "inf" if math.isinf(clip) else clip.hex(),
            "use_ema": self.use_ema,
            "ema_decay": float(self.ema_decay).hex(),
            "warmup_steps": int(self.warmup_steps),
            "shuffle": self.shuffle,
            "drop_last": False,
        }

    def digest(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_phase1_runner(
    recipe: Phase1TrainingRecipe,
    gino_config: GinoPhase1Config,
    train_data: RunnerPairDataset,
    val_data: RunnerPairDataset,
    *,
    output_dir: str,
    resume_from: str | None = None,
    resume_trusted: bool = False,
    save_every_n_steps: int = 0,
) -> Tuple[HexTrainingRunner, GinoRunnerModule]:
    """One training run on the N = 32 triangle lattice, built from ``recipe`` alone.

    Seeds torch with ``recipe.seed`` before the model is built (the seed fixes
    the initial weights) and gives the training loader its own generator from
    the same seed (the runner saves and restores it on resume). Returns the
    runner and the module it trains; after ``runner.run()`` the module holds the
    weights of the last epoch. ``resume_trusted`` is the runner's flag: a resume
    checkpoint holds Python and NumPy RNG states and is read with pickle, so it
    must only be set for a file this machine wrote.
    """
    if not isinstance(recipe, Phase1TrainingRecipe):
        raise TypeError("recipe must be a Phase1TrainingRecipe")
    lattice = triangle_hex_lattice(_TRAINING_N)
    for name, data in (("train_data", train_data), ("val_data", val_data)):
        if not isinstance(data, RunnerPairDataset) or data.num_nodes != lattice.num_nodes:
            raise ValueError(f"{name} must be a RunnerPairDataset on the N = {_TRAINING_N} triangle lattice")
        if len(data) == 0:
            raise ValueError(f"{name} is empty")
    torch.manual_seed(recipe.seed)
    model = GinoRunnerModule(gino_config, lattice)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(recipe.learning_rate), betas=tuple(float(b) for b in recipe.adam_betas),
        eps=float(recipe.adam_eps), weight_decay=float(recipe.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=recipe.epochs, eta_min=float(recipe.cosine_eta_min)
    )
    loader_generator = torch.Generator().manual_seed(recipe.seed)
    train_loader = DataLoader(train_data, batch_size=recipe.batch_size, shuffle=recipe.shuffle,
                              drop_last=False, generator=loader_generator)
    val_loader = DataLoader(val_data, batch_size=recipe.batch_size, shuffle=False, drop_last=False)
    config = RunnerConfig(
        output_dir=output_dir,
        num_epochs=recipe.epochs,
        grad_clip_max_norm=float(recipe.grad_clip_max_norm),
        use_ema=recipe.use_ema,
        ema_decay=float(recipe.ema_decay),
        use_amp=False,
        warmup_steps=recipe.warmup_steps,
        save_every_n_steps=save_every_n_steps,
        # Information only: the registered model is the last epoch, and the
        # registered selection statistic is mean_rel_l2, computed after the run.
        best_metric_name="val_loss",
        resume_from=resume_from,
        resume_trusted=resume_trusted,
    )
    runner = HexTrainingRunner(config, model, optimizer, train_loader, val_loader,
                               loss_fn=RelativeL2Loss(), scheduler=scheduler, device="cpu")
    return runner, model
