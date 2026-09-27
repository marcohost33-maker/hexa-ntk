"""
resume_utils_hex_v3.py
======================
Training resume checkpoint utilities for HexaNTK pipeline.

Key design decisions:
- build_resume_payload() separated from save for composability
- Safe vs trusted loading paths (PyTorch 2.6+)
- Restore order: model, optimizer, scheduler, grad_scaler, EMA, RNG
  (optimizer before scheduler, as in the PyTorch checkpoint recipe)
- Optional EMA (AveragedModel) state: without it a resumed run restarts
  the average from the resumed weights and is no longer equivalent to an
  uninterrupted run
- Optional DataLoader state for mid-epoch resume readiness
- requires_trusted_load flag in save report
- epoch / global_step are validated as non-negative integers (as_step_count)
  before anything is written or any state is restored

Safe load path:
    Checkpoint contains only tensors, primitives, dicts.
    Use weights_only=True.

Trusted load path:
    Checkpoint contains Python objects (random.getstate(), numpy RNG).
    Use weights_only=False. Trusted sources only.
"""
from __future__ import annotations

__all__ = [
    "as_step_count",
    "build_resume_payload",
    "save_resume_checkpoint",
    "restore_rng_states",
    "load_resume_checkpoint",
]

import math
import numbers
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


# ------------------------------------------------------------------
# Counter contract
# ------------------------------------------------------------------


def as_step_count(value: Any, *, name: str) -> int:
    """Return ``value`` as a non-negative int, or raise.

    int() alone raised a raw OverflowError for inf and silently truncated
    1.5 to 1 (issue #40). Accepted: ints, integral reals (2.0,
    Fraction(4, 2)), single-element tensors/numpy values. bool and
    non-numeric types raise TypeError; non-integral, non-finite or negative
    values raise ValueError.
    """
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) is not None:
        if math.prod(value.shape) != 1:
            raise ValueError(f"{name} must be a scalar, got shape {tuple(value.shape)}")
        value = item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}")
    message = f"{name} must be a non-negative integer, got {value!r}"
    try:
        count = int(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(message) from exc
    if count != value or count < 0:
        raise ValueError(message)
    return count


# ------------------------------------------------------------------
# Payload construction (separated for composability with atomic writers)
# ------------------------------------------------------------------


def build_resume_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: Optional[int] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
    extra: Optional[Dict[str, Any]] = None,
    include_python_rng: bool = True,
    include_numpy_rng: bool = True,
    include_torch_rng: bool = True,
    include_dataloader_state: Optional[Dict[str, Any]] = None,
    ema_state_dict: Optional[Dict[str, Any]] = None,
    ema_decay: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a resume checkpoint payload dict without writing to disk.

    ``ema_state_dict`` is ``AveragedModel.state_dict()`` (averaged weights
    plus the ``n_averaged`` counter); it holds tensors only, so it does not
    affect the safe (weights_only=True) load path. The averaging function is
    not part of it, so ``ema_decay`` is stored next to it (as a float) and
    checked on load: continuing with another decay would silently produce a
    hybrid average.

    This function is separated from save_resume_checkpoint so that
    callers (e.g. CheckpointIntegrationManager) can use their own
    atomic writing mechanism.

    Returns
    -------
    Dict[str, Any]
        Complete payload ready for torch.save().
    """
    epoch = as_step_count(epoch, name="epoch")
    if global_step is not None:
        global_step = as_step_count(global_step, name="global_step")

    payload: Dict[str, Any] = {
        "checkpoint_type": "resume",
        "checkpoint_semantics": "resume_full",
        "schema_version": "1.0",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if global_step is not None:
        payload["global_step"] = global_step

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    if grad_scaler is not None:
        payload["grad_scaler_state_dict"] = grad_scaler.state_dict()

    if include_python_rng:
        payload["python_random_state"] = random.getstate()

    if include_numpy_rng:
        payload["numpy_random_state"] = np.random.get_state()

    if include_torch_rng:
        payload["torch_random_state"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            payload["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()

    if include_dataloader_state is not None:
        payload["dataloader_state_dict"] = include_dataloader_state

    if ema_state_dict is not None:
        payload["ema_state_dict"] = ema_state_dict
        if ema_decay is not None:
            payload["ema_decay"] = float(ema_decay)

    if extra is not None:
        payload["extra"] = extra

    return payload


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------


def save_resume_checkpoint(
    path: Union[str, Path],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: Optional[int] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
    extra: Optional[Dict[str, Any]] = None,
    include_python_rng: bool = True,
    include_numpy_rng: bool = True,
    include_torch_rng: bool = True,
    include_dataloader_state: Optional[Dict[str, Any]] = None,
    ema_state_dict: Optional[Dict[str, Any]] = None,
    ema_decay: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Save a training resume checkpoint.

    Note: This writes directly via torch.save. For atomic writing,
    use build_resume_payload() + AtomicCheckpointWriter instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_resume_payload(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        scheduler=scheduler,
        grad_scaler=grad_scaler,
        extra=extra,
        include_python_rng=include_python_rng,
        include_numpy_rng=include_numpy_rng,
        include_torch_rng=include_torch_rng,
        include_dataloader_state=include_dataloader_state,
        ema_state_dict=ema_state_dict,
        ema_decay=ema_decay,
    )

    torch.save(payload, path)

    requires_trusted_load = include_python_rng or include_numpy_rng

    return {
        "checkpoint_path": str(path),
        "checkpoint_type": "resume",
        "checkpoint_semantics": "resume_full",
        "epoch": payload["epoch"],
        "global_step": payload.get("global_step"),
        "has_scheduler": scheduler is not None,
        "has_grad_scaler": grad_scaler is not None,
        "has_torch_rng": include_torch_rng,
        "has_python_rng": include_python_rng,
        "has_numpy_rng": include_numpy_rng,
        "has_dataloader_state": include_dataloader_state is not None,
        "has_ema_state": ema_state_dict is not None,
        "requires_trusted_load": requires_trusted_load,
    }


# ------------------------------------------------------------------
# RNG restore
# ------------------------------------------------------------------


def restore_rng_states(payload: Dict[str, Any]) -> Dict[str, bool]:
    """Restore RNG states from a checkpoint payload."""
    restored = {
        "python_random_state": False,
        "numpy_random_state": False,
        "torch_random_state": False,
        "torch_cuda_random_state_all": False,
    }

    if "python_random_state" in payload:
        random.setstate(payload["python_random_state"])
        restored["python_random_state"] = True

    if "numpy_random_state" in payload:
        np.random.set_state(payload["numpy_random_state"])
        restored["numpy_random_state"] = True

    if "torch_random_state" in payload:
        torch.random.set_rng_state(payload["torch_random_state"])
        restored["torch_random_state"] = True

    if "torch_cuda_random_state_all" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda_random_state_all"])
        restored["torch_cuda_random_state_all"] = True

    return restored


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------


def _load_checkpoint_payload(
    path: Path,
    *,
    map_location: Union[str, torch.device],
    trusted: bool,
) -> Dict[str, Any]:
    """Load checkpoint payload with safe or trusted path."""
    if trusted:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    else:
        payload = torch.load(path, map_location=map_location, weights_only=True)

    if not isinstance(payload, dict):
        raise TypeError("Resume checkpoint payload must be a dictionary.")

    return payload


def load_resume_checkpoint(
    path: Union[str, Path],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
    map_location: Union[str, torch.device] = "cpu",
    restore_rng: bool = True,
    strict: bool = True,
    trusted: bool = False,
    ema_model: Optional[torch.optim.swa_utils.AveragedModel] = None,
    ema_decay: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Load a training resume checkpoint.

    Parameters
    ----------
    trusted :
        If False (default), uses weights_only=True (safe path).
        If True, uses weights_only=False (for checkpoints with Python RNG states).
    ema_model :
        Optional AveragedModel, built with the same decay as when saving.
        Restored from ``ema_state_dict`` if the checkpoint carries one;
        ``report["ema_restored"]`` says whether it did.
    ema_decay :
        The decay ``ema_model`` averages with. If both it and the checkpoint's
        recorded ``ema_decay`` are known and differ, ValueError is raised
        before anything is restored (no silent hybrid EMA).

    Restore order (PyTorch convention):
        1. model state
        2. optimizer state
        3. scheduler state
        4. grad_scaler state
        5. EMA state
        6. RNG states

    The required fields and the epoch/global_step contract are validated
    before step 1, so a malformed checkpoint raises without touching the
    model or optimizer.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    payload = _load_checkpoint_payload(
        path, map_location=map_location, trusted=trusted
    )

    # Validate required fields
    if "model_state_dict" not in payload:
        raise ValueError("Resume checkpoint is missing 'model_state_dict'.")
    if "optimizer_state_dict" not in payload:
        raise ValueError("Resume checkpoint is missing 'optimizer_state_dict'.")
    if "epoch" not in payload:
        raise ValueError("Resume checkpoint is missing 'epoch'.")
    epoch = as_step_count(payload["epoch"], name="epoch")
    global_step = (
        as_step_count(payload["global_step"], name="global_step")
        if payload.get("global_step") is not None
        else None
    )
    recorded_decay = payload.get("ema_decay")
    if (
        ema_model is not None
        and "ema_state_dict" in payload
        and ema_decay is not None
        and recorded_decay is not None
        and float(recorded_decay) != float(ema_decay)
    ):
        raise ValueError(
            f"Resume checkpoint EMA was averaged with decay {recorded_decay!r}, "
            f"but the current EMA uses {ema_decay!r}; continuing would mix two "
            "averages. Resume with the same ema_decay (or without ema_model)."
        )

    # 1. Model
    model_incompat = model.load_state_dict(payload["model_state_dict"], strict=strict)

    # 2. Optimizer (before scheduler — PyTorch convention)
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    # 3. Scheduler
    scheduler_restored = False
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        scheduler_restored = True

    # 4. GradScaler
    grad_scaler_restored = False
    if grad_scaler is not None and "grad_scaler_state_dict" in payload:
        grad_scaler.load_state_dict(payload["grad_scaler_state_dict"])
        grad_scaler_restored = True

    # 5. EMA
    ema_restored = False
    if ema_model is not None and "ema_state_dict" in payload:
        ema_model.load_state_dict(payload["ema_state_dict"], strict=strict)
        ema_restored = True

    # 6. RNG states
    rng_report: Dict[str, bool] = {
        "python_random_state": False,
        "numpy_random_state": False,
        "torch_random_state": False,
        "torch_cuda_random_state_all": False,
    }
    if restore_rng:
        rng_report = restore_rng_states(payload)

    report: Dict[str, Any] = {
        "checkpoint_path": str(path),
        "checkpoint_type": payload.get("checkpoint_type", "unknown"),
        "checkpoint_semantics": payload.get("checkpoint_semantics", "resume_full"),
        "epoch": epoch,
        "global_step": global_step,
        "strict": strict,
        "trusted": trusted,
        "model_load_result": {
            "missing_keys": list(model_incompat.missing_keys),
            "unexpected_keys": list(model_incompat.unexpected_keys),
        },
        "optimizer_restored": True,
        "scheduler_restored": scheduler_restored,
        "grad_scaler_restored": grad_scaler_restored,
        "has_ema_state": "ema_state_dict" in payload,
        "ema_restored": ema_restored,
        "ema_decay": recorded_decay,
        "rng_restored": rng_report,
        "has_dataloader_state": "dataloader_state_dict" in payload,
        "has_extra": "extra" in payload,
    }

    if "dataloader_state_dict" in payload:
        report["dataloader_state_dict"] = payload["dataloader_state_dict"]

    if "extra" in payload:
        report["extra"] = payload["extra"]

    # Written by CheckpointIntegrationManager.save_resume (best-metric state).
    if "checkpoint_manager_state" in payload:
        report["checkpoint_manager_state"] = payload["checkpoint_manager_state"]

    return report
