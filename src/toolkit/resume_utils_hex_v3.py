"""
resume_utils_hex_v3.py
======================
Training resume checkpoint utilities for HexaNTK pipeline.

Key design decisions:
- build_resume_payload() separated from save for composability
- Safe vs trusted loading paths (PyTorch 2.6+)
- Scheduler restored BEFORE optimizer (PyTorch best practice)
- Optional DataLoader state for mid-epoch resume readiness
- requires_trusted_load flag in save report

Safe load path:
    Checkpoint contains only tensors, primitives, dicts.
    Use weights_only=True.

Trusted load path:
    Checkpoint contains Python objects (random.getstate(), numpy RNG).
    Use weights_only=False. Trusted sources only.
"""
from __future__ import annotations

__all__ = [
    "build_resume_payload",
    "save_resume_checkpoint",
    "restore_rng_states",
    "load_resume_checkpoint",
]

import math
import numbers
import random
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch



def _exact_nonnegative_int(value: Any, *, name: str) -> int:
    """Normalize exact non-negative integer metadata or fail with a contract error."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise TypeError(f"{name} must be a scalar integer value")
        value = value.item()

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer value, not bool")

    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise ValueError(f"{name} must be a finite exact integer, got {value!r}")
        result = int(value)
    elif isinstance(value, numbers.Integral):
        result = int(value)
    elif isinstance(value, numbers.Rational):
        if value.denominator != 1:
            raise ValueError(f"{name} must be an exact integer, got {value!r}")
        result = int(value)
    elif isinstance(value, numbers.Real):
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a finite exact integer, got {value!r}"
            ) from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be a finite exact integer, got {value!r}")
        result = int(value)
        if value != result:
            raise ValueError(f"{name} must be an exact integer, got {value!r}")
    else:
        raise TypeError(
            f"{name} must be an integer-compatible scalar, got {type(value).__name__}"
        )

    if result < 0:
        raise ValueError(f"{name} must be >= 0, got {result}")
    return result


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
) -> Dict[str, Any]:
    """
    Build a resume checkpoint payload dict without writing to disk.

    This function is separated from save_resume_checkpoint so that
    callers (e.g. CheckpointIntegrationManager) can use their own
    atomic writing mechanism.

    Returns
    -------
    Dict[str, Any]
        Complete payload ready for torch.save().
    """
    payload: Dict[str, Any] = {
        "checkpoint_type": "resume",
        "checkpoint_semantics": "resume_full",
        "schema_version": "1.0",
        "epoch": _exact_nonnegative_int(epoch, name="epoch"),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if global_step is not None:
        payload["global_step"] = _exact_nonnegative_int(
            global_step, name="global_step"
        )

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
) -> Dict[str, Any]:
    """
    Load a training resume checkpoint.

    Parameters
    ----------
    trusted :
        If False (default), uses weights_only=True (safe path).
        If True, uses weights_only=False (for checkpoints with Python RNG states).

    Restore order:
        1. model state
        2. scheduler state (scheduler already constructed with optimizer)
        3. optimizer state
        4. grad_scaler state
        5. RNG states
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

    # Validate scalar metadata before mutating any training state.
    epoch_value = _exact_nonnegative_int(payload["epoch"], name="epoch")
    global_step_value = (
        _exact_nonnegative_int(payload["global_step"], name="global_step")
        if "global_step" in payload
        else None
    )

    # 1. Model
    model_incompat = model.load_state_dict(payload["model_state_dict"], strict=strict)

    # 2. Scheduler. It must already be constructed with the optimizer.
    # Restore its state before optimizer.load_state_dict() so the optimizer
    # learning rates from the checkpoint are not overwritten by scheduler
    # initialization/state restoration.
    scheduler_restored = False
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        scheduler_restored = True

    # 3. Optimizer
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    # 4. GradScaler
    grad_scaler_restored = False
    if grad_scaler is not None and "grad_scaler_state_dict" in payload:
        grad_scaler.load_state_dict(payload["grad_scaler_state_dict"])
        grad_scaler_restored = True

    # 5. RNG states
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
        "epoch": epoch_value,
        "global_step": global_step_value,
        "strict": strict,
        "trusted": trusted,
        "model_load_result": {
            "missing_keys": list(model_incompat.missing_keys),
            "unexpected_keys": list(model_incompat.unexpected_keys),
        },
        "optimizer_restored": True,
        "scheduler_restored": scheduler_restored,
        "grad_scaler_restored": grad_scaler_restored,
        "rng_restored": rng_report,
        "has_dataloader_state": "dataloader_state_dict" in payload,
        "has_extra": "extra" in payload,
    }

    if "dataloader_state_dict" in payload:
        report["dataloader_state_dict"] = payload["dataloader_state_dict"]

    if "extra" in payload:
        report["extra"] = payload["extra"]

    return report
