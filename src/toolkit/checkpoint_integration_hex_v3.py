"""
checkpoint_integration_hex_v3.py
================================
Central orchestration layer for HexaNTK checkpoint management.

Merges:
- Doc1: Class architecture, BestCheckpointPolicy, min_delta, checkpoint_semantics
- Doc2: EMA support, is_main_process guard, FailureSignature integration

Design invariants:
- best / last / resume / warmstart are strictly separated
- checkpoint_semantics field on every save/load report
- Atomic writing for ALL checkpoint types
- Best-model always loaded via safe path (weights_only=True)
- Resume payload built via build_resume_payload for composability
- Scheduler restored BEFORE optimizer (PyTorch best practice)

PyTorch 2.6+ compatibility:
- weights_only=True is the default
- safe_globals whitelisting for legacy checkpoints
"""
from __future__ import annotations

__all__ = [
    "BestCheckpointPolicy",
    "AtomicCheckpointWriter",
    "CheckpointIntegrationManager",
]

import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from resume_utils_hex_v3 import (
    build_resume_payload,
    load_resume_checkpoint,
)
from warmstart_utils_hex_v3 import load_partial_checkpoint

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Best-model policy
# ------------------------------------------------------------------


def _finite_metric_value(value: Any) -> float:
    """Return ``value`` as a finite float, or raise ValueError."""
    message = f"metric_value must be a finite real number, got {value!r}"
    try:
        result = float(value)
    except OverflowError as exc:
        # Exact Reals beyond binary64 range (10**400, Fraction(10**400)).
        raise ValueError(message) from exc
    if not math.isfinite(result):
        raise ValueError(message)
    return result


@dataclass
class BestCheckpointPolicy:
    """
    Policy for deciding whether a metric constitutes a new best model.

    Attributes
    ----------
    mode :
        "min" for losses/errors (lower is better).
        "max" for scores (higher is better).
    min_delta :
        Minimum required improvement to count as a new best.
        Prevents saving on noise-level fluctuations.
    """

    mode: str = "min"
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"Unsupported mode: {self.mode}. Use 'min' or 'max'.")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0.")

    def is_improved(self, current: float, best: Optional[float]) -> bool:
        """Check if current metric is an improvement over best."""
        if best is None:
            return True
        if self.mode == "min":
            return current < (best - self.min_delta)
        return current > (best + self.min_delta)


# ------------------------------------------------------------------
# Atomic writer
# ------------------------------------------------------------------


class AtomicCheckpointWriter:
    """
    Write checkpoint files atomically:
    1. Write to temp file in same directory
    2. os.replace() onto target path (POSIX atomic rename)
    3. Cleanup on failure

    Thread-safe: uses unique temp file names via tempfile.mkstemp.
    """

    @staticmethod
    def save_torch_payload_atomic(
        path: Union[str, Path], payload: Dict[str, Any]
    ) -> str:
        """Save a payload dict atomically via temp file + os.replace."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path_str = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".tmp_{path.name}_",
            suffix=".pt",
        )
        os.close(fd)
        tmp_path = Path(tmp_path_str)

        try:
            torch.save(payload, tmp_path)
            # fsync to ensure data is flushed to disk before rename.
            # Without this, a power failure between torch.save() and
            # os.replace() could leave the temp file with incomplete
            # data in OS buffers. This is critical for cluster preemption.
            # (Pattern from python-atomicwrites, PyTorch Ignite DiskSaver)
            #
            # The handle MUST be opened writable ("rb+"): os.fsync() on a
            # read-only ("rb") descriptor raises OSError(EBADF, "Bad file
            # descriptor") on Windows (verified) and is not the documented
            # contract for flushing prior writes. python-atomicwrites fsyncs
            # the *write* handle, so we reopen the file writable, flush, fsync.
            with open(tmp_path, "rb+") as f:
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            # fsync the parent directory so the rename (the filename entry)
            # itself is durable. POSIX-only: os.fsync() on a directory descriptor
            # raises OSError(EACCES/EISDIR) on Windows, so it is skipped there.
            # Honest limitation (do not overclaim): on Windows the temp-file
            # CONTENT is durable (the fsync above), but the rename's directory
            # metadata is NOT explicitly flushed — CPython's os.replace maps to
            # MoveFileExW WITHOUT MOVEFILE_WRITE_THROUGH, so a crash in the narrow
            # window right after the rename can still drop the directory entry.
            # Acceptable for checkpoint/preemption use; if full rename-durability
            # on Windows is ever required, add a ctypes
            # MoveFileExW(..., MOVEFILE_WRITE_THROUGH) call here.
            if os.name == "posix":
                dir_fd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

        return str(path)

    @staticmethod
    def save_model_state_dict_atomic(
        path: Union[str, Path],
        model: nn.Module,
        *,
        extra: Optional[Dict[str, Any]] = None,
        ema_model_state: Optional[Dict[str, Any]] = None,
        checkpoint_type: str = "best_model",
    ) -> str:
        """Save a model state_dict checkpoint atomically."""
        payload: Dict[str, Any] = {
            "checkpoint_type": checkpoint_type,
            "checkpoint_semantics": checkpoint_type,
            "schema_version": "1.0",
            "model_state_dict": model.state_dict(),
        }
        if ema_model_state is not None:
            payload["ema_model_state_dict"] = ema_model_state
        if extra is not None:
            payload["extra"] = extra

        return AtomicCheckpointWriter.save_torch_payload_atomic(path, payload)


# ------------------------------------------------------------------
# Integration manager
# ------------------------------------------------------------------


class CheckpointIntegrationManager:
    """
    Central orchestration layer for:
    - best-model checkpoint (with min_delta policy + optional EMA)
    - last-model checkpoint
    - resume checkpoint (full training state)
    - warmstart loading (weights only)

    Parameters
    ----------
    root_dir :
        Directory for all checkpoints.
    best_policy :
        Policy for deciding best-model updates.
    is_main_process :
        DDP guard. If False, save operations are skipped.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        *,
        best_policy: Optional[BestCheckpointPolicy] = None,
        is_main_process: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.best_policy = best_policy or BestCheckpointPolicy(mode="min", min_delta=0.0)
        self.best_metric_value: Optional[float] = None
        self.is_main_process = is_main_process

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def best_model_path(self) -> str:
        return str(self.root_dir / "best.pt")

    @property
    def last_model_path(self) -> str:
        return str(self.root_dir / "last.pt")

    @property
    def resume_path(self) -> str:
        return str(self.root_dir / "resume_last.pt")

    # ------------------------------------------------------------------
    # State export
    # ------------------------------------------------------------------

    def export_state(self) -> Dict[str, Any]:
        """Export internal state for diagnostics/logging."""
        return {
            "root_dir": str(self.root_dir),
            "best_policy": asdict(self.best_policy),
            "best_metric_value": self.best_metric_value,
            "is_main_process": self.is_main_process,
            "paths": {
                "best_model_path": self.best_model_path,
                "last_model_path": self.last_model_path,
                "resume_path": self.resume_path,
            },
        }

    # ------------------------------------------------------------------
    # Save: best model
    # ------------------------------------------------------------------

    def save_best_model_checkpoint(
        self,
        *,
        model: nn.Module,
        metric_value: float,
        metric_name: str,
        epoch: int,
        global_step: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
        ema_model_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save a best-model checkpoint if the metric has improved.

        Returns a report dict with saved=True/False.

        Raises ValueError (before any state change) if ``metric_value`` is not
        a finite real number: NaN compares False against everything, so a NaN
        best value would block every later improvement. ``best_metric_value``
        is committed only after the checkpoint write succeeded.
        """
        metric = _finite_metric_value(metric_value)
        improved = self.best_policy.is_improved(metric, self.best_metric_value)

        if not improved:
            return {
                "saved": False,
                "reason": "not_improved",
                "checkpoint_semantics": "best_model",
                "metric_name": metric_name,
                "metric_value": metric_value,
                "best_metric_value": self.best_metric_value,
                "path": self.best_model_path,
            }

        # Build (and thereby validate) the payload on every rank before any
        # state change, so all DDP ranks accept or reject the same inputs.
        payload_extra = {
            "metric_name": metric_name,
            "metric_value": metric,
            "epoch": int(epoch),
            "global_step": int(global_step) if global_step is not None else None,
            "best_policy": asdict(self.best_policy),
        }
        if extra is not None:
            payload_extra["user_extra"] = extra

        if not self.is_main_process:
            self.best_metric_value = metric
            return {
                "saved": False,
                "reason": "not_main_process",
                "checkpoint_semantics": "best_model",
                "metric_value": metric_value,
            }

        AtomicCheckpointWriter.save_model_state_dict_atomic(
            self.best_model_path,
            model=model,
            extra=payload_extra,
            ema_model_state=ema_model_state,
            checkpoint_type="best_model",
        )
        # Commit only after the durable write: a failed save must not move
        # the best value, or the same improvement would never be saved.
        self.best_metric_value = metric

        return {
            "saved": True,
            "reason": "improved",
            "checkpoint_semantics": "best_model",
            "metric_name": metric_name,
            "metric_value": metric_value,
            "best_metric_value": self.best_metric_value,
            "path": self.best_model_path,
            "has_ema": ema_model_state is not None,
        }

    # ------------------------------------------------------------------
    # Save: last model
    # ------------------------------------------------------------------

    def save_last_model_checkpoint(
        self,
        *,
        model: nn.Module,
        epoch: int,
        global_step: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save the last model checkpoint (weights only, no optimizer).

        last.pt MUST NEVER be interpreted as a resume checkpoint.
        """
        if not self.is_main_process:
            return {
                "saved": False,
                "reason": "not_main_process",
                "checkpoint_semantics": "last_model_only",
            }

        payload_extra = {
            "epoch": int(epoch),
            "global_step": int(global_step) if global_step is not None else None,
        }
        if extra is not None:
            payload_extra["user_extra"] = extra

        path = AtomicCheckpointWriter.save_model_state_dict_atomic(
            self.last_model_path,
            model=model,
            extra=payload_extra,
            checkpoint_type="last_model_only",
        )

        return {
            "saved": True,
            "checkpoint_semantics": "last_model_only",
            "path": path,
            "epoch": int(epoch),
            "global_step": int(global_step) if global_step is not None else None,
        }

    # ------------------------------------------------------------------
    # Save: resume
    # ------------------------------------------------------------------

    def save_resume(
        self,
        *,
        model: nn.Module,
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
        Save a full resume checkpoint atomically.

        Uses build_resume_payload() + AtomicCheckpointWriter for
        atomic writing with unique temp file names.
        """
        if not self.is_main_process:
            return {
                "saved": False,
                "reason": "not_main_process",
                "checkpoint_semantics": "resume_full",
            }

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

        AtomicCheckpointWriter.save_torch_payload_atomic(self.resume_path, payload)

        requires_trusted_load = include_python_rng or include_numpy_rng

        return {
            "saved": True,
            "checkpoint_path": self.resume_path,
            "checkpoint_type": "resume",
            "checkpoint_semantics": "resume_full",
            "epoch": int(epoch),
            "global_step": int(global_step) if global_step is not None else None,
            "requires_trusted_load": requires_trusted_load,
        }

    # ------------------------------------------------------------------
    # Load: best model
    # ------------------------------------------------------------------

    def load_best_model_checkpoint(
        self,
        *,
        model: nn.Module,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = True,
        use_ema_if_available: bool = False,
    ) -> Dict[str, Any]:
        """
        Load a best-model checkpoint via the SAFE path (weights_only=True).

        Best-model checkpoints contain only tensors and are always
        loaded with weights_only=True for security.
        """
        path = Path(self.best_model_path)
        if not path.exists():
            raise FileNotFoundError(f"Best-model checkpoint not found: {path}")

        # Always safe path for best-model
        payload = torch.load(path, map_location=map_location, weights_only=True)

        if not isinstance(payload, dict):
            raise TypeError("Best-model checkpoint payload must be a dict.")

        # Validate checkpoint type
        checkpoint_type = payload.get("checkpoint_type", "unknown")
        if checkpoint_type not in ("best_model", "unknown"):
            logger.warning(
                "Loading checkpoint marked as '%s' via best-model path.",
                checkpoint_type,
            )

        # Select target state dict
        if "model_state_dict" in payload:
            target_dict = payload["model_state_dict"]
        else:
            # Fallback: assume payload is a flat state dict
            target_dict = payload

        used_ema = False
        if use_ema_if_available and "ema_model_state_dict" in payload:
            target_dict = payload["ema_model_state_dict"]
            used_ema = True

        incompat = model.load_state_dict(target_dict, strict=strict)

        return {
            "checkpoint_path": str(path),
            "checkpoint_semantics": "best_model",
            "load_path_type": "safe",
            "strict": strict,
            "used_ema": used_ema,
            "model_load_result": {
                "missing_keys": list(incompat.missing_keys),
                "unexpected_keys": list(incompat.unexpected_keys),
            },
            "has_extra": "extra" in payload,
            "extra": payload.get("extra", None),
        }

    # ------------------------------------------------------------------
    # Load: resume
    # ------------------------------------------------------------------

    def load_resume(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        grad_scaler: Optional[torch.amp.GradScaler] = None,
        map_location: Union[str, torch.device] = "cpu",
        restore_rng: bool = True,
        strict: bool = True,
        trusted: bool = False,
        path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Load a resume checkpoint.

        Delegates to load_resume_checkpoint from resume_utils_hex_v3.py
        which handles the restore order:
        1. model, 2. scheduler, 3. optimizer, 4. grad_scaler, 5. RNG

        path explicitly selects the resume checkpoint. If omitted, the
        manager-owned resume_path is used. Callers that expose a configurable
        resume path MUST forward it here so the file checked for existence is
        also the file that is actually loaded.
        """
        checkpoint_path = Path(path) if path is not None else self.resume_path
        return load_resume_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
            map_location=map_location,
            restore_rng=restore_rng,
            strict=strict,
            trusted=trusted,
        )

    # ------------------------------------------------------------------
    # Load: warmstart
    # ------------------------------------------------------------------

    def load_warmstart(
        self,
        *,
        model: nn.Module,
        checkpoint_path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
        strip_prefixes: Optional[List[str]] = None,
        prefix_map: Optional[Dict[str, str]] = None,
        allow_dtype_mismatch: bool = True,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Load a warmstart checkpoint (weights only, NEVER optimizer).

        Delegates to load_partial_checkpoint from warmstart_utils_hex_v3.py.
        """
        return load_partial_checkpoint(
            model,
            checkpoint_path,
            map_location=map_location,
            strip_prefixes=strip_prefixes,
            prefix_map=prefix_map,
            allow_dtype_mismatch=allow_dtype_mismatch,
            strict=strict,
        )
