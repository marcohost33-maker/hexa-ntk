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
- Restore order optimizer -> scheduler (PyTorch checkpoint recipe)
- Resume carries the best-metric state and (optionally) the EMA state, so a
  resumed run neither overwrites a better best.pt nor restarts its average
- In-memory best value always matches best.pt: committed after the write,
  and also when the file was replaced but the directory fsync failed

PyTorch 2.6+ compatibility:
- weights_only=True is the default
- safe_globals whitelisting for legacy checkpoints
"""
from __future__ import annotations

__all__ = [
    "BestCheckpointPolicy",
    "AtomicCheckpointWriter",
    "CheckpointDurabilityError",
    "CheckpointIntegrationManager",
]

import logging
import math
import numbers
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from resume_utils_hex_v3 import (
    as_step_count,
    build_resume_payload,
    check_ema_save_metadata,
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
        # NaN passed "< 0" and then blocked every improvement (nan compares
        # False); inf did the same (issue #40).
        if (
            isinstance(self.min_delta, bool)
            or not isinstance(self.min_delta, numbers.Real)
            or not math.isfinite(self.min_delta)
            or self.min_delta < 0
        ):
            raise ValueError(
                f"min_delta must be a finite number >= 0, got {self.min_delta!r}"
            )

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


class CheckpointDurabilityError(OSError):
    """The target file was replaced, but making the rename durable failed.

    Raised when os.replace() already succeeded and the parent-directory
    fsync failed afterwards: the new checkpoint IS at ``path`` (visible to
    readers), only its crash-durability is not guaranteed. Callers that
    track state derived from the file must treat it as written.
    """

    def __init__(self, message: str, *, path: str) -> None:
        super().__init__(message)
        self.path = path
        self.replaced = True


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
        replaced = False

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
            replaced = True
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
        except Exception as exc:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if replaced:
                # The new bytes are already at `path`; report that explicitly
                # so callers do not roll back state that matches the file.
                raise CheckpointDurabilityError(
                    f"Checkpoint written to {path}, but the directory fsync "
                    f"after the rename failed: {exc}",
                    path=str(path),
                ) from exc
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
    run_id :
        Identity of the training run, written into best.pt and the resume
        record. A fresh manager gets a random id; load_resume() adopts the id
        of the resumed run, so best.pt of that run is recognised and a best.pt
        of any other run in the same directory is not.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        *,
        best_policy: Optional[BestCheckpointPolicy] = None,
        is_main_process: bool = True,
        run_id: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.best_policy = best_policy or BestCheckpointPolicy(mode="min", min_delta=0.0)
        self.best_metric_value: Optional[float] = None
        self.best_metric_name: Optional[str] = None
        self.is_main_process = is_main_process
        # DDP: pass the same run_id on every rank if the ranks' ids must agree.
        self.run_id = run_id if run_id is not None else uuid.uuid4().hex

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
            "best_metric_name": self.best_metric_name,
            "run_id": self.run_id,
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
        ema_model_state: Optional[
            Union[Dict[str, Any], Callable[[], Dict[str, Any]]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Save a best-model checkpoint if the metric has improved.

        ``ema_model_state`` may be a zero-argument callable; it is then called
        only when this rank actually writes best.pt (improved metric, main
        process), so a non-improving epoch never copies the EMA to the host.

        Returns a report dict with saved=True/False.

        Raises ValueError (before any state change) if ``metric_value`` is not
        a finite real number: NaN compares False against everything, so a NaN
        best value would block every later improvement. ``best_metric_value``
        is committed only after the checkpoint write succeeded, or when the
        file was already replaced and only the directory fsync failed
        (CheckpointDurabilityError, re-raised): memory must match best.pt.
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
            "epoch": as_step_count(epoch, name="epoch"),
            "global_step": (
                as_step_count(global_step, name="global_step")
                if global_step is not None
                else None
            ),
            "best_policy": asdict(self.best_policy),
            "run_id": self.run_id,
        }
        if extra is not None:
            payload_extra["user_extra"] = extra

        if not self.is_main_process:
            self.best_metric_value = metric
            self.best_metric_name = metric_name
            return {
                "saved": False,
                "reason": "not_main_process",
                "checkpoint_semantics": "best_model",
                "metric_value": metric_value,
            }

        if callable(ema_model_state):
            ema_model_state = ema_model_state()
        try:
            AtomicCheckpointWriter.save_model_state_dict_atomic(
                self.best_model_path,
                model=model,
                extra=payload_extra,
                ema_model_state=ema_model_state,
                checkpoint_type="best_model",
            )
        except CheckpointDurabilityError:
            # best.pt already holds the new model (#40 P3-3): keeping the old
            # value would let a later, worse metric overwrite it.
            self.best_metric_value = metric
            self.best_metric_name = metric_name
            raise
        # Commit only after the durable write: a failed save must not move
        # the best value, or the same improvement would never be saved.
        self.best_metric_value = metric
        self.best_metric_name = metric_name

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
        epoch = as_step_count(epoch, name="epoch")
        if global_step is not None:
            global_step = as_step_count(global_step, name="global_step")

        if not self.is_main_process:
            return {
                "saved": False,
                "reason": "not_main_process",
                "checkpoint_semantics": "last_model_only",
            }

        payload_extra = {
            "epoch": epoch,
            "global_step": global_step,
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
            "epoch": epoch,
            "global_step": global_step,
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
        ema_state_dict: Optional[Dict[str, Any]] = None,
        ema_decay: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Save a full resume checkpoint atomically.

        Uses build_resume_payload() + AtomicCheckpointWriter for
        atomic writing with unique temp file names. The payload also records
        the manager's best-metric state (``checkpoint_manager_state``) so
        load_resume() can restore it, and ``ema_state_dict`` (from
        ``AveragedModel.state_dict()``) if given. ``ema_decay`` is required
        with ``ema_state_dict`` (EMADecayError otherwise); like epoch and
        global_step it is validated on every rank, so a bad call fails the
        same way everywhere instead of only on the main process.
        """
        epoch = as_step_count(epoch, name="epoch")
        if global_step is not None:
            global_step = as_step_count(global_step, name="global_step")
        check_ema_save_metadata(ema_state_dict, ema_decay)

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
            ema_state_dict=ema_state_dict,
            ema_decay=ema_decay,
        )
        payload["checkpoint_manager_state"] = {
            "best_metric_value": self.best_metric_value,
            "best_metric_name": self.best_metric_name,
            "best_policy_mode": self.best_policy.mode,
            "run_id": self.run_id,
        }

        AtomicCheckpointWriter.save_torch_payload_atomic(self.resume_path, payload)

        requires_trusted_load = include_python_rng or include_numpy_rng

        return {
            "saved": True,
            "checkpoint_path": self.resume_path,
            "checkpoint_type": "resume",
            "checkpoint_semantics": "resume_full",
            "epoch": epoch,
            "global_step": global_step,
            "has_ema_state": ema_state_dict is not None,
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
        ema_model: Optional[torch.optim.swa_utils.AveragedModel] = None,
        ema_decay: Optional[float] = None,
        metric_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a resume checkpoint.

        Delegates to load_resume_checkpoint from resume_utils_hex_v3.py
        which handles the correct restore order:
        1. model, 2. optimizer, 3. scheduler, 4. grad_scaler, 5. EMA, 6. RNG

        Afterwards the best-metric state recorded by save_resume() is
        restored from this manager's best.pt, if best.pt is consistent with
        the record (see _restore_best_metric). Without it a resumed run
        started with best=None and its first epoch replaced a better best.pt
        (issue #40). ``metric_name``, if given, must equal the recorded best
        metric's name; a resume that switches the selection metric starts a
        fresh best. The manager adopts the resumed run's ``run_id``.

        With ``ema_model`` and a checkpoint that carries EMA state,
        ``ema_decay`` is required and must equal the decay recorded next to
        the EMA state; a missing decay on either side or a mismatch raises
        EMADecayError (a ValueError) before anything is restored.

        path explicitly selects the resume checkpoint. If omitted, the
        manager-owned resume_path is used. Callers that expose a configurable
        resume path MUST forward it here so the file checked for existence is
        also the file that is actually loaded.
        """
        checkpoint_path = Path(path) if path is not None else self.resume_path
        report = load_resume_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
            map_location=map_location,
            restore_rng=restore_rng,
            strict=strict,
            trusted=trusted,
            ema_model=ema_model,
            ema_decay=ema_decay,
        )
        state = report.get("checkpoint_manager_state")
        if isinstance(state, dict) and isinstance(state.get("run_id"), str):
            # Resuming continues that run: later saves carry its identity.
            self.run_id = state["run_id"]
        # A manager that already tracked a best value (e.g. an earlier run in
        # this process) must not compare the resumed run against it: start
        # fresh and let a successful restore set the value (Codex #46 T5).
        self.best_metric_value = None
        self.best_metric_name = None
        restored, reason = self._restore_best_metric(state, metric_name=metric_name)
        report["best_metric_restored"] = restored
        report["best_metric_restore_reason"] = reason
        report["best_metric_value"] = self.best_metric_value
        if not restored and reason not in ("absent", "no_best_recorded"):
            logger.warning(
                "Resume: best metric not restored (%s); the next improvement "
                "check starts from %r.",
                reason,
                self.best_metric_value,
            )
        return report

    def _restore_best_metric(
        self, state: Any, *, metric_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Restore best_metric_value from best.pt, if the resume record allows it.

        best.pt in this manager's root_dir is the source of truth (memory must
        match the file). It is adopted when it carries the same policy mode
        and a metric at least as good as the one recorded in the resume
        checkpoint: equal = the state at save time; better = a later epoch of
        the same run improved best.pt after the last periodic resume save
        ("restored_from_newer_best"). Identity is checked first: best.pt must
        carry the recorded run_id and metric name, and the metric name must
        equal ``metric_name`` if the caller names it. A best.pt of another
        run, of another metric, a worse or a missing one is not trusted, so it
        cannot block saving a new best model. Returns (restored, reason).
        """
        if not isinstance(state, dict):
            return False, "absent"
        recorded = state.get("best_metric_value")
        if recorded is None:
            return False, "no_best_recorded"
        if state.get("best_policy_mode") != self.best_policy.mode:
            return False, "policy_mode_mismatch"
        recorded_name = state.get("best_metric_name")
        if metric_name is not None and recorded_name != metric_name:
            return False, "metric_name_mismatch"
        try:
            value = _finite_metric_value(recorded)
        except (TypeError, ValueError):
            return False, "invalid_record"

        best_path = Path(self.best_model_path)
        if not best_path.exists():
            return False, "best_checkpoint_missing"
        try:
            # mmap: read the metadata without materialising the weights.
            best_extra = torch.load(
                best_path, map_location="cpu", weights_only=True, mmap=True
            )["extra"]
            on_disk = _finite_metric_value(best_extra["metric_value"])
            disk_mode = best_extra.get("best_policy", {}).get("mode", self.best_policy.mode)
            disk_run_id = best_extra.get("run_id")
            disk_name = best_extra.get("metric_name")
        except Exception:  # unreadable/legacy best.pt: do not trust the record
            return False, "best_checkpoint_unreadable"
        if disk_run_id is None or disk_run_id != state.get("run_id"):
            return False, "run_id_mismatch"
        if disk_name != recorded_name:
            return False, "metric_name_mismatch"
        if disk_mode != self.best_policy.mode:
            return False, "policy_mode_mismatch"
        at_least_as_good = on_disk <= value if self.best_policy.mode == "min" else on_disk >= value
        if not at_least_as_good:
            return False, "best_checkpoint_mismatch"

        self.best_metric_value = on_disk
        self.best_metric_name = disk_name
        return True, "restored" if on_disk == value else "restored_from_newer_best"

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
        skip_shape_mismatch: bool = False,
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
            skip_shape_mismatch=skip_shape_mismatch,
        )
