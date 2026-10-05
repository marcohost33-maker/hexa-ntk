"""
runner_hex_v3.py
================
Canonical training runner for HexaNTK PDE operator pipeline.

Integrates:
- Anomaly detection (NaN/Inf/Loss-Spike via robust moving median)
- Replay capsule dumping for post-mortem debugging
- EMA model updates (via torch.optim.swa_utils.AveragedModel)
- Checkpoint orchestration (best/last/resume via CheckpointIntegrationManager)
- Mixed precision (AMP with GradScaler, bf16 preferred)
- Gradient clipping (clip_grad_norm_ before optimizer step)
- OOM rescue (gc.collect + cuda.empty_cache with safe variable cleanup)
- Operator adapter validation (OperatorAdapterV3)
- FailureSignature registration

Best practices applied:
- GradScaler.unscale_ BEFORE clip_grad_norm_ (PyTorch AMP requirement)
- torch.isfinite check on loss BEFORE backward (skip update on NaN)
- NaN/Inf validation metric never becomes the best checkpoint (skip + record)
- Empty validation never becomes the best checkpoint (no measurement != 0.0)
- Best checkpoint is evaluated every epoch; last/resume every
  save_every_n_epochs AND after the final epoch
- Resume restores the EMA average and the best-metric state (full state)
- val_nrmse is the pooled relative L2 error sqrt(sum|e|^2 / sum|y|^2)
- set_to_none=True in zero_grad for memory efficiency
- Moving median for spike detection (robust vs regime changes)
- Replay capsule includes model state_dict (not just batch)
- OOM handler: del pred/loss → gc.collect → cuda.empty_cache → then rescue

References:
- PyTorch AMP best practices (7 rules to avoid NaN)
- PyTorch Lightning terminate_on_nan pattern
- APEBench rollout evaluation methodology
"""
from __future__ import annotations

__all__ = [
    "TrainingAnomalyDetector",
    "RunnerConfig",
    "HexTrainingRunner",
]

import collections
import gc
import logging
import math
import statistics
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from checkpoint_integration_hex_v3 import (
    BestCheckpointPolicy,
    CheckpointIntegrationManager,
)
from ema_hex_v3 import create_ema_model
from failure_signature_hex_v3 import FailureSignature
from operator_adapter_hex_v3 import OperatorAdapterV3

logger = logging.getLogger(__name__)


def _is_non_finite_metric(value: Any) -> bool:
    """True for a NaN/+-inf metric.

    Values math.isfinite() cannot judge (10**400, non-numeric, multi-element
    tensors) return False, so the checkpoint manager's contract rejects them.
    """
    try:
        return not math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


# ------------------------------------------------------------------
# Anomaly detection
# ------------------------------------------------------------------


class TrainingAnomalyDetector:
    """
    Robust anomaly detector for training loops.

    Uses moving median for spike detection (more robust than mean
    against legitimate regime changes like curriculum switches).

    Detects:
    - NaN / Inf loss values
    - Loss spikes exceeding spike_multiplier * rolling median
    """

    def __init__(
        self,
        spike_multiplier: float = 3.0,
        window_size: int = 50,
    ) -> None:
        # "nan <= 1.0" is False, so NaN used to pass and disable detection.
        if not (math.isfinite(spike_multiplier) and spike_multiplier > 1.0):
            raise ValueError("spike_multiplier must be a finite number > 1.0")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be an int >= 1")
        self.spike_multiplier = spike_multiplier
        self.window_size = window_size
        # At least one value before a median exists: window_size=1 used to
        # call statistics.median([]) on the first loss (StatisticsError).
        self.min_history = max(1, window_size // 2)
        self.loss_history: collections.deque = collections.deque(maxlen=window_size)

    def check_loss(self, loss_val: float) -> Tuple[bool, str]:
        """
        Check a loss value for anomalies.

        Returns (is_anomaly, reason_string).
        """
        if not torch.isfinite(torch.tensor(loss_val)):
            return True, "NaN_or_Inf_Loss"

        if len(self.loss_history) >= self.min_history:
            median_loss = statistics.median(self.loss_history)
            if median_loss > 0 and loss_val > median_loss * self.spike_multiplier:
                return (
                    True,
                    f"Loss_Spike (val={loss_val:.6f}, median={median_loss:.6f})",
                )

        self.loss_history.append(loss_val)
        return False, ""


# ------------------------------------------------------------------
# Replay capsule
# ------------------------------------------------------------------


def dump_replay_capsule_safe(
    batch_data: Dict[str, Any],
    model: nn.Module,
    output_dir: Path,
    step: int,
) -> str:
    """
    Save batch inputs AND model weights for deterministic failure reproduction.

    Saves to CPU to avoid OOM during rescue operations.
    """
    capsule_id = uuid.uuid4().hex[:8]
    capsule_path = output_dir / f"replay_capsule_step{step}_{capsule_id}.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "batch_data": {},
        "model_state_dict": {},
    }

    # Save batch tensors (CPU copies)
    for k, v in batch_data.items():
        if torch.is_tensor(v):
            try:
                payload["batch_data"][k] = v.detach().cpu()
            except Exception:
                payload["batch_data"][k] = f"<failed to copy: {type(v)}>"
        else:
            payload["batch_data"][k] = str(v)

    # Save model state (CPU copies)
    try:
        payload["model_state_dict"] = {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        }
    except Exception as exc:
        payload["model_state_dict"] = f"<failed: {exc}>"

    torch.save(payload, capsule_path)
    return str(capsule_path)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Training runner configuration."""

    # Paths
    output_dir: str = "./runs/hex_v3"

    # Training
    num_epochs: int = 100
    grad_clip_max_norm: float = 1.0
    log_every_n_steps: int = 10

    # Checkpointing
    save_every_n_epochs: int = 1
    # Model selection uses the pooled relative L2 error computed in _validate
    # (E1=B, 2026-10-05). A configured metric missing from the validation
    # report raises KeyError instead of silently falling back to val_loss.
    best_metric_name: str = "val_nrmse"
    best_metric_mode: str = "min"
    best_metric_min_delta: float = 1e-4

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # AMP
    use_amp: bool = False
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # Learning rate warmup
    warmup_steps: int = 0  # 0 = disabled; set to ~5% of total steps for AMP
    warmup_start_factor: float = 0.01  # LR starts at base_lr * warmup_start_factor

    # Anomaly detection
    spike_multiplier: float = 3.0
    spike_window_size: int = 50

    # DDP
    is_main_process: bool = True

    # Resume
    resume_from: Optional[str] = None
    resume_trusted: bool = False

    def __post_init__(self) -> None:
        """Reject values that would only fail (or silently misbehave) later.

        save_every_n_epochs=0 raised ZeroDivisionError after the first
        epoch; ema_decay > 1 diverges; NaN clip norms disable clipping.
        """

        def _int_at_least(name: str, minimum: int) -> None:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")

        _int_at_least("num_epochs", 0)
        _int_at_least("log_every_n_steps", 0)
        _int_at_least("save_every_n_epochs", 1)
        _int_at_least("warmup_steps", 0)
        _int_at_least("spike_window_size", 1)
        if math.isnan(self.grad_clip_max_norm) or self.grad_clip_max_norm <= 0:
            raise ValueError(
                f"grad_clip_max_norm must be > 0 (inf disables clipping), "
                f"got {self.grad_clip_max_norm!r}"
            )
        if not (0.0 < self.warmup_start_factor <= 1.0):
            raise ValueError(
                f"warmup_start_factor must be in (0, 1], got {self.warmup_start_factor!r}"
            )
        if not (math.isfinite(self.ema_decay) and 0.0 <= self.ema_decay <= 1.0):
            raise ValueError(f"ema_decay must be in [0, 1], got {self.ema_decay!r}")
        if self.amp_dtype not in ("float16", "bfloat16"):
            raise ValueError(
                f"amp_dtype must be 'float16' or 'bfloat16', got {self.amp_dtype!r}"
            )


# ------------------------------------------------------------------
# Training runner
# ------------------------------------------------------------------


class HexTrainingRunner:
    """
    Canonical training runner for HexaNTK operator learning.

    Orchestrates:
    - Training loop with gradient clipping + AMP
    - Anomaly detection + replay capsule dumping
    - EMA model updates
    - Checkpoint management (best/last/resume)
    - Validation with operator adapter
    """

    def __init__(
        self,
        config: RunnerConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        loss_fn: Optional[Callable] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        adapter: Optional[OperatorAdapterV3] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or nn.MSELoss()
        self.scheduler = scheduler
        self.adapter = adapter
        self.device = torch.device(device)

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint manager
        self.ckpt_mgr = CheckpointIntegrationManager(
            self.output_dir / "checkpoints",
            best_policy=BestCheckpointPolicy(
                mode=config.best_metric_mode,
                min_delta=config.best_metric_min_delta,
            ),
            is_main_process=config.is_main_process,
        )

        # Anomaly detector
        self.anomaly_detector = TrainingAnomalyDetector(
            spike_multiplier=config.spike_multiplier,
            window_size=config.spike_window_size,
        )

        # EMA
        self.ema_model: Optional[torch.optim.swa_utils.AveragedModel] = None
        if config.use_ema:
            # One EMA construction path for the toolkit (validation + the
            # scalar fallback for PyTorch < 2.0 live in ema_hex_v3).
            self.ema_model = create_ema_model(
                model, decay=config.ema_decay, use_buffers=True
            )

        # AMP
        self.grad_scaler: Optional[torch.amp.GradScaler] = None
        self.amp_dtype = torch.float16
        if config.use_amp and self.device.type == "cuda":
            if config.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                # bf16 doesn't need GradScaler
                self.grad_scaler = None
            else:
                self.amp_dtype = torch.float16
                self.grad_scaler = torch.amp.GradScaler("cuda")

        # State
        self.global_step: int = 0
        self.start_epoch: int = 0
        self.train_losses: List[float] = []

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def _loader_generators(self) -> Dict[str, torch.Generator]:
        """Explicit torch.Generator objects of the train/val loaders.

        make_reproducible_dataloader_kwargs() gives a shuffled loader its own
        generator. Its state is part of the run: without it, a resumed run
        repeats the seed's first permutation instead of continuing the
        uninterrupted sequence (Codex PR #58 round 2).
        """
        gens: Dict[str, torch.Generator] = {}
        for name, loader in (("train", self.train_loader), ("val", self.val_loader)):
            gen = getattr(loader, "generator", None)
            if isinstance(gen, torch.Generator):
                gens[name] = gen
        return gens

    def _restore_loader_generators(
        self, report: Dict[str, Any], resume_path: Path
    ) -> None:
        """Restore loader generator states saved by _save_checkpoints.

        Checkpoints written before this field existed carry no state: the
        loaders then continue from their seed (the first-epoch permutation
        repeats), which is logged and reported, not silently accepted.
        """
        saved = (report.get("dataloader_state_dict") or {}).get("generator_states") or {}
        restored, missing = [], []
        for name, gen in self._loader_generators().items():
            state = saved.get(name)
            if state is None:
                missing.append(name)
                continue
            if not (torch.is_tensor(state) and state.dtype == torch.uint8):
                raise ValueError(
                    f"Resume checkpoint {resume_path}: generator state for the "
                    f"{name} loader is not a uint8 tensor"
                )
            gen.set_state(state)
            restored.append(name)
        report["loader_generators_restored"] = restored
        report["loader_generators_missing"] = missing
        if missing:
            logger.warning(
                "Resume checkpoint %s carries no generator state for the %s "
                "loader(s) (written before loader state was saved); their "
                "shuffle order restarts from the seed instead of continuing.",
                resume_path,
                ", ".join(missing),
            )

    def _loader_generator_payload(self) -> Optional[Dict[str, Any]]:
        gens = self._loader_generators()
        if not gens:
            return None
        return {"generator_states": {name: gen.get_state() for name, gen in gens.items()}}

    def maybe_resume(self) -> Optional[Dict[str, Any]]:
        """Resume from ``config.resume_from`` if configured.

        Returns None only when no resume path is configured. A configured but
        missing path raises FileNotFoundError (fail-closed): silently starting
        at epoch 0 would turn a typo in the resume path into an unnoticed
        full retraining that overwrites the run directory.
        """
        if self.config.resume_from is None:
            return None
        resume_path = Path(self.config.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"resume_from is set but the checkpoint does not exist: "
                f"{resume_path}"
            )
        report = self.ckpt_mgr.load_resume(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            grad_scaler=self.grad_scaler,
            map_location=self.device,
            trusted=self.config.resume_trusted,
            path=resume_path,
            ema_model=self.ema_model,
            ema_decay=self.config.ema_decay if self.ema_model is not None else None,
            metric_name=self.config.best_metric_name,
        )
        if self.ema_model is not None and not report.get("ema_restored"):
            logger.warning(
                "Resume checkpoint %s carries no EMA state; the EMA average "
                "restarts from the resumed weights.",
                resume_path,
            )
        self._restore_loader_generators(report, resume_path)
        self.start_epoch = report["epoch"] + 1
        if report.get("global_step") is not None:
            self.global_step = report["global_step"]
        logger.info(
            "Resumed from epoch %d, step %d",
            self.start_epoch,
            self.global_step,
        )
        return report

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_one_step(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[float, bool, Optional[float]]:
        """
        Execute one training step with all safety guards.

        Returns (loss_value, is_healthy, grad_norm_before_clip).
        """
        # Initialize to None for safe cleanup on OOM
        pred = None
        loss = None

        try:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            # Learning rate warmup (linear)
            if (
                self.config.warmup_steps > 0
                and self.global_step < self.config.warmup_steps
            ):
                warmup_factor = self.config.warmup_start_factor + (
                    1.0 - self.config.warmup_start_factor
                ) * (self.global_step / self.config.warmup_steps)
                for pg in self.optimizer.param_groups:
                    if "_base_lr" not in pg:
                        pg["_base_lr"] = pg["lr"]
                    pg["lr"] = pg["_base_lr"] * warmup_factor

            x = batch["input"].to(self.device)
            target = batch["target"].to(self.device)

            # Build context if adapter is used
            ctx = None
            if self.adapter is not None and "context" in batch:
                ctx = batch["context"]

            # Forward pass (optionally with AMP)
            use_autocast = (
                self.config.use_amp and self.device.type == "cuda"
            )

            if use_autocast:
                with torch.autocast(
                    device_type="cuda", dtype=self.amp_dtype
                ):
                    if self.adapter is not None and ctx is not None:
                        pred = self.adapter(x, ctx)
                    else:
                        pred = self.model(x)
                    loss = self.loss_fn(pred, target)
            else:
                if self.adapter is not None and ctx is not None:
                    pred = self.adapter(x, ctx)
                else:
                    pred = self.model(x)
                loss = self.loss_fn(pred, target)

            loss_val = loss.item()

            # NaN/Inf check BEFORE backward
            is_anomaly, reason = self.anomaly_detector.check_loss(loss_val)
            if is_anomaly:
                logger.error(
                    "Anomaly at step %d: %s", self.global_step, reason
                )
                cap_path = dump_replay_capsule_safe(
                    batch, self.model, self.output_dir / "failures",
                    self.global_step,
                )
                FailureSignature(
                    kind=reason,
                    step=self.global_step,
                    locus="train_step_loss_check",
                    observable=f"loss={loss_val}",
                    replay_capsule=cap_path,
                ).register(self.output_dir / "failures")

                if "NaN" in reason or "Inf" in reason:
                    # Do NOT backward or step — protect optimizer
                    return loss_val, False, None

            # Backward
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                # Unscale BEFORE clipping (PyTorch AMP requirement)
                self.grad_scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip_max_norm,
            )
            grad_norm_val = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

            # Optimizer step
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()

            # EMA update
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)

            return loss_val, True, grad_norm_val

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.critical(
                    "CUDA OOM at step %d! Attempting rescue...",
                    self.global_step,
                )
                # 1. Destroy references holding VRAM
                del pred
                del loss
                # 2. Python GC + CUDA cache (order matters)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 3. Now safe to dump capsule
                try:
                    cap_path = dump_replay_capsule_safe(
                        batch, self.model,
                        self.output_dir / "failures",
                        self.global_step,
                    )
                    FailureSignature(
                        kind="CUDA_OOM",
                        step=self.global_step,
                        locus="train_step_oom",
                        observable="OOM exception caught",
                        replay_capsule=cap_path,
                    ).register(self.output_dir / "failures")
                except Exception:
                    logger.error("Failed to dump OOM replay capsule")
                raise
            raise

    # ------------------------------------------------------------------
    # Epoch
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> Dict[str, Any]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses: List[float] = []
        epoch_grad_norms: List[float] = []
        n_anomalies = 0

        for batch_idx, batch in enumerate(self.train_loader):
            loss_val, is_healthy, grad_norm = self._train_one_step(batch)
            self.global_step += 1

            if is_healthy:
                epoch_losses.append(loss_val)
                if grad_norm is not None:
                    epoch_grad_norms.append(grad_norm)
            else:
                n_anomalies += 1

            if (
                self.config.log_every_n_steps > 0
                and self.global_step % self.config.log_every_n_steps == 0
            ):
                logger.info(
                    "epoch=%d step=%d loss=%.6f grad_norm=%.4f",
                    epoch,
                    self.global_step,
                    loss_val,
                    grad_norm if grad_norm is not None else -1.0,
                )

        # Scheduler step (per epoch)
        if self.scheduler is not None:
            self.scheduler.step()

        # No healthy step means no measurement: NaN, not a perfect 0.0.
        mean_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        )
        mean_grad_norm = (
            sum(epoch_grad_norms) / len(epoch_grad_norms)
            if epoch_grad_norms
            else float("nan")
        )

        return {
            "epoch": epoch,
            "mean_loss": mean_loss,
            "mean_grad_norm": mean_grad_norm,
            "n_batches": len(epoch_losses),
            "n_anomalies": n_anomalies,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, Any]:
        """
        Run validation.

        EMA handling:
        - If EMA is active, validate with EMA model weights
        - AveragedModel.forward() delegates to its internal module copy,
          so calling ema_model(x, **kwargs) uses the averaged weights
        - The adapter wraps the TRAINING model, not the EMA model.
          For EMA validation, we call ema_model.module directly with
          the same kwargs the adapter would build, or we temporarily
          wrap EMA's module in the adapter.
        """
        # Select which model to evaluate
        if self.ema_model is not None:
            eval_module = self.ema_model.module
        else:
            eval_module = self.model

        eval_module.eval()

        val_losses: List[float] = []
        # Pooled relative L2 error over the whole validation set, accumulated
        # in float64: nRMSE = sqrt(sum ||pred - y||^2 / sum ||y||^2).
        sq_err_sum = 0.0
        sq_ref_sum = 0.0

        for batch in self.val_loader:
            x = batch["input"].to(self.device)
            target = batch["target"].to(self.device)

            # Rebind the canonical adapter to the actual evaluation module.
            # This keeps raw and EMA validation on the same context, dispatch,
            # and output-shape contract as training instead of duplicating it.
            if self.adapter is not None and "context" in batch:
                ctx = batch["context"]
                eval_adapter = OperatorAdapterV3(eval_module, self.adapter.model_type)
                pred = eval_adapter(x, ctx)
            else:
                pred = eval_module(x)

            loss = self.loss_fn(pred, target)
            val_losses.append(float(loss.item()))
            ref = target.detach().to(torch.float64)
            err = pred.detach().to(torch.float64) - ref
            sq_err_sum += float(err.pow(2).sum().item())
            sq_ref_sum += float(ref.pow(2).sum().item())

        # Restore training mode
        self.model.train()

        if val_losses:
            mean_val_loss = sum(val_losses) / len(val_losses)
            # sq_ref_sum == 0 (all-zero targets): the relative metric is
            # undefined -- nan for 0/0, inf for e/0. Both are non-finite, so
            # the best-checkpoint skip path records them; 0/0 must never
            # become a "perfect" best value (Codex PR #58).
            if sq_ref_sum > 0:
                val_nrmse = math.sqrt(sq_err_sum / sq_ref_sum)
            elif sq_err_sum == 0:
                val_nrmse = float("nan")
            else:
                val_nrmse = float("inf")
        else:
            # No validation batch: no measurement. 0.0 used to become the
            # best value and blocked every later real improvement.
            mean_val_loss = float("nan")
            val_nrmse = float("nan")

        return {
            "epoch": epoch,
            "val_loss": mean_val_loss,
            "val_nrmse": val_nrmse,
            "n_batches": len(val_losses),
            "used_ema": self.ema_model is not None,
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Execute the full training run.

        Returns a summary dict with training history.
        """
        # Maybe resume
        resume_report = self.maybe_resume()

        history: Dict[str, List[Any]] = {
            "train": [],
            "val": [],
            "checkpoints": [],
        }

        logger.info(
            "Starting training: epochs=%d-%d, device=%s, ema=%s, amp=%s",
            self.start_epoch,
            self.config.num_epochs - 1,
            self.device,
            self.config.use_ema,
            self.config.use_amp,
        )

        for epoch in range(self.start_epoch, self.config.num_epochs):
            # Train
            train_report = self._train_one_epoch(epoch)
            history["train"].append(train_report)
            self.train_losses.append(train_report["mean_loss"])

            logger.info(
                "Epoch %d: loss=%.6f grad_norm=%.4f lr=%.2e anomalies=%d",
                epoch,
                train_report["mean_loss"],
                train_report["mean_grad_norm"],
                train_report["lr"],
                train_report["n_anomalies"],
            )

            # Validate
            val_report = self._validate(epoch)
            history["val"].append(val_report)

            logger.info(
                "Epoch %d val: loss=%.6f (ema=%s)",
                epoch,
                val_report["val_loss"],
                val_report["used_ema"],
            )

            # Checkpointing: best every epoch (validation already ran, a
            # better model at a non-save epoch was silently lost before);
            # last/resume periodically and always after the final epoch, so
            # the end state of the run is never dropped.
            is_final_epoch = epoch == self.config.num_epochs - 1
            ckpt_reports = self._save_checkpoints(
                epoch=epoch,
                val_report=val_report,
                train_report=train_report,
                save_state=(
                    epoch % self.config.save_every_n_epochs == 0 or is_final_epoch
                ),
            )
            history["checkpoints"].append(ckpt_reports)

        # Final summary
        return {
            "history": history,
            "final_epoch": self.config.num_epochs - 1,
            "global_step": self.global_step,
            "resume_report": resume_report,
        }

    # ------------------------------------------------------------------
    # Checkpoint orchestration
    # ------------------------------------------------------------------

    def _skip_non_finite_best_metric(
        self,
        epoch: int,
        metric_value: Any,
        *,
        kind: str = "NaN_or_Inf_Metric",
        reason: str = "non_finite_metric",
    ) -> Dict[str, Any]:
        """Record a NaN/inf (or absent) metric instead of saving a best checkpoint."""
        metric_name = self.config.best_metric_name
        logger.error(
            "%s: %s=%s at epoch %d; best checkpoint not updated",
            kind,
            metric_name,
            metric_value,
            epoch,
        )
        signature_path = FailureSignature(
            kind=kind,
            step=self.global_step,
            locus="save_checkpoints_best_metric",
            observable=f"{metric_name}={metric_value}",
            extra={"epoch": epoch, "metric_name": metric_name},
        ).register(self.output_dir / "failures")
        return {
            "saved": False,
            "reason": reason,
            "checkpoint_semantics": "best_model",
            "metric_name": metric_name,
            "metric_value": metric_value,
            "best_metric_value": self.ckpt_mgr.best_metric_value,
            "path": self.ckpt_mgr.best_model_path,
            "failure_signature": signature_path,
        }

    def _ema_state_snapshot(self) -> Dict[str, Any]:
        """Host copy of the averaged weights (called lazily by the manager)."""
        return {
            k: v.detach().cpu()
            for k, v in self.ema_model.module.state_dict().items()
        }

    def _save_checkpoints(
        self,
        epoch: int,
        val_report: Dict[str, Any],
        train_report: Dict[str, Any],
        *,
        save_state: bool = True,
    ) -> Dict[str, Any]:
        """Save the best checkpoint if improved; last + resume if save_state.

        The report always has the keys "best", "last" and "resume"; skipped
        periodic saves carry reason "not_scheduled".
        """
        reports: Dict[str, Any] = {}

        if self.config.best_metric_name not in val_report:
            raise KeyError(
                f"configured best metric {self.config.best_metric_name!r} "
                "is absent from validation report"
            )
        metric_value = val_report[self.config.best_metric_name]

        # EMA state for best model: a callable the manager invokes only when
        # the best policy accepted the metric on the main process, so a
        # non-improving epoch never copies the EMA to the host (Codex #46 T6).
        ema_state = self._ema_state_snapshot if self.ema_model is not None else None

        # Best model. A NaN/inf metric is never promoted: the manager would
        # reject it with ValueError; the runner records it and keeps training
        # (same policy as a NaN train loss: skip, record, continue).
        if val_report.get("n_batches", 1) == 0:
            best_report = self._skip_non_finite_best_metric(
                epoch,
                metric_value,
                kind="Empty_Validation",
                reason="empty_validation",
            )
        elif _is_non_finite_metric(metric_value):
            best_report = self._skip_non_finite_best_metric(epoch, metric_value)
        else:
            best_report = self.ckpt_mgr.save_best_model_checkpoint(
                model=self.model,
                metric_value=metric_value,
                metric_name=self.config.best_metric_name,
                epoch=epoch,
                global_step=self.global_step,
                ema_model_state=ema_state,
                extra={
                    "train_loss": train_report["mean_loss"],
                    "val_report": val_report,
                },
            )
        reports["best"] = best_report

        if not save_state:
            not_scheduled = {"saved": False, "reason": "not_scheduled"}
            reports["last"] = {**not_scheduled, "checkpoint_semantics": "last_model_only"}
            reports["resume"] = {**not_scheduled, "checkpoint_semantics": "resume_full"}
            return reports

        # Last model
        last_report = self.ckpt_mgr.save_last_model_checkpoint(
            model=self.model,
            epoch=epoch,
            global_step=self.global_step,
        )
        reports["last"] = last_report

        # Resume (full state)
        resume_report = self.ckpt_mgr.save_resume(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            global_step=self.global_step,
            scheduler=self.scheduler,
            grad_scaler=self.grad_scaler,
            include_python_rng=True,
            include_numpy_rng=True,
            include_torch_rng=True,
            ema_state_dict=(
                self.ema_model.state_dict() if self.ema_model is not None else None
            ),
            ema_decay=self.config.ema_decay if self.ema_model is not None else None,
            include_dataloader_state=self._loader_generator_payload(),
            extra={
                "train_loss": train_report["mean_loss"],
                "val_loss": val_report["val_loss"],
            },
        )
        reports["resume"] = resume_report

        if best_report.get("saved"):
            logger.info(
                "New best model saved: %s=%.6f at epoch %d",
                self.config.best_metric_name,
                metric_value,
                epoch,
            )

        return reports
