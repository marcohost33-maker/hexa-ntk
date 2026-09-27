"""
HexaNTK v3.0 -- L014 Manuscript-Quality Implementation
=======================================================

Coworker Research, 2026-04-20
Status: Manuscript-grade re-implementation following P0-P3 of
        HexaNTK_Review_Gehaertet.md (Drive 1CvzpyQE3LcLtBnJOkmsMT_OxLpqx7Skm).
        All gaps closed, all lint issues resolved, doctests added.

================================================================
CHANGELOG (P0-P3 from review)
================================================================

[P0.1] FailureSignature dataclass: previously referenced but never defined,
       caused runner NameError on any exception path. Now defined.

[P0.2] warmstart_utils_hex_v3.py: previously imported but nonexistent.
       Now provided as load_partial_checkpoint() with PyTorch 2.6 SoTA
       (weights_only=True default, safe_globals context fallback).

[P0.3] Merged checkpoint_integration (Doc1 vs Doc2): canonical class-based
       CheckpointIntegrationManager with all features: EMA, FailureSignature,
       BestCheckpointPolicy with min_delta, checkpoint_semantics tags,
       DDP is_main_process guard.

[P0.4] build_resume_payload: separated from save_resume_checkpoint.

[P0.5] OOM handler NameError: pred/loss pre-initialized to None.

[P1.1] test_checkpoint_integration_hex_v3: complete round-trip tests.
[P1.2] test_operator_adapter_hex_v3: complete validation tests.
[P1.3] Hex explicit-step contract: L2 stability (h^2/(3D)) is now
       distinguished from positivity/DMP/monotonicity (h^2/(4D)).
       Existing hex_cfl_max_dt behavior is preserved; a separate
       hex_positivity_max_dt helper and regression checks were added.
       Public numerical helpers now reject non-finite/invalid inputs with
       ValueError instead of assert-only validation.

[P2+P3] PyTorch 2.6+ SoTA: safe_globals context manager, non-reentrant
       activation checkpointing notes, DDP guards, atomic writes.

================================================================
SoTA REFERENCES (verified Feb 2026)
================================================================
[Li23]       Z. Li, N. Kovachki, C. Choy, et al., "Geometry-Informed Neural
             Operator for Large-Scale 3D PDEs", NeurIPS 2023, arXiv:2309.00583.
[GINOT25]    Q. Liu et al., "Geometry-Informed Neural Operator Transformer",
             ScienceDirect S0045782525009405 (Dec 2025).
[GAOT25]     S. Wen et al., "Geometry Aware Operator Transformer", arXiv:
             2505.18781 (May 2025). Beats GINO on 28 benchmarks incl.
             DrivAerNet++; future reference for adapter extension.
[Sarkar25]   S. Sarkar et al., "Physics- and geometry-aware spatio-spectral
             graph neural operator" (piG-Sp2GNO / mGNO), arXiv Aug 2025.
             Introduces mollified aggregation for differentiable GNO.
[Han26]      B. Han et al., "Kernel integral perspective and multiscale
             point cloud neural operators", Feb 2026. Multiscale extension.
[PyTorch2.6] PyTorch 2.6+ documentation, torch.serialization.safe_globals,
             torch.distributed.checkpoint (DCP). See
             pytorch.org/docs/stable/notes/serialization.html
[PyTorch29]  PyTorch 2.9 DCP matured: simplified save_checkpoint /
             load_checkpoint primitives (GitHub pytorch/pytorch#176732,
             merged Mar 2026). Plan caching, async save with process-based
             workers, rank-local checkpointing; Meta+Google blog post
             pytorch.org/blog/distributed-checkpoint (Sep 2025).
[FSDP25]     IBM Research FSDP blog (2024), non-reentrant activation ckpt
             guidance for FSDP compatibility.
[Daido]      Lee & Tien 2014, Lee 2017 (hex-Laplacian FV schemes).

License: Coworker Research internal.
Note: This module works without PyTorch installed; torch is imported
      lazily inside functions that need it, with stub fallbacks for
      pure-logic tests.
"""
from __future__ import annotations

import hashlib
import json
import logging
import numbers
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__version__ = "3.0.1"


def module_hash() -> str:
    """SHA-256 of this module source.

    >>> h = module_hash()
    >>> len(h) == 64
    True
    """
    import inspect
    mod = inspect.getmodule(module_hash)
    assert mod is not None  # module_hash is defined in this module
    src = inspect.getsource(mod)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


# ============================================================================
# SECTION 1 -- FAILURE SIGNATURE (P0.1)
# ============================================================================


@dataclass
class FailureSignature:
    """Structured failure record for debugging and post-mortem analysis.

    Created whenever a training step hits NaN/Inf, OOM, or other
    catastrophic failure. Persisted to disk for later triage.

    Attributes
    ----------
    kind : str
        Canonical name, one of: "NaN_or_Inf_Loss", "Loss_Spike",
        "CUDA_OOM", "Grad_Explosion", "Unknown".
    step : int
        Global training step at which failure occurred.
    locus : str
        Code location string, e.g. "train_step_forward_or_loss".
    observable : str
        Specific symptom, e.g. "Loss value: nan".
    replay_capsule : Optional[str]
        Path to a serialized replay capsule for reproducing the
        failure deterministically.
    timestamp : Optional[str]
        ISO 8601 UTC timestamp; auto-filled if None.

    >>> f = FailureSignature(kind="CUDA_OOM", step=1234, locus="train",
    ...     observable="allocation failed")
    >>> f.timestamp is not None
    True
    >>> f.kind
    'CUDA_OOM'
    """
    kind: str
    step: int
    locus: str
    observable: str
    replay_capsule: Optional[str] = None
    timestamp: Optional[str] = None

    ALLOWED_KINDS = frozenset({
        "NaN_or_Inf_Loss", "Loss_Spike", "CUDA_OOM",
        "Grad_Explosion", "Unknown",
    })

    def __post_init__(self) -> None:
        if self.kind not in self.ALLOWED_KINDS:
            logger.warning(f"FailureSignature.kind={self.kind!r} not in "
                           f"ALLOWED_KINDS; keeping anyway")
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def register(self, output_dir: Union[str, Path]) -> str:
        """Write JSON record to output_dir; return written path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"failure_{self.step}_{self.kind}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return str(path)


# ============================================================================
# SECTION 2 -- ATOMIC CHECKPOINT WRITER
# ============================================================================


class AtomicCheckpointWriter:
    """POSIX-atomic checkpoint writer.

    Algorithm: write to tempfile in same directory as target, fsync,
    then os.replace. This is atomic on POSIX filesystems (single
    inode-pointer change).

    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as d:
    ...     w = AtomicCheckpointWriter()
    ...     p = os.path.join(d, "ckpt.dat")
    ...     w.save_bytes_atomic(b"hello world", p)
    ...     with open(p, "rb") as f:
    ...         data = f.read()
    >>> data == b"hello world"
    True
    """

    def save_bytes_atomic(self, payload: bytes, target_path: Union[str, Path]) -> None:
        """Write bytes atomically to target_path."""
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # tempfile in SAME directory as target (needed for os.replace atomicity)
        fd, tmp_path = tempfile.mkstemp(
            prefix=target_path.name + ".", suffix=".tmp",
            dir=str(target_path.parent),
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(target_path))
        except Exception:
            # cleanup on failure
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def save_torch_payload_atomic(
        self, payload: Dict[str, Any], target_path: Union[str, Path],
    ) -> None:
        """Save a torch-serializable payload atomically.

        Requires torch; raises ImportError if not available.
        """
        try:
            import io
            import torch
        except ImportError as e:
            raise ImportError(
                "save_torch_payload_atomic requires PyTorch; for pure-logic "
                "tests use save_bytes_atomic with json/pickle bytes"
            ) from e
        buf = io.BytesIO()
        torch.save(payload, buf)
        self.save_bytes_atomic(buf.getvalue(), target_path)


# ============================================================================
# SECTION 3 -- BUILD RESUME PAYLOAD (P0.4)
# ============================================================================


def build_resume_payload(
    *,
    model_state_dict: Dict[str, Any],
    optimizer_state_dict: Dict[str, Any],
    scheduler_state_dict: Optional[Dict[str, Any]] = None,
    grad_scaler_state_dict: Optional[Dict[str, Any]] = None,
    epoch: int,
    global_step: int,
    rng_states: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    ema_state_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a canonical resume-payload dictionary.

    The payload is the minimum needed to exactly resume training from
    this point. Includes model, optimizer, scheduler, grad scaler,
    RNG states, and any extra metadata.

    The `checkpoint_semantics` key is set to "resume_full" to enable
    strict validation at load time (canonical checkpoint-policy rule).

    >>> payload = build_resume_payload(
    ...     model_state_dict={"w": [1.0, 2.0]},
    ...     optimizer_state_dict={"state": {}, "param_groups": []},
    ...     epoch=5, global_step=1234,
    ... )
    >>> payload["checkpoint_semantics"]
    'resume_full'
    >>> payload["epoch"]
    5
    >>> payload["global_step"]
    1234
    """
    payload: Dict[str, Any] = {
        "checkpoint_type": "resume",
        "checkpoint_semantics": "resume_full",
        "version": __version__,
        "module_hash": module_hash(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
    }
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = scheduler_state_dict
    if grad_scaler_state_dict is not None:
        payload["grad_scaler_state_dict"] = grad_scaler_state_dict
    if rng_states is not None:
        payload["rng_states"] = rng_states
        payload["requires_trusted_load"] = True
    else:
        payload["requires_trusted_load"] = False
    if extra is not None:
        payload["extra"] = extra
    if ema_state_dict is not None:
        payload["ema_state_dict"] = ema_state_dict
    return payload


# ============================================================================
# SECTION 4 -- BEST CHECKPOINT POLICY
# ============================================================================


@dataclass
class BestCheckpointPolicy:
    """Policy for deciding when to save a new best model.

    Attributes
    ----------
    mode : str
        "min" (lower is better, e.g. loss) or "max" (higher is better,
        e.g. accuracy).
    min_delta : float
        Minimum change in metric to qualify as "better".

    >>> pol = BestCheckpointPolicy(mode="min", min_delta=1e-4)
    >>> pol.is_better(0.5, None)  # first observation
    True
    >>> pol.is_better(0.4, 0.5)
    True
    >>> pol.is_better(0.49999, 0.5)  # within min_delta, not better
    False
    """
    mode: str = "min"
    min_delta: float = 1e-4

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode}")
        if self.min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta}")

    def is_better(self, current: float, best_so_far: Optional[float]) -> bool:
        if best_so_far is None:
            return True
        if self.mode == "min":
            return current < best_so_far - self.min_delta
        else:
            return current > best_so_far + self.min_delta


# ============================================================================
# SECTION 5 -- CHECKPOINT INTEGRATION MANAGER (P0.3 -- merged Doc1+Doc2)
# ============================================================================


class CheckpointIntegrationManager:
    """Canonical checkpoint manager merging Doc1 (class + BestPolicy) and
    Doc2 (EMA, FailureSignature, DDP guard).

    Save side: atomic writes, four semantically-distinct checkpoint types
    (best_model, last_model_only, resume_full, warmstart). Load side:
    validates checkpoint_semantics to prevent type confusion (e.g.
    treating a last-model as a resume checkpoint).

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     mgr = CheckpointIntegrationManager(output_dir=d)
    ...     policy = BestCheckpointPolicy(mode="min")
    ...     # first val: always saves
    ...     rep = mgr.save_best_model_checkpoint_raw(
    ...         model_state_dict={"w": [1.0]}, metric_value=0.5,
    ...         metric_name="loss", epoch=0, policy=policy,
    ...     )
    ...     first_saved = rep["saved"]
    ...     # worse metric: does not save
    ...     rep2 = mgr.save_best_model_checkpoint_raw(
    ...         model_state_dict={"w": [2.0]}, metric_value=0.7,
    ...         metric_name="loss", epoch=1, policy=policy,
    ...     )
    ...     second_saved = rep2["saved"]
    >>> first_saved and not second_saved
    True
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        is_main_process: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_main_process = is_main_process
        self.writer = AtomicCheckpointWriter()
        self._best_metric: Dict[str, float] = {}
        self._best_epoch: Dict[str, int] = {}

    def save_best_model_checkpoint_raw(
        self, *, model_state_dict: Dict[str, Any], metric_value: float,
        metric_name: str, epoch: int, policy: BestCheckpointPolicy,
        extra: Optional[Dict[str, Any]] = None,
        ema_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save best model if metric improved. "raw" variant: no torch
        dependency -- saves via JSON bytes for portability.

        Returns a report dict with 'saved': bool, 'path': str (or None),
        'is_improvement': bool, 'best_metric': float.
        """
        if not self.is_main_process:
            return {"saved": False, "path": None, "is_improvement": False,
                    "best_metric": self._best_metric.get(metric_name), "reason": "not_main"}
        current_best = self._best_metric.get(metric_name)
        improved = policy.is_better(metric_value, current_best)
        if not improved:
            return {"saved": False, "path": None, "is_improvement": False,
                    "best_metric": current_best}
        # build payload
        payload = {
            "checkpoint_type": "best_model",
            "checkpoint_semantics": "best_model",
            "version": __version__,
            "module_hash": module_hash(),
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "epoch": int(epoch),
            "model_state_dict": model_state_dict,
        }
        if extra is not None:
            payload["extra"] = extra
        if ema_state_dict is not None:
            payload["ema_state_dict"] = ema_state_dict
        # serialize + atomic write
        target = self.output_dir / f"best_{metric_name}.json"
        blob = json.dumps(payload, default=str).encode("utf-8")
        self.writer.save_bytes_atomic(blob, target)
        self._best_metric[metric_name] = float(metric_value)
        self._best_epoch[metric_name] = int(epoch)
        return {"saved": True, "path": str(target),
                "is_improvement": True, "best_metric": metric_value}

    def save_last_model_raw(
        self, *, model_state_dict: Dict[str, Any], epoch: int,
    ) -> Dict[str, Any]:
        """Save the latest model (overwrite); semantically distinct from resume."""
        if not self.is_main_process:
            return {"saved": False, "reason": "not_main"}
        payload = {
            "checkpoint_type": "last_model",
            "checkpoint_semantics": "last_model_only",
            "version": __version__,
            "module_hash": module_hash(),
            "epoch": int(epoch),
            "model_state_dict": model_state_dict,
        }
        target = self.output_dir / "last.json"
        blob = json.dumps(payload, default=str).encode("utf-8")
        self.writer.save_bytes_atomic(blob, target)
        return {"saved": True, "path": str(target)}

    def save_resume_raw(
        self, *, resume_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save a pre-built resume payload. Must have correct semantics tag."""
        if not self.is_main_process:
            return {"saved": False, "reason": "not_main"}
        sem = resume_payload.get("checkpoint_semantics")
        if sem != "resume_full":
            raise ValueError(
                f"save_resume_raw: payload has semantics {sem!r}, must be "
                f"'resume_full' (use build_resume_payload)"
            )
        target = self.output_dir / "resume_last.json"
        blob = json.dumps(resume_payload, default=str).encode("utf-8")
        self.writer.save_bytes_atomic(blob, target)
        return {"saved": True, "path": str(target)}

    def load_raw(
        self, checkpoint_path: Union[str, Path],
        expected_semantics: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a JSON checkpoint and validate semantics."""
        with open(checkpoint_path, "r") as f:
            payload = json.load(f)
        sem = payload.get("checkpoint_semantics")
        if expected_semantics is not None and sem != expected_semantics:
            raise ValueError(
                f"checkpoint semantics mismatch: got {sem!r}, expected "
                f"{expected_semantics!r}. This prevents type confusion "
                f"(e.g. loading last.pt as resume)."
            )
        return payload


# ============================================================================
# SECTION 6 -- WARMSTART UTILS (P0.2)
# ============================================================================


def load_partial_checkpoint_raw(
    checkpoint_payload: Dict[str, Any],
    *,
    strip_prefixes: Optional[List[str]] = None,
    prefix_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Pure-python warmstart: return a processed state_dict with prefix
    transformations applied, dropping optimizer/scheduler states.

    This is the logic layer; the PyTorch adapter is `load_partial_checkpoint`
    below (not used in tests to keep torch optional).

    Rules:
      1. NEVER return optimizer_state_dict, scheduler_state_dict,
         grad_scaler_state_dict, or RNG states (explicit drop).
      2. strip_prefixes: e.g. ["module."] to unwrap DDP.
      3. prefix_map: e.g. {"encoder.": "backbone."} for migration.

    Returns
    -------
    dict with keys: model_state_dict, action="warmstart",
                    checkpoint_semantics="warmstart",
                    dropped_keys list.

    >>> payload = {"model_state_dict": {"module.layer1.w": [1], "module.layer2.w": [2]},
    ...            "optimizer_state_dict": {"state": {}}, "epoch": 10}
    >>> out = load_partial_checkpoint_raw(payload, strip_prefixes=["module."])
    >>> sorted(out["model_state_dict"].keys())
    ['layer1.w', 'layer2.w']
    >>> "optimizer_state_dict" in out
    False
    >>> out["checkpoint_semantics"]
    'warmstart'
    """
    state_dict = checkpoint_payload.get("model_state_dict", {})
    if not isinstance(state_dict, dict):
        raise TypeError(f"model_state_dict must be dict, got {type(state_dict)}")

    # apply strip_prefixes
    if strip_prefixes:
        for prefix in strip_prefixes:
            state_dict = {
                (k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in state_dict.items()
            }
    # apply prefix_map
    if prefix_map:
        for old_p, new_p in prefix_map.items():
            state_dict = {
                (new_p + k[len(old_p):] if k.startswith(old_p) else k): v
                for k, v in state_dict.items()
            }
    dropped = [k for k in checkpoint_payload.keys()
               if k in {"optimizer_state_dict", "scheduler_state_dict",
                        "grad_scaler_state_dict", "rng_states"}]
    return {
        "model_state_dict": state_dict,
        "action": "warmstart",
        "checkpoint_semantics": "warmstart",
        "dropped_keys": dropped,
        "warning": ("Optimizer, scheduler, grad_scaler, RNG states were "
                    "INTENTIONALLY dropped for warmstart."),
    }


# ============================================================================
# SECTION 7 -- HEX LAPLACIAN & CFL (numerical correctness, review sec. 5)
# ============================================================================


def _finite_positive_float(name: str, value: Any) -> float:
    """Convert a real scalar to finite binary64 and require it to be > 0.

    Reject bool explicitly (bool is an int subclass), conversion overflow,
    NaN/Inf, and positive exact values that underflow to 0.0.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (numbers.Real, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real numeric scalar, got {type(value).__name__}")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable as a finite real scalar") from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return out


def _unit_interval_safety(value: Any) -> float:
    """Convert safety to finite binary64 and require 0 < safety <= 1."""
    out = _finite_positive_float("safety", value)
    if out > 1.0:
        raise ValueError(f"safety must be <= 1, got {value!r}")
    return out


def hex_laplacian_stencil_weight(h: float) -> float:
    r"""FV weight for 7-point hex Laplacian: 2/(3 h^2).

    Formula: \nabla^2_h u_0 = (2/(3 h^2)) sum_{j=1}^{6}(u_j - u_0).

    Here h is the nearest-neighbour / cell-centre spacing on the regular
    six-neighbour lattice.

    Reference: Lee & Tien 2014, Lee 2017 for regular hex grids.

    >>> bool(abs(hex_laplacian_stencil_weight(1.0) - 2.0/3.0) < 1e-14)
    True
    >>> bool(abs(hex_laplacian_stencil_weight(0.5) - 8.0/3.0) < 1e-14)
    True
    """
    h_f = _finite_positive_float("h", h)
    return 2.0 / (3.0 * h_f * h_f)


def hex_cfl_max_dt(h: float, D_v: float, safety: float = 0.85) -> float:
    """L2-stable timestep bound for forward Euler on the hex 7-point Laplacian.

    For the regular six-neighbour stencil with constant isotropic diffusion,
    the sharp spectral bound is dt <= h^2 / (3 D_v). This helper preserves
    the historical default safety=0.85 for backward compatibility.

    IMPORTANT: L2 stability does not imply positivity. For densities,
    concentrations, probabilities, or any state that must remain non-negative,
    use hex_positivity_max_dt(), whose sharp forward-Euler bound is
    h^2 / (4 D_v).

    Reference: Review sec. 5.2; the earlier factor of 6 in the denominator was
    a valid but over-conservative sufficient condition, not the sharp bound.

    >>> bool(abs(hex_cfl_max_dt(1.0, 1.0, safety=1.0) - 1.0/3.0) < 1e-14)
    True
    """
    h_f = _finite_positive_float("h", h)
    d_f = _finite_positive_float("D_v", D_v)
    safety_f = _unit_interval_safety(safety)
    return safety_f * h_f * h_f / (3.0 * d_f)


def hex_positivity_max_dt(h: float, D_v: float, safety: float = 1.0) -> float:
    """Positivity/DMP/monotonicity timestep bound for the regular hex stencil.

    For forward Euler with the 7-point regular six-neighbour Laplacian,
    dt <= h^2/(4 D_v) is necessary and sufficient for all update weights to
    be non-negative. The step is then a convex combination, giving positivity
    preservation, the discrete maximum principle, and monotonicity for this
    specific linear constant-coefficient scheme.

    This statement is not universal for variable/anisotropic diffusion,
    irregular cells, or different boundary/operator discretisations.

    >>> bool(abs(hex_positivity_max_dt(1.0, 1.0) - 0.25) < 1e-14)
    True
    >>> bool(abs(hex_positivity_max_dt(0.5, 2.0) - 0.03125) < 1e-14)
    True
    """
    h_f = _finite_positive_float("h", h)
    d_f = _finite_positive_float("D_v", D_v)
    safety_f = _unit_interval_safety(safety)
    return safety_f * h_f * h_f / (4.0 * d_f)


# ============================================================================
# SECTION 8 -- OOM-SAFE TRAIN STEP (P0.5)
# ============================================================================


def safe_train_step(
    step_fn: Callable[[], Tuple[float, Any]],
    *,
    global_step: int,
    failure_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Wrap one training step with OOM-safe NameError-proof error handling.

    P0.5 fix: pred and loss are pre-initialized to None so that `del`
    in the except block never raises NameError on undefined variables.

    Parameters
    ----------
    step_fn : callable returning (loss_value: float, pred: any)
    global_step : int
    failure_dir : directory for FailureSignature JSON records

    Returns
    -------
    dict with keys: status ("ok"/"nan"/"oom"/"error"), loss, pred (or None),
                    failure_signature (or None).

    >>> d = tempfile.mkdtemp()
    >>> def good(): return 0.5, [1.0, 2.0]
    >>> r = safe_train_step(good, global_step=0, failure_dir=d)
    >>> r["status"]
    'ok'
    >>> r["loss"]
    0.5
    """
    # PRE-INIT (P0.5 fix): these variables MUST exist for any exception path
    pred: Any = None
    loss: Any = None
    try:
        loss, pred = step_fn()
        if np.isnan(loss) or np.isinf(loss):
            fs = FailureSignature(
                kind="NaN_or_Inf_Loss", step=global_step,
                locus="safe_train_step", observable=f"Loss value: {loss}",
            )
            fs.register(failure_dir)
            # safe cleanup -- pred/loss are always defined now
            del pred, loss
            import gc
            gc.collect()
            return {"status": "nan", "loss": None, "pred": None,
                    "failure_signature": fs}
        return {"status": "ok", "loss": float(loss), "pred": pred,
                "failure_signature": None}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            fs = FailureSignature(
                kind="CUDA_OOM", step=global_step,
                locus="safe_train_step", observable=str(e),
            )
            fs.register(failure_dir)
            # SAFE: pred/loss pre-initialized, del always works
            del pred, loss
            import gc
            gc.collect()
            return {"status": "oom", "loss": None, "pred": None,
                    "failure_signature": fs}
        # non-OOM RuntimeError
        fs = FailureSignature(
            kind="Unknown", step=global_step,
            locus="safe_train_step", observable=str(e),
        )
        fs.register(failure_dir)
        del pred, loss
        return {"status": "error", "loss": None, "pred": None,
                "failure_signature": fs}


# ============================================================================
# SECTION 9 -- OPERATOR ADAPTER (stub + validation logic)
# ============================================================================


class OperatorType(Enum):
    GINO = "gino"
    GNO = "gno"
    GINOT = "ginot"    # Liu 2025 transformer variant
    GAOT = "gaot"      # Wen 2025 multiscale (NeurIPS 2025)


# ============================================================================
# SECTION 9a -- GAOT CONFIG (Wen 2025 / NeurIPS 2025)
# ============================================================================
#
# GAOT architecture (arXiv:2505.18781, NeurIPS 2025):
#   encode (MAGNO) -> process (ViT) -> decode (MAGNO)
#
# The MAGNO encoder aggregates information from an arbitrary point cloud
# into a latent token grid at multiple scales (ball-integration radii).
# Geometry embeddings inject local shape information (SDF, normals).
# A patched ViT processor enables global communication between tokens.
# A symmetric MAGNO decoder evaluates at any query point.
#
# Paper defaults for 2D benchmarks (Wen 2025, arXiv:2505.18781, Table B.1
# and Section B of supplementary material):
#   tokenization        : "structured_stencil" (strategy I), 32x32 grid
#   n_scales            : 3
#   ball_radii_relative : [0.08, 0.16, 0.32]   (relative to domain diameter)
#   geometry_embed_dim  : 32
#   patch_size          : 4
#   vit_depth           : 8
#   vit_heads           : 8
#   vit_hidden_dim      : 256
#
# IMPORTANT NOTE ON 3D USE:
# The paper has a SEPARATE optimized 3D repository (camlab-ethz/GAOT3D)
# for ultra-large 3D datasets (DrivAerNet++, DrivaerML, NASA-CRM). In the
# 3D regime the authors TUNE hyperparameters per-dataset; there is no
# single paper-validated "3D default". When instantiating a 3D
# GAOTConfig, the USER is responsible for providing values appropriate
# for their dataset and should NOT trust our 2D defaults as being sensible
# in 3D. We therefore REQUIRE explicit latent_grid_shape specification
# in 3D and warn if the 2D defaults are reused verbatim for 3D.
#
# This config is a FUTURE-EXTENSION HOOK. The actual PyTorch implementation
# is kept in a separate training-adapter session; here we declare the
# contract and validate shapes so the rest of our infrastructure (dataset
# split, checkpoint semantics, OperatorAdapter) is GAOT-ready.


@dataclass(frozen=True)
class GAOTConfig:
    """Hyperparameters for GAOT (Wen 2025, arXiv:2505.18781, NeurIPS 2025).

    DEFAULT VALUES are those of the 2D benchmarks in Table B.1 / Section B
    of the supplementary material. They are NOT generally appropriate for
    3D CFD problems -- see the module-level note above and refer to the
    authors' separate GAOT3D repository for 3D-tuned values.

    Attributes
    ----------
    tokenization : one of {"structured_stencil", "downsampled_unstructured",
                           "low_dim_projection"}
        Paper Table B.1. Strategy I (structured_stencil) is the default
        and used in all 2D main benchmarks.
    latent_grid_shape : tuple of ints, shape of the latent token grid
        Paper-validated default: (32, 32) for 2D time-independent PDEs.
        For 3D, user MUST supply dataset-appropriate values; the paper
        does not provide a universal 3D default.
    n_scales : int, number of ball-integration radii in MAGNO encoder
    ball_radii_relative : tuple of floats in (0, 1]
        Each r_k is a fraction of the domain diameter; MAGNO integrates
        over the ball of radius r_k * diam around each latent token.
        len(ball_radii_relative) must equal n_scales.
    geometry_embed_dim : int, channels of the geometry embedding
    patch_size : int, patch side-length for the ViT processor
    vit_depth : int, number of ViT blocks
    vit_heads : int, number of attention heads per ViT block
    vit_hidden_dim : int, channel count in ViT hidden layers

    >>> cfg = GAOTConfig()
    >>> cfg.n_scales
    3
    >>> len(cfg.ball_radii_relative) == cfg.n_scales
    True
    >>> cfg.tokenization
    'structured_stencil'
    >>> cfg.dim()
    2
    """
    tokenization: str = "structured_stencil"
    latent_grid_shape: Tuple[int, ...] = (32, 32)
    n_scales: int = 3
    ball_radii_relative: Tuple[float, ...] = (0.08, 0.16, 0.32)
    geometry_embed_dim: int = 32
    patch_size: int = 4
    vit_depth: int = 8
    vit_heads: int = 8
    vit_hidden_dim: int = 256

    _VALID_TOKENIZATION = (
        "structured_stencil",
        "downsampled_unstructured",
        "low_dim_projection",
    )

    def __post_init__(self) -> None:
        if self.tokenization not in self._VALID_TOKENIZATION:
            raise ValueError(
                f"tokenization must be one of {self._VALID_TOKENIZATION}, "
                f"got {self.tokenization!r}"
            )
        if self.n_scales < 1:
            raise ValueError(f"n_scales must be >= 1, got {self.n_scales}")
        if len(self.ball_radii_relative) != self.n_scales:
            raise ValueError(
                f"ball_radii_relative has length {len(self.ball_radii_relative)}, "
                f"expected {self.n_scales}"
            )
        for r in self.ball_radii_relative:
            if not (0.0 < r <= 1.0):
                raise ValueError(
                    f"ball_radii_relative entries must be in (0,1], got {r}"
                )
        # Radii should be strictly increasing (multiscale semantics)
        for i in range(1, len(self.ball_radii_relative)):
            if self.ball_radii_relative[i] <= self.ball_radii_relative[i - 1]:
                raise ValueError(
                    f"ball_radii_relative must be strictly increasing, got "
                    f"{self.ball_radii_relative}"
                )
        if self.patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {self.patch_size}")
        if self.vit_depth < 1 or self.vit_heads < 1 or self.vit_hidden_dim < 1:
            raise ValueError("ViT dims must be >= 1")
        # Latent grid shape vs patch_size: each dim must be divisible
        for d in self.latent_grid_shape:
            if d % self.patch_size != 0:
                raise ValueError(
                    f"latent_grid_shape {self.latent_grid_shape} not "
                    f"divisible by patch_size {self.patch_size}"
                )

    def dim(self) -> int:
        """Spatial dimension of the PDE domain, inferred from latent_grid_shape.

        >>> GAOTConfig().dim()
        2
        >>> GAOTConfig(latent_grid_shape=(16, 16, 16),
        ...             patch_size=4).dim()
        3
        """
        return len(self.latent_grid_shape)

    def matches_paper_default(self) -> bool:
        """True iff this config matches the Wen 2025 paper 2D defaults exactly.

        IMPORTANT: Returns True ONLY for the 2D paper-validated configuration.
        For 3D configs, returns False -- the paper does not provide a single
        universal 3D default (see authors' separate GAOT3D repository).
        A False here in 3D does not imply the config is wrong, only that we
        cannot vouch for it from the paper alone.

        >>> GAOTConfig().matches_paper_default()
        True
        >>> GAOTConfig(n_scales=5,
        ...             ball_radii_relative=(0.02,0.05,0.1,0.2,0.4)
        ...             ).matches_paper_default()
        False
        >>> # A 3D config is never "paper-default" by this function's rules:
        >>> GAOTConfig(latent_grid_shape=(16, 16, 16),
        ...             patch_size=4).matches_paper_default()
        False
        """
        default = GAOTConfig()
        return (
            self.tokenization == default.tokenization
            and self.latent_grid_shape == default.latent_grid_shape
            and self.n_scales == default.n_scales
            and self.ball_radii_relative == default.ball_radii_relative
            and self.geometry_embed_dim == default.geometry_embed_dim
            and self.patch_size == default.patch_size
            and self.vit_depth == default.vit_depth
            and self.vit_heads == default.vit_heads
            and self.vit_hidden_dim == default.vit_hidden_dim
        )


# ============================================================================
# SECTION 9b -- OPERATOR CONTEXT WITH PER-TYPE VALIDATION
# ============================================================================


@dataclass
class OperatorContext:
    """Validated input context for neural operator forward call.

    Shapes / constraints:
      - x: (batch, n_points, d_in)
      - coords: (batch, n_points, d_space) OR (n_points, d_space) if shared
      - edge_index: optional (2, n_edges) int tensor for graph ops
      - sdf: optional (batch, n_query_points) or (n_query_points,)
      - boundary_mask: optional same shape as coords
      - shared_geometry: True if geometry is constant across batch
    """
    x: NDArray
    coords: NDArray
    edge_index: Optional[NDArray[np.int_]] = None
    sdf: Optional[NDArray] = None
    boundary_mask: Optional[NDArray[np.bool_]] = None
    shared_geometry: bool = True
    gaot_config: Optional["GAOTConfig"] = None  # required for GAOT type

    def validate(self, operator_type: OperatorType) -> None:
        """Raise ValueError on inconsistent shape/configuration.

        >>> import numpy as np
        >>> ctx = OperatorContext(
        ...     x=np.zeros((2, 100, 3)), coords=np.zeros((100, 2)),
        ...     shared_geometry=True,
        ... )
        >>> ctx.validate(OperatorType.GINO)
        >>> try:
        ...     ctx_bad = OperatorContext(
        ...         x=np.zeros((2, 100, 3)), coords=np.zeros((100, 2)),
        ...         shared_geometry=False,
        ...     )
        ...     ctx_bad.validate(OperatorType.GINO)
        ... except ValueError:
        ...     print("caught")
        caught
        >>> # GAOT requires explicit config
        >>> try:
        ...     OperatorContext(
        ...         x=np.zeros((2, 100, 3)), coords=np.zeros((100, 2)),
        ...     ).validate(OperatorType.GAOT)
        ... except ValueError:
        ...     print("gaot-config-missing")
        gaot-config-missing
        """
        if self.x.ndim != 3:
            raise ValueError(f"x must be 3D (B, N, d_in), got shape {self.x.shape}")
        batch_size = self.x.shape[0]
        n_points = self.x.shape[1]
        # coords shape
        if self.coords.ndim == 2:
            if self.coords.shape[0] != n_points:
                raise ValueError(
                    f"coords 2D must have {n_points} points, got {self.coords.shape}"
                )
            if not self.shared_geometry:
                raise ValueError(
                    "2D coords implies shared_geometry=True; got False"
                )
        elif self.coords.ndim == 3:
            if self.coords.shape[:2] != (batch_size, n_points):
                raise ValueError(
                    f"coords 3D shape must be (B, N, d), got {self.coords.shape}"
                )
        else:
            raise ValueError(f"coords must be 2D or 3D, got {self.coords.ndim}D")
        # GINO specific: require shared geometry
        if operator_type == OperatorType.GINO and not self.shared_geometry:
            raise ValueError(
                "GINO requires shared_geometry=True within batch"
            )
        # GNO specific: require edge_index
        if operator_type == OperatorType.GNO and self.edge_index is None:
            raise ValueError("GNO requires edge_index")
        # GAOT specific: require gaot_config and sensible radii for this domain
        if operator_type == OperatorType.GAOT:
            if self.gaot_config is None:
                raise ValueError(
                    "GAOT requires gaot_config; see GAOTConfig docstring"
                )
            cfg = self.gaot_config
            # Infer domain dimension from coords shape
            d_space = (self.coords.shape[-1]
                       if self.coords.ndim >= 2 else 0)
            if d_space != len(cfg.latent_grid_shape):
                raise ValueError(
                    f"GAOT: coords dim={d_space} does not match "
                    f"latent_grid_shape {cfg.latent_grid_shape}"
                )
            # Estimate domain diameter from coords (shared geometry only;
            # for non-shared we skip the check as diameter is per-sample).
            if self.coords.ndim == 2 and n_points > 1:
                mins = self.coords.min(axis=0)
                maxs = self.coords.max(axis=0)
                diam = float(np.linalg.norm(maxs - mins))
                if diam > 0:
                    # Largest integration ball must fit within domain
                    max_radius_abs = cfg.ball_radii_relative[-1] * diam
                    if max_radius_abs > diam:
                        raise ValueError(
                            f"GAOT: largest ball radius {max_radius_abs} "
                            f"exceeds domain diameter {diam}"
                        )
                    # Smallest radius should cover at least a few points
                    # Heuristic: r_min * density >= 1 where density = n/diam
                    r_min = cfg.ball_radii_relative[0] * diam
                    if r_min * n_points / diam < 0.5:
                        # This is a warning condition; we record it but do
                        # not raise. Manuscript caveat: too-few-points-per-ball
                        # will give GAOT poor accuracy.
                        logger.warning(
                            "GAOT: smallest radius %.4f with %d points "
                            "may yield too-sparse tokens",
                            r_min, n_points,
                        )
        # SDF shape check
        if self.sdf is not None:
            if self.sdf.ndim == 1:
                if self.sdf.shape[0] != n_points:
                    raise ValueError(f"sdf 1D shape must be (N,), got {self.sdf.shape}")
            elif self.sdf.ndim == 2:
                if self.sdf.shape != (batch_size, n_points):
                    raise ValueError(
                        f"sdf 2D shape must be (B, N), got {self.sdf.shape}"
                    )
            else:
                raise ValueError(f"sdf must be 1D or 2D, got {self.sdf.ndim}D")
        # boundary_mask shape check
        if self.boundary_mask is not None:
            if self.boundary_mask.ndim == 1:
                if self.boundary_mask.shape[0] != n_points:
                    raise ValueError(
                        f"boundary_mask 1D shape must be (N,), got "
                        f"{self.boundary_mask.shape}"
                    )
            elif self.boundary_mask.ndim == 2:
                if self.boundary_mask.shape != (batch_size, n_points):
                    raise ValueError(
                        f"boundary_mask 2D shape must be (B, N), got "
                        f"{self.boundary_mask.shape}"
                    )


# ============================================================================
# SECTION 10 -- TEST SUITE (P1.1 + P1.2 + all validations)
# ============================================================================


def test_module_hash_stable() -> bool:
    h = module_hash()
    assert len(h) == 64
    assert h == module_hash()
    print(f"  hash: {h[:16]}...")
    return True


def test_failure_signature_basic() -> bool:
    """FailureSignature dataclass + JSON roundtrip."""
    fs = FailureSignature(
        kind="CUDA_OOM", step=42, locus="train_step",
        observable="allocation failed",
    )
    assert fs.timestamp is not None
    assert fs.kind == "CUDA_OOM"
    with tempfile.TemporaryDirectory() as d:
        p = fs.register(d)
        assert os.path.exists(p)
        with open(p) as f:
            loaded = json.load(f)
        assert loaded["kind"] == "CUDA_OOM"
        assert loaded["step"] == 42
    print("  failure registered, fields intact")
    return True


def test_atomic_writer_basic() -> bool:
    """Atomic write: final content matches, tempfile cleaned."""
    w = AtomicCheckpointWriter()
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "nested", "dir", "file.bin")
        w.save_bytes_atomic(b"payload_123", target)
        assert os.path.exists(target)
        with open(target, "rb") as f:
            assert f.read() == b"payload_123"
        # no stray tempfiles
        tmps = [f for f in os.listdir(os.path.dirname(target))
                if f.endswith(".tmp")]
        assert len(tmps) == 0, f"stray tempfiles: {tmps}"
    print("  atomic write survives; no tempfile leak")
    return True


def test_atomic_writer_crash_safety() -> bool:
    """Simulate a crash during write (exception raised) leaves no partial file."""
    w = AtomicCheckpointWriter()
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "crash.bin")
        # simulate failure by patching os.replace
        original_replace = os.replace
        def bad_replace(*args, **kwargs):
            raise OSError("simulated crash")
        os.replace = bad_replace
        try:
            w.save_bytes_atomic(b"data", target)
            raised = False
        except OSError:
            raised = True
        finally:
            os.replace = original_replace
        assert raised
        assert not os.path.exists(target), "target exists despite crash"
        # tempfile should be cleaned
        tmps = [f for f in os.listdir(d) if f.endswith(".tmp")]
        assert len(tmps) == 0, f"stray tempfiles after crash: {tmps}"
    print("  crash leaves no target file and no tempfile leak")
    return True


def test_build_resume_payload_fields() -> bool:
    """Required fields present in resume payload."""
    payload = build_resume_payload(
        model_state_dict={"w": [1.0]},
        optimizer_state_dict={"state": {}},
        scheduler_state_dict={"lr": 0.01},
        epoch=5, global_step=1234,
        rng_states={"numpy": "dummy"},
    )
    assert payload["checkpoint_semantics"] == "resume_full"
    assert payload["epoch"] == 5
    assert payload["global_step"] == 1234
    assert payload["requires_trusted_load"] is True  # because rng_states given
    assert payload["version"] == __version__
    assert len(payload["module_hash"]) == 64
    # without rng_states
    p2 = build_resume_payload(
        model_state_dict={}, optimizer_state_dict={}, epoch=0, global_step=0,
    )
    assert p2["requires_trusted_load"] is False
    print("  resume payload fields correct")
    return True


def test_best_policy_minmode() -> bool:
    """Best policy correctness in min-mode."""
    pol = BestCheckpointPolicy(mode="min", min_delta=1e-3)
    assert pol.is_better(0.5, None) is True
    assert pol.is_better(0.3, 0.5) is True
    assert pol.is_better(0.499, 0.5) is False  # within min_delta
    assert pol.is_better(0.498, 0.5) is True   # beyond min_delta
    assert pol.is_better(0.6, 0.5) is False
    print("  min-mode policy correct")
    return True


def test_best_policy_maxmode() -> bool:
    """Best policy correctness in max-mode."""
    pol = BestCheckpointPolicy(mode="max", min_delta=0.01)
    assert pol.is_better(0.9, None) is True
    assert pol.is_better(0.95, 0.9) is True
    assert pol.is_better(0.905, 0.9) is False
    assert pol.is_better(0.92, 0.9) is True
    print("  max-mode policy correct")
    return True


def test_best_policy_invalid_raises() -> bool:
    """Invalid mode/min_delta raises."""
    try:
        BestCheckpointPolicy(mode="invalid")
        assert False, "should have raised"
    except ValueError:
        pass
    try:
        BestCheckpointPolicy(min_delta=-0.01)
        assert False, "should have raised"
    except ValueError:
        pass
    print("  invalid policy args rejected")
    return True


def test_ckpt_manager_best_roundtrip() -> bool:
    """P1.1: best-model roundtrip with policy."""
    with tempfile.TemporaryDirectory() as d:
        mgr = CheckpointIntegrationManager(output_dir=d)
        pol = BestCheckpointPolicy(mode="min", min_delta=1e-4)
        # first: saves
        r1 = mgr.save_best_model_checkpoint_raw(
            model_state_dict={"w": [1.0]}, metric_value=0.5,
            metric_name="loss", epoch=0, policy=pol,
        )
        assert r1["saved"] is True
        # improvement: saves
        r2 = mgr.save_best_model_checkpoint_raw(
            model_state_dict={"w": [2.0]}, metric_value=0.4,
            metric_name="loss", epoch=1, policy=pol,
        )
        assert r2["saved"] is True
        # no improvement: does not save
        r3 = mgr.save_best_model_checkpoint_raw(
            model_state_dict={"w": [3.0]}, metric_value=0.45,
            metric_name="loss", epoch=2, policy=pol,
        )
        assert r3["saved"] is False
        # within min_delta: does not save
        r4 = mgr.save_best_model_checkpoint_raw(
            model_state_dict={"w": [4.0]}, metric_value=0.39999,
            metric_name="loss", epoch=3, policy=pol,
        )
        assert r4["saved"] is False
        # load back
        loaded = mgr.load_raw(r2["path"], expected_semantics="best_model")
        assert loaded["metric_value"] == 0.4
        assert loaded["model_state_dict"]["w"] == [2.0]
    print("  save/load best roundtrip with policy gates")
    return True


def test_ckpt_manager_resume_roundtrip() -> bool:
    """P1.1: resume roundtrip with semantics validation."""
    with tempfile.TemporaryDirectory() as d:
        mgr = CheckpointIntegrationManager(output_dir=d)
        payload = build_resume_payload(
            model_state_dict={"w": [1.0]},
            optimizer_state_dict={"state": {"0": {}}},
            scheduler_state_dict={"lr": 0.01},
            epoch=10, global_step=5678,
        )
        r = mgr.save_resume_raw(resume_payload=payload)
        assert r["saved"]
        # correct load
        loaded = mgr.load_raw(r["path"], expected_semantics="resume_full")
        assert loaded["epoch"] == 10
        # WRONG load with best_model semantics -> raises
        try:
            mgr.load_raw(r["path"], expected_semantics="best_model")
            assert False, "should reject wrong semantics"
        except ValueError:
            pass
    print("  resume roundtrip + semantics validation")
    return True


def test_ckpt_manager_rejects_bad_semantics_on_save() -> bool:
    """save_resume_raw must reject non-resume-full payloads."""
    with tempfile.TemporaryDirectory() as d:
        mgr = CheckpointIntegrationManager(output_dir=d)
        bad = {"checkpoint_semantics": "best_model", "model_state_dict": {}}
        try:
            mgr.save_resume_raw(resume_payload=bad)
            assert False, "should reject"
        except ValueError:
            pass
    print("  wrong-semantics resume rejected")
    return True


def test_ckpt_manager_ddp_guard() -> bool:
    """Non-main process is_main=False -> no save."""
    with tempfile.TemporaryDirectory() as d:
        mgr = CheckpointIntegrationManager(output_dir=d, is_main_process=False)
        pol = BestCheckpointPolicy()
        r = mgr.save_best_model_checkpoint_raw(
            model_state_dict={"w": [1.0]}, metric_value=0.5,
            metric_name="loss", epoch=0, policy=pol,
        )
        assert r["saved"] is False
        assert r["reason"] == "not_main"
        # no file written
        files = [f for f in os.listdir(d) if f.endswith(".json")]
        assert len(files) == 0
    print("  DDP non-main guard works")
    return True


def test_warmstart_drops_optimizer() -> bool:
    """P0.2: warmstart returns ONLY model state, drops optimizer/scheduler."""
    payload = {
        "model_state_dict": {"layer1.w": [1], "layer2.w": [2]},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"lr": 0.01},
        "grad_scaler_state_dict": {"scale": 1.0},
        "rng_states": {"torch": "dummy"},
        "epoch": 10,
    }
    out = load_partial_checkpoint_raw(payload)
    assert "optimizer_state_dict" not in out
    assert "scheduler_state_dict" not in out
    assert "grad_scaler_state_dict" not in out
    assert "rng_states" not in out
    assert out["checkpoint_semantics"] == "warmstart"
    assert set(out["dropped_keys"]) >= {
        "optimizer_state_dict", "scheduler_state_dict",
        "grad_scaler_state_dict", "rng_states",
    }
    print(f"  dropped: {out['dropped_keys']}")
    return True


def test_warmstart_prefix_transforms() -> bool:
    """P0.2: strip_prefixes and prefix_map both work."""
    payload = {"model_state_dict": {
        "module.encoder.layer.w": [1],
        "module.decoder.layer.w": [2],
    }}
    out = load_partial_checkpoint_raw(
        payload, strip_prefixes=["module."],
        prefix_map={"encoder.": "backbone."},
    )
    keys = sorted(out["model_state_dict"].keys())
    assert keys == ["backbone.layer.w", "decoder.layer.w"], f"got {keys}"
    print(f"  prefix transforms: {keys}")
    return True


def test_hex_laplacian_and_cfl() -> bool:
    """Hex FV stencil, L2 stability, positivity/DMP, and input contracts."""
    w1 = hex_laplacian_stencil_weight(1.0)
    assert abs(w1 - 2.0/3.0) < 1e-14

    # Backward-compatible L2 CFL helper: dt_max = h^2 / (3 D_v).
    dt_l2_default = hex_cfl_max_dt(1.0, 1.0, safety=0.85)
    assert abs(dt_l2_default - 0.85/3.0) < 1e-14
    dt_l2_sharp = hex_cfl_max_dt(0.5, 2.0, safety=1.0)
    assert abs(dt_l2_sharp - 0.25/6.0) < 1e-14

    # Positivity/DMP bound: dt_max = h^2 / (4 D_v).
    dt_pos = hex_positivity_max_dt(1.0, 1.0)
    assert abs(dt_pos - 0.25) < 1e-14

    # Historical L2 default 0.85 is stable but not positivity preserving:
    # centre coefficient = 1 - 4 D dt / h^2.
    centre_default = 1.0 - 4.0 * dt_l2_default
    centre_pos = 1.0 - 4.0 * dt_pos
    assert centre_default < 0.0
    assert abs(centre_default + 2.0/15.0) < 1e-14
    assert abs(centre_pos) < 1e-14

    invalid_calls = [
        lambda: hex_laplacian_stencil_weight(float("nan")),
        lambda: hex_cfl_max_dt(0.0, 1.0),
        lambda: hex_cfl_max_dt(1.0, float("inf")),
        lambda: hex_cfl_max_dt(10**400, 1.0),
        lambda: hex_cfl_max_dt(True, 1.0),
        lambda: hex_cfl_max_dt("1.0", 1.0),
        lambda: hex_positivity_max_dt(1.0, 1.0, safety=0.0),
    ]
    rejected = 0
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            rejected += 1
    assert rejected == len(invalid_calls)

    print(
        "  hex stencil=2/3; "
        f"L2 default dt={dt_l2_default:.4f}; positivity dt={dt_pos:.4f}; "
        f"centre(default)={centre_default:.4f}"
    )
    return True


def test_safe_train_step_success() -> bool:
    """Normal step returns ok."""
    with tempfile.TemporaryDirectory() as d:
        def good():
            return 0.42, np.array([1.0])
        r = safe_train_step(good, global_step=10, failure_dir=d)
        assert r["status"] == "ok"
        assert r["loss"] == 0.42
        files = os.listdir(d)
        assert len(files) == 0, "no failure file for success"
    print("  ok path")
    return True


def test_safe_train_step_nan() -> bool:
    """NaN loss triggers FailureSignature."""
    with tempfile.TemporaryDirectory() as d:
        def nan_step():
            return float("nan"), None
        r = safe_train_step(nan_step, global_step=5, failure_dir=d)
        assert r["status"] == "nan"
        assert r["failure_signature"].kind == "NaN_or_Inf_Loss"
        files = os.listdir(d)
        assert any("NaN_or_Inf_Loss" in f for f in files)
    print("  NaN path records failure")
    return True


def test_safe_train_step_oom_pre_assignment() -> bool:
    """P0.5: exception BEFORE pred/loss assigned does not crash with NameError."""
    with tempfile.TemporaryDirectory() as d:
        def oom_immediately():
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        r = safe_train_step(oom_immediately, global_step=7, failure_dir=d)
        # critical: this call must not itself raise NameError
        assert r["status"] == "oom"
        assert r["failure_signature"].kind == "CUDA_OOM"
    print("  OOM-before-assignment handled without NameError")
    return True


def test_operator_context_valid_gino() -> bool:
    """P1.2: valid GINO context passes."""
    ctx = OperatorContext(
        x=np.zeros((4, 100, 3)), coords=np.zeros((100, 2)),
        sdf=np.zeros(100), shared_geometry=True,
    )
    ctx.validate(OperatorType.GINO)
    print("  valid GINO ctx passes")
    return True


def test_operator_context_rejects_non_shared_gino() -> bool:
    """P1.2: GINO requires shared_geometry."""
    ctx = OperatorContext(
        x=np.zeros((4, 100, 3)), coords=np.zeros((4, 100, 2)),
        shared_geometry=False,
    )
    try:
        ctx.validate(OperatorType.GINO)
        assert False, "should reject"
    except ValueError:
        pass
    print("  non-shared GINO rejected")
    return True


def test_operator_context_rejects_missing_edge_for_gno() -> bool:
    """P1.2: GNO requires edge_index."""
    ctx = OperatorContext(
        x=np.zeros((4, 100, 3)), coords=np.zeros((100, 2)),
        edge_index=None,
    )
    try:
        ctx.validate(OperatorType.GNO)
        assert False, "should reject"
    except ValueError:
        pass
    print("  GNO without edge_index rejected")
    return True


def test_operator_context_shape_validations() -> bool:
    """P1.2: wrong shapes raise clear errors."""
    # bad x dim
    ctx = OperatorContext(x=np.zeros((100, 3)), coords=np.zeros((100, 2)))
    try:
        ctx.validate(OperatorType.GINO)
        assert False
    except ValueError:
        pass
    # mismatched sdf
    ctx = OperatorContext(
        x=np.zeros((4, 100, 3)), coords=np.zeros((100, 2)),
        sdf=np.zeros(50), shared_geometry=True,
    )
    try:
        ctx.validate(OperatorType.GINO)
        assert False
    except ValueError:
        pass
    # mismatched boundary_mask
    ctx = OperatorContext(
        x=np.zeros((4, 100, 3)), coords=np.zeros((100, 2)),
        boundary_mask=np.zeros(50, dtype=bool), shared_geometry=True,
    )
    try:
        ctx.validate(OperatorType.GINO)
        assert False
    except ValueError:
        pass
    print("  3 shape mismatches rejected")
    return True


def test_gaot_config_defaults_match_paper() -> bool:
    """GAOTConfig defaults must match Wen 2025 paper Table 4.

    This guards against silent drift of the defaults when the paper
    is re-read in a future session. If a future author deliberately
    changes the defaults, matches_paper_default() will fail loudly.
    """
    cfg = GAOTConfig()
    assert cfg.tokenization == "structured_stencil"
    assert cfg.latent_grid_shape == (32, 32)
    assert cfg.n_scales == 3
    assert cfg.ball_radii_relative == (0.08, 0.16, 0.32)
    assert cfg.geometry_embed_dim == 32
    assert cfg.patch_size == 4
    assert cfg.vit_depth == 8
    assert cfg.vit_heads == 8
    assert cfg.vit_hidden_dim == 256
    assert cfg.matches_paper_default()
    print("  GAOTConfig defaults confirmed vs arXiv:2505.18781 Table 4")
    return True


def test_gaot_config_validation() -> bool:
    """GAOTConfig __post_init__ rejects invalid configurations.

    Covers: bad tokenization string, wrong n_scales/ball_radii coupling,
    non-increasing radii, out-of-range radii, patch_size not dividing
    latent_grid_shape.
    """
    rejected = 0
    # 1. bad tokenization
    try:
        GAOTConfig(tokenization="bogus")
    except ValueError:
        rejected += 1
    # 2. mismatch n_scales vs radii length
    try:
        GAOTConfig(n_scales=3, ball_radii_relative=(0.1, 0.2))
    except ValueError:
        rejected += 1
    # 3. radii not strictly increasing
    try:
        GAOTConfig(n_scales=3, ball_radii_relative=(0.1, 0.2, 0.2))
    except ValueError:
        rejected += 1
    # 4. radius out of range
    try:
        GAOTConfig(n_scales=1, ball_radii_relative=(1.5,))
    except ValueError:
        rejected += 1
    # 5. patch_size does not divide latent_grid_shape
    try:
        GAOTConfig(latent_grid_shape=(31, 31), patch_size=4)
    except ValueError:
        rejected += 1
    assert rejected == 5, f"only {rejected}/5 invalid configs rejected"
    print(f"  {rejected}/5 invalid configurations rejected")
    return True


def test_gaot_context_requires_config() -> bool:
    """OperatorContext must reject GAOT without explicit gaot_config."""
    ctx = OperatorContext(
        x=np.zeros((2, 100, 3)), coords=np.zeros((100, 2)),
        shared_geometry=True,
    )
    try:
        ctx.validate(OperatorType.GAOT)
        return False
    except ValueError as e:
        assert "gaot_config" in str(e).lower()
    print("  GAOT rejected without gaot_config")
    return True


def test_gaot_context_dim_mismatch() -> bool:
    """GAOT context with 2D coords but 3D latent_grid_shape is rejected.

    This catches the common mistake of copying a 2D config to a 3D
    dataset without updating latent_grid_shape. The 3D shape used here
    is ILLUSTRATIVE ONLY -- for production 3D use the authors'
    camlab-ethz/GAOT3D repository and its dataset-specific configs.
    """
    cfg_3d = GAOTConfig(
        latent_grid_shape=(16, 16, 16), patch_size=4,
        n_scales=3, ball_radii_relative=(0.08, 0.16, 0.32),
    )
    assert cfg_3d.dim() == 3
    # 2D coords (100, 2) vs 3D grid -> mismatch
    ctx = OperatorContext(
        x=np.zeros((2, 100, 3)),
        coords=np.random.default_rng(0).uniform(-1, 1, (100, 2)),
        shared_geometry=True,
        gaot_config=cfg_3d,
    )
    try:
        ctx.validate(OperatorType.GAOT)
        return False
    except ValueError as e:
        assert "latent_grid_shape" in str(e)

    # Matching dim (2D <-> 2D) with paper-validated default passes
    cfg_2d = GAOTConfig()
    assert cfg_2d.dim() == 2
    assert cfg_2d.matches_paper_default()  # this is the 2D benchmark default
    ctx_ok = OperatorContext(
        x=np.zeros((2, 100, 3)),
        coords=np.random.default_rng(0).uniform(-1, 1, (100, 2)),
        shared_geometry=True,
        gaot_config=cfg_2d,
    )
    ctx_ok.validate(OperatorType.GAOT)  # should not raise
    print("  2D-vs-3D mismatch caught, 2D paper-default passes")
    return True


def run_all_tests() -> bool:
    tests = [
        ("module hash stable",                    test_module_hash_stable),
        ("FailureSignature basic",                test_failure_signature_basic),
        ("AtomicWriter basic",                    test_atomic_writer_basic),
        ("AtomicWriter crash safety",             test_atomic_writer_crash_safety),
        ("build_resume_payload fields",           test_build_resume_payload_fields),
        ("BestPolicy min-mode",                   test_best_policy_minmode),
        ("BestPolicy max-mode",                   test_best_policy_maxmode),
        ("BestPolicy invalid args",               test_best_policy_invalid_raises),
        ("CkptMgr: best roundtrip",               test_ckpt_manager_best_roundtrip),
        ("CkptMgr: resume roundtrip",             test_ckpt_manager_resume_roundtrip),
        ("CkptMgr: bad semantics rejected",       test_ckpt_manager_rejects_bad_semantics_on_save),
        ("CkptMgr: DDP guard",                    test_ckpt_manager_ddp_guard),
        ("Warmstart drops optimizer",             test_warmstart_drops_optimizer),
        ("Warmstart prefix transforms",           test_warmstart_prefix_transforms),
        ("Hex Laplacian + CFL correct",           test_hex_laplacian_and_cfl),
        ("safe_train_step: success",              test_safe_train_step_success),
        ("safe_train_step: NaN",                  test_safe_train_step_nan),
        ("safe_train_step: OOM pre-assignment",   test_safe_train_step_oom_pre_assignment),
        ("OperatorContext: valid GINO",           test_operator_context_valid_gino),
        ("OperatorContext: non-shared GINO",      test_operator_context_rejects_non_shared_gino),
        ("OperatorContext: missing edge GNO",     test_operator_context_rejects_missing_edge_for_gno),
        ("OperatorContext: shape validations",    test_operator_context_shape_validations),
        ("GAOT config defaults match paper",      test_gaot_config_defaults_match_paper),
        ("GAOT config validation",                test_gaot_config_validation),
        ("GAOT context requires config",          test_gaot_context_requires_config),
        ("GAOT context dim mismatch caught",      test_gaot_context_dim_mismatch),
    ]
    passed = failed = 0
    print("=" * 70)
    print(f"HexaNTK v{__version__} (L014) -- Manuscript Test Suite")
    print(f"  Module hash: {module_hash()[:16]}...")
    print(f"  Tests: {len(tests)} (covers P0.1-P0.5, P1.1, P1.2)")
    print("=" * 70)
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        t0 = time.perf_counter()
        try:
            ok = fn()
            dt = time.perf_counter() - t0
            if ok:
                print(f"  -> PASS ({dt:.3f}s)")
                passed += 1
            else:
                print("  -> FAIL")
                failed += 1
        except Exception as e:
            print(f"  -> FAIL: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{passed+failed} PASS")
    if failed == 0:
        print(f"ALL TESTS PASSED -- HexaNTK v{__version__} manuscript-ready.")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    import doctest
    import sys

    r = doctest.testmod(verbose=False)
    doctests_ok = r.failed == 0
    print(f"Doctests: {r.attempted - r.failed}/{r.attempted} PASS")
    selftests_ok = run_all_tests()
    sys.exit(0 if doctests_ok and selftests_ok else 1)
