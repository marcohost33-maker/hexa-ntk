"""
warmstart_utils_hex_v3.py
=========================
Safe partial checkpoint loading for model weight transfer (warmstart).

Design invariants:
- NEVER loads optimizer state
- NEVER loads scheduler state
- NEVER restores RNG states
- NEVER interprets warmstart as resume
- Always uses weights_only=True (safe path)
- Falls back to safe_globals whitelisting for legacy checkpoints
- Supports prefix stripping (DDP "module." prefix)
- Supports prefix mapping (architecture migration)
- Supports dtype mismatch tolerance

PyTorch 2.6+ compatibility:
- weights_only=True is the default since PyTorch 2.6
- Legacy checkpoints with numpy scalars require safe_globals whitelisting
"""
from __future__ import annotations

__all__ = [
    "load_partial_checkpoint",
    "diagnose_checkpoint_safety",
]

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _load_payload_safe(
    path: Path,
    map_location: Union[str, torch.device],
) -> Dict[str, Any]:
    """
    Load checkpoint payload via the safe path (weights_only=True).

    Falls back to safe_globals whitelisting for legacy checkpoints
    that contain numpy scalars (common in older PyTorch checkpoints).

    Raises
    ------
    Exception
        If the checkpoint cannot be loaded even with whitelisting.
    """
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        error_msg = str(e)
        if "Weights only load failed" not in error_msg:
            raise

        logger.warning(
            "Warmstart: weights_only=True failed for %s. "
            "Attempting safe_globals whitelisting for numpy types...",
            path,
        )

        try:
            import numpy as np
            from torch.serialization import safe_globals

            # Whitelist numpy types commonly found in legacy checkpoints.
            # NumPy 2.0 reorganized internals: np.core -> np._core
            allowed_globals = []
            for attr_path in (
                "core.multiarray.scalar",
                "_core.multiarray.scalar",
            ):
                parts = attr_path.split(".")
                obj = np
                try:
                    for p in parts:
                        obj = getattr(obj, p)
                    allowed_globals.append(obj)
                except AttributeError:
                    pass

            try:
                allowed_globals.append(np.dtype)
                allowed_globals.append(type(np.dtype(np.float64)))
            except Exception:
                pass

            with safe_globals(allowed_globals):
                payload = torch.load(
                    path, map_location=map_location, weights_only=True
                )

        except ImportError:
            logger.error(
                "safe_globals not available (PyTorch < 2.4). "
                "Cannot safely load legacy checkpoint."
            )
            raise e
        except Exception:
            logger.error(
                "safe_globals whitelisting also failed. "
                "Checkpoint may contain unsupported Python objects."
            )
            raise e

    return payload


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    """
    Extract model state_dict from various checkpoint formats.

    Supports:
    - Direct state_dict (flat dict of tensors)
    - Wrapped format with "model_state_dict" key
    - Wrapped format with "state_dict" key (common convention)
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"Checkpoint payload must be a dict, got {type(payload).__name__}"
        )

    # Try common keys in order of preference
    for key in ("model_state_dict", "state_dict"):
        if key in payload:
            candidate = payload[key]
            if isinstance(candidate, dict):
                return candidate

    # Assume the payload itself is a state_dict if it contains tensors
    has_tensors = any(torch.is_tensor(v) for v in payload.values())
    if has_tensors:
        return payload

    raise ValueError(
        "Cannot extract state_dict from checkpoint. "
        "Expected 'model_state_dict', 'state_dict', or a flat dict of tensors."
    )


def _apply_prefix_transforms(
    state_dict: Dict[str, torch.Tensor],
    strip_prefixes: Optional[List[str]],
    prefix_map: Optional[Dict[str, str]],
) -> Dict[str, torch.Tensor]:
    """Apply prefix stripping and mapping to state_dict keys."""
    result = dict(state_dict)

    # Strip prefixes (e.g. "module." from DDP checkpoints)
    if strip_prefixes:
        for prefix in strip_prefixes:
            result = {
                k[len(prefix) :] if k.startswith(prefix) else k: v
                for k, v in result.items()
            }

    # Map prefixes (e.g. "encoder." -> "backbone.")
    if prefix_map:
        for old_prefix, new_prefix in prefix_map.items():
            result = {
                new_prefix + k[len(old_prefix) :]
                if k.startswith(old_prefix)
                else k: v
                for k, v in result.items()
            }

    return result


def _handle_dtype_mismatches(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """Cast checkpoint tensors to match model dtype where they differ."""
    model_sd = model.state_dict()
    result = dict(state_dict)

    for key in result:
        if key in model_sd:
            if result[key].dtype != model_sd[key].dtype:
                logger.info(
                    "Warmstart dtype cast: %s %s -> %s",
                    key,
                    result[key].dtype,
                    model_sd[key].dtype,
                )
                result[key] = result[key].to(model_sd[key].dtype)

    return result


def load_partial_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    *,
    map_location: Union[str, torch.device] = "cpu",
    strip_prefixes: Optional[List[str]] = None,
    prefix_map: Optional[Dict[str, str]] = None,
    allow_dtype_mismatch: bool = True,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load model weights from a checkpoint (warmstart).

    This function:
    - Loads ONLY model weights
    - NEVER loads optimizer, scheduler, or RNG states
    - Uses the safe loading path (weights_only=True)
    - Supports partial loading (strict=False by default)

    Parameters
    ----------
    model :
        Target model to load weights into.
    checkpoint_path :
        Path to the checkpoint file.
    map_location :
        Device mapping for tensor loading.
    strip_prefixes :
        List of prefixes to strip from checkpoint keys.
        Common use: ["module."] for DDP checkpoints.
    prefix_map :
        Dict mapping old prefixes to new prefixes.
        Common use: {"encoder.": "backbone."} for architecture migration.
    allow_dtype_mismatch :
        If True, cast checkpoint tensors to match model dtypes.
    strict :
        If True, require exact key match between checkpoint and model.
        Default False for warmstart (partial loading expected).

    Returns
    -------
    Dict[str, Any]
        Report containing:
        - checkpoint_path: str
        - action: "warmstart"
        - checkpoint_semantics: "warmstart"
        - load_path_type: "safe"
        - strict_mode_used: bool
        - missing_keys: list of keys in model but not in checkpoint
        - unexpected_keys: list of keys in checkpoint but not in model
        - dtype_casts: number of dtype conversions performed
        - warning: reminder that optimizer/scheduler were intentionally dropped
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Warmstart checkpoint not found: {checkpoint_path}")

    # Load via safe path
    payload = _load_payload_safe(checkpoint_path, map_location)

    # Extract state_dict
    state_dict = _extract_state_dict(payload)

    # Apply prefix transforms
    state_dict = _apply_prefix_transforms(state_dict, strip_prefixes, prefix_map)

    # Handle dtype mismatches
    dtype_casts = 0
    if allow_dtype_mismatch:
        model_sd = model.state_dict()
        for key in state_dict:
            if key in model_sd and state_dict[key].dtype != model_sd[key].dtype:
                dtype_casts += 1
        state_dict = _handle_dtype_mismatches(state_dict, model)

    # Load into model
    incompat = model.load_state_dict(state_dict, strict=strict)

    return {
        "checkpoint_path": str(checkpoint_path),
        "action": "warmstart",
        "checkpoint_semantics": "warmstart",
        "load_path_type": "safe",
        "strict_mode_used": strict,
        "missing_keys": list(incompat.missing_keys),
        "unexpected_keys": list(incompat.unexpected_keys),
        "dtype_casts": dtype_casts,
        "warning": (
            "Optimizer and scheduler states were INTENTIONALLY dropped for warmstart."
        ),
    }


def diagnose_checkpoint_safety(
    checkpoint_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Diagnose whether a checkpoint can be loaded with weights_only=True.

    Uses torch.serialization.get_unsafe_globals_in_checkpoint (PyTorch 2.6+)
    to statically analyze the checkpoint without loading it.

    Returns
    -------
    Dict with:
        path: str
        safe: bool (True if weights_only=True should work)
        unsafe_globals: list of str (names of unsafe globals found)
        error: optional str if diagnostic itself failed
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return {
            "path": str(checkpoint_path),
            "safe": False,
            "unsafe_globals": [],
            "error": f"File not found: {checkpoint_path}",
        }

    try:
        from torch.serialization import get_unsafe_globals_in_checkpoint

        unsafe = get_unsafe_globals_in_checkpoint(str(checkpoint_path))
        return {
            "path": str(checkpoint_path),
            "safe": len(unsafe) == 0,
            "unsafe_globals": list(unsafe),
            "error": None,
        }
    except ImportError:
        return {
            "path": str(checkpoint_path),
            "safe": None,  # Unknown, API not available
            "unsafe_globals": [],
            "error": "get_unsafe_globals_in_checkpoint not available (PyTorch < 2.6)",
        }
    except Exception as exc:
        return {
            "path": str(checkpoint_path),
            "safe": None,
            "unsafe_globals": [],
            "error": str(exc),
        }
