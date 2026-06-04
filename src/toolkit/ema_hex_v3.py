"""
ema_hex_v3.py
=============
Exponential Moving Average (EMA) wrapper for HexaNTK models.

Uses PyTorch's native torch.optim.swa_utils.AveragedModel with
get_ema_multi_avg_fn for efficient fused EMA updates.

Key design decisions:
- Uses PyTorch-native API (not a custom implementation)
- use_buffers=True: EMA includes BatchNorm statistics
- avg_fn/multi_avg_fn are NOT saved in state_dict, must re-init on resume
- Checkpoint integration: save/load via ema_model.state_dict()

References:
- torch.optim.swa_utils.AveragedModel (PyTorch >= 1.10)
- get_ema_multi_avg_fn (PyTorch >= 2.0, fused vectorized update)
- timm ModelEmaV2 (inspiration for design patterns)
- lucidrains/ema-pytorch (warmup, update_every patterns)
"""
from __future__ import annotations

__all__ = [
    "create_ema_model",
    "save_ema_state",
    "load_ema_state",
]

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def create_ema_model(
    model: nn.Module,
    decay: float = 0.9999,
    device: Optional[Union[str, torch.device]] = None,
    use_buffers: bool = True,
) -> torch.optim.swa_utils.AveragedModel:
    """
    Create an EMA model using PyTorch's native AveragedModel.

    Parameters
    ----------
    model :
        The training model to track.
    decay :
        EMA decay factor (0.9999 typical for PDE operators).
        Higher = smoother, slower adaptation.
    device :
        Optional device for EMA model (e.g. "cpu" to save GPU memory).
    use_buffers :
        If True, EMA also tracks buffers (BatchNorm running stats).
        Recommended True for models with normalization layers.

    Returns
    -------
    torch.optim.swa_utils.AveragedModel
        The EMA-wrapped model. Call ema_model.update_parameters(model)
        after each optimizer step.

    Notes
    -----
    - avg_fn / multi_avg_fn are NOT saved in state_dict().
      On resume, you must re-create the EMA model with the same decay,
      then load the state_dict.
    - For BatchNorm, either set use_buffers=True (recommended) or call
      torch.optim.swa_utils.update_bn(loader, ema_model) after training.
    """
    try:
        # PyTorch >= 2.0: use fused multi-tensor EMA (faster)
        multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay)
        ema_model = torch.optim.swa_utils.AveragedModel(
            model,
            multi_avg_fn=multi_avg_fn,
            device=device,
            use_buffers=use_buffers,
        )
    except AttributeError:
        # PyTorch < 2.0: fallback to scalar avg_fn
        logger.info(
            "get_ema_multi_avg_fn not available (PyTorch < 2.0). "
            "Falling back to scalar EMA."
        )

        def ema_avg_fn(
            avg_model_param: torch.Tensor,
            model_param: torch.Tensor,
            num_averaged: int,
        ) -> torch.Tensor:
            return decay * avg_model_param + (1.0 - decay) * model_param

        ema_model = torch.optim.swa_utils.AveragedModel(
            model,
            avg_fn=ema_avg_fn,
            device=device,
            use_buffers=use_buffers,
        )

    return ema_model


def save_ema_state(
    ema_model: torch.optim.swa_utils.AveragedModel,
) -> Dict[str, Any]:
    """
    Extract EMA state for checkpoint integration.

    Returns a dict suitable for inclusion in checkpoint payloads:
    - ema_model_state_dict: the EMA model weights
    - n_averaged: number of EMA updates performed

    Note: avg_fn is NOT included (not serializable).
    Must be re-initialized with the same decay on resume.
    """
    state_dict = ema_model.state_dict()
    return {
        "ema_model_state_dict": {
            k: v.clone() for k, v in ema_model.module.state_dict().items()
        },
        "n_averaged": int(ema_model.n_averaged.item()),
        "ema_full_state_dict": state_dict,
    }


def load_ema_state(
    ema_model: torch.optim.swa_utils.AveragedModel,
    ema_state: Dict[str, Any],
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load EMA state from a checkpoint payload.

    Parameters
    ----------
    ema_model :
        An AveragedModel created with the same decay parameter.
    ema_state :
        Dict from save_ema_state() or a checkpoint with ema_full_state_dict.

    Returns
    -------
    Dict with load results.
    """
    if "ema_full_state_dict" in ema_state:
        ema_model.load_state_dict(ema_state["ema_full_state_dict"])
        return {
            "loaded_via": "ema_full_state_dict",
            "n_averaged": int(ema_model.n_averaged.item()),
        }

    if "ema_model_state_dict" in ema_state:
        incompat = ema_model.module.load_state_dict(
            ema_state["ema_model_state_dict"], strict=strict
        )
        if "n_averaged" in ema_state:
            ema_model.n_averaged.fill_(ema_state["n_averaged"])
        return {
            "loaded_via": "ema_model_state_dict",
            "n_averaged": int(ema_model.n_averaged.item()),
            "missing_keys": list(incompat.missing_keys),
            "unexpected_keys": list(incompat.unexpected_keys),
        }

    raise ValueError(
        "ema_state must contain 'ema_full_state_dict' or 'ema_model_state_dict'"
    )
