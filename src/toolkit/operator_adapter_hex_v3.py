"""
operator_adapter_hex_v3.py
==========================
Explicit dispatch/validation layer for operator models.

Design goals:
- No implicit None-magic
- Clear required fields per model family
- One canonical call path from runner/evaluator to model.forward(...)
- GINO shared_geometry constraint enforced

Supported model types:
- "graph_operator_v3"
- "gino_hex_v3"
- "gino_hex_v3_v2"
"""
from __future__ import annotations

__all__ = [
    "OperatorBatchContext",
    "OperatorAdapterV3",
]

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Context
# ------------------------------------------------------------------


@dataclass
class OperatorBatchContext:
    """
    Canonical batch context passed from runner/evaluator to the adapter.

    Notes
    -----
    - `coords` is the canonical geometric field for geometry-aware models.
    - `edge_index` is required for graph-style operators.
    - `dt` is optional globally, but may be required by some model types.
    - `extra` stores optional structured fields such as:
        - laplace_eigvecs
        - rel_error_threshold
        - sdf
        - boundary_mask
        - shared_geometry
    """

    coords: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    dt: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------


class OperatorAdapterV3:
    """
    Explicit dispatch/validation layer for operator models.

    Supported model types:
    - "graph_operator_v3"
    - "gino_hex_v3"
    - "gino_hex_v3_v2"
    """

    SUPPORTED_MODEL_TYPES = {
        "graph_operator_v3",
        "gino_hex_v3",
        "gino_hex_v3_v2",
    }

    def __init__(self, model: nn.Module, model_type: str) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")

        if model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Supported: {sorted(self.SUPPORTED_MODEL_TYPES)}"
            )

        self.model = model
        self.model_type = model_type

    # ------------------------------------------------------------------
    # Public call
    # ------------------------------------------------------------------

    def __call__(self, x: torch.Tensor, ctx: OperatorBatchContext) -> torch.Tensor:
        self.validate_context(x, ctx)

        if self.model_type == "graph_operator_v3":
            return self._call_graph_operator(x, ctx)

        if self.model_type in {"gino_hex_v3", "gino_hex_v3_v2"}:
            return self._call_gino(x, ctx)

        raise RuntimeError(f"Unreachable model_type branch: {self.model_type}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_context(self, x: torch.Tensor, ctx: OperatorBatchContext) -> None:
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")

        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B, N, C], got shape {tuple(x.shape)}"
            )

        if not isinstance(ctx, OperatorBatchContext):
            raise TypeError("ctx must be an OperatorBatchContext")

        if self.model_type == "graph_operator_v3":
            self._validate_graph_context(x, ctx)
            return

        if self.model_type in {"gino_hex_v3", "gino_hex_v3_v2"}:
            self._validate_gino_context(x, ctx)
            return

        raise RuntimeError(f"Unreachable model_type branch: {self.model_type}")

    def _validate_graph_context(
        self, x: torch.Tensor, ctx: OperatorBatchContext
    ) -> None:
        if ctx.edge_index is None:
            raise ValueError("graph_operator_v3 requires ctx.edge_index")

        if not torch.is_tensor(ctx.edge_index):
            raise TypeError("ctx.edge_index must be a torch.Tensor")

        if ctx.edge_index.ndim != 2 or ctx.edge_index.shape[0] != 2:
            raise ValueError(
                f"ctx.edge_index must have shape [2, E], "
                f"got {tuple(ctx.edge_index.shape)}"
            )

        if ctx.edge_index.dtype.is_floating_point:
            raise TypeError(
                f"ctx.edge_index must be integer dtype (torch.long), "
                f"got {ctx.edge_index.dtype}"
            )

        if ctx.coords is not None:
            if not torch.is_tensor(ctx.coords):
                raise TypeError("ctx.coords must be a torch.Tensor when provided")
            if ctx.coords.ndim != 2:
                raise ValueError(
                    f"ctx.coords must have shape [N, D], "
                    f"got {tuple(ctx.coords.shape)}"
                )
            if ctx.coords.shape[0] != x.shape[1]:
                raise ValueError(
                    "ctx.coords first dim must match x.shape[1] (number of nodes)"
                )

        if ctx.dt is not None:
            if not torch.is_tensor(ctx.dt):
                raise TypeError("ctx.dt must be a torch.Tensor when provided")
            if ctx.dt.numel() not in {1, x.shape[0]}:
                raise ValueError(
                    "ctx.dt must be scalar-like or batch-aligned with x.shape[0]"
                )

    def _validate_gino_context(
        self, x: torch.Tensor, ctx: OperatorBatchContext
    ) -> None:
        if ctx.coords is None:
            raise ValueError(f"{self.model_type} requires ctx.coords")

        if not torch.is_tensor(ctx.coords):
            raise TypeError("ctx.coords must be a torch.Tensor")

        if ctx.coords.ndim != 2:
            raise ValueError(
                f"ctx.coords must have shape [N, D], "
                f"got {tuple(ctx.coords.shape)}"
            )

        if ctx.coords.shape[0] != x.shape[1]:
            raise ValueError(
                "ctx.coords first dim must match x.shape[1] (number of nodes)"
            )

        if ctx.coords.shape[1] < 2:
            raise ValueError(
                f"{self.model_type} expects at least 2D coordinates, "
                f"got D={ctx.coords.shape[1]}"
            )

        if ctx.dt is not None:
            if not torch.is_tensor(ctx.dt):
                raise TypeError("ctx.dt must be a torch.Tensor when provided")
            if ctx.dt.numel() not in {1, x.shape[0]}:
                raise ValueError(
                    "ctx.dt must be scalar-like or batch-aligned with x.shape[0]"
                )

        # GINO batch constraint
        shared_geometry = ctx.extra.get("shared_geometry", True)
        if not isinstance(shared_geometry, bool):
            raise TypeError("ctx.extra['shared_geometry'] must be bool")
        if not shared_geometry:
            raise ValueError(
                f"{self.model_type} currently requires shared_geometry=True"
            )

        # Optional SDF validation
        if "sdf" in ctx.extra and ctx.extra["sdf"] is not None:
            sdf = ctx.extra["sdf"]
            if not torch.is_tensor(sdf):
                raise TypeError("ctx.extra['sdf'] must be a torch.Tensor")
            if sdf.ndim != 2:
                raise ValueError(
                    f"ctx.extra['sdf'] must have shape [N, 1]-like, "
                    f"got {tuple(sdf.shape)}"
                )
            if sdf.shape[0] != x.shape[1]:
                raise ValueError(
                    "ctx.extra['sdf'] first dim must match x.shape[1]"
                )

        # Optional boundary_mask validation
        if "boundary_mask" in ctx.extra and ctx.extra["boundary_mask"] is not None:
            bm = ctx.extra["boundary_mask"]
            if not torch.is_tensor(bm):
                raise TypeError(
                    "ctx.extra['boundary_mask'] must be a torch.Tensor"
                )
            if bm.ndim != 2:
                raise ValueError(
                    "ctx.extra['boundary_mask'] must have shape [N, 1]-like"
                )
            if bm.shape[0] != x.shape[1]:
                raise ValueError(
                    "ctx.extra['boundary_mask'] first dim must match x.shape[1]"
                )

    # ------------------------------------------------------------------
    # Dispatch implementations
    # ------------------------------------------------------------------

    def _call_graph_operator(
        self, x: torch.Tensor, ctx: OperatorBatchContext
    ) -> torch.Tensor:
        kwargs: Dict[str, Any] = {"edge_index": ctx.edge_index}

        if ctx.coords is not None:
            kwargs["coords"] = ctx.coords
        if ctx.dt is not None:
            kwargs["dt"] = ctx.dt

        for key in ("sdf", "boundary_mask"):
            if key in ctx.extra:
                kwargs[key] = ctx.extra[key]

        out = self.model(x, **kwargs)
        self._validate_output(x, out)
        return out

    def _call_gino(
        self, x: torch.Tensor, ctx: OperatorBatchContext
    ) -> torch.Tensor:
        kwargs: Dict[str, Any] = {"coords_cart": ctx.coords}

        if ctx.dt is not None:
            kwargs["dt"] = ctx.dt
        if "sdf" in ctx.extra:
            kwargs["sdf"] = ctx.extra["sdf"]
        if "boundary_mask" in ctx.extra:
            kwargs["boundary_mask"] = ctx.extra["boundary_mask"]

        out = self.model(x, **kwargs)
        self._validate_output(x, out)
        return out

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    def _validate_output(self, x: torch.Tensor, out: Any) -> None:
        if not torch.is_tensor(out):
            raise TypeError(
                f"Model output must be a torch.Tensor, got {type(out)}"
            )

        if out.shape != x.shape:
            raise ValueError(
                f"Model output shape must match input shape [B, N, C]. "
                f"Got input {tuple(x.shape)} vs output {tuple(out.shape)}"
            )
