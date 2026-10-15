"""Phase-1 adapter for the pinned external NeuralOperator GINO backend.

This module does not implement GINO. It binds HexaNTK's verified lattice/data
contracts to neuraloperator==2.0.0 while keeping the optional dependency lazy.
No training, tuning, or test-set access occurs here.
"""
from __future__ import annotations

__all__ = [
    "GINO_BACKEND_PIN",
    "GinoPhase1Config",
    "build_gino",
    "prepare_gino_batch",
    "predict_gino",
]

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from hex_fv_solver_hex_v3 import HexLattice

GINO_BACKEND_PIN = "neuraloperator==2.0.0"
_DOMAIN_HEIGHT = math.sqrt(3.0) / 2.0


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class GinoPhase1Config:
    """Architecture-side settings; optimizer/training policy lives in preregistration."""

    latent_resolution: int = 32
    n_modes: int = 12
    hidden_channels: int = 64
    n_layers: int = 4
    radius_factor: float = 2.5

    def __post_init__(self) -> None:
        resolution = _positive_int(self.latent_resolution, name="latent_resolution", minimum=8)
        modes = _positive_int(self.n_modes, name="n_modes")
        _positive_int(self.hidden_channels, name="hidden_channels")
        _positive_int(self.n_layers, name="n_layers")
        if modes >= resolution / 2:
            raise ValueError("n_modes must be strictly below latent_resolution / 2 (Nyquist)")
        radius = float(self.radius_factor)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius_factor must be finite and > 0")

    def canonical(self) -> Dict[str, Any]:
        return {
            "backend": GINO_BACKEND_PIN,
            "geometry_semantics": "shared-euclidean-v1",
            "latent_resolution": int(self.latent_resolution),
            "n_modes": int(self.n_modes),
            "hidden_channels": int(self.hidden_channels),
            "n_layers": int(self.n_layers),
            "radius_factor": float(self.radius_factor),
        }

    def digest(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_lattice(lattice: Any) -> HexLattice:
    if not isinstance(lattice, HexLattice):
        raise TypeError("lattice must be a HexLattice")
    coords = np.asarray(lattice.coords)
    if coords.shape != (lattice.num_nodes, 2) or not np.all(np.isfinite(coords)):
        raise ValueError("lattice coordinates must be finite with shape (num_nodes, 2)")
    return lattice


def prepare_gino_batch(
    lattice: HexLattice,
    u0: Any,
    config: GinoPhase1Config,
    *,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Create the exact shared-geometry tensors required by GINO 2.0.0."""
    lattice = _check_lattice(lattice)
    if not isinstance(config, GinoPhase1Config):
        raise TypeError("config must be a GinoPhase1Config")

    values = torch.as_tensor(u0, dtype=dtype)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[1] != lattice.num_nodes:
        raise ValueError(
            f"u0 must have shape ({lattice.num_nodes},) or (batch, {lattice.num_nodes})"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("u0 must be finite")

    device = values.device
    input_geom = torch.as_tensor(
        np.asarray(lattice.coords).copy(), dtype=dtype, device=device
    ).unsqueeze(0)
    axis_x = torch.linspace(0.0, 1.0, config.latent_resolution, dtype=dtype, device=device)
    axis_y = torch.linspace(
        0.0, _DOMAIN_HEIGHT, config.latent_resolution, dtype=dtype, device=device
    )
    grid_x, grid_y = torch.meshgrid(axis_x, axis_y, indexing="ij")
    latent_queries = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    return {
        "input_geom": input_geom,
        "latent_queries": latent_queries,
        "output_queries": input_geom,
        "x": values.unsqueeze(-1),
    }


def build_gino(config: GinoPhase1Config, training_lattice: HexLattice) -> Any:
    """Instantiate the pinned external GINO with a radius fixed from train h."""
    training_lattice = _check_lattice(training_lattice)
    if training_lattice.domain != "triangle_dirichlet":
        raise ValueError("Phase-1 GINO must be configured from the triangle training lattice")
    if not isinstance(config, GinoPhase1Config):
        raise TypeError("config must be a GinoPhase1Config")
    try:
        from neuralop.models import GINO
    except ImportError as exc:
        raise RuntimeError(
            f"optional backend missing; install {GINO_BACKEND_PIN}"
        ) from exc

    radius = float(config.radius_factor) * float(training_lattice.h)
    return GINO(
        in_channels=1,
        out_channels=1,
        gno_coord_dim=2,
        in_gno_radius=radius,
        out_gno_radius=radius,
        fno_n_modes=(config.n_modes, config.n_modes),
        fno_hidden_channels=config.hidden_channels,
        fno_n_layers=config.n_layers,
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
    )


def predict_gino(
    model: Any,
    lattice: HexLattice,
    u0: Any,
    config: GinoPhase1Config,
) -> torch.Tensor:
    """Run GINO and return shape (batch, num_nodes), failing closed on drift."""
    batch = prepare_gino_batch(lattice, u0, config)
    out = model(**batch)
    if not isinstance(out, torch.Tensor):
        raise TypeError("GINO backend must return a torch.Tensor")
    expected = (batch["x"].shape[0], lattice.num_nodes, 1)
    if tuple(out.shape) != expected:
        raise ValueError(f"GINO output shape {tuple(out.shape)} != expected {expected}")
    if not bool(torch.isfinite(out).all()):
        raise ValueError("GINO output must be finite")
    return out[..., 0]
