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
import importlib.metadata
import json
import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from hex_fv_solver_hex_v3 import HexLattice, torus_hex_lattice, triangle_hex_lattice

GINO_BACKEND_PIN = "neuraloperator==2.0.0"
# Bound untrusted lattice metadata before any O(n**2) canonical reconstruction.
_MAX_ADAPTER_N = 512
_MAX_LATENT_RESOLUTION = 128
_MAX_HIDDEN_CHANNELS = 512
_MAX_LAYERS = 16
_MAX_RADIUS_FACTOR = 16.0
# Conservative spectral-weight budget for a dense complex64 FNO stack.
# This excludes GNO/MLP/optimizer/activation allocations, and is NOT a
# complete peak-memory estimate. All arithmetic uses Python integers.
_MAX_SPECTRAL_WEIGHT_BYTES = 256 * 1024 * 1024
_RESOURCE_POLICY = "GINO_ADAPTER_PHASE1_v2"
_DOMAIN_HEIGHT = math.sqrt(3.0) / 2.0
# The GNO radius is fixed from the N=32 training lattice (h = 1/32), see build_gino.
_TRAINING_N = 32
# A node-latent distance within this relative band of the GNO radius is a tie:
# float32 and float64 then build different neighbour graphs (spec v2, B4).
_RADIUS_TIE_RTOL = 1e-6
_TIE_CHUNK_ELEMENTS = 1 << 21
_TIE_CACHE: Dict[tuple, int] = {}


def _latent_grid64(latent_resolution: int) -> np.ndarray:
    """The latent query points of prepare_gino_batch, in float64, shape (R*R, 2)."""
    axis_x = np.linspace(0.0, 1.0, latent_resolution)
    axis_y = np.linspace(0.0, _DOMAIN_HEIGHT, latent_resolution)
    grid_x, grid_y = np.meshgrid(axis_x, axis_y, indexing="ij")
    return np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1)


def _radius_ties(lattice: HexLattice, config: "GinoPhase1Config") -> int:
    """Count node-latent distances d with |d - r| <= 1e-6 r for the GNO radius r.

    The lattice is canonical (checked by _check_lattice), so the count depends
    only on (domain, n, latent_resolution, radius_factor) and is cached.
    """
    key = (lattice.domain, lattice.n, int(config.latent_resolution), float(config.radius_factor))
    cached = _TIE_CACHE.get(key)
    if cached is not None:
        return cached
    radius = float(config.radius_factor) / _TRAINING_N
    band = _RADIUS_TIE_RTOL * radius
    queries = _latent_grid64(int(config.latent_resolution))
    coords = np.asarray(lattice.coords, dtype=np.float64)
    chunk = max(1, _TIE_CHUNK_ELEMENTS // queries.shape[0])
    ties = 0
    for start in range(0, coords.shape[0], chunk):
        diff = coords[start:start + chunk, None, :] - queries[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        ties += int(np.count_nonzero(np.abs(dist - radius) <= band))
    _TIE_CACHE[key] = ties
    return ties


def _refuse_radius_ties(lattice: HexLattice, config: "GinoPhase1Config") -> None:
    ties = _radius_ties(lattice, config)
    if ties != 0:
        raise ValueError(
            f"{ties} node-latent distances lie within {_RADIUS_TIE_RTOL:g} r of the GNO radius "
            f"(radius_factor={float(config.radius_factor)!r}, latent_resolution="
            f"{int(config.latent_resolution)}, {lattice.domain} n={lattice.n}); a radius tie makes "
            "the neighbour graph depend on rounding, choose another radius_factor"
        )


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
        channels = _positive_int(self.hidden_channels, name="hidden_channels")
        layers = _positive_int(self.n_layers, name="n_layers")
        if (resolution > _MAX_LATENT_RESOLUTION or channels > _MAX_HIDDEN_CHANNELS
                or layers > _MAX_LAYERS):
            raise ValueError("GINO configuration exceeds adapter resource limit")
        if modes >= resolution / 2:
            raise ValueError("n_modes must be strictly below latent_resolution / 2 (Nyquist)")
        # neuraloperator 2.0.0 uses dense complex spectral weights for the
        # default (non-factorized) FNO, with the final RFFT axis halved.
        # Even-round odd mode counts UP, so the estimate is conservative.
        effective_modes = modes + (modes % 2)
        spectral_bytes = (
            layers * channels * channels * effective_modes
            * (effective_modes // 2 + 1) * 8
        )
        if spectral_bytes > _MAX_SPECTRAL_WEIGHT_BYTES:
            raise ValueError(
                "GINO combined spectral weight resource limit exceeded: "
                f"estimated {spectral_bytes} bytes > {_MAX_SPECTRAL_WEIGHT_BYTES} bytes"
            )
        if isinstance(self.radius_factor, (bool, str)) or not isinstance(self.radius_factor, numbers.Real):
            raise TypeError("radius_factor must be a real numeric scalar")
        try:
            radius = float(self.radius_factor)
        except OverflowError as exc:
            raise ValueError("radius_factor must fit in a finite float") from exc
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius_factor must be finite and > 0")
        if radius > _MAX_RADIUS_FACTOR:
            raise ValueError("GINO radius_factor exceeds adapter resource limit")

    def canonical(self) -> Dict[str, Any]:
        return {
            "backend": GINO_BACKEND_PIN,
            "resource_policy": _RESOURCE_POLICY,
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
    """Fail closed on fabricated/mutated geometry and adjacency at the public boundary."""
    if not isinstance(lattice, HexLattice):
        raise TypeError("lattice must be a HexLattice")
    if type(lattice.n) is not int or lattice.n < 3:
        raise ValueError("lattice n must be a canonical integer")
    n = lattice.n
    if n > _MAX_ADAPTER_N:
        raise ValueError(f"GINO lattice n exceeds adapter resource limit {_MAX_ADAPTER_N}")
    if lattice.domain == "triangle_dirichlet":
        expected_nodes = (n - 1) * (n - 2) // 2
        factory = triangle_hex_lattice
    elif lattice.domain == "torus_periodic":
        if n < 4 or n % 2:
            raise ValueError("torus lattice n must be even and >= 4")
        expected_nodes = n * n
        factory = torus_hex_lattice
    else:
        raise ValueError("lattice domain is not supported")
    # Never allocate the canonical O(n^2) lattice for malformed caller shapes.
    if (
        not isinstance(lattice.coords, np.ndarray)
        or not isinstance(lattice.neighbors, np.ndarray)
        or lattice.coords.shape != (expected_nodes, 2)
        or lattice.neighbors.shape != (expected_nodes, 6)
        or lattice.coords.dtype != np.dtype("float64")
        or lattice.neighbors.dtype != np.dtype("int64")
        or lattice.h != 1.0 / n
    ):
        raise ValueError("lattice must have canonical shapes, dtypes and spacing")
    canonical = factory(n)
    if (
        lattice.h != canonical.h
        or not isinstance(lattice.coords, np.ndarray)
        or not isinstance(lattice.neighbors, np.ndarray)
        or lattice.coords.dtype != canonical.coords.dtype
        or lattice.neighbors.dtype != canonical.neighbors.dtype
        or not np.array_equal(lattice.coords, canonical.coords)
        or not np.array_equal(lattice.neighbors, canonical.neighbors)
    ):
        raise ValueError("lattice must have canonical geometry and neighbor topology")
    # The adapter must never use mutable caller-owned lattice arrays after check.
    return canonical


def prepare_gino_batch(
    lattice: HexLattice,
    u0: Any,
    config: GinoPhase1Config,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Dict[str, torch.Tensor]:
    """Create the exact shared-geometry tensors required by GINO 2.0.0."""
    lattice = _check_lattice(lattice)
    if not isinstance(config, GinoPhase1Config):
        raise TypeError("config must be a GinoPhase1Config")
    _refuse_radius_ties(lattice, config)

    if dtype not in (torch.float32, torch.float64):
        raise TypeError("GINO inputs require float32 or float64")
    raw = torch.as_tensor(u0)
    if raw.dtype == torch.bool or raw.is_complex():
        raise TypeError("u0 must be a real numeric field, not bool or complex")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("u0 must be finite before dtype conversion")
    values = raw.to(dtype=dtype, device=device)
    # A nonzero field value silently becoming zero is not a valid conversion.
    if raw.is_floating_point() and raw.dtype != dtype:
        converted_source = values.to(device=raw.device)
        if bool(((raw != 0) & (converted_source == 0)).any()):
            raise ValueError("u0 conversion underflows a nonzero field entry")
    # Protect against post-validation writes through caller-owned tensor storage.
    values = values.clone()
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
    if training_lattice.domain != "triangle_dirichlet" or training_lattice.n != 32:
        raise ValueError("Phase-1 GINO requires the fixed N=32 triangle training lattice")
    if not isinstance(config, GinoPhase1Config):
        raise TypeError("config must be a GinoPhase1Config")
    _refuse_radius_ties(training_lattice, config)
    try:
        actual_version = importlib.metadata.version("neuraloperator")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"optional backend missing; install {GINO_BACKEND_PIN}") from exc
    if actual_version != "2.0.0":
        raise RuntimeError(
            f"backend version mismatch: required {GINO_BACKEND_PIN}, got {actual_version}"
        )
    try:
        from neuralop.models import GINO
    except ImportError as exc:
        raise RuntimeError(
            f"optional backend missing; install {GINO_BACKEND_PIN}"
        ) from exc

    radius = float(config.radius_factor) * float(training_lattice.h)
    model = GINO(
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
    # n_modes, hidden_channels, n_layers and radius_factor are baked into the model;
    # predict_gino must not attribute a run to another config's digest (Codex P2, #61).
    setattr(model, CONFIG_DIGEST_ATTR, config.digest())
    return model


CONFIG_DIGEST_ATTR = "_hexantk_gino_config_digest"


def _check_config_binding(model: Any, config: GinoPhase1Config) -> None:
    """A model built by build_gino carries its config digest; any other config is refused.
    Models not built here (test doubles, foreign wrappers) carry no binding and pass."""
    bound = getattr(model, CONFIG_DIGEST_ATTR, None)
    if bound is not None and bound != config.digest():
        raise ValueError(
            f"model was built with config digest {bound[:16]}..., "
            f"predict_gino got {config.digest()[:16]}...; use the construction config"
        )


def _model_device_dtype(model: Any) -> tuple[torch.device | None, torch.dtype]:
    """Use a Module's actual floating parameters, not a guessed CPU/float32 default.

    Complex spectral weights are legal, but all parameters/buffers must live on
    the same device. All real floating tensors must have a consistent dtype.
    """
    if not isinstance(model, torch.nn.Module):
        return None, torch.float32
    tensors = list(model.parameters()) + list(model.buffers())
    if not tensors:
        return None, torch.float32
    device = tensors[0].device
    if any(t.device != device for t in tensors):
        raise ValueError("GINO model parameters/buffers span multiple devices")
    floating = [t.dtype for t in tensors if t.is_floating_point()]
    dtype = floating[0] if floating else torch.float32
    if dtype not in (torch.float32, torch.float64) or any(d != dtype for d in floating):
        raise ValueError("GINO model parameters/buffers have inconsistent floating dtypes")
    expected_complex = torch.complex64 if dtype == torch.float32 else torch.complex128
    if any(t.is_complex() and t.dtype != expected_complex for t in tensors):
        raise ValueError("GINO complex spectral weights do not match the real parameter dtype")
    if dtype == torch.float64 and any(
        type(sub).__module__.startswith("neuralop.") for sub in model.modules()
    ):
        # neuraloperator 2.0.0 allocates in float32/complex64 regardless of the model:
        # layers/segment_csr.py:76 (torch.zeros without dtype), spectral_convolution.py:444.
        # model.modules() includes the model itself, so a caller-side wrapper outside
        # neuralop.* around a float64 neuraloperator submodule is refused too (2026-10-15).
        raise ValueError("float64 GINO is not supported by the neuraloperator 2.0.0 backend; use float32")
    return device, dtype


def predict_gino(
    model: Any,
    lattice: HexLattice,
    u0: Any,
    config: GinoPhase1Config,
) -> torch.Tensor:
    """Run GINO and return shape (batch, num_nodes), failing closed on drift."""
    _check_config_binding(model, config)
    device, dtype = _model_device_dtype(model)
    batch = prepare_gino_batch(lattice, u0, config, dtype=dtype, device=device)
    out = model(**batch)
    if not isinstance(out, torch.Tensor):
        raise TypeError("GINO backend must return a torch.Tensor")
    expected = (batch["x"].shape[0], lattice.num_nodes, 1)
    if tuple(out.shape) != expected:
        raise ValueError(f"GINO output shape {tuple(out.shape)} != expected {expected}")
    if not bool(torch.isfinite(out).all()):
        raise ValueError("GINO output must be finite")
    return out[..., 0]
