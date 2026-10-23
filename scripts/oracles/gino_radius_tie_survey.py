#!/usr/bin/env python3
"""GINO radius-tie and float32-graph survey: decision support for preregistration D4 (owner decision).

``spec/GINO_ADAPTER_PHASE1_v2.md`` (amendment 2026-10-15) refuses any lattice
on which a node-latent distance lies within ``1e-6 r`` of the GNO radius
``r = radius_factor / 32``: float32 and float64 would then build different
neighbour graphs. With the Phase-1 default (latent 32, radius_factor 2.5) the
resolution-axis lattices ``N = 64`` and ``N = 128`` carry 4 ties each, so the
preregistered resolution OOD cannot be evaluated until the owner fixes a
tie-free ``(latent_resolution, radius_factor)`` for every evaluation lattice.

For every candidate pair on the five preregistered lattices (triangle
``N = 16, 32, 64, 128``; torus ``N = 32``) this script reports:

1. the tie count of the adapter's own rule (``_radius_ties``);
2. (2026-10-20, review finding F-S4: "tie-free" is not "float32 and float64
   build the same graph") the neighbour graphs the pinned backend builds.
   neuraloperator 2.0.0 with ``gno_use_open3d=False`` (``build_gino``) searches
   with ``native_neighbor_search``: ``torch.cdist(queries, data)``, a zero
   distance replaced by ``1e-7``, edge iff ``<= radius``. The input GNO queries
   the latent grid against the nodes, the output GNO the nodes against the
   latent grid. Both graphs are rebuilt here with the same torch calls on the
   geometry of ``prepare_gino_batch`` (lattice coordinates and ``torch.linspace``
   latent axes in the target dtype), in float32 and in float64; every graph is
   also built by the backend's own ``native_neighbor_search`` and must match.
   ``flips`` counts the edges present in exactly one of the two dtypes;
3. ``min_relative_margin``: ``min |d - r| / r`` over all node-latent pairs, in
   float64;
4. ``float32_error_near_radius``: the largest ``|d32 - d64| / r`` among pairs
   with ``|d64 - r| <= 0.1 r``, i.e. the float32 rounding of ``torch.cdist`` on
   this machine where it can decide an edge. A candidate whose margin is not
   clearly above it is flip-free here only by the rounding of this BLAS/torch.

It decides nothing: the owner picks from the table. Exit 0 iff at least one
candidate is tie-free AND flip-free, the flip detector saw at least one flip
(positive control), the backend (neuraloperator 2.0.0) was present and every
cross-check matched, and the source revision is bound; otherwise 1.

    python scripts/oracles/gino_radius_tie_survey.py --output results/<name>.json
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src" / "toolkit"))

import numpy  # noqa: E402
import torch  # noqa: E402

from gino_adapter_hex_v3 import (  # noqa: E402
    _DOMAIN_HEIGHT,
    _RADIUS_TIE_RTOL,
    _TRAINING_N,
    GinoPhase1Config,
    _latent_grid64,
    _radius_ties,
)
from hex_fv_solver_hex_v3 import torus_hex_lattice, triangle_hex_lattice  # noqa: E402
from hex_heat_tri_reference_oracle import _report_source_binding, _source_revision  # noqa: E402

# Preregistration D3: train/ID test on the triangle N = 32, resolution OOD on
# the triangle N = 16, 64, 128, composite domain/topology/BC OOD on the torus N = 32.
LATTICES = (
    ("triangle_dirichlet", 16),
    ("triangle_dirichlet", 32),
    ("triangle_dirichlet", 64),
    ("triangle_dirichlet", 128),
    ("torus_periodic", 32),
)
LATENT_RESOLUTIONS = (8, 16, 24, 32, 48, 64)
RADIUS_FACTORS = (1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0)
DEFAULT = GinoPhase1Config()
BACKEND_PIN = "2.0.0"
# native_neighbor_search keeps a zero distance as a neighbour by replacing it with this value.
_ZERO_DISTANCE = 1e-7
# Pairs with |d64 - r| <= this fraction of r enter the float32 error figure.
_NEAR_RADIUS_BAND = 0.1
_MARGIN_CHUNK_ELEMENTS = 1 << 21


def _lattice(domain: str, n: int):
    return triangle_hex_lattice(n) if domain == "triangle_dirichlet" else torus_hex_lattice(n)


def _config(latent_resolution: int, radius_factor: float) -> GinoPhase1Config:
    # n_modes does not enter the tie count; it only has to satisfy the Nyquist rule.
    n_modes = max(1, min(DEFAULT.n_modes, latent_resolution // 2 - 1))
    return GinoPhase1Config(latent_resolution=latent_resolution, n_modes=n_modes, radius_factor=radius_factor)


def _backend_search():
    """The backend's native_neighbor_search, or None when neuraloperator 2.0.0 is absent."""
    try:
        version = importlib.metadata.version("neuraloperator")
    except importlib.metadata.PackageNotFoundError:
        return None, None
    if version != BACKEND_PIN:
        return None, version
    from neuralop.layers.neighbor_search import native_neighbor_search
    return native_neighbor_search, version


def _geometry(lattice, latent_resolution: int, dtype: torch.dtype):
    """Nodes and flattened latent queries exactly as prepare_gino_batch and GINO.forward build them."""
    nodes = torch.as_tensor(numpy.asarray(lattice.coords).copy(), dtype=dtype)
    axis_x = torch.linspace(0.0, 1.0, latent_resolution, dtype=dtype)
    axis_y = torch.linspace(0.0, _DOMAIN_HEIGHT, latent_resolution, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(axis_x, axis_y, indexing="ij")
    latent = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
    return nodes, latent


def _search(queries: torch.Tensor, data: torch.Tensor, radius: float):
    """Distances and edge matrix of native_neighbor_search (neuraloperator 2.0.0)."""
    dist = torch.cdist(queries, data)
    dist = torch.where(dist == 0.0, _ZERO_DISTANCE, dist)
    return dist, (dist <= radius).numpy()


def _backend_matrix(result: dict, shape: tuple) -> numpy.ndarray:
    splits = result["neighbors_row_splits"].numpy()
    index = result["neighbors_index"].numpy()
    if splits.shape != (shape[0] + 1,) or int(splits[-1]) != index.shape[0]:
        raise RuntimeError("backend neighbour search returned an inconsistent CSR structure")
    out = numpy.zeros(shape, dtype=bool)
    out[numpy.repeat(numpy.arange(shape[0]), numpy.diff(splits)), index] = True
    return out


def _graph_study(lattice, latent_resolution: int, radius: float, backend) -> dict:
    """float32 vs float64 neighbour graphs of both GNO directions on one lattice."""
    flips, error, checks, mismatches = {}, 0.0, 0, 0
    geometry = {dtype: _geometry(lattice, latent_resolution, dtype) for dtype in (torch.float32, torch.float64)}
    for direction in ("in", "out"):
        built = {}
        for dtype, (nodes, latent) in geometry.items():
            queries, data = (latent, nodes) if direction == "in" else (nodes, latent)
            dist, edges = _search(queries, data, radius)
            if backend is not None:
                checks += 1
                if not numpy.array_equal(_backend_matrix(backend(data, queries, radius), edges.shape), edges):
                    mismatches += 1
            built[dtype] = (dist, edges)
        (d32, e32), (d64, e64) = built[torch.float32], built[torch.float64]
        flips[direction] = int(numpy.count_nonzero(e32 != e64))
        d64 = d64.numpy()
        near = numpy.abs(d64 - radius) <= _NEAR_RADIUS_BAND * radius
        if near.any():
            gap = numpy.abs(d32.double().numpy()[near] - d64[near])
            error = max(error, float(numpy.max(gap)) / radius)
        del built, d32, d64, e32, e64
    return {"flips": flips, "float32_error_near_radius": error, "checks": checks, "mismatches": mismatches}


def _min_relative_margin(lattice, latent_resolution: int, radius: float) -> float:
    queries = _latent_grid64(latent_resolution)
    coords = numpy.asarray(lattice.coords, dtype=numpy.float64)
    chunk = max(1, _MARGIN_CHUNK_ELEMENTS // queries.shape[0])
    best = numpy.inf
    for start in range(0, coords.shape[0], chunk):
        diff = coords[start:start + chunk, None, :] - queries[None, :, :]
        dist = numpy.sqrt(numpy.sum(diff * diff, axis=-1))
        best = min(best, float(numpy.min(numpy.abs(dist - radius))))
    return best / radius


def survey(backend) -> dict:
    lattices = {f"{domain}/n={n}": _lattice(domain, n) for domain, n in LATTICES}
    training_h = float(triangle_hex_lattice(_TRAINING_N).h)
    rows = []
    checks = mismatches = 0
    for latent in LATENT_RESOLUTIONS:
        for factor in RADIUS_FACTORS:
            config = _config(latent, factor)
            # build_gino passes radius_factor * h of the N = 32 training lattice to both GNOs.
            radius = float(config.radius_factor) * training_h
            ties, flips, margins, errors = {}, {}, {}, {}
            for name, lattice in lattices.items():
                ties[name] = int(_radius_ties(lattice, config))
                study = _graph_study(lattice, latent, radius, backend)
                checks += study["checks"]
                mismatches += study["mismatches"]
                flips[name] = study["flips"]
                errors[name] = study["float32_error_near_radius"]
                margins[name] = _min_relative_margin(lattice, latent, radius)
            min_margin, max_error = min(margins.values()), max(errors.values())
            total_flips = sum(v["in"] + v["out"] for v in flips.values())
            tie_free = all(v == 0 for v in ties.values())
            rows.append({
                "latent_resolution": latent,
                "radius_factor": factor,
                "radius": radius,
                "ties": ties,
                "tie_free_on_all": tie_free,
                "float32_flips": flips,
                "float32_flips_total": total_flips,
                "flip_free_on_all": tie_free and total_flips == 0,
                "min_relative_margin": min_margin,
                "min_relative_margin_per_lattice": margins,
                "float32_error_near_radius": max_error,
                "margin_over_float32_error": (min_margin / max_error) if max_error > 0 else None,
            })

    def pick(row):
        return {"latent_resolution": row["latent_resolution"], "radius_factor": row["radius_factor"],
                "min_relative_margin": row["min_relative_margin"],
                "float32_error_near_radius": row["float32_error_near_radius"],
                "margin_over_float32_error": row["margin_over_float32_error"]}

    default_row = next(r for r in rows if r["latent_resolution"] == DEFAULT.latent_resolution
                       and r["radius_factor"] == DEFAULT.radius_factor)
    detector_live = any(r["float32_flips_total"] > 0 for r in rows)
    flip_free = [pick(r) for r in rows if r["flip_free_on_all"]]
    return {
        "rule": {
            "tie": f"|d - r| <= {_RADIUS_TIE_RTOL:g} r in float64, r = radius_factor / {_TRAINING_N}",
            "latent_grid": "linspace(0, 1, R) x linspace(0, sqrt(3)/2, R), as prepare_gino_batch",
            "source": "gino_adapter_hex_v3._radius_ties (spec/GINO_ADAPTER_PHASE1_v2.md, amendment 2026-10-15)",
            "graph": ("neuraloperator 2.0.0 native_neighbor_search (gno_use_open3d=False): "
                      "torch.cdist(queries, data), 0 -> 1e-7, edge iff <= radius; input GNO "
                      "queries=latent grid, data=nodes; output GNO queries=nodes, data=latent grid"),
            "flips": "edges present in exactly one of the float32 and float64 graphs (both GNO directions)",
            "margin": "min |d - r| / r over all node-latent pairs, float64",
            "float32_error_near_radius": f"max |d32 - d64| / r over pairs with |d64 - r| <= {_NEAR_RADIUS_BAND:g} r",
        },
        "lattices": [{"domain": d, "n": n, "num_nodes": lattices[f"{d}/n={n}"].num_nodes} for d, n in LATTICES],
        "candidates": rows,
        "default": {"latent_resolution": DEFAULT.latent_resolution, "radius_factor": DEFAULT.radius_factor,
                    "ties": default_row["ties"], "float32_flips": default_row["float32_flips"]},
        "tie_free_candidates": [{"latent_resolution": r["latent_resolution"], "radius_factor": r["radius_factor"]}
                                for r in rows if r["tie_free_on_all"]],
        "flip_free_candidates": flip_free,
        "positive_control": {"flip_detector_saw_a_flip": detector_live,
                             "default_flips_total": default_row["float32_flips_total"]},
        "backend_crosscheck": {"graphs_checked": checks, "mismatches": mismatches},
        "passed": bool(flip_free) and detector_live and checks > 0 and mismatches == 0,
    }


def build_report() -> dict:
    backend, version = _backend_search()
    report = {
        "source": _source_revision(),
        "environment": {"python": platform.python_version(), "numpy": numpy.__version__,
                        "torch": torch.__version__, "neuraloperator": version,
                        "platform": platform.platform()},
        "survey": survey(backend),
    }
    report["survey"]["backend_crosscheck"]["backend"] = (
        f"neuraloperator {version}" if backend is not None
        else f"absent (need neuraloperator=={BACKEND_PIN}, found {version!r})")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the JSON report here (default: stdout)")
    args = parser.parse_args(argv)
    report = build_report()
    text = json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    result = report["survey"]
    default = result["default"]
    cross = result["backend_crosscheck"]
    print(
        f"[GINO-RADIUS-TIES] default latent={default['latent_resolution']} radius_factor="
        f"{default['radius_factor']} ties={default['ties']} flips={default['float32_flips']} | "
        f"tie-free: {len(result['tie_free_candidates'])}/{len(result['candidates'])} | "
        f"tie-free and flip-free: {len(result['flip_free_candidates'])} | "
        f"backend cross-check {cross['graphs_checked']} graphs, {cross['mismatches']} mismatches "
        f"({cross['backend']}) | {'PASS' if result['passed'] else 'FAIL'}",
        file=sys.stderr,
    )
    bound = _report_source_binding(report["source"], "GINO-RADIUS-TIES")
    return 0 if result["passed"] and bound else 1


if __name__ == "__main__":
    sys.exit(main())
