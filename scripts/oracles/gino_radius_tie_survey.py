#!/usr/bin/env python3
"""GINO radius-tie survey: decision support for preregistration D4 (owner decision).

``spec/GINO_ADAPTER_PHASE1_v2.md`` (amendment 2026-10-15) refuses any lattice
on which a node-latent distance lies within ``1e-6 r`` of the GNO radius
``r = radius_factor / 32``: float32 and float64 would then build different
neighbour graphs. With the Phase-1 default (latent 32, radius_factor 2.5) the
resolution-axis lattices ``N = 64`` and ``N = 128`` carry 4 ties each, so the
preregistered resolution OOD cannot be evaluated until the owner fixes a
tie-free ``(latent_resolution, radius_factor)`` for every evaluation lattice.

This script measures, with the adapter's own rule (``_radius_ties``), the tie
count of every candidate pair on the five preregistered lattices (triangle
``N = 16, 32, 64, 128``; torus ``N = 32``) and lists the candidates that are
tie-free on all of them. It decides nothing: the owner picks from the table.
Exit 0 iff at least one tie-free candidate exists AND the source revision is
bound (clean work tree, verified HEAD); otherwise 1.

    python scripts/oracles/gino_radius_tie_survey.py --output results/<name>.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src" / "toolkit"))

import numpy  # noqa: E402
import torch  # noqa: E402

from gino_adapter_hex_v3 import (  # noqa: E402
    _RADIUS_TIE_RTOL,
    _TRAINING_N,
    GinoPhase1Config,
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


def _lattice(domain: str, n: int):
    return triangle_hex_lattice(n) if domain == "triangle_dirichlet" else torus_hex_lattice(n)


def _config(latent_resolution: int, radius_factor: float) -> GinoPhase1Config:
    # n_modes does not enter the tie count; it only has to satisfy the Nyquist rule.
    n_modes = max(1, min(DEFAULT.n_modes, latent_resolution // 2 - 1))
    return GinoPhase1Config(latent_resolution=latent_resolution, n_modes=n_modes, radius_factor=radius_factor)


def survey() -> dict:
    lattices = {f"{domain}/n={n}": _lattice(domain, n) for domain, n in LATTICES}
    rows = []
    for latent in LATENT_RESOLUTIONS:
        for factor in RADIUS_FACTORS:
            config = _config(latent, factor)
            ties = {name: int(_radius_ties(lattice, config)) for name, lattice in lattices.items()}
            rows.append({
                "latent_resolution": latent,
                "radius_factor": factor,
                "radius": factor / _TRAINING_N,
                "ties": ties,
                "tie_free_on_all": all(v == 0 for v in ties.values()),
            })
    default_row = next(r for r in rows if r["latent_resolution"] == DEFAULT.latent_resolution
                       and r["radius_factor"] == DEFAULT.radius_factor)
    tie_free = [(r["latent_resolution"], r["radius_factor"]) for r in rows if r["tie_free_on_all"]]
    return {
        "rule": {
            "tie": f"|d - r| <= {_RADIUS_TIE_RTOL:g} r in float64, r = radius_factor / {_TRAINING_N}",
            "latent_grid": "linspace(0, 1, R) x linspace(0, sqrt(3)/2, R), as prepare_gino_batch",
            "source": "gino_adapter_hex_v3._radius_ties (spec/GINO_ADAPTER_PHASE1_v2.md, amendment 2026-10-15)",
        },
        "lattices": [{"domain": d, "n": n, "num_nodes": lattices[f"{d}/n={n}"].num_nodes} for d, n in LATTICES],
        "candidates": rows,
        "default": {"latent_resolution": DEFAULT.latent_resolution, "radius_factor": DEFAULT.radius_factor,
                    "ties": default_row["ties"]},
        "tie_free_candidates": [{"latent_resolution": r, "radius_factor": f} for r, f in tie_free],
        "passed": bool(tie_free),
    }


def build_report() -> dict:
    return {
        "source": _source_revision(),
        "environment": {"python": platform.python_version(), "numpy": numpy.__version__,
                        "torch": torch.__version__, "platform": platform.platform()},
        "survey": survey(),
    }


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
    print(
        f"[GINO-RADIUS-TIES] default latent={default['latent_resolution']} radius_factor="
        f"{default['radius_factor']} ties={default['ties']} | tie-free candidates: "
        f"{len(result['tie_free_candidates'])}/{len(result['candidates'])} | "
        f"{'PASS' if result['passed'] else 'FAIL no tie-free candidate'}",
        file=sys.stderr,
    )
    bound = _report_source_binding(report["source"], "GINO-RADIUS-TIES")
    return 0 if result["passed"] and bound else 1


if __name__ == "__main__":
    sys.exit(main())
