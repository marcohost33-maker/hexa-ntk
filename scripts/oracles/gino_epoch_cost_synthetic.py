#!/usr/bin/env python3
"""GINO epoch cost on synthetic tensors: decision support for preregistration D4/D5 (owner decision).

``spec/PREREGISTRATION_PHASE1_v1.md`` line 14 forbids any fit, tuning or
evaluation on T1 data before approval and allows "smoke tests on synthetic
tensors". This script stays on that side of the line:

- no T1 data: ``heat_operator_task_hex_v3`` is not imported, no dataset is
  generated or loaded. Inputs and targets are standard-normal tensors from a
  fixed ``torch.Generator``; their values carry no information about T1;
- the geometry is the N = 32 triangle training lattice (465 nodes), from
  ``triangle_hex_lattice(32)`` -- lattice coordinates are not T1 data;
- the model is the pinned neuraloperator 2.0.0 GINO built by the adapter
  (``build_gino``, ``predict_gino``), loss relative L2, Adam, cosine schedule,
  as in preregistration section 3.

For every grid configuration it runs ``--warmup-steps`` untimed minibatches,
then times ``--repeats`` measured epochs of ``--n-train`` samples (1800 = the 90 % train split of D3)
at ``--batch-size``, plus a no-grad pass over ``--n-val`` samples, and reports
the median. The numbers are wall-clock seconds on THIS machine at the given
thread count; they are a cost estimate for budgeting the D4 epoch count, not
a result about GINO on T1 and not a latency measurement of section 4.

It decides nothing. Exit 0 iff every configuration trained with finite losses,
the backend is neuraloperator 2.0.0 and the source revision is bound;
otherwise 1 (argument errors: 2).

    python scripts/oracles/gino_epoch_cost_synthetic.py --threads 4 --output results/<name>.json
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src" / "toolkit"))
sys.path.insert(0, str(_REPO / "scripts" / "oracles"))

import numpy  # noqa: E402
import torch  # noqa: E402

from gino_adapter_hex_v3 import (  # noqa: E402
    _DOMAIN_HEIGHT,
    _TRAINING_N,
    GinoPhase1Config,
    build_gino,
    predict_gino,
)
from hex_fv_solver_hex_v3 import torus_hex_lattice, triangle_hex_lattice  # noqa: E402
from hex_heat_tri_reference_oracle import _report_source_binding, _source_revision  # noqa: E402

# Candidate grid (decision support, spec/PREREGISTRATION_PHASE1_v1.md section 8, 2026-10-29).
WIDTHS = (32, 64)
LATENT_RESOLUTIONS = (16, 24)
LEARNING_RATES = (1e-3, 3e-4)
# n_modes follows the adapter default ratio 12/32 (GinoPhase1Config): 6 for latent 16,
# 9 for latent 24. The default 12 is refused for both (n_modes < latent/2).
N_MODES = {16: 6, 24: 9}
N_LAYERS = 4
RADIUS_FACTOR = 1.5
# Neighbourhood statistics only (geometry, no T1 data): the five preregistered lattices.
LATTICES = (
    ("triangle_dirichlet", 16),
    ("triangle_dirichlet", 32),
    ("triangle_dirichlet", 64),
    ("triangle_dirichlet", 128),
    ("torus_periodic", 32),
)
SURVEY_RADIUS_FACTORS = (1.5, 2.25, 2.5, 2.75)
DATA_SEED = 20261029
SHUFFLE_SEED = 0
MODEL_SEED = 0


def _positive(kind):
    def parse(text: str):
        value = kind(text)
        if isinstance(value, float) and not math.isfinite(value):
            raise argparse.ArgumentTypeError(f"must be finite, got {text!r}")
        if value <= 0:
            raise argparse.ArgumentTypeError(f"must be > 0, got {text!r}")
        return value
    return parse


def _cpu_load_percent() -> int | None:
    """Whole-machine CPU load (Win32_Processor.LoadPercentage); other processes are not controlled."""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor | Select-Object -First 1).LoadPercentage"],
                             capture_output=True, text=True, check=True, timeout=60,
                             stdin=subprocess.DEVNULL).stdout.strip()
        return int(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _hardware() -> dict:
    info: dict = {"platform": platform.platform(), "machine": platform.machine(),
                  "processor": platform.processor(), "os_cpu_count": os.cpu_count()}
    if sys.platform == "win32":
        command = ("$p = Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,"
                   "NumberOfLogicalProcessors,MaxClockSpeed; $m = (Get-CimInstance Win32_ComputerSystem)."
                   "TotalPhysicalMemory; $o = Get-CimInstance Win32_OperatingSystem | Select-Object -First 1 "
                   "Caption,Version,BuildNumber; @{cpu=$p; ram_bytes=$m; os=$o} | ConvertTo-Json -Compress")
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True,
                                 text=True, check=True, timeout=60, stdin=subprocess.DEVNULL).stdout
            raw = json.loads(out)
            info["cpu_name"] = str(raw["cpu"]["Name"]).strip()
            info["physical_cores"] = int(raw["cpu"]["NumberOfCores"])
            info["logical_processors"] = int(raw["cpu"]["NumberOfLogicalProcessors"])
            info["max_clock_mhz"] = int(raw["cpu"]["MaxClockSpeed"])
            info["ram_bytes"] = int(raw["ram_bytes"])
            info["os"] = f"{raw['os']['Caption']} {raw['os']['Version']} build {raw['os']['BuildNumber']}"
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
            info["query_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (torch.linalg.vector_norm(pred - target, dim=1)
            / torch.linalg.vector_norm(target, dim=1)).mean()


def _neighbourhoods() -> list[dict]:
    """Smallest number of latent points within r of a node (output GNO), float64, per candidate."""
    rows = []
    for latent in LATENT_RESOLUTIONS:
        axis_x = numpy.linspace(0.0, 1.0, latent)
        axis_y = numpy.linspace(0.0, _DOMAIN_HEIGHT, latent)
        gx, gy = numpy.meshgrid(axis_x, axis_y, indexing="ij")
        grid = numpy.stack((gx.ravel(), gy.ravel()), axis=-1)
        for factor in SURVEY_RADIUS_FACTORS:
            radius = factor / _TRAINING_N
            per_lattice = {}
            for domain, n in LATTICES:
                lattice = triangle_hex_lattice(n) if domain == "triangle_dirichlet" else torus_hex_lattice(n)
                counts = numpy.zeros(lattice.coords.shape[0], dtype=numpy.int64)
                for start in range(0, lattice.coords.shape[0], 2048):
                    block = lattice.coords[start:start + 2048, None, :] - grid[None, :, :]
                    dist = numpy.sqrt(numpy.sum(block * block, axis=-1))
                    counts[start:start + 2048] = numpy.count_nonzero(dist <= radius, axis=1)
                per_lattice[f"{domain}/n={n}"] = {"min": int(counts.min()), "mean": float(counts.mean())}
            rows.append({"latent_resolution": latent, "radius_factor": factor,
                         "min_latent_neighbours_per_node": min(v["min"] for v in per_lattice.values()),
                         "per_lattice": per_lattice})
    return rows


def _time_config(width: int, latent: int, lr: float, args, lattice, u0, target, u0_val, target_val) -> dict:
    config = GinoPhase1Config(latent_resolution=latent, n_modes=N_MODES[latent], hidden_channels=width,
                              n_layers=N_LAYERS, radius_factor=RADIUS_FACTOR)
    torch.manual_seed(MODEL_SEED)
    model = build_gino(config, lattice)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.repeats)
    shuffle = torch.Generator().manual_seed(SHUFFLE_SEED)

    def step(index: torch.Tensor) -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = _relative_l2(predict_gino(model, lattice, u0[index], config), target[index])
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    # Warm-up: the first --warmup-steps minibatches, untimed (allocator, kernel selection).
    order = torch.randperm(args.n_train, generator=shuffle)
    start = time.perf_counter()
    for k in range(args.warmup_steps):
        first = (k * args.batch_size) % args.n_train
        step(order[first:first + args.batch_size])
    warmup_seconds = time.perf_counter() - start
    epoch_seconds, losses = [], []
    for _ in range(args.repeats):
        order = torch.randperm(args.n_train, generator=shuffle)
        start = time.perf_counter()
        last = math.nan
        for first in range(0, args.n_train, args.batch_size):
            last = step(order[first:first + args.batch_size])
        scheduler.step()
        epoch_seconds.append(time.perf_counter() - start)
        losses.append(last)
    val_seconds = []
    model.eval()
    for _ in range(1 + args.repeats):
        start = time.perf_counter()
        with torch.no_grad():
            for first in range(0, args.n_val, args.batch_size):
                predict_gino(model, lattice, u0_val[first:first + args.batch_size], config)
        val_seconds.append(time.perf_counter() - start)
    measured = epoch_seconds
    measured_val = val_seconds[1:]
    return {
        "hidden_channels": width, "latent_resolution": latent, "n_modes": N_MODES[latent],
        "n_layers": N_LAYERS, "radius_factor": RADIUS_FACTOR, "learning_rate": lr,
        "config_digest": config.digest(), "n_parameters": int(n_params),
        "warmup_steps_seconds": warmup_seconds, "warmup_val_pass_seconds": val_seconds[0],
        "epoch_seconds": measured, "epoch_seconds_median": statistics.median(measured),
        "val_pass_seconds": measured_val, "val_pass_seconds_median": statistics.median(measured_val),
        "last_batch_loss_per_epoch": losses,
        "losses_finite": all(math.isfinite(x) for x in losses),
    }


def build_report(args, argv: list[str]) -> dict:
    source = _source_revision()
    torch.set_num_threads(args.threads)
    try:
        backend = importlib.metadata.version("neuraloperator")
    except importlib.metadata.PackageNotFoundError:
        backend = None
    lattice = triangle_hex_lattice(_TRAINING_N)
    data = torch.Generator().manual_seed(DATA_SEED)
    n_nodes = lattice.num_nodes
    u0 = torch.randn((args.n_train, n_nodes), generator=data, dtype=torch.float32)
    target = torch.randn((args.n_train, n_nodes), generator=data, dtype=torch.float32)
    u0_val = torch.randn((args.n_val, n_nodes), generator=data, dtype=torch.float32)
    target_val = torch.randn((args.n_val, n_nodes), generator=data, dtype=torch.float32)
    load_start = _cpu_load_percent()
    configs = []
    for width in WIDTHS:
        for latent in LATENT_RESOLUTIONS:
            for lr in LEARNING_RATES:
                configs.append(_time_config(width, latent, lr, args, lattice, u0, target, u0_val, target_val))
                print(f"[GINO-EPOCH-COST] width={width} latent={latent} lr={lr:g} "
                      f"epoch median {configs[-1]['epoch_seconds_median']:.3f} s", file=sys.stderr)
    medians = [c["epoch_seconds_median"] for c in configs]
    passed = backend == "2.0.0" and all(c["losses_finite"] for c in configs)
    return {
        "source": source,
        "command": [Path(sys.executable).name, "scripts/oracles/gino_epoch_cost_synthetic.py", *argv],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": numpy.__version__, "neuraloperator": backend,
                        "torch_num_threads": torch.get_num_threads(),
                        "torch_num_interop_threads": torch.get_num_interop_threads(),
                        "cuda_available": bool(torch.cuda.is_available()), "dtype": "float32", "amp": False},
        "hardware": _hardware(),
        "machine_load_percent": {"start": load_start, "end": _cpu_load_percent(),
                                 "note": "shared machine; other processes ran during the measurement"},
        "data": {"kind": "synthetic standard-normal tensors (torch.Generator), no T1 data",
                 "data_seed": DATA_SEED, "shuffle_seed": SHUFFLE_SEED, "model_seed": MODEL_SEED,
                 "geometry": f"triangle_hex_lattice({_TRAINING_N}), {n_nodes} nodes",
                 "n_train": args.n_train, "n_val": args.n_val, "batch_size": args.batch_size,
                 "steps_per_epoch": math.ceil(args.n_train / args.batch_size)},
        "protocol": {"warmup_steps": args.warmup_steps, "warmup_val_passes": 1,
                     "measured_epochs": args.repeats, "statistic": "median",
                     "clock": "time.perf_counter", "optimizer": "Adam", "schedule": "CosineAnnealingLR",
                     "loss": "relative L2 (mean over batch)"},
        "configs": configs,
        "summary": {"epoch_seconds_median_max": max(medians), "epoch_seconds_median_min": min(medians),
                    "epoch_seconds_median_sum": sum(medians)},
        "neighbourhoods": _neighbourhoods(),
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=_positive(int), required=True, help="torch.set_num_threads")
    parser.add_argument("--repeats", type=_positive(int), default=3)
    parser.add_argument("--warmup-steps", type=_positive(int), default=10)
    parser.add_argument("--n-train", type=_positive(int), default=1800)
    parser.add_argument("--n-val", type=_positive(int), default=200)
    parser.add_argument("--batch-size", type=_positive(int), default=20)
    parser.add_argument("--output", type=Path, help="write the JSON report here (default: stdout)")
    args = parser.parse_args(argv)
    if args.batch_size > args.n_train:
        parser.error("--batch-size must not exceed --n-train")
    report = build_report(args, argv)
    text = json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    print(f"[GINO-EPOCH-COST] {len(report['configs'])} configs, epoch median "
          f"{report['summary']['epoch_seconds_median_min']:.3f}..{report['summary']['epoch_seconds_median_max']:.3f}"
          f" s at {report['environment']['torch_num_threads']} threads | {'PASS' if report['passed'] else 'FAIL'}",
          file=sys.stderr)
    bound = _report_source_binding(report["source"], "GINO-EPOCH-COST")
    return 0 if report["passed"] and bound else 1


if __name__ == "__main__":
    sys.exit(main())
