#!/usr/bin/env python3
"""Evidence writer: the pinned GINO (GinoRunnerModule) under DistributedDataParallel on gloo.

Contract: spec/GINO_TRAINING_PHASE1_v1.md, section 8a. Synthetic tensors only
(standard-normal fields on the N = 32 triangle lattice); no T1 data is
generated, loaded or fit (spec/PREREGISTRATION_PHASE1_v1.md, lines 14-15).

Part A, "premise": two gloo ranks. torch 2.6.0's gloo backend refuses to
broadcast complex tensors ("Invalid scalar type") while it all-reduces them,
and DistributedDataParallel broadcasts every parameter at construction. The
pinned GINO holds its spectral weights as complex64 parameters, so
``DDP(GinoRunnerModule(...))`` fails on every rank before the first step. If
that no longer happens (another torch or backend), the premise of section 8a
has changed and the script reports it.

Part B, "parametrize": ``torch.nn.utils.parametrize`` cannot hold the complex
weight as a real parameter, because tltorch's ``FactorizedTensor`` refuses the
class parametrize injects. Recorded as an option that does not work.

Part C, "exclude and average": the option section 8a puts to the owner, measured
only, not adopted in the module. The complex parameters are excluded from DDP
(``DDP._set_params_and_buffers_to_ignore_for_model``), broadcast once as their
real view, and their gradients are averaged by an all-reduce in a
post-accumulate-grad hook. Two gloo ranks with per-rank batch 2 against the
single process with batch 4, through HexTrainingRunner, 3 epochs, once with SGD
and once with Adam (learning rate 1e-2, no clipping, no EMA).

Exit 0 only if part A fails as recorded on both ranks, part B is refused as
recorded, part C ends on both ranks with bitwise equal weights and, with SGD,
within 1e-6 of the single-process run, the backend is neuraloperator 2.0.0 and
the source revision is bound (clean work tree, verified HEAD); otherwise 1.

    python scripts/oracles/gino_ddp_gloo_oracle.py --output results/<name>.json
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src" / "toolkit"))
sys.path.insert(0, str(_REPO / "scripts" / "oracles"))

import numpy  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from ddp_hex_v3 import ExactDistributedEvalSampler  # noqa: E402
from gino_adapter_hex_v3 import GinoPhase1Config  # noqa: E402
from gino_training_hex_v3 import GinoRunnerModule, RelativeL2Loss, RunnerPairDataset  # noqa: E402
from hex_fv_solver_hex_v3 import triangle_hex_lattice  # noqa: E402
from hex_heat_tri_reference_oracle import _report_source_binding, _source_revision  # noqa: E402
from runner_hex_v3 import HexTrainingRunner, RunnerConfig  # noqa: E402

_LATTICE = triangle_hex_lattice(32)
_NODES = _LATTICE.num_nodes
_CONFIG = GinoPhase1Config(latent_resolution=16, n_modes=4, hidden_channels=8, n_layers=2, radius_factor=2.5)
_EPOCHS = 3
_SEED = 3
_LR = 1e-2
_PER_RANK = 2
_WORLD = 2
_SGD_TOLERANCE = 1e-6
_PREMISE_ERROR = "Invalid scalar type"


class _Source:
    def __init__(self, count: int, seed: int) -> None:
        rng = numpy.random.default_rng(seed)
        self.u0 = rng.standard_normal((count, _NODES))
        self.u_t = rng.standard_normal((count, _NODES))

    def __len__(self) -> int:
        return len(self.u0)

    def __getitem__(self, index: int) -> dict:
        return {"trajectory_id": 100 + index, "u0": self.u0[index], "uT": self.u_t[index]}


def _data() -> tuple:
    return RunnerPairDataset(_Source(8, 1), _NODES), RunnerPairDataset(_Source(6, 2), _NODES)


def _module() -> GinoRunnerModule:
    torch.manual_seed(_SEED)
    return GinoRunnerModule(_CONFIG, _LATTICE)


def _flat(module: nn.Module) -> torch.Tensor:
    return torch.cat([(torch.view_as_real(p.detach()) if p.is_complex() else p.detach()).reshape(-1)
                      for p in module.parameters()])


def _complex_mask(module: nn.Module) -> torch.Tensor:
    return torch.cat([torch.full(((torch.view_as_real(p) if p.is_complex() else p).numel(),), p.is_complex())
                      for p in module.parameters()])


def _runner(out: Path, model: nn.Module, train: DataLoader, val: DataLoader, optimizer: str,
            *, distributed: bool = False, rank: int = 0) -> HexTrainingRunner:
    params = model.parameters()
    opt = torch.optim.SGD(params, lr=_LR) if optimizer == "sgd" else torch.optim.Adam(params, lr=_LR)
    config = RunnerConfig(output_dir=str(out), num_epochs=_EPOCHS, grad_clip_max_norm=float("inf"),
                          use_ema=False, use_amp=False, log_every_n_steps=0, best_metric_name="val_loss",
                          distributed=distributed, is_main_process=rank == 0)
    return HexTrainingRunner(config, model, opt, train, val, loss_fn=RelativeL2Loss(),
                             scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_EPOCHS),
                             device="cpu")


def _exclude_and_average(module: nn.Module) -> DDP:
    """Part C: complex parameters outside DDP, broadcast as real views, gradients averaged by a hook."""
    names = [n for n, p in module.named_parameters() if p.is_complex()]
    DDP._set_params_and_buffers_to_ignore_for_model(module, names)
    world = dist.get_world_size()
    for p in module.parameters():
        if p.is_complex():
            with torch.no_grad():
                dist.broadcast(torch.view_as_real(p.data), 0)

            def average(param: torch.Tensor, world: int = world) -> None:
                grad = torch.view_as_real(param.grad)
                dist.all_reduce(grad)
                grad.div_(world)

            p.register_post_accumulate_grad_hook(average)
    return DDP(module)


def _outcome(fn) -> dict:
    try:
        return {"ok": True, "value": fn()}
    except Exception as exc:  # recorded, judged by the parent
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:300]}


def _worker(rank: int, root: str, part: str, optimizer: str) -> None:
    logging.disable(logging.CRITICAL)
    dist.init_process_group("gloo", init_method=(Path(root) / "rendezvous").as_uri(), rank=rank,
                            world_size=_WORLD, timeout=timedelta(seconds=120))
    try:
        if part == "premise":
            def premise() -> dict:
                values = torch.ones(2, dtype=torch.complex64)
                dist.all_reduce(values)
                return {"all_reduce_complex": [values.real.tolist(), values.imag.tolist()]}
            result = {"all_reduce": _outcome(premise),
                      "broadcast_complex": _outcome(lambda: dist.broadcast(
                          torch.ones(2, dtype=torch.complex64), 0) and None),
                      "ddp_construction": _outcome(lambda: DDP(_module()) and None)}
        else:
            def run() -> list:
                module = _module()
                train, val = _data()
                model = _exclude_and_average(module)
                runner = _runner(Path(root) / "out", model,
                                 DataLoader(train, batch_size=_PER_RANK,
                                            sampler=DistributedSampler(train, _WORLD, rank, shuffle=False)),
                                 DataLoader(val, batch_size=_PER_RANK, sampler=ExactDistributedEvalSampler(val)),
                                 optimizer, distributed=True, rank=rank)
                runner.run()
                return _flat(module).tolist()
            result = {"run": _outcome(run)}
    finally:
        (Path(root) / f"rank{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


def _job(part: str, optimizer: str = "") -> list:
    with tempfile.TemporaryDirectory() as root:
        mp.start_processes(_worker, args=(root, part, optimizer), nprocs=_WORLD, start_method="spawn", join=True)
        return [json.loads((Path(root) / f"rank{r}.json").read_text()) for r in range(_WORLD)]


def _premise_report() -> dict:
    ranks = _job("premise")
    as_recorded = all(
        r["all_reduce"]["ok"] and not r["broadcast_complex"]["ok"] and not r["ddp_construction"]["ok"]
        and _PREMISE_ERROR in r["broadcast_complex"]["error"] and _PREMISE_ERROR in r["ddp_construction"]["error"]
        for r in ranks)
    complex_params = {n: list(p.shape) for n, p in _module().named_parameters() if p.is_complex()}
    return {"ranks": ranks, "complex_parameters": complex_params, "as_recorded": as_recorded}


def _parametrize_report() -> dict:
    import torch.nn.utils.parametrize as parametrize

    class _AsComplex(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.view_as_complex(x)

        def right_inverse(self, z: torch.Tensor) -> list:
            return [torch.view_as_real(z).clone()]

    module = _module()
    holder = next(m for m in module.modules() if isinstance(m._parameters.get("tensor"), torch.Tensor)
                  and m._parameters["tensor"].is_complex())
    outcome = _outcome(lambda: parametrize.register_parametrization(holder, "tensor", _AsComplex(),
                                                                     unsafe=True) and None)
    outcome["holder_type"] = f"{type(holder).__module__}.{type(holder).__qualname__}"
    outcome["as_recorded"] = (not outcome["ok"]) and outcome.get("error_type") == "TypeError"
    return outcome


def _single_process(optimizer: str) -> torch.Tensor:
    module = _module()
    train, val = _data()
    with tempfile.TemporaryDirectory() as tmp:
        _runner(Path(tmp), module, DataLoader(train, batch_size=_WORLD * _PER_RANK),
                DataLoader(val, batch_size=len(val)), optimizer).run()
    return _flat(module).double()


def _exclude_and_average_report() -> dict:
    mask = _complex_mask(_module())
    report = {}
    for optimizer in ("sgd", "adam"):
        ranks = _job("option", optimizer)
        entry = {"ranks_ok": [r["run"]["ok"] for r in ranks]}
        if all(r["run"]["ok"] for r in ranks):
            r0, r1 = (torch.tensor(r["run"]["value"], dtype=torch.float64) for r in ranks)
            single = _single_process(optimizer)
            diff = (r0 - single).abs()
            entry.update(ranks_bitwise_equal=bool(torch.equal(r0, r1)),
                         max_abs_vs_single=float(diff.max()),
                         max_abs_vs_single_complex=float(diff[mask].max()),
                         max_abs_vs_single_real=float(diff[~mask].max()),
                         max_abs_weight=float(single.abs().max()))
        else:
            entry["errors"] = [r["run"].get("error") for r in ranks]
        report[optimizer] = entry
    sgd = report["sgd"]
    report["holds"] = bool(all(report[o].get("ranks_bitwise_equal") for o in ("sgd", "adam"))
                           and sgd.get("max_abs_vs_single", float("inf")) <= _SGD_TOLERANCE)
    return report


def build_report() -> dict:
    return {
        "source": _source_revision(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": numpy.__version__,
                        "neuraloperator": importlib.metadata.version("neuraloperator"),
                        "platform": platform.platform(), "torch_threads": torch.get_num_threads()},
        "data": "synthetic standard-normal tensors on the N = 32 triangle lattice; no T1 data",
        "gino_config": _CONFIG.canonical(),
        "setup": {"world": _WORLD, "per_rank_batch": _PER_RANK, "epochs": _EPOCHS, "learning_rate": _LR,
                  "seed": _SEED, "sgd_tolerance": _SGD_TOLERANCE},
        "premise": _premise_report(),
        "parametrize": _parametrize_report(),
        "exclude_and_average": _exclude_and_average_report(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the JSON report here (default: stdout)")
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    report = build_report()
    text = json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    checks = {
        "backend_is_2.0.0": report["environment"]["neuraloperator"] == "2.0.0",
        "premise_as_recorded": report["premise"]["as_recorded"],
        "parametrize_refused_as_recorded": report["parametrize"]["as_recorded"],
        "exclude_and_average_holds": report["exclude_and_average"]["holds"],
    }
    for name, passed in checks.items():
        print(f"[GINO-DDP] {name:34s} {'PASS' if passed else 'FAIL'}", file=sys.stderr)
    ok = _report_source_binding(report["source"], "GINO-DDP") and all(checks.values())
    print(f"[GINO-DDP] {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
