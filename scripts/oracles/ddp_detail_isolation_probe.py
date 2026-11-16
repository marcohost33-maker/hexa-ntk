"""Isolate the DETAIL SIGSEGV of 007acc2 R14 section 6: minimal gloo vs minimal DDP vs the runner.

Usage (repo root):
    TORCH_DISTRIBUTED_DEBUG=DETAIL python scripts/oracles/ddp_detail_isolation_probe.py \
        --level {gloo,ddp,runner} --world N [--toolkit src/toolkit]

The debug level comes from TORCH_DISTRIBUTED_DEBUG in the environment (inherited by the
spawned ranks). Prints exactly one JSON line with every rank's exit code; exit 0 only if
every rank exited 0 within the deadline. `--level ddp` also reports the GLOO backend class
before DDP and the `gloo_num_threads` DDP's logger read: torch's Logger::set_env_variables
static_casts the GLOO backend to ProcessGroupGloo, which under DETAIL is a
_ProcessGroupWrapper (results/2026-11-16_r14-ddp-detail-isolation.log).
Not run in CI: a diagnosis tool, evidence in the results log.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _level_gloo(rank: int, world: int, _root: Path) -> None:
    t = torch.tensor([float(rank + 1)])
    dist.all_reduce(t)
    assert t.item() == world * (world + 1) / 2, t


def _level_ddp(rank: int, world: int, _root: Path) -> None:
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.manual_seed(0)
    (_root / f"rank{rank}_pre.json").write_text(json.dumps({
        "gloo_backend_class_before_ddp": type(dist.group.WORLD._get_backend(torch.device("cpu"))).__name__}))
    model = DDP(torch.nn.Linear(4, 2))
    # torch's Logger::set_env_variables static_casts the GLOO backend to ProcessGroupGloo and
    # reads getNumThreads(); under DETAIL that backend is a ProcessGroupWrapper. Record what it read.
    data = model._get_ddp_logging_data()
    (_root / f"rank{rank}.json").write_text(json.dumps({
        "gloo_num_threads": data.get("gloo_num_threads"),
        "torch_distributed_debug": data.get("torch_distributed_debug"),
        "backend_name": data.get("backend_name"),
        "gloo_backend_class": type(dist.group.WORLD._get_backend(torch.device("cpu"))).__name__,
    }))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in range(3):
        x = torch.full((2, 4), float(rank + step))
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    ref = flat.clone()
    dist.broadcast(ref, 0)
    assert torch.equal(flat, ref), "replicas diverged"


def _level_runner(rank: int, world: int, root: Path) -> None:
    # the ORIGINAL #107 runner, imported from the checkout under test
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from ddp_hex_v3 import ExactDistributedEvalSampler
    from runner_hex_v3 import HexTrainingRunner, RunnerConfig

    class _Data(Dataset):
        def __init__(self, n: int, seed: int) -> None:
            g = torch.Generator().manual_seed(seed)
            self.x = torch.randn(n, 3, generator=g, dtype=torch.float64)
            self.y = self.x @ torch.ones(3, 2, dtype=torch.float64)

        def __len__(self) -> int:
            return len(self.x)

        def __getitem__(self, i: int):
            return {"input": self.x[i], "target": self.y[i]}

    torch.manual_seed(0)
    inner = torch.nn.Sequential(torch.nn.Linear(3, 8, dtype=torch.float64), torch.nn.Tanh(),
                                torch.nn.Linear(8, 2, dtype=torch.float64))
    model = DDP(inner)
    train = _Data(8, 1)
    val = _Data(5, 2)
    runner = HexTrainingRunner(
        RunnerConfig(output_dir=str(root / "out"), num_epochs=2, log_every_n_steps=0, ema_decay=0.9,
                     best_metric_min_delta=0.0, distributed=True, is_main_process=rank == 0),
        model,
        torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        DataLoader(train, batch_size=2, sampler=DistributedSampler(train, world, rank, shuffle=False)),
        DataLoader(val, batch_size=2, sampler=ExactDistributedEvalSampler(val)),
    )
    runner.run()


LEVELS = {"gloo": _level_gloo, "ddp": _level_ddp, "runner": _level_runner}


def _rank_main(rank: int, world: int, root: str, level: str, toolkit: str) -> None:
    faulthandler.enable()
    if toolkit:
        sys.path.insert(0, toolkit)
    dist.init_process_group("gloo", init_method=(Path(root) / "rendezvous").as_uri(), rank=rank,
                            world_size=world, timeout=timedelta(seconds=60))
    try:
        LEVELS[level](rank, world, Path(root))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=sorted(LEVELS), required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--toolkit", default="")
    ap.add_argument("--deadline", type=float, default=180.0)
    a = ap.parse_args()
    if a.world < 2:
        ap.error("--world must be >= 2")
    if a.level == "runner" and not a.toolkit:
        ap.error("--level runner needs --toolkit")
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as root:
        procs = [ctx.Process(target=_rank_main, args=(r, a.world, root, a.level, a.toolkit)) for r in range(a.world)]
        for p in procs:
            p.start()
        deadline = time.monotonic() + a.deadline
        for p in procs:
            p.join(max(0.0, deadline - time.monotonic()))
        hung = [i for i, p in enumerate(procs) if p.is_alive()]
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join()
        codes = [p.exitcode for p in procs]
        per_rank = []
        for r in range(a.world):
            f = Path(root) / f"rank{r}.json"
            pre = Path(root) / f"rank{r}_pre.json"
            entry = json.loads(f.read_text()) if f.exists() else {}
            if pre.exists():
                entry.update(json.loads(pre.read_text()))
            per_rank.append(entry or None)
    ok = not hung and all(c == 0 for c in codes)
    print(json.dumps({"level": a.level, "world": a.world, "debug": os.environ.get("TORCH_DISTRIBUTED_DEBUG", "<unset>"),
                      "torch": torch.__version__, "python": sys.version.split()[0], "exit_codes": codes,
                      "hung": hung, "ok": ok, "per_rank": per_rank}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
