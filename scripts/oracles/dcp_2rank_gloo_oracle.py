#!/usr/bin/env python3
"""Two-rank CPU/gloo DDP + Distributed Checkpoint differential oracle."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.nn.parallel import DistributedDataParallel as DDP


def nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 8, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 2, dtype=torch.float64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def rank_data(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(700 + rank)
    x = torch.randn(12, 4, generator=generator, dtype=torch.float64)
    y = torch.randn(12, 2, generator=generator, dtype=torch.float64)
    return x, y


def step(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = torch.mean((model(x) - y) ** 2)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
    result_dir: str,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(100)
        model = DDP(Tiny())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x, y = rank_data(rank)
        for _ in range(3):
            step(model, optimizer, x, y)

        model_state, optim_state = get_state_dict(model, optimizer)
        dcp.save(
            {"model": model_state, "optim": optim_state},
            checkpoint_id=Path(checkpoint_dir),
        )
        dist.barrier()

        torch.manual_seed(999 + rank)
        restored = DDP(Tiny())
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
        restored_model_state, restored_optim_state = get_state_dict(
            restored,
            restored_optimizer,
        )
        payload = {
            "model": restored_model_state,
            "optim": restored_optim_state,
        }
        dcp.load(payload, checkpoint_id=Path(checkpoint_dir))
        set_state_dict(
            restored,
            restored_optimizer,
            model_state_dict=payload["model"],
            optim_state_dict=payload["optim"],
        )

        before_model = nested_equal(model.state_dict(), restored.state_dict())
        before_optimizer = nested_equal(
            optimizer.state_dict(),
            restored_optimizer.state_dict(),
        )
        loss_source = step(model, optimizer, x, y)
        loss_restored = step(restored, restored_optimizer, x, y)
        after_model = nested_equal(model.state_dict(), restored.state_dict())
        after_optimizer = nested_equal(
            optimizer.state_dict(),
            restored_optimizer.state_dict(),
        )
        result = {
            "rank": rank,
            "before_model_equal": before_model,
            "before_optimizer_equal": before_optimizer,
            "next_loss_equal": loss_source == loss_restored,
            "after_one_step_model_equal": after_model,
            "after_one_step_optimizer_equal": after_optimizer,
        }
        result["all_exact"] = all(result[key] for key in result if key != "rank")
        Path(result_dir, f"rank{rank}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run() -> dict[str, Any]:
    world_size = 2
    with tempfile.TemporaryDirectory(prefix="hexantk_dcp2_") as temp_dir:
        root = Path(temp_dir)
        result_dir = root / "results"
        result_dir.mkdir()
        mp.spawn(
            worker,
            args=(
                world_size,
                str(root / "pg_init"),
                str(root / "checkpoint"),
                str(result_dir),
            ),
            nprocs=world_size,
            join=True,
        )
        ranks = [
            json.loads((result_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
            for rank in range(world_size)
        ]
    return {
        "world_size": world_size,
        "backend": "gloo",
        "wrapper": "DistributedDataParallel",
        "torch_version": str(torch.__version__),
        "rank_results": ranks,
        "all_ranks_exact": all(item["all_exact"] for item in ranks),
        "claim_ceiling": (
            "two-rank single-node CPU/gloo DDP model+Adam DCP roundtrip and "
            "one-next-step differential; no FSDP, GPU/NCCL, elastic, or loader-state claim"
        ),
    }


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_ranks_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
