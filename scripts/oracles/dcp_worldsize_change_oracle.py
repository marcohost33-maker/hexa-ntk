#!/usr/bin/env python3
"""DCP topology-change control for replicated model + Adam state: 2->1 and 1->2."""
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

from dcp_2rank_gloo_oracle import nested_equal


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 8, dtype=torch.float64),
            torch.nn.GELU(),
            torch.nn.Linear(8, 2, dtype=torch.float64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def rank_data(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(555 + rank)
    return (
        torch.randn(10, 4, generator=generator, dtype=torch.float64),
        torch.randn(10, 2, generator=generator, dtype=torch.float64),
    )


def step(model, optimizer, x: torch.Tensor, y: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = torch.mean((model(x) - y) ** 2)
    loss.backward()
    optimizer.step()


def save_two_rank(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
    reference_path: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(71)
        model = DDP(Tiny())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        x, y = rank_data(rank)
        for _ in range(4):
            step(model, optimizer, x, y)
        model_state, optim_state = get_state_dict(model, optimizer)
        dcp.save(
            {"model": model_state, "optim": optim_state},
            checkpoint_id=Path(checkpoint_dir),
        )
        if rank == 0:
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "optim": optimizer.state_dict(),
                },
                reference_path,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def load_one_from_two(checkpoint_dir: Path, reference_path: Path) -> dict[str, bool]:
    reference = torch.load(reference_path, weights_only=True)
    torch.manual_seed(999)
    model = Tiny()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    model_state, optim_state = get_state_dict(model, optimizer)
    payload = {"model": model_state, "optim": optim_state}
    dcp.load(payload, checkpoint_id=checkpoint_dir)
    set_state_dict(
        model,
        optimizer,
        model_state_dict=payload["model"],
        optim_state_dict=payload["optim"],
    )
    return {
        "model_equal": nested_equal(reference["model"], model.state_dict()),
        "optimizer_equal": nested_equal(reference["optim"], optimizer.state_dict()),
    }


def save_one(checkpoint_dir: Path, reference_path: Path) -> None:
    torch.manual_seed(123)
    model = Tiny()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    x, y = rank_data(0)
    for _ in range(4):
        step(model, optimizer, x, y)
    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_state, "optim": optim_state},
        checkpoint_id=checkpoint_dir,
    )
    torch.save(
        {"model": model.state_dict(), "optim": optimizer.state_dict()},
        reference_path,
    )


def load_two_rank(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
    reference_path: str,
    result_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        reference = torch.load(reference_path, weights_only=True)
        torch.manual_seed(1000 + rank)
        model = DDP(Tiny())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        model_state, optim_state = get_state_dict(model, optimizer)
        payload = {"model": model_state, "optim": optim_state}
        dcp.load(payload, checkpoint_id=Path(checkpoint_dir))
        set_state_dict(
            model,
            optimizer,
            model_state_dict=payload["model"],
            optim_state_dict=payload["optim"],
        )
        result = {
            "rank": rank,
            "model_equal": nested_equal(reference["model"], model.module.state_dict()),
            "optimizer_equal": nested_equal(
                reference["optim"],
                optimizer.state_dict(),
            ),
        }
        result["all_exact"] = result["model_equal"] and result["optimizer_equal"]
        Path(result_dir, f"rank{rank}.json").write_text(
            json.dumps(result, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hexantk_worldsize_") as temp_dir:
        root = Path(temp_dir)

        two_checkpoint = root / "checkpoint_two"
        two_reference = root / "reference_two.pt"
        mp.spawn(
            save_two_rank,
            args=(
                2,
                str(root / "init_two"),
                str(two_checkpoint),
                str(two_reference),
            ),
            nprocs=2,
            join=True,
        )
        two_to_one = load_one_from_two(two_checkpoint, two_reference)
        two_to_one["all_exact"] = (
            two_to_one["model_equal"] and two_to_one["optimizer_equal"]
        )

        one_checkpoint = root / "checkpoint_one"
        one_reference = root / "reference_one.pt"
        save_one(one_checkpoint, one_reference)
        result_dir = root / "results_one_to_two"
        result_dir.mkdir()
        mp.spawn(
            load_two_rank,
            args=(
                2,
                str(root / "init_one_to_two"),
                str(one_checkpoint),
                str(one_reference),
                str(result_dir),
            ),
            nprocs=2,
            join=True,
        )
        rank_results = [
            json.loads((result_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
            for rank in range(2)
        ]
        one_to_two = {
            "rank_results": rank_results,
            "all_ranks_exact": all(item["all_exact"] for item in rank_results),
        }

    return {
        "torch_version": str(torch.__version__),
        "backend": "gloo",
        "two_rank_ddp_to_one_plain": two_to_one,
        "one_plain_to_two_rank_ddp": one_to_two,
        "all_exact": two_to_one["all_exact"] and one_to_two["all_ranks_exact"],
        "claim_ceiling": (
            "single-node CPU/gloo topology-change control for replicated DDP/plain "
            "model+Adam; not FSDP sharded resharding, elastic training, GPU/NCCL, or loader state"
        ),
    }


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
