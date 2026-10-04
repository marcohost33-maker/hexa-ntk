#!/usr/bin/env python3
"""StatefulDataLoader mid-epoch resume differential oracle.

Claim ceiling: deterministic map-style Dataset, in_order=True, CPU. This checks
batch order plus model/optimizer/scheduler/EMA/loss continuity. It does not
claim rank-wise distributed recovery or stochastic worker-transform recovery.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader


class RegressionDataset(Dataset):
    def __init__(self, n: int = 64) -> None:
        generator = torch.Generator().manual_seed(314159)
        self.x = torch.randn(n, 5, generator=generator, dtype=torch.float64)
        weights = torch.tensor([[0.7], [-0.4], [0.2], [0.9], [-0.1]], dtype=torch.float64)
        self.y = self.x @ weights + 0.05

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input": self.x[index],
            "target": self.y[index],
            "index": torch.tensor(index),
        }


class TinyRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(5, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SimpleEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9) -> None:
        self.decay = decay
        self.state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            self.state[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "state": {key: value.clone() for key, value in self.state.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay mismatch")
        self.state = {key: value.clone() for key, value in state["state"].items()}


def nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def make_objects(seed: int):
    torch.manual_seed(seed)
    model = TinyRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1.0,
        total_iters=6,
    )
    ema = SimpleEMA(model)
    return model, optimizer, scheduler, ema


def make_loader(dataset: Dataset, *, seed: int, num_workers: int) -> StatefulDataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": 8,
        "shuffle": True,
        "num_workers": num_workers,
        "generator": torch.Generator().manual_seed(seed),
        "in_order": True,
        "snapshot_every_n_steps": 1,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return StatefulDataLoader(**kwargs)


def run_steps(model, optimizer, scheduler, ema, iterator, *, limit: int | None = None):
    indices: list[list[int]] = []
    losses: list[float] = []
    for batch_number, batch in enumerate(iterator):
        x = batch["input"]
        y = batch["target"]
        x = x + 1e-4 * torch.randn_like(x)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)
        indices.append([int(value) for value in batch["index"].tolist()])
        losses.append(float(loss.detach().item()))
        if limit is not None and batch_number + 1 >= limit:
            break
    return indices, losses


def run_oracle(num_workers: int, *, seed: int = 20261004, interrupt_after: int = 3) -> dict[str, Any]:
    dataset = RegressionDataset()

    base_model, base_opt, base_sched, base_ema = make_objects(seed)
    base_loader = make_loader(dataset, seed=seed, num_workers=num_workers)
    base_indices, base_losses = run_steps(
        base_model,
        base_opt,
        base_sched,
        base_ema,
        iter(base_loader),
    )

    int_model, int_opt, int_sched, int_ema = make_objects(seed)
    first_loader = make_loader(dataset, seed=seed, num_workers=num_workers)
    first_iter = iter(first_loader)
    prefix_indices, prefix_losses = run_steps(
        int_model,
        int_opt,
        int_sched,
        int_ema,
        first_iter,
        limit=interrupt_after,
    )
    loader_state = deepcopy(first_loader.state_dict())
    model_state = deepcopy(int_model.state_dict())
    optimizer_state = deepcopy(int_opt.state_dict())
    scheduler_state = deepcopy(int_sched.state_dict())
    ema_state = deepcopy(int_ema.state_dict())
    rng_state = torch.random.get_rng_state().clone()

    res_model, res_opt, res_sched, res_ema = make_objects(seed)
    resumed_loader = make_loader(dataset, seed=seed, num_workers=num_workers)
    resumed_loader.load_state_dict(loader_state)
    res_model.load_state_dict(model_state)
    res_sched.load_state_dict(scheduler_state)
    res_opt.load_state_dict(optimizer_state)
    res_ema.load_state_dict(ema_state)
    torch.random.set_rng_state(rng_state)

    suffix_indices, suffix_losses = run_steps(
        res_model,
        res_opt,
        res_sched,
        res_ema,
        iter(resumed_loader),
    )
    resumed_indices = prefix_indices + suffix_indices
    resumed_losses = prefix_losses + suffix_losses

    checks = {
        "same_batch_sequence": resumed_indices == base_indices,
        "same_losses": resumed_losses == base_losses,
        "same_model": nested_equal(res_model.state_dict(), base_model.state_dict()),
        "same_optimizer": nested_equal(res_opt.state_dict(), base_opt.state_dict()),
        "same_scheduler": nested_equal(res_sched.state_dict(), base_sched.state_dict()),
        "same_ema": nested_equal(res_ema.state_dict(), base_ema.state_dict()),
    }
    return {
        "torch_version": str(torch.__version__),
        "num_workers": num_workers,
        "interrupt_after": interrupt_after,
        **checks,
        "all_exact": all(checks.values()),
        "claim_ceiling": (
            "CPU, deterministic map-style dataset, in_order=True; no rank-wise distributed "
            "claim and no stochastic worker-transform RNG claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, choices=(0, 2), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_oracle(args.num_workers)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
