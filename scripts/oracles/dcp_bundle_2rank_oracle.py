#!/usr/bin/env python3
"""Two-rank DCP bundle commit/digest oracle with negative controls."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.nn.parallel import DistributedDataParallel as DDP

from distributed_checkpoint_bundle_v1 import finalize_bundle, validate_bundle

MAIN_SHA = "e4a0e5e5f402f4b1df67443e090d6d2276fc8aba"


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def worker(rank: int, init_file: str, bundle_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(77)
        model = DDP(Tiny())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        generator = torch.Generator().manual_seed(100 + rank)
        x = torch.randn(8, 3, generator=generator, dtype=torch.float64)
        y = torch.randn(8, 2, generator=generator, dtype=torch.float64)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x) - y) ** 2)
            loss.backward()
            optimizer.step()

        model_state, optim_state = get_state_dict(model, optimizer)
        dcp.save(
            {"model": model_state, "optim": optim_state},
            checkpoint_id=Path(bundle_dir) / "dcp",
        )
        dist.barrier()
        if rank == 0:
            finalize_bundle(
                Path(bundle_dir),
                {
                    "main_sha": MAIN_SHA,
                    "world_size": 2,
                    "backend": "gloo",
                    "torch_version": str(torch.__version__),
                    "loader_state_protocol": "separate_committed_cursor_v1",
                },
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def rejected(callable_obj) -> str:
    try:
        callable_obj()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return "UNEXPECTED_PASS"


def run() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="hexantk_bundle_") as temp_dir:
        root = Path(temp_dir)
        bundle_dir = root / "bundle"
        bundle_dir.mkdir()
        mp.spawn(
            worker,
            args=(str(root / "init"), str(bundle_dir)),
            nprocs=2,
            join=True,
        )
        good = validate_bundle(bundle_dir, expected_main_sha=MAIN_SHA)

        partial = root / "partial"
        shutil.copytree(bundle_dir, partial)
        (partial / "COMMITTED.json").unlink()
        missing_commit = rejected(lambda: validate_bundle(partial))

        corrupt = root / "corrupt"
        shutil.copytree(bundle_dir, corrupt)
        candidates = [
            path
            for path in (corrupt / "dcp").iterdir()
            if path.is_file() and path.name != ".metadata"
        ]
        target = candidates[0] if candidates else corrupt / "dcp" / ".metadata"
        with target.open("ab") as stream:
            stream.write(b"X")
        tampered = rejected(lambda: validate_bundle(corrupt))

        wrong_sha = rejected(
            lambda: validate_bundle(bundle_dir, expected_main_sha="0" * 40)
        )

    negatives = [missing_commit, tampered, wrong_sha]
    return {
        "valid_bundle": good["valid"],
        "manifest_file_count": len(good["manifest"]["files"]),
        "missing_commit_rejected": missing_commit,
        "tampered_shard_rejected": tampered,
        "main_sha_mismatch_rejected": wrong_sha,
        "all_negative_controls_rejected": all(
            "UNEXPECTED_PASS" not in item for item in negatives
        ),
        "claim_ceiling": (
            "shared-filesystem two-rank CPU/gloo DCP commit/digest protocol; "
            "no object-store transactional guarantee"
        ),
    }


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid_bundle"] and result["all_negative_controls_rejected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
