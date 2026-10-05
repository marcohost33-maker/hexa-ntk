#!/usr/bin/env python3
"""Two-rank DCP bundle commit/digest oracle with negative controls."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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

_SHA = re.compile(r"[0-9a-f]{40}")


def resolve_source_sha(explicit: str | None = None) -> str:
    """Revision the bundle is produced from (Codex #54 T0).

    Order: explicit argument, GITHUB_SHA (the checked-out workflow revision),
    then `git rev-parse HEAD`. Fails closed if none yields a 40-hex SHA.
    """
    candidates = [explicit, os.environ.get("GITHUB_SHA")]
    if not any(candidates):
        try:
            candidates.append(subprocess.run(
                ["git", "rev-parse", "HEAD"], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, check=True,
                cwd=Path(__file__).resolve().parent,
            ).stdout.strip())
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("cannot resolve source SHA (no --source-sha, GITHUB_SHA or git)") from exc
    value = next((c for c in candidates if c), "")
    if not _SHA.fullmatch(value):
        raise RuntimeError(f"source SHA must be 40 lowercase hex characters, got {value!r}")
    return value


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def worker(rank: int, init_file: str, bundle_dir: str, source_sha: str) -> None:
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
                    "source_sha": source_sha,
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


def run(source_sha: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="hexantk_bundle_") as temp_dir:
        root = Path(temp_dir)
        bundle_dir = root / "bundle"
        bundle_dir.mkdir()
        mp.spawn(
            worker,
            args=(str(root / "init"), str(bundle_dir), source_sha),
            nprocs=2,
            join=True,
        )
        good = validate_bundle(bundle_dir, expected_source_sha=source_sha)

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
            lambda: validate_bundle(bundle_dir, expected_source_sha="0" * 40)
        )

        writing = root / "writing"
        shutil.copytree(bundle_dir, writing)
        marker = json.loads((writing / "COMMITTED.json").read_text(encoding="utf-8"))
        marker["state"] = "WRITING"
        (writing / "COMMITTED.json").write_text(json.dumps(marker), encoding="utf-8")
        not_committed = rejected(lambda: validate_bundle(writing))

    negatives = [missing_commit, tampered, wrong_sha, not_committed]
    return {
        "valid_bundle": good["valid"],
        "manifest_file_count": len(good["manifest"]["files"]),
        "missing_commit_rejected": missing_commit,
        "tampered_shard_rejected": tampered,
        "source_sha": source_sha,
        "source_sha_mismatch_rejected": wrong_sha,
        "non_committed_state_rejected": not_committed,
        "all_negative_controls_rejected": all(
            "UNEXPECTED_PASS" not in item for item in negatives
        ),
        "claim_ceiling": (
            "shared-filesystem two-rank CPU/gloo DCP commit/digest protocol; "
            "no object-store transactional guarantee"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", default=None,
                        help="revision to record; default GITHUB_SHA, then git HEAD")
    args = parser.parse_args()
    result = run(resolve_source_sha(args.source_sha))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid_bundle"] and result["all_negative_controls_rejected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
