#!/usr/bin/env python3
"""ddp_crash_matrix_oracle.py - crash/stall fault matrix of the real runner under DDP (gloo).

What it measures (spec/DDP_RUNNER_v1.md section 8, 007acc2 A7 R2-HX107-01):
``HexTrainingRunner`` itself (not a model of its collectives) runs as a real
2- or 3-rank gloo job on one host. One rank is made to die without cleanup
(``os._exit``, as after SIGKILL or an OOM kill) or to stall (alive, silent) at
one of seven points of the second epoch. For every case the parent checks:

  I1 bounded end   every surviving rank ends by itself before the deadline
                   (process-group timeout + slack); none has to be killed.
  I2 no success    every surviving rank ends in an exception; none reports a
                   completed run.
  I3 checkpoint    a ``resume_last.pt`` left behind loads, and it is from an
                   epoch every rank had finished or was just saving.
  I4 recovery      the whole job restarted in the same output directory from
                   that checkpoint (all ranks, no fault) runs to the end; the
                   ranks' weights, EMA, learning rate and step count are
                   bitwise equal to each other, and equal to the uninterrupted
                   run of the same job: bitwise with 2 ranks, within
                   REL_TOL with 3 (see the premise below).

A case whose fault never fired is a failure of the oracle, not a pass.

Premise, measured by this script in every run (``allreduce_order_premise``): with
3 or more ranks gloo's all-reduce rounds the same per-rank values differently
depending on where they lie in the buffer (the ranks are added in another order;
with 2 ranks the sum is commutative), and DDP rebuilds its gradient buckets after
the first iteration of a process. The first step of a resumed job therefore
all-reduces with another layout than the same step of the uninterrupted job: the
gradient differs by an ulp, the ranks still agree bitwise with each other. A
resumed 3-rank run cannot be bitwise equal to the uninterrupted one; that is a
property of DDP + gloo, not of the checkpoint.

Limits (named, not covered): gloo on one host and CPU only - no NCCL, no CUDA,
no second host, no network partition; a stalled rank is a sleeping process, not
a frozen kernel thread; a crash inside backward (inside DDP's gradient
all-reduce) is not injected; the faults hit one rank, once.

Run (Python >= 3.12, torch 2.6.0; about 80 short jobs):
    python scripts/oracles/ddp_crash_matrix_oracle.py [--detail] [--json OUT] [--jobs N]
``--detail`` runs every job under TORCH_DISTRIBUTED_DEBUG=DETAIL.
Exit codes: 0 every case holds I1-I4; 1 a case violates one; 2 a fault did not fire
or a reference run failed (the oracle proves nothing then).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent.parent
TOOLKIT = REPO / "src" / "toolkit"

EPOCHS = 4
TIMEOUT_S = 15  # process-group timeout of every job
REL_TOL = 1e-9  # I4 with >= 3 ranks (measured: about 1e-15 after 4 resumed steps)
SLACK_S = 60  # start-up (torch import on every rank) + slow CI hosts
FAULT_EXIT_CODE = 41

# point -> (runner method, fire on "entry" or "exit" of its n-th call, n).
# 2 training steps per epoch and rank, so call 4 of a step method and call 2 of
# an epoch method are in the second epoch: a checkpoint of the first exists.
POINTS: dict[str, tuple[str, str, int]] = {
    "before_step": ("_train_one_step", "entry", 4),
    "after_backward_before_step_agreement": ("_apply_update", "entry", 4),
    "after_step_agreement_before_optimizer_step": ("_step_optimizer", "entry", 4),
    "in_validation": ("_validate_local", "entry", 2),
    "before_epoch_end_state_agreement": ("_agree_shared_state", "entry", 2),
    "inside_agreed_checkpoint_save_before_write": ("_save_checkpoints_local", "entry", 2),
    "inside_agreed_checkpoint_save_after_write": ("_save_checkpoints_local", "exit", 2),
}
# a stall costs the full timeout per case; two points, the last rank
STALL_POINTS = ("before_step", "before_epoch_end_state_agreement")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# one rank
# ---------------------------------------------------------------------------


def probe_main(args: argparse.Namespace) -> None:
    """The premise: all-reduce rounding by buffer position, DDP gradient by iteration."""
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    rank, world = args.rank, args.world
    dist.init_process_group(
        "gloo", init_method=Path(args.rendezvous).as_uri(), rank=rank, world_size=world,
        timeout=timedelta(seconds=TIMEOUT_S),
    )
    g = torch.Generator().manual_seed(100 + rank)
    local = torch.randn(7, generator=g, dtype=torch.float64)
    sums = []
    for size, offset in ((7, 0), (58, 0), (58, 20), (58, 51), (1000, 500), (1000, 993)):
        buffer = torch.zeros(size, dtype=torch.float64)
        buffer[offset:offset + 7] = local
        dist.all_reduce(buffer)
        sums.append(buffer[offset:offset + 7].numpy().tobytes().hex())
    torch.manual_seed(0)
    model = DDP(nn.Sequential(nn.Linear(3, 8, dtype=torch.float64), nn.Tanh(),
                              nn.Linear(8, 2, dtype=torch.float64)))
    start = {k: v.clone() for k, v in model.state_dict().items()}
    g = torch.Generator().manual_seed(200 + rank)
    x = torch.randn(2, 3, generator=g, dtype=torch.float64)
    y = torch.randn(2, 2, generator=g, dtype=torch.float64)
    grads = []
    for _ in range(4):  # the same weights and data every time
        model.load_state_dict(start)
        model.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        grads.append(torch.cat([p.grad.reshape(-1) for p in model.parameters()]).clone())
    _write_json(Path(args.result), {
        "rank": rank, "outcome": "completed", "allreduce_sums": sums,
        "ddp_gradient_equal_to_first_iteration": [bool(torch.equal(grads[0], gr)) for gr in grads],
        "ddp_gradient_max_rel_diff_to_first_iteration": [
            float(((grads[0] - gr).abs() / grads[0].abs().clamp_min(1e-300)).max()) for gr in grads],
    })
    dist.destroy_process_group()


def rank_main(args: argparse.Namespace) -> None:
    if args.probe:
        probe_main(args)
        return
    sys.path.insert(0, str(TOOLKIT))
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset, DistributedSampler

    from ddp_hex_v3 import ExactDistributedEvalSampler
    from runner_hex_v3 import HexTrainingRunner, RunnerConfig

    class Data(Dataset):
        def __init__(self, n: int, seed: int) -> None:
            g = torch.Generator().manual_seed(seed)
            self.x = torch.randn(n, 3, generator=g, dtype=torch.float64)
            w = torch.tensor([[0.5, -1.0], [2.0, 0.25], [-0.75, 1.5]], dtype=torch.float64)
            self.y = self.x @ w + 0.1 * torch.randn(n, 2, generator=g, dtype=torch.float64)

        def __len__(self) -> int:
            return len(self.x)

        def __getitem__(self, i: int) -> dict[str, Any]:
            return {"input": self.x[i], "target": self.y[i]}

    rank, world, out = args.rank, args.world, Path(args.out)
    result_path = Path(args.result)
    dist.init_process_group(
        "gloo", init_method=Path(args.rendezvous).as_uri(), rank=rank, world_size=world,
        timeout=timedelta(seconds=TIMEOUT_S),
    )
    torch.manual_seed(0)
    model = DDP(nn.Sequential(nn.Linear(3, 8, dtype=torch.float64), nn.Tanh(),
                              nn.Linear(8, 2, dtype=torch.float64)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    train = Data(8, seed=1)
    val = Data(5, seed=2)
    config = RunnerConfig(
        output_dir=str(out), num_epochs=EPOCHS, log_every_n_steps=0, ema_decay=0.9,
        best_metric_min_delta=0.0, distributed=True, is_main_process=rank == 0,
        resume_from=args.resume or None, resume_trusted=bool(args.resume),
    )
    runner = HexTrainingRunner(
        config, model, optimizer,
        DataLoader(train, batch_size=2, sampler=DistributedSampler(train, world, rank, shuffle=True, seed=7)),
        DataLoader(val, batch_size=2, sampler=ExactDistributedEvalSampler(val)),
        scheduler=scheduler,
    )

    if args.fault_point and args.fault_rank == rank:
        name, when, nth = POINTS[args.fault_point]
        original = getattr(runner, name)
        calls = [0]

        def fire() -> None:
            _write_json(Path(args.marker), {"point": args.fault_point, "global_step": runner.global_step,
                                            "time": time.time()})
            if args.fault_kind == "exit":
                os._exit(FAULT_EXIT_CODE)  # no cleanup, no atexit: sockets close as after SIGKILL
            time.sleep(10 ** 6)  # stall: alive and silent until the parent kills it

        def faulty(*a: Any, **k: Any) -> Any:
            calls[0] += 1
            if when == "entry" and calls[0] == nth:
                fire()
            value = original(*a, **k)
            if when == "exit" and calls[0] == nth:
                fire()
            return value

        setattr(runner, name, faulty)

    def digest(module: nn.Module) -> str:
        h = hashlib.sha256()
        for key, tensor in sorted(module.state_dict().items()):
            h.update(key.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    record: dict[str, Any] = {"rank": rank}
    try:
        runner.run()
        record["outcome"] = "completed"
    except Exception as exc:  # the outcome under test
        record.update(outcome="error", error=type(exc).__name__, message=str(exc)[:400])
    def values(module: nn.Module) -> list[float]:
        return [x for _, t in sorted(module.state_dict().items()) for x in t.detach().reshape(-1).tolist()]

    record.update(
        time=time.time(), global_step=runner.global_step,
        weights=digest(runner.raw_model),
        ema=digest(runner.ema_model.module) if runner.ema_model is not None else None,
        lr=[group["lr"] for group in optimizer.param_groups],
        values=values(runner.raw_model) + (values(runner.ema_model.module) if runner.ema_model is not None else []),
    )
    _write_json(result_path, record)
    # no destroy_process_group: with a dead peer it may wait; the result is on disk
    os._exit(0 if record["outcome"] == "completed" else 3)


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------


def run_job(job_dir: Path, out: Path, world: int, *, fault: tuple[str, int, str] | None = None,
            resume: Path | None = None, detail: bool = False, probe: bool = False) -> dict[str, Any]:
    """One WORLD-rank job; returns per-rank exit codes, wall times and result records."""
    job_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("TORCH_DISTRIBUTED_DEBUG", None)
    if detail:
        env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    marker = job_dir / "fault.json"
    procs = []
    start = time.time()
    for rank in range(world):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--rank", str(rank), "--world", str(world),
               "--out", str(out), "--rendezvous", str(job_dir / "rendezvous"),
               "--result", str(job_dir / f"rank{rank}.json"), "--marker", str(marker)]
        if fault is not None:
            cmd += ["--fault-point", fault[0], "--fault-rank", str(fault[1]), "--fault-kind", fault[2]]
        if resume is not None:
            cmd += ["--resume", str(resume)]
        if probe:
            cmd += ["--probe"]
        log = open(job_dir / f"rank{rank}.log", "w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    deadline = start + TIMEOUT_S + SLACK_S
    stalled = fault[1] if fault is not None and fault[2] == "stall" else None
    ended: dict[int, float] = {}
    while time.time() < deadline:
        for rank, (proc, _) in enumerate(procs):
            if rank not in ended and proc.poll() is not None:
                ended[rank] = time.time()
        if all(rank in ended for rank in range(world) if rank != stalled):
            break
        time.sleep(0.05)
    ranks = []
    for rank, (proc, log) in enumerate(procs):
        killed = proc.poll() is None
        if killed:
            proc.kill()
            proc.wait()
        log.close()
        try:
            record = json.loads((job_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
        ranks.append({"rank": rank, "returncode": proc.returncode, "killed_by_parent": killed,
                      "wall_s": round(ended.get(rank, time.time()) - start, 3), "record": record})
    fired = None
    if marker.exists():
        fired = json.loads(marker.read_text(encoding="utf-8"))
    return {"ranks": ranks, "fault_fired": fired}


def _checkpoint_epoch(path: Path) -> int | None:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    return int(state["epoch"]) if isinstance(state, dict) and "epoch" in state else None


def run_case(root: Path, world: int, point: str, fault_rank: int, kind: str, reference: dict[str, Any],
             detail: bool) -> dict[str, Any]:
    name = f"w{world}_{point}_rank{fault_rank}_{kind}"
    case_dir = root / name
    out = case_dir / "run"
    faulted = run_job(case_dir / "faulted", out, world, fault=(point, fault_rank, kind), detail=detail)
    case: dict[str, Any] = {"case": name, "world": world, "point": point, "fault_rank": fault_rank,
                            "kind": kind, "violations": []}
    fired = faulted["fault_fired"]
    if fired is None:
        case["violations"].append("ORACLE: the fault did not fire")
        case["status"] = "NOT_FIRED"
        return case
    case["fault_global_step"] = fired["global_step"]
    survivors = [r for r in faulted["ranks"] if r["rank"] != fault_rank]
    faulty = faulted["ranks"][fault_rank]
    if kind == "exit" and faulty["returncode"] != FAULT_EXIT_CODE:
        case["violations"].append(f"ORACLE: faulty rank ended with {faulty['returncode']}")
    errors = []
    latencies = []
    for r in survivors:
        if r["killed_by_parent"]:
            case["violations"].append(f"I1: rank {r['rank']} did not end within {TIMEOUT_S + SLACK_S} s (hang)")
            continue
        record = r["record"]
        if record is None:
            case["violations"].append(f"I2: rank {r['rank']} ended ({r['returncode']}) without a result")
            continue
        if record["outcome"] != "error":
            case["violations"].append(f"I2: rank {r['rank']} reported a completed run after its peer's fault")
            continue
        if record["weights"] == reference["weights"]:  # sensitivity control of the I4 comparison
            case["violations"].append(f"ORACLE: rank {r['rank']} stopped early with the reference's final weights")
        errors.append(record["error"])
        latencies.append(round(record["time"] - fired["time"], 3))
    case["survivor_errors"] = sorted(errors)
    case["survivor_end_after_fault_s"] = latencies
    case["survivor_global_steps"] = [r["record"]["global_step"] for r in survivors if r["record"]]

    checkpoint = out / "checkpoints" / "resume_last.pt"
    resume = None
    if checkpoint.exists():
        try:
            epoch = _checkpoint_epoch(checkpoint)
        except Exception as exc:
            epoch = None
            case["violations"].append(f"I3: resume_last.pt does not load: {type(exc).__name__}: {exc}")
        case["checkpoint_epoch"] = epoch
        # the fault is in the second epoch (index 1): epoch 0 is finished. Epoch 1 only
        # if rank 0 reached its write: the fault came after it, or it hit another rank
        # inside the agreed save, while rank 0 was writing
        in_save = point.startswith("inside_agreed_checkpoint_save")
        allowed = (0, 1) if in_save and (fault_rank != 0 or point.endswith("after_write")) else (0,)
        if epoch is not None and epoch not in allowed:
            case["violations"].append(f"I3: resume_last.pt is from epoch {epoch}, allowed {allowed}")
        resume = checkpoint
    else:
        case["checkpoint_epoch"] = None
        case["violations"].append("I3: no resume_last.pt although the first epoch was completed")

    recovered = run_job(case_dir / "recovery", out, world, resume=resume, detail=detail)
    states = set()
    values: list[float] | None = None
    for r in recovered["ranks"]:
        record = r["record"]
        if r["killed_by_parent"] or record is None or record["outcome"] != "completed":
            detail_text = (record or {}).get("error", "no result")
            case["violations"].append(f"I4: recovery rank {r['rank']} did not complete ({detail_text}: "
                                      f"{(record or {}).get('message', '')[:160]})")
            continue
        states.add((record["weights"], record["ema"], tuple(record["lr"]), record["global_step"]))
        values = record["values"]
    expected = (reference["weights"], reference["ema"], tuple(reference["lr"]), reference["global_step"])
    case["recovered_ranks_bitwise_equal"] = len(states) == 1
    case["recovered_bitwise_equal_to_reference"] = states == {expected}
    case["recovered_max_rel_diff_to_reference"] = None
    if len(states) > 1:
        case["violations"].append("I4: the recovered ranks ended in different states")
    elif states:
        state = next(iter(states))
        case["recovered_max_rel_diff_to_reference"] = _max_rel_diff(values or [], reference["values"])
        if state[2:] != expected[2:]:
            case["violations"].append("I4: recovered learning rate or step count differ from the uninterrupted run")
        elif world == 2 and state != expected:
            case["violations"].append("I4: recovered weights/EMA are not bitwise the uninterrupted run's (2 ranks)")
        elif case["recovered_max_rel_diff_to_reference"] > REL_TOL:
            case["violations"].append(f"I4: recovered weights/EMA differ from the uninterrupted run by "
                                      f"{case['recovered_max_rel_diff_to_reference']:.3g} > {REL_TOL}")
    case["status"] = "PASS" if not case["violations"] else "FAIL"
    return case


def _max_rel_diff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return float("inf")
    return max(abs(x - y) / max(abs(y), 1e-300) for x, y in zip(a, b))


def premise_run(root: Path, world: int, detail: bool) -> dict[str, Any]:
    job = run_job(root / f"premise_w{world}", root / f"premise_w{world}" / "unused", world, detail=detail, probe=True)
    records = [r["record"] for r in job["ranks"]]
    if any(r is None for r in records):
        raise RuntimeError(f"premise probe with {world} ranks failed: {job}")
    first = records[0]
    return {
        "ranks_agree_bitwise": all(r["allreduce_sums"] == first["allreduce_sums"] for r in records),
        "allreduce_distinct_results_over_6_buffer_positions": len(set(first["allreduce_sums"])),
        "ddp_gradient_equal_to_first_iteration": first["ddp_gradient_equal_to_first_iteration"],
        "ddp_gradient_max_rel_diff_to_first_iteration": first["ddp_gradient_max_rel_diff_to_first_iteration"],
    }


def reference_run(root: Path, world: int, detail: bool) -> dict[str, Any]:
    job = run_job(root / f"reference_w{world}" / "job", root / f"reference_w{world}" / "run", world, detail=detail)
    records = [r["record"] for r in job["ranks"]]
    if any(r is None or r["outcome"] != "completed" for r in records):
        raise RuntimeError(f"reference run with {world} ranks failed: {job}")
    states = {(r["weights"], r["ema"], tuple(r["lr"]), r["global_step"]) for r in records}
    if len(states) != 1:
        raise RuntimeError(f"reference run with {world} ranks: the ranks ended in different states")
    first = records[0]
    return {"weights": first["weights"], "ema": first["ema"], "lr": first["lr"],
            "global_step": first["global_step"], "values": first["values"]}


def matrix() -> list[tuple[int, str, int, str]]:
    cases = []
    for world in (2, 3):
        for point in POINTS:
            for fault_rank in range(world):
                cases.append((world, point, fault_rank, "exit"))
        for point in STALL_POINTS:
            cases.append((world, point, world - 1, "stall"))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world", type=int)
    parser.add_argument("--out")
    parser.add_argument("--rendezvous")
    parser.add_argument("--result")
    parser.add_argument("--marker")
    parser.add_argument("--fault-point", default="")
    parser.add_argument("--fault-rank", type=int, default=-1)
    parser.add_argument("--fault-kind", default="exit", choices=("exit", "stall"))
    parser.add_argument("--resume", default="")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--json", default="")
    parser.add_argument("--jobs", type=int, default=2, help="cases run at once")
    args = parser.parse_args()
    if args.rank is not None:
        rank_main(args)
        return 0

    import numpy
    import torch

    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            references = {world: reference_run(root, world, args.detail) for world in (2, 3)}
            premise = {world: premise_run(root, world, args.detail) for world in (2, 3)}
        except RuntimeError as exc:
            print(f"[DDP-CRASH-MATRIX] {exc} | FAIL")
            return 2
        cases = matrix()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            results = list(pool.map(
                lambda c: run_case(root, c[0], c[1], c[2], c[3], references[c[0]], args.detail), cases))
    for case in results:
        line = (f"{case['status']:<9} {case['case']} | survivors {','.join(case.get('survivor_errors', [])) or '-'}"
                f" after {case.get('survivor_end_after_fault_s', '-')} s | checkpoint epoch "
                f"{case.get('checkpoint_epoch')} | recovered: ranks bitwise equal "
                f"{case.get('recovered_ranks_bitwise_equal')}, bitwise the reference "
                f"{case.get('recovered_bitwise_equal_to_reference')}, max rel diff "
                f"{case.get('recovered_max_rel_diff_to_reference')}")
        print(line)
        for violation in case["violations"]:
            print(f"          {violation}")
    for world, found in premise.items():
        print(f"premise   {world} ranks: all-reduce gives {found['allreduce_distinct_results_over_6_buffer_positions']}"
              f" distinct results over 6 buffer positions (ranks agree: {found['ranks_agree_bitwise']}); DDP gradient"
              f" of iteration 1-4 equal to iteration 1: {found['ddp_gradient_equal_to_first_iteration']}")
    passed = sum(case["status"] == "PASS" for case in results)
    not_fired = sum(case["status"] == "NOT_FIRED" for case in results)
    report = {
        "schema": "hexantk/ddp-crash-matrix/v1",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": numpy.__version__, "platform": platform.platform(),
                        "backend": "gloo", "hosts": 1, "cuda": torch.cuda.is_available(),
                        "torch_distributed_debug": "DETAIL" if args.detail else "OFF"},
        "job": {"epochs": EPOCHS, "process_group_timeout_s": TIMEOUT_S, "deadline_slack_s": SLACK_S,
                "points": {k: list(v) for k, v in POINTS.items()}},
        "rel_tol_3_ranks": REL_TOL,
        "allreduce_order_premise": {str(w): v for w, v in premise.items()},
        "references": {str(w): {k: v for k, v in ref.items() if k != "values"} for w, ref in references.items()},
        "cases": results,
        "summary": {"cases": len(results), "passed": passed, "not_fired": not_fired,
                    "wall_s": round(time.time() - started, 1)},
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    ok = passed == len(results)
    print(f"[DDP-CRASH-MATRIX] torch {torch.__version__}, gloo, debug "
          f"{'DETAIL' if args.detail else 'OFF'} | {passed}/{len(results)} cases hold I1-I4 | "
          f"{'PASS' if ok else 'FAIL'}")
    if not_fired:
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
