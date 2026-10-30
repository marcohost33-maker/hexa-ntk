"""
test_ddp_hex_v3.py
==================
Tests for the distributed runner path (spec/DDP_RUNNER_v1.md).

Three layers:
- ExactDistributedEvalSampler and DistributedRunContext without a process
  group (argument checks, partition algebra);
- an in-process gloo group of world size 1: the distributed code path runs
  under coverage and must reproduce the single-process runner bit for bit;
- two real gloo ranks (torch.multiprocessing, file:// rendezvous): equality
  with a single-process reference run, and every agreed failure (padded
  validation sampler, NaN loss on one rank, checkpoint write failure on
  rank 0, unequal batch counts, resume file missing or different on one rank)
  ends on both ranks with an error instead of a hang.
"""
from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset

import numpy as np

from ddp_hex_v3 import (
    DistributedContractError,
    DistributedRunContext,
    ExactDistributedEvalSampler,
    PeerRankError,
    restore_rng_state_tensors,
    rng_state_tensors,
)
from operator_adapter_hex_v3 import OperatorAdapterV3
from runner_hex_v3 import HexTrainingRunner, RunnerConfig

# ------------------------------------------------------------------
# Fixtures shared by the in-process tests and the spawned ranks
# ------------------------------------------------------------------


class _Data(Dataset):
    """Deterministic float64 regression samples as dict batches."""

    def __init__(self, n: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, generator=g, dtype=torch.float64)
        w = torch.tensor([[0.5, -1.0], [2.0, 0.25], [-0.75, 1.5]], dtype=torch.float64)
        self.y = self.x @ w + 0.1 * torch.randn(n, 2, generator=g, dtype=torch.float64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {"input": self.x[i], "target": self.y[i]}


class _Stream(IterableDataset):
    def __iter__(self):
        yield {"input": torch.zeros(3, dtype=torch.float64), "target": torch.zeros(2, dtype=torch.float64)}


class _BN(nn.Module):
    """A model with buffers: DDP broadcasts them at the start of every forward."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(3, 8, dtype=torch.float64)
        self.bn = nn.BatchNorm1d(8, dtype=torch.float64)
        self.out = nn.Linear(8, 2, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.bn(self.lin(x))))


class _LyingStream(IterableDataset):
    """Declares 4 samples, yields ``n``."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return 4

    def __iter__(self):
        data = _Data(4, seed=1)
        for i in range(self.n):
            yield data[i]


class _NoTarget(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {"input": _Data(4, seed=1)[i]["input"]}


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(3, 8, dtype=torch.float64), nn.Tanh(), nn.Linear(8, 2, dtype=torch.float64)
    )


def _config(out: Path, **kw: Any) -> RunnerConfig:
    base = dict(
        output_dir=str(out),
        num_epochs=3,
        log_every_n_steps=0,
        ema_decay=0.9,
        best_metric_min_delta=0.0,
    )
    base.update(kw)
    return RunnerConfig(**base)


def _flat_params(module: nn.Module) -> List[float]:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()]).tolist()


def _single_process_reference(out: Path) -> Dict[str, Any]:
    """The run the 2-rank DDP job must reproduce: global batch 4, no sharding."""
    model = _model()
    runner = HexTrainingRunner(
        _config(out),
        model,
        torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        DataLoader(_Data(8, seed=1), batch_size=4),
        DataLoader(_Data(5, seed=2), batch_size=5),
    )
    result = runner.run()
    return {
        "params": _flat_params(model),
        "ema": _flat_params(runner.ema_model.module),
        "val_nrmse": [v["val_nrmse"] for v in result["history"]["val"]],
        "mean_loss": [t["mean_loss"] for t in result["history"]["train"]],
    }


# ------------------------------------------------------------------
# No process group
# ------------------------------------------------------------------


class TestExactDistributedEvalSampler(unittest.TestCase):
    def test_partition_is_exact_for_every_length_and_world_size(self) -> None:
        for n in range(0, 13):
            for world in range(1, 5):
                parts = [list(ExactDistributedEvalSampler(n, num_replicas=world, rank=r)) for r in range(world)]
                flat = [i for p in parts for i in p]
                self.assertEqual(sorted(flat), list(range(n)), (n, world))
                for r in range(world):
                    self.assertEqual(
                        len(ExactDistributedEvalSampler(n, num_replicas=world, rank=r)), len(parts[r])
                    )

    def test_r89_counterexample_distributed_sampler_pads(self) -> None:
        # The stock sampler repeats sample 0 for n=5, world 2; ours does not.
        stock = [list(DistributedSampler(range(5), num_replicas=2, rank=r, shuffle=False)) for r in range(2)]
        self.assertEqual(stock, [[0, 2, 4], [1, 3, 0]])
        exact = [list(ExactDistributedEvalSampler(range(5), num_replicas=2, rank=r)) for r in range(2)]
        self.assertEqual(exact, [[0, 2, 4], [1, 3]])

    def test_rejects_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            ExactDistributedEvalSampler(4, num_replicas=0, rank=0)
        with self.assertRaises(ValueError):
            ExactDistributedEvalSampler(4, num_replicas=2, rank=2)
        with self.assertRaises(ValueError):
            ExactDistributedEvalSampler(-1, num_replicas=1, rank=0)
        with self.assertRaises(TypeError):
            ExactDistributedEvalSampler(True, num_replicas=1, rank=0)
        with self.assertRaises(TypeError):
            ExactDistributedEvalSampler(4, num_replicas=2.0, rank=0)
        with self.assertRaises(TypeError):
            ExactDistributedEvalSampler(4, num_replicas=2, rank=True)



class TestRunnerWithoutProcessGroup(unittest.TestCase):
    def test_needs_a_process_group_for_implicit_rank(self) -> None:
        if dist.is_initialized():
            self.skipTest("a process group is active in this process")
        with self.assertRaises(DistributedContractError):
            ExactDistributedEvalSampler(4)
        with self.assertRaises(DistributedContractError):
            DistributedRunContext()

    def test_distributed_runner_needs_a_process_group(self) -> None:
        if dist.is_initialized():
            self.skipTest("a process group is active in this process")
        with tempfile.TemporaryDirectory() as tmp:
            model = _model()
            with self.assertRaisesRegex(DistributedContractError, "not initialized"):
                HexTrainingRunner(
                    _config(Path(tmp), distributed=True),
                    model,
                    torch.optim.SGD(model.parameters(), lr=0.1),
                    DataLoader(_Data(4, 1), batch_size=2),
                    DataLoader(_Data(4, 2), batch_size=2),
                )

    def test_rng_state_tensors_round_trip_exactly(self) -> None:
        random.seed(5)
        random.gauss(0.0, 1.0)  # leaves a cached gauss_next
        np.random.seed(6)
        np.random.standard_normal()  # leaves a cached gaussian
        torch.manual_seed(7)
        states = rng_state_tensors()
        self.assertEqual(states["python"].dtype, torch.int64)
        self.assertEqual(states["numpy"].numel(), 627)
        want = (random.gauss(0.0, 1.0), random.random(), float(np.random.standard_normal()),
                float(np.random.rand()), torch.rand(3).tolist())
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        restore_rng_state_tensors(states)
        got = (random.gauss(0.0, 1.0), random.random(), float(np.random.standard_normal()),
               float(np.random.rand()), torch.rand(3).tolist())
        self.assertEqual(got, want)

    def test_error_types(self) -> None:
        # callers that catch RuntimeError (e.g. the OOM rescue path) see both
        self.assertTrue(issubclass(PeerRankError, RuntimeError))
        self.assertTrue(issubclass(DistributedContractError, RuntimeError))

    def test_config_flags_must_be_bool(self) -> None:
        with self.assertRaises(ValueError):
            RunnerConfig(distributed=1)
        with self.assertRaises(ValueError):
            RunnerConfig(is_main_process=0)

    def test_set_epoch_reaches_the_sampler_once(self) -> None:
        calls: List[int] = []

        class _Recording(torch.utils.data.Sampler):
            def __init__(self) -> None:
                self.n = 4

            def set_epoch(self, epoch: int) -> None:
                calls.append(epoch)

            def __iter__(self):
                return iter(range(self.n))

            def __len__(self) -> int:
                return self.n

        with tempfile.TemporaryDirectory() as tmp:
            model = _model()
            runner = HexTrainingRunner(
                _config(Path(tmp), num_epochs=2),
                model,
                torch.optim.SGD(model.parameters(), lr=0.01),
                DataLoader(_Data(4, 1), batch_size=2, sampler=_Recording()),
                DataLoader(_Data(4, 2), batch_size=2),
            )
            runner.run()
        self.assertEqual(calls, [0, 1])


# ------------------------------------------------------------------
# In-process gloo group, world size 1
# ------------------------------------------------------------------


class TestWorldSizeOne(unittest.TestCase):
    """The distributed path under coverage; world 1 must equal no DDP at all."""

    @classmethod
    def setUpClass(cls) -> None:
        if dist.is_initialized():
            raise unittest.SkipTest("a process group is already active")
        cls._tmp = tempfile.TemporaryDirectory()
        dist.init_process_group(
            "gloo",
            init_method=(Path(cls._tmp.name) / "rendezvous").as_uri(),
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=60),
        )
        cls.ctx = DistributedRunContext()

    @classmethod
    def tearDownClass(cls) -> None:
        dist.destroy_process_group()
        cls._tmp.cleanup()

    def _runner(self, out: Path, *, distributed: bool, **kw: Any) -> HexTrainingRunner:
        model = _model()
        wrapped = DDP(model) if distributed else model
        train = kw.pop("train", DataLoader(_Data(8, seed=1), batch_size=4))
        val = kw.pop("val", DataLoader(_Data(5, seed=2), batch_size=2))
        return HexTrainingRunner(
            _config(out, distributed=distributed, **kw),
            wrapped,
            torch.optim.SGD(wrapped.parameters(), lr=0.1, momentum=0.9),
            train,
            val,
        )

    def test_world_one_reproduces_single_process_bitwise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._runner(Path(tmp) / "plain", distributed=False)
            ddp = self._runner(Path(tmp) / "ddp", distributed=True)
            a, b = plain.run(), ddp.run()
            self.assertEqual(_flat_params(plain.raw_model), _flat_params(ddp.raw_model))
            self.assertEqual(_flat_params(plain.ema_model.module), _flat_params(ddp.ema_model.module))
            for key in ("val_nrmse", "val_loss", "n_batches"):
                self.assertEqual([v[key] for v in a["history"]["val"]], [v[key] for v in b["history"]["val"]])
            for key in ("mean_loss", "mean_grad_norm", "n_batches"):
                self.assertEqual(
                    [t[key] for t in a["history"]["train"]], [t[key] for t in b["history"]["train"]]
                )
            # checkpoints hold the unwrapped module: no "module." prefix
            payload = torch.load(Path(tmp) / "ddp" / "checkpoints" / "best.pt", weights_only=True)
            self.assertEqual(sorted(payload["model_state_dict"]), sorted(plain.raw_model.state_dict()))

    def test_world_one_resume_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            full = self._runner(out / "full", distributed=True, num_epochs=4)
            full.run()
            first = self._runner(out / "split", distributed=True, num_epochs=2)
            first.run()
            second = self._runner(
                out / "split",
                distributed=True,
                num_epochs=4,
                resume_from=str(out / "split" / "checkpoints" / "resume_last.pt"),
                resume_trusted=True,
            )
            report = second.run()["resume_report"]
            self.assertEqual(report["epoch"], 1)
            self.assertEqual(second.ckpt_mgr.run_id, first.ckpt_mgr.run_id)
            self.assertEqual(_flat_params(full.raw_model), _flat_params(second.raw_model))

    def test_is_main_process_must_match_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "rank == 0"):
                self._runner(Path(tmp), distributed=True, is_main_process=False)

    def test_runner_refuses_distributed_without_ddp_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = _model()
            with self.assertRaisesRegex(ValueError, "needs a DistributedDataParallel model"):
                HexTrainingRunner(
                    _config(Path(tmp), distributed=True),
                    model,
                    torch.optim.SGD(model.parameters(), lr=0.1),
                    DataLoader(_Data(4, 1), batch_size=2),
                    DataLoader(_Data(4, 2), batch_size=2),
                )

    def test_adapter_around_the_inner_module_is_refused(self) -> None:
        # The adapter runs the training forward; around model.module it would
        # bypass DDP and the gradients would never be all-reduced.
        with tempfile.TemporaryDirectory() as tmp:
            model = DDP(_model())
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            loaders = (DataLoader(_Data(4, 1), batch_size=2), DataLoader(_Data(4, 2), batch_size=2))
            with self.assertRaisesRegex(ValueError, "bypasses DDP"):
                HexTrainingRunner(_config(Path(tmp) / "a", distributed=True), model, optimizer, *loaders,
                                  adapter=OperatorAdapterV3(model.module, "graph_operator_v3"))
            runner = HexTrainingRunner(_config(Path(tmp) / "b", distributed=True), model, optimizer, *loaders,
                                       adapter=OperatorAdapterV3(model, "graph_operator_v3"))
            self.assertIs(runner.adapter.model, model)

    def test_forward_error_is_raised_after_the_agreement(self) -> None:
        def broken(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            raise KeyError("loss exploded (injected)")

        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True)
            runner.loss_fn = broken
            with self.assertRaisesRegex(KeyError, "injected"):
                runner.run()

    def test_nondeterministic_validation_sampler_is_refused(self) -> None:
        class _Shifting(torch.utils.data.Sampler):
            def __init__(self) -> None:
                self.calls = 0

            def __iter__(self):
                self.calls += 1
                order = [0, 1, 2, 3]
                return iter(order[self.calls % 4:] + order[:self.calls % 4])

            def __len__(self) -> int:
                return 4

        with tempfile.TemporaryDirectory() as tmp:
            val = DataLoader(_Data(4, 2), batch_size=2, sampler=_Shifting())
            runner = self._runner(Path(tmp), distributed=True, val=val, num_epochs=1)
            with self.assertRaisesRegex(DistributedContractError, "different index sequence"):
                runner.run()

    def test_rank_rng_restore_reports_why_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True)
            for extra, reason in ((None, False), ({"rank_rng_states": {"world_size": 2}}, False)):
                report = {"extra": extra}
                with self.assertLogs("runner_hex_v3", level="WARNING"):
                    runner._restore_rank_rng(report)
                self.assertIs(report["rank_rng_restored"], reason)

    def test_resume_checkpoint_carries_every_rank_rng(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._runner(Path(tmp), distributed=True, num_epochs=1).run()
            payload = torch.load(Path(tmp) / "checkpoints" / "resume_last.pt", weights_only=False)
            states = payload["extra"]["rank_rng_states"]
            self.assertEqual(states["world_size"], 1)
            self.assertEqual([len(states[k]) for k in ("torch", "python", "numpy")], [1, 1, 1])

    def test_oom_while_preparing_gets_the_rescue(self) -> None:
        # Review round 2: preparing runs in _agreed_next, outside _train_one_step's handler.
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True)

            def oom(batch: Any) -> Any:
                raise RuntimeError("CUDA out of memory (injected)")

            runner._prepare_step = oom
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                runner.run()
            self.assertTrue(list(runner.failure_dir.glob("*CUDA_OOM*.json")))

    def test_effective_amp_names_scaler_and_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True)
            plain = runner._effective_amp()
            self.assertIn("autocast=False", plain)
            runner.grad_scaler = object()
            self.assertNotEqual(runner._effective_amp(), plain)

    def test_decision_sums_are_rank0_bound(self) -> None:
        self.assertEqual(self.ctx.all_reduce_sum_rank0([1.25, -0.0]), [1.25, -0.0])
        self.assertEqual(self.ctx.max_int(2), 2)

    def test_ddp_model_without_distributed_flag_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = DDP(_model())
            with self.assertRaisesRegex(ValueError, "rank-local"):
                HexTrainingRunner(
                    _config(Path(tmp)),
                    model,
                    torch.optim.SGD(model.parameters(), lr=0.1),
                    DataLoader(_Data(4, 1), batch_size=2),
                    DataLoader(_Data(4, 2), batch_size=2),
                )

    def test_failure_records_go_to_a_rank_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True)
            self.assertEqual(runner.failure_dir, Path(tmp) / "failures" / "rank0")

    def test_validation_partition_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # every sample twice: not an exact partition
            twice = DataLoader(_Data(4, 2), batch_size=2, sampler=[0, 1, 2, 3, 0, 1, 2, 3])
            runner = self._runner(Path(tmp), distributed=True, val=twice)
            with self.assertRaisesRegex(DistributedContractError, "not an exact partition"):
                runner.run()

    def test_validation_without_batch_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unbatched = DataLoader(_Data(3, 2), batch_size=None)
            runner = self._runner(Path(tmp), distributed=True, val=unbatched, num_epochs=1)
            self.assertEqual(runner._val_indices(), [0, 1, 2])

    def test_iterable_validation_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True, val=DataLoader(_Stream(), batch_size=1))
            with self.assertRaisesRegex(DistributedContractError, "map-style"):
                runner.run()

    def test_unsized_train_loader_runs(self) -> None:
        # The end of the data is agreed per step, so an IterableDataset can train.
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True, train=DataLoader(_Stream(), batch_size=1),
                                  num_epochs=2)
            history = runner.run()["history"]["train"]
            self.assertEqual([t["n_batches"] for t in history], [1, 1])
            self.assertEqual(runner.global_step, 2)

    def test_config_digest_ignores_rank_local_fields_only(self) -> None:
        from runner_hex_v3 import _config_digest
        base = RunnerConfig(output_dir="/a")
        self.assertEqual(_config_digest(base), _config_digest(RunnerConfig(output_dir="/b", is_main_process=False)))
        self.assertEqual(_config_digest(RunnerConfig(resume_from="/x/r.pt")),
                         _config_digest(RunnerConfig(resume_from="/y/r.pt")))
        for changed in (RunnerConfig(resume_from="/x/r.pt"), RunnerConfig(num_epochs=7),
                        RunnerConfig(best_metric_min_delta=0.5), RunnerConfig(use_ema=False)):
            self.assertNotEqual(_config_digest(base), _config_digest(changed))

    def test_missing_resume_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True, resume_from=str(Path(tmp) / "nope.pt"))
            with self.assertRaises(FileNotFoundError):
                runner.run()

    # -- context primitives ------------------------------------------------

    def test_partition_check_branches(self) -> None:
        self.ctx.check_exact_partition([2, 0, 1], 3, what="t")
        self.ctx.check_exact_partition([], 0, what="t")
        with self.assertRaisesRegex(DistributedContractError, "1 sample\\(s\\) on more"):
            self.ctx.check_exact_partition([0, 0, 1, 2], 3, what="t")
        with self.assertRaisesRegex(DistributedContractError, "1 of 3 sample\\(s\\) on none"):
            self.ctx.check_exact_partition([0, 1], 3, what="t")
        with self.assertRaisesRegex(DistributedContractError, "outside"):
            self.ctx.check_exact_partition([0, 1, 3], 3, what="t")
        with self.assertRaisesRegex(DistributedContractError, "outside"):
            self.ctx.check_exact_partition([-1, 0, 1], 3, what="t")
        with self.assertRaisesRegex(DistributedContractError, "outside"):
            self.ctx.check_exact_partition([0], 0, what="t")

    def test_run_agreed(self) -> None:
        self.assertEqual(self.ctx.run_agreed(lambda: 7, what="t"), 7)
        with self.assertRaisesRegex(KeyError, "boom"):
            self.ctx.run_agreed(lambda: {}["boom"], what="t")

    def test_reductions_and_broadcast(self) -> None:
        self.assertEqual(self.ctx.all_reduce_sum([1.5, 2.0]), [1.5, 2.0])
        self.assertTrue(math.isnan(self.ctx.all_reduce_sum([math.nan])[0]))
        self.assertFalse(self.ctx.any_true(False))
        self.assertTrue(self.ctx.any_true(True))
        self.assertEqual(self.ctx.require_equal_int(3, what="t"), 3)
        self.ctx.require_rank0_agreement([math.nan, 1.0, -0.0], what="t")
        self.assertEqual(self.ctx.broadcast_bytes(b"\x01\x02", 2), b"\x01\x02")
        with self.assertRaisesRegex(DistributedContractError, "2-byte"):
            self.ctx.broadcast_bytes(b"\x01", 2)
        with self.assertRaisesRegex(DistributedContractError, "2-byte"):
            self.ctx.broadcast_bytes(None, 2)

    def test_implicit_rank_from_group(self) -> None:
        sampler = ExactDistributedEvalSampler(3)
        self.assertEqual((sampler.num_replicas, sampler.rank, list(sampler)), (1, 0, [0, 1, 2]))


# ------------------------------------------------------------------
# Two real gloo ranks
# ------------------------------------------------------------------


def _ddp_runner(
    rank: int, out: Path, *, train: DataLoader, val: DataLoader, model_seed: int = 0, **kw: Any
) -> HexTrainingRunner:
    model = DDP(_model(model_seed))
    return HexTrainingRunner(
        _config(out, distributed=True, is_main_process=rank == 0, **kw),
        model,
        torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        train,
        val,
    )


def _train_loader(rank: int, world: int, *, shuffle: bool = False) -> DataLoader:
    ds = _Data(8, seed=1)
    return DataLoader(
        ds, batch_size=2, sampler=DistributedSampler(ds, world, rank, shuffle=shuffle, seed=7)
    )


def _exact_val_loader() -> DataLoader:
    ds = _Data(5, seed=2)
    return DataLoader(ds, batch_size=2, sampler=ExactDistributedEvalSampler(ds))


def _digest(module: nn.Module) -> str:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()]).numpy().tobytes().hex()


def _sc_equivalence(rank: int, world: int, out: Path) -> Dict[str, Any]:
    runner = _ddp_runner(rank, out, train=_train_loader(rank, world), val=_exact_val_loader())
    result = runner.run()
    best = torch.load(out / "checkpoints" / "best.pt", weights_only=True) if rank == 0 else None
    return {
        "params": _flat_params(runner.raw_model),
        "ema": _flat_params(runner.ema_model.module),
        "val_nrmse": [v["val_nrmse"] for v in result["history"]["val"]],
        "val_n_batches": [v["n_batches"] for v in result["history"]["val"]],
        "mean_loss": [t["mean_loss"] for t in result["history"]["train"]],
        "best_keys": sorted(best["model_state_dict"]) if best else None,
        "run_id": runner.ckpt_mgr.run_id,
    }


def _sc_padded_val(rank: int, world: int, out: Path) -> Any:
    ds = _Data(5, seed=2)
    padded = DataLoader(ds, batch_size=2, sampler=DistributedSampler(ds, world, rank, shuffle=False))
    _ddp_runner(rank, out, train=_train_loader(rank, world), val=padded, num_epochs=1).run()


def _sc_nan_on_rank1(rank: int, world: int, out: Path) -> Dict[str, Any]:
    calls = {"n": 0}
    mse = nn.MSELoss()

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = mse(pred, target)
        if pred.requires_grad:  # training steps only
            calls["n"] += 1
            if rank == 1 and calls["n"] == 2:
                return loss * float("nan")
        return loss

    model = DDP(_model())
    runner = HexTrainingRunner(
        _config(out, distributed=True, is_main_process=rank == 0, num_epochs=2),
        model,
        torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        _train_loader(rank, world),
        _exact_val_loader(),
        loss_fn=loss_fn,
    )
    result = runner.run()
    return {
        "digest": _digest(runner.raw_model),
        "n_anomalies": [t["n_anomalies"] for t in result["history"]["train"]],
        "n_batches": [t["n_batches"] for t in result["history"]["train"]],
        "failure_files": sorted(p.name for p in runner.failure_dir.glob("*.json")),
    }


def _nan_skip_under(ddp_kwargs: Dict[str, Any]) -> Callable[[int, int, Path], Dict[str, Any]]:
    """An agreed skip runs forward without backward on every rank; DDP's
    find_unused_parameters and static_graph modes must accept that too."""

    def scenario(rank: int, world: int, out: Path) -> Dict[str, Any]:
        calls = {"n": 0}
        mse = nn.MSELoss()

        def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            loss = mse(pred, target)
            if pred.requires_grad:
                calls["n"] += 1
                if rank == 0 and calls["n"] == 1:
                    return loss * float("nan")
            return loss

        model = DDP(_model(), **ddp_kwargs)
        runner = HexTrainingRunner(
            _config(out, distributed=True, is_main_process=rank == 0, num_epochs=2),
            model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
            _train_loader(rank, world), _exact_val_loader(), loss_fn=loss_fn,
        )
        result = runner.run()
        return {"digest": _digest(runner.raw_model),
                "n_anomalies": [t["n_anomalies"] for t in result["history"]["train"]]}

    return scenario


def _sc_forward_error_rank1(rank: int, world: int, out: Path) -> Any:
    mse = nn.MSELoss()

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.requires_grad and rank == 1:
            raise KeyError("model error on rank 1 (injected)")
        return mse(pred, target)

    model = DDP(_model())
    HexTrainingRunner(
        _config(out, distributed=True, is_main_process=rank == 0, num_epochs=1),
        model, torch.optim.SGD(model.parameters(), lr=0.1),
        _train_loader(rank, world), _exact_val_loader(), loss_fn=loss_fn,
    ).run()


def _sc_construction_error_rank1(rank: int, world: int, out: Path) -> Any:
    model = DDP(_model())
    # rank 1 claims to be the main process: refused there, agreed everywhere
    HexTrainingRunner(
        _config(out, distributed=True, is_main_process=True, num_epochs=1),
        model, torch.optim.SGD(model.parameters(), lr=0.1),
        _train_loader(rank, world), _exact_val_loader(),
    )


def _bn_runner(rank: int, world: int, out: Path, *, epochs: int = 3, shuffle: bool = False,
               **kw: Any) -> HexTrainingRunner:
    model = DDP(_BN())
    return HexTrainingRunner(
        _config(out, distributed=True, is_main_process=rank == 0, num_epochs=epochs, **kw),
        model, torch.optim.SGD(model.parameters(), lr=0.1),
        _train_loader(rank, world, shuffle=shuffle), _exact_val_loader(),
    )


def _buffer_digest(module: nn.Module) -> str:
    return b"".join(b.detach().reshape(-1).to(torch.float64).numpy().tobytes()
                    for _, b in sorted(module.named_buffers())).hex()


def _sc_bn_buffers(rank: int, world: int, out: Path) -> Dict[str, Any]:
    """Review finding 2: validation must evaluate one model, the one rank 0 saves."""
    runner = _bn_runner(rank, world, out)
    result = runner.run()
    value: Dict[str, Any] = {
        "raw": _buffer_digest(runner.raw_model), "ema": _buffer_digest(runner.ema_model),
        "val_nrmse": result["history"]["val"][-1]["val_nrmse"],
    }
    if rank == 0:
        # rank 0's EMA model, evaluated alone on the whole validation set
        module = runner.ema_model.module.eval()
        data = _Data(5, seed=2)
        with torch.no_grad():
            pred = module(data.x)
        value["full_set_nrmse"] = float(((pred - data.y).pow(2).sum() / data.y.pow(2).sum()).sqrt())
    return value


def _sc_bn_resume(rank: int, world: int, out: Path) -> Dict[str, Any]:
    full = _bn_runner(rank, world, out / "full", epochs=4, shuffle=True)
    full.run()
    _bn_runner(rank, world, out / "split", epochs=2, shuffle=True).run()
    resumed = _bn_runner(rank, world, out / "split", epochs=4, shuffle=True,
                         resume_from=str(out / "split" / "checkpoints" / "resume_last.pt"), resume_trusted=True)
    resumed.run()
    return {"full": _digest(full.raw_model) + _buffer_digest(full.ema_model),
            "resumed": _digest(resumed.raw_model) + _buffer_digest(resumed.ema_model)}


def _sc_prepare_error_bn(rank: int, world: int, out: Path) -> Any:
    """Review finding 1: an error before the forward of a model with buffers."""
    ds = _NoTarget() if rank == 1 else _Data(4, seed=1)
    model = DDP(_BN())
    HexTrainingRunner(
        _config(out, distributed=True, is_main_process=rank == 0, num_epochs=1),
        model, torch.optim.SGD(model.parameters(), lr=0.1),
        DataLoader(ds, batch_size=2), _exact_val_loader(),
    ).run()


def _sc_resume_from_on_rank0_only(rank: int, world: int, out: Path) -> Any:
    """Review finding 3: resume_from set on one rank only."""
    _ddp_runner(rank, out, train=_train_loader(rank, world), val=_exact_val_loader(), num_epochs=1).run()
    resume = str(out / "checkpoints" / "resume_last.pt") if rank == 0 else None
    _ddp_runner(rank, out, train=_train_loader(rank, world), val=_exact_val_loader(), num_epochs=2,
                resume_from=resume, resume_trusted=True).run()


def _sc_lying_len(rank: int, world: int, out: Path) -> Any:
    """Review finding 4: equal len(), unequal number of batches."""
    train = DataLoader(_LyingStream(3 + rank), batch_size=1)
    _ddp_runner(rank, out, train=train, val=_exact_val_loader(), num_epochs=1).run()


def _sc_resume_rank_generators(rank: int, world: int, out: Path) -> Dict[str, Any]:
    """Review finding 5: per-rank shards shuffled by per-rank loader generators."""

    def runner(directory: Path, epochs: int, **kw: Any) -> HexTrainingRunner:
        shard = torch.utils.data.Subset(_Data(8, seed=1), list(range(rank, 8, world)))
        gen = torch.Generator().manual_seed(42 + rank)
        return _ddp_runner(rank, directory, train=DataLoader(shard, batch_size=2, shuffle=True, generator=gen),
                           val=_exact_val_loader(), num_epochs=epochs, **kw)

    full = runner(out / "full", 4)
    full.run()
    runner(out / "split", 2).run()
    resumed = runner(out / "split", 4, resume_from=str(out / "split" / "checkpoints" / "resume_last.pt"),
                     resume_trusted=True)
    resumed.run()
    return {"full": _digest(full.raw_model), "resumed": _digest(resumed.raw_model)}


def _sc_allreduce_noise(rank: int, world: int, out: Path) -> Dict[str, Any]:
    """Review finding 6: decision values must not rely on bitwise-equal all-reduce results.

    Fault injection: on rank 1 every float64 all-reduce result is moved by one ulp,
    as a backend whose reduction order differs per rank could. The validation
    values are rank-0-bound, so both ranks must still decide on the same numbers.
    """
    original = dist.all_reduce

    def noisy(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        work = original(tensor, *args, **kwargs)
        if rank == 1 and tensor.dtype == torch.float64:
            tensor.copy_(torch.nextafter(tensor, torch.full_like(tensor, math.inf)))
        return work

    runner = _ddp_runner(rank, out, train=_train_loader(rank, world), val=_exact_val_loader())
    dist.all_reduce = noisy
    try:
        result = runner.run()
    finally:
        dist.all_reduce = original
    return {"val": [[v["val_nrmse"], v["val_loss"]] for v in result["history"]["val"]],
            "best_metric_value": runner.ckpt_mgr.best_metric_value}


class _Cache(nn.Module):
    """A non-persistent buffer whose shape follows the last batch."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(3, 2, dtype=torch.float64)
        self.register_buffer("last", torch.zeros(0, 3, dtype=torch.float64), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last = x.detach()
        return self.lin(x)


def _sc_buffer_shape_mismatch(rank: int, world: int, out: Path) -> Any:
    """Review round 2, finding 1: equal names, different shapes -> agreed error, not SIGABRT."""
    model = DDP(_Cache(), broadcast_buffers=False)
    train = DataLoader(_Data(5 - rank, seed=1), batch_size=3)  # last batch: 2 vs 1 samples
    HexTrainingRunner(
        # no EMA: AveragedModel(use_buffers=True) cannot average a buffer that changes
        # shape (a symmetric RuntimeError on every rank) and would hide the sync path
        _config(out, distributed=True, is_main_process=rank == 0, num_epochs=1, use_ema=False),
        model, torch.optim.SGD(model.parameters(), lr=0.1), train, _exact_val_loader(),
    ).run()


def _sc_effective_amp_differs(rank: int, world: int, out: Path) -> Any:
    """Review round 2, finding 4: one host resolves bfloat16 to float16 + GradScaler."""
    original = HexTrainingRunner._effective_amp
    if rank == 1:
        HexTrainingRunner._effective_amp = lambda self: "autocast=True|dtype=torch.float16|grad_scaler=True"
    try:
        _ddp_runner(rank, out, train=_train_loader(rank, world), val=_exact_val_loader(), num_epochs=1)
    finally:
        HexTrainingRunner._effective_amp = original


def _sc_save_failure_rank0(rank: int, world: int, out: Path) -> Any:
    runner = _ddp_runner(rank, out, train=_train_loader(rank, world), val=_exact_val_loader(), num_epochs=1)
    if rank == 0:
        def broken(**_kw: Any) -> Any:
            raise OSError("disk full (injected)")

        runner.ckpt_mgr.save_last_model_checkpoint = broken
    runner.run()


def _sc_unequal_batches(rank: int, world: int, out: Path) -> Any:
    train = DataLoader(_Data(2 + rank, seed=1), batch_size=1)
    _ddp_runner(rank, out, train=train, val=_exact_val_loader(), num_epochs=1).run()


def _sc_resume_equivalence(rank: int, world: int, out: Path) -> Dict[str, Any]:
    full = _ddp_runner(rank, out / "full", train=_train_loader(rank, world, shuffle=True),
                       val=_exact_val_loader(), num_epochs=4)
    full.run()
    split = out / "split"
    first = _ddp_runner(rank, split, train=_train_loader(rank, world, shuffle=True),
                        val=_exact_val_loader(), num_epochs=2)
    first.run()
    second = _ddp_runner(
        rank, split, train=_train_loader(rank, world, shuffle=True), val=_exact_val_loader(),
        model_seed=123, num_epochs=4,
        resume_from=str(split / "checkpoints" / "resume_last.pt"), resume_trusted=True,
    )
    second.run()
    return {
        "full": _digest(full.raw_model),
        "resumed": _digest(second.raw_model),
        "same_run_id": second.ckpt_mgr.run_id == first.ckpt_mgr.run_id,
    }


class _Dropout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(3, 16, dtype=torch.float64), nn.Tanh(), nn.Dropout(0.5),
                                 nn.Linear(16, 2, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sc_resume_rank_rng(rank: int, world: int, out: Path) -> Dict[str, Any]:
    """Per-rank seeds + dropout: each rank must continue its own RNG stream."""

    mse = nn.MSELoss()

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = mse(pred, target)
        if pred.requires_grad:  # the Python and NumPy streams shape the update too
            loss = loss * (1.0 + 0.01 * random.random()) * (1.0 + 0.01 * float(np.random.rand()))
        return loss

    def runner(directory: Path, epochs: int, **kw: Any) -> HexTrainingRunner:
        model = DDP(_Dropout())
        torch.manual_seed(1000 + rank)  # rank-specific streams (dropout masks, loss weights)
        random.seed(2000 + rank)
        np.random.seed(3000 + rank)
        return HexTrainingRunner(
            _config(directory, distributed=True, is_main_process=rank == 0, num_epochs=epochs, **kw),
            model, torch.optim.SGD(model.parameters(), lr=0.1),
            _train_loader(rank, world, shuffle=True), _exact_val_loader(), loss_fn=loss_fn,
        )

    full = runner(out / "full", 4)
    full.run()
    runner(out / "split", 2).run()
    resumed = runner(out / "split", 4, resume_from=str(out / "split" / "checkpoints" / "resume_last.pt"),
                     resume_trusted=True)
    report = resumed.run()["resume_report"]
    return {"full": _digest(full.raw_model), "resumed": _digest(resumed.raw_model),
            "rank_rng_restored": report["rank_rng_restored"]}


def _sc_resume_missing_on_rank1(rank: int, world: int, out: Path) -> Any:
    first = _ddp_runner(rank, out, train=_train_loader(rank, world), val=_exact_val_loader(), num_epochs=1)
    first.run()
    path = out / "checkpoints" / "resume_last.pt"
    resume_from = str(path) if rank == 0 else str(out / "missing.pt")
    _ddp_runner(rank, out, train=_train_loader(rank, world), val=_exact_val_loader(), num_epochs=2,
                resume_from=resume_from, resume_trusted=True).run()


def _sc_resume_other_weights_on_rank1(rank: int, world: int, out: Path) -> Any:
    first = _ddp_runner(rank, out, train=_train_loader(rank, world), val=_exact_val_loader(), num_epochs=1)
    first.run()
    path = out / "checkpoints" / "resume_last.pt"
    if rank == 1:
        payload = torch.load(path, weights_only=False)
        for value in payload["model_state_dict"].values():
            value.add_(1.0)
        other = out / "rank1_copy.pt"
        torch.save(payload, other)
        path = other
    _ddp_runner(rank, out, train=_train_loader(rank, world), val=_exact_val_loader(), num_epochs=2,
                resume_from=str(path), resume_trusted=True).run()


_SCENARIOS: Dict[str, Callable[[int, int, Path], Any]] = {
    "equivalence": _sc_equivalence,
    "padded_val": _sc_padded_val,
    "nan_on_rank1": _sc_nan_on_rank1,
    "nan_skip_find_unused": _nan_skip_under({"find_unused_parameters": True}),
    "nan_skip_static_graph": _nan_skip_under({"static_graph": True}),
    "forward_error_rank1": _sc_forward_error_rank1,
    "construction_error_rank1": _sc_construction_error_rank1,
    "save_failure_rank0": _sc_save_failure_rank0,
    "buffer_shape_mismatch": _sc_buffer_shape_mismatch,
    "effective_amp_differs": _sc_effective_amp_differs,
    "bn_buffers": _sc_bn_buffers,
    "bn_resume": _sc_bn_resume,
    "prepare_error_bn": _sc_prepare_error_bn,
    "resume_from_on_rank0_only": _sc_resume_from_on_rank0_only,
    "lying_len": _sc_lying_len,
    "resume_rank_generators": _sc_resume_rank_generators,
    "allreduce_noise": _sc_allreduce_noise,
    "unequal_batches": _sc_unequal_batches,
    "resume_equivalence": _sc_resume_equivalence,
    "resume_rank_rng": _sc_resume_rank_rng,
    "resume_missing_on_rank1": _sc_resume_missing_on_rank1,
    "resume_other_weights_on_rank1": _sc_resume_other_weights_on_rank1,
}


def _write_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)  # a rank killed mid-write leaves the previous file


def _worker(rank: int, world: int, root: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),  # file:///C:/... on Windows
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=30),  # a desynchronised collective fails fast
    )
    results: Dict[str, Any] = {}
    out = Path(root) / f"rank{rank}.json"
    try:
        for name, scenario in _SCENARIOS.items():
            try:
                results[name] = {"ok": True, "value": scenario(rank, world, Path(root) / name)}
            except Exception as exc:  # recorded and asserted by the parent
                results[name] = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
            # after every scenario: a later hang must not erase earlier results
            _write_atomic(out, results)
            try:
                dist.barrier()
            except Exception as exc:  # ranks out of step: stop, the parent fails the rest
                results["_job"] = f"barrier after {name}: {type(exc).__name__}: {exc}"
                _write_atomic(out, results)
                break
    finally:
        dist.destroy_process_group()


class TestTwoRanks(unittest.TestCase):
    """Real 2-rank gloo job; every scenario must end, none may hang."""

    WORLD = 2
    DEADLINE_S = 400

    @classmethod
    def setUpClass(cls) -> None:
        if not dist.is_gloo_available():
            raise unittest.SkipTest("gloo backend not available")
        if os.environ.get("HEXANTK_SKIP_MULTIPROCESS") == "1":
            raise unittest.SkipTest("HEXANTK_SKIP_MULTIPROCESS=1")
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls._tmp.name
        # A hang or a crashed rank must turn the scenario tests red, not end
        # in a setUpClass error: record the job error, assert per scenario.
        cls.job_error = None
        ctx = mp.start_processes(_worker, args=(cls.WORLD, root), nprocs=cls.WORLD,
                                 join=False, start_method="spawn")
        deadline = time.monotonic() + cls.DEADLINE_S
        try:
            while not ctx.join(timeout=5):
                if time.monotonic() > deadline:
                    for p in ctx.processes:
                        p.kill()
                    cls.job_error = f"2-rank job did not finish within {cls.DEADLINE_S} s (hang)"
                    break
        except Exception as exc:  # a rank process died
            cls.job_error = f"{type(exc).__name__}: {exc}"
        cls.results = []
        for r in range(cls.WORLD):
            path = Path(root) / f"rank{r}.json"
            try:
                cls.results.append(json.loads(path.read_text()))
            except (OSError, ValueError) as exc:
                cls.results.append({"_job": f"rank {r} result unreadable: {exc}"})
        with tempfile.TemporaryDirectory() as ref_dir:
            cls.reference = _single_process_reference(Path(ref_dir))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _ok(self, name: str) -> List[Any]:
        values = []
        for rank, res in enumerate(self.results):
            entry = res.get(name)
            self.assertIsNotNone(entry, f"{name} rank {rank}: no result ({self.job_error}, {res.get('_job')})")
            self.assertTrue(entry["ok"], f"{name} rank {rank}: {entry}")
            values.append(entry["value"])
        return values

    def _errors(self, name: str) -> List[str]:
        return [res.get(name, {}).get("error", "<no result>") for res in self.results]

    def test_two_ranks_reproduce_the_single_process_run(self) -> None:
        r0, r1 = self._ok("equivalence")
        ref = self.reference
        self.assertEqual(r0["params"], r1["params"])
        self.assertEqual(r0["val_nrmse"], r1["val_nrmse"])
        for got, want in zip(r0["params"], ref["params"]):
            self.assertAlmostEqual(got, want, delta=1e-12)
        for got, want in zip(r0["ema"], ref["ema"]):
            self.assertAlmostEqual(got, want, delta=1e-12)
        for got, want in zip(r0["val_nrmse"], ref["val_nrmse"]):
            self.assertAlmostEqual(got, want, delta=1e-12)
        for got, want in zip(r0["mean_loss"], ref["mean_loss"]):
            self.assertAlmostEqual(got, want, delta=1e-12)
        # 5 samples, batch 2: rank 0 has [0, 2], [4]; rank 1 has [1, 3]
        self.assertEqual(r0["val_n_batches"], [3, 3, 3])
        self.assertEqual(r0["best_keys"], ["0.bias", "0.weight", "2.bias", "2.weight"])
        self.assertEqual(r0["run_id"], r1["run_id"])

    def test_padded_validation_sampler_fails_on_both_ranks(self) -> None:
        self.assertEqual(self._errors("padded_val"), ["DistributedContractError"] * 2)
        self.assertIn("1 sample(s) on more than one rank", self.results[0].get("padded_val", {}).get("message", ""))

    def test_nan_on_one_rank_skips_the_step_everywhere(self) -> None:
        r0, r1 = self._ok("nan_on_rank1")
        self.assertEqual(r0["digest"], r1["digest"])
        self.assertEqual(r0["n_anomalies"], [1, 0])
        self.assertEqual(r1["n_anomalies"], [1, 0])
        self.assertEqual(r0["n_batches"], [1, 2])
        self.assertEqual(r0["failure_files"], [])
        self.assertTrue(any("NaN" in name for name in r1["failure_files"]), r1["failure_files"])

    def test_nan_skip_under_find_unused_parameters_and_static_graph(self) -> None:
        for name in ("nan_skip_find_unused", "nan_skip_static_graph"):
            r0, r1 = self._ok(name)
            self.assertEqual(r0["digest"], r1["digest"], name)
            self.assertEqual(r0["n_anomalies"], [1, 0], name)
            self.assertEqual(r1["n_anomalies"], [1, 0], name)

    def test_forward_error_on_one_rank_fails_both(self) -> None:
        self.assertEqual(self._errors("forward_error_rank1"), ["PeerRankError", "KeyError"])

    def test_construction_error_on_one_rank_fails_both(self) -> None:
        self.assertEqual(self._errors("construction_error_rank1"), ["PeerRankError", "ValueError"])

    def test_bn_buffers_agree_and_the_metric_is_rank0s_model(self) -> None:
        r0, r1 = self._ok("bn_buffers")
        self.assertEqual(r0["raw"], r1["raw"])
        self.assertEqual(r0["ema"], r1["ema"])
        self.assertEqual(r0["val_nrmse"], r1["val_nrmse"])
        self.assertAlmostEqual(r0["val_nrmse"], r0["full_set_nrmse"], delta=1e-12)

    def test_bn_resume_reproduces_the_uninterrupted_run(self) -> None:
        for value in self._ok("bn_resume"):
            self.assertEqual(value["full"], value["resumed"])

    def test_prepare_error_with_buffers_fails_both(self) -> None:
        self.assertEqual(self._errors("prepare_error_bn"), ["PeerRankError", "KeyError"])

    def test_resume_from_on_one_rank_only_fails_both(self) -> None:
        self.assertEqual(self._errors("resume_from_on_rank0_only"), ["DistributedContractError"] * 2)

    def test_lying_len_fails_both(self) -> None:
        self.assertEqual(self._errors("lying_len"), ["DistributedContractError"] * 2)
        self.assertIn("ran out of training batches", self.results[0].get("lying_len", {}).get("message", ""))

    def test_resume_continues_each_ranks_loader_generator(self) -> None:
        for value in self._ok("resume_rank_generators"):
            self.assertEqual(value["full"], value["resumed"])

    def test_decisions_survive_rank_dependent_allreduce_rounding(self) -> None:
        r0, r1 = self._ok("allreduce_noise")
        self.assertEqual(r0, r1)

    def test_buffer_shape_mismatch_fails_both_instead_of_aborting(self) -> None:
        self.assertEqual(self._errors("buffer_shape_mismatch"), ["DistributedContractError"] * 2)
        self.assertIn("shapes", self.results[0].get("buffer_shape_mismatch", {}).get("message", ""))

    def test_effective_amp_difference_fails_both(self) -> None:
        self.assertEqual(self._errors("effective_amp_differs"), ["DistributedContractError"] * 2)

    def test_checkpoint_write_failure_on_rank0_fails_both(self) -> None:
        self.assertEqual(self._errors("save_failure_rank0"), ["OSError", "PeerRankError"])

    def test_unequal_batch_counts_fail_both(self) -> None:
        self.assertEqual(self._errors("unequal_batches"), ["DistributedContractError"] * 2)
        # the declared counts already differ: the fast fail before the first step,
        # not the per-step agreement after a partial epoch (lying_len covers that)
        for res in self.results:
            self.assertIn("number of training batches per epoch", res.get("unequal_batches", {}).get("message", ""))

    def test_resume_reproduces_the_uninterrupted_run(self) -> None:
        for value in self._ok("resume_equivalence"):
            self.assertEqual(value["full"], value["resumed"])
            self.assertTrue(value["same_run_id"])

    def test_resume_continues_each_ranks_own_rng_stream(self) -> None:
        for value in self._ok("resume_rank_rng"):
            self.assertTrue(value["rank_rng_restored"])
            self.assertEqual(value["full"], value["resumed"])

    def test_resume_file_missing_on_one_rank_fails_both(self) -> None:
        self.assertEqual(self._errors("resume_missing_on_rank1"), ["PeerRankError", "FileNotFoundError"])

    def test_resume_with_other_weights_on_one_rank_fails_both(self) -> None:
        self.assertEqual(self._errors("resume_other_weights_on_rank1"), ["DistributedContractError"] * 2)


if __name__ == "__main__":
    unittest.main()
