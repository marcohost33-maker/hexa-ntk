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
  rank 0, unequal batch counts, resume file missing or different on one rank,
  an error after backward on one rank) ends on both ranks with an error
  instead of a hang;
- three real gloo ranks (spec section 8, v1.1): equality with a single-process
  run on an uneven validation partition (2/2/1) and the same agreements.
"""
from __future__ import annotations

import errno
import gc
import json
import math
import os
import random
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest import mock

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
from failure_signature_hex_v3 import FailureSignature
from operator_adapter_hex_v3 import OperatorAdapterV3
from runner_hex_v3 import (
    _OPTIMIZER_INSTANCE_OVERRIDES_TOLERATED,
    _TRUSTED_GRAD_SCALERS,
    _TRUSTED_OPTIMIZERS,
    _TRUSTED_SCHEDULERS,
    HexTrainingRunner,
    RunnerConfig,
    _comparable_state,
    _module_digest,
    _state_digest,
)

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
class TestResumeStateDigest(unittest.TestCase):
    """The canonical digest behind the post-resume agreement (spec section 6)."""

    def test_equal_state_gives_equal_digest_independent_of_dict_order(self) -> None:
        a = {"scale": 1024.0, "growth_tracker": 3, "nested": [1, (2.5, None), "x"]}
        b = {"nested": [1, (2.5, None), "x"], "growth_tracker": 3, "scale": 1024.0}
        self.assertEqual(_state_digest(a), _state_digest(b))

    def test_every_value_change_is_seen(self) -> None:
        base = {"scale": 1024.0, "n": 3, "flag": True, "name": "sgd", "t": torch.zeros(2)}
        changed = [
            {**base, "scale": math.nextafter(1024.0, 2048.0)},  # one ulp
            {**base, "scale": 1024},  # int vs float
            {**base, "n": 4},
            {**base, "flag": False},
            {**base, "flag": 1},  # bool vs int
            {**base, "name": "sgd "},
            {**base, "t": torch.zeros(2, dtype=torch.float64)},  # dtype
            {**base, "t": torch.zeros(1, 2)},  # shape
            {**base, "t": torch.tensor([0.0, -0.0])},  # bytes (-0.0)
            {**base, "extra": None},
        ]
        ref = _state_digest(base)
        for other in changed:
            with self.subTest(other=other):
                self.assertNotEqual(_state_digest(other), ref)
        self.assertNotEqual(_state_digest(0.0), _state_digest(-0.0))
        self.assertEqual(_state_digest(float("nan")), _state_digest(float("nan")))
        self.assertNotEqual(_state_digest([1, 2]), _state_digest([[1, 2]]))

    def test_sparse_tensors_digest_canonically(self) -> None:
        # before: .contiguous() refused every non-strided layout (RuntimeError)
        i = torch.tensor([[0, 1, 1], [2, 0, 0]])
        v = torch.tensor([1.0, 2.0, 3.0])
        uncoalesced = torch.sparse_coo_tensor(i, v, (2, 3))  # duplicate (1, 0) entries
        coalesced = uncoalesced.coalesce()
        dense = uncoalesced.to_dense()
        self.assertEqual(_state_digest(uncoalesced), _state_digest(coalesced))  # same value
        self.assertEqual(_state_digest(dense.to_sparse_csr()), _state_digest(coalesced))  # layout-free value
        self.assertNotEqual(_state_digest(coalesced), _state_digest(dense))  # sparse vs strided typed
        other = torch.sparse_coo_tensor(i, torch.tensor([1.0, 2.0, 4.0]), (2, 3))
        self.assertNotEqual(_state_digest(other), _state_digest(coalesced))
        self.assertNotEqual(_state_digest(torch.sparse_coo_tensor(i, v, (2, 4))), _state_digest(uncoalesced))
        self.assertNotEqual(_state_digest(uncoalesced.double()), _state_digest(uncoalesced))
        module = nn.Module()
        module.register_buffer("s", coalesced)
        self.assertEqual(len(_module_digest(module)), 32)

    def test_opaque_state_values_are_refused_not_named(self) -> None:
        # R8 (cross-family GPT-6 A7-R2, "opaque fallback"): before, an unknown object
        # was digested by its type name only, so two classes of one name -- or one
        # class with other hidden attributes -- agreed between ranks
        def opaque_class(hidden: int) -> type:
            return type("Opaque", (), {"hidden": hidden})

        first, second = opaque_class(1)(), opaque_class(2)()
        self.assertEqual(type(first).__qualname__, type(second).__qualname__)  # the collision
        # the strongest forger: an unrelated class carrying an admitted type's name
        named_float = type("float", (), {"__module__": "builtins"})()
        for value in (first, object(), {"nested": [1, first]}, np.array([first], dtype=object), named_float):
            with self.subTest(value=type(value).__name__), self.assertRaisesRegex(TypeError, "cannot be compared"):
                _state_digest(value)

        # subclasses of admitted types are refused too: they may write other bytes
        class LyingFloat(float):
            def hex(self) -> str:
                return (0.5).hex()

        class LyingDict(dict):
            pass

        for value in (LyingFloat(0.25), type("I", (int,), {})(3), LyingDict(a=1),
                      torch.zeros(1).as_subclass(type("T", (torch.Tensor,), {}))):
            with self.subTest(value=type(value).__name__), self.assertRaisesRegex(TypeError, "cannot be compared"):
                _state_digest({"v": value})
        # positive control: what built-in states hold stays digestible
        import collections

        for value in (collections.Counter({2: 1}), np.float32(1.5), np.int64(2), np.bool_(True),
                      torch.zeros(2), nn.Parameter(torch.ones(1)), (1, None), frozenset({1})):
            with self.subTest(value=type(value).__name__):
                self.assertEqual(len(_state_digest({"v": value})), 32)

    def test_module_extra_state_is_digested_and_opaque_extra_state_refused(self) -> None:
        class WithExtra(nn.Module):
            def __init__(self, extra: Any) -> None:
                super().__init__()
                self.lin = nn.Linear(2, 1)
                self.extra = extra

            def get_extra_state(self) -> Any:
                return self.extra

            def set_extra_state(self, state: Any) -> None:
                self.extra = state

        def module(extra: Any) -> nn.Module:
            torch.manual_seed(0)
            return WithExtra(extra)

        # before: non-tensor entries were skipped, other extra state agreed
        self.assertEqual(_module_digest(module({"k": 1})), _module_digest(module({"k": 1})))
        self.assertNotEqual(_module_digest(module({"k": 1})), _module_digest(module({"k": 2})))
        with self.assertRaisesRegex(TypeError, "cannot be compared"):
            _module_digest(module(object()))

    def test_numpy_bytes_sets_and_complex_values_are_seen(self) -> None:
        # before: every one of these fell into the type-name branch, so two
        # different values gave the same digest (a difference went unnoticed)
        pairs = [
            (np.int64(1), np.int64(2)), (np.bool_(True), np.bool_(False)),
            (np.float32(1.5), np.float32(2.5)), (np.array([1.0]), np.array([2.0])),
            (np.array([1.0]), np.array([1.0], dtype=np.float32)), (np.array([[1, 2]]), np.array([1, 2])),
            (np.array([{"a": 1}], dtype=object), np.array([{"a": 2}], dtype=object)),
            (b"a", b"b"), ({1}, {2}), (frozenset({1}), frozenset({1, 2})), (1 + 2j, 1 + 3j),
            (1, np.int64(1)), (1, 1.0), (b"a", "a"),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertNotEqual(_state_digest(a), _state_digest(b))
        self.assertEqual(_state_digest({1, "x", 2.5}), _state_digest({2.5, 1, "x"}))
        # a structured scalar with an object field holds pointers: digest its values
        rec = [("n", np.int64), ("o", object)]
        first = np.array([(1, "a" * 3)], dtype=rec)[0]
        second = np.array([(1, "".join(["a"] * 3))], dtype=rec)[0]  # equal value, other str object
        self.assertNotEqual(first.tobytes(), second.tobytes())  # the premise: tobytes sees addresses
        self.assertEqual(_state_digest(first), _state_digest(second))
        self.assertNotEqual(_state_digest(first), _state_digest(np.array([(1, "b")], dtype=rec)[0]))
        self.assertEqual(_state_digest(np.array([3.0, -0.0])), _state_digest(np.array([3.0, -0.0])))

    def test_set_digest_is_independent_of_the_ranks_hash_seed(self) -> None:
        # str hashing is randomised per process (PYTHONHASHSEED), so a set of
        # strings iterates in another order on another rank: unsorted, equal
        # state would digest differently (a false refusal of a correct resume)
        import subprocess
        import sys

        code = ("import sys; sys.path.insert(0, %r); from runner_hex_v3 import _state_digest; "
                "print(_state_digest({'s': {'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', "
                "'theta', 'iota', 'kappa'}, 'f': frozenset({'x', 'y', 'z', 'w'})}).hex()); "
                "print(list({'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta'}))"
                % str(Path(__file__).resolve().parent))
        outputs = []
        for seed in ("1", "2", "3"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                  timeout=120, check=True)
            outputs.append(done.stdout.splitlines())
        self.assertEqual(len({out[0] for out in outputs}), 1)
        # the premise: the iteration order itself does differ between the seeds
        self.assertGreater(len({out[1] for out in outputs}), 1)

    def test_no_false_positive_across_real_optimizers_and_schedulers(self) -> None:
        """Equal state built twice (other object addresses) or loaded from a
        state_dict must digest equally, else a correct resume would be refused;
        one more step must be seen. 6 optimizers x 6 schedulers here; the wider
        6 x 11 matrix was measured on 2026-11-10 (gate log)."""
        from torch.optim import lr_scheduler as lrs

        optimizers = {
            "SGD": lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, nesterov=True),
            "Adam": lambda p: torch.optim.Adam(p, lr=0.01, amsgrad=True),
            "AdamW": lambda p: torch.optim.AdamW(p, lr=0.01),
            "RMSprop": lambda p: torch.optim.RMSprop(p, lr=0.01, centered=True),
            "Adagrad": lambda p: torch.optim.Adagrad(p, lr=0.01),
            "LBFGS": lambda p: torch.optim.LBFGS(p, lr=0.1),
        }
        schedulers = {
            "none": lambda o: None,
            "MultiStepLR": lambda o: lrs.MultiStepLR(o, [1, 2, 2]),
            "LambdaLR": lambda o: lrs.LambdaLR(o, lambda e: 0.9 ** e),
            "Plateau": lambda o: lrs.ReduceLROnPlateau(o),
            "Sequential": lambda o: lrs.SequentialLR(o, [lrs.LinearLR(o), lrs.ExponentialLR(o, 0.9)],
                                                     milestones=[2]),
            "Chained": lambda o: lrs.ChainedScheduler([lrs.ConstantLR(o), lrs.ExponentialLR(o, 0.9)]),
        }

        def build(opt_name: str, sched_name: str, steps: int) -> Any:
            torch.manual_seed(0)
            model = nn.Linear(3, 2, dtype=torch.float64)
            opt = optimizers[opt_name](model.parameters())
            sched = schedulers[sched_name](opt)
            x = torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(4, 3)

            def closure() -> torch.Tensor:
                opt.zero_grad()
                loss = model(x).pow(2).sum()
                loss.backward()
                return loss

            for _ in range(steps):
                opt.step(closure)
                if sched is not None:
                    sched.step(1.0) if sched_name == "Plateau" else sched.step()
            return opt, sched

        def digest(obj: Any) -> bytes:
            return _state_digest(None if obj is None else obj.state_dict())

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # scheduler-order warnings of the toy loop
            for opt_name in optimizers:
                for sched_name in schedulers:
                    with self.subTest(optimizer=opt_name, scheduler=sched_name):
                        a, sa = build(opt_name, sched_name, 3)
                        b, sb = build(opt_name, sched_name, 3)
                        c, sc = build(opt_name, sched_name, 0)
                        c.load_state_dict(a.state_dict())
                        if sc is not None:
                            sc.load_state_dict(sa.state_dict())
                        e, _ = build(opt_name, sched_name, 4)
                        self.assertEqual(digest(a), digest(b))
                        self.assertEqual(digest(sa), digest(sb))
                        self.assertEqual(digest(c), digest(a))
                        self.assertEqual(digest(sc), digest(sa))
                        self.assertNotEqual(digest(e), digest(a))

    def test_real_optimizer_scheduler_and_scaler_states(self) -> None:
        def states(momentum: float) -> List[bytes]:
            model = _model()
            opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=momentum)
            model(torch.ones(1, 3, dtype=torch.float64)).sum().backward()
            opt.step()
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
            return [_state_digest(opt.state_dict()), _state_digest(sched.state_dict()),
                    _state_digest(torch.amp.GradScaler("cpu").state_dict())]

        self.assertEqual(states(0.9), states(0.9))
        self.assertNotEqual(states(0.9)[0], states(0.8)[0])


# ------------------------------------------------------------------


class TestComparableState(unittest.TestCase):
    """R8 (cross-family GPT-6 R8-01): optimizer, scheduler and GradScaler states are
    compared between ranks only for built-in classes held by identity whose state
    determines their behaviour (spec section 6)."""

    REFUSED = "not one of the built-in classes"

    @staticmethod
    def _opt(cls: type = torch.optim.SGD, **kw: Any) -> torch.optim.Optimizer:
        return cls(nn.Linear(2, 1).parameters(), lr=0.1, **kw)

    def _refused(self, obj: Any, trusted: Any, pattern: str, **kw: Any) -> None:
        with self.assertRaisesRegex(DistributedContractError, pattern):
            _comparable_state(obj, trusted, "component", **kw)

    def test_forged_scheduler_class_is_refused(self) -> None:
        lrs = torch.optim.lr_scheduler
        nameonly = _forged(object, "StepLR", "torch.optim.lr_scheduler", state_dict=lambda self: {})
        subclass = _forged_step_lr()

        class Meta(type):  # the strongest forger: equal to StepLR, same hash
            def __eq__(cls, other: Any) -> bool:
                return True

            def __hash__(cls) -> int:
                return hash(lrs.StepLR)

        meta = Meta("StepLR", (lrs.StepLR,), {"__module__": "torch.optim.lr_scheduler"})

        class ClassLiar(lrs.StepLR):
            @property
            def __class__(self) -> type:  # type: ignore[override]
                return lrs.StepLR

        # premises: each forger fools the check it targets
        for cls in (nameonly, subclass, meta):
            self.assertEqual((cls.__module__, cls.__name__), ("torch.optim.lr_scheduler", "StepLR"))
        self.assertIn(meta, {lrs.StepLR})  # a set/dict lookup would admit it
        liar = ClassLiar(self._opt(), step_size=1)
        self.assertTrue(isinstance(liar, lrs.StepLR) and liar.__class__ is lrs.StepLR)
        for obj in (nameonly(), subclass(self._opt(), step_size=1), meta(self._opt(), step_size=1), liar):
            with self.subTest(forger=type(obj).__qualname__):
                self._refused(obj, _TRUSTED_SCHEDULERS, self.REFUSED)
        # negative control: the genuine class passes and yields its own state
        real = lrs.StepLR(self._opt(), step_size=1)
        self.assertEqual(_state_digest(self._admitted(real, _TRUSTED_SCHEDULERS)),
                         _state_digest(real.state_dict()))

    def test_class_method_patched_process_wide_is_refused(self) -> None:
        # Quella F2-A: `type(obj) is SGD` still holds after SGD.step is replaced on the class
        for owner, method in ((torch.optim.SGD, "step"), (torch.optim.Optimizer, "state_dict"),
                              (torch.optim.lr_scheduler.StepLR, "get_lr")):
            with self.subTest(patched=f"{owner.__qualname__}.{method}"):
                original = owner.__dict__[method]
                obj = (torch.optim.lr_scheduler.StepLR(self._opt(), step_size=1)
                       if method == "get_lr" else self._opt())
                trusted = _TRUSTED_SCHEDULERS if method == "get_lr" else _TRUSTED_OPTIMIZERS
                self._admitted(obj, trusted, tolerated=frozenset({"step"}))  # before the patch
                try:
                    setattr(owner, method, lambda self, *a, **k: None)
                    self.assertTrue(any(type(obj) is c for c in trusted))  # identity alone still passes
                    self._refused(obj, trusted, "replaced since import", tolerated=frozenset({"step"}))
                finally:
                    setattr(owner, method, original)
        # strongest forger: torch's own step wrapper (same code object) around a foreign function
        opt = self._opt()
        original = torch.optim.SGD.__dict__["step"]
        try:
            forged = torch.optim.Optimizer.profile_hook_step(lambda self, *a, **k: None)
            forged.hooked = True  # type: ignore[attr-defined]
            torch.optim.SGD.step = forged  # type: ignore[method-assign]
            self.assertIs(forged.__code__, original.__code__)  # the wrapper is indistinguishable by code
            self._refused(opt, _TRUSTED_OPTIMIZERS, "replaced since import", tolerated=frozenset({"step"}))
        finally:
            torch.optim.SGD.step = original  # type: ignore[method-assign]
        # negative control: a patch on a base method StepLR overrides does not reach
        # StepLR's behaviour and must not cause a false refusal
        base = torch.optim.lr_scheduler.LRScheduler
        original = base.__dict__["get_lr"]
        try:
            base.get_lr = lambda self: []  # type: ignore[method-assign]
            self._admitted(torch.optim.lr_scheduler.StepLR(self._opt(), step_size=1), _TRUSTED_SCHEDULERS)
        finally:
            base.get_lr = original  # type: ignore[method-assign]
        # negative control: after restoring, the genuine classes pass again
        self._admitted(self._opt(), _TRUSTED_OPTIMIZERS, tolerated=frozenset({"step"}))
        self._admitted(torch.optim.lr_scheduler.StepLR(self._opt(), step_size=1), _TRUSTED_SCHEDULERS)

    def test_optimizer_load_state_dict_hook_is_refused(self) -> None:
        for register in ("register_load_state_dict_pre_hook", "register_load_state_dict_post_hook"):
            with self.subTest(hook=register):
                opt = self._opt()
                getattr(opt, register)(lambda *a, **k: None)
                self._refused(opt, _TRUSTED_OPTIMIZERS, "state_dict hooks", tolerated=frozenset({"step"}))

    def _admitted(self, obj: Any, trusted: Any, **kw: Any) -> Any:
        """A positive control: a refusal here is a false refusal, reported as a test failure."""
        try:
            return _comparable_state(obj, trusted, "component", **kw)
        except DistributedContractError as exc:
            self.fail(f"false refusal of a built-in {type(obj).__qualname__}: {exc}")

    def test_every_trusted_component_is_compared(self) -> None:
        # positive control over the whole allowlist: no false refusal of a built-in
        built = {name: make(self._opt(), 0) for name, make in _SCHEDULER_BUILDERS.items()}
        self.assertEqual({type(s) for s in built.values()}, set(_TRUSTED_SCHEDULERS))
        for name, sched in built.items():
            with self.subTest(scheduler=name):
                self.assertEqual(self._admitted(sched, _TRUSTED_SCHEDULERS).keys(), sched.state_dict().keys())
        for cls in _TRUSTED_OPTIMIZERS:
            with self.subTest(optimizer=cls.__name__):
                opt = _build_optimizer(cls, 0.1)[1]
                torch.optim.lr_scheduler.StepLR(opt, step_size=1)  # torch puts step on the instance
                self.assertIn("step", vars(opt))
                self._admitted(opt, _TRUSTED_OPTIMIZERS, tolerated=_OPTIMIZER_INSTANCE_OVERRIDES_TOLERATED)
        self._admitted(_cpu_scaler(), _TRUSTED_GRAD_SCALERS)
        self.assertIsNone(_comparable_state(None, _TRUSTED_SCHEDULERS, "scheduler"))

    def test_trusted_scheduler_state_determines_the_learning_rates(self) -> None:
        """The oracle behind the allowlist: load a state into a scheduler built with
        other arguments, step on -- the learning rates must be the original's."""
        for name, make in _SCHEDULER_BUILDERS.items():
            with self.subTest(scheduler=name):
                original, other = _lr_continuations(make)
                self.assertEqual(original, other)

    def test_lambda_multiplicative_and_custom_cyclic_lr_are_refused(self) -> None:
        lrs = torch.optim.lr_scheduler
        untrusted = {
            "LambdaLR": lambda o, k: lrs.LambdaLR(o, [lambda e: 0.9 ** e, lambda e: 0.8 ** e][k]),
            "MultiplicativeLR": lambda o, k: lrs.MultiplicativeLR(o, [lambda e: 0.9, lambda e: 0.8][k]),
            "CyclicLR": lambda o, k: lrs.CyclicLR(o, 0.01, 0.1, scale_fn=[lambda x: 1.0, lambda x: 0.5][k]),
        }
        for name, make in untrusted.items():
            with self.subTest(scheduler=name):
                original, other = _lr_continuations(make)
                self.assertNotEqual(original, other)  # premise: equal state, other learning rates
                self._refused(make(self._opt(), 0), _TRUSTED_SCHEDULERS, self.REFUSED)

    def test_instance_override_of_state_export_or_behaviour_is_refused(self) -> None:
        lrs = torch.optim.lr_scheduler
        for attr, value in (("state_dict", lambda: {}), ("get_lr", lambda: [0.5]), ("step", lambda *a: None)):
            sched = lrs.StepLR(self._opt(), step_size=1)
            setattr(sched, attr, value)
            with self.subTest(scheduler_override=attr):
                self._refused(sched, _TRUSTED_SCHEDULERS, f"overrides {attr}")
        opt = self._opt()
        opt.state_dict = lambda: {}
        self._refused(opt, _TRUSTED_OPTIMIZERS, "overrides state_dict",
                      tolerated=_OPTIMIZER_INSTANCE_OVERRIDES_TOLERATED)
        for register in ("register_state_dict_pre_hook", "register_state_dict_post_hook"):
            opt = self._opt()
            getattr(opt, register)(lambda *a: None)
            with self.subTest(hook=register):
                self._refused(opt, _TRUSTED_OPTIMIZERS, "state_dict hooks",
                              tolerated=_OPTIMIZER_INSTANCE_OVERRIDES_TOLERATED)
        # negative control: plain instance data (no method shadowed) is not refused
        sched = lrs.StepLR(self._opt(), step_size=1)
        sched.note = "user data"
        self._admitted(sched, _TRUSTED_SCHEDULERS)

    def test_composite_scheduler_with_forged_or_overridden_child_is_refused(self) -> None:
        lrs = torch.optim.lr_scheduler
        forged_exp = _forged(lrs.ExponentialLR, "ExponentialLR", "torch.optim.lr_scheduler")
        for composite in ("ChainedScheduler", "SequentialLR"):
            def build(child: Callable[[Any], Any]) -> Any:
                o = self._opt()
                inner = [lrs.ConstantLR(o), child(o)]
                return (lrs.ChainedScheduler(inner) if composite == "ChainedScheduler"
                        else lrs.SequentialLR(o, inner, milestones=[2]))

            with self.subTest(composite, child="genuine"):
                self._admitted(build(lambda o: lrs.ExponentialLR(o, 0.9)), _TRUSTED_SCHEDULERS)
            with self.subTest(composite, child="forged"):
                self._refused(build(lambda o: forged_exp(o, 0.9)), _TRUSTED_SCHEDULERS,
                              f"inside {composite}.*{self.REFUSED}")
            with self.subTest(composite, child="lambda"):
                self._refused(build(lambda o: lrs.LambdaLR(o, lambda e: 1.0)), _TRUSTED_SCHEDULERS,
                              f"inside {composite}")
            overridden = build(lambda o: lrs.ExponentialLR(o, 0.9))
            overridden._schedulers[1].get_lr = lambda: [0.5]
            with self.subTest(composite, child="instance override"):
                self._refused(overridden, _TRUSTED_SCHEDULERS, "overrides get_lr")
            hidden = build(lambda o: lrs.ExponentialLR(o, 0.9))
            hidden._schedulers = type("L", (list,), {})(hidden._schedulers)
            with self.subTest(composite, child="foreign container"):
                self._refused(hidden, _TRUSTED_SCHEDULERS, "unknown container")

    def test_forged_optimizer_and_grad_scaler_are_refused(self) -> None:
        forged_sgd = _forged(torch.optim.SGD, "SGD", "torch.optim.sgd")
        self._refused(forged_sgd(nn.Linear(2, 1).parameters(), lr=0.1), _TRUSTED_OPTIMIZERS, self.REFUSED,
                      tolerated=_OPTIMIZER_INSTANCE_OVERRIDES_TOLERATED)
        forged_scaler = _forged(torch.amp.GradScaler, "GradScaler", "torch.amp.grad_scaler")
        self._refused(forged_scaler("cpu"), _TRUSTED_GRAD_SCALERS, self.REFUSED)
        # the tolerated instance-level step exists for optimizers only
        sched = torch.optim.lr_scheduler.StepLR(self._opt(), step_size=1)
        sched.step = lambda *a: None
        self._refused(sched, _TRUSTED_SCHEDULERS, "overrides step")

    def test_trusted_optimizer_state_determines_the_update(self) -> None:
        """Oracle for the optimizer allowlist: two optimizers built with other
        learning rates, both loaded with one saved state, step alike."""
        for cls in _TRUSTED_OPTIMIZERS:
            with self.subTest(optimizer=cls.__name__):
                model, opt, closure, x = _build_optimizer(cls, 0.05)
                for _ in range(3):
                    opt.step(closure)
                saved = _roundtrip({"model": model.state_dict(), "opt": opt.state_dict()})
                params = []
                for lr in (0.05, 0.5):
                    m, o, c, _ = _build_optimizer(cls, lr)
                    m.load_state_dict(_roundtrip(saved["model"]))
                    o.load_state_dict(_roundtrip(saved["opt"]))  # a copy: the step tensor is not copied by torch
                    for _ in range(3):
                        o.step(c)
                    params.append([p.detach().clone() for p in m.parameters()])
                for a, b in zip(*params):
                    self.assertTrue(torch.equal(a, b))


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

    def test_shared_state_refuses_untrusted_optimizer_scheduler_and_scaler(self) -> None:
        # R8 at the runner: each component goes through the identity check, at
        # construction (the initial agreement) and whenever the digests are built
        with tempfile.TemporaryDirectory() as tmp:
            model = DDP(_model())
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            with self.assertRaisesRegex(DistributedContractError, "scheduler: .*LambdaLR is not one"):
                HexTrainingRunner(_config(Path(tmp) / "a", distributed=True), model, opt,
                                  DataLoader(_Data(8, seed=1), batch_size=4),
                                  DataLoader(_Data(5, seed=2), batch_size=2),
                                  scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0))
            runner = self._runner(Path(tmp) / "b", distributed=True)
            self.assertEqual(len(runner._shared_state_digests()), 5)  # positive control
            forged_sgd = _forged(torch.optim.SGD, "SGD", "torch.optim.sgd")
            components = (
                ("optimizer", forged_sgd(runner.raw_model.parameters(), lr=0.1)),
                ("scheduler", _forged_step_lr()(runner.optimizer, step_size=1)),
                ("grad_scaler", _forged(torch.amp.GradScaler, "GradScaler", "torch.amp.grad_scaler")("cpu")),
            )
            for attr, forged in components:
                saved = getattr(runner, attr)
                setattr(runner, attr, forged)
                with self.subTest(attr), self.assertRaisesRegex(DistributedContractError, "is not one of the built-in"):
                    runner._shared_state_digests()
                setattr(runner, attr, saved)

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

    def test_mid_epoch_rank_buffers_must_match_the_model(self) -> None:
        # broadcast_buffers=False: the mid-epoch record carries every rank's
        # buffers; a record whose buffers are not the model's is refused inside
        # the agreed resume (a KeyError in the copy would hit one rank only)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            kw = {"model_fn": _RunningScale, "ddp_kw": {"broadcast_buffers": False}}
            with self.assertRaises(_Preempted):
                _mid_runner(0, 1, out, crash_at=3, **kw).run()
            path = out / "checkpoints" / "resume_last.pt"
            payload = torch.load(path, weights_only=False)
            buffers = payload["extra"]["mid_epoch"]["rank_buffers"]
            self.assertEqual(sorted(buffers), ["scale"])
            self.assertEqual(len(buffers["scale"]), 1)
            buffers["other"] = buffers.pop("scale")
            torch.save(payload, path)
            resumed = _mid_runner(0, 1, out, resume_from=str(path), resume_trusted=True, **kw)
            with self.assertRaisesRegex(ValueError, r"do not match the model's persistent buffers \['scale'\]"):
                resumed.run()

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

    def test_error_after_backward_is_held_until_the_next_agreement(self) -> None:
        # Spec section 3.6: the step returns, the next agreement point raises.
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True, num_epochs=1)

            def oom() -> float:
                raise RuntimeError("CUDA out of memory in optimizer.step (injected)")

            runner._apply_update = oom
            batch = next(iter(runner.train_loader))
            loss_val, healthy, grad_norm = runner._train_one_step(batch)
            self.assertFalse(healthy)
            self.assertIsNone(grad_norm)
            self.assertIsInstance(runner._post_step_error, RuntimeError)
            # the OOM got the rescue before it was held
            self.assertTrue(list(runner.failure_dir.glob("*CUDA_OOM*.json")))
            runner.global_step = 1
            with self.assertRaisesRegex(RuntimeError, "optimizer.step"):
                runner._agreed_next(iter(runner.train_loader))
            self.assertIsNone(runner._post_step_error)  # raised once, not again

    def test_own_error_survives_a_failed_rank_identification(self) -> None:
        # Codex P2 on #97: the failure flag is agreed, then the one-hot
        # all-reduce naming the failing ranks fails (a peer disconnects).
        # The caller must still see this rank's exception, and its log must
        # already hold the traceback.
        original = self.ctx.sum_ints

        def broken(values: Any) -> Any:
            raise RuntimeError("Connection closed by peer during identification (injected)")

        self.ctx.sum_ints = broken  # any_true (the flag agreement) still works
        try:
            with self.assertLogs("ddp_hex_v3", level="ERROR") as logs:
                with self.assertRaisesRegex(KeyError, "boom") as caught:
                    self.ctx.run_agreed(lambda: {}["boom"], what="t")
        finally:
            self.ctx.sum_ints = original
        self.assertIn("the agreement on this error failed too (RuntimeError: Connection closed",
                      " ".join(caught.exception.__notes__))
        self.assertIn("Traceback", logs.output[0])

    def test_unscale_error_is_held_too(self) -> None:
        # GradScaler.unscale_ runs after DDP's gradient all-reduce as well.
        class _Scaler:
            def scale(self, loss: torch.Tensor) -> torch.Tensor:
                return loss

            def unscale_(self, optimizer: Any) -> None:
                raise RuntimeError("unscale_ failed (injected)")

        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True, num_epochs=1)
            runner.grad_scaler = _Scaler()
            with self.assertLogs("runner_hex_v3", level="ERROR") as logs:
                _, healthy, _ = runner._train_one_step(next(iter(runner.train_loader)))
            self.assertFalse(healthy)
            self.assertIn("held until the next agreement point", " ".join(logs.output))
            self.assertRegex(str(runner._post_step_error), "unscale_ failed")
            runner.global_step = 1
            with self.assertRaisesRegex(RuntimeError, "unscale_ failed") as caught:
                runner._agreed_next(iter(runner.train_loader))
            self.assertIn("training step 0 after backward", " ".join(caught.exception.__notes__))

    def test_error_recording_a_skipped_update_is_held_too(self) -> None:
        # HX102-01: the capsule and FailureSignature of a skipped update are written
        # after DDP's gradient all-reduce as well; an OSError there (a full disk) on
        # one rank must be held and agreed like an optimizer error, not raised alone.
        def full(*args: Any, **kwargs: Any) -> None:
            raise OSError(errno.ENOSPC, "No space left on device (injected)")

        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True, num_epochs=1)
            runner._apply_update = lambda: (float("nan"), False)
            with mock.patch.object(FailureSignature, "register", full):
                with self.assertLogs("runner_hex_v3", level="ERROR") as logs:
                    _, healthy, grad_norm = runner._train_one_step(next(iter(runner.train_loader)))
            self.assertFalse(healthy)
            self.assertIsNone(grad_norm)
            self.assertIn("held until the next agreement point", " ".join(logs.output))
            self.assertIsInstance(runner._post_step_error, OSError)
            runner.global_step = 1
            with self.assertRaisesRegex(OSError, "No space left") as caught:
                runner._agreed_next(iter(runner.train_loader))
            self.assertIn("training step 0 after backward", " ".join(caught.exception.__notes__))
            self.assertIsNone(runner._post_step_error)

    def test_held_error_survives_a_failed_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True, num_epochs=1)
            runner._post_step_error = ValueError("optimizer failed (injected)")

            def broken(values: Any) -> Any:
                raise RuntimeError("Application timeout caused pair closure (injected)")

            runner.dist.sum_ints = broken
            try:
                with self.assertRaisesRegex(ValueError, "optimizer failed") as caught:
                    runner._agreed_next(iter(runner.train_loader))
            finally:
                del runner.dist.sum_ints
            self.assertIn("the agreement on this error failed too", " ".join(caught.exception.__notes__))

    def test_error_after_backward_stops_the_mid_epoch_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True, num_epochs=1, save_every_n_steps=1)

            def broken() -> float:
                raise ValueError("EMA update failed (injected)")

            runner._apply_update = broken
            with self.assertRaisesRegex(ValueError, "EMA update failed"):
                runner.run()
            self.assertFalse((Path(tmp) / "checkpoints" / "resume_last.pt").exists())

    def test_oom_rescue_never_raises(self) -> None:
        # Review round 3: a rescue that raised would leave the agreement early.
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp), distributed=True)

            def oom(batch: Any) -> Any:
                raise RuntimeError("CUDA out of memory (injected)")

            def broken_gc() -> int:
                raise MemoryError("gc failed (injected)")

            runner._prepare_step = oom
            original_gc = gc.collect
            gc.collect = broken_gc
            try:
                with self.assertRaisesRegex(RuntimeError, "out of memory"):
                    runner.run()
            finally:
                gc.collect = original_gc
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
        with self.assertLogs("ddp_hex_v3", level="ERROR") as logs:
            with self.assertRaisesRegex(KeyError, "boom") as caught:
                self.ctx.run_agreed(lambda: {}["boom"], what="t")
        self.assertIn("[rank 0] t: agreed failure on rank(s) [0]; every rank raises", caught.exception.__notes__)
        # "see their logs": the failing rank logged it, with traceback
        self.assertIn("rank 0: t failed", logs.output[0])
        self.assertIn("Traceback", logs.output[0])

    def test_failed_ranks_is_a_one_hot_sum(self) -> None:
        self.assertEqual(self.ctx.failed_ranks(True), [0])
        self.assertEqual(self.ctx.failed_ranks(False), [])

    def test_own_error_survives_a_failed_agreement(self) -> None:
        # The agreement collective itself fails (a peer died, the timeout): the
        # caller must still see this rank's exception, the root cause.
        original = self.ctx.any_true

        def broken(flag: bool) -> bool:
            raise RuntimeError("Connection closed by peer (injected)")

        self.ctx.any_true = broken
        try:
            with self.assertRaisesRegex(KeyError, "boom") as caught:
                self.ctx.run_agreed(lambda: {}["boom"], what="t")
            notes = " ".join(caught.exception.__notes__)
            self.assertIn("the agreement on this error failed too (RuntimeError: Connection closed", notes)
            self.assertIsInstance(caught.exception.__context__, RuntimeError)
            # without an own error the transport error itself propagates
            with self.assertRaisesRegex(RuntimeError, "Connection closed"):
                self.ctx.run_agreed(lambda: 7, what="t")
        finally:
            self.ctx.any_true = original

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
class _BNDrop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(3, 8, dtype=torch.float64), nn.BatchNorm1d(8, dtype=torch.float64),
                                 nn.Tanh(), nn.Dropout(0.3), nn.Linear(8, 2, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _RunningScale(nn.Module):
    """A buffer the training forward reads and then updates from the local batch."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(3, 2, dtype=torch.float64)
        self.register_buffer("scale", torch.ones((), dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin(x) * self.scale.clone()
        if self.training:
            with torch.no_grad():
                self.scale.mul_(0.5).add_(0.5 * x.abs().mean())
        return out


class _Preempted(RuntimeError):
    pass


def _mid_runner(rank: int, world: int, directory: Path, *, crash_at: int = -1,
                model_fn: Callable[[], nn.Module] = _BNDrop, ddp_kw: Optional[Dict[str, Any]] = None,
                **kw: Any) -> HexTrainingRunner:
    model = DDP(model_fn(), **(ddp_kw or {}))
    torch.manual_seed(1000 + rank)
    r = HexTrainingRunner(
        _config(directory, distributed=True, is_main_process=rank == 0, num_epochs=4,
                save_every_n_steps=3, **kw),
        model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        _train_loader(rank, world, shuffle=True), _exact_val_loader(),
    )
    if crash_at >= 0:
        original = r._train_one_step

        def crashing(batch: Any, prepared: Any = None) -> Any:
            if r.global_step == crash_at:  # every rank: a preemption of the whole job
                raise _Preempted("injected preemption")
            return original(batch, prepared)

        r._train_one_step = crashing
    return r


def _mid_epoch_pair(rank: int, world: int, out: Path, **kw: Any) -> Dict[str, Any]:
    """Uninterrupted run vs crash after the mid-epoch save at step 3 (epoch 1) + resume."""
    full = _mid_runner(rank, world, out / "full", **kw)
    reference = full.run()
    try:
        _mid_runner(rank, world, out / "split", crash_at=3, **kw).run()
    except _Preempted:
        pass
    resumed = _mid_runner(rank, world, out / "split", resume_trusted=True,
                          resume_from=str(out / "split" / "checkpoints" / "resume_last.pt"), **kw)
    result = resumed.run()
    return {
        # this rank's own model: parameters, buffers and EMA
        "full": _digest(full.raw_model) + _buffer_digest(full.raw_model) + _buffer_digest(full.ema_model),
        "resumed": _digest(resumed.raw_model) + _buffer_digest(resumed.raw_model)
        + _buffer_digest(resumed.ema_model),
        "batches_done": result["resume_report"]["mid_epoch_batches_done"],
        "same_reports": result["history"]["train"] == reference["history"]["train"][1:]
        and result["history"]["val"] == reference["history"]["val"][1:],
        "same_history": list(resumed.anomaly_detector.loss_history) == list(full.anomaly_detector.loss_history),
        "history_len": len(full.anomaly_detector.loss_history),
    }


def _sc_mid_epoch_resume(rank: int, world: int, out: Path) -> Dict[str, Any]:
    """Mid-epoch resume under DDP: BatchNorm + dropout + rank seeds, crash inside epoch 1."""
    return _mid_epoch_pair(rank, world, out)


def _sc_mid_epoch_rank_buffers(rank: int, world: int, out: Path) -> Dict[str, Any]:
    """broadcast_buffers=False and a buffer the training forward reads: each rank resumes its own."""
    return _mid_epoch_pair(rank, world, out, model_fn=_RunningScale, ddp_kw={"broadcast_buffers": False})


def _sc_resume_other_position_on_rank1(rank: int, world: int, out: Path) -> Any:
    """Same weights and step, rank 1's copy one batch earlier: the resume agreement refuses."""
    try:
        _mid_runner(rank, world, out, crash_at=3).run()
    except _Preempted:
        pass
    path = out / "checkpoints" / "resume_last.pt"
    if rank == 1:
        payload = torch.load(path, weights_only=False)
        mid = payload["extra"]["mid_epoch"]
        mid["batches_done"] -= 1  # still consistent: one healthy loss less per rank
        for key in ("losses", "grad_norms"):
            mid[key] = [part[:-1] for part in mid[key]]
        path = out / "rank1_copy.pt"
        torch.save(payload, path)
    _mid_runner(rank, world, out, resume_from=str(path), resume_trusted=True).run()


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


def _cpu_scaler() -> torch.amp.GradScaler:
    # The runner builds a GradScaler only on CUDA; torch >= 2.3 has the same
    # scaler on the CPU, so AMP's scaler path under DDP is testable with gloo.
    return torch.amp.GradScaler("cpu", init_scale=1024.0, growth_interval=2)


# ------------------------------------------------------------------
# R8: trusted classes by identity (TestComparableState and the 2-rank job)
def _forged(base: type, name: str, module: str, **body: Any) -> type:
    """A class carrying a built-in's name and module (cross-family GPT-6 R8-01:
    ``type('StepLR', ..., {'__module__': 'torch.optim.lr_scheduler'})``)."""
    return type(name, (base,), {"__module__": module, "__qualname__": name, **body})


def _forged_step_lr() -> type:
    """Named torch.optim.lr_scheduler.StepLR, StepLR's state, half its learning rates."""
    step_lr = torch.optim.lr_scheduler.StepLR

    def get_lr(self: Any) -> List[float]:
        return [0.5 * lr for lr in step_lr.get_lr(self)]

    return _forged(step_lr, "StepLR", "torch.optim.lr_scheduler", get_lr=get_lr)


def _roundtrip(state: Any) -> Any:
    """Through torch.save/torch.load like a checkpoint: no tensor shared with the source."""
    import io

    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


def _scheduler_builders() -> Dict[str, Callable[[torch.optim.Optimizer, int], Any]]:
    """One builder per trusted scheduler; ``k`` selects other constructor arguments."""
    lrs = torch.optim.lr_scheduler
    return {
        "StepLR": lambda o, k: lrs.StepLR(o, 2 + k, gamma=0.5 + 0.1 * k),
        "MultiStepLR": lambda o, k: lrs.MultiStepLR(o, [1 + k, 3], gamma=0.5 + 0.1 * k),
        "ConstantLR": lambda o, k: lrs.ConstantLR(o, factor=0.3 + 0.1 * k, total_iters=3 + k),
        "LinearLR": lambda o, k: lrs.LinearLR(o, 0.2 + 0.1 * k, total_iters=4 + k),
        "ExponentialLR": lambda o, k: lrs.ExponentialLR(o, 0.9 - 0.1 * k),
        "PolynomialLR": lambda o, k: lrs.PolynomialLR(o, 6 + k, power=1.0 + k),
        "CosineAnnealingLR": lambda o, k: lrs.CosineAnnealingLR(o, 5 + k, eta_min=0.001 * k),
        "CosineAnnealingWarmRestarts": lambda o, k: lrs.CosineAnnealingWarmRestarts(o, 3 + k, T_mult=1 + k),
        "ReduceLROnPlateau": lambda o, k: lrs.ReduceLROnPlateau(o, patience=1 + k, factor=0.5 + 0.1 * k),
        "OneCycleLR": lambda o, k: lrs.OneCycleLR(o, 1.0 + k, total_steps=20 + k,
                                                  anneal_strategy=("cos", "linear")[k]),
        "SequentialLR": lambda o, k: lrs.SequentialLR(
            o, [lrs.LinearLR(o, 0.2 + 0.1 * k), lrs.ExponentialLR(o, 0.9 - 0.1 * k)], [2 + k]),
        "ChainedScheduler": lambda o, k: lrs.ChainedScheduler(
            [lrs.ConstantLR(o, 0.3 + 0.1 * k), lrs.ExponentialLR(o, 0.9 - 0.1 * k)]),
    }


_SCHEDULER_BUILDERS = _scheduler_builders()


def _lr_continuations(make: Callable[[torch.optim.Optimizer, int], Any]) -> List[List[float]]:
    """Learning rates of 3 steps after loading one saved (optimizer, scheduler) state into
    a scheduler built like the original (k=0) and one built with other arguments (k=1)."""
    import warnings

    def run(k: int, steps: int, saved: Optional[Any] = None) -> Any:
        opt = torch.optim.SGD(nn.Linear(2, 1).parameters(), lr=0.1, momentum=0.9)
        sched = make(opt, k)
        if saved is not None:
            opt.load_state_dict(_roundtrip(saved[0]))
            sched.load_state_dict(_roundtrip(saved[1]))
        lrs = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # optimizer.step without gradients, toy loop
            for _ in range(steps):
                opt.step()
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(1.0)
                else:
                    sched.step()
                lrs.append(opt.param_groups[0]["lr"])
        return opt, sched, lrs

    opt, sched, _ = run(0, 3)
    saved = (opt.state_dict(), sched.state_dict())
    return [run(k, 3, saved)[2] for k in (0, 1)]


def _build_optimizer(cls: type, lr: float) -> Any:
    """(model, optimizer, closure, input) for one trusted optimizer class, float64."""
    torch.manual_seed(0)
    if cls is getattr(torch.optim, "SparseAdam", None):
        model: nn.Module = nn.Embedding(5, 3, sparse=True, dtype=torch.float64)
        x = torch.tensor([0, 2, 4])
    else:
        model = nn.Linear(3, 2, dtype=torch.float64)
        x = torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(4, 3)
    opt = cls(model.parameters(), lr=lr)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = model(x).pow(2).sum()
        loss.backward()
        return loss

    return model, opt, closure, x


def _scheduler_differs(kind: str) -> Callable[[int, int, Path], Any]:
    """R8 (cross-family GPT-6 R8-01, A7-R2): equal scheduler state, other behaviour on
    rank 1. ``kind="lambda"``: LambdaLR with another function on rank 1 (the function is
    not in the state); ``kind="forged"``: rank 1's scheduler is a class named
    torch.optim.lr_scheduler.StepLR that halves the learning rate. Before, both
    digests agreed and both ranks trained to the end with diverging weights."""

    def scenario(rank: int, world: int, out: Path) -> Any:
        model = DDP(_model())
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        if kind == "lambda":
            factor = 0.8 if rank == 1 else 0.9
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: factor ** e)
        else:
            cls = _forged_step_lr() if rank == 1 else torch.optim.lr_scheduler.StepLR
            sched = cls(opt, step_size=1, gamma=0.5)
        runner = HexTrainingRunner(
            _config(out, distributed=True, is_main_process=rank == 0, num_epochs=2),
            model, opt, _train_loader(rank, world), _exact_val_loader(), scheduler=sched,
        )
        runner.run()
        return {"digest": _digest(runner.raw_model)}

    return scenario


def _resume_with_rank1_copy(mutate: Callable[[Dict[str, Any]], None], *,
                            scaler: bool = False) -> Callable[[int, int, Path], Any]:
    """Rank 1 resumes from a copy of rank 0's file whose weights are equal but
    whose ``mutate``-d state is not (a stale copy, no shared file system).

    Before the fix only the weights were compared: the run went on and the
    ranks applied the same averaged gradients differently from then on.
    """

    def scenario(rank: int, world: int, out: Path) -> Any:
        first = _ddp_runner(rank, out, train=_train_loader(rank, world), val=_exact_val_loader(), num_epochs=1)
        if scaler:
            first.grad_scaler = _cpu_scaler()
        first.run()
        path = out / "checkpoints" / "resume_last.pt"
        if rank == 1:
            payload = torch.load(path, weights_only=False)
            mutate(payload)
            other = out / "rank1_copy.pt"
            torch.save(payload, other)
            path = other
        second = _ddp_runner(rank, out, train=_train_loader(rank, world), val=_exact_val_loader(), num_epochs=3,
                             resume_from=str(path), resume_trusted=True)
        if scaler:
            second.grad_scaler = _cpu_scaler()
        second.run()
        return {"digest": _digest(second.raw_model)}

    return scenario


def _construction_differs(part: str) -> Callable[[int, int, Path], Any]:
    """Rank 1 builds its optimizer with another learning rate (``part="optimizer"``)
    or its scheduler with another step size (``part="scheduler"``), with the
    same RunnerConfig. Before the check both ranks trained to the end and
    diverged: DDP averages the gradients, each rank applied them its own way."""

    def scenario(rank: int, world: int, out: Path) -> Any:
        model = DDP(_model())
        lr = 0.2 if (part == "optimizer" and rank == 1) else 0.1
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        step_size = 2 if (part == "scheduler" and rank == 1) else 1
        runner = HexTrainingRunner(
            _config(out, distributed=True, is_main_process=rank == 0, num_epochs=2),
            model, opt, _train_loader(rank, world), _exact_val_loader(),
            scheduler=torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.5),
        )
        runner.run()
        return {"digest": _digest(runner.raw_model)}

    return scenario


def _sc_state_dict_error_rank1(rank: int, world: int, out: Path) -> Any:
    """Round 3: building the shared-state digest fails on rank 1 only (here: its
    scheduler carries an instance-level state_dict that raises; since R8 that
    override is refused before it runs). Unagreed, rank 1 left while rank 0
    waited in the digest broadcast until the process-group timeout."""
    model = DDP(_model())
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    if rank == 1:
        def broken() -> Dict[str, Any]:
            raise _StepFailure("scheduler state_dict failed (injected)")

        sched.state_dict = broken
    t0 = time.monotonic()
    try:
        HexTrainingRunner(_config(out, distributed=True, is_main_process=rank == 0, num_epochs=1),
                          model, opt, _train_loader(rank, world), _exact_val_loader(), scheduler=sched)
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc), "seconds": time.monotonic() - t0}
    return {"error": None, "message": "", "seconds": time.monotonic() - t0}


def _bump_momentum(payload: Dict[str, Any]) -> None:
    for state in payload["optimizer_state_dict"]["state"].values():
        state["momentum_buffer"].add_(1.0)


def _halve_scale(payload: Dict[str, Any]) -> None:
    payload["grad_scaler_state_dict"]["scale"] = payload["grad_scaler_state_dict"]["scale"] / 2


def _move_ema(payload: Dict[str, Any]) -> None:
    for value in payload["ema_state_dict"].values():
        if torch.is_tensor(value) and value.is_floating_point():
            value.add_(1.0)


def _amp_run(rank: int, world: int, out: Path, *, poison_rank: int = -1, poison_call: int = -1) -> Dict[str, Any]:
    """A CPU GradScaler on every rank; ``poison_rank`` gets a NaN gradient in its
    ``poison_call``-th training step (the all-reduce spreads it to every rank)."""
    calls = {"n": 0}
    mse = nn.MSELoss()

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.requires_grad:
            calls["n"] += 1
            if rank == poison_rank and calls["n"] == poison_call:
                pred = _PoisonGrad.apply(pred)
        return mse(pred, target)

    model = DDP(_model())
    runner = HexTrainingRunner(
        _config(out, distributed=True, is_main_process=rank == 0),
        model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        _train_loader(rank, world), _exact_val_loader(), loss_fn=loss_fn,
    )
    runner.grad_scaler = _cpu_scaler()
    result = runner.run()
    return {
        "params": _flat_params(runner.raw_model),
        "ema": _flat_params(runner.ema_model.module),
        "digest": _digest(runner.raw_model),
        "scale": runner.grad_scaler.get_scale(),
        "mean_loss": [t["mean_loss"] for t in result["history"]["train"]],
        "n_anomalies": [t["n_anomalies"] for t in result["history"]["train"]],
        "failure_files": sorted(p.name for p in runner.failure_dir.glob("*.json")),
    }


def _amp_single_process_reference(out: Path) -> Dict[str, Any]:
    """The run the 2-rank AMP job must reproduce: global batch 4, the same CPU GradScaler."""
    model = _model()
    runner = HexTrainingRunner(
        _config(out), model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        DataLoader(_Data(8, seed=1), batch_size=4), DataLoader(_Data(5, seed=2), batch_size=5),
    )
    runner.grad_scaler = _cpu_scaler()
    result = runner.run()
    return {
        "params": _flat_params(model),
        "ema": _flat_params(runner.ema_model.module),
        "scale": runner.grad_scaler.get_scale(),
        "mean_loss": [t["mean_loss"] for t in result["history"]["train"]],
    }


def _sc_amp_equivalence(rank: int, world: int, out: Path) -> Dict[str, Any]:
    return _amp_run(rank, world, out)


def _sc_amp_nan_grad_on_rank1(rank: int, world: int, out: Path) -> Dict[str, Any]:
    return _amp_run(rank, world, out, poison_rank=1, poison_call=2)


class _StepFailure(RuntimeError):
    """Injected error after backward (optimizer step / EMA update)."""


def _failing_sgd(params: Any, *, fail_at: int, **kw: Any) -> torch.optim.SGD:
    """A genuine SGD whose ``fail_at``-th step raises (1-based; -1 never).

    Injected by a step pre-hook, not a subclass: since R8 the shared-state check
    refuses an optimizer that is not exactly a built-in class."""
    opt = torch.optim.SGD(params, **kw)
    calls = [0]

    def fail(optimizer: Any, args: Any, kwargs: Any) -> None:
        calls[0] += 1
        if calls[0] == fail_at:
            raise _StepFailure(f"optimizer step {calls[0]} failed (injected)")

    opt.register_step_pre_hook(fail)
    return opt


def _post_step_outcome(runner: HexTrainingRunner, out: Path) -> Dict[str, Any]:
    """Run, expecting a failure; report it and whether any checkpoint was written."""
    try:
        runner.run()
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc), "notes": list(getattr(exc, "__notes__", [])),
                "checkpoints": sorted(p.name for p in (out / "checkpoints").glob("*.pt"))}
    return {"error": None, "message": "", "checkpoints": []}


def _failing_step_runner(rank: int, world: int, out: Path, *, failing_rank: int, fail_at: int,
                         **kw: Any) -> HexTrainingRunner:
    model = DDP(_model())
    return HexTrainingRunner(
        _config(out, distributed=True, is_main_process=rank == 0, **kw),
        model,
        _failing_sgd(model.parameters(), fail_at=fail_at if rank == failing_rank else -1, lr=0.1),
        _train_loader(rank, world), _exact_val_loader(),
    )


def _sc_step_error_last_step_rank1(rank: int, world: int, out: Path) -> Dict[str, Any]:
    """Section 3.6: optimizer.step raises on rank 1 in the last step of epoch 0.

    Without the agreement rank 1 leaves while its peers ask for the next batch
    (the epoch's end): the next collective each side issues no longer matches.
    """
    runner = _failing_step_runner(rank, world, out, failing_rank=1, fail_at=2, num_epochs=2)
    return _post_step_outcome(runner, out)


def _sc_step_error_before_mid_save_rank1(rank: int, world: int, out: Path) -> Dict[str, Any]:
    """Section 3.6: optimizer.step raises on rank 1 in step 0, a mid-epoch save is due after it."""
    runner = _failing_step_runner(rank, world, out, failing_rank=1, fail_at=1, num_epochs=1,
                                  save_every_n_steps=1)
    return _post_step_outcome(runner, out)


def _sc_ema_error_rank0(rank: int, world: int, out: Path) -> Dict[str, Any]:
    """Section 3.6: the EMA update raises on rank 0 (the rank that writes checkpoints)."""
    runner = _failing_step_runner(rank, world, out, failing_rank=-1, fail_at=-1, num_epochs=1)
    if rank == 0:
        def broken(_model: nn.Module) -> None:
            raise _StepFailure("EMA update failed (injected)")

        runner.ema_model.update_parameters = broken
    return _post_step_outcome(runner, out)


def _equivalence_reference_3(out: Path) -> Dict[str, Any]:
    """The run the 3-rank job must reproduce: 12 samples, global batch 6, no sharding."""
    model = _model()
    runner = HexTrainingRunner(
        _config(out), model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        DataLoader(_Data(12, seed=1), batch_size=6), DataLoader(_Data(5, seed=2), batch_size=5),
    )
    result = runner.run()
    return {
        "params": _flat_params(model),
        "ema": _flat_params(runner.ema_model.module),
        "val_nrmse": [v["val_nrmse"] for v in result["history"]["val"]],
        "mean_loss": [t["mean_loss"] for t in result["history"]["train"]],
    }


def _sc_equivalence_12(rank: int, world: int, out: Path) -> Dict[str, Any]:
    """12 training samples, batch 2 per rank: with world 3 one step sees samples 0..5."""
    ds = _Data(12, seed=1)
    train = DataLoader(ds, batch_size=2, sampler=DistributedSampler(ds, world, rank, shuffle=False))
    runner = _ddp_runner(rank, out, train=train, val=_exact_val_loader())
    result = runner.run()
    return {
        "params": _flat_params(runner.raw_model),
        "ema": _flat_params(runner.ema_model.module),
        "val_nrmse": [v["val_nrmse"] for v in result["history"]["val"]],
        "mean_loss": [t["mean_loss"] for t in result["history"]["train"]],
        "val_indices": list(iter(runner.val_loader.sampler)),
    }


class _PoisonGrad(torch.autograd.Function):
    """Identity forward; backward multiplies the incoming gradient by NaN."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return grad * float("nan")


def _sc_nan_grad_on_rank1(rank: int, world: int, out: Path) -> Dict[str, Any]:
    """Section 3.7: a finite loss with a NaN gradient on rank 1 only (its step 1).

    DDP's all-reduce spreads the NaN to every rank's gradient, so every rank
    skips that update on its own; the agreement points confirm it. Before the
    fix all ranks stepped with NaN and trained on NaN weights from then on.
    """
    calls = {"n": 0}
    mse = nn.MSELoss()

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.requires_grad:
            calls["n"] += 1
            if rank == 1 and calls["n"] == 2:
                pred = _PoisonGrad.apply(pred)
        return mse(pred, target)

    model = DDP(_model())
    runner = HexTrainingRunner(
        _config(out, distributed=True, is_main_process=rank == 0, num_epochs=2, save_every_n_steps=1),
        model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        _train_loader(rank, world), _exact_val_loader(), loss_fn=loss_fn,
    )
    result = runner.run()
    return {
        "digest": _digest(runner.raw_model),
        "finite": all(bool(torch.isfinite(p).all()) for p in runner.raw_model.parameters())
        and all(bool(torch.isfinite(p).all()) for p in runner.ema_model.module.parameters()),
        "n_anomalies": [t["n_anomalies"] for t in result["history"]["train"]],
        "n_batches": [t["n_batches"] for t in result["history"]["train"]],
        "failure_files": sorted(p.name for p in runner.failure_dir.glob("*.json")),
    }


def _grad_skip_split(save_every_n_steps: int) -> Callable[[int, int, Path], Dict[str, Any]]:
    """Section 3.7 fault injection: rank 1 alone skips step 0 (as if only its
    reduced gradient were non-finite), the peers apply it. The weights now
    differ; the next agreement point must stop every rank before any save."""

    def scenario(rank: int, world: int, out: Path) -> Dict[str, Any]:
        runner = _failing_step_runner(rank, world, out, failing_rank=-1, fail_at=-1, num_epochs=1,
                                      save_every_n_steps=save_every_n_steps)
        if rank == 1:
            original = runner._apply_update
            calls = {"n": 0}

            def split() -> Any:
                calls["n"] += 1
                if calls["n"] == 1:
                    runner.optimizer.zero_grad(set_to_none=True)
                    return float("nan"), False
                return original()

            runner._apply_update = split
        return _post_step_outcome(runner, out)

    return scenario


def _sc_skip_record_error_rank1(rank: int, world: int, out: Path) -> Dict[str, Any]:
    """HX102-01: a NaN gradient on rank 1 (step 0) makes every rank skip the update;
    writing the FailureSignature of that skip fails on rank 1 only (a full disk).

    Before the fix rank 1 raised at once and its peers waited in the next
    agreement until the collective timeout; now the error is held and agreed.
    """
    calls = {"n": 0}
    mse = nn.MSELoss()

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.requires_grad:
            calls["n"] += 1
            if rank == 1 and calls["n"] == 1:
                pred = _PoisonGrad.apply(pred)
        return mse(pred, target)

    def full(*args: Any, **kwargs: Any) -> None:
        raise OSError(errno.ENOSPC, "No space left on device (injected)")

    model = DDP(_model())
    runner = HexTrainingRunner(
        _config(out, distributed=True, is_main_process=rank == 0, num_epochs=1),
        model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        _train_loader(rank, world), _exact_val_loader(), loss_fn=loss_fn,
    )
    if rank != 1:
        return _post_step_outcome(runner, out)
    with mock.patch.object(FailureSignature, "register", full):
        return _post_step_outcome(runner, out)


def _skip_limit(poison: str, preset_rank1: int) -> Callable[[int, int, Path], Dict[str, Any]]:
    """HX102-03: max_consecutive_skips=2 under DDP; rank 1 poisons its first two steps.

    ``poison="grad"``: a NaN gradient, spread to every rank by the all-reduce;
    both ranks reach the limit together in step 1 and each raises its own
    ConsecutiveSkipsError at the agreement. ``poison="loss"`` with rank 1's
    count preset to 1 (a desynchronised count, fault injection): only rank 1
    reaches the limit, in step 0, on the non-finite-loss path outside the
    held region of the update; the stop must still be agreed, not a lone exit.
    """

    def scenario(rank: int, world: int, out: Path) -> Dict[str, Any]:
        calls = {"n": 0}
        mse = nn.MSELoss()

        def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if pred.requires_grad:
                calls["n"] += 1
                if rank == 1 and calls["n"] <= 2:
                    if poison == "loss":
                        return mse(pred, target) * float("nan")
                    pred = _PoisonGrad.apply(pred)
            return mse(pred, target)

        model = DDP(_model())
        runner = HexTrainingRunner(
            _config(out, distributed=True, is_main_process=rank == 0, num_epochs=2, max_consecutive_skips=2),
            model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
            _train_loader(rank, world), _exact_val_loader(), loss_fn=loss_fn,
        )
        if rank == 1:
            runner._consecutive_skips = preset_rank1
        return _post_step_outcome(runner, out)

    return scenario


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
    "mid_epoch_resume": _sc_mid_epoch_resume,
    "mid_epoch_rank_buffers": _sc_mid_epoch_rank_buffers,
    "resume_other_position_on_rank1": _sc_resume_other_position_on_rank1,
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
    "resume_other_optimizer_state_on_rank1": _resume_with_rank1_copy(_bump_momentum),
    "resume_other_scaler_state_on_rank1": _resume_with_rank1_copy(_halve_scale, scaler=True),
    "resume_other_ema_state_on_rank1": _resume_with_rank1_copy(_move_ema),
    "resume_same_scaler_state": _resume_with_rank1_copy(lambda payload: None, scaler=True),
    "construction_other_lr_on_rank1": _construction_differs("optimizer"),
    "construction_other_scheduler_on_rank1": _construction_differs("scheduler"),
    "lambda_lr_other_function_on_rank1": _scheduler_differs("lambda"),
    "forged_step_lr_on_rank1": _scheduler_differs("forged"),
    "amp_equivalence": _sc_amp_equivalence,
    "amp_nan_grad_on_rank1": _sc_amp_nan_grad_on_rank1,
    "step_error_last_step_rank1": _sc_step_error_last_step_rank1,
    "step_error_before_mid_save_rank1": _sc_step_error_before_mid_save_rank1,
    "ema_error_rank0": _sc_ema_error_rank0,
    "nan_grad_on_rank1": _sc_nan_grad_on_rank1,
    "grad_skip_split_rank1": _grad_skip_split(0),
    "grad_skip_split_before_mid_save_rank1": _grad_skip_split(1),
    # last: before the fix it desynchronised the job; the scenarios above stay valid
    "skip_record_error_rank1": _sc_skip_record_error_rank1,
    "skip_limit_both": _skip_limit("grad", 0),
    "skip_limit_desync_rank1": _skip_limit("loss", 1),
    # very last: unfixed it desynchronises the job (rank 0 times out), and every
    # later scenario would fail with it -- the mutation probe's killed_by showed
    # 6 unrelated tests red for that mutant before this moved here
    "state_dict_error_rank1": _sc_state_dict_error_rank1,
}

# The 3-rank job (spec section 8, v1.1): equivalence with an uneven validation
# partition (5 samples -> 2/2/1) and the agreements, each ending on all ranks.
_SCENARIOS_3: Dict[str, Callable[[int, int, Path], Any]] = {
    "equivalence_12": _sc_equivalence_12,
    "padded_val": _sc_padded_val,
    "nan_on_rank1": _sc_nan_on_rank1,
    "forward_error_rank1": _sc_forward_error_rank1,
    "step_error_last_step_rank1": _sc_step_error_last_step_rank1,
    "step_error_before_mid_save_rank1": _sc_step_error_before_mid_save_rank1,
    "ema_error_rank0": _sc_ema_error_rank0,
    "nan_grad_on_rank1": _sc_nan_grad_on_rank1,
    "grad_skip_split_rank1": _grad_skip_split(0),
    "construction_other_lr_on_rank1": _construction_differs("optimizer"),
    "state_dict_error_rank1": _sc_state_dict_error_rank1,
}

_SUITES: Dict[str, Dict[str, Callable[[int, int, Path], Any]]] = {"two": _SCENARIOS, "three": _SCENARIOS_3}


def _write_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)  # a rank killed mid-write leaves the previous file


def _worker(rank: int, world: int, root: str, suite: str = "two") -> None:
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
        for name, scenario in _SUITES[suite].items():
            try:
                results[name] = {"ok": True, "value": scenario(rank, world, Path(root) / name)}
            except Exception as exc:  # recorded and asserted by the parent
                results[name] = {"ok": False, "error": type(exc).__name__, "message": str(exc),
                                 "notes": list(getattr(exc, "__notes__", []))}
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


class _MultiRankJob(unittest.TestCase):
    """Base: a real gloo job of WORLD ranks running SUITE; every scenario must end, none may hang."""

    WORLD = 2
    SUITE = "two"
    DEADLINE_S = 400

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _MultiRankJob:
            raise unittest.SkipTest("base class")
        if not dist.is_gloo_available():
            raise unittest.SkipTest("gloo backend not available")
        if os.environ.get("HEXANTK_SKIP_MULTIPROCESS") == "1":
            raise unittest.SkipTest("HEXANTK_SKIP_MULTIPROCESS=1")
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls._tmp.name
        # A hang or a crashed rank must turn the scenario tests red, not end
        # in a setUpClass error: record the job error, assert per scenario.
        cls.job_error = None
        ctx = mp.start_processes(_worker, args=(cls.WORLD, root, cls.SUITE), nprocs=cls.WORLD,
                                 join=False, start_method="spawn")
        deadline = time.monotonic() + cls.DEADLINE_S
        try:
            while not ctx.join(timeout=5):
                if time.monotonic() > deadline:
                    for p in ctx.processes:
                        p.kill()
                    cls.job_error = f"{cls.WORLD}-rank job did not finish within {cls.DEADLINE_S} s (hang)"
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
            cls.reference = cls._reference(Path(ref_dir))

    @staticmethod
    def _reference(out: Path) -> Dict[str, Any]:
        return _single_process_reference(out)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls is not _MultiRankJob:
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

    def _post_step(self, name: str, failing_rank: int, step: int, before: str) -> List[Dict[str, Any]]:
        """Section 3.6: the failing rank raises its own error, every peer PeerRankError."""
        values = self._ok(name)
        for rank, value in enumerate(values):
            with self.subTest(name, rank=rank):
                if rank == failing_rank:
                    self.assertEqual(value["error"], "_StepFailure")
                    # its own exception, annotated with the agreed failing ranks
                    self.assertTrue(any(f"agreed failure on rank(s) [{failing_rank}]" in n for n in value["notes"]),
                                    value["notes"])
                else:
                    self.assertEqual(value["error"], "PeerRankError")
                    self.assertIn(f"training step {step} after backward", value["message"])
                    self.assertIn(f"failed on rank(s) [{failing_rank}]", value["message"])
                    self.assertIn(before, value["message"])
        return values

    def _check_named_rank(self, name: str, failing_rank: int, what: str) -> None:
        """Every peer's PeerRankError names the failing rank; the failing rank's own error carries the note."""
        for rank, res in enumerate(self.results):
            entry = res.get(name, {})
            with self.subTest(name, rank=rank):
                if rank == failing_rank:
                    self.assertTrue(any(f"agreed failure on rank(s) [{failing_rank}]" in n
                                        for n in entry.get("notes", [])), entry)
                else:
                    self.assertIn(f"{what} failed on rank(s) [{failing_rank}]", entry.get("message", ""))

    def _check_post_step_scenarios(self) -> None:
        # last step of epoch 0 (2 steps per epoch): agreed before the epoch may end
        values = self._post_step("step_error_last_step_rank1", 1, 1, "stopped before step 2")
        # the epoch-0 save never ran: no checkpoint at all
        self.assertEqual(values[0]["checkpoints"], [])
        # a mid-epoch save due right after the failed step 0: no rank writes it
        values = self._post_step("step_error_before_mid_save_rank1", 1, 0, "the mid-epoch save")
        self.assertEqual(values[0]["checkpoints"], [])
        values = self._post_step("ema_error_rank0", 0, 0, "stopped before step 1")
        self.assertEqual(values[0]["checkpoints"], [])

    def _check_grad_skip_scenarios(self) -> None:
        """Section 3.7: a NaN gradient skips on every rank; a split skip stops every rank."""
        values = self._ok("nan_grad_on_rank1")
        for rank, value in enumerate(values):
            with self.subTest("nan_grad_on_rank1", rank=rank):
                self.assertTrue(value["finite"])
                self.assertEqual(value["digest"], values[0]["digest"])
                self.assertEqual(value["n_anomalies"], [1, 0])
                self.assertIn("failure_step1_NaN_or_Inf_Grad.json", value["failure_files"])
        names = ["grad_skip_split_rank1"]
        if "grad_skip_split_before_mid_save_rank1" in _SUITES[self.SUITE]:
            # bound to the suite, not to the results: a scenario missing from a
            # desynchronised job must fail in _ok, not be skipped
            names.append("grad_skip_split_before_mid_save_rank1")
        for name in names:
            before = "the mid-epoch save" if "mid_save" in name else "stopped before step 1"
            for rank, value in enumerate(self._ok(name)):
                with self.subTest(name, rank=rank):
                    self.assertEqual(value["error"], "DistributedContractError")
                    self.assertIn(f"1 of {self.WORLD} ranks skipped training step 0", value["message"])
                    self.assertIn(before, value["message"])
                    self.assertEqual(value["checkpoints"], [])


class TestTwoRanks(_MultiRankJob):
    """Real 2-rank gloo job; every scenario must end, none may hang."""

    WORLD = 2
    SUITE = "two"

    def test_error_after_backward_on_one_rank_fails_both(self) -> None:
        self._check_post_step_scenarios()

    def test_non_finite_gradient_skips_on_both_and_a_split_stops_both(self) -> None:
        self._check_grad_skip_scenarios()

    def test_error_recording_a_skipped_update_on_one_rank_fails_both(self) -> None:
        # HX102-01: rank 1 raises its own OSError, rank 0 the agreed PeerRankError;
        # neither waits for the collective timeout
        values = self._ok("skip_record_error_rank1")
        self.assertEqual(values[1]["error"], "OSError")
        self.assertIn("No space left on device", values[1]["message"])
        self.assertTrue(any("agreed failure on rank(s) [1]" in n for n in values[1]["notes"]), values[1]["notes"])
        self.assertEqual(values[0]["error"], "PeerRankError")
        for part in ("training step 0 after backward", "failed on rank(s) [1]", "stopped before step 1"):
            self.assertIn(part, values[0]["message"])
        self.assertEqual(values[0]["checkpoints"], [])

    def test_consecutive_skip_limit_stops_both_ranks_alike(self) -> None:
        # HX102-03: equal counts, both ranks raise their own error at the same agreement
        for rank, value in enumerate(self._ok("skip_limit_both")):
            with self.subTest("skip_limit_both", rank=rank):
                self.assertEqual(value["error"], "ConsecutiveSkipsError")
                self.assertIn("2 training steps in a row were skipped (the last: step 1", value["message"])
                self.assertTrue(any("agreed failure on rank(s) [0, 1]" in n for n in value["notes"]), value["notes"])
                self.assertEqual(value["checkpoints"], [])
        # a desynchronised count: rank 1 alone at the limit, rank 0 stops with it, no hang
        values = self._ok("skip_limit_desync_rank1")
        self.assertEqual(values[1]["error"], "ConsecutiveSkipsError")
        self.assertTrue(any("agreed failure on rank(s) [1]" in n for n in values[1]["notes"]), values[1]["notes"])
        self.assertEqual(values[0]["error"], "PeerRankError")
        for part in ("training step 0", "failed on rank(s) [1]", "stopped before step 1"):
            self.assertIn(part, values[0]["message"])

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
        self._check_named_rank("forward_error_rank1", 1, "training step 0 (forward or loss)")

    def test_construction_error_on_one_rank_fails_both(self) -> None:
        self.assertEqual(self._errors("construction_error_rank1"), ["PeerRankError", "ValueError"])
        self._check_named_rank("construction_error_rank1", 1, "runner construction")

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
    def test_mid_epoch_resume_reproduces_the_uninterrupted_run(self) -> None:
        for name in ("mid_epoch_resume", "mid_epoch_rank_buffers"):
            for rank, value in enumerate(self._ok(name)):
                with self.subTest(name, rank=rank):
                    self.assertEqual(value["batches_done"], 1)
                    self.assertEqual(value["full"], value["resumed"])
                    self.assertTrue(value["same_reports"])
                    # the rank's own anomaly history, not an empty window
                    self.assertGreater(value["history_len"], 1)
                    self.assertTrue(value["same_history"])

    def test_another_mid_epoch_position_on_rank1_fails_both(self) -> None:
        self.assertEqual(self._errors("resume_other_position_on_rank1"), ["DistributedContractError"] * 2)
        for res in self.results:
            self.assertIn("mid-epoch position", res.get("resume_other_position_on_rank1", {}).get("message", ""))

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
        for res in self.results:
            self.assertIn(": model weights", res.get("resume_other_weights_on_rank1", {}).get("message", ""))

    def test_resume_with_other_shared_state_on_one_rank_fails_both(self) -> None:
        # equal weights, another optimizer / GradScaler / EMA state on rank 1: red
        # before the fix (the run went on and the ranks drifted apart)
        for name, part in (("resume_other_optimizer_state_on_rank1", "optimizer state"),
                           ("resume_other_scaler_state_on_rank1", "GradScaler state"),
                           ("resume_other_ema_state_on_rank1", "EMA state")):
            # unfixed, the run completes and the ranks' weights drift apart: show the digests
            self.assertEqual(self._errors(name), ["DistributedContractError"] * 2,
                             f"{name}: {[res.get(name) for res in self.results]}")
            for rank, res in enumerate(self.results):
                with self.subTest(name, rank=rank):
                    message = res.get(name, {}).get("message", "")
                    self.assertTrue(message.endswith(f"on at least one rank: {part}"), message)

    def test_other_optimizer_or_scheduler_at_construction_fails_both(self) -> None:
        # same RunnerConfig, another learning rate / scheduler on rank 1: red before
        # (both ranks trained to the end with diverging weights)
        # another learning rate also shows in the scheduler's base_lrs/_last_lr
        for name, part in (("construction_other_lr_on_rank1", "optimizer state, scheduler state"),
                           ("construction_other_scheduler_on_rank1", "scheduler state")):
            self.assertEqual(self._errors(name), ["DistributedContractError"] * 2,
                             f"{name}: {[res.get(name) for res in self.results]}")
            for rank, res in enumerate(self.results):
                with self.subTest(name, rank=rank):
                    message = res.get(name, {}).get("message", "")
                    self.assertEqual(message, f"initial state differs from rank 0 on at least one rank: {part}")

    def test_untrusted_scheduler_is_refused_on_both_ranks(self) -> None:
        # R8: equal state, other behaviour -- red before (both ranks ran to the end)
        self.assertEqual(self._errors("lambda_lr_other_function_on_rank1"), ["DistributedContractError"] * 2,
                         [res.get("lambda_lr_other_function_on_rank1") for res in self.results])
        for res in self.results:
            message = res["lambda_lr_other_function_on_rank1"]["message"]
            self.assertIn("scheduler: torch.optim.lr_scheduler.LambdaLR is not one of the built-in classes", message)
        # forged on rank 1 only: rank 1 refuses, rank 0 is told (agreed, no hang)
        name = "forged_step_lr_on_rank1"
        self.assertEqual(self._errors(name), ["PeerRankError", "DistributedContractError"],
                         [res.get(name) for res in self.results])
        self.assertIn("scheduler: torch.optim.lr_scheduler.StepLR is not one of the built-in classes",
                      self.results[1][name]["message"])
        self.assertIn("failed on rank(s) [1]", self.results[0][name]["message"])

    def test_state_digest_error_on_one_rank_is_agreed(self) -> None:
        # rank 1's scheduler carries an instance-level state_dict: since R8 it is
        # refused before it runs -- the refusal on one rank is agreed like any
        # digest failure (before R8: the injected state_dict raised _StepFailure)
        r0, r1 = self._ok("state_dict_error_rank1")
        self.assertEqual(r1["error"], "DistributedContractError")
        self.assertIn("overrides state_dict", r1["message"])
        self.assertEqual(r0["error"], "PeerRankError", r0)
        self.assertIn("failed on rank(s) [1]", r0["message"])
        # agreed at once, not after the 30 s process-group timeout of this job
        self.assertLess(r0["seconds"], 15.0)

    def test_resume_with_equal_scaler_state_runs(self) -> None:
        # positive control: the same file on both ranks, GradScaler included
        r0, r1 = self._ok("resume_same_scaler_state")
        self.assertEqual(r0["digest"], r1["digest"])

    def test_amp_grad_scaler_under_ddp_reproduces_the_single_process_run(self) -> None:
        r0, r1 = self._ok("amp_equivalence")
        self.assertEqual(r0["params"], r1["params"])
        self.assertEqual(r0["ema"], r1["ema"])
        self.assertEqual(r0["scale"], r1["scale"])
        with tempfile.TemporaryDirectory() as ref_dir:
            ref = _amp_single_process_reference(Path(ref_dir))
        # growth_interval=2 over 6 healthy steps: 1024 doubled three times
        self.assertEqual(r0["scale"], ref["scale"])
        self.assertEqual(ref["scale"], 8192.0)
        for key in ("params", "ema", "mean_loss"):
            for got, want in zip(r0[key], ref[key]):
                self.assertAlmostEqual(got, want, delta=1e-12, msg=key)
        self.assertEqual(r0["n_anomalies"], [0, 0, 0])

    def test_amp_nan_gradient_on_one_rank_backs_off_on_both(self) -> None:
        r0, r1 = self._ok("amp_nan_grad_on_rank1")
        clean, _ = self._ok("amp_equivalence")
        for value in (r0, r1):
            self.assertTrue(all(math.isfinite(x) for x in value["params"] + value["ema"]))
            # the scaler's skip is agreed like any skip and counted, routine: no capsule
            self.assertEqual(value["n_anomalies"], [1, 0, 0])
            self.assertEqual(value["failure_files"], [])
        self.assertEqual(r0["digest"], r1["digest"])
        self.assertEqual(r0["scale"], r1["scale"])
        # step 0 healthy, step 1 backs off (1024 -> 512, tracker reset), steps 2-5
        # grow twice: 2048, against 8192 in the clean run
        self.assertEqual(r0["scale"], 2048.0)
        self.assertEqual(clean["scale"], 8192.0)


class TestThreeRanks(_MultiRankJob):
    """Real 3-rank gloo job (spec section 8, v1.1): more than two ranks, uneven validation shards."""

    WORLD = 3
    SUITE = "three"
    DEADLINE_S = 300

    @staticmethod
    def _reference(out: Path) -> Dict[str, Any]:
        return _equivalence_reference_3(out)

    def test_three_ranks_reproduce_the_single_process_run(self) -> None:
        values = self._ok("equivalence_12")
        # 5 validation samples on 3 ranks: 2/2/1, together exactly range(5)
        self.assertEqual([v["val_indices"] for v in values], [[0, 3], [1, 4], [2]])
        for other in values[1:]:
            for key in ("params", "ema", "val_nrmse", "mean_loss"):
                self.assertEqual(values[0][key], other[key], key)
        for key in ("params", "ema", "val_nrmse", "mean_loss"):
            for got, want in zip(values[0][key], self.reference[key]):
                self.assertAlmostEqual(got, want, delta=1e-12, msg=key)

    def test_padded_validation_sampler_fails_on_every_rank(self) -> None:
        self.assertEqual(self._errors("padded_val"), ["DistributedContractError"] * 3)

    def test_nan_on_one_rank_skips_the_step_everywhere(self) -> None:
        values = self._ok("nan_on_rank1")
        self.assertEqual(len({v["digest"] for v in values}), 1)
        for value in values:
            self.assertEqual(value["n_anomalies"], [1, 0])

    def test_forward_error_on_one_rank_fails_all(self) -> None:
        self.assertEqual(self._errors("forward_error_rank1"), ["PeerRankError", "KeyError", "PeerRankError"])
        # rank 1 of 3: the one-hot identification maps the middle rank correctly
        self._check_named_rank("forward_error_rank1", 1, "training step 0 (forward or loss)")

    def test_error_after_backward_on_one_rank_fails_all(self) -> None:
        self._check_post_step_scenarios()

    def test_non_finite_gradient_skips_on_all_and_a_split_stops_all(self) -> None:
        self._check_grad_skip_scenarios()

    def test_shared_state_of_the_middle_rank_is_checked_on_all(self) -> None:
        # rank 1 of 3 differs: every rank raises, with the same named components
        name = "construction_other_lr_on_rank1"
        self.assertEqual(self._errors(name), ["DistributedContractError"] * 3)
        for res in self.results:
            self.assertEqual(res.get(name, {}).get("message"),
                             "initial state differs from rank 0 on at least one rank: optimizer state, scheduler state")
        # a digest failure on the middle rank: its own error there, PeerRankError naming it elsewhere
        # (since R8 its instance-level state_dict is refused before it runs)
        values = self._ok("state_dict_error_rank1")
        self.assertEqual([v["error"] for v in values], ["PeerRankError", "DistributedContractError", "PeerRankError"])
        self.assertIn("overrides state_dict", values[1]["message"])
        for rank in (0, 2):
            self.assertIn("failed on rank(s) [1]", values[rank]["message"])
            self.assertLess(values[rank]["seconds"], 15.0)


if __name__ == "__main__":
    unittest.main()
