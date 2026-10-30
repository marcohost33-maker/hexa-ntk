"""
test_mid_epoch_resume_hex_v3.py
===============================
Mid-epoch resume of the training runner (spec/MID_EPOCH_RESUME_v1.md).

The claim is equality: a run interrupted inside an epoch and resumed from its
last mid-epoch checkpoint ends with the same weights, EMA, epoch reports and
anomaly history as the run that was never interrupted -- with shuffling through
an explicit generator, dropout, loss weights drawn from the Python and NumPy
streams, and with DataLoader workers whose transforms draw from their own RNG.
The 2-rank DDP case is in test_ddp_hex_v3.TestTwoRanks (mid_epoch_resume).
"""
from __future__ import annotations

import random
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from runner_hex_v3 import HexTrainingRunner, RunnerConfig, _check_mid_epoch_record


class _Data(Dataset):
    """Regression samples; with ``noise`` each fetch adds torch noise (a worker-RNG transform)."""

    def __init__(self, n: int = 8, noise: bool = False) -> None:
        g = torch.Generator().manual_seed(11)
        self.x = torch.randn(n, 3, generator=g, dtype=torch.float64)
        self.y = self.x @ torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.0, 0.75]], dtype=torch.float64)
        self.noise = noise

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        x = self.x[i]
        if self.noise:
            x = x + 0.01 * torch.randn(3, dtype=torch.float64)
        return {"input": x, "target": self.y[i]}


class _Stream(IterableDataset):
    """The same samples as an unsized stream (no len: only the fast-forward sees a change)."""

    def __init__(self, n: int) -> None:
        self.data = _Data(n)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return (self.data[i] for i in range(len(self.data)))


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(3, 16, dtype=torch.float64), nn.Tanh(), nn.Dropout(0.3),
                                 nn.Linear(16, 2, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = torch.mean((pred - target) ** 2)
    if pred.requires_grad:  # the Python and NumPy streams shape the update
        loss = loss * (1.0 + 0.01 * random.random()) * (1.0 + 0.01 * float(np.random.rand()))
    return loss


class _Crash(RuntimeError):
    pass


def _runner(out: Path, *, workers: int = 0, crash_at: int = -1, n: int = 8, generator: bool = True,
            persistent: bool = False, batch_size: int = 2, stream: bool = False,
            nan_at: Tuple[int, ...] = (), **kw: Any) -> HexTrainingRunner:
    torch.manual_seed(5)
    random.seed(6)
    np.random.seed(7)
    model = _Net()
    # without an explicit generator the sampler permutation and the worker base
    # seed come from the global torch RNG at the start of the epoch
    gen = torch.Generator().manual_seed(123) if generator else None
    if stream:
        train = DataLoader(_Stream(n), batch_size=batch_size)
    else:
        train = DataLoader(_Data(n, noise=workers > 0), batch_size=batch_size, shuffle=True,
                           generator=gen, num_workers=workers, persistent_workers=persistent)

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = _loss(pred, target)
        if pred.requires_grad and runner.global_step in nan_at:
            loss = loss * float("nan")  # a skipped step
        return loss

    config = RunnerConfig(output_dir=str(out), num_epochs=3, log_every_n_steps=0, ema_decay=0.9,
                          best_metric_min_delta=0.0, **{"save_every_n_steps": 3, **kw})
    runner = HexTrainingRunner(config, model, torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9),
                               train, DataLoader(_Data(5), batch_size=5), loss_fn=loss_fn,
                               scheduler=None)
    if crash_at >= 0:
        original = runner._train_one_step

        def crashing(batch: Any, prepared: Any = None) -> Any:
            if runner.global_step == crash_at:
                raise _Crash(f"injected preemption before step {crash_at}")
            return original(batch, prepared)

        runner._train_one_step = crashing
    return runner


def _state(runner: HexTrainingRunner) -> Dict[str, Any]:
    return {
        "params": torch.cat([p.detach().reshape(-1) for p in runner.raw_model.parameters()]).tolist(),
        "ema": torch.cat([p.detach().reshape(-1) for p in runner.ema_model.module.parameters()]).tolist(),
        "history": list(runner.anomaly_detector.loss_history),
        "step": runner.global_step,
    }


class TestMidEpochResume(unittest.TestCase):
    def _check(self, workers: int, generator: bool = True, **kw: Any) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            full = _runner(tmp / "full", workers=workers, generator=generator, **kw)
            reference = full.run()
            interrupted = _runner(tmp / "split", workers=workers, crash_at=7, generator=generator, **kw)
            with self.assertRaises(_Crash):
                interrupted.run()
            # last save at step 6 = epoch 1, 2 of 4 batches done
            resumed = _runner(tmp / "split", workers=workers, generator=generator,
                              resume_from=str(tmp / "split" / "checkpoints" / "resume_last.pt"),
                              resume_trusted=True, **kw)
            result = resumed.run()
            self.assertEqual(result["resume_report"]["mid_epoch_batches_done"], 2)
            self.assertEqual(result["resume_report"]["epoch"], 1)
            self.assertEqual(_state(resumed), _state(full))
            # epochs 1 and 2 report exactly what the uninterrupted run reported
            self.assertEqual(result["history"]["train"], reference["history"]["train"][1:])
            self.assertEqual(result["history"]["val"], reference["history"]["val"][1:])
            return reference

    def test_interrupted_equals_uninterrupted(self) -> None:
        self._check(workers=0)

    def test_interrupted_equals_uninterrupted_with_worker_rng(self) -> None:
        self._check(workers=2)

    def test_interrupted_equals_uninterrupted_without_loader_generator(self) -> None:
        self._check(workers=0, generator=False)
        self._check(workers=2, generator=False)

    def test_a_step_skipped_before_the_save_is_carried(self) -> None:
        # NaN at step 5 (epoch 1, batch 2): the save at step 6 records 1 skip
        reference = self._check(workers=0, nan_at=(5,))
        self.assertEqual([e["n_anomalies"] for e in reference["history"]["train"]], [0, 1, 0])

    def test_without_the_fast_forward_state_the_result_differs(self) -> None:
        # positive control of the comparison: an epoch-boundary resume of the same
        # interruption (the mid-epoch record dropped) does not reproduce the run
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            full = _runner(tmp / "full")
            full.run()
            interrupted = _runner(tmp / "split", crash_at=7)
            with self.assertRaises(_Crash):
                interrupted.run()
            path = tmp / "split" / "checkpoints" / "resume_last.pt"
            payload = torch.load(path, weights_only=False)
            del payload["extra"]["mid_epoch"]
            torch.save(payload, path)
            resumed = _runner(tmp / "split", resume_from=str(path), resume_trusted=True)
            resumed.run()
            self.assertNotEqual(_state(resumed)["params"], _state(full)["params"])

    def test_a_changed_loader_is_refused(self) -> None:
        # batch size 4 would let the fast-forward eat the whole epoch, 12 samples
        # would shift the positions: both used to run through without an error
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            interrupted = _runner(tmp, crash_at=7)
            with self.assertRaises(_Crash):
                interrupted.run()
            for i, change in enumerate(({"batch_size": 1}, {"batch_size": 4}, {"n": 12}, {"n": 2})):
                with self.subTest(**change):
                    run = tmp / f"resume{i}"  # each from its own copy of the mid-epoch save
                    shutil.copytree(tmp / "checkpoints", run / "checkpoints")
                    resumed = _runner(run, resume_from=str(run / "checkpoints" / "resume_last.pt"),
                                      resume_trusted=True, **change)
                    with self.assertRaisesRegex(ValueError, r"train loader with .* = \[4, 2, 8\]"):
                        resumed.run()

    def test_a_shrunk_stream_is_refused(self) -> None:
        # unsized: the shape check sees nothing, the fast-forward runs dry
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            interrupted = _runner(tmp, crash_at=7, stream=True)
            with self.assertRaises(_Crash):
                interrupted.run()
            resumed = _runner(tmp, n=2, stream=True, resume_from=str(tmp / "checkpoints" / "resume_last.pt"),
                              resume_trusted=True)
            with self.assertRaisesRegex(ValueError, "yields only 1 batches"):
                resumed.run()

    def test_persistent_workers_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaisesRegex(ValueError, "save_every_n_steps > 0, but .* persistent_workers=True"):
                _runner(tmp / "a", workers=1, persistent=True)
            # a mid-epoch checkpoint resumed by a persistent-worker loader (no
            # mid-epoch saves of its own) is refused at the resume
            interrupted = _runner(tmp / "b", crash_at=7)
            with self.assertRaises(_Crash):
                interrupted.run()
            resumed = _runner(tmp / "b", workers=1, persistent=True, save_every_n_steps=0,
                              resume_from=str(tmp / "b" / "checkpoints" / "resume_last.pt"),
                              resume_trusted=True)
            with self.assertRaisesRegex(ValueError, "mid-epoch position, but .* persistent_workers=True"):
                resumed.run()

    def test_persistent_workers_break_the_replay(self) -> None:
        # the reason for the refusal (torch 2.6): epoch 2 of one persistent-worker
        # iterator is not the epoch a fresh iterator draws from the restored
        # epoch-start state, with or without an explicit generator
        for generator in (True, False):
            with self.subTest(generator=generator):
                data = list(range(12))
                torch.manual_seed(0)
                gen = torch.Generator().manual_seed(5) if generator else None
                loader = DataLoader(data, batch_size=3, shuffle=True, num_workers=1,
                                    persistent_workers=True, generator=gen)
                list(loader)
                start = (gen.get_state() if generator else None, torch.random.get_rng_state())
                epoch2 = [b.tolist() for b in loader]
                gen2 = None
                if generator:
                    gen2 = torch.Generator()
                    gen2.set_state(start[0])
                torch.random.set_rng_state(start[1])
                fresh = DataLoader(data, batch_size=3, shuffle=True, num_workers=1,
                                   persistent_workers=True, generator=gen2)
                self.assertNotEqual([b.tolist() for b in fresh], epoch2)

    def test_save_every_n_steps_is_validated(self) -> None:
        for bad in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                RunnerConfig(save_every_n_steps=bad)


def _record(**change: Any) -> Dict[str, Any]:
    record = {
        "version": 1, "epoch": 2, "batches_done": 3, "world_size": 1, "n_anomalies": 1,
        "losses": [[0.5, 0.25]], "grad_norms": [[1.0, 2.0]],
        "epoch_start": {"torch_rng": [torch.random.get_rng_state()],
                        "loader_generators": {"train": [torch.Generator().get_state()]}},
        "train_loader": [[4, 2, 8]],
    }
    record.update(change)
    return record


class TestMidEpochRecord(unittest.TestCase):
    def test_valid_record_gives_this_ranks_part(self) -> None:
        part = _check_mid_epoch_record(_record(), epoch=2, path=Path("r.pt"), loader_shape=[4, 2, 8])
        self.assertEqual((part["batches_done"], part["losses"], part["n_anomalies"]), (3, [0.5, 0.25], 1))
        self.assertIn("train", part["epoch_start"]["loader_generators"])
        self.assertIsNone(part["buffers"])
        part = _check_mid_epoch_record(_record(rank_buffers={"b": [torch.ones(2)]}), epoch=2,
                                       path=Path("r.pt"), loader_shape=[4, 2, 8])
        self.assertEqual(part["buffers"]["b"].tolist(), [1.0, 1.0])

    def test_invalid_records_are_refused(self) -> None:
        cases = {
            "version": _record(version=2),
            "names epoch": _record(epoch=1),
            "another world size": _record(world_size=2),
            "batches_done": _record(batches_done=-1),
            "n_anomalies": _record(n_anomalies=4),
            "does not add up": _record(batches_done=4),
            "incomplete": _record(losses=[]),
            "list of floats": _record(losses=[[0.5, "x"]]),
            "uint8 RNG": _record(epoch_start={"torch_rng": [torch.zeros(3)], "loader_generators": {}}),
            "train loader with": _record(train_loader=[[4, 1, 8]]),
            "is incomplete": _record(train_loader=[]),
            "not a tensor": _record(rank_buffers={"b": [[1.0, 1.0]]}),
        }
        for why, record in cases.items():
            with self.subTest(why):
                with self.assertRaisesRegex(ValueError, why):
                    _check_mid_epoch_record(record, epoch=2, path=Path("r.pt"), loader_shape=[4, 2, 8])


if __name__ == "__main__":
    unittest.main()
