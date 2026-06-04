"""
test_dataloader_utils_hex_v3.py
===============================
Tests for reproducible DataLoader utilities.
"""
from __future__ import annotations

import unittest

import torch

from dataloader_utils_hex_v3 import (
    make_reproducible_dataloader_kwargs,
    make_reproducible_generator,
    make_worker_init_fn,
)


class TestReproducibleGenerator(unittest.TestCase):
    def test_same_seed_same_sequence(self) -> None:
        g1 = make_reproducible_generator(42)
        g2 = make_reproducible_generator(42)
        vals1 = [torch.randint(0, 1000, (1,), generator=g1).item() for _ in range(10)]
        vals2 = [torch.randint(0, 1000, (1,), generator=g2).item() for _ in range(10)]
        self.assertEqual(vals1, vals2)

    def test_different_seed_different_sequence(self) -> None:
        g1 = make_reproducible_generator(42)
        g2 = make_reproducible_generator(99)
        vals1 = [torch.randint(0, 1000, (1,), generator=g1).item() for _ in range(10)]
        vals2 = [torch.randint(0, 1000, (1,), generator=g2).item() for _ in range(10)]
        self.assertNotEqual(vals1, vals2)


class TestWorkerInitFn(unittest.TestCase):
    def test_callable(self) -> None:
        fn = make_worker_init_fn(42)
        self.assertTrue(callable(fn))

    def test_deterministic_seeding(self) -> None:
        import random
        fn = make_worker_init_fn(42)
        fn(0)
        val1 = random.random()
        fn(0)  # Reset same worker
        val2 = random.random()
        self.assertEqual(val1, val2)

    def test_different_workers_different_seeds(self) -> None:
        import random
        fn = make_worker_init_fn(42)
        fn(0)
        val_w0 = random.random()
        fn(1)
        val_w1 = random.random()
        self.assertNotEqual(val_w0, val_w1)


class TestMakeReproducibleKwargs(unittest.TestCase):
    def test_contains_generator(self) -> None:
        kwargs = make_reproducible_dataloader_kwargs(seed=42)
        self.assertIn("generator", kwargs)
        self.assertIsInstance(kwargs["generator"], torch.Generator)

    def test_no_worker_init_fn_for_zero_workers(self) -> None:
        kwargs = make_reproducible_dataloader_kwargs(seed=42, num_workers=0)
        self.assertNotIn("worker_init_fn", kwargs)

    def test_has_worker_init_fn_for_multiworker(self) -> None:
        kwargs = make_reproducible_dataloader_kwargs(seed=42, num_workers=4)
        self.assertIn("worker_init_fn", kwargs)
        self.assertTrue(callable(kwargs["worker_init_fn"]))
        self.assertEqual(kwargs["num_workers"], 4)


if __name__ == "__main__":
    unittest.main()
