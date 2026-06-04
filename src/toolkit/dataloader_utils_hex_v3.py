"""
dataloader_utils_hex_v3.py
==========================
Reproducible DataLoader utilities for HexaNTK pipeline.

Provides:
- worker_init_fn for deterministic multi-worker seeding
- Explicit Generator for DataLoader shuffling
- Convenience factory for reproducible DataLoaders

PyTorch reproducibility caveats (from official docs):
- Full reproducibility is NOT guaranteed across releases,
  platforms, or CPU/GPU boundaries
- Reproducibility is scoped to: same software + same hardware + same seed
- Multi-worker DataLoaders need explicit worker_init_fn + generator

References:
- PyTorch Reproducibility Guide
- PyTorch DataLoader worker_init_fn documentation
"""
from __future__ import annotations

__all__ = [
    "make_worker_init_fn",
    "make_reproducible_generator",
    "make_reproducible_dataloader_kwargs",
]

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset


def make_worker_init_fn(base_seed: int):
    """
    Create a worker_init_fn that seeds each DataLoader worker deterministically.

    Each worker gets seed = base_seed + worker_id, ensuring:
    - Different workers produce different random sequences
    - Same base_seed reproduces the same sequences across runs

    Usage::

        loader = DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=make_worker_init_fn(seed=42),
        )
    """
    import numpy as np
    import random

    def worker_init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2 ** 32))

    return worker_init_fn


def make_reproducible_generator(seed: int) -> torch.Generator:
    """
    Create a torch.Generator for reproducible DataLoader shuffling.

    Usage::

        g = make_reproducible_generator(seed=42)
        loader = DataLoader(dataset, shuffle=True, generator=g)
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_reproducible_dataloader_kwargs(
    seed: int,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    Build kwargs dict for reproducible DataLoader construction.

    Usage::

        kwargs = make_reproducible_dataloader_kwargs(seed=42, num_workers=4)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, **kwargs)

    Returns
    -------
    Dict with:
        generator: torch.Generator (for shuffle reproducibility)
        worker_init_fn: callable (for multi-worker seed consistency)
        num_workers: int
    """
    kwargs: Dict[str, Any] = {
        "generator": make_reproducible_generator(seed),
        "num_workers": num_workers,
    }

    if num_workers > 0:
        kwargs["worker_init_fn"] = make_worker_init_fn(seed)

    return kwargs
