"""
ddp_hex_v3.py
=============
Rank coordination for the HexaNTK training runner under DistributedDataParallel
(contract: spec/DDP_RUNNER_v1.md).

DDP synchronises gradients, nothing else. Every decision the runner takes
outside the gradient all-reduce -- skip a step, call a validation empty, accept
a checkpoint, resume from a file -- must come out the same on every rank, or
one rank enters a collective its peers never reach and the job hangs (or, worse,
the ranks silently disagree about the best model). This module holds the few
primitives the runner needs for that:

- ExactDistributedEvalSampler: every sample on exactly one rank. The stock
  DistributedSampler pads with repeated indices when ``len(dataset)`` is not a
  multiple of the world size (n=5, world 2 -> [[0, 2, 4], [1, 3, 0]]), so a
  pooled metric would count sample 0 twice.
- DistributedRunContext: tensor-only collectives (no pickled objects) for
  float64 sums, integer agreement, an "any rank" flag, a rank-0 broadcast, an
  exact-partition check and run_agreed(), which turns a local exception into an
  error on every rank instead of a hang.

Pooled metrics are reduced as sufficient statistics (sum of squared errors, sum
of squared references, counts), never as a mean of per-rank ratios: for
(sq_err, sq_ref) = (9, 4) and (1, 96) the mean of the rank nRMSEs is 0.801, the
pooled value sqrt(10 / 100) = 0.316.

References:
- PyTorch DistributedSampler (padding, drop_last, set_epoch):
  torch/utils/data/distributed.py
- PyTorch distributed: all_reduce/broadcast on tensors; the *_object
  collectives unpickle peer data and are not used here.
"""
from __future__ import annotations

__all__ = [
    "DistributedContractError",
    "PeerRankError",
    "ExactDistributedEvalSampler",
    "DistributedRunContext",
    "rng_state_tensors",
    "restore_rng_state_tensors",
]

import hashlib
import random
import struct
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, TypeVar, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

T = TypeVar("T")


class DistributedContractError(RuntimeError):
    """A distributed precondition failed; raised identically on every rank."""


class PeerRankError(RuntimeError):
    """This rank's part succeeded, a peer rank's part failed."""


def _require_initialized() -> None:
    if not (dist.is_available() and dist.is_initialized()):
        raise DistributedContractError(
            "torch.distributed is not initialized; call "
            "torch.distributed.init_process_group() before building the runner"
        )


def _as_rank_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _float_bits(value: float) -> int:
    return struct.unpack("<q", struct.pack("<d", float(value)))[0]


def _bits_float(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<q", int(bits)))[0]


def rng_state_tensors() -> Dict[str, torch.Tensor]:
    """This process's torch-CPU, Python and NumPy (legacy) RNG states as tensors.

    Fixed-length tensors, so ranks can all_gather them without pickling:
    torch as its uint8 state; Python as int64 [version, 625 state words,
    has_gauss, gauss bits]; NumPy as int64 [624 key words, pos, has_gauss,
    cached_gaussian bits]. Floats travel as their IEEE-754 bits (exact).
    """
    version, internal, gauss = random.getstate()
    py = [version, *internal, 0 if gauss is None else 1, _float_bits(0.0 if gauss is None else gauss)]
    name, keys, pos, has_gauss, cached = np.random.get_state(legacy=True)
    if name != "MT19937":
        raise DistributedContractError(f"unsupported numpy bit generator {name!r}")
    npy = [*(int(k) for k in keys), int(pos), int(has_gauss), _float_bits(cached)]
    return {
        "torch": torch.random.get_rng_state().clone(),
        "python": torch.tensor(py, dtype=torch.int64),
        "numpy": torch.tensor(npy, dtype=torch.int64),
    }


def restore_rng_state_tensors(states: Dict[str, torch.Tensor]) -> None:
    """Inverse of rng_state_tensors() (exact)."""
    py = [int(v) for v in states["python"].tolist()]
    gauss = _bits_float(py[-1]) if py[-2] else None
    random.setstate((py[0], tuple(py[1:-2]), gauss))
    npy = [int(v) for v in states["numpy"].tolist()]
    np.random.set_state(
        ("MT19937", np.array(npy[:-3], dtype=np.uint32), npy[-3], npy[-2], _bits_float(npy[-1]))
    )
    torch.random.set_rng_state(states["torch"].to(torch.uint8).cpu())


class ExactDistributedEvalSampler(Sampler[int]):
    """Strided, padding-free partition of ``range(n)`` for evaluation.

    Rank r gets indices r, r + W, r + 2W, ... (W = world size). The ranks'
    lists are disjoint and their union is ``range(n)``; ranks may differ by
    one sample. Order is fixed (no shuffle): evaluation needs coverage, not
    randomness. Use it only where no collective runs per batch (the runner's
    validation loop runs on the unwrapped module, see spec section 3).
    """

    def __init__(
        self,
        data_source: Any,
        *,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if isinstance(data_source, bool):
            raise TypeError("data_source must be a sized dataset or an int length")
        n = data_source if isinstance(data_source, int) else len(data_source)
        if n < 0:
            raise ValueError(f"dataset length must be >= 0, got {n}")
        if num_replicas is None or rank is None:
            _require_initialized()
        num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
        rank = dist.get_rank() if rank is None else rank
        num_replicas = _as_rank_count(num_replicas, "num_replicas")
        rank = _as_rank_count(rank, "rank")
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.dataset_len = n
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_len, self.num_replicas))

    def __len__(self) -> int:
        return len(range(self.rank, self.dataset_len, self.num_replicas))


class DistributedRunContext:
    """Collectives the runner uses to keep its ranks in agreement.

    All payloads are tensors: float64 for sums, int64 for counts, flags and
    bytes. On the NCCL backend the tensors live on ``device`` if it is a CUDA
    device (the runner passes its training device), else on the current CUDA
    device -- call ``torch.cuda.set_device(local_rank)`` before; elsewhere on
    the CPU. Every method is a collective: all ranks must call it in the same
    order. Only the default process group is used.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        _require_initialized()
        self.rank: int = dist.get_rank()
        self.world_size: int = dist.get_world_size()
        self.backend: str = str(dist.get_backend())
        if self.backend == "nccl":
            requested = torch.device(device) if device is not None else None
            if requested is not None and requested.type == "cuda" and requested.index is not None:
                self.device = requested
            else:
                self.device = torch.device("cuda", torch.cuda.current_device())
        else:
            self.device = torch.device("cpu")

    # -- primitives ------------------------------------------------------

    def all_reduce_sum(self, values: Sequence[float]) -> List[float]:
        """Element-wise float64 sum over ranks (NaN/inf propagate)."""
        t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return [float(v) for v in t.cpu().tolist()]

    def all_reduce_sum_rank0(self, values: Sequence[float]) -> List[float]:
        """all_reduce_sum, then rank 0's result broadcast to every rank.

        For values a decision depends on (best checkpoint, empty validation):
        NCCL and gloo describe all-reduce results as identical on every rank,
        but no backend documents that bit for bit for every algorithm; the
        broadcast makes every rank decide on the same bytes by construction.
        """
        t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        dist.broadcast(t, src=0)
        return [float(v) for v in t.cpu().tolist()]

    def _all_reduce_int(self, values: Sequence[int], op: Any) -> List[int]:
        t = torch.tensor([int(v) for v in values], dtype=torch.int64, device=self.device)
        dist.all_reduce(t, op=op)
        return [int(v) for v in t.cpu().tolist()]

    def sum_ints(self, values: Sequence[int]) -> List[int]:
        """Element-wise int64 sum over ranks (e.g. how many ranks failed / ran dry)."""
        return self._all_reduce_int(values, dist.ReduceOp.SUM)

    def max_int(self, value: int) -> int:
        """The largest ``value`` over all ranks (a status code: 0 ok, higher worse)."""
        return self._all_reduce_int([value], dist.ReduceOp.MAX)[0]

    def any_true(self, flag: bool) -> bool:
        """True on every rank iff ``flag`` is true on at least one rank."""
        return self.max_int(1 if flag else 0) == 1

    def broadcast_bytes(self, data: Optional[bytes], length: int) -> bytes:
        """Rank 0's ``data`` (exactly ``length`` bytes) on every rank."""
        t = torch.zeros(length, dtype=torch.uint8, device=self.device)
        bad = False
        if self.rank == 0:
            if not isinstance(data, (bytes, bytearray)) or len(data) != length:
                bad = True
            else:
                t.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        if self.any_true(bad):
            raise DistributedContractError(
                f"rank 0 has no {length}-byte payload to broadcast"
            )
        dist.broadcast(t, src=0)
        return bytes(t.cpu().tolist())

    def all_gather_tensor(self, tensor: torch.Tensor, *, what: str) -> List[torch.Tensor]:
        """Every rank's 1-D ``tensor`` (same length and dtype on all ranks), on the CPU."""
        self.require_equal_int(tensor.numel(), what=f"{what}: length")
        local = tensor.reshape(-1).to(self.device)
        parts = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(parts, local)
        return [part.cpu() for part in parts]

    def require_same_names(self, names: Sequence[str], *, what: str) -> None:
        """The ordered ``names`` must be identical on every rank (SHA-256 vs rank 0).

        Run before a sequence of per-name collectives: a rank with one name
        more would issue one collective more and hang its peers.
        """
        mine = hashlib.sha256("\n".join(names).encode("utf-8")).digest()
        ref = self.broadcast_bytes(mine if self.rank == 0 else None, 32)
        if self.any_true(ref != mine):
            raise DistributedContractError(f"{what} differ between ranks")

    def sync_buffers_from_rank0(self, module: torch.nn.Module, *, what: str) -> None:
        """Overwrite every buffer of ``module`` with rank 0's (in place).

        DDP broadcasts rank 0's buffers at the start of each forward, but a
        training forward then updates them with the local batch (BatchNorm
        running statistics), and an EMA copy averages those local buffers.
        Before validation and a checkpoint save every rank must hold the model
        rank 0 writes, or the pooled metric belongs to no single model.
        """
        buffers = list(module.named_buffers())
        # Name, dtype and shape: a broadcast of buffers whose sizes differ
        # between ranks is not an exception but an abort (gloo EnforceNotMet,
        # SIGABRT), e.g. a non-persistent cache that follows the last batch
        # under broadcast_buffers=False (review round 2).
        self.require_same_names(
            [f"{n}|{b.dtype}|{tuple(b.shape)}" for n, b in buffers],
            what=f"{what} buffer names, dtypes or shapes",
        )
        for _, buf in buffers:
            dist.broadcast(buf.data, src=0)

    # -- agreement -------------------------------------------------------

    def run_agreed(self, fn: Callable[[], T], *, what: str) -> T:
        """Run ``fn`` on every rank; if it raised anywhere, raise everywhere.

        The rank that failed re-raises its own exception; the others raise
        PeerRankError. The closing all-reduce is also the barrier: no rank
        leaves before every rank finished ``fn`` (e.g. rank 0's checkpoint
        write is complete before any rank continues).
        """
        result: Any = None
        error: Optional[BaseException] = None
        try:
            result = fn()
        except Exception as exc:  # re-raised below, after the peers know
            error = exc
        failed_somewhere = self.any_true(error is not None)
        if error is not None:
            raise error
        if failed_somewhere:
            raise PeerRankError(f"{what} failed on a peer rank (see that rank's log)")
        return result

    def require_equal_int(self, value: int, *, what: str) -> int:
        """``value`` must be the same on every rank (min == max)."""
        lo, neg_hi = self._all_reduce_int([value, -value], dist.ReduceOp.MIN)
        hi = -neg_hi
        if lo != hi:
            raise DistributedContractError(
                f"{what} differs between ranks (min {lo}, max {hi})"
            )
        return value

    def require_rank0_agreement(self, values: Sequence[float], *, what: str) -> None:
        """Every rank's float64 ``values`` must equal rank 0's, bit for bit.

        Bitwise, so NaN == NaN and 0.0 != -0.0 (None is passed as NaN by
        callers). Mismatch raises DistributedContractError on every rank.
        """
        mine = torch.tensor([float(v) for v in values], dtype=torch.float64)
        ref = mine.clone().to(self.device)
        dist.broadcast(ref, src=0)
        mismatch = not torch.equal(
            mine.view(torch.int64), ref.cpu().view(torch.int64)
        )
        if self.any_true(mismatch):
            raise DistributedContractError(f"{what} differs from rank 0 on at least one rank")

    def check_exact_partition(
        self, indices: Sequence[int], dataset_len: int, *, what: str
    ) -> None:
        """The ranks' ``indices`` must cover ``range(dataset_len)`` exactly once.

        Counts every index over all ranks (one int64 all-reduce of length
        ``dataset_len``). Duplicates (DistributedSampler padding), gaps
        (drop_last, a non-distributed sampler on some ranks) and foreign
        indices raise DistributedContractError on every rank.
        """
        self.require_equal_int(dataset_len, what=f"{what}: dataset length")
        idx = torch.as_tensor(list(indices), dtype=torch.int64)
        out_of_range = bool(idx.numel()) and bool(((idx < 0) | (idx >= dataset_len)).any())
        if self.any_true(out_of_range):
            raise DistributedContractError(
                f"{what}: an index outside [0, {dataset_len}) on at least one rank"
            )
        if dataset_len == 0:
            return  # nothing to cover; any index was out of range above
        counts = torch.bincount(idx, minlength=dataset_len).to(torch.int64).to(self.device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        counts = counts.cpu()
        duplicated = int((counts > 1).sum())
        missing = int((counts == 0).sum())
        if duplicated or missing:
            raise DistributedContractError(
                f"{what} is not an exact partition over {self.world_size} ranks: "
                f"{duplicated} sample(s) on more than one rank or batch, "
                f"{missing} of {dataset_len} sample(s) on none. Use "
                f"ExactDistributedEvalSampler (DistributedSampler pads with repeated "
                f"samples when the length is not a multiple of the world size)."
            )
