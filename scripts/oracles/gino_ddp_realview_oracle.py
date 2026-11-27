#!/usr/bin/env python3
"""Evidence writer: the pinned GINO under DDP on gloo with its complex weights held as real views.

Contract: spec/GINO_TRAINING_PHASE1_v1.md, section 8a, option 4 (measured, not
adopted). Synthetic tensors only (standard-normal fields on the N = 32 triangle
lattice, as in ``gino_ddp_gloo_oracle.py``); no T1 data is generated, loaded or fit.

The wrapper ``RealViews`` (defined here, measurement only, nothing in the toolkit
uses it) registers one float32 parameter per complex parameter of the module, a
``torch.view_as_real`` view of the SAME storage, and calls the module through
``torch.func.functional_call`` with ``torch.view_as_complex`` of those views in
place of the complex parameters. The complex parameters are left out of DDP
(``_ddp_params_and_buffers_to_ignore``, the attribute DDP reads; the same private
mechanism as option 2) and out of the optimizer. DDP then broadcasts and
all-reduces only float32 tensors, inside its own reducer: unlike option 2 there
is no collective outside DDP.

Part A, "premise": ``DDP(GinoRunnerModule)`` on two gloo ranks fails with
"Invalid scalar type" (the finding of section 8a, reproduced).

Part B, "gradients": ``DDP(RealViews(module))`` on two gloo ranks, one forward and
backward, rank r on half r of a batch of 4 (one torch thread per process). The
reference is one process without DDP and without the wrapper. Checks:
I1 every gradient on both ranks equals ``g1 * 0.5 + g2 * 0.5`` bitwise, where g1,
g2 are the single-process gradients of the two halves (this is the reducer's own
arithmetic for world 2: each rank multiplies by 1/2, then gloo sums two terms);
I2 every gradient is within ``64 * eps32 * S`` of the full-batch gradient G, with
S = max over entries of ``(|g1| + |g2|) / 2 + |G|`` (the operand scale of the
float32 sums that differ; NaN fails the check).

Part C, "optimizer": one SGD and one Adam step (lr 1e-2), three times, on a complex
parameter and on its real view with the same gradient. Recorded, not required:
Adam (torch 2.6.0 steps complex parameters on their real view) agrees bitwise;
SGD does not in every entry, because ``add_(grad, alpha=-lr)`` rounds a complex
tensor differently from its real view (up to 1 ulp).

Part D, "alias": after an optimizer step on the views the module's complex
parameters hold the same values in the same storage (data_ptr equal).

Part E, "runner": two gloo ranks through HexTrainingRunner with distributed=True,
3 epochs, SGD and Adam, with ``TORCH_DISTRIBUTED_DEBUG=DETAIL`` (the collective
order check of spec/DDP_RUNNER_v1.md R14), in two variants. "Aliases saved": the
wrapper's state dict holds both the complex weights and their views; the
runner's first checkpoint fails ("Cannot save multiple tensors or storages that
view the same data as different types", torch.save). "Views dropped": the wrapper
leaves the views out of its state dict; both ranks must finish with bitwise equal
weights. Against one process (batch 4) the distance is recorded only: it is the
batch-split rounding that part D of gino_ddp_gloo_oracle.py measured. Resume from
such a checkpoint is not measured.

Exit 0 only if A fails as recorded, I1 and I2 hold, D holds, E "aliases saved"
fails as recorded and E "views dropped" finishes on both ranks with equal
weights, the backend is neuraloperator 2.0.0 and the source revision is bound
(clean work tree, verified HEAD); otherwise 1.

    python scripts/oracles/gino_ddp_realview_oracle.py --output results/<name>.json
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src" / "toolkit"))
sys.path.insert(0, str(_REPO / "scripts" / "oracles"))

import numpy  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.func import functional_call  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from ddp_hex_v3 import ExactDistributedEvalSampler  # noqa: E402
from gino_ddp_gloo_oracle import _CONFIG, _EPOCHS, _LR, _PREMISE_ERROR, _data, _module, _runner  # noqa: E402
from gino_training_hex_v3 import RelativeL2Loss  # noqa: E402
from hex_heat_tri_reference_oracle import _report_source_binding, _source_revision  # noqa: E402
from runner_hex_v3 import HexTrainingRunner, RunnerConfig  # noqa: E402

_WORLD = 2
_PER_RANK = 2
_EPS32 = float(numpy.finfo(numpy.float32).eps)
_TOL_FACTOR = 64
_ALIAS_ERROR = "view the same data as different types"


class RealViews(nn.Module):
    """The module with its complex parameters presented as float32 views of the same storage."""

    def __init__(self, inner: nn.Module, *, drop_views_from_state: bool = False) -> None:
        super().__init__()
        self.inner = inner
        self.drop_views_from_state = drop_views_from_state
        self.complex_names = [n for n, p in inner.named_parameters() if p.is_complex()]
        self.views = nn.ParameterList(
            [nn.Parameter(torch.view_as_real(inner.get_parameter(n).detach())) for n in self.complex_names])
        for n in self.complex_names:
            inner.get_parameter(n).requires_grad_(False)
        self._ddp_params_and_buffers_to_ignore = [f"inner.{n}" for n in self.complex_names]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swap = {n: torch.view_as_complex(v) for n, v in zip(self.complex_names, self.views)}
        return functional_call(self.inner, swap, (x,), strict=False)

    def trainable(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def state_dict(self, *, destination=None, prefix: str = "", keep_vars: bool = False):
        # The views alias the complex weights (other dtype, same storage); torch.save
        # refuses such a pair, so variant "views dropped" leaves them out.
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.drop_views_from_state:
            for key in [k for k in state if k.startswith(prefix + "views.")]:
                del state[key]
        return state


def _batch() -> tuple:
    train, _ = _data()
    x = torch.stack([train[i]["input"] for i in range(4)])
    y = torch.stack([train[i]["target"] for i in range(4)])
    return x, y


def _real(t: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(t) if t.is_complex() else t


def _named_grads(wrapped: RealViews) -> dict:
    grads = {n: p.grad.detach().clone() for n, p in wrapped.inner.named_parameters() if p.grad is not None}
    for n, v in zip(wrapped.complex_names, wrapped.views):
        if v.grad is not None:
            grads[n] = v.grad.detach().clone()
    return grads


def _outcome(fn) -> dict:
    try:
        return {"ok": True, "value": fn()}
    except Exception as exc:  # recorded, judged by the parent
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:300]}


def _worker(rank: int, root: str, part: str, optimizer: str) -> None:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    if part.startswith("runner"):
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    dist.init_process_group("gloo", init_method=(Path(root) / "rendezvous").as_uri(), rank=rank,
                            world_size=_WORLD, timeout=timedelta(seconds=120))
    try:
        if part == "premise":
            result = {"ddp_construction": _outcome(lambda: DDP(_module()) and None)}
        elif part == "grads":
            def grads() -> dict:
                model = DDP(RealViews(_module()))
                x, y = _batch()
                rows = slice(_PER_RANK * rank, _PER_RANK * rank + _PER_RANK)
                RelativeL2Loss()(model(x[rows]), y[rows]).backward()
                torch.save(_named_grads(model.module), Path(root) / f"grads{rank}.pt")
                return {"complex_params_without_grad": all(
                    model.module.inner.get_parameter(n).grad is None for n in model.module.complex_names)}
            result = {"run": _outcome(grads)}
        else:
            def run() -> list:
                module = _module()
                wrapped = RealViews(module, drop_views_from_state=part == "runner_views_dropped")
                train, val = _data()
                model = DDP(wrapped)
                runner = _runner_on(Path(root) / "out", model, wrapped.trainable(), _loader(train, rank),
                                    _loader(val, rank, True), optimizer, distributed=True, rank=rank)
                runner.run()
                return _flat(module).tolist()
            result = {"run": _outcome(run)}
    finally:
        (Path(root) / f"rank{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


def _runner_on(out: Path, model: nn.Module, params: list, train: DataLoader, val: DataLoader, optimizer: str,
               *, distributed: bool, rank: int) -> HexTrainingRunner:
    """``gino_ddp_gloo_oracle._runner`` with the optimizer on ``params`` (the views, not the complex weights)."""
    opt = torch.optim.SGD(params, lr=_LR) if optimizer == "sgd" else torch.optim.Adam(params, lr=_LR)
    config = RunnerConfig(output_dir=str(out), num_epochs=_EPOCHS, grad_clip_max_norm=float("inf"),
                          use_ema=False, use_amp=False, log_every_n_steps=0, best_metric_name="val_loss",
                          distributed=distributed, is_main_process=rank == 0)
    return HexTrainingRunner(config, model, opt, train, val, loss_fn=RelativeL2Loss(),
                             scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_EPOCHS),
                             device="cpu")


def _loader(dataset, rank: int, evaluation: bool = False) -> DataLoader:
    sampler = (ExactDistributedEvalSampler(dataset) if evaluation
               else DistributedSampler(dataset, _WORLD, rank, shuffle=False))
    return DataLoader(dataset, batch_size=_PER_RANK, sampler=sampler)


def _flat(module: nn.Module) -> torch.Tensor:
    return torch.cat([_real(p.detach()).reshape(-1) for p in module.parameters()])


def _job(part: str, optimizer: str = "", keep: bool = False) -> list:
    with tempfile.TemporaryDirectory() as root:
        mp.start_processes(_worker, args=(root, part, optimizer), nprocs=_WORLD, start_method="spawn", join=True)
        ranks = [json.loads((Path(root) / f"rank{r}.json").read_text()) for r in range(_WORLD)]
        if keep:
            for r in range(_WORLD):
                ranks[r]["grads"] = torch.load(Path(root) / f"grads{r}.pt") if ranks[r]["run"]["ok"] else None
        return ranks


def _premise_report() -> dict:
    ranks = _job("premise")
    as_recorded = all(not r["ddp_construction"]["ok"] and _PREMISE_ERROR in r["ddp_construction"]["error"]
                      for r in ranks)
    return {"ranks": ranks, "as_recorded": as_recorded}


def _single_grads(rows: slice) -> dict:
    torch.set_num_threads(1)
    module = _module()
    x, y = _batch()
    RelativeL2Loss()(module(x[rows]), y[rows]).backward()
    return {n: _real(p.grad.detach()).clone() for n, p in module.named_parameters() if p.grad is not None}


def _gradient_report() -> dict:
    ranks = _job("grads", keep=True)
    full, g1, g2 = _single_grads(slice(0, 4)), _single_grads(slice(0, 2)), _single_grads(slice(2, 4))
    rows, i1, i2 = {}, True, True
    ran = all(r["run"]["ok"] and r["run"]["value"]["complex_params_without_grad"] for r in ranks)
    names_match = ran and all(sorted(r["grads"]) == sorted(full) for r in ranks)
    for n in sorted(full):
        if not names_match:
            break
        d0, d1 = ranks[0]["grads"][n], ranks[1]["grads"][n]
        halves = g1[n] * 0.5 + g2[n] * 0.5
        bitwise = bool(torch.equal(d0, d1) and torch.equal(d0, halves))
        scale = float((0.5 * (g1[n].abs() + g2[n].abs()) + full[n].abs()).max())
        diff = float((d0 - full[n]).abs().max())
        tol = _TOL_FACTOR * _EPS32 * scale
        within = diff <= tol  # NaN compares False: fail-closed
        i1 &= bitwise
        i2 &= within
        rows[n] = {"bitwise_vs_halves": bitwise, "max_abs_diff_vs_full_batch": diff, "tolerance": tol,
                   "operand_scale": scale}
    for r in ranks:
        r.pop("grads", None)
    return {"ranks": ranks, "per_parameter": rows, "ran": ran, "names_match": names_match,
            "I1_bitwise_vs_halves": names_match and i1, "I2_within_tolerance_vs_full_batch": names_match and i2}


def _optimizer_report() -> dict:
    out = {}
    for name in ("sgd", "adam"):
        torch.manual_seed(7)
        z = torch.randn(8, 8, 4, 3, dtype=torch.complex64)
        grad = torch.randn(8, 8, 4, 3, dtype=torch.complex64)
        pc = nn.Parameter(z.clone())
        pr = nn.Parameter(torch.view_as_real(z).clone())
        kind = torch.optim.SGD if name == "sgd" else torch.optim.Adam
        oc, orr = kind([pc], lr=1e-2), kind([pr], lr=1e-2)
        for _ in range(3):
            pc.grad = grad.clone()
            pr.grad = torch.view_as_real(grad).clone()
            oc.step()
            orr.step()
        diff = (torch.view_as_real(pc.detach()) - pr.detach()).abs()
        out[name] = {"bitwise": bool(torch.equal(torch.view_as_real(pc.detach()), pr.detach())),
                     "max_abs_diff": float(diff.max()), "entries_differing": int((diff > 0).sum()),
                     "entries": diff.numel()}
    return out


def _alias_report() -> dict:
    module = _module()
    wrapped = RealViews(module)
    x, y = _batch()
    opt = torch.optim.Adam(wrapped.trainable(), lr=1e-2)
    RelativeL2Loss()(wrapped(x), y).backward()
    before = [module.get_parameter(n).detach().clone() for n in wrapped.complex_names]
    opt.step()
    rows = {}
    for n, v, old in zip(wrapped.complex_names, wrapped.views, before):
        p = module.get_parameter(n)
        rows[n] = {"same_storage": p.data_ptr() == v.data_ptr(),
                   "values_equal": bool(torch.equal(torch.view_as_real(p.detach()), v.detach())),
                   "moved": not bool(torch.equal(p.detach(), old))}
    holds = bool(rows) and all(all(r.values()) for r in rows.values())
    return {"per_parameter": rows, "holds": holds}


def _runner_variant(part: str, optimizer: str) -> dict:
    ranks = _job(part, optimizer)
    ok = all(r["run"]["ok"] for r in ranks)
    equal = ok and ranks[0]["run"]["value"] == ranks[1]["run"]["value"]
    entry = {"ranks_ok": ok, "ranks_bitwise_equal": equal,
             "errors": [r["run"].get("error") for r in ranks if not r["run"]["ok"]]}
    if ok:
        with tempfile.TemporaryDirectory() as tmp:
            module = _module()
            train, val = _data()
            _runner(Path(tmp), module, DataLoader(train, batch_size=4), DataLoader(val, batch_size=4),
                    optimizer).run()
            single = _flat(module)
        entry["max_abs_diff_vs_single_process"] = float(
            (torch.tensor(ranks[0]["run"]["value"]) - single).abs().max())
    return entry


def _runner_report() -> dict:
    """Variant "aliases saved": the runner's checkpoint refuses the state (recorded).
    Variant "views dropped": the views are left out of the state dict; both ranks must end equal."""
    saved = {o: _runner_variant("runner_aliases_saved", o) for o in ("sgd", "adam")}
    dropped = {o: _runner_variant("runner_views_dropped", o) for o in ("sgd", "adam")}
    refused_as_recorded = all(not e["ranks_ok"] and any(_ALIAS_ERROR in (m or "") for m in e["errors"])
                              for e in saved.values())
    return {"aliases_saved": saved, "views_dropped": dropped,
            "aliases_saved_refused_as_recorded": refused_as_recorded,
            "views_dropped_ranks_equal": all(e["ranks_bitwise_equal"] for e in dropped.values())}


def build_report() -> dict:
    return {
        "source": _source_revision(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": numpy.__version__,
                        "neuraloperator": importlib.metadata.version("neuraloperator"),
                        "platform": platform.platform(), "torch_threads_in_ranks": 1},
        "data": "synthetic standard-normal tensors on the N = 32 triangle lattice; no T1 data",
        "gino_config": _CONFIG.canonical(),
        "setup": {"world": _WORLD, "per_rank_batch": _PER_RANK, "tolerance_factor": _TOL_FACTOR,
                  "eps32": _EPS32},
        "premise": _premise_report(),
        "gradients": _gradient_report(),
        "optimizer": _optimizer_report(),
        "alias": _alias_report(),
        "runner": _runner_report(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the JSON report here (default: stdout)")
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    report = build_report()
    text = json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    checks = {
        "backend_is_2.0.0": report["environment"]["neuraloperator"] == "2.0.0",
        "premise_as_recorded": report["premise"]["as_recorded"],
        "I1_bitwise_vs_halves": report["gradients"]["I1_bitwise_vs_halves"],
        "I2_within_tolerance": report["gradients"]["I2_within_tolerance_vs_full_batch"],
        "alias_holds": report["alias"]["holds"],
        "runner_aliases_refused": report["runner"]["aliases_saved_refused_as_recorded"],
        "runner_views_dropped_equal": report["runner"]["views_dropped_ranks_equal"],
    }
    for name, passed in checks.items():
        print(f"[GINO-DDP-RV] {name:28s} {'PASS' if passed else 'FAIL'}", file=sys.stderr)
    ok = _report_source_binding(report["source"], "GINO-DDP-RV") and all(checks.values())
    print(f"[GINO-DDP-RV] {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
