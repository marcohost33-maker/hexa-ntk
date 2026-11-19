#!/usr/bin/env python3
"""Evidence writer: the pinned GINO through HexTrainingRunner, without and with gino_training_hex_v3.

Contract: spec/GINO_TRAINING_PHASE1_v1.md. Synthetic tensors only (standard
normal fields on the N = 32 triangle lattice); no T1 data is generated, loaded
or fit (spec/PREREGISTRATION_PHASE1_v1.md, lines 14-15).

Part A, "plain": the smallest wrapper that makes the runner's ``model(x)`` call
possible and does nothing else. It records what the runner does with the
backend's own state (spec section 1, findings 2-4). These are the failures the
module exists for; if one of them no longer occurs (another backend or torch
version), the premise has changed and the script reports it.

Part B, "module": the same runs through ``build_phase1_runner``.

Exit 0 only if every plain run fails as recorded in the spec, every module
check holds, the backend is neuraloperator 2.0.0 and the source revision is
bound (clean work tree, verified HEAD); otherwise 1.

    python scripts/oracles/gino_runner_integration_oracle.py --output results/<name>.json
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
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src" / "toolkit"))
sys.path.insert(0, str(_REPO / "scripts" / "oracles"))

import numpy  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from gino_adapter_hex_v3 import GinoPhase1Config, build_gino, predict_gino, prepare_gino_batch  # noqa: E402
from gino_training_hex_v3 import (  # noqa: E402
    Phase1TrainingRecipe,
    RelativeL2Loss,
    RunnerPairDataset,
    build_phase1_runner,
    mean_rel_l2,
)
from hex_fv_solver_hex_v3 import triangle_hex_lattice  # noqa: E402
from hex_heat_tri_reference_oracle import _report_source_binding, _source_revision  # noqa: E402
from runner_hex_v3 import HexTrainingRunner, RunnerConfig  # noqa: E402

_LATTICE = triangle_hex_lattice(32)
_NODES = _LATTICE.num_nodes
_CONFIG = GinoPhase1Config(latent_resolution=16, n_modes=4, hidden_channels=8, n_layers=2, radius_factor=2.5)
_EPOCHS = 4
_SEED = 3

# What part A must show (spec section 1): scenario -> text the error must contain.
_PLAIN_EXPECTED = {
    "ema_run": "'dict' object has no attribute 'detach'",
    "weights_only_load_best": "torch._C._nn.gelu",
    "weights_only_load_last": "torch._C._nn.gelu",
    "trusted_resume": 'Unexpected key(s) in state_dict: "_metadata"',
}


class _Source:
    """Standard-normal {"u0", "uT"} samples; the values carry no information about T1."""

    def __init__(self, count: int, seed: int) -> None:
        rng = numpy.random.default_rng(seed)
        self.u0 = rng.standard_normal((count, _NODES))
        self.u_t = rng.standard_normal((count, _NODES))

    def __len__(self) -> int:
        return len(self.u0)

    def __getitem__(self, index: int) -> dict:
        return {"trajectory_id": index, "u0": self.u0[index], "uT": self.u_t[index]}


def _data() -> tuple[RunnerPairDataset, RunnerPairDataset]:
    return RunnerPairDataset(_Source(8, 1), _NODES), RunnerPairDataset(_Source(6, 2), _NODES)


class _PlainWrapper(nn.Module):
    """model(x) for the runner; the backend's state is passed on untouched."""

    def __init__(self) -> None:
        super().__init__()
        self.gino = build_gino(_CONFIG, _LATTICE)
        geometry = prepare_gino_batch(_LATTICE, numpy.zeros(_NODES), _CONFIG)
        self.register_buffer("input_geom", geometry["input_geom"], persistent=False)
        self.register_buffer("latent_queries", geometry["latent_queries"], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gino(input_geom=self.input_geom, latent_queries=self.latent_queries,
                         output_queries=self.input_geom, x=x.unsqueeze(-1))[..., 0]


def _plain_runner(output_dir: str, *, use_ema: bool, resume_from: str | None = None) -> HexTrainingRunner:
    train, val = _data()
    torch.manual_seed(_SEED)
    model = _PlainWrapper()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_EPOCHS)
    config = RunnerConfig(output_dir=output_dir, num_epochs=_EPOCHS, use_ema=use_ema,
                          grad_clip_max_norm=float("inf"), best_metric_name="val_loss",
                          resume_from=resume_from, resume_trusted=resume_from is not None)
    return HexTrainingRunner(
        config, model, optimizer,
        DataLoader(train, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(_SEED)),
        DataLoader(val, batch_size=4), loss_fn=RelativeL2Loss(), scheduler=scheduler,
    )


def _outcome(action) -> dict:
    try:
        action()
    except Exception as exc:  # the oracle records whatever the run raises
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:2000]}
    return {"ok": True, "error_type": None, "error": None}


def _plain_report() -> dict:
    report = {}
    with tempfile.TemporaryDirectory() as tmp:
        report["ema_run"] = _outcome(lambda: _plain_runner(tmp, use_ema=True).run())
    with tempfile.TemporaryDirectory() as tmp:
        report["no_ema_run"] = _outcome(lambda: _plain_runner(tmp, use_ema=False).run())
        checkpoints = os.path.join(tmp, "checkpoints")
        for name in ("best", "last"):
            path = os.path.join(checkpoints, f"{name}.pt")
            report[f"weights_only_load_{name}"] = _outcome(lambda p=path: torch.load(p, weights_only=True))
        resume = os.path.join(checkpoints, "resume_last.pt")
        report["trusted_resume"] = _outcome(
            lambda: _plain_runner(tmp, use_ema=False, resume_from=resume).run())
    for name, needle in _PLAIN_EXPECTED.items():
        report[name]["expected_error_contains"] = needle
        report[name]["as_recorded"] = (not report[name]["ok"]) and needle in (report[name]["error"] or "")
    report["no_ema_run"]["as_recorded"] = report["no_ema_run"]["ok"]
    return report


def _equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class _Stop(Exception):
    pass


def _module_runner(output_dir: str, *, use_ema: bool, steps: int = 0, resume_from: str | None = None):
    recipe = Phase1TrainingRecipe(epochs=_EPOCHS, learning_rate=1e-3, batch_size=4, seed=_SEED, use_ema=use_ema)
    train, val = _data()
    return build_phase1_runner(recipe, _CONFIG, train, val, output_dir=output_dir, resume_from=resume_from,
                               resume_trusted=resume_from is not None, save_every_n_steps=steps)


def _resumed_equals_full(*, use_ema: bool, mid_epoch: bool) -> bool:
    steps = 1 if mid_epoch else 0
    with tempfile.TemporaryDirectory() as full_dir, tempfile.TemporaryDirectory() as cut_dir:
        full_runner, full_model = _module_runner(full_dir, use_ema=use_ema, steps=steps)
        full_runner.run()
        cut_runner, _ = _module_runner(cut_dir, use_ema=use_ema, steps=steps)
        if mid_epoch:
            one_step, calls = cut_runner._train_one_step, [0]

            def stopping(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 4:  # second step of the second epoch
                    raise _Stop()
                return one_step(*args, **kwargs)

            cut_runner._train_one_step = stopping
            try:
                cut_runner.run()
            except _Stop:
                pass
            else:
                return False
        else:
            cut_runner.config.num_epochs = 2
            cut_runner.run()
        resume = os.path.join(cut_dir, "checkpoints", "resume_last.pt")
        resumed_runner, resumed_model = _module_runner(cut_dir, use_ema=use_ema, steps=steps, resume_from=resume)
        if resumed_runner.run()["resume_report"] is None:
            return False
        same = _equal(full_model.state_dict(), resumed_model.state_dict())
        if use_ema:
            same = same and _equal(full_runner.ema_model.state_dict(), resumed_runner.ema_model.state_dict())
        return same


def _module_report() -> dict:
    checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        _, probe = _module_runner(tmp, use_ema=False)
        probe.eval()
        field = torch.randn(3, _NODES, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            checks["forward_equals_predict_gino_bitwise"] = bool(
                torch.equal(probe(field), predict_gino(probe.gino, _LATTICE, field, _CONFIG)))
        checks["state_is_tensors_only"] = all(isinstance(v, torch.Tensor) for v in probe.state_dict().values())
    for use_ema in (False, True):
        tag = "ema" if use_ema else "no_ema"
        with tempfile.TemporaryDirectory() as tmp:
            runner, model = _module_runner(tmp, use_ema=use_ema)
            initial = {k: v.clone() for k, v in model.state_dict().items()}
            summary = runner.run()
            checks[f"{tag}_run_completes"] = summary["global_step"] == 2 * _EPOCHS
            checks[f"{tag}_weights_moved"] = not _equal(initial, model.state_dict())
            for name in ("best", "last"):
                outcome = _outcome(lambda n=name: torch.load(os.path.join(tmp, "checkpoints", f"{n}.pt"),
                                                             weights_only=True))
                checks[f"{tag}_weights_only_load_{name}"] = outcome["ok"]
            checks[f"{tag}_mean_rel_l2_finite"] = bool(numpy.isfinite(mean_rel_l2(model, _data()[1])))
        checks[f"{tag}_resume_at_epoch_boundary_bitwise"] = _resumed_equals_full(use_ema=use_ema, mid_epoch=False)
        checks[f"{tag}_resume_inside_epoch_bitwise"] = _resumed_equals_full(use_ema=use_ema, mid_epoch=True)
    return checks


def build_report() -> dict:
    backend = importlib.metadata.version("neuraloperator")
    return {
        "source": _source_revision(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": numpy.__version__, "neuraloperator": backend,
                        "platform": platform.platform(), "torch_threads": torch.get_num_threads()},
        "data": "synthetic standard-normal tensors on the N = 32 triangle lattice; no T1 data",
        "gino_config": _CONFIG.canonical(),
        "plain_wrapper": _plain_report(),
        "module": _module_report(),
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
    ok = report["environment"]["neuraloperator"] == "2.0.0"
    for name, entry in report["plain_wrapper"].items():
        verdict = "as recorded" if entry["as_recorded"] else "CHANGED"
        print(f"[GINO-RUNNER] plain  {name:24s} {'ok' if entry['ok'] else entry['error_type']:20s} {verdict}",
              file=sys.stderr)
        ok = ok and entry["as_recorded"]
    for name, passed in report["module"].items():
        print(f"[GINO-RUNNER] module {name:44s} {'PASS' if passed else 'FAIL'}", file=sys.stderr)
        ok = ok and passed
    ok = _report_source_binding(report["source"], "GINO-RUNNER") and ok
    print(f"[GINO-RUNNER] {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
