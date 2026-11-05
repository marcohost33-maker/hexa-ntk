#!/usr/bin/env python3
"""T1 baseline refinement and linear comparator on pilot ids: decision support for D-T1-1 / D-T1-2 (owner).

``spec/PHASE1_PROTOCOL_v1.md`` sections 2a and 2b (amendment 2026-11-05). On the
two preregistered domains at the ID lattice (triangle ``N = 32``, torus
``N = 32``) this script measures, with the registered task configuration [D1],
the registered timing protocol (3 warm-ups, 10 repeats) and the registered steps
grid ``1 .. 64``:

1. the Crank-Nicolson cost-accuracy curve with the proposed refinements
   ``k = 1, 2, 4`` (``N_solve = 32, 64, 128``), errors on the ``N = 32`` nodes;
2. the Pareto frontier of the registered curve (``k = 1``) and of the proposed
   one (``k = 1, 2, 4``), their floors and the "window" of errors a model could
   reach to be "more accurate at equal runtime" against the first but not the
   second;
3. the exact linear map (``exact_linear_map``, an oracle from the basis: no data,
   no fit) as a ``LinearComparatorPoint`` on the same inputs.

Pilot ids ``900100 .. 900149`` (dataset seed 1) lie outside every registered id
range of the preregistration (train pool ``0 .. 1999``, test sets
``100000 .. 100499`` and ``200000 .. 200499``). No model is fitted: the POD-ridge
comparator is not run here (preregistration: no model on T1 data before
approval). Times are this machine's; they are not the D5 machine.

It decides nothing: the owner amends the preregistration or does not. Exit 0
iff the instrument checks hold (``k = 1`` equals the unrefined curve, the
spatial error falls at second order between ``k = 1, 2, 4`` at ``steps = 64``,
the exact map reproduces the truth to ``1e-12``) and the source revision is
bound; otherwise 1.

    python scripts/oracles/t1_baseline_refinement_pilot.py --output results/<name>.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src" / "toolkit"))
sys.path.insert(0, str(_REPO / "scripts" / "oracles"))

import numpy  # noqa: E402
import torch  # noqa: E402

import phase1_protocol_hex_v3 as pp  # noqa: E402
from heat_operator_task_hex_v3 import HeatOperatorTaskConfig, generate_dataset  # noqa: E402
from hex_fv_solver_hex_v3 import torus_hex_lattice, triangle_hex_lattice  # noqa: E402
from hex_heat_tri_reference_oracle import _report_source_binding, _source_revision  # noqa: E402

SCHEMA = "hexantk_t1_baseline_refinement_pilot_v1"
PILOT_IDS = tuple(range(900100, 900150))
DATASET_SEED = 1
TIMED_SAMPLES = 10
EVAL_N = 32
CASES = (("triangle_dirichlet", triangle_hex_lattice), ("torus_periodic", torus_hex_lattice))
# Registered id ranges of the preregistration (section 2 [D3]); the pilot must avoid all of them.
_REGISTERED_RANGES = (range(0, 2000), range(100000, 100500), range(200000, 200500))


def _frontier(points: list) -> list:
    times = [p.median_wall_time_s for p in points]
    errors = [p.mean_rel_l2 for p in points]
    return [{"n_solve": points[i].n_solve, "steps": points[i].steps, "median_wall_time_s": times[i],
             "mean_rel_l2": errors[i]} for i in pp.pareto_frontier(times, errors)]


# Hypothetical GINO outcomes for the decision map (no model: a grid of latencies and upper IQM bounds).
_MAP_LATENCIES = tuple(10.0 ** (-4.0 + 0.1 * i) for i in range(41))      # 0.1 ms .. 1 s
_MAP_UPPERS = tuple(10.0 ** (-5.0 + 0.1 * i) for i in range(46))         # 1e-5 .. 10^-0.5
# Named examples: the Gesamtbericht's case (1e-3 at 20 ms) and two neighbours.
_MAP_EXAMPLES = ((0.020, 1e-3), (0.010, 1.5e-3), (0.002, 3e-3))


def _outcome(upper: float, latency: float, times: list, errors: list) -> dict:
    runtime, accuracy, verdict = pp._rules(upper, latency, times, errors)
    floor = (not accuracy.evaluable) and accuracy.reason.startswith("baseline floor")
    return {"verdict": verdict, "runtime_evaluable": runtime.evaluable, "accuracy_evaluable": accuracy.evaluable,
            "baseline_floor": floor, "runtime_reason": runtime.reason, "accuracy_reason": accuracy.reason}


def _decision_map(registered: list, proposed: list) -> dict:
    """How the preregistered rules (section 6) answer hypothetical outcomes under both curves."""
    curves = {name: ([p.median_wall_time_s for p in pts], [p.mean_rel_l2 for p in pts])
              for name, pts in (("registered_k1", registered), ("proposed_k124", proposed))}
    counts = {"cells": 0, "same_verdict": 0, "h0_to_advantage": 0, "advantage_to_h0": 0,
              "floor_k1_evaluated_k124": 0, "runtime_unevaluable_k1_evaluable_k124": 0}
    for latency in _MAP_LATENCIES:
        for upper in _MAP_UPPERS:
            a = _outcome(upper, latency, *curves["registered_k1"])
            b = _outcome(upper, latency, *curves["proposed_k124"])
            counts["cells"] += 1
            counts["same_verdict"] += a["verdict"] == b["verdict"]
            counts["h0_to_advantage"] += a["verdict"] == "H0" and b["verdict"] == "advantage"
            counts["advantage_to_h0"] += a["verdict"] == "advantage" and b["verdict"] == "H0"
            counts["floor_k1_evaluated_k124"] += a["baseline_floor"] and b["accuracy_evaluable"]
            counts["runtime_unevaluable_k1_evaluable_k124"] += (not a["runtime_evaluable"]) and b["runtime_evaluable"]
    examples = [{"latency_s": t, "upper": e, "registered_k1": _outcome(e, t, *curves["registered_k1"]),
                 "proposed_k124": _outcome(e, t, *curves["proposed_k124"])} for t, e in _MAP_EXAMPLES]
    return {"hypothetical": "no model: latencies 1e-4..1 s and upper IQM bounds 1e-5..10^-0.5 on a log grid, "
                            "step 0.1 decade; the preregistered rules of section 6 (phase1_protocol _rules)",
            "counts": counts, "examples": examples}


def _case(domain: str, factory) -> dict:
    config = HeatOperatorTaskConfig(domain=domain)
    lattice = factory(EVAL_N)
    data = generate_dataset(config, n=EVAL_N, sample_ids=PILOT_IDS, dataset_seed=DATASET_SEED)
    started = time.perf_counter()
    points = pp.baseline_cost_accuracy_curve(config, data["u0"], data["uT"], lattice,
                                             timed_samples=TIMED_SAMPLES, refinements=pp.PROPOSED_REFINEMENTS,
                                             coefficients=data["coefficients"])
    curve_s = time.perf_counter() - started
    unrefined = pp.baseline_cost_accuracy_curve(config, data["u0"], data["uT"], lattice, timed_samples=1)
    exact = pp.linear_comparator_point(config, data["u0"], data["uT"], lattice, timed_samples=TIMED_SAMPLES)

    registered = [p for p in points if p.n_solve == EVAL_N]
    frontier_k1 = _frontier(registered)
    frontier_all = _frontier(list(points))
    floor_k1 = min(f["mean_rel_l2"] for f in frontier_k1)
    floor_all = min(f["mean_rel_l2"] for f in frontier_all)
    t_all = [p.median_wall_time_s for p in points]
    e_all = [p.mean_rel_l2 for p in points]
    # At every time of the registered frontier: the error the proposed frontier reaches at that time.
    at_registered_times = [{"median_wall_time_s": f["median_wall_time_s"], "registered_error": f["mean_rel_l2"],
                            "proposed_error": pp.error_at_time(t_all, e_all, f["median_wall_time_s"])}
                           for f in frontier_k1]
    floor64 = {p.n_solve: p.mean_rel_l2 for p in points if p.steps == 64}
    checks = {
        # Errors are deterministic: the k = 1 block must equal the unrefined curve's errors exactly.
        "k1_equals_unrefined_errors": [p.mean_rel_l2 for p in registered] == [p.mean_rel_l2 for p in unrefined],
        "second_order_in_h_at_steps64": all(3.0 < floor64[k * EVAL_N] / floor64[2 * k * EVAL_N] < 5.0
                                            for k in (1, 2)),
        "exact_map_reproduces_truth": exact.mean_rel_l2 <= 1e-12,
    }
    return {
        "domain": domain, "n": EVAL_N, "num_nodes": lattice.num_nodes,
        "sample_ids": [PILOT_IDS[0], PILOT_IDS[-1]], "sample_count": len(PILOT_IDS),
        "dataset_sha256": data["manifest"]["dataset_sha256"],
        "curve_wall_time_s": curve_s,
        "points": [p.to_dict() for p in points],
        "exact_linear_map": exact.to_dict(),
        "frontier_registered_k1": frontier_k1,
        "frontier_proposed_k124": frontier_all,
        "floor_registered_k1": floor_k1,
        "floor_proposed_k124": floor_all,
        "floor_ratio": floor_k1 / floor_all,
        "spurious_window": {
            "meaning": ("a model error e with floor_proposed < e < floor_registered, at a latency inside the "
                        "proposed frontier's time range, is 'more accurate at equal runtime' against the "
                        "registered curve only if the registered frontier at that latency lies above e; the "
                        "proposed frontier answers the same latency with proposed_error"),
            "low": floor_all, "high": floor_k1,
        },
        "registered_frontier_vs_proposed": at_registered_times,
        "error_at_steps64_by_n_solve": {str(k): v for k, v in sorted(floor64.items())},
        "decision_map": _decision_map(registered, list(points)),
        "checks": checks,
    }


def build_report() -> dict:
    assert not any(i in r for i in PILOT_IDS for r in _REGISTERED_RANGES), "pilot ids overlap a registered range"
    cases = [_case(domain, factory) for domain, factory in CASES]
    passed = all(all(c["checks"].values()) for c in cases)
    return {
        "schema": SCHEMA,
        "claim_boundary": ("instrument measurement on pilot ids outside the registered id ranges; not a Phase 1 "
                           "result; no model fitted; wall times are this machine's, not the D5 machine's; "
                           "decision support for the owner amendments D-T1-1 / D-T1-2 (ADR-003)"),
        "source": _source_revision(),
        "environment": {
            "python": platform.python_version(), "numpy": numpy.__version__, "torch": torch.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                         "MKL_NUM_THREADS")},
        },
        "task_config": {d: HeatOperatorTaskConfig(domain=d).canonical() for d, _ in CASES},
        "timing_protocol": pp.TimingProtocol().canonical(),
        "steps_grid": list(pp._DEFAULT_STEPS_GRID),
        "refinements": list(pp.PROPOSED_REFINEMENTS),
        "registered_refinements": list(pp._REGISTERED_REFINEMENTS),
        "timed_samples": TIMED_SAMPLES,
        "dataset_seed": DATASET_SEED,
        "cases": cases,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the JSON report here (default: stdout)")
    args = parser.parse_args(argv)
    report = build_report()
    text = json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    for case in report["cases"]:
        exact = case["exact_linear_map"]
        print(f"[T1-REFINEMENT] {case['domain']} N={case['n']}: floor k=1 {case['floor_registered_k1']:.3e}, "
              f"k=1,2,4 {case['floor_proposed_k124']:.3e} (x{case['floor_ratio']:.2f}); exact map "
              f"{exact['mean_rel_l2']:.1e} at {exact['median_wall_time_s']:.2e} s; checks {case['checks']}",
              file=sys.stderr)
    bound = _report_source_binding(report["source"], "T1-REFINEMENT")
    ok = report["passed"] and bound
    print(f"[T1-REFINEMENT] passed={report['passed']} bound={bound} | {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
