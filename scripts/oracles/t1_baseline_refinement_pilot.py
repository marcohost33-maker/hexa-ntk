#!/usr/bin/env python3
"""T1 baseline refinement, fair vs. oracle, and the linear comparator on pilot ids (D-T1-1 / D-T1-2, owner).

``spec/PHASE1_PROTOCOL_v1.md`` sections 2a and 2b (amendment 2026-11-05). On the
preregistered domains at the ID lattice (``N = 32``) this script measures, with
the registered task configuration [D1], the registered timing protocol (3 warm-ups,
10 repeats) and the registered steps grid ``1 .. 64``:

1. the Crank-Nicolson cost-accuracy curve with the proposed refinements
   ``k = 1, 2, 4`` (``N_solve = 32, 64, 128``), errors on the ``N = 32`` nodes, for
   both ways the solve lattice gets its initial condition: ``prolongation="linear"``
   (the timed piecewise-linear prolongation of ``u0`` from ``N``: what a
   practitioner with ``u0`` on ``N`` alone can do, the fair baseline) and
   ``prolongation="exact"`` (the untimed exact field of the mode coefficients:
   an oracle upper bound, information the model never sees); each with the
   Richardson extrapolation in ``h`` of consecutive levels (``scheme
   cn_richardson_h``, reported only);
2. the interpolation error of the linear prolongation alone (the initial-condition
   error budget that bounds what the fair refinement can gain);
3. the Pareto frontiers of the registered curve (``k = 1``), the fair and the
   oracle curves (with and without Richardson), their floors, and a map of how the
   preregistered decision rules (section 6) answer hypothetical model outcomes under
   each frontier compared with the registered one;
4. the exact linear map (``exact_linear_map``, an oracle from the basis: no data,
   no fit) as a ``LinearComparatorPoint`` on the same inputs.

Pilot ids ``900100 ..`` (dataset seed 1) lie outside every registered id range of
the preregistration (train pool ``0 .. 1999``, test sets ``100000 .. 100499`` and
``200000 .. 200499``). No model is fitted: the POD-ridge comparator is not run
(preregistration: no model on T1 data before approval). Times are this machine's;
``--threads`` pins the BLAS/OpenMP/torch thread count (recorded), the default
leaves it to the libraries.

It decides nothing: the owner amends the preregistration or does not. Exit 0 iff
the instrument checks hold (the ``k = 1`` block is bitwise the same under both
prolongations, the oracle error falls at second order between the levels at
``steps = 64``, the linear prolongation alone is second order on the inputs, the
exact initial condition is the given ``u0`` on the nested nodes, the exact map
reproduces the truth to ``1e-12``) and the source revision is bound; otherwise 1.

    python scripts/oracles/t1_baseline_refinement_pilot.py --output results/<name>.json [--threads 4]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the JSON report here (default: stdout)")
    parser.add_argument("--threads", type=int, default=None,
                        help="pin OMP/OpenBLAS/MKL and torch to this many threads (set before numpy loads)")
    parser.add_argument("--samples", type=int, default=50, help="pilot samples per domain (default 50)")
    parser.add_argument("--timed-samples", type=int, default=10, help="samples timed with the protocol (default 10)")
    parser.add_argument("--eval-n", type=int, default=32, help="evaluation lattice N (default 32, the ID lattice)")
    parser.add_argument("--first-id", type=int, default=900100, help="first pilot sample id (default 900100)")
    parser.add_argument("--domains", nargs="+", default=["triangle_dirichlet", "torus_periodic"])
    parser.add_argument("--prolongations", nargs="+", default=["linear", "exact"])
    parser.add_argument("--refinements", nargs="+", type=int, default=None, help="default: PROPOSED_REFINEMENTS")
    parser.add_argument("--no-richardson", action="store_true", help="skip the Richardson pairs")
    return parser.parse_args(argv)


_ARGS = _parse(sys.argv[1:]) if __name__ == "__main__" else None
if _ARGS is not None and _ARGS.threads is not None:
    if _ARGS.threads < 1:
        raise SystemExit("--threads must be >= 1")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(_ARGS.threads)      # read by the BLAS at load time, so before numpy

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src" / "toolkit"))
sys.path.insert(0, str(_REPO / "scripts" / "oracles"))

import numpy  # noqa: E402
import torch  # noqa: E402

import phase1_protocol_hex_v3 as pp  # noqa: E402
from heat_operator_task_hex_v3 import HeatOperatorTaskConfig, evaluate_sample, generate_dataset  # noqa: E402
from hex_fv_solver_hex_v3 import (  # noqa: E402
    nested_node_index,
    prolongate_linear,
    torus_hex_lattice,
    triangle_hex_lattice,
)
from hex_heat_tri_reference_oracle import _report_source_binding, _source_revision  # noqa: E402

SCHEMA = "hexantk_t1_baseline_refinement_pilot_v2"
DATASET_SEED = 1
FACTORIES = {"triangle_dirichlet": triangle_hex_lattice, "torus_periodic": torus_hex_lattice}
# Registered id ranges of the preregistration (section 2 [D3]); the pilot must avoid all of them.
_REGISTERED_RANGES = (range(0, 2000), range(100000, 100500), range(200000, 200500))
# Hypothetical model outcomes for the decision map (no model: a log grid of latencies and upper IQM bounds).
_MAP_LATENCIES = tuple(10.0 ** (-4.0 + 0.1 * i) for i in range(41))      # 0.1 ms .. 1 s
_MAP_UPPERS = tuple(10.0 ** (-5.0 + 0.1 * i) for i in range(46))         # 1e-5 .. 10^-0.5
# Named examples: the Gesamtbericht's case (1e-3 at 20 ms) and two neighbours.
_MAP_EXAMPLES = ((0.020, 1e-3), (0.010, 1.5e-3), (0.002, 3e-3))


def _point_rows(points: list) -> list:
    return [{"scheme": p.scheme, "n_solve": p.n_solve, "prolongation": p.prolongation, "steps": p.steps,
             "median_wall_time_s": p.median_wall_time_s, "mean_rel_l2": p.mean_rel_l2} for p in points]


def _frontier(points: list) -> list:
    times = [p.median_wall_time_s for p in points]
    errors = [p.mean_rel_l2 for p in points]
    return [_point_rows([points[i]])[0] for i in pp.pareto_frontier(times, errors)]


def _outcome(upper: float, latency: float, times: list, errors: list) -> dict:
    runtime, accuracy, verdict = pp._rules(upper, latency, times, errors)
    floor = (not accuracy.evaluable) and accuracy.reason.startswith("baseline floor")
    return {"verdict": verdict, "runtime_evaluable": runtime.evaluable, "accuracy_evaluable": accuracy.evaluable,
            "baseline_floor": floor, "runtime_reason": runtime.reason, "accuracy_reason": accuracy.reason}


def _decision_map(registered: list, candidate: list) -> dict:
    """How the preregistered rules (section 6) answer hypothetical outcomes under both curves."""
    reg = ([p.median_wall_time_s for p in registered], [p.mean_rel_l2 for p in registered])
    cand = ([p.median_wall_time_s for p in candidate], [p.mean_rel_l2 for p in candidate])
    counts = {"cells": 0, "same_verdict": 0, "h0_to_advantage": 0, "advantage_to_h0": 0,
              "floor_registered_evaluated_candidate": 0, "runtime_unevaluable_registered_evaluable_candidate": 0}
    for latency in _MAP_LATENCIES:
        for upper in _MAP_UPPERS:
            a, b = _outcome(upper, latency, *reg), _outcome(upper, latency, *cand)
            counts["cells"] += 1
            counts["same_verdict"] += a["verdict"] == b["verdict"]
            counts["h0_to_advantage"] += a["verdict"] == "H0" and b["verdict"] == "advantage"
            counts["advantage_to_h0"] += a["verdict"] == "advantage" and b["verdict"] == "H0"
            counts["floor_registered_evaluated_candidate"] += a["baseline_floor"] and b["accuracy_evaluable"]
            counts["runtime_unevaluable_registered_evaluable_candidate"] += (
                (not a["runtime_evaluable"]) and b["runtime_evaluable"])
    examples = [{"latency_s": t, "upper": e, "registered": _outcome(e, t, *reg), "candidate": _outcome(e, t, *cand)}
                for t, e in _MAP_EXAMPLES]
    return {"counts": counts, "examples": examples}


def _prolongation_error_budget(config: HeatOperatorTaskConfig, data: dict, lattice, factory, refinements: tuple
                               ) -> dict:
    """Relative L2 error of the linear prolongation of u0 alone, N -> k N, against the exact fine field;
    and its order: N -> 2N against 2N -> 4N (the latter on the exact 2N field), first 10 samples."""
    out = {}
    for k in refinements:
        if k == 1:
            continue
        fine = factory(k * lattice.n)
        errors = []
        for a, u0 in zip(data["coefficients"], data["u0"]):
            exact_fine = evaluate_sample(config, a, fine)[0]
            errors.append(float(numpy.linalg.norm(prolongate_linear(u0, lattice, fine) - exact_fine)
                                / numpy.linalg.norm(exact_fine)))
        out[str(k)] = float(numpy.mean(errors))
    mid, fine = factory(2 * lattice.n), factory(4 * lattice.n)
    ratios = []
    for a, u0 in zip(data["coefficients"][:10], data["u0"][:10]):
        exact_mid, exact_fine = evaluate_sample(config, a, mid)[0], evaluate_sample(config, a, fine)[0]
        coarse_e = numpy.linalg.norm(prolongate_linear(u0, lattice, mid) - exact_mid) / numpy.linalg.norm(exact_mid)
        fine_e = numpy.linalg.norm(prolongate_linear(exact_mid, mid, fine) - exact_fine) / numpy.linalg.norm(exact_fine)
        ratios.append(float(coarse_e / fine_e))
    out["order_ratio_N_to_2N_over_2N_to_4N"] = {"min": min(ratios), "max": max(ratios),
                                                "mean": float(numpy.mean(ratios))}
    return out


def _case(domain: str, args: argparse.Namespace, refinements: tuple) -> dict:
    factory = FACTORIES[domain]
    config = HeatOperatorTaskConfig(domain=domain)
    lattice = factory(args.eval_n)
    ids = tuple(range(args.first_id, args.first_id + args.samples))
    data = generate_dataset(config, n=args.eval_n, sample_ids=ids, dataset_seed=DATASET_SEED)
    nested_ulps = 0.0
    for k in refinements:
        if k == 1:
            continue
        fine = factory(k * args.eval_n)
        index = nested_node_index(lattice, fine)
        for a, u0 in zip(data["coefficients"], data["u0"]):
            fine_u0 = evaluate_sample(config, a, fine)[0]
            nested_ulps = max(nested_ulps, float(numpy.max(numpy.abs(fine_u0[index] - u0))
                                                 / (numpy.finfo(float).eps * numpy.max(numpy.abs(u0)))))
    curves: dict = {}
    timings: dict = {}
    for mode in args.prolongations:
        started = time.perf_counter()
        curves[mode] = pp.baseline_cost_accuracy_curve(
            config, data["u0"], data["uT"], lattice, timed_samples=args.timed_samples, refinements=refinements,
            prolongation=mode, coefficients=data["coefficients"], richardson=not args.no_richardson)
        timings[mode] = time.perf_counter() - started
    exact_map = pp.linear_comparator_point(config, data["u0"], data["uT"], lattice, timed_samples=args.timed_samples)
    budget = _prolongation_error_budget(config, data, lattice, factory, refinements)

    first = args.prolongations[0]
    registered = [p for p in curves[first] if p.n_solve == args.eval_n and p.scheme == "cn"]
    tag = "k" + "".join(map(str, refinements))
    candidates = {}
    for mode, points in curves.items():
        candidates[f"{mode}_{tag}"] = [p for p in points if p.scheme == "cn"]
        if not args.no_richardson:
            candidates[f"{mode}_{tag}_richardson"] = list(points)
    frontiers = {"registered_k1": _frontier(registered)}
    frontiers.update({name: _frontier(pts) for name, pts in candidates.items()})
    floors = {name: min(f["mean_rel_l2"] for f in front) for name, front in frontiers.items()}
    gains = {name: floors["registered_k1"] / floor for name, floor in floors.items() if name != "registered_k1"}
    maps = {name: _decision_map(registered, pts) for name, pts in candidates.items()}
    steps64 = {mode: {f"{p.scheme}:{p.n_solve}": p.mean_rel_l2 for p in points if p.steps == 64}
               for mode, points in curves.items()}

    checks: dict = {"exact_map_reproduces_truth": exact_map.mean_rel_l2 <= 1e-12,
                    "exact_initial_condition_is_u0_on_nested_nodes": nested_ulps <= pp._REFINED_INPUT_ULPS}
    k1_blocks = [[p.to_dict() for p in points if p.n_solve == args.eval_n and p.scheme == "cn"]
                 for points in curves.values()]
    checks["k1_block_identical_errors_under_every_prolongation"] = all(
        [b["mean_rel_l2"] for b in block] == [b["mean_rel_l2"] for b in k1_blocks[0]]
        and [b["cg_iterations_mean"] for b in block] == [b["cg_iterations_mean"] for b in k1_blocks[0]]
        for block in k1_blocks[1:])
    if "exact" in curves:
        levels = [k * args.eval_n for k in refinements]
        checks["oracle_second_order_in_h_at_steps64"] = all(
            3.0 < steps64["exact"][f"cn:{a}"] / steps64["exact"][f"cn:{b}"] < 5.0 for a, b in zip(levels, levels[1:]))
    order = budget["order_ratio_N_to_2N_over_2N_to_4N"]
    checks["linear_prolongation_second_order_on_the_inputs"] = 3.5 < order["min"] and order["max"] < 4.5
    return {
        "domain": domain, "n": args.eval_n, "num_nodes": lattice.num_nodes,
        "sample_ids": [ids[0], ids[-1]], "sample_count": len(ids), "timed_samples": args.timed_samples,
        "dataset_sha256": data["manifest"]["dataset_sha256"],
        "curve_wall_time_s": timings,
        "points": {mode: [p.to_dict() for p in points] for mode, points in curves.items()},
        "exact_linear_map": exact_map.to_dict(),
        "exact_initial_condition_max_deviation_ulps": nested_ulps,
        "linear_prolongation_error_budget": budget,
        "error_at_steps64": steps64,
        "frontiers": frontiers,
        "floors": floors,
        "floor_gain_over_registered": gains,
        "decision_maps_vs_registered": {
            "hypothetical": "no model: latencies 1e-4..1 s and upper IQM bounds 1e-5..10^-0.5 on a log grid, step "
                            "0.1 decade; the preregistered rules of section 6 (phase1_protocol _rules); "
                            "'h0_to_advantage' counts cells undecidable or H0 under the registered curve that the "
                            "candidate decides for the model, 'advantage_to_h0' the reverse",
            **maps},
        "checks": checks,
    }


def build_report(args: argparse.Namespace) -> dict:
    if args.samples < 1 or args.timed_samples < 1 or args.timed_samples > args.samples:
        raise SystemExit("need 1 <= --timed-samples <= --samples")
    refinements = tuple(pp.PROPOSED_REFINEMENTS if args.refinements is None else args.refinements)
    for mode in args.prolongations:
        if mode not in pp._PROLONGATIONS:
            raise SystemExit(f"--prolongations: unknown mode {mode!r}")
    ids = range(args.first_id, args.first_id + args.samples)
    if any(i in r for i in ids for r in _REGISTERED_RANGES):
        raise SystemExit("pilot ids overlap a registered id range of the preregistration")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    cases = [_case(domain, args, refinements) for domain in args.domains]
    passed = all(all(c["checks"].values()) for c in cases)
    return {
        "schema": SCHEMA,
        "claim_boundary": ("instrument measurement on pilot ids outside the registered id ranges; not a Phase 1 "
                           "result; no model fitted; wall times are this machine's, not the D5 machine's; "
                           "decision support for the owner amendments D-T1-1 / D-T1-2 (ADR-003). 'exact' is an "
                           "oracle initial condition the model never sees; 'linear' is the fair prolongation"),
        "source": _source_revision(),
        "environment": {
            "python": platform.python_version(), "numpy": numpy.__version__, "torch": torch.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(), "threads_pinned": args.threads,
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                         "MKL_NUM_THREADS")},
        },
        "task_config": {d: HeatOperatorTaskConfig(domain=d).canonical() for d in args.domains},
        "timing_protocol": pp.TimingProtocol().canonical(),
        "steps_grid": list(pp._DEFAULT_STEPS_GRID),
        "refinements": list(refinements),
        "registered": {"refinements": list(pp._REGISTERED_REFINEMENTS), "prolongation": pp._REGISTERED_PROLONGATION,
                       "scheme": pp._REGISTERED_SCHEME},
        "prolongations": list(args.prolongations),
        "richardson": not args.no_richardson,
        "dataset_seed": DATASET_SEED,
        "cases": cases,
        "passed": passed,
    }


def main(args: argparse.Namespace) -> int:
    report = build_report(args)
    text = json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    for case in report["cases"]:
        gains = ", ".join(f"{name} x{gain:.2f}" for name, gain in sorted(case["floor_gain_over_registered"].items()))
        exact = case["exact_linear_map"]
        print(f"[T1-REFINEMENT] {case['domain']} N={case['n']}: floor k=1 {case['floors']['registered_k1']:.3e}; "
              f"gains: {gains}; prolongation error budget {case['linear_prolongation_error_budget']}; exact map "
              f"{exact['mean_rel_l2']:.1e} at {exact['median_wall_time_s']:.2e} s; checks {case['checks']}",
              file=sys.stderr)
    bound = _report_source_binding(report["source"], "T1-REFINEMENT")
    ok = report["passed"] and bound
    print(f"[T1-REFINEMENT] passed={report['passed']} bound={bound} | {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(_ARGS))
