"""
test_preregistration_candidate_hex_v3.py
========================================
Binds the 2026-10-29 approval-candidate paragraph of
spec/PREREGISTRATION_PHASE1_v1.md (section 8) to the evidence it cites:

- the radius table to ``margin_over_float32_error`` of the committed survey
  results/2026-10-20_gino_radius_tie_survey_float32-graphs.json;
- the D4 grid, the epoch seconds and the CPU budget to the cost measurement
  results/2026-10-29_gino_epoch_cost_synthetic.json;
- the D5 machine table to the machine recorded in that measurement;
- the status line to DRAFT (the candidate is not an approval).

Every number in the paragraph is recomputed from the JSON in the format the
paragraph prints it; a changed number fails. A missing row fails by count,
not by silence. Reads files only; no model, no T1 data.
"""

from __future__ import annotations

import json
import pathlib
import re
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PREREG = _ROOT / "spec" / "PREREGISTRATION_PHASE1_v1.md"
_SURVEY = _ROOT / "results" / "2026-10-20_gino_radius_tie_survey_float32-graphs.json"
_COST = _ROOT / "results" / "2026-10-29_gino_epoch_cost_synthetic.json"
_MARKER = "2026-10-29 (approval candidate, no decision)"
_LATENTS = (16, 24)
_RADIUS_FACTORS = (1.5, 2.25, 2.5, 2.75)
_FINAL_SEEDS = 5


def _paragraph() -> str:
    text = _PREREG.read_text(encoding="utf-8").replace("\r\n", "\n")
    start = text.find(_MARKER)
    if start < 0:
        raise AssertionError(f"paragraph {_MARKER!r} missing in {_PREREG.name}")
    end = text.find("\n## 9.", start)
    if end < 0:
        raise AssertionError("section 9 heading missing after the candidate paragraph")
    return text[start:end]


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _margins() -> dict:
    out = {}
    for c in _load(_SURVEY)["survey"]["flip_free_candidates"]:
        out[(int(c["latent_resolution"]), float(c["radius_factor"]))] = float(c["margin_over_float32_error"])
    return out


def _budget_hours(cost: dict, epochs: int) -> float:
    """Selection (every configuration, seed 0) + final (5 seeds of the most expensive one), plus one
    validation pass per run; the selected configuration is unknown in advance, so the worst case."""
    configs = cost["configs"]
    train = [c["epoch_seconds_median"] for c in configs]
    val = [c["val_pass_seconds_median"] for c in configs]
    seconds = epochs * (sum(train) + _FINAL_SEEDS * max(train)) + sum(val) + _FINAL_SEEDS * max(val)
    return seconds / 3600.0


class TestApprovalCandidate20261029(unittest.TestCase):
    def test_status_line_stays_draft(self):
        lines = _PREREG.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[2], "**Status:** DRAFT — NOT binding.")

    def test_radius_table_matches_the_survey(self):
        margins = _margins()
        rows = re.findall(r"^\| `(\d+(?:\.\d+)?)` \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$",
                          _paragraph(), flags=re.M)
        self.assertEqual(sorted(float(r[0]) for r in rows), sorted(_RADIUS_FACTORS))
        for factor, m16, m24, low in rows:
            factor = float(factor)
            with self.subTest(radius_factor=factor):
                expected = [margins[(latent, factor)] for latent in _LATENTS]
                self.assertEqual([m16, m24, low], [f"{expected[0]:.2f}", f"{expected[1]:.2f}",
                                                   f"{min(expected):.2f}"])

    def test_proposed_radius_has_the_largest_minimum_margin(self):
        margins = _margins()
        best = max(_RADIUS_FACTORS, key=lambda f: min(margins[(latent, f)] for latent in _LATENTS))
        proposed = re.findall(r"Proposed: `radius_factor = (\d+(?:\.\d+)?)`", " ".join(_paragraph().split()))
        self.assertEqual([float(p) for p in proposed], [best])
        cost = _load(_COST)
        self.assertEqual({c["radius_factor"] for c in cost["configs"]}, {best})

    def test_grid_table_matches_the_cost_measurement(self):
        cost = _load(_COST)
        rows = re.findall(r"^\| G(\d) \| (\d+) \| (\d+) \| (\d+) \| ([0-9.e-]+) \| ([0-9.]+) \| ([0-9.]+) \|$",
                          _paragraph(), flags=re.M)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(cost["configs"]), 8)
        self.assertEqual([int(r[0]) for r in rows], list(range(1, 9)))
        for row, config in zip(rows, cost["configs"]):
            _, width, latent, modes, lr, epoch, val = row
            with self.subTest(config=row[0]):
                self.assertEqual((int(width), int(latent), int(modes), float(lr)),
                                 (config["hidden_channels"], config["latent_resolution"],
                                  config["n_modes"], config["learning_rate"]))
                self.assertEqual(epoch, f"{config['epoch_seconds_median']:.1f}")
                self.assertEqual(val, f"{config['val_pass_seconds_median']:.1f}")

    def test_epoch_budget_and_cpu_hours_match_the_measurement(self):
        cost = _load(_COST)
        paragraph = " ".join(_paragraph().split())
        epochs = re.findall(r"`E = (\d+)` epochs per configuration", paragraph)
        self.assertEqual(len(epochs), 1)
        hours = re.findall(r"worst case \*\*([0-9.]+) h\*\*", paragraph)
        self.assertEqual(hours, [f"{_budget_hours(cost, int(epochs[0])):.1f}"])
        budget = re.findall(r"within the proposed D5 budget of \*\*(\d+) h\*\*", paragraph)
        self.assertEqual(len(budget), 1)
        self.assertLessEqual(_budget_hours(cost, int(epochs[0])), float(budget[0]))
        fixed = re.findall(r"batch size `(\d+)`, `(\d+)` training samples", paragraph)
        self.assertEqual(fixed, [(str(cost["data"]["batch_size"]), str(cost["data"]["n_train"]))])
        # Self-check (same class one level up): the prose numbers that come from the JSON are bound too.
        protocol = re.findall(r"median of (\d+) timed epochs after (\d+) untimed warm-up steps; validation "
                              r"pass = (\d+) samples", paragraph)
        self.assertEqual(protocol, [(str(cost["protocol"]["measured_epochs"]), str(cost["protocol"]["warmup_steps"]),
                                     str(cost["data"]["n_val"]))])
        modes = sorted({(c["latent_resolution"], c["n_modes"]) for c in cost["configs"]})
        self.assertEqual(re.findall(r"\((\d+) and (\d+); the default 12 is refused", paragraph),
                         [tuple(str(m) for _, m in modes)])

    def test_machine_table_matches_the_measurement(self):
        cost = _load(_COST)
        hw, env = cost["hardware"], cost["environment"]
        rows = dict(re.findall(r"^\| D5 ([a-z_]+) \| (.+?) \|$", _paragraph(), flags=re.M))
        expected = {
            "cpu": hw["cpu_name"],
            "cores": f"{hw['physical_cores']} physical, {hw['logical_processors']} logical",
            "ram": f"{hw['ram_bytes'] / 2**30:.2f} GiB ({hw['ram_bytes']} bytes)",
            "os": hw["os"],
            "python": env["python"],
            "torch": env["torch"],
            "neuraloperator": env["neuraloperator"],
            "numpy": env["numpy"],
            "threads": f"{env['torch_num_threads']} (torch.set_num_threads)",
            "gpu": "none (CUDA available: False)" if not env["cuda_available"] else "CUDA available",
        }
        self.assertEqual(rows, expected)

    def test_float_budget_verdicts_match_the_protocol_instruments(self):
        import math

        import phase1_protocol_hex_v3 as pp

        best = float(re.findall(r"Proposed: `radius_factor = (\d+(?:\.\d+)?)`", _paragraph())[0])
        survey = {(int(c["latent_resolution"]), float(c["radius_factor"])): c
                  for c in _load(_SURVEY)["survey"]["flip_free_candidates"]}
        # spec/GINO_ADAPTER_PHASE1_v2.md amendment 2026-10-21: R = sqrt(7/4), input error 4 u R.
        norm = math.sqrt(7 / 4)
        u = pp.FLOAT32_UNIT_ROUNDOFF
        ratios, independent, measured = [], set(), set()
        for latent in _LATENTS:
            c = survey[(latent, best)]
            budget = pp.cdist_rounding_margin(best / 32, max_point_norm=norm, input_abs_error=4 * u * norm)
            ratios.append(f"{c['min_relative_margin'] / budget:.2f}")
            independent.add(pp.radius_graph_stability(c["min_relative_margin"], budget_relative=budget,
                                                      frozen_order_flips=0).status)
            measured.add(pp.radius_graph_stability(c["min_relative_margin"],
                                                   budget_relative=c["float32_error_near_radius"],
                                                   frozen_order_flips=0).status)
        paragraph = " ".join(_paragraph().split())
        found = re.findall(r"both pairs are `(\w+)` \(margin over budget ([0-9.]+) for latent 16 and "
                           r"([0-9.]+) for latent 24\)", paragraph)
        self.assertEqual(len(independent), 1)
        self.assertEqual(found, [(independent.pop(), *ratios)])
        found = re.findall(r"frozen_order_flips = 0\)` both pairs are `(\w+)`", paragraph)
        self.assertEqual(len(measured), 1)
        self.assertEqual(found, [measured.pop()])

    def test_neighbourhood_load_and_runner_up_numbers(self):
        cost = _load(_COST)
        margins = _margins()
        mins = {f: min(margins[(latent, f)] for latent in _LATENTS) for f in _RADIUS_FACTORS}
        best = max(_RADIUS_FACTORS, key=mins.get)
        runner = max((f for f in _RADIUS_FACTORS if f != best), key=mins.get)
        neighbours = {(r["latent_resolution"], r["radius_factor"]): r["min_latent_neighbours_per_node"]
                      for r in cost["neighbourhoods"]}
        paragraph = " ".join(_paragraph().split())
        self.assertEqual(re.findall(r"For the owner: `([0-9.]+)` is close \(([0-9.]+)\)", paragraph),
                         [(f"{runner:g}", f"{mins[runner]:.2f}")])
        self.assertEqual(re.findall(r"latent 16 some node has only (\d+) latent point\(s\) within the radius "
                                    r"\(`([0-9.]+)`: (\d+);", paragraph),
                         [(str(neighbours[(16, best)]), f"{runner:g}", str(neighbours[(16, runner)]))])
        self.assertEqual(re.findall(r"largest minimum ([0-9.]+)\)", paragraph), [f"{mins[best]:.2f}"])
        load = cost["machine_load_percent"]
        self.assertEqual(re.findall(r"load was (\d+) % at the start and (\d+) % at the end", paragraph),
                         [(str(load["start"]), str(load["end"]))])
        every = [s for c in cost["configs"] for s in c["epoch_seconds"]]
        self.assertEqual(re.findall(r"single timed epochs ranged from ([0-9.]+) to ([0-9.]+) s", paragraph),
                         [(f"{min(every):.1f}", f"{max(every):.1f}")])

    def test_cited_measurement_is_bound_synthetic_and_passed(self):
        cost = _load(_COST)
        self.assertIs(cost["source"]["bound"], True)
        self.assertIs(cost["passed"], True)
        self.assertIn("no T1 data", cost["data"]["kind"])
        self.assertEqual(cost["environment"]["neuraloperator"], "2.0.0")
        self.assertEqual(cost["protocol"]["measured_epochs"], 3)
        paragraph = " ".join(_paragraph().split())
        self.assertIn(f"source `{cost['source']['git_sha'][:7]}`", paragraph)
        self.assertIn("`" + " ".join(["python", *cost["command"][1:]]) + "`", paragraph)


if __name__ == "__main__":
    unittest.main()
