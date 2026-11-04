"""Independent red/green probes for the optional PR #91 preapproval cost auditor."""

from __future__ import annotations

import copy
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest

import preapproval_budget_hex_v3 as a


TIMES = [
    [140.0510689, 96.8989879, 116.7454365],
    [121.6922723, 82.1184399, 163.6932249],
    [176.7709585, 190.5805331, 153.9892088],
    [143.6006271, 143.3071911, 144.6444171],
    [155.2208342, 152.1394185, 141.6742947],
    [132.7519261, 133.0927062, 134.4273592],
    [292.0969895, 314.4684043, 334.9752801],
    [331.2872989, 332.1334125, 329.2665466],
]
VALS = [
    [2.0875956, 2.5743917, 1.6786124],
    [2.2509176, 2.2806079, 1.9592339],
    [2.2119669, 2.072528, 2.0418393],
    [2.0806183, 2.1285641, 2.1642533],
    [1.2937631, 1.2804657, 1.2958012],
    [1.3407439, 1.3264872, 1.3049701],
    [2.7918853, 2.7095247, 2.733618],
    [2.6872382, 2.7619987, 2.7137446],
]


def evidence() -> dict:
    rows = []
    for i, (times, val) in enumerate(zip(TIMES, VALS)):
        rows.append(
            {
                "hidden_channels": 32 if i < 4 else 64,
                "latent_resolution": 16 if i % 4 < 2 else 24,
                "n_modes": 6 if i % 4 < 2 else 9,
                "learning_rate": 0.001 if i % 2 == 0 else 0.0003,
                "radius_factor": 1.5,
                "epoch_seconds": list(times),
                "epoch_seconds_median": sorted(times)[1],
                "val_pass_seconds": list(val),
                "val_pass_seconds_median": sorted(val)[1],
            }
        )
    return {
        "source": {"bound": True, "worktree_dirty": False, "git_sha": "35f82dcff7821f41073537fdf514ec9ff3987774"},
        "passed": True,
        "data": {"kind": "synthetic standard-normal tensors, no T1 data"},
        "hardware": {"cpu_name": "test fixture"},
        "environment": {"torch": "2.6.0+cpu"},
        "protocol": {"measured_epochs": 3, "statistic": "median"},
        "configs": rows,
    }


class TestPreapprovalBudget(unittest.TestCase):
    def test_historical_cost_numbers_from_transcribed_evidence(self):
        r = a.audit_cost(evidence())
        self.assertAlmostEqual(r["estimated_median_hours"], 43.70609893, places=6)
        self.assertGreater(r["observed_sample_max_scenario_hours"], 45.4334)
        self.assertLess(r["observed_sample_max_scenario_hours"], 45.435)
        self.assertEqual(r["release_verdict"], "HOLD_OWNER_APPROVAL_REQUIRED")
        self.assertEqual(r["research_result"], "NONE_NO_T1_MODEL_FIT")

    def test_overhead_crosses_budget(self):
        r = a.audit_cost(evidence())
        self.assertLess(r["stress_hours_median_base"]["1.00x"], 48)
        self.assertGreater(r["stress_hours_median_base"]["1.10x"], 48)
        self.assertGreater(r["stress_hours_sample_max_base"]["1.10x"], 48)
        self.assertGreater(r["median_break_even_train_slowdown_factor"], 1.0)
        self.assertLess(r["median_break_even_train_slowdown_factor"], 1.10)

    def test_more_final_seeds_cannot_reduce_cost(self):
        x = evidence()
        self.assertGreater(
            a.audit_cost(x, final_seeds=6)["estimated_median_hours"],
            a.audit_cost(x, final_seeds=5)["estimated_median_hours"],
        )

    def test_more_epochs_cannot_reduce_cost(self):
        x = evidence()
        self.assertGreater(
            a.audit_cost(x, epochs=51)["estimated_median_hours"], a.audit_cost(x, epochs=50)["estimated_median_hours"]
        )

    def test_dirty_tree_rejected(self):
        x = evidence()
        x["source"]["worktree_dirty"] = True
        with self.assertRaisesRegex(a.EvidenceError, "source must be bound"):
            a.audit_cost(x)

    def test_unbound_source_rejected(self):
        x = evidence()
        x["source"]["bound"] = False
        with self.assertRaises(a.EvidenceError):
            a.audit_cost(x)

    def test_malformed_git_sha_rejected(self):
        x = evidence()
        x["source"]["git_sha"] = "35f82dc"
        with self.assertRaises(a.EvidenceError):
            a.audit_cost(x)

    def test_false_pass_rejected(self):
        x = evidence()
        x["passed"] = False
        with self.assertRaises(a.EvidenceError):
            a.audit_cost(x)

    def test_t1_measurement_rejected(self):
        x = evidence()
        x["data"]["kind"] = "T1 training sample"
        with self.assertRaises(a.EvidenceError):
            a.audit_cost(x)

    def test_stale_median_rejected(self):
        x = evidence()
        x["configs"][0]["epoch_seconds_median"] += 5
        with self.assertRaisesRegex(a.EvidenceError, "median"):
            a.audit_cost(x)

    def test_duplicate_grid_rejected(self):
        x = evidence()
        x["configs"][1] = copy.deepcopy(x["configs"][0])
        with self.assertRaisesRegex(a.EvidenceError, "duplicate configuration"):
            a.audit_cost(x)

    def test_wrong_sample_count_rejected(self):
        x = evidence()
        x["configs"][1]["epoch_seconds"].pop()
        with self.assertRaisesRegex(a.EvidenceError, "wrong epoch sample count"):
            a.audit_cost(x)

    def test_nonfinite_epoch_rejected(self):
        x = evidence()
        x["configs"][0]["epoch_seconds"][0] = math.inf
        with self.assertRaises(a.EvidenceError):
            a.audit_cost(x)

    def test_missing_validation_times_rejected(self):
        x = evidence()
        x["configs"][0].pop("val_pass_seconds")
        with self.assertRaisesRegex(a.EvidenceError, "validation sample count"):
            a.audit_cost(x)

    def test_stale_validation_median_rejected(self):
        x = evidence()
        x["configs"][0]["val_pass_seconds_median"] += 1.0
        with self.assertRaisesRegex(a.EvidenceError, "validation median"):
            a.audit_cost(x)

    def test_bool_numeric_rejected(self):
        x = evidence()
        x["configs"][0]["val_pass_seconds_median"] = True
        with self.assertRaises(a.EvidenceError):
            a.audit_cost(x)

    def test_zero_and_negative_epochs_rejected(self):
        for n in (0, -2, True, 5.2):
            with self.subTest(n=n), self.assertRaises(a.EvidenceError):
                a.audit_cost(evidence(), epochs=n)

    def test_nan_budget_rejected(self):
        with self.assertRaises(a.EvidenceError):
            a.audit_cost(evidence(), budget_hours=float("nan"))

    def test_strict_json_duplicate_key(self):
        with tempfile.TemporaryDirectory() as folder:
            p = pathlib.Path(folder) / "bad.json"
            p.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(a.EvidenceError, "duplicate JSON key"):
                a.read_evidence(p)

    def test_strict_json_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            p = pathlib.Path(folder) / "bad.json"
            p.write_text('{"a": NaN}', encoding="utf-8")
            with self.assertRaises(a.EvidenceError):
                a.read_evidence(p)

    def test_byte_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            p = pathlib.Path(folder) / "oversize.json"
            p.write_bytes(b" " * (a.MAX_SOURCE_BYTES + 1))
            with self.assertRaisesRegex(a.EvidenceError, "byte budget"):
                a.read_evidence(p)

    def test_cli_fixture_mode_explicit_and_non_authoritative(self):
        with tempfile.TemporaryDirectory() as folder:
            p = pathlib.Path(folder) / "fixture.json"
            p.write_text(json.dumps(evidence(), allow_nan=False), encoding="utf-8")
            cmd = [sys.executable, str(pathlib.Path(a.__file__)), str(p), "--fixture-only"]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            output = json.loads(run.stdout)
            self.assertEqual(output["release_verdict"], "HOLD_UNPINNED_TEST_FIXTURE")
            self.assertEqual(output["evidence_tier"], "TEST_FIXTURE_NOT_ORIGINAL_NO_RELEASE_EVIDENCE")
            self.assertEqual(len(output["input_file_sha256"]), 64)

    def test_cli_defaults_to_pinned_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            p = pathlib.Path(folder) / "fixture.json"
            p.write_text(json.dumps(evidence(), allow_nan=False), encoding="utf-8")
            cmd = [sys.executable, str(pathlib.Path(a.__file__)), str(p)]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertEqual(json.loads(run.stdout)["release_verdict"], "HOLD_INVALID_EVIDENCE")

    def test_manifest_digest_from_actual_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            p = pathlib.Path(folder) / "good.json"
            p.write_text(json.dumps(evidence(), sort_keys=True), encoding="utf-8")
            obj, digest = a.read_evidence(p)
            self.assertEqual(a.audit_cost(obj)["config_count"], 8)
            self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
