"""PR #91 pinned candidate contract and adversarial red/green probes.

All tests use explicitly fabricated metadata or transcribed fixture values; they
never assert an official GitHub measurement verification. No PyTorch dependency.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import preapproval_budget_hex_v3 as audit
from test_preapproval_budget_hex_v3 import evidence


def candidate_like() -> dict:
    x = evidence()
    x["source"]["github_sha"] = None
    x["source"]["git_sha"] = audit.PR91_MEASUREMENT_TREE
    x["environment"] = {
        "python": "3.12.10",
        "torch": "2.6.0+cpu",
        "numpy": "2.5.3",
        "neuraloperator": "2.0.0",
        "dtype": "float32",
        "amp": False,
        "cuda_available": False,
        "torch_num_threads": 4,
    }
    x["hardware"] = {
        "cpu_name": "Intel(R) Core(TM) i5-1035G1 CPU @ 1.00GHz",
        "physical_cores": 4,
        "logical_processors": 8,
        "ram_bytes": 8260374528,
        "os": "Microsoft Windows 11 Home 10.0.26200 build 26200",
    }
    x["data"].update(
        {
            "geometry": "triangle_hex_lattice(32), 465 nodes",
            "batch_size": 20,
            "n_train": 1800,
            "n_val": 200,
            "steps_per_epoch": 90,
            "data_seed": 20261029,
            "shuffle_seed": 0,
            "model_seed": 0,
        }
    )
    x["protocol"].update(
        {
            "warmup_steps": 10,
            "warmup_val_passes": 1,
            "clock": "time.perf_counter",
            "optimizer": "Adam",
            "schedule": "CosineAnnealingLR",
            "loss": "relative L2 (mean over batch)",
        }
    )
    x["command"] = [
        "python.exe",
        "scripts/oracles/gino_epoch_cost_synthetic.py",
        "--threads",
        "4",
        "--output",
        "results/2026-10-29_gino_epoch_cost_synthetic.json",
    ]
    digests = {}
    for cfg in x["configs"]:
        cfg.update(
            {
                "n_layers": 4,
                "n_parameters": 1024,
                "warmup_steps_seconds": 2.1,
                "warmup_val_pass_seconds": 1.2,
                "losses_finite": True,
                "last_batch_loss_per_epoch": [1.0, 0.9, 0.8],
            }
        )
        pair = (cfg["hidden_channels"], cfg["latent_resolution"])
        if pair not in digests:
            digests[pair] = f"{len(digests) + 1:064x}"
        cfg["config_digest"] = digests[pair]
    medians = [statistics.median(c["epoch_seconds"]) for c in x["configs"]]
    x["summary"] = {
        "epoch_seconds_median_min": min(medians),
        "epoch_seconds_median_max": max(medians),
        "epoch_seconds_median_sum": sum(medians),
    }
    return x


def survey_like() -> dict:
    rows = []
    ratios = {
        (16, 1.5): 3.53,
        (24, 1.5): 1.14,
        (16, 2.25): 2.10,
        (24, 2.25): 0.82,
        (16, 2.5): 1.11,
        (24, 2.5): 1.32,
        (16, 2.75): 2.20,
        (24, 2.75): 0.80,
        (48, 2.75): 0.32,
    }
    for (latent, factor), ratio in ratios.items():
        rows.append(
            {
                "latent_resolution": latent,
                "radius_factor": factor,
                "margin_over_float32_error": ratio,
                "float32_error_near_radius": 0.0001,
                "min_relative_margin": ratio * 0.0001,
            }
        )
    return {
        "source": {"bound": True, "git_sha": audit.PR91_SURVEY_TREE, "worktree_dirty": False},
        "environment": {"neuraloperator": "2.0.0", "numpy": "2.5.3", "python": "3.12.10", "torch": "2.6.0+cpu"},
        "survey": {
            "passed": True,
            "positive_control": {"flip_detector_saw_a_flip": True, "default_flips_total": 16},
            "flip_free_candidates": rows,
        },
    }


class TestPR91Contract(unittest.TestCase):
    def assert_invalid(
        self, modify, pattern="PR91|invalid|mismatch|incomplete|finite|digest changed|integer in|duplicate"
    ):
        x = candidate_like()
        modify(x)
        with self.assertRaisesRegex(audit.EvidenceError, pattern):
            audit.verify_pr91_contract(x)

    def test_contract_proposal_fixture_passes_but_is_not_source(self):
        r = audit.verify_pr91_contract(candidate_like())
        self.assertEqual(r["configuration_count"], 8)
        self.assertFalse(r["research_data"])
        self.assertFalse(r["checkpoint_io_budgeted"])
        self.assertNotEqual(r["measurement_scheduler_t_max"], r["planned_scheduler_t_max"])

    def test_original_sha256_constant_and_blob_pin_are_full_length(self):
        self.assertEqual(len(audit.PR91_EVIDENCE_SHA256), 64)
        self.assertEqual(len(audit.PR91_EVIDENCE_BLOB), 40)
        self.assertEqual(len(audit.PR91_HEAD), 40)

    def test_non_original_fixture_denied_by_both_pins(self):
        b = json.dumps(candidate_like(), sort_keys=True).encode()
        with self.assertRaisesRegex(audit.EvidenceError, "pinned commit"):
            audit.verify_pr91_bytes(b)

    def test_single_byte_flip_is_rejected(self):
        source = b'{"x":123}'
        for mutation in (source, source[:-1] + b' ', source + b' '):
            with self.subTest(mutation=mutation), self.assertRaises(audit.EvidenceError):
                audit.verify_pr91_bytes(mutation)

    def test_oversized_evidence_bytes_rejected(self):
        with self.assertRaisesRegex(audit.EvidenceError, "byte stream"):
            audit.verify_pr91_bytes(b' ' * (audit.MAX_SOURCE_BYTES + 1))

    def test_source_revision_forgery(self):
        self.assert_invalid(lambda x: x["source"].__setitem__("git_sha", "a" * 40))

    def test_unregistered_github_sha(self):
        self.assert_invalid(lambda x: x["source"].__setitem__("github_sha", "b" * 40))

    def test_wrong_config_width(self):
        self.assert_invalid(lambda x: x["configs"][0].__setitem__("hidden_channels", 48))

    def test_wrong_learning_rate(self):
        self.assert_invalid(lambda x: x["configs"][0].__setitem__("learning_rate", 0.01))

    def test_grid_missing_configuration(self):
        self.assert_invalid(lambda x: x["configs"].pop(), "eight")

    def test_grid_duplicate_configuration(self):
        self.assert_invalid(lambda x: x["configs"].__setitem__(1, copy.deepcopy(x["configs"][0])))

    def test_layer_mismatch(self):
        self.assert_invalid(lambda x: x["configs"][0].__setitem__("n_layers", 1))

    def test_radius_mismatch(self):
        self.assert_invalid(lambda x: x["configs"][0].__setitem__("radius_factor", 2.5))

    def test_n_modes_mismatch(self):
        self.assert_invalid(lambda x: x["configs"][0].__setitem__("n_modes", 7))

    def test_config_digest_malformed(self):
        self.assert_invalid(lambda x: x["configs"][0].__setitem__("config_digest", "x" * 64))

    def test_model_digest_differs_by_lr(self):
        self.assert_invalid(lambda x: x["configs"][1].__setitem__("config_digest", "0" * 64))

    def test_zero_parameter_count(self):
        self.assert_invalid(lambda x: x["configs"][0].__setitem__("n_parameters", 0))

    def test_wrong_cpu_is_rejected(self):
        self.assert_invalid(lambda x: x["hardware"].__setitem__("cpu_name", "different"))

    def test_wrong_memory_is_rejected(self):
        self.assert_invalid(lambda x: x["hardware"].__setitem__("ram_bytes", 100))

    def test_torch_version_is_pinned(self):
        self.assert_invalid(lambda x: x["environment"].__setitem__("torch", "2.7.0"))

    def test_cuda_is_not_silently_enabled(self):
        self.assert_invalid(lambda x: x["environment"].__setitem__("cuda_available", True))

    def test_amp_is_not_silently_enabled(self):
        self.assert_invalid(lambda x: x["environment"].__setitem__("amp", True))

    def test_wrong_training_samples(self):
        self.assert_invalid(lambda x: x["data"].__setitem__("n_train", 1000))

    def test_wrong_batch_size(self):
        self.assert_invalid(lambda x: x["data"].__setitem__("batch_size", 40))

    def test_wrong_geometry(self):
        self.assert_invalid(lambda x: x["data"].__setitem__("geometry", "torus"))

    def test_wrong_optimizer(self):
        self.assert_invalid(lambda x: x["protocol"].__setitem__("optimizer", "SGD"))

    def test_missing_scheduler(self):
        self.assert_invalid(lambda x: x["protocol"].__setitem__("schedule", "none"))

    def test_wrong_clock(self):
        self.assert_invalid(lambda x: x["protocol"].__setitem__("clock", "time.time"))

    def test_extra_sampled_epoch(self):
        self.assert_invalid(lambda x: x["protocol"].__setitem__("measured_epochs", 4))

    def test_missing_loss_series(self):
        self.assert_invalid(lambda x: x["configs"][0].__setitem__("last_batch_loss_per_epoch", []))

    def test_nonfinite_loss_rejected(self):
        self.assert_invalid(lambda x: x["configs"][0]["last_batch_loss_per_epoch"].__setitem__(1, math.inf))

    def test_false_losses_finite_rejected(self):
        self.assert_invalid(lambda x: x["configs"][0].__setitem__("losses_finite", False))

    def test_summary_mismatch_rejected(self):
        self.assert_invalid(lambda x: x["summary"].__setitem__("epoch_seconds_median_sum", 0))

    def test_summary_max_mismatch_rejected(self):
        self.assert_invalid(lambda x: x["summary"].__setitem__("epoch_seconds_median_max", 0))

    def test_forged_median_pass_flag_still_rejected(self):
        self.assert_invalid(lambda x: x["configs"][0].__setitem__("epoch_seconds_median", 999), "median")

    def test_recursion_bomb_fails_with_evidence_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "bomb.json"
            p.write_bytes(b'[' * 2200 + b'0' + b']' * 2200)
            with self.assertRaises(audit.EvidenceError):
                audit.read_evidence(p)

    def test_cli_fixture_is_explicit_not_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "fake.json"
            p.write_text(json.dumps(candidate_like(), allow_nan=False), encoding="utf-8")
            cmd = [sys.executable, str(pathlib.Path(audit.__file__)), str(p)]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=20)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["release_verdict"], "HOLD_INVALID_EVIDENCE")
            proc2 = subprocess.run(cmd + ["--fixture-only"], capture_output=True, text=True, check=False, timeout=20)
            self.assertEqual(proc2.returncode, 0)
            self.assertEqual(json.loads(proc2.stdout)["release_verdict"], "HOLD_UNPINNED_TEST_FIXTURE")

    def test_extreme_budget_is_controlled_rejection(self):
        with self.assertRaisesRegex(audit.EvidenceError, "overflow"):
            audit.audit_cost(candidate_like(), budget_hours=1e308)

    def test_valid_json_decoded_from_one_byte_snapshot(self):
        raw = json.dumps(candidate_like(), allow_nan=False).encode("utf-8")
        self.assertEqual(audit._decode_evidence(raw), candidate_like())

    def test_radius_survey_contract_fixture(self):
        r = audit.verify_radius_survey(survey_like())
        self.assertEqual(r["selected_radius_factor"], 1.5)
        self.assertFalse(r["cross_hardware_claim"])

    def test_radius_survey_must_have_positive_control(self):
        d = survey_like()
        d["survey"]["positive_control"]["flip_detector_saw_a_flip"] = False
        with self.assertRaises(audit.EvidenceError):
            audit.verify_radius_survey(d)

    def test_radius_survey_rejects_margin_forgery(self):
        d = survey_like()
        d["survey"]["flip_free_candidates"][0]["margin_over_float32_error"] = 6
        with self.assertRaisesRegex(audit.EvidenceError, "ratio mismatch"):
            audit.verify_radius_survey(d)

    def test_radius_survey_rejects_unstable_pair(self):
        d = survey_like()
        d["survey"]["flip_free_candidates"][1]["margin_over_float32_error"] = 0.8
        d["survey"]["flip_free_candidates"][1]["min_relative_margin"] = 0.00008
        with self.assertRaisesRegex(audit.EvidenceError, "too close"):
            audit.verify_radius_survey(d)

    def test_radius_survey_rejects_missing_pair(self):
        d = survey_like()
        d["survey"]["flip_free_candidates"][1]["latent_resolution"] = 22
        with self.assertRaisesRegex(audit.EvidenceError, "absent"):
            audit.verify_radius_survey(d)

    def test_radius_survey_rejects_changed_best_radius(self):
        d = survey_like()
        for c in d["survey"]["flip_free_candidates"]:
            if c["radius_factor"] == 2.5:
                c["margin_over_float32_error"] = 5.0
                c["min_relative_margin"] = 0.0005
        with self.assertRaisesRegex(audit.EvidenceError, "highest minimum"):
            audit.verify_radius_survey(d)

    def test_strict_preflight_end_to_end_with_rebound_test_pins_only(self):
        """Exercise full strict logic using fixture-owned local hashes (NOT GitHub evidence)."""
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            cost = root / "fake_cost.json"
            survey = root / "fake_survey.json"
            cost.write_text(json.dumps(candidate_like(), sort_keys=True), encoding="utf8")
            survey.write_text(json.dumps(survey_like(), sort_keys=True), encoding="utf8")

            def git_blob(raw):
                return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest()

            x = cost.read_bytes()
            y = survey.read_bytes()
            with (
                mock.patch.object(audit, "PR91_EVIDENCE_SHA256", hashlib.sha256(x).hexdigest()),
                mock.patch.object(audit, "PR91_EVIDENCE_BLOB", git_blob(x)),
                mock.patch.object(audit, "PR91_SURVEY_SHA256", hashlib.sha256(y).hexdigest()),
                mock.patch.object(audit, "PR91_SURVEY_BLOB", git_blob(y)),
            ):
                r = audit.preflight_pr91(cost, survey)
                self.assertEqual(r["release_verdict"], "HOLD_OWNER_APPROVAL_REQUIRED")
                self.assertEqual(r["radius_survey_gate"]["selected_radius_factor"], 1.5)
                self.assertEqual(r["config_count"], 8)

    def test_strict_preflight_rejects_unpinned_radius_survey(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            c = root / "fake_cost.json"
            s = root / "fake_survey.json"
            c.write_text(json.dumps(candidate_like(), sort_keys=True), encoding="utf8")
            s.write_text(json.dumps(survey_like(), sort_keys=True), encoding="utf8")
            x = c.read_bytes()
            with (
                mock.patch.object(audit, "PR91_EVIDENCE_SHA256", hashlib.sha256(x).hexdigest()),
                mock.patch.object(
                    audit, "PR91_EVIDENCE_BLOB", hashlib.sha1(b"blob " + str(len(x)).encode() + b"\x00" + x).hexdigest()
                ),
            ):
                with self.assertRaisesRegex(audit.EvidenceError, "radius survey"):
                    audit.preflight_pr91(c, s)

    def test_cost_has_no_false_confidence_bound(self):
        r = audit.audit_cost(candidate_like())
        self.assertEqual(r["fit_epochs_counted"], 650)
        self.assertEqual(r["validation_passes_counted"], 13)
        self.assertLess(r["median_training_hours"], r["estimated_median_hours"])
        self.assertLess(r["estimated_median_hours"], 48)
        self.assertGreater(r["stress_hours_median_base"]["1.10x"], 48)


if __name__ == "__main__":
    unittest.main()
