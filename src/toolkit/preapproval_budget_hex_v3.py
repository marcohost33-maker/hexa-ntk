"""Pre-approval synthetic GINO cost stress audit (no model training).

Standalone, stdlib-only read-only verifier. This is *not* a PR #91 approval,
a scientific runtime upper bound, or an independent reproduction of the
input timings. Apply against the exact pinned GitHub evidence file in repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics
from typing import Any

POLICY = "PREAPPROVAL_BUDGET_STRESS_v2"
MAX_SOURCE_BYTES = 1024 * 1024
_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class EvidenceError(ValueError):
    """A cost record is invalid or its provenance contract is violated."""


def _mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label}: object required")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise EvidenceError(f"{label}: finite positive number required")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise EvidenceError(f"{label}: unrepresentable number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise EvidenceError(f"{label}: finite positive number required")
    return number


def _integer(value: Any, label: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise EvidenceError(f"{label}: integer in [{low},{high}] required")
    return value


def _duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise EvidenceError(f"nonstandard JSON number: {token}")


def _decode_evidence(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or len(raw) > MAX_SOURCE_BYTES:
        raise EvidenceError("source byte budget exceeded")
    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_duplicate_keys, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceError("not strict UTF-8 JSON") from exc
    return _mapping(parsed, "root")


def read_evidence(path: str | pathlib.Path) -> tuple[dict, str]:
    """Read a bounded JSON file, hashing *the same bytes* that are parsed."""
    with pathlib.Path(path).open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    return _decode_evidence(raw), hashlib.sha256(raw).hexdigest()


def audit_cost(
    data: dict,
    *,
    epochs: int = 50,
    final_seeds: int = 5,
    budget_hours: float = 48.0,
    expected_configs: int = 8,
) -> dict:
    """Recompute selection+final-training budget; no statistical runtime guarantee.

    The selected configuration is unknown: use the maximum observed cost
    among configurations for each one of `final_seeds` final fits.
    Validation costs use the raw per-pass observations: median in the
    nominal scenario and observed maxima in the stress scenario.
    Max-sample scenarios are NOT upper bounds.
    """
    data = _mapping(data, "root")
    epochs = _integer(epochs, "epochs", 1, 10000)
    final_seeds = _integer(final_seeds, "final_seeds", 1, 100)
    expected_configs = _integer(expected_configs, "expected_configs", 1, 64)
    budget_hours = _number(budget_hours, "budget_hours")
    if not math.isfinite(budget_hours * 3600.0):
        raise EvidenceError("budget_hours overflow in seconds")
    source = _mapping(data.get("source"), "source")
    if source.get("bound") is not True or source.get("worktree_dirty") is not False:
        raise EvidenceError("source must be bound and clean")
    sha = source.get("git_sha")
    if not isinstance(sha, str) or not _SHA40.fullmatch(sha):
        raise EvidenceError("source.git_sha must be full lowercase SHA-1")
    if data.get("passed") is not True:
        raise EvidenceError("measurement did not report passed=true")
    kind = _mapping(data.get("data"), "data").get("kind")
    if not isinstance(kind, str) or "synthetic" not in kind.lower() or "no T1 data" not in kind:
        raise EvidenceError("cost record is not explicitly marked synthetic/no T1")
    _mapping(data.get("hardware"), "hardware")
    _mapping(data.get("environment"), "environment")
    protocol = _mapping(data.get("protocol"), "protocol")
    n_observations = _integer(protocol.get("measured_epochs"), "measured_epochs", 1, 100)
    if protocol.get("statistic") != "median":
        raise EvidenceError("protocol must declare median statistic")
    configs = data.get("configs")
    if not isinstance(configs, list) or len(configs) != expected_configs:
        raise EvidenceError("unexpected number of configurations")
    medians: list[float] = []
    maxima: list[float] = []
    validations: list[float] = []
    validation_maxima: list[float] = []
    seen: set[tuple] = set()
    for ix, raw in enumerate(configs):
        row = _mapping(raw, f"configs[{ix}]")
        label = f"configs[{ix}]"
        width = _integer(row.get("hidden_channels"), label + ".hidden_channels", 1, 4096)
        latent = _integer(row.get("latent_resolution"), label + ".latent_resolution", 1, 4096)
        modes = _integer(row.get("n_modes"), label + ".n_modes", 1, 4096)
        lr = _number(row.get("learning_rate"), label + ".learning_rate")
        radius = _number(row.get("radius_factor"), label + ".radius_factor")
        key = (width, latent, modes, lr, radius)
        if key in seen:
            raise EvidenceError(f"duplicate configuration: {label}")
        seen.add(key)
        observations = row.get("epoch_seconds")
        if not isinstance(observations, list) or len(observations) != n_observations:
            raise EvidenceError(f"{label}: wrong epoch sample count")
        values = [_number(x, f"{label}.epoch_seconds[{j}]") for j, x in enumerate(observations)]
        median = statistics.median(values)
        declared = _number(row.get("epoch_seconds_median"), label + ".epoch_seconds_median")
        if not math.isclose(median, declared, rel_tol=1e-11, abs_tol=1e-8):
            raise EvidenceError(f"{label}: stale or fabricated epoch median")
        val_observations = row.get("val_pass_seconds")
        if not isinstance(val_observations, list) or len(val_observations) != n_observations:
            raise EvidenceError(f"{label}: wrong validation sample count")
        val_values = [_number(v, f"{label}.val_pass_seconds[{j}]") for j, v in enumerate(val_observations)]
        val = statistics.median(val_values)
        declared_val = _number(row.get("val_pass_seconds_median"), label + ".val_pass_seconds_median")
        if not math.isclose(val, declared_val, rel_tol=1e-11, abs_tol=1e-8):
            raise EvidenceError(f"{label}: stale or fabricated validation median")
        medians.append(median)
        maxima.append(max(values))
        validations.append(val)
        validation_maxima.append(max(val_values))

    val_cost_s = sum(validations) + final_seeds * max(validations)
    val_cost_max_s = sum(validation_maxima) + final_seeds * max(validation_maxima)
    median_train_s = epochs * (sum(medians) + final_seeds * max(medians))
    observed_max_train_s = epochs * (sum(maxima) + final_seeds * max(maxima))

    def hours(train: float, multiplier: float = 1.0, *, validation_max: bool = False) -> float:
        seconds = train * multiplier + (val_cost_max_s if validation_max else val_cost_s)
        if not math.isfinite(seconds):
            raise EvidenceError("projected duration overflow")
        return seconds / 3600.0

    nominal = hours(median_train_s)
    max_sample_scenario = hours(observed_max_train_s, validation_max=True)
    # Break-even factors are sensitivity thresholds, NOT confidence intervals.
    median_factor = (budget_hours * 3600.0 - val_cost_s) / median_train_s
    max_factor = (budget_hours * 3600.0 - val_cost_max_s) / observed_max_train_s
    return {
        "policy": POLICY,
        "release_verdict": "HOLD_OWNER_APPROVAL_REQUIRED",
        "research_result": "NONE_NO_T1_MODEL_FIT",
        "evidence_source_git_sha": sha,
        "cost_evidence_kind": "SYNTHETIC_OBSERVATIONAL_NOT_A_TAIL_BOUND",
        "config_count": len(configs),
        "measured_epochs_per_config": n_observations,
        "epochs_per_fit": epochs,
        "final_seeds_assumed": final_seeds,
        "budget_hours": budget_hours,
        "estimated_median_hours": nominal,
        "observed_sample_max_scenario_hours": max_sample_scenario,
        "median_headroom_hours": budget_hours - nominal,
        "median_training_hours": median_train_s / 3600.0,
        "median_validation_hours": val_cost_s / 3600.0,
        "validation_passes_counted": len(configs) + final_seeds,
        "fit_epochs_counted": epochs * (len(configs) + final_seeds),
        "observed_sample_max_headroom_hours": budget_hours - max_sample_scenario,
        "median_break_even_train_slowdown_factor": median_factor,
        "observed_max_break_even_train_slowdown_factor": max_factor,
        "stress_hours_median_base": {
            "1.00x": hours(median_train_s, 1.00),
            "1.05x": hours(median_train_s, 1.05),
            "1.10x": hours(median_train_s, 1.10),
            "1.20x": hours(median_train_s, 1.20),
        },
        "stress_hours_sample_max_base": {
            "1.00x": hours(observed_max_train_s, 1.00, validation_max=True),
            "1.05x": hours(observed_max_train_s, 1.05, validation_max=True),
            "1.10x": hours(observed_max_train_s, 1.10, validation_max=True),
            "1.20x": hours(observed_max_train_s, 1.20, validation_max=True),
        },
        "limitations": [
            "Three epoch observations per configuration cannot establish an upper runtime bound.",
            "The cost record uses synthetic tensors; model and dataset shift cannot be inferred.",
            "Validation maxima are observed sample maxima, not statistical bounds.",
            "Warm-up, checkpoint I/O, retries, system load, and evaluation overhead are not bounded here.",
            "The evidence's passed=true and clean-tree flags are source claims, not authenticated by this audit.",
            "The eight-run cost is conservatively followed by five selected-config runs; "
            "actual overlap is not assumed.",
        ],
    }


# Pin values obtained from the GitHub connector by fetching the exact PR #91 head,
# independently of the JSON's self-reported source metadata. The SHA256 was
# computed over the 14,345 ASCII bytes fetched at the pinned commit; SHA1 is
# the corresponding Git blob object ID, not a claim of signed provenance.
PR91_HEAD = "1a54cd8d9326e762ee493d3fc8d9a80202174304"
PR91_EVIDENCE_BLOB = "cc6bfbfcd95127915d520f283740ec7c61b206f5"
PR91_EVIDENCE_SHA256 = "8d9212597ba5e827f2c1d9fd707dd20d6c5755d5e70d1e693869dd1ec82f61ed"
PR91_MEASUREMENT_TREE = "35f82dcff7821f41073537fdf514ec9ff3987774"
PR91_SURVEY_TREE = "69740d1943f905ec4653021e8f681f03a8a00f27"
PR91_SURVEY_SHA256 = "3111a480fa934e0bf027a092f4fa389fdb4faa500edf60a6a737160df471b786"
PR91_SURVEY_BLOB = "9073f51b26cf77165cc4cea37bb8377fd170dd05"
PR91_CONFIGS = {(w, n, modes, lr, 1.5) for w in (32, 64) for n, modes in ((16, 6), (24, 9)) for lr in (0.001, 0.0003)}


def _exact(value: Any, expected: Any, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise EvidenceError(f"PR91 {label}: expected {expected!r}, got {value!r}")


def verify_pr91_contract(data: dict) -> dict:
    """Bind the cost JSON to PR #91 D4/D5 *proposal*, not to owner approval.

    Returns a description of the eligible evidence assumptions, never an
    authorization to run T1 or scientific superiority verdict.
    """
    data = _mapping(data, "root")
    source = _mapping(data.get("source"), "source")
    _exact(source.get("git_sha"), PR91_MEASUREMENT_TREE, "measurement source revision")
    _exact(source.get("github_sha"), None, "source.github_sha")
    _exact(data.get("passed"), True, "passed")
    env = _mapping(data.get("environment"), "environment")
    for key, val in {
        "python": "3.12.10",
        "torch": "2.6.0+cpu",
        "neuraloperator": "2.0.0",
        "numpy": "2.5.3",
        "dtype": "float32",
        "amp": False,
        "cuda_available": False,
        "torch_num_threads": 4,
    }.items():
        _exact(env.get(key), val, "environment." + key)
    hw = _mapping(data.get("hardware"), "hardware")
    for key, val in {
        "cpu_name": "Intel(R) Core(TM) i5-1035G1 CPU @ 1.00GHz",
        "physical_cores": 4,
        "logical_processors": 8,
        "ram_bytes": 8260374528,
        "os": "Microsoft Windows 11 Home 10.0.26200 build 26200",
    }.items():
        _exact(hw.get(key), val, "hardware." + key)
    d = _mapping(data.get("data"), "data")
    for key, val in {
        "geometry": "triangle_hex_lattice(32), 465 nodes",
        "batch_size": 20,
        "n_train": 1800,
        "n_val": 200,
        "steps_per_epoch": 90,
        "data_seed": 20261029,
        "shuffle_seed": 0,
        "model_seed": 0,
    }.items():
        _exact(d.get(key), val, "data." + key)
    protocol = _mapping(data.get("protocol"), "protocol")
    for key, val in {
        "measured_epochs": 3,
        "warmup_steps": 10,
        "warmup_val_passes": 1,
        "statistic": "median",
        "clock": "time.perf_counter",
        "optimizer": "Adam",
        "schedule": "CosineAnnealingLR",
        "loss": "relative L2 (mean over batch)",
    }.items():
        _exact(protocol.get(key), val, "protocol." + key)
    _exact(
        data.get("command"),
        [
            "python.exe",
            "scripts/oracles/gino_epoch_cost_synthetic.py",
            "--threads",
            "4",
            "--output",
            "results/2026-10-29_gino_epoch_cost_synthetic.json",
        ],
        "command",
    )
    rows = data.get("configs")
    if not isinstance(rows, list) or len(rows) != 8:
        raise EvidenceError("PR91 requires exactly eight measured configurations")
    # Validates types and medians of every row first: the checks below hash the
    # configuration tuple and read epoch_seconds directly.
    audit = audit_cost(data)
    seen: set[tuple] = set()
    width_latent_to_digest: dict[tuple, str] = {}
    for i, row in enumerate(rows):
        row = _mapping(row, f"configs[{i}]")
        name = f"configs[{i}]"
        _exact(row.get("n_layers"), 4, name + ".n_layers")
        cfg = (
            row.get("hidden_channels"),
            row.get("latent_resolution"),
            row.get("n_modes"),
            row.get("learning_rate"),
            row.get("radius_factor"),
        )
        if cfg not in PR91_CONFIGS or cfg in seen:
            raise EvidenceError(f"PR91 grid mismatch or duplicate: {name}")
        seen.add(cfg)
        digest = row.get("config_digest")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise EvidenceError(f"{name}: invalid config digest")
        group = cfg[:2]
        if group in width_latent_to_digest and width_latent_to_digest[group] != digest:
            raise EvidenceError(f"{name}: config digest changed across learning rates")
        width_latent_to_digest[group] = digest
        _integer(row.get("n_parameters"), name + ".n_parameters", 1, 100_000_000)
        _number(row.get("warmup_steps_seconds"), name + ".warmup_steps_seconds")
        _number(row.get("warmup_val_pass_seconds"), name + ".warmup_val_pass_seconds")
        _exact(row.get("losses_finite"), True, name + ".losses_finite")
        losses = row.get("last_batch_loss_per_epoch")
        if not isinstance(losses, list) or len(losses) != 3:
            raise EvidenceError(f"{name}: loss series incomplete")
        for j, loss in enumerate(losses):
            _number(loss, f"{name}.last_batch_loss_per_epoch[{j}]")
    summary = _mapping(data.get("summary"), "summary")
    medians = [statistics.median(x["epoch_seconds"]) for x in rows]
    for key, expected in {
        "epoch_seconds_median_min": min(medians),
        "epoch_seconds_median_max": max(medians),
        "epoch_seconds_median_sum": sum(medians),
    }.items():
        value = _number(summary.get(key), "summary." + key)
        if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-9):
            raise EvidenceError(f"PR91 summary mismatch: {key}")
    return {
        "source_revision": PR91_MEASUREMENT_TREE,
        "configuration_count": len(seen),
        "hardware_verified_against_candidate": True,
        "operator_version": env["neuraloperator"],
        "research_data": False,
        "checkpoint_io_budgeted": False,
        "measurement_scheduler_t_max": 3,
        "planned_scheduler_t_max": 50,
        "disclaimer": "Candidate-binding only; not a Phase-1 scientific or owner approval.",
        "estimated_median_hours": audit["estimated_median_hours"],
    }


def verify_pr91_bytes(raw: bytes) -> dict:
    """Verify two independent content-addressed fingerprints of original PR source.

    Git SHA-1 is checked as a normal Git blob; SHA-256 is an additional byte pin.
    A locally transcribed fixture must fail here by construction.
    """
    if not isinstance(raw, bytes) or len(raw) > MAX_SOURCE_BYTES:
        raise EvidenceError("invalid evidence byte stream")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    actual_git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw).hexdigest()
    if actual_sha256 != PR91_EVIDENCE_SHA256 or actual_git_blob != PR91_EVIDENCE_BLOB:
        raise EvidenceError("PR91 source bytes differ from pinned commit file")
    return {"sha256": actual_sha256, "git_blob": actual_git_blob, "head_revision": PR91_HEAD}


def verify_radius_survey(data: dict) -> dict:
    """Check the PR #91 radius decision on pinned survey evidence, not performance."""
    data = _mapping(data, "survey_file")
    source = _mapping(data.get("source"), "survey_file.source")
    _exact(source.get("bound"), True, "survey source.bound")
    _exact(source.get("worktree_dirty"), False, "survey source.worktree_dirty")
    _exact(source.get("git_sha"), PR91_SURVEY_TREE, "survey source git SHA")
    env = _mapping(data.get("environment"), "survey_file.environment")
    for key, expected in {
        "python": "3.12.10",
        "torch": "2.6.0+cpu",
        "numpy": "2.5.3",
        "neuraloperator": "2.0.0",
    }.items():
        _exact(env.get(key), expected, "survey environment." + key)
    survey = _mapping(data.get("survey"), "survey_file.survey")
    _exact(survey.get("passed"), True, "survey.passed")
    positive = _mapping(survey.get("positive_control"), "survey.positive_control")
    _exact(positive.get("flip_detector_saw_a_flip"), True, "survey.positive_control")
    if _integer(positive.get("default_flips_total"), "default_flips_total", 1, 1000000) < 1:
        raise EvidenceError("positive control invalid")
    candidates = survey.get("flip_free_candidates")
    if not isinstance(candidates, list) or len(candidates) != 9:
        raise EvidenceError("survey flip-free candidate count mismatch")
    margins = {}
    for i, row in enumerate(candidates):
        row = _mapping(row, f"survey.candidates[{i}]")
        n = _integer(row.get("latent_resolution"), "latent_resolution", 1, 4096)
        factor = _number(row.get("radius_factor"), "radius_factor")
        measured = _number(row.get("margin_over_float32_error"), "margin_over_float32_error")
        absmargin = _number(row.get("min_relative_margin"), "min_relative_margin")
        error = _number(row.get("float32_error_near_radius"), "float32_error_near_radius")
        if not math.isclose(absmargin / error, measured, rel_tol=1e-10, abs_tol=1e-10):
            raise EvidenceError("radius measured margin ratio mismatch")
        pair = (n, factor)
        if pair in margins:
            raise EvidenceError("duplicate survey margin coordinate")
        margins[pair] = measured
    pairs = [(16, 1.5), (24, 1.5)]
    if not all(p in margins for p in pairs):
        raise EvidenceError("PR91 chosen radius pairs absent from survey")
    if not all(margins[p] > 1.0 for p in pairs):
        raise EvidenceError("PR91 chosen radius too close to float32 error")
    factors = (1.5, 2.25, 2.5, 2.75)
    if not all((n, f) in margins for n in (16, 24) for f in factors):
        raise EvidenceError("PR91 radius-choice grid incomplete")
    minimax = {f: min(margins[(16, f)], margins[(24, f)]) for f in factors}
    best = max(factors, key=minimax.get)
    if best != 1.5:
        raise EvidenceError("PR91 selected radius is not highest minimum measured margin")
    return {
        "selected_radius_factor": 1.5,
        "latent16_margin_ratio": margins[(16, 1.5)],
        "latent24_margin_ratio": margins[(24, 1.5)],
        "label": "FLIP_FREE_OBSERVED_ON_SURVEY_MACHINE_ONLY",
        "cross_hardware_claim": False,
    }


def _verify_named_bytes(raw: bytes, *, sha256_pin: str, blob_pin: str, label: str) -> dict:
    if not isinstance(raw, bytes) or len(raw) > MAX_SOURCE_BYTES:
        raise EvidenceError(f"{label}: source byte budget exceeded")
    actual_sha = hashlib.sha256(raw).hexdigest()
    actual_blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw).hexdigest()
    if actual_sha != sha256_pin or actual_blob != blob_pin:
        raise EvidenceError(f"{label}: bytes differ from pinned source")
    return {"sha256": actual_sha, "git_blob": actual_blob}


def preflight_pr91(path: str | pathlib.Path, survey_path: str | pathlib.Path) -> dict:
    """Fail closed unless BOTH source files and D4/D5 candidate assumptions match."""
    with pathlib.Path(path).open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    source_pin = verify_pr91_bytes(raw)
    record = _decode_evidence(raw)
    with pathlib.Path(survey_path).open("rb") as stream:
        raw_survey = stream.read(MAX_SOURCE_BYTES + 1)
    survey_pin = _verify_named_bytes(
        raw_survey, sha256_pin=PR91_SURVEY_SHA256, blob_pin=PR91_SURVEY_BLOB, label="radius survey"
    )
    checked = verify_pr91_contract(record)
    radius_checked = verify_radius_survey(_decode_evidence(raw_survey))
    result = audit_cost(record)
    result["source_pin"] = source_pin
    result["radius_survey_pin"] = survey_pin
    result["radius_survey_gate"] = radius_checked
    result["candidate_contract"] = checked
    result["evidence_tier"] = "TWO_EVIDENCE_FILES_BYTE_PINNED_PR91_CANDIDATE_ONLY"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cost_json", help="Exact cost JSON from PR #91")
    parser.add_argument(
        "--fixture-only",
        action="store_true",
        help="Explicitly allow exploratory calculation on untrusted transcribed fixtures",
    )
    parser.add_argument(
        "--radius-survey", default=None, help="Pinned float32 graph survey JSON from PR #91 (required in strict mode)"
    )
    args = parser.parse_args()
    try:
        if args.fixture_only:
            data, digest = read_evidence(args.cost_json)
            result = audit_cost(data)
            result["input_file_sha256"] = digest
            result["evidence_tier"] = "TEST_FIXTURE_NOT_ORIGINAL_NO_RELEASE_EVIDENCE"
            result["release_verdict"] = "HOLD_UNPINNED_TEST_FIXTURE"
        else:
            if args.radius_survey is None:
                raise EvidenceError("strict PR91 audit requires --radius-survey")
            result = preflight_pr91(args.cost_json, args.radius_survey)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except (EvidenceError, OSError) as exc:
        print(
            json.dumps(
                {"policy": POLICY, "release_verdict": "HOLD_INVALID_EVIDENCE", "error": str(exc)}, sort_keys=True
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
