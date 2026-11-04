"""Read-only, pre-approval GINO Phase-1 budget-envelope calculator.

Synthetic observational timings are not tail bounds, full-run validation or
scientific results.  Strict mode calls R2's byte-pinned *two-source* verifier.
The only possible research/owner verdict is HOLD.  No torch, I/O writes or fit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import statistics
from typing import Any

from preapproval_budget_hex_v3 import (
    EvidenceError,
    audit_cost,
    preflight_pr91,
    read_evidence,
)

POLICY = "PHASE1_RUNTIME_ENVELOPE_v1"
HOURS = 3600.0
EPOCHS = 50  # D4 draft; immutable until owner changes preregistration.
EXTRA_SEEDS = 5  # Conservatively re-fit selected seed 0 as an extra run.
N_CONFIGS = 8
FIT_COUNT = N_CONFIGS + EXTRA_SEEDS


def _integer(value: Any, name: str, *, lo: int, hi: int) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise EvidenceError(f"{name} must be an integer in [{lo},{hi}]")
    return value


def _finite(value: Any, name: str, *, lo: float, hi: float) -> float:
    if type(value) not in (int, float):
        raise EvidenceError(f"{name} must be a number, not boolean or string")
    try:
        v = float(value)
    except (OverflowError, ValueError) as exc:
        raise EvidenceError(f"{name} cannot be represented") from exc
    if not math.isfinite(v) or not lo <= v <= hi:
        raise EvidenceError(f"{name} must be finite and in [{lo},{hi}]")
    return v


@dataclass(frozen=True)
class Scenario:
    # Each timing/overhead value is a planning ASSUMPTION until separately measured.
    validation_interval_epochs: int = 50
    checkpoint_interval_epochs: int = 50
    checkpoint_seconds_each: float = 0.0
    fit_setup_seconds_each: float = 0.0
    data_generation_seconds: float = 0.0
    final_evaluation_seconds: float = 0.0
    contingency_seconds: float = 0.0
    training_multiplier: float = 1.0
    validation_multiplier: float = 1.0
    budget_hours: float = 48.0

    @classmethod
    def from_mapping(cls, value: Any) -> "Scenario":
        if not isinstance(value, dict):
            raise EvidenceError("scenario must be a JSON object")
        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise EvidenceError("unknown scenario fields: " + ",".join(sorted(unknown)))
        obj = cls(**value)
        _integer(obj.validation_interval_epochs, "validation_interval_epochs", lo=1, hi=EPOCHS)
        _integer(obj.checkpoint_interval_epochs, "checkpoint_interval_epochs", lo=1, hi=EPOCHS)
        for name in (
            "checkpoint_seconds_each",
            "fit_setup_seconds_each",
            "data_generation_seconds",
            "final_evaluation_seconds",
            "contingency_seconds",
        ):
            _finite(getattr(obj, name), name, lo=0.0, hi=1e8)
        for name in ("training_multiplier", "validation_multiplier"):
            _finite(getattr(obj, name), name, lo=1.0, hi=100.0)
        _finite(obj.budget_hours, "budget_hours", lo=1.0, hi=1e6)
        return obj

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _passes(epochs: int, interval: int) -> int:
    """At epochs interval, 2*interval, ..., E; always include final E."""
    return (epochs + interval - 1) // interval


def _measured_stat(record: dict, field: str, *, maximum: bool) -> list[float]:
    result: list[float] = []
    for i, row in enumerate(record["configs"]):
        values = row[field]
        if not isinstance(values, list) or not values:
            raise EvidenceError(f"invalid {field} observations at index {i}")
        result.append(max(values) if maximum else statistics.median(values))
    return result


def _sum_and_selected(rows: list[float]) -> float:
    return sum(rows) + EXTRA_SEEDS * max(rows)


def _source_scheduler_label(record: dict) -> str:
    # The PR#91 source used T_max=args.repeats=3. Its protocol contains no
    # separately recorded value, so never infer that its timing is Phase-1-faithful.
    protocol = record.get("protocol", {})
    t_max = protocol.get("scheduler_t_max")
    if t_max == 50 and type(t_max) is int:
        return "CLAIMED_SCHEDULE_50_NOT_INDEPENDENTLY_BENCHMARKED"
    return "PR91_THREE_EPOCH_SCHEDULER_NOT_PHASE1_FIDELITY"


def evaluate_envelope(record: dict, scenario: Scenario, *, evidence_tier: str) -> dict:
    """Extrapolate exact D4 candidate counts, with explicit unmeasured overhead.

    Validations and checkpoints are counted PER FIT, not once per experiment.
    Neither max-of-three observations nor a user-chosen multiplier is a bound.
    """
    if not isinstance(scenario, Scenario):
        raise EvidenceError("scenario must be a Scenario")
    scenario = Scenario.from_mapping(scenario.to_dict())
    if evidence_tier not in {"PINNED_TWO_SOURCE_CANDIDATE", "TRANSCRIBED_TEST_FIXTURE"}:
        raise EvidenceError("unknown evidence tier")
    # Validates source marker, raw median consistency and exact number of rows.
    basis = audit_cost(
        record, epochs=EPOCHS, final_seeds=EXTRA_SEEDS, budget_hours=scenario.budget_hours, expected_configs=N_CONFIGS
    )
    n_validations = _passes(EPOCHS, scenario.validation_interval_epochs)
    n_checkpoints = _passes(EPOCHS, scenario.checkpoint_interval_epochs)
    extras = {
        "dataset_generation_seconds": scenario.data_generation_seconds,
        "fit_setup_seconds": FIT_COUNT * scenario.fit_setup_seconds_each,
        "checkpoint_seconds": FIT_COUNT * n_checkpoints * scenario.checkpoint_seconds_each,
        "final_evaluation_seconds": scenario.final_evaluation_seconds,
        "contingency_seconds": scenario.contingency_seconds,
    }
    overhead = math.fsum(extras.values())

    def budget(*, sample_max: bool) -> dict:
        train = _measured_stat(record, "epoch_seconds", maximum=sample_max)
        val = _measured_stat(record, "val_pass_seconds", maximum=sample_max)
        training = EPOCHS * _sum_and_selected(train) * scenario.training_multiplier
        validation = n_validations * _sum_and_selected(val) * scenario.validation_multiplier
        total = math.fsum([training, validation, overhead])
        if not math.isfinite(total) or total <= 0:
            raise EvidenceError("invalid projected time")
        headroom = scenario.budget_hours * HOURS - total
        # Solved in closed form; this is sensitivity, not an empirical confidence bound.
        base_training = EPOCHS * _sum_and_selected(train)
        nontraining = validation + overhead
        admissible = scenario.budget_hours * HOURS - nontraining
        train_factor_cap = admissible / base_training if admissible > 0 else None
        speedup_required = (training / admissible) if admissible > 0 else None
        return {
            "train_hours": training / HOURS,
            "validation_hours": validation / HOURS,
            "assumed_overhead_hours": overhead / HOURS,
            "total_hours": total / HOURS,
            "headroom_hours": headroom / HOURS,
            "maximum_training_multiplier_for_this_assumed_scenario": train_factor_cap,
            "minimum_speedup_from_assumed_training_time_to_fit": speedup_required,
            "budget_fit_class": "ESTIMATE_WITHIN_CAP_NOT_CERTIFIED" if headroom >= 0 else "ESTIMATE_EXCEEDS_CAP",
        }

    return {
        "policy": POLICY,
        "source": "PR91_SYNTHETIC_TIMINGS",
        "source_revision": basis["evidence_source_git_sha"],
        "evidence_tier": evidence_tier,
        "research_status": "NONE_NO_T1_MODEL_FIT",
        "owner_verdict": "HOLD_PREREGISTRATION_DRAFT",
        "cost_measurement_fidelity": _source_scheduler_label(record),
        "statistical_upper_bound": False,
        "scientific_cost_comparison": False,
        "frozen_d4_candidate_epoch_count": EPOCHS,
        "selection_configurations": N_CONFIGS,
        "additional_selected_fits_charged": EXTRA_SEEDS,
        "full_fits_budgeted": FIT_COUNT,
        "total_training_epochs_budgeted": FIT_COUNT * EPOCHS,
        "validation_passes_per_fit": n_validations,
        "validation_passes_budgeted": FIT_COUNT * n_validations,
        "checkpoint_writes_per_fit_assumed": n_checkpoints,
        "checkpoint_writes_budgeted": FIT_COUNT * n_checkpoints,
        "overhead_breakdown_assumed_seconds": extras,
        "assumptions": scenario.to_dict(),
        "nominal_median_scenario": budget(sample_max=False),
        "observed_sample_max_scenario": budget(sample_max=True),
        "limitations": [
            "Synthetic three-epoch scheduling differs from D4's 50-epoch cosine scheduler.",
            "No statistical upper tail can be established from three timing samples per configuration.",
            "Overheads/multipliers are user-supplied assumptions unless independently measured.",
            "Selected seed 0 is deliberately charged twice until owner makes a reuse decision.",
            "No T1 fit, source remeasurement, repository integration, or approval is implied.",
        ],
    }


def parse_json_scenario(path: Path) -> Scenario:
    with path.open("rb") as f:
        raw = f.read(65537)
    if len(raw) > 65536:
        raise EvidenceError("scenario exceeds 64 KiB")
    from preapproval_budget_hex_v3 import _decode_evidence

    return Scenario.from_mapping(_decode_evidence(raw))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cost_json", type=Path)
    parser.add_argument("--radius-survey", type=Path, help="mandatory for strict byte-pinned source evidence")
    parser.add_argument("--scenario", type=Path, help="JSON planning assumptions; omit for R2-equivalent defaults")
    parser.add_argument(
        "--fixture-only",
        action="store_true",
        help="allow transcribed untrusted test data, never count as source evidence",
    )
    args = parser.parse_args(argv)
    try:
        scenario = parse_json_scenario(args.scenario) if args.scenario else Scenario.from_mapping({})
        if args.fixture_only:
            if args.radius_survey is not None:
                raise EvidenceError("fixture-only may not imply a radius-survey evidence gate")
            record, digest = read_evidence(args.cost_json)
            evidence_tier = "TRANSCRIBED_TEST_FIXTURE"
        else:
            if args.radius_survey is None:
                raise EvidenceError("two-source strict verification requires --radius-survey")
            strict = preflight_pr91(args.cost_json, args.radius_survey)
            record, digest = read_evidence(args.cost_json)
            if strict["source_pin"]["sha256"] != digest:
                raise EvidenceError("source changed since verification")
            evidence_tier = "PINNED_TWO_SOURCE_CANDIDATE"
        out = evaluate_envelope(record, scenario, evidence_tier=evidence_tier)
        out["source_file_sha256"] = digest
        print(json.dumps(out, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (EvidenceError, OSError, OverflowError) as exc:
        print(json.dumps({"policy": POLICY, "owner_verdict": "HOLD_INVALID_INPUT", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
