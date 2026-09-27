#!/usr/bin/env python3
"""mutation_probe.py - reproducible mutation probe for the 2026-09-27 review fixes.

Each mutant reverts one fix (issues #40, #42 and the review findings) in place
in a toolkit module, runs the named test class, and restores the file. A
mutant must turn its test class red ("KILLED"); a surviving mutant means the
tests do not pin that fix.

Why this script exists: earlier ROADMAP entries cited "11/11 mutants" and
"600/600 differential" from scripts that never entered the repo (issue #40
P3-4). This probe is the in-repo evidence for its own claim.

Validity (Codex review on #46): a nonzero exit alone is not a kill. Every
target test class must first pass on the unmutated tree (baseline); otherwise
missing dependencies or import errors would make every mutant look killed. A
mutant counts as KILLED only if its run executed tests ("Ran N tests", N > 0)
and ended in "FAILED (...)".

Run from anywhere (every file is restored, also on failure):
    python scripts/mutation_probe.py
Exit codes: 0 every mutant killed; 1 a mutant survived or a run was not a
valid test failure; 2 a mutant no longer applies (source moved on); 3 the
baseline is red (the probe proves nothing then).
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from typing import NamedTuple

TOOLKIT = pathlib.Path(__file__).resolve().parent.parent / "src" / "toolkit"


class Mutant(NamedTuple):
    module: str
    original: str
    mutated: str
    test: str


CK = "checkpoint_integration_hex_v3.py"
DS = "dataset_split_hex_v3.py"
RU = "runner_hex_v3.py"

MUTANTS = (
    # #40 P3-2: non-main ranks store the raw metric object.
    Mutant(CK, "            self.best_metric_value = metric\n            self.best_metric_name = metric_name\n"
           "            return {\n",
           "            self.best_metric_value = metric_value\n            self.best_metric_name = metric_name\n"
           "            return {\n",
           "test_checkpoint_integration_hex_v3.TestNonMainRankStoresValidatedFloat"),
    # #40 P3-3: no commit when the file was replaced but dir-fsync failed.
    Mutant(CK, "            self.best_metric_value = metric\n            self.best_metric_name = metric_name\n"
           "            raise\n", "            raise\n",
           "test_checkpoint_integration_hex_v3.TestDurabilityErrorAfterReplace"),
    # Codex review on #46: run identity / metric identity / EMA decay.
    Mutant(CK, '        if disk_run_id is None or disk_run_id != state.get("run_id"):\n',
           "        if False:\n",
           "test_checkpoint_integration_hex_v3.TestResumeRestoresBestMetric"),
    Mutant(CK, "        if disk_name != recorded_name:\n", "        if False:\n",
           "test_checkpoint_integration_hex_v3.TestResumeRestoresBestMetric"),
    Mutant("resume_utils_hex_v3.py", "        and float(recorded_decay) != float(ema_decay)\n",
           "        and False\n",
           "test_checkpoint_integration_hex_v3.TestResumeEmaState"),
    # Resume best restore: trust the record without checking best.pt.
    Mutant(CK, "        if not at_least_as_good:\n", "        if False:\n",
           "test_checkpoint_integration_hex_v3.TestResumeRestoresBestMetric"),
    Mutant(CK, 'on_disk <= value if self.best_policy.mode == "min" else on_disk >= value',
           "on_disk <= value",
           "test_checkpoint_integration_hex_v3.TestResumeRestoresBestMetric"),
    Mutant(CK, '        if not best_path.exists():\n            return False, "best_checkpoint_missing"\n', "",
           "test_checkpoint_integration_hex_v3.TestResumeRestoresBestMetric"),
    # #40 counters: accept non-integral / negative values.
    Mutant("resume_utils_hex_v3.py", "    if count != value or count < 0:\n", "    if count < 0:\n",
           "test_resume_utils_hex_v3.TestAsStepCount"),
    Mutant("resume_utils_hex_v3.py", "    if count != value or count < 0:\n", "    if count != value:\n",
           "test_resume_utils_hex_v3.TestAsStepCount"),
    # Detector: median of an empty window for window_size=1.
    Mutant(RU, "        self.min_history = max(1, window_size // 2)", "        self.min_history = window_size // 2",
           "test_runner_hex_v3.TestDetectorContract"),
    # Resume without the EMA average.
    Mutant(RU, "            ema_model=self.ema_model,\n            ema_decay=",
           "            ema_decay=",
           "test_runner_hex_v3.TestFullResumeEquivalence"),
    # Final epoch not checkpointed.
    Mutant(RU, "epoch % self.config.save_every_n_epochs == 0 or is_final_epoch",
           "epoch % self.config.save_every_n_epochs == 0",
           "test_runner_hex_v3.TestCheckpointSchedule"),
    # Empty validation treated as a measurement.
    Mutant(RU, '        if val_report.get("n_batches", 1) == 0:', "        if False:",
           "test_runner_hex_v3.TestEmptyValidationAndTrainEpoch"),
    # #42: precision-split dt keys accepted again.
    Mutant(DS, "    if group_by_dt:\n        _reject_precision_split_dt_keys",
           "    if False:\n        _reject_precision_split_dt_keys",
           "test_dataset_split_hex_v3.TestDtPrecisionSplitIssue42"),
    # #40 P3-5: bool ids / non-canonical strings accepted again.
    Mutant(DS, "    if isinstance(value, (bool, bytes, bytearray)):", "    if isinstance(value, (bytes, bytearray)):",
           "test_dataset_split_hex_v3.TestTrajectoryIdContractIssue40"),
    Mutant(DS, '_CANONICAL_INT_STR = re.compile(r"0|-?[1-9][0-9]*")', '_CANONICAL_INT_STR = re.compile(r"-?[0-9]+")',
           "test_dataset_split_hex_v3.TestTrajectoryIdContractIssue40"),
    # Warmstart: silent key collision.
    Mutant("warmstart_utils_hex_v3.py", "        if new_key in result:\n", "        if False:\n",
           "test_warmstart_utils_hex_v3.TestPrefixTransforms"),
    # EMA: decay outside [0, 1] accepted.
    Mutant("ema_hex_v3.py", "    decay = _validate_decay(decay)\n", "", "test_ema_hex_v3.TestDecayContract"),
)


_RAN = re.compile(r"^Ran (\d+) tests?", re.MULTILINE)


def _run_tests(test: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "unittest", test],
        cwd=TOOLKIT,
        capture_output=True,
        text=True,
    )


def _tests_ran(output: str) -> int:
    match = _RAN.search(output)
    return int(match.group(1)) if match else 0


def run_mutant(mutant: Mutant) -> str:
    """Apply one mutant, run its test class, restore.

    Returns "KILLED", "SURVIVED" or "INVALID" (the run failed without
    executing tests, e.g. a crash before discovery).
    """
    path = TOOLKIT / mutant.module
    original = path.read_text(encoding="utf-8")
    if original.count(mutant.original) != 1:
        raise LookupError(f"{mutant.module}: mutation site not found exactly once: {mutant.original!r}")
    path.write_text(original.replace(mutant.original, mutant.mutated), encoding="utf-8")
    try:
        result = _run_tests(mutant.test)
    finally:
        path.write_text(original, encoding="utf-8")
    if result.returncode == 0:
        return "SURVIVED"
    output = result.stderr + result.stdout
    if _tests_ran(output) > 0 and "FAILED (" in output:
        return "KILLED"
    return "INVALID"


def check_baseline() -> list[str]:
    """Every target class must pass unmutated; return the red ones."""
    red = []
    for test in sorted({m.test for m in MUTANTS}):
        result = _run_tests(test)
        if result.returncode != 0 or _tests_ran(result.stderr + result.stdout) == 0:
            red.append(test)
    return red


def main() -> int:
    red = check_baseline()
    if red:
        for test in red:
            print(f"[BASELINE RED] {test}")
        print("[MUTATION-PROBE] baseline red - no mutant result is valid | FAIL")
        return 3
    killed = 0
    for index, mutant in enumerate(MUTANTS, 1):
        try:
            status = run_mutant(mutant)
        except LookupError as exc:
            print(f"M{index:02d} STALE     {exc}")
            return 2
        killed += status == "KILLED"
        print(f"M{index:02d} {status:<8}  {mutant.module} -> {mutant.test.rsplit('.', 1)[-1]}")
    ok = killed == len(MUTANTS)
    print(f"[MUTATION-PROBE] baseline green | {killed}/{len(MUTANTS)} killed | {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
