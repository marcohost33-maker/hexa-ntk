#!/usr/bin/env python3
"""mutation_probe.py - reproducible mutation probe for the 2026-09-27 review fixes.

Each mutant reverts one fix (issues #40, #42, #59, the #60 reference cases,
solver layer and Phase 1 instruments, and the review findings) in place in a
toolkit module (or, for a path with a "/", a repo-relative file such as an
oracle under scripts/), runs the named test class, and restores the file. A
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

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOLKIT = REPO / "src" / "toolkit"


class Mutant(NamedTuple):
    module: str
    original: str
    mutated: str
    test: str


CK = "checkpoint_integration_hex_v3.py"
DS = "dataset_split_hex_v3.py"
RU = "runner_hex_v3.py"
DCP = "scripts/oracles/distributed_checkpoint_bundle_v1.py"
RC = "reference_case_hex_v3.py"
FV = "hex_fv_solver_hex_v3.py"
T1 = "heat_operator_task_hex_v3.py"
P1 = "phase1_protocol_hex_v3.py"
NT = "ntk_trajectory_hex_v3.py"
DDP = "ddp_hex_v3.py"
T2 = "test_ddp_hex_v3.TestTwoRanks"

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
    Mutant("resume_utils_hex_v3.py", "        if recorded != current_decay:\n",
           "        if False:\n",
           "test_checkpoint_integration_hex_v3.TestResumeEmaState"),
    # Codex review #46 (PRRT_kwDOSwvSC86m2eei): the decay is mandatory with EMA
    # state -- on save, for the current EMA on load, and in the checkpoint.
    Mutant("resume_utils_hex_v3.py", "    if ema_state_dict is None:\n        return None\n",
           "    if ema_state_dict is None or ema_decay is None:\n        return None\n",
           "test_resume_utils_hex_v3.TestEmaDecayContract"),
    Mutant("resume_utils_hex_v3.py",
           '    if ema_model is not None and "ema_state_dict" in payload:\n        # Fail-closed',
           '    if ema_model is not None and "ema_state_dict" in payload and ema_decay is not None:\n'
           "        # Fail-closed",
           "test_resume_utils_hex_v3.TestEmaDecayContract"),
    Mutant("resume_utils_hex_v3.py",
           "        if recorded_decay is None:\n            raise EMADecayError(\n",
           "        if recorded_decay is None:\n            recorded_decay = ema_decay\n"
           "        if False:\n            raise EMADecayError(\n",
           "test_resume_utils_hex_v3.TestEmaDecayContract"),
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
    # #59 P2: shard contents are no longer flushed before the marker.
    Mutant(DCP, '    for item in manifest["files"]:\n        fsync_file(bundle_path / item["path"])\n', "",
           "test_distributed_checkpoint_bundle_v1.TestShardContentDurabilityIssue59"),
    Mutant(DCP, '    with path.open("rb+") as stream:\n        os.fsync(stream.fileno())\n',
           '    with path.open("rb+") as stream:\n        pass\n',
           "test_distributed_checkpoint_bundle_v1.TestShardContentDurabilityIssue59"),
    # #59 P2: metric_semantics_label missing from __all__ again.
    Mutant("benchmark_hex_v3.py", '    "metric_semantics_label",\n', "",
           "test_benchmark_hex_v3.TestPublicApiMatchesMetricSpecV4"),
    # #59 P3: dt keys compared purely relatively again (0.0 vs 1e-46 accepted).
    Mutant(DS, "    if bits_lower is not None and bits_upper is not None:\n", "    if False:\n",
           "test_dataset_split_hex_v3.TestDtPrecisionSplitIssue42"),
    # #59 P3: raw bit patterns compared, so -0.0 and 0.0 count as different.
    Mutant(DS, 'struct.unpack("<f", bits_lower)[0] == struct.unpack("<f", bits_upper)[0]',
           "bits_lower == bits_upper", "test_dataset_split_hex_v3.TestDtPrecisionSplitIssue42"),
    # Reality Anchor Phase 0a (hex-heat-tri, #60): operator, analysis, gate.
    Mutant(FV, "        return weight * float(kappa) * (gathered.sum(axis=1) - 6.0 * arr)\n",
           "        return weight * float(kappa) * (gathered.sum(axis=1) - (nb >= 0).sum(axis=1) * arr)\n",
           "test_reference_case_hex_v3.TestLameModesAreExactDiscreteEigenvectors"),
    Mutant(RC, "    p = abs(math.log(abs(quotient))) / math.log(r)\n", "    p = abs(math.log(abs(quotient)))\n",
           "test_reference_case_hex_v3.TestConvergenceAnalysis"),
    Mutant(RC, '        "C5_observed_order": _within(celik_fine["observed_order"], ORDER_BAND),\n',
           '        "C5_observed_order": True,\n', "test_reference_case_hex_v3.TestGateDiscrimination"),
    Mutant(RC, '        "C9_gci_covers_exact_error": celik_fine["gci_fine"] >= exact_rel_error_fine,\n',
           '        "C9_gci_covers_exact_error": True,\n', "test_reference_case_hex_v3.TestGateDiscrimination"),
    Mutant(RC, "    if all(criteria[k] for k in keys):\n", "    if True:\n",
           "test_reference_case_hex_v3.TestGateDiscrimination"),
    Mutant(FV, "    if np.any(off_arr < 0.0):\n", "    if False:\n",
           "test_hex_fv_solver_hex_v3.TestPositivityGateIssue60"),
    Mutant(FV, "    if dt > bound:\n", "    if False:\n", "test_hex_fv_solver_hex_v3.TestPositivityGateIssue60"),
    # Reality Anchor Phase 0b (hex-heat-torus, hex-mms-tri, solver layer).
    Mutant(FV, "        ni = np.where(nj == n, ni + n // 2, ni)\n", "        ni = np.where(nj == n, ni, ni)\n",
           "test_hex_fv_solver_hex_v3.TestTorusLattice"),
    Mutant(FV, "        midpoints = lattice.coords + 0.5 * lattice.h * SLOT_DIRECTIONS[slot]\n",
           "        midpoints = lattice.coords + 0.0 * lattice.h * SLOT_DIRECTIONS[slot]\n",
           "test_hex_fv_solver_hex_v3.TestEdgeConductances"),
    Mutant(FV, "        if not np.array_equal(c[rows, slot], c[lattice.neighbors[rows, slot], slot + 3]):\n",
           "        if False:\n", "test_hex_fv_solver_hex_v3.TestEdgeConductances"),
    Mutant(FV, "        if not curvature > 0.0:\n", "        if False:\n",
           "test_hex_fv_solver_hex_v3.TestConjugateGradient"),
    Mutant(FV, '        raise ValueError("solve_steady_diffusion supports the Dirichlet triangle only (singular on the '
               'torus)")\n',
           "        pass\n", "test_hex_fv_solver_hex_v3.TestSteadySolve"),
    Mutant(RC, "        return self.kappa(xy) * neg_laplacian - (self.kappa_gradient(xy) * self.solution_gradient(xy))"
               ".sum(axis=1)\n",
           "        return self.kappa(xy) * neg_laplacian\n", "test_reference_case_hex_v3.TestHexMmsTriStudyGate"),
    Mutant(RC, '        "O2_source_matches_flux_divergence": source_error <= _MMS_SOURCE_TOL,\n',
           '        "O2_source_matches_flux_divergence": True,\n',
           "test_reference_case_hex_v3.TestMmsGateDiscrimination"),
    # 2026-10-12 (E1012-07): the I1 rule moved into _i1_level_holds (V1/V2); same defect.
    Mutant(RC, '    return bool(level["cg_relative_residual"] <= bound\n'
               '                and level["iterative_to_discretisation_ratio"] <= _MMS_ITERATIVE_RATIO)\n',
           '    return bool(level["cg_relative_residual"] <= bound)\n',
           "test_reference_case_hex_v3.TestMmsGateDiscrimination"),
    Mutant(RC, "        criteria[\"C10_conservation\"] = all(level[\"conservation_residual\"] <= _CONSERVATION_TOL for "
               "level in levels)\n",
           "        criteria[\"C10_conservation\"] = True\n", "test_reference_case_hex_v3.TestHexHeatTorusStudyGate"),
    # Phase 1 instruments (T1 hex-heat-operator, Crank-Nicolson baseline).
    Mutant(FV, "    half = 0.5 * dt\n", "    half = dt\n", "test_hex_fv_solver_hex_v3.TestCrankNicolson"),
    Mutant(FV, "            raise ValueError(\"x0 is not representable at the scale of b\")\n"
               "        r = rhs - np.asarray(apply_a(x), dtype=np.float64)\n",
           "            raise ValueError(\"x0 is not representable at the scale of b\")\n"
           "        r = rhs.copy()\n",
           "test_hex_fv_solver_hex_v3.TestConjugateGradient"),
    # 2026-10-12 P1: the stopping rule is decided on the TRUE residual b - A x.
    Mutant(FV, "        # Pre-check passed: the true residual decides.\n"
               "        r = rhs - np.asarray(apply_a(x), dtype=np.float64)\n"
               "        rr = float(r @ r)\n",
           "        rr = rr_new\n",
           "test_hex_fv_solver_hex_v3.TestConjugateGradient"),
    Mutant(T1, "            for n in range(m, bound + 1):\n", "            for n in range(1, bound + 1):\n",
           "test_heat_operator_task_hex_v3.TestBasis"),
    Mutant(T1, "    sigma = (eigenvalues.min() / eigenvalues) ** float(config.spectral_decay)\n",
           "    sigma = (eigenvalues.min() / eigenvalues) ** 0.0\n",
           "test_heat_operator_task_hex_v3.TestPriorAndTruth"),
    Mutant(T1, "    digest.update(np.ascontiguousarray(u_t, dtype=\"<f8\").tobytes())\n", "",
           "test_heat_operator_task_hex_v3.TestManifest"),
    Mutant(T1, "        [-lattice.cell_area * float(row @ apply_hex_laplacian(row, lattice)) for row in v],\n",
           "        [lattice.cell_area * float(row @ row) for row in v],\n",
           "test_heat_operator_task_hex_v3.TestMetrics"),
    # 2026-10-12 Codex P2: the negative-energy band is per sample, not batch-wide.
    Mutant(T1, "    tolerance = 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(values))\n",
           "    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(values))))\n",
           "test_heat_operator_task_hex_v3.TestMetrics"),
    # 2026-10-23 HNT-R86-03/-04 (issue #82, cross-family GPT-6 R86): whole-string digests,
    # break-even beyond the float range.
    Mutant(P1, '_HEX40 = re.compile(r"^[0-9a-f]{40}\\Z")\n', '_HEX40 = re.compile(r"^[0-9a-f]{40}$")\n',
           "test_phase1_protocol_hex_v3.TestDigestStringsAreWholeStrings"),
    Mutant(P1, '_HEX64 = re.compile(r"^[0-9a-f]{64}\\Z")\n', '_HEX64 = re.compile(r"^[0-9a-f]{64}$")\n',
           "test_phase1_protocol_hex_v3.TestDigestStringsAreWholeStrings"),
    Mutant(P1, "    if math.isfinite(quotient):\n", "    if True:\n",
           "test_phase1_protocol_hex_v3.TestBreakEvenRange"),
    # 2026-10-22 NTK_TRAJECTORY_v1: each guard and each closed-form metric of the trajectory.
    Mutant(NT, "    if given != steps:\n", "    if False:\n", "test_ntk_trajectory_hex_v3.TestFailClosed"),
    # 2026-10-23 HNT-R25-01 (cross-family GPT-6 R25): the duplicate-bytes refusal was
    # retracted (equal bytes prove neither a copy nor stationarity), so its mutant is gone;
    # these pin the markers that replace it and the record check that enforces them.
    Mutant(NT, "        unchanged, stationary = _stationarity(previous, state_digest, kernel_digest)\n",
           "        unchanged, stationary = False, None\n", "test_ntk_trajectory_hex_v3.TestRepeatedStates"),
    Mutant(NT, "        unchanged, stationary = _stationarity(previous, state_digest, kernel_digest)\n",
           "        unchanged, stationary = _stationarity(entries[0] if entries else None, "
           "state_digest, kernel_digest)\n",
           "test_ntk_trajectory_hex_v3.TestRepeatedStates"),
    Mutant(NT, "    return True, \"measured_zero\" if previous.kernel_sha256 == kernel_digest else \"not_measured\"\n",
           "    return True, \"measured_zero\"\n", "test_ntk_trajectory_hex_v3.TestRepeatedStates"),
    Mutant(NT, "            repeated_state_digest_of=None if repeated_of == step else repeated_of,\n",
           "            repeated_state_digest_of=None,\n", "test_ntk_trajectory_hex_v3.TestRepeatedStates"),
    Mutant(NT, "            if e[key] != value or type(e[key]) is not type(value):\n", "            if False:\n",
           "test_ntk_trajectory_hex_v3.TestRecordCheck"),
    Mutant(NT, " or type(e[key]) is not type(value):\n", ":\n", "test_ntk_trajectory_hex_v3.TestRecordCheck"),
    Mutant(NT, "        if value.dtype != ref.dtype or value.shape != ref.shape:\n",
           "        if value.shape != ref.shape:\n", "test_ntk_trajectory_hex_v3.TestFailClosed"),
    Mutant(NT, "        if tensor_sha256(inputs) != inputs_digest:\n", "        if False:\n",
           "test_ntk_trajectory_hex_v3.TestFailClosed"),
    Mutant(NT, "    if any(m.training for m in model.modules()):\n", "    if False:\n",
           "test_ntk_trajectory_hex_v3.TestFailClosed"),
    Mutant(NT, "    return min(1.0, max(0.0, 1.0 - cosine))\n", "    return min(1.0, max(0.0, 1.0 - cosine ** 2))\n",
           "test_ntk_trajectory_hex_v3.TestClosedForms"),
    Mutant(NT, "    if np.array_equal(sa, sb):\n", "    if False:\n", "test_ntk_trajectory_hex_v3.TestBinding"),
    Mutant(NT, "        k0 = entries[0].kernel if entries else kernel\n",
           "        k0 = entries[-1].kernel if entries else kernel\n", "test_ntk_trajectory_hex_v3.TestClosedForms"),
    Mutant(NT, "dist_prev / (step - previous.step)", "dist_prev", "test_ntk_trajectory_hex_v3.TestClosedForms"),
    Mutant(NT, "    effective_rank = float(np.sum(diag)) ** 2 / fro2\n",
           "    effective_rank = float(np.sum(diag)) / math.sqrt(fro2)\n",
           "test_ntk_trajectory_hex_v3.TestClosedForms"),
    Mutant(NT, "(math.sqrt(fro2) * float(target @ target))", "float(target @ target)",
           "test_ntk_trajectory_hex_v3.TestClosedForms"),
    Mutant(NT, "        h.update(encoded)\n", "", "test_ntk_trajectory_hex_v3.TestBinding"),
    Mutant(NT, "    h.update(_tensor_header(t))\n", "", "test_ntk_trajectory_hex_v3.TestBinding"),
    # 2026-10-23 HNT-R85-01: the loader gets the private snapshot; loading the caller's
    # dict would let neuraloperator's loader pop "_metadata" from it.
    Mutant(NT, "        twin.load_state_dict(snapshot, strict=True)\n",
           "        twin.load_state_dict(state, strict=True)\n", "test_ntk_trajectory_hex_v3.TestConstructorMetadata"),
    Mutant(NT, "        if state_dict_sha256(snapshot) != state_digest:\n", "        if False:\n",
           "test_ntk_trajectory_hex_v3.TestPreflightBinding"),
    # 2026-10-23 HNT-R86-01/-02 (cross-family GPT-6 R86): metadata aliasing, drift range.
    Mutant(NT, "            if k == _QUALIFIED_NAME:\n", "            if False:\n",
           "test_ntk_trajectory_hex_v3.TestMetadataAliasing"),
    Mutant(NT, "            if type(k) is not str:\n", "            if False:\n",
           "test_ntk_trajectory_hex_v3.TestMetadataAliasing"),
    Mutant(NT, "    if math.isfinite(drift):\n        return drift  # the v1 formula",
           "    if True:\n        return drift  # the v1 formula", "test_ntk_trajectory_hex_v3.TestDriftRange"),
    Mutant(NT, "    if not math.isfinite(drift):\n        raise NTKTrajectoryError(\"relative drift exceeds",
           "    if False:\n        raise NTKTrajectoryError(\"relative drift exceeds",
           "test_ntk_trajectory_hex_v3.TestDriftRange"),
    Mutant(NT, " or _metadata_bytes(key, value) != _metadata_bytes(key, ref):\n", ":\n",
           "test_ntk_trajectory_hex_v3.TestConstructorMetadata"),
    # 2026-10-26 XF-R88-M2 / XF-R34 (cross-family GPT-6 R88, R34): a callable binds only
    # under a name that leads back to it; the loaded copy must give back the digest.
    Mutant(NT, " and _names_itself(obj, module, qualname)):\n", "):\n",
           "test_ntk_trajectory_hex_v3.TestCrossFamilyR88R98"),
    Mutant(NT, "    if target is obj:\n", "    if True:\n", "test_ntk_trajectory_hex_v3.TestCrossFamilyR88R98"),
    Mutant(NT, "qualname.rpartition(\".\")[2] == name", "True", "test_ntk_trajectory_hex_v3.TestCrossFamilyR88R98"),
    Mutant(NT, "vars(namespace).get(name) is obj", "vars(namespace).get(name) is not None",
           "test_ntk_trajectory_hex_v3.TestCrossFamilyR88R98"),
    Mutant(NT, "        if _loaded_digest(twin, reference) != state_digest:\n", "        if False:\n",
           "test_ntk_trajectory_hex_v3.TestCrossFamilyR88R98"),
    Mutant(NT, "        if _loaded_digest(twin, reference) != state_digest:\n",
           "        if state_dict_sha256(twin.state_dict()) != state_digest:\n",
           "test_ntk_trajectory_hex_v3.TestCrossFamilyR88R98"),
    # DDP runner (spec/DDP_RUNNER_v1.md, ROADMAP "DDP-Integration des Runners"): every
    # rank decision is agreed. A mutant that makes ranks disagree ends in a gloo
    # timeout (60 s) or a wrong result; both turn TestTwoRanks red.
    Mutant(RU, "        status = self.dist.max_int(local)\n", "        status = local\n", T2),
    Mutant(RU, "        if status == 2:\n", "        if False:\n", T2),
    Mutant(RU, '        self.dist.run_agreed(self._build, what="runner construction")\n',
           "        self._build()\n", T2),
    Mutant(RU, "            loss_sum, n_batches, sq_err_sum, sq_ref_sum = self.dist.all_reduce_sum_rank0(local)\n",
           "            loss_sum, n_batches, sq_err_sum, sq_ref_sum = local\n", T2),
    Mutant(RU, "            self.dist.check_exact_partition(\n                indices,",
           "            (lambda *a, **k: None)(\n                indices,", T2),
    Mutant(RU, "        if self.dist is None:\n            return self._save_checkpoints_local(",
           "        if True:\n            return self._save_checkpoints_local(", T2),
    Mutant(RU, "        if self.dist is not None:\n            self._require_equal_batch_count()\n", "", T2),
    Mutant(RU, "        if self.dist.any_true(ref != mine):\n"
           "            raise DistributedContractError(\"resumed model",
           "        if False:\n            raise DistributedContractError(\"resumed model", T2),
    Mutant(RU, '        report = self.dist.run_agreed(self._resume_local, what="resume")\n',
           "        report = self._resume_local()\n", T2),
    Mutant(RU, "            loss_sum, loss_n, grad_sum, grad_n = self.dist.all_reduce_sum(\n"
           "                [loss_sum, loss_n, grad_sum, grad_n]\n            )\n", "            pass\n", T2),
    Mutant(RU, "        self.ckpt_mgr.run_id = self.dist.broadcast_bytes(\n"
           "            bytes.fromhex(self.ckpt_mgr.run_id) if self.dist.rank == 0 else None, 16\n        ).hex()\n",
           "", T2),
    Mutant(DDP, "        failed_somewhere = self.any_true(error is not None)\n",
           "        failed_somewhere = False\n", T2),
    Mutant(DDP, "        if lo != hi:\n", "        if False:\n", T2),
    # Per-rank RNG streams across a resume (dropout + rank seeds must continue).
    Mutant(RU, "        if self.dist is not None:\n            self._restore_rank_rng(report)\n",
           "        pass\n", T2),
    Mutant(RU, "        rank_rng = self._gather_rank_rng_states() if save_state else None\n",
           "        rank_rng = None\n", T2),
    # The same contract, pinned in-process (world size 1 and no process group).
    Mutant(RU, "            best_report = self.ckpt_mgr.save_best_model_checkpoint(\n"
           "                model=self.raw_model,",
           "            best_report = self.ckpt_mgr.save_best_model_checkpoint(\n"
           "                model=self.model,", "test_ddp_hex_v3.TestWorldSizeOne"),
    Mutant(RU, "        if self.adapter is not None and self.adapter.model is not self.model:\n",
           "        if False:\n", "test_ddp_hex_v3.TestWorldSizeOne"),
    Mutant(RU, "        if first != second:\n", "        if False:\n", "test_ddp_hex_v3.TestWorldSizeOne"),
    Mutant(DDP, "    gauss = _bits_float(py[-1]) if py[-2] else None\n", "    gauss = None\n",
           "test_ddp_hex_v3.TestRunnerWithoutProcessGroup"),
    Mutant(RU, '            self.failure_dir = self.failure_dir / f"rank{self.dist.rank}"\n', "            pass\n",
           "test_ddp_hex_v3.TestWorldSizeOne"),
    Mutant(RU, "            if callable(set_epoch):\n                set_epoch(epoch)\n",
           "            if False:\n                set_epoch(epoch)\n",
           "test_ddp_hex_v3.TestRunnerWithoutProcessGroup"),
    Mutant(DDP, "        if duplicated or missing:\n", "        if missing:\n", "test_ddp_hex_v3.TestWorldSizeOne"),
    Mutant(DDP, "        if self.any_true(out_of_range):\n", "        if False:\n", "test_ddp_hex_v3.TestWorldSizeOne"),
    Mutant(DDP, "        return iter(range(self.rank, self.dataset_len, self.num_replicas))\n",
           "        padded = -(-self.dataset_len // self.num_replicas) * self.num_replicas\n"
           "        return iter([i % max(self.dataset_len, 1) for i in range(self.rank, padded, self.num_replicas)])\n",
           "test_ddp_hex_v3.TestExactDistributedEvalSampler"),
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
    path = (REPO if "/" in mutant.module else TOOLKIT) / mutant.module
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
