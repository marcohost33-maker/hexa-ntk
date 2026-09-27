# _SOURCES_raw.md — hexa-ntk (Staging 2026-06-04)

> **Pfad-Entschaerfung 2026-08-26.** Die Quelle-Spalte trug vollstaendige lokale Pfade
> des Arbeitsrechners (Benutzername, Cloud-Ordner). Vor einem Sichtbarkeits-Wechsel
> ist das eine unnoetige Preisgabe ohne Nutzen fuer die Nachvollziehbarkeit; die
> maschinen-spezifischen Praefixe sind durch Platzhalter ersetzt
> (`<staging-drive>` 17x, `<staging-lokal>` 2x). Laufwerksbuchstabe, Projektordner
> und Dateiname bleiben stehen. Die SHA-256-Praefixe sind unveraendert;
> `python scripts/verify_sources.py` deckt weiterhin 17/17 Eintraege (exit 0).
> Achtung: die alten Pfade bleiben in der Git-Historie.


| SHA256 (16) | mtime | Ziel | Quelle |
|---|---|---|---|
| A4B6EF52247ED5AA | 2026-05-25 14:33 | src | C:\<staging-lokal>\coworker-dde\src\hexantk_v3.py |
| 234106F54BFF22FD | 2026-04-06 22:50 | spec | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\HexaNTK_Review_Gehaertet.md |
| 4A97524E450E91CD | 2026-05-26 14:05 | results | C:\<staging-lokal>\coworker-dde\evidence\2026-05-26_gate-run\logs\hexantk_v3.log |

## src/toolkit/ — 16 Module (append-only, ergaenzt 2026-06-19)

> **Schema.** `SHA256 (16)` = `sha256(<repo-Datei>)[:16].upper()` (gleiche Konvention wie oben,
> nicht die Quell-Datei). `mtime` = mtime der **Quell-Datei** (0Archiv-Snapshot, READ-ONLY).
> `Quelle` = Herkunft; "(in-repo modifiziert, Commit X)" markiert Module, deren aktuelle Bytes vom
> 04-06-Snapshot **abweichen**, weil sie nach der Uebernahme im Repo gehaertet wurden (Lint-Baseline
> Commit `0865915`, Atomic-Checkpoint-Fix `c6628d8`, Durability-Praezisierung `5b88d29`) —
> Provenance = Original-Quelle + abgeleitete Form (SLSA-Praxis: beide festhalten). Re-verifizierbar
> mit `python scripts/verify_sources.py` (rechnet die SHA-256 der Repo-Dateien gegen diese Tabelle).
> Die 8 NICHT als "in-repo modifiziert" markierten Module sind byte-identisch zum 04-06-Snapshot.

| 7D48D5B659D08A05 | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\benchmark_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| 2BBFD0218DEBBF0F | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\checkpoint_integration_hex_v3.py (in-repo modifiziert, Commit 5b88d29) |
| A8ACCD446EC287D6 | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\dataloader_utils_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| DF201FDE60688DAF | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\dataset_split_hex_v3.py |
| 72019F9045DF105D | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\ema_hex_v3.py |
| 4C0BC9B6D6233180 | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\failure_signature_hex_v3.py |
| D0192688CE5222F1 | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\operator_adapter_hex_v3.py |
| BB8B7E456C1B7DEB | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\resume_utils_hex_v3.py |
| 8B1886090583A412 | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\runner_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| DE5B1FE6EA89314D | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_benchmark_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| E18BF23374494EE9 | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_checkpoint_integration_hex_v3.py (in-repo modifiziert, Commit c6628d8) |
| 54FE8C5313B45824 | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_dataloader_utils_hex_v3.py |
| 574B8851CE411016 | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_ema_hex_v3.py |
| C0EAD8D722BCE4D4 | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_operator_adapter_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| AB14B563E7D9EA9C | 2026-07-03 00:00 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_runner_hex_v3.py (in-repo modifiziert: HexTrainingRunner-Coverage-Tests 25%->79%, vero/hexa-ntk-coverage-ratchet-runner) |
| 1FAB88FF9CCA6627 | 2026-04-06 22:50 | src/toolkit | G:\<staging-drive>\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\warmstart_utils_hex_v3.py |

## src/toolkit/ — Ergaenzung 2026-07-03 (append-only)

> Neues Testfile, **in-repo erstellt** (kein 0Archiv-Snapshot als Quelle): schliesst die
> Coverage-Luecke von `dataset_split_hex_v3.py` (vorher 0% — einziges Modul ohne Tests).
> `mtime` = Erstellzeit im Repo.

| 8890E69A5CC1AA6B | 2026-07-03 03:51 | src/toolkit | in-repo erstellt, Branch claude/code-analysis-optimization-ajgyce: test_dataset_split_hex_v3.py |

## src/toolkit/ — Derived forms 2026-09-23 (append-only, PR #31)

> Imported-original locators above remain immutable. These rows record the
> current derived bytes for the explicit resume_from path fix and its tests.

| B8BE0568BB7AF782 | 2026-09-23 00:00 | src/toolkit | in-repo derived, PR #31 explicit resume path: checkpoint_integration_hex_v3.py |
| 0A917FA7C56F2222 | 2026-09-23 00:00 | src/toolkit | in-repo derived, PR #31 explicit resume path + fail-closed missing resume_from: runner_hex_v3.py |
| 00AC58F5C9B3E19F | 2026-09-23 00:00 | src/toolkit | in-repo derived, PR #31 regression tests: test_checkpoint_integration_hex_v3.py |
| 385A6A2DC36E4AC9 | 2026-09-23 00:00 | src/toolkit | in-repo derived, PR #31 regression tests (+ own-checkpoint-present case, F31-1; missing resume_from fails closed): test_runner_hex_v3.py |
## src/toolkit/ — Derived forms 2026-09-23 (append-only, PR #30)

> The imported-original rows above remain immutable. These additional rows pin
> the current in-repo derived bytes for the H1/benchmark hardening; the final
> SHA values are filled from the fail-closed provenance gate.

| 97339519C3A96F28 | 2026-09-23 00:00 | src/toolkit | in-repo derived, PR #30 H1/benchmark hardening (+ float32 accumulation, per-step H1 loop, single validation): benchmark_hex_v3.py |
| 2466DB271C4DAE8D | 2026-09-23 00:00 | src/toolkit | in-repo derived, PR #30 regression tests (+ exact-gradient H1 oracle, F30-1; fp16/memory/validation, round 2): test_benchmark_hex_v3.py |

## src/toolkit/ — H1 error-only path 2026-09-24 (append-only, PR #30)

> The unnormalised H1 RMSE path no longer computes the target gradient that
> only the normalised metric needs. The regression binds one stencil call per
> rollout step while H1 nRMSE keeps both error and target terms.

| 0392725750BAF4D9 | 2026-09-24 00:00 | src/toolkit | in-repo derived, PR #30 error-only H1 path: benchmark_hex_v3.py |
| B6B1E9916FD640A4 | 2026-09-24 00:00 | src/toolkit | in-repo derived, PR #30 error-only H1 regression: test_benchmark_hex_v3.py |
## src/toolkit/ — Ergaenzung 2026-09-23 (append-only)

> Neue Stress-/Shift-Vertragsdateien, in-repo erstellt. Die SHA-Pins werden vom
> fail-closed Provenance-Gate gegen die aktuellen Repo-Bytes verifiziert.

| 7074616F72C9870D | 2026-09-23 00:21 | src/toolkit | in-repo erstellt, PR #32: stress_hex_v3.py |
| 736E7595EDAAF3AC | 2026-09-23 00:21 | src/toolkit | in-repo erstellt, PR #32: test_stress_hex_v3.py |

## src/toolkit/ — Review-hardening derived forms 2026-09-23 (append-only, PR #32)

> The original PR #32 creation rows above are retained as lineage. These rows
> pin the post-review immutable/self-consistent stress evidence implementation.

| 34C92C112A40BE46 | 2026-09-23 00:49 | src/toolkit | in-repo derived, PR #32 review-hardening: stress_hex_v3.py |
| 75AEA74A3CF101F0 | 2026-09-23 00:49 | src/toolkit | in-repo derived, PR #32 review-hardening tests: test_stress_hex_v3.py |

## spec/ + src/toolkit/ — Second-review hardening 2026-09-23 (append-only, PR #32)

> The authoritative stress contract itself is now SHA-indexed. Prior toolkit
> rows remain as lineage; the following rows pin the current underflow-hardened
> implementation and tests.

| A9EB2FEE22AB4A97 | 2026-09-23 01:00 | spec | in-repo created/updated, PR #32: STRESS_SHIFT_CONTRACT_v1.md |
| 34876C4B56EFACA1 | 2026-09-23 01:00 | src/toolkit | in-repo derived, PR #32 underflow hardening: stress_hex_v3.py |
| 4A0685F445D0A679 | 2026-09-23 01:00 | src/toolkit | in-repo derived, PR #32 underflow regression tests: test_stress_hex_v3.py |

## spec/ + src/toolkit/ — Third-review hardening 2026-09-23 (append-only, PR #32)

> Equalita F32-1/F32-2: direction-preserving degradation floor
> (`max(E_s, eps) / max(E_0, eps)` + `degradation_floored`) and the spec/ target
> in `scripts/verify_sources.py` (the spec pin is now gate-checked). Prior rows
> remain as lineage; the gate reads the last row per file name.

| 95100B4D8B668153 | 2026-09-23 10:49 | spec | in-repo updated, PR #32 direction-preserving floor: STRESS_SHIFT_CONTRACT_v1.md |
| 23EA1FBDA46C5E5A | 2026-09-23 10:47 | src/toolkit | in-repo derived, PR #32 direction-preserving floor: stress_hex_v3.py |
| 4FE5220068122DCE | 2026-09-23 10:56 | src/toolkit | in-repo derived, PR #32 floor/underflow regression tests: test_stress_hex_v3.py |
| C4E4199FE420E0D6 | 2026-09-23 10:51 | src/toolkit | in-repo erstellt, PR #32 spec-target gate tests: test_verify_sources_gate.py |

## src/toolkit/ — Gate covers every file type 2026-09-23 (append-only, PR #32, E7)

> scripts/verify_sources.py now enumerates all files under src/toolkit/ and spec/
> (git ls-files, any extension); the gate tests below probe type, case,
> sub-directory and symlink. Prior rows remain as lineage.

| 84B550E7AECC05BC | 2026-09-23 12:17 | src/toolkit | in-repo derived, PR #32 E7 all-file-type gate tests: test_verify_sources_gate.py |

## src/toolkit/ — Git-mode exclusion regression 2026-09-24 (append-only, PR #32)

> The documented __pycache__/ exception is now enforced in Git mode too;
> a tracked cache artifact does not become a provenance source merely because
> it is present in the index.

| 0C8B6BD519D34E81 | 2026-09-24 00:00 | src/toolkit | in-repo derived, PR #32 tracked-pycache regression: test_verify_sources_gate.py |

## src/toolkit/ — Final-review fail-closed hardening 2026-09-24 (append-only, PR #32)

> Final re-review hardening: filesystem target-root symlinks, traversal errors
> and broken Git metadata no longer downgrade to a passing archive scan; exact
> Real values are sign-checked before binary64 conversion and nonzero values
> that underflow to zero are refused. Prior rows remain lineage.

| 0E7E5F55A8F6DEC9 | 2026-09-24 00:15 | src/toolkit | in-repo derived, PR #32 exact-Real numeric boundary: stress_hex_v3.py |
| A76CB45A01242259 | 2026-09-24 00:15 | src/toolkit | in-repo derived, PR #32 exact-Real regressions: test_stress_hex_v3.py |
| 8051560D9B84B2F8 | 2026-09-24 00:15 | src/toolkit | in-repo derived, PR #32 archive/git traversal regressions: test_verify_sources_gate.py |

## src/toolkit/ — binary64 overflow contract 2026-09-25 (append-only, follow-up to PR #32)

> Exact Reals beyond binary64 range (Fraction(10**400), 10**400) made float()
> raise OverflowError instead of the contract ValueError; the conversion in
> _finite_nonnegative and in metadata normalization now re-raises it as
> ValueError (from exc). Prior rows remain lineage.

| 0B0A80BFDC117DC1 | 2026-09-25 11:22 | src/toolkit | in-repo derived, PR #32 follow-up binary64 overflow contract: stress_hex_v3.py |
| A02425898B8A21BC | 2026-09-25 11:22 | src/toolkit | in-repo derived, PR #32 follow-up overflow regressions: test_stress_hex_v3.py |

## src/toolkit/ — binary64 overflow contract, sibling validators 2026-09-25 (append-only, PR #35 E2)

> Same class in benchmark_hex_v3 (_coerce_real_scalar, _validate_nonnegative_finite;
> _validate_positive_finite via _coerce_real_scalar) and dataset_split_hex_v3
> (_extract_dt_value, int dt): OverflowError from float() is re-raised as
> ValueError (from exc), keeping each module's existing wording where one exists.
> Prior rows remain lineage.

| 929022D78BCA8E15 | 2026-09-25 11:51 | src/toolkit | in-repo derived, PR #35 E2 binary64 overflow contract: benchmark_hex_v3.py |
| 4C18ACC89FDB92A9 | 2026-09-25 11:51 | src/toolkit | in-repo derived, PR #35 E2 overflow regressions: test_benchmark_hex_v3.py |
| 51FBAC90CF50961F | 2026-09-25 11:51 | src/toolkit | in-repo derived, PR #35 E2 dt binary64 overflow: dataset_split_hex_v3.py |
| 331CA34CF4FEA7E8 | 2026-09-25 11:51 | src/toolkit | in-repo derived, PR #35 E2 dt overflow regressions: test_dataset_split_hex_v3.py |

## src/toolkit/ — metadata overflow cause assert 2026-09-25 (append-only, PR #35 P3-2)

> Review finding P3-2: the `from exc` chain in the stress metadata path was
> untested (mutant `from None` stayed green). Test-only change; prior rows
> remain lineage.

| EE8D734CA0F9567A | 2026-09-25 12:22 | src/toolkit | in-repo derived, PR #35 P3-2 metadata cause assert: test_stress_hex_v3.py |

## src/toolkit/ — trajectory_id and best-metric contracts 2026-09-25 (append-only, issues #36 #37)

> #36: int(trajectory_id) raised a raw OverflowError for inf and silently
> truncated non-integral reals (1.5 -> group 1); _extract_trajectory_id now
> raises the contract ValueError (from exc) and keeps every exact integer key.
> #37: save_best_model_checkpoint validated the metric after changing state and
> promoted NaN as best; the metric is now checked first (NaN/inf/10**400 ->
> ValueError), best_metric_value is committed only after the write succeeds,
> and the runner skips a NaN/inf validation metric with a FailureSignature
> (kind NaN_or_Inf_Metric, documented in failure_signature). Prior rows remain
> lineage.

| F30EF2CE260AA5E1 | 2026-09-25 22:26 | src/toolkit | in-repo derived, #36 trajectory_id finite-integer contract: dataset_split_hex_v3.py |
| 8292860F737550D6 | 2026-09-25 22:26 | src/toolkit | in-repo derived, #36 trajectory_id regressions: test_dataset_split_hex_v3.py |
| CB266B033EDD716A | 2026-09-25 22:26 | src/toolkit | in-repo derived, #37 validate-write-commit best metric: checkpoint_integration_hex_v3.py |
| 133BD71D42FCAEB3 | 2026-09-25 22:26 | src/toolkit | in-repo derived, #37 best-metric regressions: test_checkpoint_integration_hex_v3.py |
| BC73D943B9725297 | 2026-09-25 22:26 | src/toolkit | in-repo derived, #37 non-finite validation metric skip: runner_hex_v3.py |
| 9D4C04AE428D1952 | 2026-09-25 22:26 | src/toolkit | in-repo derived, #37 runner regressions: test_runner_hex_v3.py |
| 6D2F401A0C197440 | 2026-09-25 22:26 | src/toolkit | in-repo derived, #37 NaN_or_Inf_Metric kind documented: failure_signature_hex_v3.py |

## src/toolkit/ — dt finiteness contract 2026-09-26 (append-only, issue #39)

> #39: with group_by_dt=True a non-finite dt became part of the group key;
> nan != nan, so every sample with a fresh nan formed its own group and one
> trajectory could land in both train and val (reproduced 50/50 seeds).
> _extract_dt_value now raises the contract ValueError for nan/inf/-inf
> (float incl. numpy float64, scalar tensor) before the key is grouped; dt is
> still read only under group_by_dt. Prior rows remain lineage.

| 626F935200206F98 | 2026-09-26 11:23 | src/toolkit | in-repo derived, #39 dt finiteness contract: dataset_split_hex_v3.py |
| 8860CF78DF346255 | 2026-09-26 11:23 | src/toolkit | in-repo derived, #39 non-finite dt regressions + leakage invariant: test_dataset_split_hex_v3.py |

## src/toolkit/ — review hardening 2026-09-27 (append-only, issues #40, #42)

> Review of main 9538e87: every finding reproduced before the fix
> (results/2026-09-27_review-hardening_local-gate.log). Resume now restores
> the EMA average and the best metric (verified against best.pt); the runner
> evaluates best every epoch and saves the final epoch; val_nrmse is the
> pooled relative L2 error; epoch/global_step/min_delta/decay/dt/trajectory_id
> contracts fail closed. Two new test modules (resume utils, warmstart utils)
> need their first rows here (SHA-First). Prior rows remain lineage.

| 42D60044E19B8342 | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 #40 min_delta/step-count contract, durability commit, best+EMA resume state: checkpoint_integration_hex_v3.py |
| 00A84DBEEAE647D2 | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 #42 precision-split dt keys fail closed, #40 trajectory_id numpy/bool/str contract: dataset_split_hex_v3.py |
| 4790DFE7CBA298A2 | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 EMA decay contract [0, 1], strict on full-state load: ema_hex_v3.py |
| B4B2D20101A4A8CA | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 Empty_Validation kind documented: failure_signature_hex_v3.py |
| A39A16D67EFFAC0C | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 as_step_count, validate-before-restore, EMA resume state: resume_utils_hex_v3.py |
| D8A9ECF8874EF347 | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 full resume (EMA + best), best every epoch, final-epoch state, real val_nrmse, config contract: runner_hex_v3.py |
| 2DEBCC0F014D7F11 | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 regressions #40 counters/durability/resume best+EMA: test_checkpoint_integration_hex_v3.py |
| A072FA6B50E991D5 | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 regressions #42 and #40 P3-1/5/6: test_dataset_split_hex_v3.py |
| 89BE5BA18AED0777 | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 EMA decay and load-path tests: test_ema_hex_v3.py |
| B8A76790A0B83F81 | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 new test module (resume utils had none): test_resume_utils_hex_v3.py |
| 54C56D743B99827C | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 regressions resume equivalence, schedule, empty validation, nRMSE, config: test_runner_hex_v3.py |
| 53663241136220FA | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 new test module (warmstart 43% coverage): test_warmstart_utils_hex_v3.py |
| 45DE93FC82AEA5ED | 2026-09-27 00:00 | src/toolkit | in-repo derived, review 2026-09-27 fail-closed prefix collisions, skip_shape_mismatch, numpy._core first: warmstart_utils_hex_v3.py |

## src/toolkit/ — Codex review round on PR #46 (append-only, 2026-09-27)

> Three P2 findings, all reproduced and fixed: (1) a numerically better best.pt
> of another run or metric was adopted on resume -> run_id and metric name are
> now recorded and must match; (2) resuming with another ema_decay silently
> mixed two averages -> decay recorded, mismatch raises before any restore;
> (3) scripts/mutation_probe.py counted any nonzero exit as a kill -> baseline
> must be green and a kill needs executed, failing tests. Rows above remain lineage.

| F06CE491B62990BA | 2026-09-27 00:00 | src/toolkit | in-repo derived, Codex review on PR #46: run_id + metric-name identity for best restore, ema_decay record: checkpoint_integration_hex_v3.py |
| 56735A1D24018408 | 2026-09-27 00:00 | src/toolkit | in-repo derived, Codex review on PR #46: ema_decay recorded and checked before restore: resume_utils_hex_v3.py |
| 4721CC2E0CA4A9E4 | 2026-09-27 00:00 | src/toolkit | in-repo derived, Codex review on PR #46: passes ema_decay and best_metric_name on resume: runner_hex_v3.py |
| 9B5DA9254E8CA458 | 2026-09-27 00:00 | src/toolkit | in-repo derived, Codex review on PR #46: regressions foreign best.pt, metric switch, decay mismatch: test_checkpoint_integration_hex_v3.py |
| 0425499B2E3F9453 | 2026-09-27 00:00 | src/toolkit | in-repo derived, Codex review on PR #46: regression resume with other ema_decay: test_runner_hex_v3.py |

## src/toolkit/ — review of PR #46, E0927-03 + H7 (append-only, 2026-09-28)

> Owner decision E0927-03 (issue #42): no trajectory in train AND val, always,
> independent of dt types and group_by_dt. The split unit is now the trajectory;
> dt groups of one trajectory stay together (before: 28/50 seeds split one
> trajectory under group_by_dt with dt 0.1+0.2). Runner test for resume with a
> switched best metric (Equalita H7). Rows above remain lineage.

| 8C279370377CAE96 | 2026-09-28 00:00 | src/toolkit | in-repo derived, E0927-03 trajectory is the split unit, override may not split a trajectory: dataset_split_hex_v3.py |
| 4CC7EC1399C817F1 | 2026-09-28 00:00 | src/toolkit | in-repo derived, E0927-03 invariant tests over seeds, single-dt split pinned: test_dataset_split_hex_v3.py |
| 8D8857A4ED6CF6F8 | 2026-09-28 00:00 | src/toolkit | in-repo derived, H7 regression resume with switched best metric: test_runner_hex_v3.py |
