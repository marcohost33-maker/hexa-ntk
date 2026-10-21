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

## spec/ + src/toolkit/ — Post-merge review follow-up 2026-09-27 (append-only, PR #45)

> Codex review threads on PR #45 (open after merge): the spec now carries the
> positivity/DMP contract h^2/(4 D_v) next to the L2 bound (section 5.2a), and
> new regression tests pin the overflow-safe, toward-zero timestep bounds and
> the README source anchors. Prior rows remain lineage.

| FA25DE9F8A724345 | 2026-09-27 00:00 | spec | in-repo updated, PR #45 follow-up positivity contract (5.2a): HexaNTK_Review_Gehaertet.md |
| 1AAB1765CE69D520 | 2026-09-27 00:00 | src/toolkit | in-repo created, PR #45 follow-up timestep-contract regressions: test_hex_timestep_contract.py |

## src/toolkit/ — Post-merge review round 2 2026-09-27 (append-only, PR #45)

> Equalita round 2: bounds are now computed from the exact value of the GIVEN
> numbers.Real (Fraction), safety is checked exactly, and the AGENTS.md anchor
> plus the README toolkit counts are pinned by tests. Prior rows remain lineage.

| 5ACC8D1F335C55F7 | 2026-09-27 00:00 | src/toolkit | in-repo derived, PR #45 follow-up round 2 (exact given-Real bounds, doc anchors): test_hex_timestep_contract.py |

## src/toolkit/ — Codex review on PR #47 2026-09-28 (append-only)

> Operands are validated on their exact rational value; only a non-representable
> result raises (e.g. h=10**400, D_v=10**800 -> exactly 0.25). Prior rows remain lineage.

| 9BF8C7A71221F307 | 2026-09-28 00:00 | src/toolkit | in-repo derived, PR #47 Codex follow-up (exact operand validation): test_hex_timestep_contract.py |

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

## src/toolkit/ — Codex review of PR #46 after merging main (append-only, 2026-09-28)

> PRRT_kwDOSwvSC86m2eei: EMA resume state now requires its decay on save and the
> current decay on load (EMADecayError, fail-closed; checkpoints without decay
> metadata resume only without EMA). PRRT_kwDOSwvSC86m2eeU: README counts pinned
> to the tree in TestDocsContract. Rows above remain lineage.

| BA047FF01BB4E3F5 | 2026-09-28 00:00 | src/toolkit | in-repo derived, Codex review on PR #46: save_resume validates the EMA decay on every rank: checkpoint_integration_hex_v3.py |
| B9FD4DF90F15C832 | 2026-09-28 00:00 | src/toolkit | in-repo derived, Codex review on PR #46: EMADecayError, decay mandatory on save and load: resume_utils_hex_v3.py |
| 48AC0F8C6AA0A75C | 2026-09-28 00:00 | src/toolkit | in-repo derived, Codex review on PR #46: manager round-trip with decay, save/load/legacy decay contract: test_checkpoint_integration_hex_v3.py |
| 1D9EF5DEA81373F6 | 2026-09-28 00:00 | src/toolkit | in-repo derived, Codex review on PR #46: TestEmaDecayContract: test_resume_utils_hex_v3.py |
| 5FB2BB8609C96209 | 2026-09-28 00:00 | src/toolkit | in-repo derived, Codex review on PR #46: runner records decay, legacy resume rejected with EMA: test_runner_hex_v3.py |
| ED29A89E81B871F6 | 2026-09-28 00:00 | src/toolkit | in-repo derived, Codex review on PR #46: README counts pinned to the tree: test_hex_timestep_contract.py |

## src/toolkit/ — Runner evidence semantics 2026-10-04 (Codex)

> Validation no longer aliases loss to nRMSE, empty validation fails closed,
> configured best metrics must exist, and EMA/raw validation shares the canonical
> OperatorAdapter path. Prior rows remain lineage.

| C51BEA39593F07A7 | 2026-10-04 23:00 | src/toolkit | in-repo derived, codex/runner-evidence-semantics-p0: runner_hex_v3.py |
| BFD8DC30774789CF | 2026-10-04 23:00 | src/toolkit | in-repo derived, codex/runner-evidence-semantics-p0: test_runner_hex_v3.py |

## Integration probe step02_pr48 (append-only, 2026-10-05, vero/hexa-integration-probe)

> Probe-only pins for merged bytes. Prior rows remain lineage.

| 6E571FB9C282E945 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration probe step02_pr48 (merged bytes): runner_hex_v3.py |
| B40A2DCCC489FCCC | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration probe step02_pr48 (merged bytes): test_runner_hex_v3.py |

## src/toolkit/ — Warmstart/EMA coverage ratchet 2026-10-04 (Codex)

> Test-only salvage from the useful, still-valid parts of draft PR #20.
> No production-code semantics are changed; prior rows remain lineage.

| B5C8477B5F985FD3 | 2026-10-04 23:00 | src/toolkit | in-repo derived, codex/coverage-ratchet-warmstart-ema: test_ema_hex_v3.py |
| 665DC949ADA471EA | 2026-10-04 23:00 | src/toolkit | in-repo created, codex/coverage-ratchet-warmstart-ema: test_warmstart_utils_hex_v3.py |

## Integration probe step03_pr50 (append-only, 2026-10-05, vero/hexa-integration-probe)

> Probe-only pins for merged bytes. Prior rows remain lineage.

| 9A325FE96FA64572 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration probe step03_pr50 (merged bytes): test_ema_hex_v3.py |
| 665DC949ADA471EA | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration probe step03_pr50 (merged bytes): test_warmstart_utils_cov_hex_v3.py |
| 53663241136220FA | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration probe step03_pr50 (merged bytes): test_warmstart_utils_hex_v3.py |

## src/toolkit/ — MetricSpec v4 2026-10-04 (Codex)

> Adds explicit mean-of-per-sample nRMSE, forecast-only time aggregation,
> threshold-crossing terminology, a metric-spec digest, and legacy-report
> preservation. Existing v3 functions remain available for reproduction.

| DD71A9F23937E5D7 | 2026-10-04 23:00 | src/toolkit | in-repo derived, codex/metricspec-v4: benchmark_hex_v3.py |
| 1EA2A54085C9C5A4 | 2026-10-04 23:00 | src/toolkit | in-repo derived, codex/metricspec-v4: test_benchmark_hex_v3.py |

## Integration B1-decisions-E1-E6 (append-only, 2026-10-05, vero/hexa-integration-2026-10-05)

> Pins for the integrated bytes of this block. Prior rows remain lineage.

| 40161F27978DC8D3 | 2026-10-05 00:00 | scripts | in-repo derived, integration B1-decisions-E1-E6: engine_gate.py |
| AA0904D57AF71012 | 2026-10-05 00:00 | scripts | in-repo derived, integration B1-decisions-E1-E6: mutation_probe.py |
| 1FDD515C48BE308B | 2026-10-05 00:00 | scripts/oracles | in-repo derived, integration B1-decisions-E1-E6: dcp_2rank_gloo_oracle.py |
| 5E189A1FC5A73D07 | 2026-10-05 00:00 | scripts/oracles | in-repo derived, integration B1-decisions-E1-E6: dcp_bundle_2rank_oracle.py |
| 7C7806D9817628E9 | 2026-10-05 00:00 | scripts/oracles | in-repo derived, integration B1-decisions-E1-E6: dcp_worldsize_change_oracle.py |
| 29DFF31B34EFD286 | 2026-10-05 00:00 | scripts/oracles | in-repo derived, integration B1-decisions-E1-E6: distributed_checkpoint_bundle_v1.py |
| F33D9BBD8B056B3B | 2026-10-05 00:00 | scripts/oracles | in-repo derived, integration B1-decisions-E1-E6: stateful_loader_resume_oracle.py |
| 83500C7EEC1D5B12 | 2026-10-05 00:00 | scripts | in-repo derived, integration B1-decisions-E1-E6: verify_sources.py |
| B54757BA32C68E50 | 2026-10-05 00:00 | spec | in-repo derived, integration B1-decisions-E1-E6: HexaNTK_Review_Gehaertet.md |
| 854EBC8CB813E22E | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B1-decisions-E1-E6: resume_utils_hex_v3.py |
| 9A238491660B94CA | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B1-decisions-E1-E6: runner_hex_v3.py |
| A8DD8B9F987443AB | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B1-decisions-E1-E6: test_resume_utils_hex_v3.py |
| D52C7C4AA03DD105 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B1-decisions-E1-E6: test_runner_hex_v3.py |
| EEFE5FBCEEB67ADE | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B1-decisions-E1-E6: test_verify_sources_gate.py |

## Integration B1-decisions-E1-E6-final (append-only, 2026-10-05, vero/hexa-integration-2026-10-05)

> Pins for the integrated bytes of this block. Prior rows remain lineage.

| 36D3CBADC59CE6C3 | 2026-10-05 00:00 | scripts | in-repo derived, integration B1-decisions-E1-E6-final: verify_sources.py |

## Integration B2-B3-thread-fixes (append-only, 2026-10-05, vero/hexa-integration-2026-10-05)

> Pins for the integrated bytes of this block. Prior rows remain lineage.

| 02F3C8E1BE282F1C | 2026-10-05 00:00 | scripts/oracles | in-repo derived, integration B2-B3-thread-fixes: dcp_bundle_2rank_oracle.py |
| 688BCE7FF32501CA | 2026-10-05 00:00 | scripts/oracles | in-repo derived, integration B2-B3-thread-fixes: distributed_checkpoint_bundle_v1.py |
| B18E21FADC714D84 | 2026-10-05 00:00 | spec | in-repo derived, integration B2-B3-thread-fixes: METRIC_SPEC_v4.md |
| 319D9E0A8CB1697A | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B2-B3-thread-fixes: benchmark_hex_v3.py |
| A77713B3C2F450AB | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B2-B3-thread-fixes: checkpoint_integration_hex_v3.py |
| 4F1DE45D7416AAB8 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B2-B3-thread-fixes: runner_hex_v3.py |
| 45F2096FCADED21E | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B2-B3-thread-fixes: test_benchmark_hex_v3.py |
| 39C298784E6824E9 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B2-B3-thread-fixes: test_checkpoint_integration_hex_v3.py |
| 35931254F6B5D720 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B2-B3-thread-fixes: test_distributed_checkpoint_bundle_v1.py |
| 1A5842152C2CD33E | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B2-B3-thread-fixes: test_resume_utils_cov_hex_v3.py |
| ACDCAA5B545E7CEC | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration B2-B3-thread-fixes: test_runner_hex_v3.py |

## Integration R2-codex-pr58 (append-only, 2026-10-05, vero/hexa-integration-2026-10-05)

> Pins for the integrated bytes of this block. Prior rows remain lineage.

| 2922F97F47844325 | 2026-10-05 00:00 | scripts/oracles | in-repo derived, integration R2-codex-pr58: distributed_checkpoint_bundle_v1.py |
| 7B75D532ED7CE67A | 2026-10-05 00:00 | spec | in-repo derived, integration R2-codex-pr58: METRIC_SPEC_v4.md |
| BFDF3E26044116C0 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R2-codex-pr58: benchmark_hex_v3.py |
| 0F9EEA942316B6BA | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R2-codex-pr58: runner_hex_v3.py |
| E1B87B6B8B1D549E | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R2-codex-pr58: test_benchmark_hex_v3.py |
| 87144BAAED06222C | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R2-codex-pr58: test_distributed_checkpoint_bundle_v1.py |
| BB9F68C9BE2146D3 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R2-codex-pr58: test_runner_hex_v3.py |

## Integration R2-codex-pr58-path-unit (append-only, 2026-10-05, vero/hexa-integration-2026-10-05)

> Pins for the integrated bytes of this block. Prior rows remain lineage.

| 991FAA5046DEBD99 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R2-codex-pr58-path-unit: test_distributed_checkpoint_bundle_v1.py |

## Integration R2-codex-pr58-tests (append-only, 2026-10-05, vero/hexa-integration-2026-10-05)

> Pins for the integrated bytes of this block. Prior rows remain lineage.

| 36D725225BF19466 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R2-codex-pr58-tests: test_benchmark_hex_v3.py |
| CEE8332E93D341A9 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R2-codex-pr58-tests: test_distributed_checkpoint_bundle_v1.py |

## Integration R3-codex-pr58-round2 (append-only, 2026-10-05, vero/hexa-integration-2026-10-05)

> Pins for the integrated bytes of this block. Prior rows remain lineage.

| AF58C837A63847CD | 2026-10-05 00:00 | scripts/oracles | in-repo derived, integration R3-codex-pr58-round2: distributed_checkpoint_bundle_v1.py |
| 120C32D0DC287451 | 2026-10-05 00:00 | scripts/oracles | in-repo derived, integration R3-codex-pr58-round2: stateful_loader_resume_oracle.py |
| FCD004306D3AEB06 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R3-codex-pr58-round2: runner_hex_v3.py |
| 6C34E8424D5800F1 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R3-codex-pr58-round2: test_distributed_checkpoint_bundle_v1.py |
| FF8594FFD085E64D | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R3-codex-pr58-round2: test_resume_utils_hex_v3.py |
| D061AF66600A09C6 | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R3-codex-pr58-round2: test_runner_hex_v3.py |

## Integration R3-test-assertion (append-only, 2026-10-05, vero/hexa-integration-2026-10-05)

> Pins for the integrated bytes of this block. Prior rows remain lineage.

| 99FA3780E73F53DA | 2026-10-05 00:00 | src/toolkit | in-repo derived, integration R3-test-assertion: test_runner_hex_v3.py |

## Issue #59 follow-ups (append-only, 2026-10-07, claude/hexa-ntk-project-overview-mzkd4p)

> Pins for the bytes of this block. Prior rows remain lineage.

| E7A3BD3FC8159373 | 2026-10-07 00:00 | scripts | in-repo derived, issue #59 DCP content fsync, v4 export, float32 dt: mutation_probe.py |
| F765A1425BFA7CFE | 2026-10-07 00:00 | scripts/oracles | in-repo derived, issue #59 DCP content fsync, v4 export, float32 dt: distributed_checkpoint_bundle_v1.py |
| 51013971AA5CB76F | 2026-10-07 00:00 | spec | in-repo derived, issue #59 DCP content fsync, v4 export, float32 dt: METRIC_SPEC_v4.md |
| 412AA56028E8CDBF | 2026-10-07 00:00 | src/toolkit | in-repo derived, issue #59 DCP content fsync, v4 export, float32 dt: benchmark_hex_v3.py |
| C0076272235EDB25 | 2026-10-07 00:00 | src/toolkit | in-repo derived, issue #59 DCP content fsync, v4 export, float32 dt: dataset_split_hex_v3.py |
| 137812E571113CFC | 2026-10-07 00:00 | src/toolkit | in-repo derived, issue #59 DCP content fsync, v4 export, float32 dt: test_benchmark_hex_v3.py |
| 76572D45CAC3C78B | 2026-10-07 00:00 | src/toolkit | in-repo derived, issue #59 DCP content fsync, v4 export, float32 dt: test_dataset_split_hex_v3.py |
| B050C6EBDAD7E045 | 2026-10-07 00:00 | src/toolkit | in-repo derived, issue #59 DCP content fsync, v4 export, float32 dt: test_distributed_checkpoint_bundle_v1.py |


## src/toolkit/ — ReferenceSpec v1 2026-09-23 (append-only)

> New in-repo ReferenceSpec implementation and contract tests. SHA prefixes are
> filled from the fail-closed provenance gate before review readiness.

| 892D90A81D2DE830 | 2026-09-23 00:00 | src/toolkit | in-repo created, ReferenceSpec v1: reference_spec_hex_v3.py |
| 74158FD15466A368 | 2026-09-23 00:00 | src/toolkit | in-repo created, ReferenceSpec v1 tests: test_reference_spec_hex_v3.py |

## ReferenceSpec v1 integration (PR #33) (append-only, 2026-10-07, claude/hexa-ntk-project-overview-mzkd4p)

> Pins for the bytes of this block. Prior rows remain lineage.

| 1BE50B493049465C | 2026-10-07 00:00 | spec | in-repo derived, PR #33 ReferenceSpec v1 contract, merged into the issue #59 / Phase 0 branch: REFERENCE_SPEC_v1.md |
| 84A83BFDA8185621 | 2026-10-07 00:00 | src/toolkit | in-repo derived, PR #33 ReferenceSpec v1 contract at head 932e9f9 (the 2026-09-23 pins predate its last hardening commits): reference_spec_hex_v3.py |
| 8E5CFC60C7267FAC | 2026-10-07 00:00 | src/toolkit | in-repo derived, PR #33 ReferenceSpec v1 contract at head 932e9f9 (the 2026-09-23 pins predate its last hardening commits): test_reference_spec_hex_v3.py |

## Reality Anchor Phase 0a hex-heat-tri (#60) (append-only, 2026-10-07, claude/hexa-ntk-project-overview-mzkd4p)

> Pins for the bytes of this block. Prior rows remain lineage.

| B907BF72F3A9D6D3 | 2026-10-07 00:00 | spec | in-repo derived, Phase 0a reference case hex-heat-tri, W1-K08: REFERENCE_CASE_HEX_HEAT_TRI_v1.md |
| A2C90CF65D11EE6F | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase 0a reference case hex-heat-tri, W1-K08: reference_case_hex_v3.py |
| 7D5D372881BB4A20 | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase 0a reference case hex-heat-tri, W1-K08: test_reference_case_hex_v3.py |
| 5BE5EBA7E235317E | 2026-10-07 00:00 | scripts/oracles | in-repo derived, Phase 0a reference case hex-heat-tri, W1-K08: hex_heat_tri_reference_oracle.py |
| 4C43E5DAA853E381 | 2026-10-07 00:00 | scripts | in-repo derived, Phase 0a reference case hex-heat-tri, W1-K08: mutation_probe.py |

## Phase 0a self-review: Celik refinement ratio (append-only, 2026-10-07, claude/hexa-ntk-project-overview-mzkd4p)

> Pins for the bytes of this block. Prior rows remain lineage.

| E97EDC20216DC873 | 2026-10-07 00:00 | spec | in-repo derived, Phase 0a self-review, refinement ratio r > 1.3 (Celik 2008): REFERENCE_CASE_HEX_HEAT_TRI_v1.md |
| A84C3F4971B1B6AD | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase 0a self-review, refinement ratio r > 1.3 (Celik 2008): reference_case_hex_v3.py |
| 1B1D7386C8E3952A | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase 0a self-review, refinement ratio r > 1.3 (Celik 2008): test_reference_case_hex_v3.py |

## Reality Anchor Phase 0b: torus, MMS, solver layer (#60) (append-only, 2026-10-07, claude/hexa-ntk-project-overview-mzkd4p)

> Pins for the bytes of this block. Prior rows remain lineage.

| 2A4C4D0DFDA9B63A | 2026-10-07 00:00 | scripts | in-repo derived, Phase 0b hex-heat-torus + hex-mms-tri + hex FV solver layer, W1-K08: mutation_probe.py |
| 0CC4857820C31E00 | 2026-10-07 00:00 | scripts/oracles | in-repo derived, Phase 0b hex-heat-torus + hex-mms-tri + hex FV solver layer, W1-K08: hex_phase0_reference_oracle.py |
| 7B01D64FB125C95C | 2026-10-07 00:00 | spec | in-repo derived, Phase 0b hex-heat-torus + hex-mms-tri + hex FV solver layer, W1-K08: HEX_FV_SOLVER_v1.md |
| 591AF4505C676D73 | 2026-10-07 00:00 | spec | in-repo derived, Phase 0b hex-heat-torus + hex-mms-tri + hex FV solver layer, W1-K08: REFERENCE_CASES_HEX_PHASE0B_v1.md |
| 4D90EEA7AECB7913 | 2026-10-07 00:00 | spec | in-repo derived, Phase 0b hex-heat-torus + hex-mms-tri + hex FV solver layer, W1-K08: REFERENCE_CASE_HEX_HEAT_TRI_v1.md |
| E73B0159AD87B43A | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase 0b hex-heat-torus + hex-mms-tri + hex FV solver layer, W1-K08: hex_fv_solver_hex_v3.py |
| CA7F77E711CF94FC | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase 0b hex-heat-torus + hex-mms-tri + hex FV solver layer, W1-K08: test_hex_fv_solver_hex_v3.py |
| 6E97268A81CF9B9D | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase 0b hex-heat-torus + hex-mms-tri + hex FV solver layer, W1-K08: reference_case_hex_v3.py |
| 932549A2E8F158FC | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase 0b hex-heat-torus + hex-mms-tri + hex FV solver layer, W1-K08: test_reference_case_hex_v3.py |

## Phase 1 instruments: Crank-Nicolson, T1 task, preregistration draft (#60) (append-only, 2026-10-07, claude/hexa-ntk-project-overview-mzkd4p)

> Pins for the bytes of this block. Prior rows remain lineage.

| F9D54B4D541B71ED | 2026-10-07 00:00 | scripts | in-repo derived, Phase 1 instruments (CN baseline, T1 hex-heat-operator) and preregistration DRAFT, W1-K08: mutation_probe.py |
| 7AF057326D34E5D9 | 2026-10-07 00:00 | spec | in-repo derived, Phase 1 instruments (CN baseline, T1 hex-heat-operator) and preregistration DRAFT, W1-K08: HEX_FV_SOLVER_v1.md |
| CFED3030667A3155 | 2026-10-07 00:00 | spec | in-repo derived, Phase 1 instruments (CN baseline, T1 hex-heat-operator) and preregistration DRAFT, W1-K08: HEAT_OPERATOR_TASK_v1.md |
| 599E9065E77427F7 | 2026-10-07 00:00 | spec | in-repo derived, Phase 1 instruments (CN baseline, T1 hex-heat-operator) and preregistration DRAFT, W1-K08: PREREGISTRATION_PHASE1_v1.md |
| C79474FB3B42A241 | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase 1 instruments (CN baseline, T1 hex-heat-operator) and preregistration DRAFT, W1-K08: hex_fv_solver_hex_v3.py |
| 1FC5DA42498FCF93 | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase 1 instruments (CN baseline, T1 hex-heat-operator) and preregistration DRAFT, W1-K08: test_hex_fv_solver_hex_v3.py |
| C11A3434A3C95697 | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase 1 instruments (CN baseline, T1 hex-heat-operator) and preregistration DRAFT, W1-K08: heat_operator_task_hex_v3.py |
| 57B53E08ECBFAE83 | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase 1 instruments (CN baseline, T1 hex-heat-operator) and preregistration DRAFT, W1-K08: test_heat_operator_task_hex_v3.py |

## Independent PR #61 hardening — GPT-5.6 Sol (append-only, 2026-10-07)

> Independent review of exact head 1acca25: recompute ReferenceSpec observed
> order, bind T1 lattice/coefficients into dataset integrity, fail closed on
> negative H1 energy, and remove the confounded pure-BC label from the torus
> OOD axis. Prior rows remain lineage.

| F386BDF9F14F6022 | 2026-10-07 00:00 | spec | in-repo derived, independent PR #61 credibility hardening: REFERENCE_SPEC_v1.md |
| 84DAB72D604BCA2B | 2026-10-07 00:00 | src/toolkit | in-repo derived, independent PR #61 credibility hardening: reference_spec_hex_v3.py |
| 012B910B89916082 | 2026-10-07 00:00 | src/toolkit | in-repo derived, independent PR #61 credibility hardening: test_reference_spec_hex_v3.py |
| C441BBE421F690A7 | 2026-10-07 00:00 | spec | in-repo derived, independent PR #61 T1 evidence hardening: HEAT_OPERATOR_TASK_v1.md |
| 9EBBA2DBC6724B91 | 2026-10-07 00:00 | spec | in-repo derived, independent PR #61 preregistration hardening: PREREGISTRATION_PHASE1_v1.md |
| 8142EE80BA07B24E | 2026-10-07 00:00 | src/toolkit | in-repo derived, independent PR #61 T1 evidence hardening: heat_operator_task_hex_v3.py |
| 9905598D73DAB0B1 | 2026-10-07 00:00 | src/toolkit | in-repo derived, independent PR #61 T1 evidence hardening: test_heat_operator_task_hex_v3.py |
| E8BA656865FF9032 | 2026-10-07 00:00 | spec | in-repo derived, independent PR #61 formula-text correction after CI review: REFERENCE_SPEC_v1.md |

## dt grouping quantisation contract (#59) — GPT-5.6 Sol (append-only, 2026-10-07)

> Close the ambiguous negative-ndigits path: decimal quantisation is None or a
> non-negative number of decimal places. Prior rows remain lineage.

| 3C9695B66E17255F | 2026-10-07 00:00 | src/toolkit | in-repo derived, issue #59 dt quantisation contract: dataset_split_hex_v3.py |
| 64C9F1BDB1FD7B49 | 2026-10-07 00:00 | src/toolkit | in-repo derived, issue #59 dt quantisation contract: test_dataset_split_hex_v3.py |

> **2026-10-12, PR #61 P1 (CG stopping rule).** `conjugate_gradient` now decides
> `||b - A x||_2 <= rtol ||b||_2` on the true residual (spec/HEX_FV_SOLVER_v1.md
> section 5). Consequence: Phase 0b I1 fails on N=192 (true residual stagnates
> at 2.34e-13 > 1e-13); the threshold is unchanged (owner decision). Prior rows
> remain lineage.

| CAB44CD8B761E223 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 P1 CG true-residual stopping rule (spec section 5): hex_fv_solver_hex_v3.py |
| 8A70864BFFFCE627 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 P1 CG true-residual stopping rule (spec section 5): test_hex_fv_solver_hex_v3.py |
| 0F496A82B8D39E49 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 P1 CG true-residual stopping rule (spec section 5): reference_case_hex_v3.py |
| A534D04BAAA1075B | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 P1 CG true-residual stopping rule (spec section 5): test_reference_case_hex_v3.py |
| 2CC63B7B63B566D3 | 2026-10-12 00:00 | spec | in-repo derived, 2026-10-12 PR #61 P1 CG true-residual stopping rule (spec section 5): HEX_FV_SOLVER_v1.md |
| E7E3849689E6F44B | 2026-10-12 00:00 | scripts | in-repo derived, 2026-10-12 PR #61 P1 CG true-residual stopping rule (spec section 5): mutation_probe.py |
| A0E8AFC37C6782D1 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 P1 test hardening (crash is an assertion failure): test_reference_case_hex_v3.py |
| BD4C5559B4B4B6CD | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 reject non-positive conductance arrays: hex_fv_solver_hex_v3.py |
| BB64FA00F04987A9 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 reject non-positive conductance arrays: test_hex_fv_solver_hex_v3.py |
| 6D37E04541C52F33 | 2026-10-12 00:00 | spec | in-repo derived, 2026-10-12 PR #61 Codex P2 reject non-positive conductance arrays: HEX_FV_SOLVER_v1.md |
| 2B1529B0FB815450 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 overflow-safe CG rhs norm (exact power-of-two scaling): hex_fv_solver_hex_v3.py |
| CEEE96C6CA0BD802 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 overflow-safe CG rhs norm (exact power-of-two scaling): test_hex_fv_solver_hex_v3.py |
| AFA0C6BFE9827222 | 2026-10-12 00:00 | spec | in-repo derived, 2026-10-12 PR #61 Codex P2 overflow-safe CG rhs norm (exact power-of-two scaling): HEX_FV_SOLVER_v1.md |
| 3ACF36EB34286B1A | 2026-10-12 00:00 | scripts | in-repo derived, 2026-10-12 PR #61 Codex P2 overflow-safe CG rhs norm (exact power-of-two scaling): mutation_probe.py |
| 03C9A61A6B74D006 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 GPT6 W9-F01 canonical lattice contract: hex_fv_solver_hex_v3.py |
| 22885F2D2B0CEB3A | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 GPT6 W9-F01 canonical lattice contract: test_hex_fv_solver_hex_v3.py |
| 11815B955B7ADDA5 | 2026-10-12 00:00 | spec | in-repo derived, 2026-10-12 PR #61 GPT6 W9-F01 canonical lattice contract: HEX_FV_SOLVER_v1.md |
| 675C90BC29E4AC30 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 per-sample H1 negative-energy band: heat_operator_task_hex_v3.py |
| 8FA2B49EDEAA92A6 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 per-sample H1 negative-energy band: test_heat_operator_task_hex_v3.py |
| 796B7DF840C87D64 | 2026-10-12 00:00 | scripts | in-repo derived, 2026-10-12 PR #61 Codex P2 M49 retargeted, per-sample band mutant: mutation_probe.py |
| DD05384B671624BE | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 test hardening (failed rescaled solve is an assertion failure): test_hex_fv_solver_hex_v3.py |
| 7711B0D09ADF1C45 | 2026-10-12 00:00 | scripts/oracles | in-repo derived, 2026-10-12 PR #61 Codex P2 evidence writers fail closed on an unbound source revision: hex_heat_tri_reference_oracle.py |
| 1C0CD9DC2F08626F | 2026-10-12 00:00 | scripts/oracles | in-repo derived, 2026-10-12 PR #61 Codex P2 evidence writers fail closed on an unbound source revision: hex_phase0_reference_oracle.py |
| 7338EDC5172914B4 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 evidence writers fail closed on an unbound source revision: test_reference_oracle_source_binding.py |

| D5D9EF2923B2C07C | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 underflow-safe true CG residual norm: hex_fv_solver_hex_v3.py |
| 823A6B4BC89BD24C | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 underflow-safe true CG residual norm: test_hex_fv_solver_hex_v3.py |
| D8CBCF33A208AF2B | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 metric inputs exactly [S, M], scaled L2/max norms: heat_operator_task_hex_v3.py |
| 3F1667E3CE9AA1C9 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 metric inputs exactly [S, M], scaled L2/max norms: test_heat_operator_task_hex_v3.py |
| 723931AABC8DD449 | 2026-10-12 00:00 | spec | in-repo derived, 2026-10-12 PR #61 Codex P2 metric inputs exactly [S, M], scaled L2/max norms: HEAT_OPERATOR_TASK_v1.md |
| 4063390225430176 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 manifest task_id checked, sample ids and seed exact: heat_operator_task_hex_v3.py |
| 833FF2BEECC507B3 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 manifest task_id checked, sample ids and seed exact: test_heat_operator_task_hex_v3.py |
| 495893B4DE75E0D6 | 2026-10-12 00:00 | spec | in-repo derived, 2026-10-12 PR #61 Codex P2 manifest task_id checked, sample ids and seed exact: HEAT_OPERATOR_TASK_v1.md |
| AF7D1DFFFEA29988 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 oscillatory study serialised as negative evidence: reference_case_hex_v3.py |
| 3925442C24A56C39 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 oscillatory study serialised as negative evidence: test_reference_case_hex_v3.py |
| BA52937C53F030BE | 2026-10-12 00:00 | spec | in-repo derived, 2026-10-12 PR #61 Codex P2 oscillatory study serialised as negative evidence: REFERENCE_CASE_HEX_HEAT_TRI_v1.md |
| 008ECFEDAA323F20 | 2026-10-12 00:00 | spec | in-repo derived, 2026-10-12 PR #61 Codex P2 CG underflow rule; measured residual replacement keeping p rejected: HEX_FV_SOLVER_v1.md |
| FB8C074236D46EB4 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 PR #61 Codex P2 underflow test fails by assertion for any exception type: test_hex_fv_solver_hex_v3.py |

> **2026-10-12, E1012-07 = A (CG stopping rule V2).** New spec
> `spec/HEX_FV_SOLVER_v2.md`: accept `||b - A x|| <= tol_eff ||b||`,
> `tol_eff = max(rtol, c u (|| |A||x| || + ||b||) / ||b||)`, `c = 8` for the hex
> operator, status `converged` / `at_rounding_floor`, backward error reported.
> Phase 0b I1 is judged by V2; the V1 rule stays the solver default and the V1
> verdict of I1 (red on N=192) stays as pinned negative evidence. The I1
> mutant of `mutation_probe.py` follows the rule into `_i1_level_holds`. Prior
> rows remain lineage.

| 95F135F92D66644F | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 E1012-07 CG stopping rule V2 (rounding floor), Phase 0b I1 judged by V2, V1 kept as negative evidence: hex_fv_solver_hex_v3.py |
| 8162854050D99784 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 E1012-07 CG stopping rule V2 (rounding floor), Phase 0b I1 judged by V2, V1 kept as negative evidence: test_hex_fv_solver_hex_v3.py |
| BF172F48069DCE33 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 E1012-07 CG stopping rule V2 (rounding floor), Phase 0b I1 judged by V2, V1 kept as negative evidence: reference_case_hex_v3.py |
| 2728169C296B2886 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 E1012-07 CG stopping rule V2 (rounding floor), Phase 0b I1 judged by V2, V1 kept as negative evidence: test_reference_case_hex_v3.py |
| E70C1C24727BE491 | 2026-10-12 00:00 | spec | in-repo derived, 2026-10-12 E1012-07 CG stopping rule V2 (rounding floor), Phase 0b I1 judged by V2, V1 kept as negative evidence: HEX_FV_SOLVER_v1.md |
| 025012A9B9F8E024 | 2026-10-12 00:00 | spec | in-repo new, 2026-10-12 E1012-07 CG stopping rule V2 (rounding floor), Phase 0b I1 judged by V2, V1 kept as negative evidence: HEX_FV_SOLVER_v2.md |
| 34B1815ACAD6ECB2 | 2026-10-12 00:00 | spec | in-repo derived, 2026-10-12 E1012-07 CG stopping rule V2 (rounding floor), Phase 0b I1 judged by V2, V1 kept as negative evidence: REFERENCE_CASES_HEX_PHASE0B_v1.md |
| 9B689DCC93F9B9F2 | 2026-10-12 00:00 | scripts | in-repo derived, 2026-10-12 E1012-07 CG stopping rule V2 (rounding floor), Phase 0b I1 judged by V2, V1 kept as negative evidence: mutation_probe.py |
| 725151FC09B80F9C | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 E1012-07 V2 non-finite floor test fails by assertion for any exception type: test_hex_fv_solver_hex_v3.py |
| F70F0D7AB9D8444E | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 E1012-07 I1 level field cg_converged only where rtol was reached, cg_returned for I1: reference_case_hex_v3.py |
| 71769EC73D62DE94 | 2026-10-12 00:00 | src/toolkit | in-repo derived, 2026-10-12 E1012-07 I1 level field cg_converged only where rtol was reached, cg_returned for I1: test_reference_case_hex_v3.py |
| E6BE1CB1FEF2FFD7 | 2026-10-13 00:00 | src/toolkit | in-repo new, 2026-10-13 spectral audit B1 (input Jacobian projected on the T1 modes, fail-closed): spectral_audit_hex_v3.py |
| 0D37A6EDD1C091C9 | 2026-10-13 00:00 | src/toolkit | in-repo new, 2026-10-13 spectral audit B1 (input Jacobian projected on the T1 modes, fail-closed): test_spectral_audit_hex_v3.py |
| 763B5F2160112190 | 2026-10-13 00:00 | src/toolkit | in-repo new, 2026-10-13 spectral audit B2 (oracles: semidiscrete semigroup, Crank-Nicolson, negative controls, metamorphic): test_spectral_audit_oracles_hex_v3.py |
| 880A7A1923CBBC0D | 2026-10-13 00:00 | src/toolkit | in-repo derived, 2026-10-13 spectral audit DK-SA03/DK-SA09 were BLIND: base-point NaN test, coupling-only verdict: test_spectral_audit_hex_v3.py |
| EA13FFBFE0DEE643 | 2026-10-13 00:00 | src/toolkit | in-repo derived, 2026-10-13 spectral audit DK-SA03/DK-SA09 were BLIND: base-point NaN test, coupling-only verdict: test_spectral_audit_oracles_hex_v3.py |
| 28D86FE9CC41299B | 2026-10-13 00:00 | src/toolkit | in-repo new, 2026-10-13 B3 linear POD-ridge baseline with leakage guard, checked by the spectral audit: linear_baseline_hex_v3.py |
| 35F70875438E85C4 | 2026-10-13 00:00 | src/toolkit | in-repo new, 2026-10-13 B3 linear POD-ridge baseline with leakage guard, checked by the spectral audit: test_linear_baseline_hex_v3.py |
| 76D0289C9F3E1531 | 2026-10-13 00:00 | src/toolkit | in-repo derived, 2026-10-13 DK-LB04 was red by IndexError only: all-zero training test fails by assertion: test_linear_baseline_hex_v3.py |
| 02C803DBBE6DEC98 | 2026-10-13 00:00 | src/toolkit | in-repo derived, 2026-10-13 B4 citation narrowed to Higham 2002 section 3.1: spectral_audit_hex_v3.py |
| 996CB2636D266724 | 2026-10-13 00:00 | spec | in-repo new, 2026-10-13 B4 spectral audit and linear POD-ridge baseline contract (secondary instrument, not preregistered): SPECTRAL_AUDIT_v1.md |
| 3EC5B4A4B6B3C5DA | 2026-10-13 13:13 | spec | in-repo derived, owner decision 2026-10-13 (spectral audit adopted as secondary measurement): PREREGISTRATION_PHASE1_v1.md |
| 29C3C1737A70F951 | 2026-10-13 13:13 | spec | in-repo derived, owner decision 2026-10-13 (spectral audit adopted as secondary measurement): SPECTRAL_AUDIT_v1.md |
## Phase-1 external GINO adapter v1 — GPT-5.6 Sol (append-only, 2026-10-07)

> Optional neuraloperator==2.0.0 adapter, synthetic forward smoke and explicit
> geometry/claim contract. No training or held-out test access.

| BF4314F84A2CC2A0 | 2026-10-07 00:00 | spec | in-repo derived, Phase-1 external GINO adapter contract: GINO_ADAPTER_PHASE1_v1.md |
| 1C275A6198354313 | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase-1 external GINO adapter: gino_adapter_hex_v3.py |
| FBF08A0858F074F0 | 2026-10-07 00:00 | src/toolkit | in-repo derived, Phase-1 GINO adapter contract/smoke tests: test_gino_adapter_hex_v3.py |

## Phase-1 GINO adapter — repo-gate SHA correction (append-only, 2026-10-07)

> Authoritative prefixes below are the values computed by
> scripts/verify_sources.py from the GitHub checkout. They supersede the
> earlier connector-side pre-write hashes without deleting lineage.

| EED780CF4C2653DE | 2026-10-07 00:00 | spec | repo-gate verified current bytes: GINO_ADAPTER_PHASE1_v1.md |
| 953D456F86FC9276 | 2026-10-07 00:00 | src/toolkit | repo-gate verified current bytes: gino_adapter_hex_v3.py |
| 386F8AD29D21BE33 | 2026-10-07 00:00 | src/toolkit | repo-gate verified current bytes: test_gino_adapter_hex_v3.py |


## Independent GINO adapter review hardening (2026-10-11; stacked on PR #62)

> Append-only pin for the canonical lattice guard, actual backend-version validation,
> N=32 fixed radius, geometric and config digests in tests.

| 198BC9DEA76DE6B2 | 2026-10-11 00:00 | src/toolkit | in-repo derived, independent GINO review hardening: gino_adapter_hex_v3.py |
| 238B26A645BD6656 | 2026-10-11 00:00 | src/toolkit | in-repo derived, independent GINO review hardening: test_gino_adapter_hex_v3.py |
| 4A193E9774BB32DA | 2026-10-11 00:01 | src/toolkit | in-repo derived, Wave17 CI Ruff E501 regression correction: test_gino_adapter_hex_v3.py |


## Independent Wave18 GINO device/dtype and synthetic gradient hardening (2026-10-11)

> Appended on reviewed PR63 head, no V1 MMS/preregistrated training changes.

| 650AED2652C23789 | 2026-10-11 03:45 | src/toolkit | in-repo derived, Wave18 device/dtype/autograd and synthetic optimizer regression: gino_adapter_hex_v3.py |
| 74196F46C89AF814 | 2026-10-11 03:45 | src/toolkit | in-repo derived, Wave18 device/dtype/autograd and synthetic optimizer regression: test_gino_adapter_hex_v3.py |
| 20451528C1C82C41 | 2026-10-11 04:45 | src/toolkit | in-repo derived, Wave19 bounded early GINO lattice preflight: gino_adapter_hex_v3.py |
| 7C7EF604A3E2BD7D | 2026-10-11 04:45 | src/toolkit | in-repo derived, Wave19 bounded early GINO lattice preflight: test_gino_adapter_hex_v3.py |

## Wave19 resource-budget hardening final bytes
| 512A666E4213BE26 | 2026-10-11 04:50 | src/toolkit | in-repo derived, Wave19 final budgeted GINO preflight: gino_adapter_hex_v3.py |
| 8F37B365FE332FCA | 2026-10-11 04:50 | src/toolkit | in-repo derived, Wave19 final budgeted GINO preflight: test_gino_adapter_hex_v3.py |
| 9F91DCBDC101FD94 | 2026-10-11 05:00 | src/toolkit | in-repo derived, Wave20 combined spectral-memory guard: gino_adapter_hex_v3.py |
| 09ED11DAEB4341A7 | 2026-10-11 05:00 | src/toolkit | in-repo derived, Wave20 regression tests: test_gino_adapter_hex_v3.py |
| BFC8D4BD243E2982 | 2026-10-11 05:00 | spec | in-repo derived, Wave20 explicit operational v2 contract: GINO_ADAPTER_PHASE1_v2.md |
| D3F24028185AEF60 | 2026-10-13 10:40 | src/toolkit | in-repo derived, float64/complex-dtype guard after real neuraloperator 2.0.0 run: gino_adapter_hex_v3.py |
| 67119E4E5E380C72 | 2026-10-13 10:40 | src/toolkit | in-repo derived, real-backend float64 refusal + complex-dtype tests: test_gino_adapter_hex_v3.py |
| 3ED18F2893BD0265 | 2026-10-13 10:40 | spec | in-repo derived, dtype contract + radius-tie error class (measured): GINO_ADAPTER_PHASE1_v2.md |

## CG V2 tests: BLAS-kernel-independent premise (append-only, 2026-10-14)

> PR #61 head 6d219d9 was red in `tests (cpu)`: two V2 tests assumed rtol=1e-16
> is unreachable for the n=12 system, which holds on the SkylakeX/Sandybridge
> OpenBLAS kernels but not on Haswell/Zen/Prescott (the CI run failed the same way).
> Prior rows remain lineage.

| D58885C8EA67AE42 | 2026-10-14 00:00 | src/toolkit | in-repo derived, 2026-10-14 CG V2 tests use _RTOL_BELOW_ROUNDING=1e-17 (kernel-independent stagnation premise), scaling test pins status: test_hex_fv_solver_hex_v3.py |

| F80BB66072969523 | 2026-10-15 00:00 | spec | in-repo derived, 2026-10-15 E1015M-01 D5 = CPU entered from the approved source, D4 grid and D5 machine still open, status stays DRAFT: PREREGISTRATION_PHASE1_v1.md |

> **2026-10-15, hardening negative-first (B3a, B4, B9, B6).** GINO adapter: float64
> refusal over every submodule; runtime radius-tie refusal in build_gino and
> prepare_gino_batch. POD-ridge: content leakage via canonical row hashes
> (-0.0 -> 0.0). Spectral audit: determinism preflight (two forwards bitwise
> equal, global RNG unchanged), counted in the evaluation budget. Prior rows
> remain lineage.

| E209AE93528788AE | 2026-10-15 00:00 | src/toolkit | in-repo derived, 2026-10-15 hardening B3a/B4/B9/B6 (negative-first, mutants in results/): gino_adapter_hex_v3.py |
| 8625301D301E8588 | 2026-10-15 00:00 | src/toolkit | in-repo derived, 2026-10-15 hardening B3a/B4/B9/B6 (negative-first, mutants in results/): test_gino_adapter_hex_v3.py |
| E06B6E507C83A789 | 2026-10-15 00:00 | src/toolkit | in-repo derived, 2026-10-15 hardening B3a/B4/B9/B6 (negative-first, mutants in results/): linear_baseline_hex_v3.py |
| 65676123D70A84DE | 2026-10-15 00:00 | src/toolkit | in-repo derived, 2026-10-15 hardening B3a/B4/B9/B6 (negative-first, mutants in results/): test_linear_baseline_hex_v3.py |
| CCDEFBC9270C645A | 2026-10-15 00:00 | src/toolkit | in-repo derived, 2026-10-15 hardening B3a/B4/B9/B6 (negative-first, mutants in results/): spectral_audit_hex_v3.py |
| 7EC0CFA6715BC616 | 2026-10-15 00:00 | src/toolkit | in-repo derived, 2026-10-15 hardening B3a/B4/B9/B6 (negative-first, mutants in results/): test_spectral_audit_hex_v3.py |
| 1314D4BF9583957B | 2026-10-15 00:00 | src/toolkit | in-repo derived, 2026-10-15 hardening B3a/B4/B9/B6 (negative-first, mutants in results/): test_spectral_audit_oracles_hex_v3.py |
| 6DA42EA6DFEDF763 | 2026-10-15 00:00 | spec | in-repo derived, 2026-10-15 hardening B3a/B4/B9/B6 (negative-first, mutants in results/): GINO_ADAPTER_PHASE1_v2.md |
| D1D3D3B2D2DB39A7 | 2026-10-15 00:00 | spec | in-repo derived, 2026-10-15 hardening B3a/B4/B9/B6 (negative-first, mutants in results/): SPECTRAL_AUDIT_v1.md |
| 2EF9B114873DAFF8 | 2026-10-16 | src/toolkit | Codex R on #61 dd24538 (V2 last-iterate, canonical lattice, dataset [S,M], GINO config binding): hex_fv_solver_hex_v3.py |
| 6D8721D56E9B9942 | 2026-10-16 | src/toolkit | Codex R on #61 dd24538 (V2 last-iterate, canonical lattice, dataset [S,M], GINO config binding): test_hex_fv_solver_hex_v3.py |
| 5F14BC566BD97CE8 | 2026-10-16 | src/toolkit | Codex R on #61 dd24538 (V2 last-iterate, canonical lattice, dataset [S,M], GINO config binding): heat_operator_task_hex_v3.py |
| C2E9300E82426EA8 | 2026-10-16 | src/toolkit | Codex R on #61 dd24538 (V2 last-iterate, canonical lattice, dataset [S,M], GINO config binding): test_heat_operator_task_hex_v3.py |
| A18A6C82BB0DC292 | 2026-10-16 | src/toolkit | Codex R on #61 dd24538 (V2 last-iterate, canonical lattice, dataset [S,M], GINO config binding): gino_adapter_hex_v3.py |
| 55DCBCF24EFAB6D4 | 2026-10-16 | src/toolkit | Codex R on #61 dd24538 (V2 last-iterate, canonical lattice, dataset [S,M], GINO config binding): test_gino_adapter_hex_v3.py |

## PR #75 engine-ownership wording (append-only, 2026-10-14, fix/public-metadata-claims)

> Docstring only: the engine's maintenance location is open (Issue #74). Prior rows remain lineage.

| 54464888B78D897F | 2026-10-14 00:00 | scripts | in-repo derived, PR #75 engine-ownership wording: engine_gate.py |

| 73DB4059487654BA | 2026-10-16 | src/toolkit | merge #61 1b9c2c4 + CN oracle tolerance derived from CG rtol (E5): gino_adapter_hex_v3.py |
| 5822365C94CED266 | 2026-10-16 | src/toolkit | merge #61 1b9c2c4 + CN oracle tolerance derived from CG rtol (E5): test_gino_adapter_hex_v3.py |
| 3176497F2398A499 | 2026-10-16 | src/toolkit | merge #61 1b9c2c4 + CN oracle tolerance derived from CG rtol (E5): test_spectral_audit_oracles_hex_v3.py |

| BEAD5188822CAE4A | 2026-10-16 | src/toolkit | PARAM_NTK_v1 (E1015M-03): parameter NTK, float64 shim + float32 cross-check: param_ntk_hex_v3.py |
| EB2540E4E6DF9FAC | 2026-10-16 | src/toolkit | PARAM_NTK_v1 (E1015M-03): parameter NTK, float64 shim + float32 cross-check: test_param_ntk_hex_v3.py |
| 08BA7746C3BBE57A | 2026-10-16 | src/toolkit | PARAM_NTK_v1 (E1015M-03): parameter NTK, float64 shim + float32 cross-check: test_gino_adapter_hex_v3.py |
| 0FFA4E5D26AEDAF1 | 2026-10-16 | spec | PARAM_NTK_v1 (E1015M-03): parameter NTK, float64 shim + float32 cross-check: PARAM_NTK_v1.md |

| 0409908D3F5BB815 | 2026-10-16 | src/toolkit | Codex R on #71 0372a0f (param NTK: eval/orthonormal/device; ridge scaling; scaled spectral norms; spec status): param_ntk_hex_v3.py |
| 02CA14BCB98BA89D | 2026-10-16 | src/toolkit | Codex R on #71 0372a0f (param NTK: eval/orthonormal/device; ridge scaling; scaled spectral norms; spec status): test_param_ntk_hex_v3.py |
| 3CB31A92B9ED1DD7 | 2026-10-16 | src/toolkit | Codex R on #71 0372a0f (param NTK: eval/orthonormal/device; ridge scaling; scaled spectral norms; spec status): linear_baseline_hex_v3.py |
| CC76BABD1ABCEB68 | 2026-10-16 | src/toolkit | Codex R on #71 0372a0f (param NTK: eval/orthonormal/device; ridge scaling; scaled spectral norms; spec status): test_linear_baseline_hex_v3.py |
| 12AD7404789A08D5 | 2026-10-16 | src/toolkit | Codex R on #71 0372a0f (param NTK: eval/orthonormal/device; ridge scaling; scaled spectral norms; spec status): spectral_audit_hex_v3.py |
| E2F65F122FE10146 | 2026-10-16 | src/toolkit | Codex R on #71 0372a0f (param NTK: eval/orthonormal/device; ridge scaling; scaled spectral norms; spec status): test_spectral_audit_hex_v3.py |
| 83BC52A73A92B66F | 2026-10-16 | spec | Codex R on #71 0372a0f (param NTK: eval/orthonormal/device; ridge scaling; scaled spectral norms; spec status): SPECTRAL_AUDIT_v1.md |

## PR #71 FD check on the kernel rows (append-only, 2026-10-17)

> Pins for the bytes of this block. Prior rows remain lineage.

| A07219EA8B48FAFC | 2026-10-17 00:00 | src/toolkit | FD check validates the kernel rows mode by mode (007acc2 v6 finding, 2026-10-17): param_ntk_hex_v3.py |
| 2D71C953BB65872E | 2026-10-17 00:00 | src/toolkit | FD check validates the kernel rows mode by mode (007acc2 v6 finding, 2026-10-17): test_param_ntk_hex_v3.py |

## Issue #76 Codex follow-up to #71 (append-only, 2026-10-17)

> Pins for the bytes of this block. Prior rows remain lineage.

| 9E5E59D9397613A3 | 2026-10-17 12:00 | src/toolkit | Codex follow-up to #71 (issue #76): linear_baseline_hex_v3.py |
| 47C2D0A9D4B6D514 | 2026-10-17 12:00 | src/toolkit | Codex follow-up to #71 (issue #76): param_ntk_hex_v3.py |
| 5EFC3E34567E1F28 | 2026-10-17 12:00 | src/toolkit | Codex follow-up to #71 (issue #76): spectral_audit_hex_v3.py |
| 1013013F44771EDD | 2026-10-17 12:00 | src/toolkit | Codex follow-up to #71 (issue #76): test_linear_baseline_hex_v3.py |
| EFE1C5D512872227 | 2026-10-17 12:00 | src/toolkit | Codex follow-up to #71 (issue #76): test_param_ntk_hex_v3.py |
| BA8531392DA74D11 | 2026-10-17 12:00 | src/toolkit | Codex follow-up to #71 (issue #76): test_spectral_audit_hex_v3.py |
| F37C83501D40A4A8 | 2026-10-17 12:00 | spec | Codex follow-up to #71 (issue #76): PARAM_NTK_v1.md |
| FE8BE1A0DD5372F3 | 2026-10-17 12:00 | spec | Codex follow-up to #71 (issue #76): SPECTRAL_AUDIT_v1.md |

## PR #77 Codex round (append-only, 2026-10-17)

> Pins for the bytes of this block. Prior rows remain lineage.

| 56E08D448D0D8312 | 2026-10-17 13:00 | src/toolkit | Codex round on #77 (torch_device under auto, alpha underflow): linear_baseline_hex_v3.py |
| DEE4F9877D3084B6 | 2026-10-17 13:00 | src/toolkit | Codex round on #77 (torch_device under auto, alpha underflow): spectral_audit_hex_v3.py |
| 2B2B46B400902638 | 2026-10-17 13:00 | src/toolkit | Codex round on #77 (torch_device under auto, alpha underflow): test_linear_baseline_hex_v3.py |
| 0E250206AC0EF5E9 | 2026-10-17 13:00 | src/toolkit | Codex round on #77 (torch_device under auto, alpha underflow): test_spectral_audit_hex_v3.py |

## Phase 1 protocol instruments + D4 radius-tie survey (append-only, 2026-10-19, claude/hexa-ntk-neural-operator-sh43u7)

> Pins for the bytes of this block. Prior rows remain lineage.

| 5C684F9C961FF410 | 2026-10-19 00:00 | src/toolkit | in-repo created, Phase 1 protocol instruments (spec/PHASE1_PROTOCOL_v1.md): phase1_protocol_hex_v3.py |
| C05A442FA9716B6D | 2026-10-19 00:00 | src/toolkit | in-repo created, oracles/negative controls for the protocol instruments: test_phase1_protocol_hex_v3.py |
| F54A49A15D157F5F | 2026-10-19 00:00 | spec | in-repo created, contract for preregistration sections 4-7: PHASE1_PROTOCOL_v1.md |
| B63BAD75B28B4D96 | 2026-10-19 00:00 | spec | cross-reference to PHASE1_PROTOCOL_v1 + D4 radius-tie decision support (no decision, status DRAFT unchanged): PREREGISTRATION_PHASE1_v1.md |
| 92C3031ECC5476D4 | 2026-10-19 00:00 | spec | amendment 2026-10-19 radius-tie survey (decision support only): GINO_ADAPTER_PHASE1_v2.md |
| D4D36437B9A9B2B7 | 2026-10-19 00:00 | scripts/oracles | in-repo created, D4 decision support with the adapter tie rule: gino_radius_tie_survey.py |

## Phase 1 protocol tests: asymmetric latency designs (append-only, 2026-10-19, claude/hexa-ntk-neural-operator-sh43u7)

> Pin for the bytes of this block. Prior rows remain lineage.

| 1ED5FF0D1DD2F535 | 2026-10-19 12:00 | src/toolkit | mutation probe in git clones (M09 median->mean survived a symmetric design): asymmetric latency designs: test_phase1_protocol_hex_v3.py |

## Phase 1 protocol review fixes (append-only, 2026-10-20, claude/hexa-ntk-neural-operator-sh43u7)

> Pins for the bytes of each fix block (Equalita review of f1b54bc). Prior rows remain lineage; the last row per file is authoritative.

| 77321258A0FF2A58 | 2026-10-20 10:45 | src/toolkit | F-E1: decide_test_set bound to the preregistered statistic (0.95 / 10000 / 0.25, single-source constants), DecisionInputError, verdict carries the summary: phase1_protocol_hex_v3.py |
| 67AA8F179A1AC793 | 2026-10-20 10:45 | src/toolkit | F-E1 tests: confidence-0.5 flip refused, malformed summaries fail closed: test_phase1_protocol_hex_v3.py |
| 577A6CD39743FA8F | 2026-10-20 10:45 | spec | F-E1 amendment to section 5 (binding to the preregistered statistic): PHASE1_PROTOCOL_v1.md |
| 69860D34BC97A56D | 2026-10-20 10:35 | src/toolkit | F-E1 test asserts the "gino" key before reading it (mutant to_dict-without-gino died by KeyError, not by assertion): test_phase1_protocol_hex_v3.py |
| 0120C11C5BB2A1CE | 2026-10-20 10:50 | src/toolkit | review test gap D4/D5/D10: binomial closed-form 95 % oracle at default arguments, exact law of the seed-axis IQM, constants pinned to the preregistration text: test_phase1_protocol_hex_v3.py |
| 51B0BD2F0CF1D8D1 | 2026-10-20 11:05 | scripts/oracles | F-S4: float32 vs float64 neighbour graphs of both GNO directions (torch.cdist as neuraloperator 2.0.0 native_neighbor_search, cross-checked against the backend), minimal relative margin, float32 distance error near the radius: gino_radius_tie_survey.py |
| E8D9E5CDB1554287 | 2026-10-20 11:10 | spec | amendment 2026-10-20 float32 graph check (F-S4): 9 tie- and flip-free candidates, margin over float32 error, open owner questions; 2026-10-19 list marked superseded: GINO_ADAPTER_PHASE1_v2.md |
| 20D080BA9E8B14FC | 2026-10-20 11:10 | spec | D4 decision support corrected 2026-10-20 (F-S4, tie-free and flip-free; status DRAFT unchanged): PREREGISTRATION_PHASE1_v1.md |
| 101FC9F9B68FE738 | 2026-10-20 11:20 | src/toolkit | E3: decision reasons name the Pareto-frontier range next to the range of all measured points (behaviour unchanged): phase1_protocol_hex_v3.py |
| 06519E807F079D32 | 2026-10-20 11:20 | src/toolkit | E3 test with dominated points outside the frontier range: test_phase1_protocol_hex_v3.py |
| D87F420DA34BD810 | 2026-10-20 11:20 | spec | E3 amendment to section 5 (the frontier range counts, conservative reading): PHASE1_PROTOCOL_v1.md |
| 6071A853C6E66A0D | 2026-10-20 11:20 | spec | section 6 clarified: "measured range" is the Pareto-efficient part (status DRAFT unchanged): PREREGISTRATION_PHASE1_v1.md |
| 6DA63CDA3F9849A4 | 2026-10-20 12:10 | src/toolkit | Equalita R2: decide_test_set binds n_seeds == 5 and n_samples == 500, compares as Python numbers (NEP 50), requires low >= 0, reads the validated high once (TOCTOU): phase1_protocol_hex_v3.py |
| 287AAE9833753B0F | 2026-10-20 12:10 | src/toolkit | Equalita R2 forgers (seed/sample subsets, float32 label, negative low) + TOCTOU test; binding test on the preregistered 5x500 shape: test_phase1_protocol_hex_v3.py |
| 50C52B56667C09B5 | 2026-10-20 12:10 | spec | section 5 amendment: seed and sample counts bound, provenance not bound (#82), unverified "400 designs" replaced: PHASE1_PROTOCOL_v1.md |
| 094E1326D5CDD3FF | 2026-10-20 12:10 | spec | section 8: flip-free is machine-specific; (48, 2.75) no longer recommended, 8 candidates (status DRAFT unchanged): PREREGISTRATION_PHASE1_v1.md |
| 73874D088CF6EFFE | 2026-10-20 12:10 | spec | amendment: (48, 2.75) flips under 248 of 864 emulated float32 summation orders, only 8 candidates may be named: GINO_ADAPTER_PHASE1_v2.md |
| 06F964CF13846ACF | 2026-10-21 10:10 | src/toolkit | GPT-6 R2 pairing: paired_bootstrap_iqm pairs matrices only by equal sample ids (sample_ids, PairingError): phase1_protocol_hex_v3.py |
| 118E2C34FFA85A17 | 2026-10-21 10:10 | src/toolkit | pairing tests: shape-only, disjoint (triangle/torus), reordered and asserted ids refused; positive control bit-identical: test_phase1_protocol_hex_v3.py |
| 551410BA7A6495D3 | 2026-10-21 10:10 | spec | section 4 amendment: pairing by sample id, not by shape (API change): PHASE1_PROTOCOL_v1.md |
| 3E2AB0E0E7A66D67 | 2026-10-21 10:20 | src/toolkit | GPT-6 HEX-P1-01: decide_preregistered from raw 5x500 errors (registered seeds/ids/lattices, manifest digest, bootstrap inside, record SHA-256), verify_preregistered_decision; decide_test_set diagnostic; error matrices in C order: phase1_protocol_hex_v3.py |
| 1041F8991DB1A202 | 2026-10-21 10:20 | src/toolkit | HEX-P1-01 forger tests (dataclasses.replace on summary, object and serialised record, resealed digest), canonical order, registration refusals, claim boundary; memory-layout test: test_phase1_protocol_hex_v3.py |
| A6AE29206BE48805 | 2026-10-21 10:20 | spec | section 5 diagnostic helper, new section 5a decision-grade entry with claim boundary (unattested), C-order note (API change): PHASE1_PROTOCOL_v1.md |
| 13F4ECD979D37BA9 | 2026-10-21 10:26 | src/toolkit | GPT-6 HEX-P2-02 D4 tools: cdist_rounding_margin (order-independent float32 bound, Higham gamma_6) and radius_graph_stability (budget fixed beforehand, frozen-order branch, NaN fail-closed): phase1_protocol_hex_v3.py |
| 9A206F29217304F8 | 2026-10-21 10:26 | src/toolkit | HEX-P2-02 tests: closed forms, 48-order float32 emulation oracle (positive/negative control), committed survey oracle, verdict truth table: test_phase1_protocol_hex_v3.py |
| F7B392170B94C6C0 | 2026-10-21 10:30 | src/toolkit | diskriminierung DK-20261021T082910: grade check asserts instead of crashing with KeyError when the field is missing: test_phase1_protocol_hex_v3.py |
| A147F2A1B522E7D7 | 2026-10-21 10:26 | spec | new section 11: D4 float-order stability instruments (no threshold changed): PHASE1_PROTOCOL_v1.md |
| 08863B0165DEA61F | 2026-10-21 10:26 | spec | amendment 2026-10-21: float budget criterion; no tie-free candidate clears the order-independent bound; tie band and prereg unchanged: GINO_ADAPTER_PHASE1_v2.md |
| 9FC024E59280694C | 2026-10-21 12:12 | src/toolkit | GPT-6 HEXA-TIM-01: decide_preregistered requires exactly TimingProtocol().repeats (10) timed calls per sample, shape (10,) or (samples, 10), else DecisionInputError: phase1_protocol_hex_v3.py |
| 72878AD3048FBE56 | 2026-10-21 12:20 | src/toolkit | HEXA-TIM-01 tests: 1/9/11/20 flat and (5,9)/(5,11)/(10,1) refused, positive control (10,)/(1,10)/(5,10)/(10,10) unchanged and a refusal fails by assertion; bad-latency fixtures carry 10 calls: test_phase1_protocol_hex_v3.py |
| 954C50E9B4819382 | 2026-10-21 12:12 | spec | section 5a decision input: exactly 10 measured repeats per sample (HEXA-TIM-01); warm-ups not checkable from the numbers (open, attestation): PHASE1_PROTOCOL_v1.md |
| BA6302C071C61737 | 2026-10-21 13:23 | src/toolkit | Equalita R3 F1/F2/F4: decide_preregistered binds the baseline curve to the registered steps grid (7 points, steps and n in the record) and the run-manifest content the preregistration fixes (split ids in the train pool, test-set dataset digest, preregistration file digest, cuda False, float32, neuraloperator 2.0.0); radius_graph_stability refuses flips that contradict a cleared budget: phase1_protocol_hex_v3.py |
| BF40DC9C4A55ABCA | 2026-10-21 13:23 | src/toolkit | Equalita R3 F1/F2/F4/F8/F9 tests: steps-grid subsets refused (forger positive control on the diagnostic helper), manifest content forgeries refused with positive control, contradicted budget refused, latency median, baseline values bound in the record; fixture 7-point curve and real preregistration digest: test_phase1_protocol_hex_v3.py |
| CFC877FC7E21409E | 2026-10-21 13:23 | spec | section 5a: steps-grid binding and manifest content checks with the unchecked fields named; "no parameter accepts a summary" and "any violation raises" made exact (Equalita R3 F7); section 11 flips contradict the budget: PHASE1_PROTOCOL_v1.md |
| 8A51AEE6BFDDC84D | 2026-10-21 13:38 | src/toolkit | diskriminierung DK-20261021T112713: F9 record check and the unreadable-preregistration case assert instead of crashing (KeyError, FileNotFoundError): test_phase1_protocol_hex_v3.py |
| BE2D86527B4CEF34 | 2026-10-21 13:53 | src/toolkit | F1 self-check (same class one level up): baseline curve of 7 points must not repeat a time or an error; padding the {1, 64} subset to 7 by repetition was accepted as advantage at ef4e797: phase1_protocol_hex_v3.py |
| 07EF379B8F7A99F3 | 2026-10-21 13:53 | src/toolkit | F1 self-check tests: three padded subsets refused: test_phase1_protocol_hex_v3.py |
| A20139D11AA81C7D | 2026-10-21 13:53 | spec | section 5a: no time and no error of the baseline curve may repeat: PHASE1_PROTOCOL_v1.md |
| 9290C02717471FE6 | 2026-10-21 15:09 | src/toolkit | Equalita R4-F1: decide_preregistered takes baseline_curve (CostAccuracyPoint tuple, API break, float lists refused); each point checked by its own steps (registered order), n, domain, sample_count 500, timed_samples; record written from the points (baseline_points); CostAccuracyPoint gains domain; R4-F3 hardware amp False / autocast False; repeat-check comment narrowed to literal repetition (R4-F2): phase1_protocol_hex_v3.py |
| 9ED8181814ED71CD | 2026-10-21 15:25 | src/toolkit | R4-F1 self-check (same class one level up): one snapshot tuple(baseline_curve) serves every check and the record; a sequence yielding other points on a later pass had one set checked and another decided and recorded: phase1_protocol_hex_v3.py |
| B203D951C0A7C9E9 | 2026-10-21 15:25 | src/toolkit | R4-F1 self-check test: switching sequence (switch after pass 1, 2, 3) gives the registered record: test_phase1_protocol_hex_v3.py |
| F151A8913A587FCB | 2026-10-21 15:09 | src/toolkit | R4-F1 tests: Equalita's genuine foreign-grid and n=64 curves refused, registered genuine curve H0, every coordinate, genuine producer output, old form refused, relabel boundary; R4-F3 amp/autocast; R4-F6 pins (times-only/errors-only repeat, flips=1, neuraloperator 2.0.1, file_sha256 reason): test_phase1_protocol_hex_v3.py |
| 102B992AC5E4784D | 2026-10-21 15:09 | spec | section 2 domain/sample_count fields; section 5a baseline_curve coordinates, AMP, claim boundary (input identity, horizon, relabel, near-copies), API change R4-F1: PHASE1_PROTOCOL_v1.md |
| B203D951C0A7C9E9 | 2026-10-21 15:28 | src/toolkit | same bytes as the B203D951 line above, repeated at the end: that line was inserted above the F151A891 line, and the last line per file wins: test_phase1_protocol_hex_v3.py |
