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
