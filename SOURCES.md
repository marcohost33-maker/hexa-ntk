# _SOURCES_raw.md — hexa-ntk (Staging 2026-06-04)

| SHA256 (16) | mtime | Ziel | Quelle |
|---|---|---|---|
| A4B6EF52247ED5AA | 2026-05-25 14:33 | src | C:\Users\marco\OneDrive\Desktop\Coworker-Research\coworker-dde\src\hexantk_v3.py |
| 234106F54BFF22FD | 2026-04-06 22:50 | spec | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\HexaNTK_Review_Gehaertet.md |
| 4A97524E450E91CD | 2026-05-26 14:05 | results | C:\Users\marco\OneDrive\Desktop\Coworker-Research\coworker-dde\evidence\2026-05-26_gate-run\logs\hexantk_v3.log |

## src/toolkit/ — 16 Module (append-only, ergaenzt 2026-06-19)

> **Schema.** `SHA256 (16)` = `sha256(<repo-Datei>)[:16].upper()` (gleiche Konvention wie oben,
> nicht die Quell-Datei). `mtime` = mtime der **Quell-Datei** (0Archiv-Snapshot, READ-ONLY).
> `Quelle` = Herkunft; "(in-repo modifiziert, Commit X)" markiert Module, deren aktuelle Bytes vom
> 04-06-Snapshot **abweichen**, weil sie nach der Uebernahme im Repo gehaertet wurden (Lint-Baseline
> Commit `0865915`, Atomic-Checkpoint-Fix `c6628d8`, Durability-Praezisierung `5b88d29`) —
> Provenance = Original-Quelle + abgeleitete Form (SLSA-Praxis: beide festhalten). Re-verifizierbar
> mit `python scripts/verify_sources.py` (rechnet die SHA-256 der Repo-Dateien gegen diese Tabelle).
> Die 8 NICHT als "in-repo modifiziert" markierten Module sind byte-identisch zum 04-06-Snapshot.

| 7D48D5B659D08A05 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\benchmark_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| 2BBFD0218DEBBF0F | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\checkpoint_integration_hex_v3.py (in-repo modifiziert, Commit 5b88d29) |
| A8ACCD446EC287D6 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\dataloader_utils_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| DF201FDE60688DAF | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\dataset_split_hex_v3.py |
| 72019F9045DF105D | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\ema_hex_v3.py |
| 4C0BC9B6D6233180 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\failure_signature_hex_v3.py |
| D0192688CE5222F1 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\operator_adapter_hex_v3.py |
| BB8B7E456C1B7DEB | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\resume_utils_hex_v3.py |
| 8B1886090583A412 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\runner_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| DE5B1FE6EA89314D | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_benchmark_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| E18BF23374494EE9 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_checkpoint_integration_hex_v3.py (in-repo modifiziert, Commit c6628d8) |
| 54FE8C5313B45824 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_dataloader_utils_hex_v3.py |
| B5C8477B5F985FD3 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_ema_hex_v3.py (in-repo modifiziert: Fallback-/Load-Pfad-Tests 56%->100%, claude/hexa-ntk-repo-analysis-3vcvxi) |
| C0EAD8D722BCE4D4 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_operator_adapter_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| AB14B563E7D9EA9C | 2026-07-03 00:00 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_runner_hex_v3.py (in-repo modifiziert: HexTrainingRunner-Coverage-Tests 25%->79%, vero/hexa-ntk-coverage-ratchet-runner) |
| 1FAB88FF9CCA6627 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\warmstart_utils_hex_v3.py |

## src/toolkit/ — Ergaenzung 2026-07-03 (append-only)

> Neues Testfile, **in-repo erstellt** (kein 0Archiv-Snapshot als Quelle): schliesst die
> Coverage-Luecke von `dataset_split_hex_v3.py` (vorher 0% — einziges Modul ohne Tests).
> `mtime` = Erstellzeit im Repo.

| 8890E69A5CC1AA6B | 2026-07-03 03:51 | src/toolkit | in-repo erstellt, Branch claude/code-analysis-optimization-ajgyce: test_dataset_split_hex_v3.py |

## src/toolkit/ — Ergaenzung 2026-08-08 (append-only)

> Zwei neue Testfiles, **in-repo erstellt** (kein 0Archiv-Snapshot als Quelle): schliessen die
> Coverage-Luecken von `warmstart_utils_hex_v3.py` (43% -> 94%) und `resume_utils_hex_v3.py`
> (76% -> 96%) — beide Module hatten vorher kein eigenes Testfile, nur inzidentelle Abdeckung.
> `mtime` = Erstellzeit im Repo.

| C8B771AE5244D0D3 | 2026-08-08 14:10 | src/toolkit | in-repo erstellt, Branch claude/hexa-ntk-repo-analysis-3vcvxi: test_warmstart_utils_hex_v3.py |
| 0C8A071D597A6494 | 2026-08-08 14:10 | src/toolkit | in-repo erstellt, Branch claude/hexa-ntk-repo-analysis-3vcvxi: test_resume_utils_hex_v3.py |
