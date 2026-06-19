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

| 622095D493953C80 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\benchmark_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| 5A74A94E68A466B1 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\checkpoint_integration_hex_v3.py (in-repo modifiziert, Commit 5b88d29) |
| 24E5EA3184949672 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\dataloader_utils_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| DF201FDE60688DAF | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\dataset_split_hex_v3.py |
| 72019F9045DF105D | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\ema_hex_v3.py |
| 4C0BC9B6D6233180 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\failure_signature_hex_v3.py |
| D0192688CE5222F1 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\operator_adapter_hex_v3.py |
| BB8B7E456C1B7DEB | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\resume_utils_hex_v3.py |
| D689D9342ED5F76F | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\runner_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| 9ABCD5952E6B9D9C | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_benchmark_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| 0AC850C8BF2BCD28 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_checkpoint_integration_hex_v3.py (in-repo modifiziert, Commit c6628d8) |
| 54FE8C5313B45824 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_dataloader_utils_hex_v3.py |
| 574B8851CE411016 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_ema_hex_v3.py |
| 7B05FFD8CF18DE68 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_operator_adapter_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| 80D97A66E19D95C2 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\test_runner_hex_v3.py (in-repo modifiziert, Commit 0865915) |
| 1FAB88FF9CCA6627 | 2026-04-06 22:50 | src/toolkit | G:\Meine Ablage\0Archiv Snapshots\Hex-HexaNTK\3HexaNTK\warmstart_utils_hex_v3.py |
