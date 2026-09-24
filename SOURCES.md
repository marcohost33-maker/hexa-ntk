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


## src/toolkit/ — Cluster Fabric safety spine 2026-09-24

> Neue Dateien, **in-repo erstellt**. Der Router selbst ist bewusst nicht gelernt:
> er adjudiziert nur Vorschlaege eines spaeteren learned/heuristic Routers gegen
> harte Capability-, Plane-, Budget-, Confidence- und Independence-Vertraege.

| 7FC19156DC55C6B8 | 2026-09-24 01:47 | src/toolkit | in-repo erstellt, Branch codex/cluster-fabric-safety-spine-v1: cluster_routing_hex_v1.py |
| DF10E18049BE07DD | 2026-09-24 01:47 | src/toolkit | in-repo erstellt, Branch codex/cluster-fabric-safety-spine-v1: test_cluster_routing_hex_v1.py |


## src/toolkit/ — Hex grid fabric + shadow routing 2026-09-24

> Neue Dateien, in-repo erstellt. Die Topologie trennt physisches Hex-Lattice von
> logischem 6-Projekt-Ring; der Shadow-Benchmark fuehrt keine geroutete Aufgabe aus.

| 3D1313BE66A29CF1 | 2026-09-24 02:20 | src/toolkit | in-repo erstellt, Branch codex/cluster-fabric-safety-spine-v1: hex_topology_v1.py |
| 67EA791411BFD9C5 | 2026-09-24 02:20 | src/toolkit | in-repo erstellt, Branch codex/cluster-fabric-safety-spine-v1: test_hex_topology_v1.py |
| D32397464F55BF37 | 2026-09-24 02:20 | src/toolkit | in-repo erstellt, Branch codex/cluster-fabric-safety-spine-v1: shadow_routing_hex_v1.py |
| E83BC132C6291EB2 | 2026-09-24 02:20 | src/toolkit | in-repo erstellt, Branch codex/cluster-fabric-safety-spine-v1: test_shadow_routing_hex_v1.py |


## src/toolkit/ — Cluster fabric hardening derived forms 2026-09-24

> Append-only derived bytes after semantic-hop, shadow-metric and malformed-confidence
> hardening on branch codex/cluster-fabric-safety-spine-v1.

| B8EF5FC32416FB03 | 2026-09-24 02:45 | src/toolkit | in-repo derived: cluster_routing_hex_v1.py |
| 93F2D4703A840EC3 | 2026-09-24 02:45 | src/toolkit | in-repo derived: test_cluster_routing_hex_v1.py |
| 6B4E284D12AFEF60 | 2026-09-24 02:45 | src/toolkit | in-repo derived: hex_topology_v1.py |
| 226C3B0EF0410D28 | 2026-09-24 02:45 | src/toolkit | in-repo derived: test_hex_topology_v1.py |
| 9A024D35B990FB25 | 2026-09-24 02:45 | src/toolkit | in-repo derived: shadow_routing_hex_v1.py |
| B15BA18DC4C8B54B | 2026-09-24 02:45 | src/toolkit | in-repo derived: test_shadow_routing_hex_v1.py |


## src/toolkit/ — State-of-the-art hardening derived forms 2026-09-24

> Append-only derived bytes after budget-search, shared-risk independence,
> calibration, physical-lattice precision and auditable message-envelope hardening.

| 532A28F5198C1297 | 2026-09-24 10:17 | src/toolkit | in-repo derived: cluster_routing_hex_v1.py |
| 731D7830D2978872 | 2026-09-24 10:17 | src/toolkit | in-repo derived: test_cluster_routing_hex_v1.py |
| 92AE3198DDAE5B69 | 2026-09-24 10:17 | src/toolkit | in-repo derived: hex_topology_v1.py |
| E60163049656A5E7 | 2026-09-24 10:17 | src/toolkit | in-repo derived: test_hex_topology_v1.py |
| 23B7F28F27D71110 | 2026-09-24 10:17 | src/toolkit | in-repo created: message_envelope_hex_v1.py |
| 2AA8827DF4DCD60E | 2026-09-24 10:17 | src/toolkit | in-repo created: test_message_envelope_hex_v1.py |


## src/toolkit/ — Final routing/messaging hardening derived forms 2026-09-24

> Append-only derived bytes after feasible-subset routing, shared-risk independence,
> explicit cardinality/calibration contracts, assurance set-cover, precise lattice
> coordinates and producer-scoped causal message identity.

| B119CE30543A9D9F | 2026-09-24 10:17 | src/toolkit | in-repo derived: cluster_routing_hex_v1.py |
| 395316F68FA13818 | 2026-09-24 10:17 | src/toolkit | in-repo derived: test_cluster_routing_hex_v1.py |
| 67B3E28B85C80EA4 | 2026-09-24 10:17 | src/toolkit | in-repo derived: hex_topology_v1.py |
| E60163049656A5E7 | 2026-09-24 10:17 | src/toolkit | in-repo derived: test_hex_topology_v1.py |
| F019FAFDC70AEEA3 | 2026-09-24 10:17 | src/toolkit | in-repo derived: message_envelope_hex_v1.py |
| F729849DF2C702B9 | 2026-09-24 10:17 | src/toolkit | in-repo derived: test_message_envelope_hex_v1.py |


## src/toolkit/ — Complementary compute coverage derived forms 2026-09-24

> Append-only derived bytes after collective compute-capability coverage hardening.

| CB6CDB37B98E0BB9 | 2026-09-24 10:17 | src/toolkit | in-repo derived: cluster_routing_hex_v1.py |
| 9EFD6F2FF8025823 | 2026-09-24 10:17 | src/toolkit | in-repo derived: test_cluster_routing_hex_v1.py |


## src/toolkit/ — Independence-attestation hardening derived forms 2026-09-24

> Append-only derived bytes after fail-closed independence-attestation hardening.

| EBA5AF0E096C3B6E | 2026-09-24 10:17 | src/toolkit | in-repo derived: cluster_routing_hex_v1.py |
| 45FBED0A7936F2AB | 2026-09-24 10:17 | src/toolkit | in-repo derived: test_cluster_routing_hex_v1.py |


## src/toolkit/ — HexaFabric benchmark corpus and topology baselines 2026-09-24

> New benchmark schema/evaluator and structural topology baselines. Historical task
> content is not copied into toolkit modules; the seed corpus is separately versioned.

| 81E91832F4B29644 | 2026-09-24 11:39 | src/toolkit | in-repo created: benchmark_corpus_hex_v1.py |
| C031AB097F61EC5E | 2026-09-24 11:39 | src/toolkit | in-repo created: test_benchmark_corpus_hex_v1.py |
| 2631E7DAA2F9DD3F | 2026-09-24 11:39 | src/toolkit | in-repo created: coordination_topology_hex_v1.py |
| B2661C99ADF1F5D4 | 2026-09-24 11:39 | src/toolkit | in-repo created: test_coordination_topology_hex_v1.py |
