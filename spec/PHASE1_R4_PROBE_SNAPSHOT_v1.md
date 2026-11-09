# PHASE1_R4_PROBE_SNAPSHOT_v1

**Status: Technical candidate; scientific preregistration is DRAFT.**

## 1. Vertrauensgrenzen
- Input: exakt der lokale Git-Commit `1a54cd8d9326e762ee493d3fc8d9a80202174304` und seine Objekte (kein Working-Tree-Fallback).
- Vier gepinnte Dateien: Kosten-JSON, Radius-JSON, PR-#91-Vorregistrierung, synthetisches Benchmarkskript; SHA-1-Git-Blob-Pins plus SHA-256 der JSONs und Vorregistrierung.
- Commit-Pin, SHA-256-Prüfung und unveränderte Quellfiles werden **vor** der Ableitung der Patches geprüft.
- Das vollständige Projekt unter `src`, `spec` und `scripts/oracles` stammt aus `git archive` des gebundenen Commits. Das Archiv wird durch einen eigenen, sicherheitsorientierten Extractor eingelesen; keine `extractall`-Aufrufe, keine Symlinks/Hardlinks, kein Pfad-Traversal, keine Überschreibungen und begrenzte Ressourcennutzung.
- Output: neues Zielverzeichnis, atomar umbenannte Staging-Dateien, `R4_PREPARATION_MANIFEST.json`; Original-Skripte bleiben unverändert.
- Rücklesung: `verify_prepared_snapshot_r4.py` rekonstruiert sämtliche abgeleiteten Quellbytes erneut aus der unveränderten Originalquelle und verifiziert sie unabhängig von den Hashes im Manifest.
- Manifest und SHA sind **keine signierte Attestierung** eines vertrauenswürdigen Builds. Besitz eines Archivs und ein Self-Hash sind kein wissenschaftlicher Nachweis.

## 2. Fachliche Änderung des synthetischen Oracles
- `CosineAnnealingLR(T_max=50)` ist vom Timing-Sample `--repeats=3` entkoppelt. Ein Epochenschritt des Schedulers bleibt ein Epochenfortschritt, kein Minibatch-Fortschritt.
- `torch.isfinite(loss)` vor Backward/Optimizer.
- `torch.isfinite(grad)` vor jedem Optimizer-Step; `torch.isfinite(parameter)` nach jedem Update.
- Nicht-endliche Werte -> explizite Ausnahme, kein publizierbarer Bericht.
- **Kostenvergleich:** Gradienten-/Parameter-Scans verursachen zusätzliche CPU-Arbeit. PR-#91-Laufzeiten mit `T_max=3` und ohne diese Guards sind für R4 nur historischer Kontext, keine Kalibrierung der neuen Implementierung.

## 3. Budget und Owner
Der R3-Plan wurde nicht nachträglich freigegeben. Der 48h-Budgetvergleich verlangt echte Messungen auf der genau benannten D5-Maschine. In R4 wird keine `GO`-Entscheidung erstellt. Auch ein erfolgreicher Source-Gate liefert `owner_decision=HOLD_NO_APPROVAL`.

## 4. Testmatrix
- Pfade: `../`, absolute, `./`, `//`, `\\`, DOS-Gerätenamen, casefold-Kollision, Reservierte Windows-Endungen.
- TAR: Symlink, Hardlink, FIFO, doppelte Mitgliedschaft, Parent-Datei-als-Verzeichnis, Einzel- und Gesamtgrößenlimite.
- Git: echter temporärer Commit, `git archive`, manipulierter Working Tree, Write-Safety des Originals, bestehendes Output-Verzeichnis.
- Code: Source-Anker, R3→R4-Transformation, Syntax, korrekte Position von Gradientencheck vor Optimizer, erneute Anwendung verboten.
- Manifest: Original- und Derived-Hashes, Vorregistrierungsstatus, Status-HOLD, gezielte Manipulationen an Candidate und Prereg-Bytes.
- Regression: alle R2/R3-Tests mitgeführt, insgesamt 139/139 bestanden.

## 5. Offen
Echter Original-PR-Checkout im Container fehlt; keine eigene D5-Maschinenmessung, keine Torch-GINO-Ausführung des Kandidatenskripts, keine T1-Güte, keine Owner-Freigabe. Nächste wissenschaftliche Arbeit folgt erst nach dieser klaren Gate-Reihenfolge. CUDA/NCCL, Multi-Host und T2 sind eigene, spätere Linien.
