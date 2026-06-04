# ADR-001: Warum dieses Repo existiert

**Status:** accepted | **Datum:** 2026-06-04 | **Entscheider:** Marco (User1) + Vero

## Kontext
HexaNTK-Material lag verstreut (Google Drive 0-Folder, Desktop Universe 2.0, coworker-dde).
Marco-Auftrag 2026-06-04: pro Forschungsthema ein sauberes privates GitHub-Repo; Daten zuvor
akribisch suchen, ordnen, SHA-dedupen (64 exakte Dubletten quarantaeniert, 24 dank SHA-Check
behalten); Quell-Ordner (Projekte/Projekte 2/Universe 2.0) bleiben unveraendert.

## Entscheidung
Eigenes Repo `hexa-ntk` nach `arbeitsschablone_forschungs-repo-anlage` (Phase 0-6): SHA-Provenance
in SOURCES.md, ehrliche Status-Zeilen, CI = Syntax-Gate, kanonische v3-Engine-Quelle bleibt
coworker-dde (byte-identisch verifiziert 2026-06-04, Equalita-Audit 4/4 PASS).

## Konsequenzen
+ Ein Ort pro Thema, nachvollziehbare Lineage, CI-Schutz.
- Engine-Doppelhaltung mit coworker-dde -> Sync-Disziplin noetig (AGENTS.md §Project).
