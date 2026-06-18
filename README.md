# HexaNTK

> **Status:** Forschungs-Repo (privat) | Gehaertet + abgeschlossen (Toolkit-Stand 04-06, Engine-Stand 05-25). PyTorch >= 2.6 fuer Engine-Lauf noetig (nicht in CI).
> **Lineage/Provenance:** siehe `SOURCES.md` (SHA-256 je Quelldatei) | **Lizenz:** Apache-2.0

Neural-Operator-Toolkit (GINO/GAOT-Familie) fuer physics-informed ML auf hexagonalen Gittern: Checkpoint/EMA/DDP-Integration, Dataset-Split, Failure-Signatures, Benchmark.

## Kern

- **Engine:** `src/hexantk_v3.py` (kanonisch = coworker-dde/src@main) + `src/toolkit/` (16 gehaertete Module, 2026-04-06)
- **Evidenz:** Gate-Run 2026-05-26: **26/26 Selftests + 47/47 Doctests PASS** (`results/hexantk_v3.log`); gehaertetes Experten-Review in `spec/`.

## Struktur

```
src/        Engines + Experiment-Code
spec/       Specs, Theorie, Audits (gehaertete Versionen)
results/    Selftest-/Gate-Logs, Reports, Negativ-Results
archive/    Vorgaenger-Versionen (Lineage)
docs/ADR/   Architektur-Entscheide
SOURCES.md  Provenance: Quelle + SHA-256 + mtime je Datei
```

## Reproduzieren

Die Selftests laufen direkt in der Engine (`python <engine>.py --selftest` bzw. eingebaute Runner;
Details im Engine-Header). CI prueft Syntax (`compileall`) + Lint (`ruff check src`, **blocking**,
gepinnt; Konfiguration in `pyproject.toml`; kanonische Engine `src/hexantk_v3.py` ist vom Lint-Gate
ausgenommen und wird ueber coworker-dde gepflegt) — volle physikalische Gate-Runs brauchen
numpy/scipy/torch und laufen lokal.

## Verwandte Repos

`coworker-dde` (kanonische v3-Engine-Quelle, 3DDE-Kanon) | `HQST` (p1_tasks-Floquet-Linie) |
`phi-hex` `hex-hqst` `u2-dualspec` `u6-bifurcation` `hexa-ntk` (diese Serie, 2026-06-04)

---
*Coworker Research / Coworkerz | Repo-Anlage 2026-06-04 nach arbeitsschablone_forschungs-repo-anlage*
