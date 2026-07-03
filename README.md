# HexaNTK

> **Status:** Forschungs-Repo (privat) | Gehaertet + abgeschlossen (Toolkit-Stand 04-06, Engine-Stand 05-25). PyTorch >= 2.6 fuer Engine-Lauf noetig (CPU-Wheel jetzt auch im CI-Test-Gate).
> **Lineage/Provenance:** siehe `SOURCES.md` (SHA-256 je Quelldatei) | **Lizenz:** Apache-2.0

Neural-Operator-Toolkit (GINO/GAOT-Familie) fuer physics-informed ML auf hexagonalen Gittern: Checkpoint/EMA/DDP-Integration, Dataset-Split, Failure-Signatures, Benchmark.

## Kern

- **Engine:** `src/hexantk_v3.py` (kanonisch = coworker-dde/src@main) + `src/toolkit/` (16 gehaertete Module, 2026-04-06)
- **Evidenz:** Gate-Run 2026-05-26: **26/26 Selftests + 47/47 Doctests PASS** (`results/hexantk_v3.log`); Re-Verifikation 2026-07-03 inkl. **146/146 Toolkit-Unit-Tests + 68% Branch-Coverage + Provenance 17/17** (`results/2026-07-03_coverage-baseline_gate.log`); gehaertetes Experten-Review in `spec/`.

## Struktur

```
src/        Engines + Experiment-Code
scripts/    Gate-/Hilfs-Skripte (engine_gate.py = exit-korrektes Engine-Gate)
spec/       Specs, Theorie, Audits (gehaertete Versionen)
results/    Selftest-/Gate-Logs, Reports, Negativ-Results
archive/    Vorgaenger-Versionen (Lineage)
docs/ADR/   Architektur-Entscheide
docs/ROADMAP.md  Naechste Schritte + offene Lint-Schuld (ehrlich getrackt)
SOURCES.md  Provenance: Quelle + SHA-256 + mtime je Datei
```

## Reproduzieren

Lokal (volle Evidenz):

```bash
python scripts/engine_gate.py                                  # 47 Doctests + 26 Selftests
python scripts/verify_sources.py                               # Provenance-Gate (SOURCES.md)
python -m coverage run -m unittest discover -s src/toolkit -t src/toolkit -p "test_*.py"
python -m coverage report                                      # 146 Unit-Tests + Coverage-Gate
pre-commit install && pre-commit run --all-files                # Lint-Hooks (optional)
```

CI (`.github/workflows/ci.yml`) hat zwei Gates, beide **blocking**:
- **gate** — Syntax (`compileall`) + Lint (`ruff check src scripts`, ruff gepinnt; Konfig in
  `pyproject.toml`; kanonische Engine `src/hexantk_v3.py` ist vom Lint-Gate ausgenommen und wird
  ueber coworker-dde gepflegt) + Provenance (`scripts/verify_sources.py`, fail-closed gegen
  `SOURCES.md`).
- **tests (cpu)** — installiert CPU-`torch` + `numpy` + `coverage` (gepinnt), faehrt
  Engine-Doctests/Selftests (`scripts/engine_gate.py`, exit-korrekt) + die 146
  Toolkit-Unit-Tests unter Branch-Coverage (`fail_under` in `pyproject.toml`).

## Verwandte Repos

`coworker-dde` (kanonische v3-Engine-Quelle, 3DDE-Kanon) | `HQST` (p1_tasks-Floquet-Linie) |
`phi-hex` `hex-hqst` `u2-dualspec` `u6-bifurcation` `hexa-ntk` (diese Serie, 2026-06-04)

---
*Coworker Research / Coworkerz | Repo-Anlage 2026-06-04 nach arbeitsschablone_forschungs-repo-anlage*
