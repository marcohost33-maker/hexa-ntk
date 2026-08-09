# HexaNTK

> **Status:** Forschungs-Repo (privat) | Gehaertet + abgeschlossen (Toolkit-Stand 04-06, Engine-Stand 05-25). PyTorch >= 2.6 fuer Engine-Lauf noetig (CPU-Wheel jetzt auch im CI-Test-Gate).
> **Lineage/Provenance:** siehe `SOURCES.md` (SHA-256 je Quelldatei) | **Lizenz:** Apache-2.0

Neural-Operator-Toolkit (GINO/GAOT-Familie) fuer physics-informed ML auf hexagonalen Gittern: Checkpoint/EMA/DDP-Integration, Dataset-Split, Failure-Signatures, Benchmark.

## Kern

- **Engine:** `src/hexantk_v3.py` (kanonisch = coworker-dde/src@main) + `src/toolkit/` (19 gehaertete Module: 10 Kern + 9 Testfiles; Basis 2026-04-06, Testausbau 07-03/08-08)
- **Evidenz:** Gate-Run 2026-05-26: **26/26 Selftests + 47/47 Doctests PASS** (`results/hexantk_v3.log`); Re-Verifikation 2026-08-08 inkl. **231/231 Toolkit-Unit-Tests + 94% Branch-Coverage + Provenance 19/19** (`results/2026-08-08_selfreview-ratchet-91_gate.log`); gehaertetes Experten-Review in `spec/`.

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
requirements-dev.txt  Zentrale CI-/Dev-Tool-Pins fuer Ruff/Coverage
```

## Reproduzieren

Lokal in einem frischen Python-3.12-Environment (Floor py3.12 seit #19):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt                 # zentrale Tool-Pins: ruff/coverage
python -m pip install numpy
python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
python scripts/engine_gate.py                                  # 47 Doctests + 26 Selftests
python scripts/verify_sources.py                               # Provenance-Gate (SOURCES.md)
python -m coverage run -m unittest discover -s src/toolkit -t src/toolkit -p "test_*.py"
python -m coverage report                                      # 231 Unit-Tests + Coverage-Gate
python -m pip install pre-commit && pre-commit install && pre-commit run --all-files  # Hooks optional
```

CI (`.github/workflows/ci.yml`) hat zwei Gates, beide **blocking**:
- **gate** — installiert zentrale Dev-Tool-Pins aus `requirements-dev.txt`; Syntax (`compileall`) + Lint (`ruff check src scripts`, Konfig in `pyproject.toml`; kanonische Engine `src/hexantk_v3.py` ist vom Lint-Gate ausgenommen und wird ueber coworker-dde gepflegt) + Provenance (`scripts/verify_sources.py`, fail-closed gegen `SOURCES.md`).
- **tests (cpu)** — installiert zentrale Dev-Tool-Pins aus `requirements-dev.txt`, CPU-`torch` + `numpy`, faehrt Engine-Doctests/Selftests (`scripts/engine_gate.py`, exit-korrekt) + die 231 Toolkit-Unit-Tests unter Branch-Coverage (`fail_under` in `pyproject.toml`).

## Verwandte Repos

`coworker-dde` (kanonische v3-Engine-Quelle, 3DDE-Kanon) | `HQST` (p1_tasks-Floquet-Linie) |
`phi-hex` `hex-hqst` `u2-dualspec` `u6-bifurcation` `hexa-ntk` (diese Serie, 2026-06-04)

---
*Coworker Research / Coworkerz | Repo-Anlage 2026-06-04 nach arbeitsschablone_forschungs-repo-anlage*
