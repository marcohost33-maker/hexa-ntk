# HexaNTK

> **Status:** Forschungs-Repo (Coworker Research / Coworkerz). **Reifegrad: gehaertete
> Trainings-Infrastruktur mit gruenen Gates — kein publiziertes wissenschaftliches
> Ergebnis, kein Produkt, kein trainiertes Modell.**
> **Lizenz:** Apache-2.0 | **Provenance:** `SOURCES.md` (SHA-256 je Datei, fail-closed Gate)

## Was das hier ist

Trainings- und Experiment-**Infrastruktur** in PyTorch fuer Neural-Operator-Laeufe auf
hexagonalen Gittern: atomare Checkpoints, Resume/Warmstart, EMA, DDP-Guards,
Dataset-Split, Failure-Signatures, Benchmark-Harness — plus **Konfigurations- und
Shape-Validierung** fuer die Operator-Familien GINO / GINOT / GAOT.

Die Operator-Familien selbst sind **fremde, publizierte Verfahren** (Referenzen unten).
Dieses Repo implementiert sie **nicht**; es haelt den Vertrag (`OperatorType`,
`GAOTConfig`, `OperatorContext`) und validiert Shapes, damit die Infrastruktur fuer sie
vorbereitet ist. Der Engine-Header sagt das woertlich:

> "This config is a FUTURE-EXTENSION HOOK. The actual PyTorch implementation is kept in a
> separate training-adapter session; here we declare the contract and validate shapes so
> the rest of our infrastructure (dataset split, checkpoint semantics, OperatorAdapter) is
> GAOT-ready." — `src/hexantk_v3.py:717-720`

Hex-spezifisch im engeren Sinn sind zwei verifizierte Formeln:
`hex_laplacian_stencil_weight` (7-Punkt-FV-Gewicht `2/(3h^2)`, `src/hexantk_v3.py:562`)
und `hex_cfl_max_dt` (Stabilitaets-Schranke `dt <= h^2/(3 D)`, `src/hexantk_v3.py:578`).

## Was das ausdruecklich NICHT ist

- **Keine GINO-/GINOT-/GAOT-Implementierung.** Kein Operator-Netz, kein Forward-Pass,
  kein Training-Loop fuer diese Architekturen — nur Config-Vertrag + Validierung.
- **Keine Reproduktion der Paper-Ergebnisse.** Aussagen wie "schlaegt X auf 28 Benchmarks"
  stehen im Engine-Header als **Zitat der jeweiligen Autoren** (`[GAOT25]`), nicht als
  Ergebnis dieses Repos. Hier wurde **nichts davon nachgemessen**.
- **Keine trainierten Gewichte, keine Datensaetze, kein Benchmark-Resultat.**
- **Kein betriebener Dienst, keine Bibliothek mit stabiler API.** Kein PyPI-Paket, kein
  SemVer-Versprechen; Import-Pfade koennen sich aendern.
- **Kein Nachweis physikalischer Korrektheit** einer PDE-Loesung. Die Gates pruefen
  Software-Integritaet (Determinismus, Round-Trips, Shapes), nicht Physik.

## Reifegrad / Evidenz (Lauf 2026-08-26, lokal, Exit-Codes gesehen)

| Gate | Kommando | Ergebnis |
|---|---|---|
| Engine | `python scripts/engine_gate.py` | 47/47 Doctests + 26/26 Selftests, exit 0 |
| Provenance | `python scripts/verify_sources.py` | 17/17 Toolkit-Eintraege, exit 0 |
| Toolkit-Tests | `python -m unittest discover -s src/toolkit -t src/toolkit -p "test_*.py"` | 166/166, exit 0 |
| Coverage | `python -m coverage run ... && python -m coverage report` | 82 % gesamt (`fail_under = 79`), exit 0 |

Lokale Umgebung dieses Laufs: Python 3.14.4, `torch` 2.12.1+cpu, `numpy` 2.4.6.
Die CI faehrt dieselbe Kette auf **Python 3.12** mit `torch==2.6.0` (CPU-Wheel).
Die Zahlen gelten fuer diese beiden Umgebungen; ausserhalb ist nichts zugesichert.

**Resume-Pfad (in diesem Branch korrigiert):** `HexTrainingRunner.maybe_resume()`
reicht `config.resume_from` jetzt explizit an den Checkpoint-Loader weiter.
Existenzpruefung und tatsaechlich geladene Datei bezeichnen damit denselben Pfad.
Manager- und Runner-Regressionstests decken einen Checkpoint ausserhalb des
aktuellen `output_dir` mit abweichender Epoche und `global_step` ab, auch wenn
im `output_dir` bereits ein eigener Resume-Checkpoint liegt (der fruehere
Fehlerfall: dieser wurde still geladen). Eine gesetzte, aber fehlende
`resume_from`-Datei bricht mit `FileNotFoundError` ab (fail-closed), statt
still bei Epoche 0 neu zu trainieren. Die
Trust-/`weights_only`-Semantik bleibt unveraendert. Der isolierte Fix ersetzt
den entsprechenden Teil des aelteren Sammel-PR #20.

Toolkit-Stand 2026-04-06, Engine-Stand 2026-05-25. Das gehaertete Experten-Review liegt
in `spec/`, offene Lint-/Test-Schuld in `docs/ROADMAP.md`.

## Kern

- **Engine:** `src/hexantk_v3.py` (1595 Zeilen; Checkpoints, EMA-Policy, OOM-sicherer
  Train-Step, Operator-Vertraege, 26 Selftests inline)
- **Toolkit:** `src/toolkit/` — 10 Module + 7 Testmodule (Benchmark, Checkpoint-Integration,
  Dataloader-Utils, Dataset-Split, EMA, Failure-Signature, Operator-Adapter, Resume-Utils,
  Runner, Warmstart-Utils)

## Reproduzieren

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt                  # zentrale Tool-Pins: ruff/coverage
python -m pip install numpy
python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
python scripts/engine_gate.py                                  # 47 Doctests + 26 Selftests
python scripts/verify_sources.py                               # Provenance-Gate (SOURCES.md)
python -m coverage run -m unittest discover -s src/toolkit -t src/toolkit -p "test_*.py"
python -m coverage report                                      # 170 Unit-Tests + Coverage-Gate
```

CI (`.github/workflows/ci.yml`), beide Jobs **blocking**:
- **gate** — Lint (`ruff check src scripts`; die kanonische Engine `src/hexantk_v3.py` ist
  vom Lint-Gate ausgenommen und wird stromaufwaerts gepflegt), `compileall`, Provenance.
- **tests (cpu)** — Engine-Gate + 170 Toolkit-Unit-Tests unter Branch-Coverage.

## Struktur

```text
src/        Engine + Toolkit
scripts/    Gate-Skripte (engine_gate.py = exit-korrektes Engine-Gate)
spec/       Review/Audit (gehaertete Fassung)
results/    Gate-Logs
docs/ADR/   Architektur-Entscheide
SOURCES.md  Provenance: Quelle + SHA-256 + mtime je Datei
```

## Referenzen (fremde Verfahren — nicht Ergebnisse dieses Repos)

Der vollstaendige Referenz-Block mit Kontext steht in `src/hexantk_v3.py:37-60`.

- **[Li23]** Z. Li, N. Kovachki, C. Choy et al., *Geometry-Informed Neural Operator for
  Large-Scale 3D PDEs*, NeurIPS 2023, arXiv:2309.00583. — Ursprung von **GINO**.
- **[GINOT25]** Q. Liu et al., *Geometry-Informed Neural Operator Transformer*,
  ScienceDirect S0045782525009405 (Dez. 2025).
- **[GAOT25]** S. Wen et al., *Geometry Aware Operator Transformer*, arXiv:2505.18781.
  Die 2D-Default-Hyperparameter in `GAOTConfig` sind aus dieser Arbeit uebernommen;
  fuer 3D gibt es dort **keinen** validierten Default (siehe `src/hexantk_v3.py:707-715`).
- **[Sarkar25]** S. Sarkar et al., *Physics- and geometry-aware spatio-spectral graph
  neural operator* (piG-Sp2GNO / mGNO), arXiv Aug. 2025.
- Hex-FV-Stencil: Lee & Tien 2014 / Lee 2017 (`src/hexantk_v3.py:567`).

Die Herkunft jeder Datei (inkl. der in-repo gehaerteten Ableitungen) ist in `SOURCES.md`
mit SHA-256 gepinnt und mit `scripts/verify_sources.py` nachpruefbar.

## Verwandte Repos

`coworker-dde` ist die stromaufwaerts liegende Quelle der kanonischen v3-Engine;
`HQST`, `hex-hqst`, `u2-dualspec`, `u6-bifurcation` gehoeren zur selben Serie.
**Diese Repos sind privat und von aussen nicht erreichbar.** Oeffentlich zugaenglich ist
aus der Serie nur [`phi-hex`](https://github.com/marcohost33-maker/phi-hex).

---
*Coworker Research / Coworkerz | Repo-Anlage 2026-06-04*
