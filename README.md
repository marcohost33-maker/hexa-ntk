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
> GAOT-ready." — `src/hexantk_v3.py:856-859`

Hex-spezifisch im engeren Sinn sind drei getrennte numerische Aussagen implementiert:
`hex_laplacian_stencil_weight` (7-Punkt-FV-Gewicht `2/(3h^2)`),
`hex_cfl_max_dt` (scharfe spektrale/L2-Forward-Euler-Grenze
`dt <= h^2/(3 D)`) und `hex_positivity_max_dt` (scharfe
Positivitaets-/DMP-/Monotonie-Grenze dieses regulaeren linearen Schemas
`dt <= h^2/(4 D)`). Der historische `hex_cfl_max_dt`-Default
`safety=0.85` bleibt aus Kompatibilitaetsgruenden erhalten, ist aber fuer
nichtnegative Zustaende nicht als Positivitaetsgarantie zu verwenden.

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

## Reifegrad / Evidenz (Lauf 2026-09-27, lokal, Exit-Codes gesehen)

| Gate | Kommando | Ergebnis |
|---|---|---|
| Lint | `ruff check src scripts` | 0 Befunde, exit 0 |
| Engine | `python scripts/engine_gate.py` | 49/49 Doctests + 26/26 Selftests, exit 0 |
| Provenance | `python scripts/verify_sources.py` | 22 Toolkit- + 2 Spec-Eintraege, exit 0 |
| Toolkit-Tests | `python -m unittest discover -s src/toolkit -t src/toolkit -p "test_*.py"` | 378/378, exit 0 |
| Coverage | `python -m coverage run ... && python -m coverage report` | 89,6 % gesamt (Branch, `fail_under = 87`), exit 0 |
| Mutation | `python scripts/mutation_probe.py` | 19/19 Mutanten getoetet (Baseline gruen), exit 0 |

Lokale Umgebung dieses Laufs: Python 3.12.3, `torch` 2.6.0 (PyPI-Wheel `+cu124`, auf
CPU ausgefuehrt — der CPU-Wheel-Index war in der Laufumgebung gesperrt), `numpy` 2.5.3.
Die CI faehrt dieselbe Kette auf **Python 3.12** mit `torch==2.6.0` (CPU-Wheel).
Log: `results/2026-09-27_review-hardening_local-gate.log`. Fruehere Laeufe (z. B.
2026-08-26: 47 Doctests, 166 Tests, 82 %) bleiben dort als historische Evidenz stehen.

**Resume ist vollstaendig (seit 2026-09-27).** `HexTrainingRunner.maybe_resume()` laedt
exakt `config.resume_from` (fail-closed bei fehlender Datei, PR #31) und stellt neben
Modell/Optimizer/Scheduler/RNG jetzt auch den **EMA-Mittelwert** und den **Best-Wert**
wieder her. Vorher startete das EMA nach einem Resume neu, und der Best-Wert begann bei
`None` — die erste Epoche nach dem Resume ueberschrieb ein besseres `best.pt`. Quelle der
Wahrheit ist `best.pt` im eigenen Checkpoint-Ordner: uebernommen wird dessen Wert, wenn er
dieselbe Policy traegt und mindestens so gut ist wie der im Resume notierte (besser = spaetere
Epoche desselben Laufs); fehlt `best.pt` oder ist es schlechter (Fremd-Resume, ersetzte Datei),
wird nichts uebernommen und nichts blockiert. Zusaetzlich muessen Lauf-ID und Metrikname
passen, und ein Resume mit anderem `ema_decay` wird abgelehnt (kein stilles Hybrid-EMA).
Regressionstest:
unterbrochener Lauf (2 + 2 Epochen) == ununterbrochener Lauf (4 Epochen), bitgleich fuer
Modell und EMA.

**Branch-Hinweis:** Der Fix `codex/hex-cfl-positivity-contract` (PR #45) fuegt zwei Doctests hinzu
(49 statt 47 erwartet) und hebt die Engine-Version auf 3.0.1. Die obige Tabelle ist
bewusst historische Evidenz des Laufs vom 2026-08-26 und wird nicht als neuer Lauf
umetikettiert; ein neuer PASS wird erst nach CI/Engine-Gate als Evidenz eingetragen.

**Metrik:** `val_nrmse` ist jetzt der gepoolte relative L2-Fehler
`sqrt(sum |pred - y|^2 / sum |y|^2)` ueber den Validierungssatz (vorher eine Kopie des
MSE-`val_loss`, als "Placeholder" markiert, aber Default-Kriterium fuer `best.pt`).

Toolkit-Stand 2026-04-06, Engine-Stand 2026-05-25. Das gehaertete Experten-Review liegt
in `spec/`, offene Lint-/Test-Schuld in `docs/ROADMAP.md`.

## Kern

- **Engine:** `src/hexantk_v3.py` (1772 Zeilen; Checkpoints, EMA-Policy, OOM-sicherer
  Train-Step, Operator-Vertraege, 26 Selftests inline)
- **Toolkit:** `src/toolkit/` — 11 Module + 12 Testmodule (Benchmark, Checkpoint-Integration,
  Dataloader-Utils, Dataset-Split, EMA, Failure-Signature, Operator-Adapter, Resume-Utils,
  Runner, Stress, Warmstart-Utils; dazu die Tests des Provenance-Gates und des Zeitschritt-Vertrags)

## Reproduzieren

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt                  # zentrale Tool-Pins: ruff/coverage
python -m pip install numpy
python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
python scripts/engine_gate.py                                  # 49 Doctests + 26 Selftests
python scripts/verify_sources.py                               # Provenance-Gate (SOURCES.md)
python -m coverage run -m unittest discover -s src/toolkit -t src/toolkit -p "test_*.py"
python -m coverage report                                      # 378 Unit-Tests (2026-09-27) + Coverage-Gate
python scripts/mutation_probe.py                               # optional: 19 Mutanten der 09-27-Fixes
```

CI (`.github/workflows/ci.yml`), beide Jobs scheitern mit exit != 0 (ob `tests (cpu)` auch
den Merge blockiert, entscheidet der Branch-Schutz — laut externem Bericht vom 2026-09-21 ist
nur `gate` Required Check; nur der Repo-Owner kann `tests (cpu)` hinzufuegen):
- **gate** — Lint (`ruff check src scripts`; die kanonische Engine `src/hexantk_v3.py` ist
  vom Lint-Gate ausgenommen und wird stromaufwaerts gepflegt), `compileall`, Provenance.
- **tests (cpu)** — Engine-Gate + Toolkit-Unit-Tests (378, Stand 2026-09-27) unter Branch-Coverage.

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

Der vollstaendige Referenz-Block mit Kontext steht in `src/hexantk_v3.py:43-67`.

- **[Li23]** Z. Li, N. Kovachki, C. Choy et al., *Geometry-Informed Neural Operator for
  Large-Scale 3D PDEs*, NeurIPS 2023, arXiv:2309.00583. — Ursprung von **GINO**.
- **[GINOT25]** Q. Liu et al., *Geometry-Informed Neural Operator Transformer*,
  ScienceDirect S0045782525009405 (Dez. 2025).
- **[GAOT25]** S. Wen et al., *Geometry Aware Operator Transformer*, arXiv:2505.18781.
  Die 2D-Default-Hyperparameter in `GAOTConfig` sind aus dieser Arbeit uebernommen;
  fuer 3D gibt es dort **keinen** validierten Default (siehe `src/hexantk_v3.py:846-854`).
- **[Sarkar25]** S. Sarkar et al., *Physics- and geometry-aware spatio-spectral graph
  neural operator* (piG-Sp2GNO / mGNO), arXiv Aug. 2025.
- Hex-FV-Stencil: Lee & Tien 2014 / Lee 2017 (`src/hexantk_v3.py:658`).

Die Herkunft jeder Datei (inkl. der in-repo gehaerteten Ableitungen) ist in `SOURCES.md`
mit SHA-256 gepinnt und mit `scripts/verify_sources.py` nachpruefbar.

## Verwandte Repos

`coworker-dde` ist die stromaufwaerts liegende Quelle der kanonischen v3-Engine;
`HQST`, `hex-hqst`, `u2-dualspec`, `u6-bifurcation` gehoeren zur selben Serie.
**Diese Repos sind privat und von aussen nicht erreichbar.** Oeffentlich zugaenglich ist
aus der Serie nur [`phi-hex`](https://github.com/marcohost33-maker/phi-hex).

---
*Coworker Research / Coworkerz | Repo-Anlage 2026-06-04*
