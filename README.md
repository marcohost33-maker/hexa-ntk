# HexaNTK

> **Status:** Forschungs-Repo (Coworker Research / Coworkerz). **Reifegrad: gehaertete
> Trainings-Infrastruktur mit gruenen Gates — kein publiziertes wissenschaftliches
> Ergebnis, kein Produkt, kein trainiertes Modell.**
> **Lizenz:** Apache-2.0 | **Provenance:** `SOURCES.md` (SHA-256 je Datei, fail-closed Gate)

## Was das hier ist

Trainings- und Experiment-**Infrastruktur** in PyTorch fuer Neural-Operator-Laeufe auf
hexagonalen Gittern: atomare Checkpoints, Resume/Warmstart, EMA, DDP-Integration des Runners (2-Rank-gloo getestet),
Dataset-Split, Failure-Signatures, Benchmark-Harness — plus **Konfigurations- und
Shape-Validierung** fuer die Operator-Familien GINO / GINOT / GAOT.

Die Operator-Familien selbst sind **fremde, publizierte Verfahren** (Referenzen unten).
Dieses Repo implementiert sie **nicht**; es haelt den Vertrag (`OperatorType`,
`GAOTConfig`, `OperatorContext`) und validiert Shapes, damit die Infrastruktur fuer sie
vorbereitet ist. Der Engine-Header sagt das woertlich:

> "This config is a FUTURE-EXTENSION HOOK. The actual PyTorch implementation is kept in a
> separate training-adapter session; here we declare the contract and validate shapes so
> the rest of our infrastructure (dataset split, checkpoint semantics, OperatorAdapter) is
> GAOT-ready." — `src/hexantk_v3.py:857-860`

Hex-spezifisch im engeren Sinn sind drei getrennte numerische Aussagen implementiert:
`hex_laplacian_stencil_weight` (7-Punkt-FV-Gewicht `2/(3h^2)`),
`hex_cfl_max_dt` (scharfe spektrale/L2-Forward-Euler-Grenze
`dt <= h^2/(3 D)`) und `hex_positivity_max_dt` (scharfe
Positivitaets-/DMP-/Monotonie-Grenze dieses regulaeren linearen Schemas
`dt <= h^2/(4 D)`). Der historische `hex_cfl_max_dt`-Default
`safety=0.85` bleibt aus Kompatibilitaetsgruenden erhalten, ist aber fuer
nichtnegative Zustaende nicht als Positivitaetsgarantie zu verwenden.

## Was das ausdruecklich NICHT ist

- **Keine interne GINO-/GINOT-/GAOT-Modellimplementierung, kein Training-Loop.**
  Phase 1 bietet einen optionalen Adapter zum externen GINO 2.0.0 samt echtem
  Forward-Smoke auf synthetischen Daten (`gino_adapter_hex_v3.py`). Kein
  trainiertes Modell und kein T1-Training/Tuning vor Praeregistrierung.
- **Keine Reproduktion der Paper-Ergebnisse.** Aussagen wie "schlaegt X auf 28 Benchmarks"
  stehen im Engine-Header als **Zitat der jeweiligen Autoren** (`[GAOT25]`), nicht als
  Ergebnis dieses Repos. Hier wurde **nichts davon nachgemessen**.
- **Keine trainierten Gewichte, keine Datensaetze, kein Benchmark-Resultat.**
- **Kein betriebener Dienst, keine Bibliothek mit stabiler API.** Kein PyPI-Paket, kein
  SemVer-Versprechen; Import-Pfade koennen sich aendern.
- **Kein Nachweis physikalischer Korrektheit** einer PDE-Loesung (keine Validierung). Die Gates
  pruefen Software-Integritaet (Determinismus, Round-Trips, Shapes). Einzige Verifikation einer
  Diskretisierung sind die Phase-0-Referenzfaelle (unten): Code- und Loesungsverifikation gegen
  exakte bzw. hergestellte Loesungen, eng begrenzte Contexts of Use, keine Physik-Aussage.

## Reifegrad / Evidenz (Lauf 2026-10-15: #61-Stapel #62..#66, #70, #72 + Spektral-Audit #71 + Haertungen; lokal Windows, Exit-Codes gesehen)

> **Phase 0b I1: V1 BLOCKED (historisch), V2 erfuellt (E1012-07 = A, 2026-10-12).**
> - **V1** (`rtol = 1e-13` auf dem **wahren** Residuum `||b - A x||`, `spec/HEX_FV_SOLVER_v1.md`
>   Abschnitt 5) bleibt rot und als Negativ-Befund stehen: N=192 stagniert bei 2.344e-13 > 1e-13
>   (die eps-Untergrenze `eps * || |A| |x| || / ||b||` liegt dort bei 1.11e-12). Ein Test nagelt
>   dieses Rot fest (`TestMmsI1V1Historical`).
> - **V2** (`spec/HEX_FV_SOLVER_v2.md`): Annahme bei `||b - A x|| <= tol_eff ||b||`,
>   `tol_eff = max(rtol, 8 u (|| |A| |x| || + ||b||) / ||b||)`, `u = 2^-53`, aber erst, wo V1
>   aufgibt. Ergebnis @d4acb27 (`results/2026-10-12_phase0_reference_i1-v2.json`, exit 0):
>   N=24/48/96 `converged` (7.19e-14 / 5.19e-14 / 9.29e-14 <= 1e-13); **N=192
>   `at_rounding_floor`**: 2.344e-13 > 1e-13, <= tol_eff 4.437e-12, Rueckwaertsfehler 4.69e-17.
>   Auf N=192 ist 1e-13 **nicht** erreicht. Loeser-Decke von `hex-mms-tri` wieder
>   `verified_numerical_reference`, jetzt unter I1 V2.

| Gate | Kommando | Ergebnis |
|---|---|---|
| Lint | `ruff check src scripts` | 0 Befunde, exit 0; die Engine `src/hexantk_v3.py` ist seit 2026-10-27 im Gate |
| Engine | `python scripts/engine_gate.py` | 49/49 Doctests + 26/26 Selftests, exit 0 |
| Engine-Abdeckung | `coverage run --include=src/hexantk_v3.py` (eigene rcfile, nicht `[tool.coverage]`) | kein Gate, nur Messung: Branch-Coverage der Engine 86 % aus dem Engine-Gate allein, 87 % mit den Toolkit-Tests, die sie per Pfad laden (2026-10-27 @733961a, Windows; `results/2026-10-27_gesamtbericht_local-gate.log`). Die 95 % der Coverage-Zeile gelten nur fuer `src/toolkit/` |
| Provenance | `python scripts/verify_sources.py` | 50 Toolkit- + 17 Spec-Eintraege + 11 Script-Eintraege, exit 0 |
| Toolkit-Tests | `python -m unittest discover -s src/toolkit -t src/toolkit -p "test_*.py"` | 1040 Tests, Ergebnis und Exit-Code im Gate-Log `results/2026-10-30_issue82_local-gate.log` (Issue #82: endliche Daten liefern nie `inf`, Manifest-Slot `worktree_dirty`, Break-even an die Gleich-Genauigkeits-Regel gebunden, Pigeonhole-Zitat; 10 neue Tests, 8 davon rot gegen a19c992, dazu zwei Codex-Befunde auf #93 rot reproduziert und behoben); Vorlaeufer `results/2026-10-30_ddp-runner_local-gate.log` (1030 Tests; DDP-Integration des Runners, `spec/DDP_RUNNER_v1.md`: 55 neue Tests, davon 22 aus einem echten 2-Rank-gloo-Lauf; Linux-Container, Python 3.12, torch 2.6.0); Vorlaeufer `results/2026-10-27_gesamtbericht_local-gate.log` (975 Tests; Gesamtbericht 2026-10-27: README-Phase-0-Tabelle an `results/2026-10-12_phase0_reference_i1-v2.json` gebunden (Torus p 1.997 -> 1.996), Engine im Lint-Gate, nicht importierbare Testmodule werden benannt statt gezaehlt; jeder neue Befund-Test gegen c5f5072 rot); Vorlaeufer `results/2026-10-26_xf-r88-r98_local-gate.log` (971 Tests; Cross-Family GPT-6 R88..R98: XF-R88-M2 Callable nur unter einem Namen, der zu ihm fuehrt; XF-R34 geladene Kopie muss den Checkpoint-Digest zurueckgeben; Windows, Python 3.12, torch 2.6.0+cpu, neuraloperator-Lane lokal mitgefahren); Vorlaeufer `results/2026-10-23_r87-metadata-budget_local-gate.log` (961 Tests; Codex-Runde #80 @426c97f: obere Fehlerschranke 0 wird entschieden, result_digests-Werte als SHA-256 geprueft; jeder neue Befund-Test gegen 426c97f rot); Vorlaeufer `results/2026-10-23_phase1-r86_local-gate.log` (895 Tests, HNT-R86-03 ganze Digest-Strings, HNT-R86-04 Break-even jenseits des Float-Bereichs, Issue #82, Cross-Family GPT-6 R86/R87; jeder neue Befund-Test gegen den alten Code rot); Vorlaeufer `results/2026-10-21_phase1-equalita-r5_local-gate.log` (891 Tests; Equalita R5-F1: jeder Baseline-Punkt traegt auch seine Einstellungen kappa/t_final/Task-Konfig-Digest/rtol/warmup/repeats/Timing-Digest/clock und wird daran geprueft, R5-F3 einmaliges Lesen, R5-F4 Plausibilitaet; 46/46 Mutanten rot); Vorlaeufer `results/2026-10-21_phase1-equalita-r4_local-gate.log` (888 Tests; Equalita R4-F1: jeder Baseline-Punkt traegt seine Mess-Koordinaten steps/n/domain/sample_count und wird daran geprueft, Record aus den Punkten, API-Bruch `baseline_curve`; R4-F3 AMP; 17 Mutanten rot); Vorlaeufer `results/2026-10-21_phase1-equalita-r3_local-gate.log` (882 Tests; Windows, Python 3.12, torch 2.6.0+cpu, ohne neuraloperator; Equalita R3: Baseline-Kurve an das vorregistrierte `steps`-Gitter gebunden, Manifest-Inhalt an die Vorregistrierung, Float-Budget-Widerspruch abgelehnt, 13/13 Mutanten rot); Vorlaeufer `results/2026-10-21_phase1-hexa-tim-01_local-gate.log` (877 Tests; GPT-6 HEXA-TIM-01: Latenz nur mit den vorregistrierten 10 Wiederholungen je Sample, 4/4 Mutanten rot), `results/2026-10-21_phase1-protocol_crossfamily_local-gate.log` (875 Tests; GPT-6 HEX-P1-01 Entscheidung aus Rohfehlern, HEX-P2-02 Float-Budget-Werkzeug, Paarung per `sample_id`, 12/12 Mutanten rot), `results/2026-10-20_phase1-protocol_r2_local-gate.log` (863 Tests; zweite Review-Runde: Seed-/Sample-Bindung, NEP 50, `low >= 0`, TOCTOU, 5/5 Mutanten rot), `results/2026-10-20_phase1-protocol_fix_local-gate.log` (862 Tests; Review-Fixes F-E1, D4/D5/D10, F-S4, E3, Mutanten per `diskriminierung.py` rot), `results/2026-10-19_phase1-protocol_local-gate.log` (856 Tests, Linux-Container; Phase-1-Protokoll-Instrumente, 12/12 Mutanten des neuen Moduls rot), `results/2026-10-17_issue76-codex-followup_local-gate.log` (819 Tests, Windows; Codex-Folgebefunde #76, jeder neue Test gegen den alten Code rot), `results/2026-10-17_pr71-fd-kernel-rows_local-gate.log` (809 Tests, FD-Pruefung auf den Kernel-Zeilen), `results/2026-10-16_pr71-codex-round_local-gate.log` (808 Tests, @59693d8; #61-Stapel + Spektral-Audit + Haertungen 2026-10-15, mit Negativ-zuerst- und Mutanten-Beleg), `results/2026-10-13_spectral-audit_local-gate.log` (@80a91d3) und 701 Tests im Gate-Log `results/2026-10-12_pr61-i1-v2_local-gate.log` (I1 nach V2; das V1-Rot ist ein erwarteter Befund im Test `TestMmsI1V1Historical`, kein roter Test); Skips je Plattform: Gate-Log |
| Coverage | `python -m coverage run ... && python -m coverage report` | Branch-Coverage ueber `fail_under = 87`, exit 0; Ist-Wert nur im Gate-Log (O2: eine ungepinnte Prozentzahl im README veraltet mit jedem Merge) |
| Mutation | `python scripts/mutation_probe.py` | 125/125 Mutanten getoetet, volle Probe 2026-10-30 unter Linux (Container, Python 3.12, torch 2.6.0) nach Issue #82 (5 neue Mutanten: Ueberlauf-Rueckfall, Skalierung, nur nicht-endliche Eintraege ersetzen, `worktree_dirty`-Typ, Break-even-Bereich), Baseline gruen, exit 0; Log `results/2026-10-30_issue82_local-gate.log`. Vorlauf: 120/120 Mutanten getoetet, volle Probe 2026-10-30 unter Linux, Baseline gruen, exit 0; 35 neue Mutanten fuer die DDP-Integration (je ein Waechter aus: Buffer-Signatur mit Shape, OOM-Rettung beim Vorbereiten und nie werfend, effektiver AMP-Modus, Batch-Vorbereitung/Datenende vor dem Forward, Buffer-Sync, Config-Digest, Loader-Generatoren je Rank, Rank-0-Bindung unter Fault-Injection, Schritt-Status-Agreement inkl. Forward-Fehler, abgestimmte Konstruktion, Summen-Reduktion, Partitionspruefung, deterministischer Val-Sampler, Checkpoint-/Resume-Agreement, gleiche Schrittzahl, Gewichts-Digest, run_id, rank-eigene RNG-Stroeme, Adapter-Bypass, rohes Modell im Checkpoint, Rank-Ordner, set_epoch, Sampler-Padding); Log `results/2026-10-30_ddp-runner_local-gate.log`. Vorlauf: 84/85 Mutanten getoetet, volle Probe 2026-10-26 @82b974a unter Windows, Baseline gruen, exit 1: einzig M02 ueberlebt (Testklasse POSIX-only); 6 neue Mutanten fuer XF-R88-M2/XF-R34 getoetet. Die Probe davor (@56f9cc7) gab 82/85: M62 (dtype-Waechter, durch die neue Nachkontrolle maskiert) und M76 (Schluesseltyp-Waechter, seit R87-P2 #84 durch das Budget maskiert) ueberlebten, beide Tests auf die Meldung ihres Waechters geschaerft; Log `results/2026-10-26_xf-r88-r98_local-gate.log`. Vorlauf: 78/79, volle Probe 2026-10-23 nach Merge von main (#83) unter Windows, Baseline gruen, exit 1: einzig M02 ueberlebt (Testklasse POSIX-only, unter Windows nicht messbar); 3 neue Phase-1-Mutanten fuer HNT-R86-03/-04 getoetet; Log `results/2026-10-23_phase1-r86_local-gate.log`. Vorlauf: 50/51, volle Probe @9db9cbc unter Windows, Baseline gruen, exit 1: einzig M02 ueberlebt, und das ist unter Windows nicht messbar (Testklasse POSIX-only, alle skipped; so schon 2026-09-28). Der I1-Mutant (Iterationsfehler-Verhaeltnis entfernt) zielt seit 2026-10-12 auf `_i1_level_holds`. Letzter voller Lauf unter Linux: 49/49, exit 0, 2026-10-07 (alte Zaehlung) |

Lokale Umgebung des Laufs 2026-10-15: Windows 11, Python 3.12.10, `torch` 2.6.0+cpu, `numpy`
2.5.3, coverage 7.16.2, ruff 0.16.9, GINO-Gate mit `neuraloperator` 2.0.0, OpenBLAS-Kernel-Schleife;
Log `results/2026-10-15_pr71-hardening_local-gate.log` (Vorlaeufer
`results/2026-10-15_pr61-stack-collection_local-gate.log`, `results/2026-10-12_pr61-i1-v2_local-gate.log`, `results/2026-10-12_pr61-hardening_local-gate.log`)
(gepruefter Commit-SHA, Kommandos, Exit-Codes, Zahlen; der Commit mit Log und README aendert
keinen Code). Vorlauf 2026-10-07 (659 Tests am Commit f4137f8; danach kamen vier Tests ohne
eigenen Gate-Lauf hinzu, Codex P2 auf #61) in folgender Umgebung:
Linux (Cloud-Container), Python 3.12.3, `torch` 2.6.0+cu124
(PyPI-Wheel, CPU-Ausfuehrung ohne CUDA-Geraet; der CPU-Index download.pytorch.org war aus dem
Container gesperrt), `numpy` 2.5.3; Stand: Zweig `claude/hexa-ntk-project-overview-mzkd4p` auf
`main` bb4f986 (Issue #59, ReferenceSpec #33, Reality Anchor Phase 0, Phase-1-Instrumente). Die CI
faehrt dieselbe Kette auf **Python 3.12** mit `torch==2.6.0` (CPU-Wheel, Linux). Log:
`results/2026-10-07_phase1-instruments_local-gate.log` (Commit-SHA, Kommandos, Exit-Codes, Zahlen);
Vorlaeufe: `results/2026-10-07_phase0b_local-gate.log`, `results/2026-10-07_phase0-hex-heat-tri_local-gate.log`,
`results/2026-10-07_issue59_local-gate.log`. Letzter Windows-Lauf (Integration #58):
`results/2026-10-05_integration-r3_local-gate.log`; fruehere Integrationslaeufe:
`results/2026-10-05_integration-r2_local-gate.log`, `results/2026-10-05_integration_local-gate.log`.
Vorlauf 2026-09-28: `results/2026-09-28_codex46-review_local-gate.log`. Fruehere Laeufe
(2026-09-27: 378 Tests, 19/19 Mutanten; 2026-08-26: 166 Tests, 82 %) bleiben in ihren Logs
als historische Evidenz stehen. Die Zaehlangaben dieses Abschnitts zum aktuellen Baum
(Tests, Provenance-Eintraege, Doctests, Mutanten) nagelt `TestDocsContract` in
`src/toolkit/test_hex_timestep_contract.py` am Baum selbst fest.

**Resume an der Epochengrenze (seit 2026-09-27, Loader-Generator seit 2026-10-05).**
`HexTrainingRunner.maybe_resume()` laedt exakt `config.resume_from` (fail-closed bei
fehlender Datei, PR #31) und stellt neben Modell/Optimizer/Scheduler/RNG auch den
**EMA-Mittelwert**, den **Best-Wert** und den Zustand expliziter `torch.Generator` der
Train-/Val-Loader wieder her (z.B. aus `make_reproducible_dataloader_kwargs`; ohne ihn
wiederholte ein Resume die erste Permutation des Seeds). Nicht Teil des Zustands: eine
Position innerhalb einer Epoche, Worker-interner Zustand, eigene Sampler ohne `generator`.
Aeltere Checkpoints ohne Generator-Zustand: Warnung im Log, `loader_generators_missing` im
Bericht, der Shuffle beginnt beim Seed. Vorher startete das EMA nach einem Resume neu, und der Best-Wert begann bei
`None` — die erste Epoche nach dem Resume ueberschrieb ein besseres `best.pt`. Quelle der
Wahrheit ist `best.pt` im eigenen Checkpoint-Ordner: uebernommen wird dessen Wert, wenn er
dieselbe Policy traegt und mindestens so gut ist wie der im Resume notierte (besser = spaetere
Epoche desselben Laufs); fehlt `best.pt` oder ist es schlechter (Fremd-Resume, ersetzte Datei),
wird nichts uebernommen und nichts blockiert. Zusaetzlich muessen Lauf-ID und Metrikname
passen. EMA-Zustand wird nur zusammen mit seinem `ema_decay` gespeichert; ein Resume mit EMA
verlangt den aktuellen Decay und lehnt fehlenden oder anderen Decay mit `EMADecayError` ab,
bevor irgendetwas wiederhergestellt ist (kein stilles Hybrid-EMA). Aeltere Resume-Checkpoints
mit EMA-Zustand, aber ohne Decay-Metadaten, lassen sich nur ohne EMA fortsetzen (das EMA
startet dann neu) — ihr Decay ist nicht mehr pruefbar.
Regressionstest:
unterbrochener Lauf (2 + 2 Epochen) == ununterbrochener Lauf (4 Epochen), bitgleich fuer
Modell und EMA.

**Mid-Epoch-Resume (seit 2026-10-30, `spec/MID_EPOCH_RESUME_v1.md`).** Mit
`RunnerConfig(save_every_n_steps=n)` laesst sich ein Lauf auch mitten in einer Epoche fortsetzen:
der Iterator wird aus dem gesicherten Epochen-Startzustand neu erzeugt, die erledigten Batches
werden geladen und verworfen (Worker-RNG inklusive), dann laeuft das Training mit dem RNG der
Unterbrechung weiter -- bitgleich zum ununterbrochenen Lauf (getestet mit Workern, Dropout,
Python/NumPy-Stroemen und 2 DDP-Ranks mit BatchNorm). Grenze: `persistent_workers=True`, CUDA-RNG.

**DDP (seit 2026-10-30, `spec/DDP_RUNNER_v1.md`).** `RunnerConfig(distributed=True)` mit einem
`DistributedDataParallel`-Modell: `set_epoch` auf den Samplern, gleiche Schrittzahl je Rank, ein
gemeinsamer NaN-Skip vor `backward` (sonst Deadlock im Gradienten-All-Reduce), Validierung auf dem
ungewrappten Modul mit gepruefter exakter Partition (`ExactDistributedEvalSampler`; ein gepaddeter
`DistributedSampler` wird abgelehnt) und `val_nrmse` aus ueber die Ranks summierten float64-Summen;
Konstruktion (inkl. gleicher Konfiguration), Batch-Vorbereitung, Forward-Fehler, Datenende,
Checkpoint-Schritt und Resume enden bei einem Fehler auf irgendeinem Rank auf allen Ranks mit einer
Ausnahme statt einem Haenger; ein Adapter muss das DDP-Modell selbst wrappen (sonst laeuft der
Forward an DDP vorbei); Buffer (BatchNorm, EMA) werden vor Validierung und Save von Rank 0
uebernommen, damit alle Ranks das Modell bewerten, das Rank 0 speichert; rank-eigene RNG-Stroeme
und Loader-Generatoren ueberleben ein Resume.
Getestet mit echten 2 gloo-Ranks auf CPU (== Single-Process-Lauf bis 1e-12, Resume bitgleich);
NCCL/CUDA, mehr Ranks und Hosts sind nicht gelaufen.

**Metrik:** `val_nrmse` ist jetzt der gepoolte relative L2-Fehler
`sqrt(sum |pred - y|^2 / sum |y|^2)` ueber den Validierungssatz (vorher eine Kopie des
MSE-`val_loss`, als "Placeholder" markiert, aber Default-Kriterium fuer `best.pt`).

Toolkit-Stand 2026-04-06, Engine-Stand 2026-05-25. Das gehaertete Experten-Review liegt
in `spec/`, offene Lint-/Test-Schuld in `docs/ROADMAP.md`.

## Kern

- **Engine:** `src/hexantk_v3.py` (1773 Zeilen; Checkpoints, EMA-Policy, OOM-sicherer
  Train-Step, Operator-Vertraege, 26 Selftests inline)
- **Toolkit:** `src/toolkit/` — 22 Module + 28 Testmodule (DDP-Koordination, Parameter-NTK, NTK-Trajektorie, Benchmark, Checkpoint-Integration,
  Dataloader-Utils, Dataset-Split, EMA, Failure-Signature, Hex-FV-Loeser, Lineare POD-Ridge-Baseline,
  Operator-Adapter,
  Phase-0-Referenzfaelle, Phase-1-Aufgabe T1, Phase-1-Protokoll, ReferenceSpec, Resume-Utils, Runner,
  Spektral-Audit, Stress, Warmstart-Utils; dazu die Tests des Provenance-Gates, des DCP-Bundle-Protokolls und des
  Zeitschritt-Vertrags)

## Reality Anchor Phase 0: Referenzfaelle (#60)

Drei kanonische Hex-Faelle mit exakter oder hergestellter Loesung verifizieren den
mechanistischen Loeser `src/toolkit/hex_fv_solver_hex_v3.py` (`spec/HEX_FV_SOLVER_v1.md`:
Dreieck mit Dirichlet-Raendern, periodischer Hex-Torus, Two-Point-Flux mit symmetrischen
Kantenleitfaehigkeiten, Positivitaets-Gate, explizites Euler, CG) und durch ihn die
Engine-Funktion `hex_laplacian_stencil_weight`. Alle drei Studien laufen bei jedem CI-Lauf in
`tests (cpu)`; Gate: beobachtete Ordnung p in [1.95, 2.05] (Celik 2008), GCI berichtet und den
exakten Fehler deckend, vier Level mit konstantem r > 1.3.

| Fall | Spec | Was geprueft wird | p (fein / grob) | GCI_fine vs. exakter Fehler | Decke |
|---|---|---|---|---|---|
| `hex-heat-tri` | `REFERENCE_CASE_HEX_HEAT_TRI_v1` | Lame-Moden = exakte diskrete Eigenvektoren, Dirichlet | 1.998 / 1.993 | 2.5e-4 / 2.0e-4 | `verified_numerical_reference` |
| `hex-heat-torus` | `REFERENCE_CASES_HEX_PHASE0B_v1` | ebene Wellen, periodische Assemblierung, Erhaltung, Isotropie in 0 / 10.9 / 30 Grad | 1.996 / 1.986 | 4.4e-4 / 3.5e-4 | `verified_numerical_reference` |
| `hex-mms-tri` | `REFERENCE_CASES_HEX_PHASE0B_v1` | variables kappa (Mittelpunkt-Leitfaehigkeiten), Quelle, CG; Quelle gegen unabhaengige FD | 2.001 / 2.006 | 1.1e-4 / 9.1e-5 | Orakel `mathematical_oracle`, Loeser `verified_numerical_reference` unter I1 V2 (seit 2026-10-12, E1012-07; N=192 nur `at_rounding_floor`); unter I1 V1 `comparison_only` (historisch, BLOCKED); bis 2026-10-07 `verified_numerical_reference` mit rekursivem Residuum, ueberholt |

Jede Decke gilt **nur** fuer ihren Context of Use. Diskriminierung (Tests + Mutationsprobe):
falsches Gewicht, Gewichtsfehler erster Ordnung, Neumann-artige Slots, kappa am Knoten statt
am Kantenmittelpunkt (nur Ordnung 1), Quelle ohne `grad kappa . grad u`, vorzeitiger CG-Abbruch,
nicht-konservativer Operator und falscher Torus-Umbruch fallen durch; das arithmetische Mittel
der Knoten-kappa (ebenfalls zweiter Ordnung) besteht. Positivitaets-Gate auf dem
assemblierten Operator (#60 Punkt 1): Vorzeichenstruktur + `dt <= min 1/|diag|`, sonst
`ValueError`. Evidenz: `python scripts/oracles/hex_phase0_reference_oracle.py` ->
`results/2026-10-07_phase0_reference.json` (Stand vor der Wahr-Residuum-Stoppregel). Am
2026-10-12 meldete derselbe Schreiber unter I1 V1 `[PHASE0] FAIL` (exit 1): `hex-mms-tri`
verfehlt I1 auf N=192. Unter I1 V2 meldet er `[PHASE0] PASS` (exit 0) @d4acb27, gebunden
(sauberer Arbeitsbaum, verifizierter HEAD): `results/2026-10-12_phase0_reference_i1-v2.json`.
Er gibt nur exit 0, wenn der Quellstand gebunden ist. Phase 0 ist damit unter I1 V2 gruen;
das ist lokal unter Windows gemessen, CI offen. Kein Lern-Backend vor gruener Phase 0 (#60).

## Phase 1: Instrumente und Vorregistrierung (Entwurf)

Vor jedem Training festgelegt, ohne Modell und ohne Lern-Claim:

- **Aufgabe T1 `hex-heat-operator`** (`spec/HEAT_OPERATOR_TASK_v1.md`,
  `src/toolkit/heat_operator_task_hex_v3.py`): Waerme-Halbgruppe `u0 -> u(T)` auf dem
  Dirichlet-Dreieck (ID) und dem Hex-Torus (OOD BC-Familie). Exakte Wahrheit aus der
  Eigenfunktionsbasis (15 bzw. 40 Moden, spektraler Prior, Seeds je `sample_id`, also
  aufloesungsunabhaengig); ID-Manifest mit SHA-256 je Sample und Datensatz; getrennte Pruefung
  von Integritaet (Bytes) und Reproduzierbarkeit (Neugenerierung, 1e-12 - plattformuebergreifende
  Bitgleichheit wird nicht versprochen). Metriken relativ L2 (Zellflaechen), relativ H1
  (Energieform des verifizierten Operators, Dirichlet-Kanten inklusive), relativ Max.
- **Mechanistische Baseline**: Crank-Nicolson mit warm gestartetem CG (`HEX_FV_SOLVER_v1` §4),
  verifiziert zweiter Ordnung in Raum und Zeit (rel. L2 2.2e-2 -> 5.4e-3 -> 1.3e-3 -> 3.3e-4).
- **Vorregistrierung** `spec/PREREGISTRATION_PHASE1_v1.md`: **DRAFT, nicht bindend**. Bindend erst
  nach Owner-Freigabe (Entscheidungen D1-D8); bis dahin kein Training/Tuning auf T1-Daten.
  GINO (`neuraloperator==2.0.0`, MIT, auf PyPI verifiziert) gegen diese Baseline nach
  McGreivy & Hakim (gleiche Laufzeit / gleiche Genauigkeit, Kosten-Genauigkeits-Kurve,
  Break-even inkl. Training und Datengenerierung), >= 5 Seeds, IQM + stratifiziertes Bootstrap.
- **Spektral-Audit** (`spec/SPECTRAL_AUDIT_v1.md`, `src/toolkit/spectral_audit_hex_v3.py`;
  Sekundaerinstrument, als Sekundaermessung Teil der Vorregistrierung:
  `spec/PREREGISTRATION_PHASE1_v1.md` verlangt es fuer jeden trainierten Seed): Eingabe-Jacobi `J` eines beliebigen
  Operators (numpy, torch, `nn.Module`), projiziert auf die W-orthonormierte T1-Basis,
  `S = Q^T W J Q`. Liefert Frequenzgang je Mode, Modenkopplung und Antwort ausserhalb der Basis
  gegen geschlossene Orakel (semidiskret `exp(-kappa lambda_h T)`, Crank-Nicolson, T1-Wahrheit
  `exp(-kappa lambda T)`). Fail-closed bei NaN/Inf, Budget, nicht orthonormaler Basis. Ist
  NICHT die Parameter-NTK (die liegt separat in `src/toolkit/param_ntk_hex_v3.py`, Spec
  `spec/PARAM_NTK_v1.md`). Dazu eine lineare POD-Ridge-Baseline
  (`src/toolkit/linear_baseline_hex_v3.py`) mit vorab fixierten Rang-/Regularisierungsregeln
  und Leckage-Waechter, nur auf synthetischen Daten getestet.
- **Protokoll-Instrumente** (`spec/PHASE1_PROTOCOL_v1.md`, `src/toolkit/phase1_protocol_hex_v3.py`,
  2026-10-19; numpy, kein Modell, kein T1-Fit): legen VOR jedem Lauf fest, wie die Zahlen der
  Vorregistrierungs-Abschnitte 4-7 berechnet werden. Timing-Protokoll (3 Warm-ups, 10 Wiederholungen,
  Median + Perzentil-Bootstrap), Kosten-Genauigkeits-Kurve der Crank-Nicolson-Baseline je `(N, steps)`,
  Pareto-Front mit log-log-Interpolation ohne Extrapolation, IQM (getrimmtes Mittel, Agarwal et al. 2021)
  mit zweistufigem gepaartem Bootstrap (Seeds, dann Sample-IDs; ein Indexsatz fuer alle Testsets, damit
  die Aufloesungsachse gepaart bleibt), Entscheidungsregeln (gleiche Laufzeit / gleiche Genauigkeit /
  Baseline-Floor / kein Claim ausserhalb des Messbereichs), Break-even-Zahl und Lauf-Manifest mit
  SHA-256. Orakel: IQM-Closed-Forms, exakte Reproduktion eines Potenzgesetzes, Paarung unter Skalierung
  bitgenau, Sweep der Baseline zweiter Ordnung in `dt` gegen die exakte semidiskrete Loesung; 12/12 Mutanten
  (ohne Trimmen, Paarung gebrochen, lineare Interpolation, Extrapolation, `<=` statt `<`, ...) rot.
- **D4-Entscheidungshilfe Radius-Ties** (`scripts/oracles/gino_radius_tie_survey.py`,
  `results/2026-10-19_gino_radius_tie_survey.json`): mit der Adapter-Regel aus
  `spec/GINO_ADAPTER_PHASE1_v2.md` gemessen, dass der Phase-1-Default (latent 32, `radius_factor` 2.5) auf
  den Aufloesungs-Gittern N=64 und N=128 je 4 Ties hat (dort verweigert der Adapter), waehrend
  `latent_resolution` 16, 24 oder 48 mit `radius_factor` 1.5 / 2.25 / 2.5 / 2.75 auf allen fuenf
  vorregistrierten Gittern tie-frei sind (12 von 48 Kandidaten). Die Wahl ist Owner-Entscheid D4; der Default
  bleibt unveraendert. **Korrektur 2026-10-20 (Review-Befund F-S4):** tie-frei heisst nicht "float32 und
  float64 bauen denselben Graphen". Die Survey vergleicht jetzt beide GNO-Nachbargraphen, wie das gepinnte
  Backend sie baut (neuraloperator 2.0.0, native Suche; 960 Graphen gegen die Backend-Suche geprueft, 0
  Abweichungen): latent 48 mit 1.5 / 2.25 / 2.5 flippt (8 / 8 / 3 Kanten), empfohlen werden nur noch die
  9 tie- und flip-freien Kandidaten (latent 16 und 24 mit 1.5 / 2.25 / 2.5 / 2.75, latent 48 mit 2.75;
  `results/2026-10-20_gino_radius_tie_survey_float32-graphs.json`). Nur 3 davon haben eine Marge von
  mindestens dem Doppelten des gemessenen float32-Distanzfehlers; geforderte Marge und eine breitere
  Tie-Toleranz im Adapter sind offene Owner-Fragen. **Nachtrag (zweite Review-Runde):** "flip-frei" gilt
  fuer diese Maschine; unter 864 emulierten float32-Summationsreihenfolgen flippt latent 48 / 2.75 in 248,
  die uebrigen acht in keiner. Empfohlen bleiben damit **8** Kandidaten (latent 16 und 24); Linux/CUDA
  nicht gemessen.
- **NTK-Trajektorie** (`spec/NTK_TRAJECTORY_v1.md`, `src/toolkit/ntk_trajectory_hex_v3.py`;
  PROPOSED, nicht Teil der Vorregistrierung): die Parameter-NTK an vorab deklarierten
  Checkpoints eines Laufs auf fester Eingabe. Berichtet relative Drift, Kernel-Distanz und
  -Geschwindigkeit (Fort et al. 2020), Modenkopplung in der Laplace-Basis, effektiven Rang und
  optional Kernel-Target-Alignment. Fail-closed: Checkpoint-Folge = deklarierte Folge,
  Schluessel/Dtype/Shape exakt, Eingabe-Mutation erkannt; byte-gleiche Checkpoints werden
  gemessen und je Eintrag markiert, nicht verweigert (HNT-R25-01: gleiche Bytes beweisen weder
  eine Kopie noch echte Stationaritaet);
  SHA-256 ueber die tatsaechlich gemessenen Bytes (inkl. neuraloperator-`_metadata`), Herkunft
  `unattested`. Orakel: geschlossene Formen, Skalen-/Basis-Invarianz, Lazy-Regime (breites Netz
  driftet weniger, Jacot 2018 / Lee 2019), echter GINO (2 Checkpoints, synthetischer Schritt).

## Reproduzieren

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt                  # zentrale Tool-Pins: ruff/coverage
python -m pip install "numpy==2.5.3"                         # wie CI und Evidence-Lanes
python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
python scripts/engine_gate.py                                  # 49 Doctests + 26 Selftests
python scripts/verify_sources.py                               # Provenance-Gate (SOURCES.md)
# GINO-Gate nur mit optionalen Paketen, wie im verpflichtenden ci/gate:
# python -m pip install -r requirements-operator.txt
# python -m unittest discover -s src/toolkit -t src/toolkit -p "test_gino_adapter_hex_v3.py" -v
python -m coverage run -m unittest discover -s src/toolkit -t src/toolkit -p "test_*.py"
python -m coverage report                                      # 1040 Unit-Tests (2026-10-30 Issue #82, DDP-Runner; 2026-10-27 Gesamtbericht; 2026-10-26 Cross-Family GPT-6 R88..R98, R87-P2 Metadaten-Budget, Merge main/#83, Phase-1-Protokoll, Codex #80 @426c97f, Cross-Family-Befunde GPT-6 bis R86, Equalita R3/R4/R5) + Coverage-Gate
python scripts/mutation_probe.py                               # optional: 125 Mutanten der Review-Fixes
python scripts/oracles/hex_phase0_reference_oracle.py         # Phase-0-Referenzfaelle (exit 0 = alle Gates gruen)
python scripts/oracles/gino_radius_tie_survey.py              # D4-Entscheidungshilfe: Radius-Ties + float32/float64-Graphvergleich je (latent, radius_factor) x Gitter (braucht requirements-operator.txt)
```

CI (`.github/workflows/ci.yml`), beide Jobs scheitern mit exit != 0 (beide sind
Required Checks: Branch-Schutz `main` gemessen 2026-10-27 per
`gh api repos/marcohost33-maker/hexa-ntk/branches/main/protection/required_status_checks`:
`contexts = ["gate", "tests (cpu)"]`, `strict = true`; der Bericht vom 2026-09-21 "nur `gate`" ist
damit ueberholt; Branch-Schutz liegt ausserhalb des Repos und kann sich ohne Commit aendern):
- **gate** — Lint (`ruff check src scripts`, inklusive Engine `src/hexantk_v3.py`: bis 2026-10-27 ausgenommen,
  solange ihr Pflegeort offen war; Issue #74 hat dieses Repo als kanonisch entschieden), `compileall`, Provenance,
  **zusaetzlich** bei vorhandenem GINO-Adapter: Installation von `torch==2.6.0` und
  `requirements-operator.txt` in einer isolierten CI-Jobumgebung, `neuraloperator==2.0.0`
  Versions-Check, dann `python -m unittest discover -s src/toolkit -t src/toolkit
  -p "test_gino_adapter_hex_v3.py" -v` inklusive echtem Forward, Backward und
  synthetischem Optimizerschritt (kein T1-Training). Fehlendes Backend => FAIL.
  Achtung: Abweichung von der noch nicht genehmigten D7-Preregistrierung, die
  `CI stays torch-only` vorsieht: siehe `docs/ADR/ADR-002-Phase1-GINO-required-gate-proposal.md`.
- **tests (cpu)** — Engine-Gate + Toolkit-Unit-Tests (1040, Stand 2026-10-30; Issue #82, DDP-Runner, Gesamtbericht 2026-10-27, Cross-Family GPT-6 R88..R98, R87-P2 Metadaten-Budget, Merge main/#83, Phase-1-Protokoll, Codex #80 @426c97f, Cross-Family-Befunde GPT-6 bis R86, Equalita R3/R4/R5) unter Branch-Coverage;
  danach die torch-freien FV-/Reference-Suites unter jedem portablen OpenBLAS-Kernel
  (`OPENBLAS_CORETYPE` Prescott/Sandybridge/Haswell/Zen, SkylakeX bei AVX-512), weil die
  CG-Stoppregel-Tests an der Rundung von BLAS-`ddot` haengen (2026-10-14). Das Spektral-Orakel (torch) bleibt ausserhalb: unter erzwungenem OPENBLAS_CORETYPE stuerzt es auf dem Runner ab (Exit 139, Haswell, 2026-10-16); seine CN-Toleranz ist stattdessen aus dem CG-Abbruch hergeleitet und lokal unter vier Kerneln gruen.

**Adapter-Ressourcenbudget (Wave19):** Vor kanonischer Gitterallokation werden
Arrayform, n und Dtype geprüft. Die GINO-Linie akzeptiert maximal n=512,
latent_resolution=128, hidden_channels=512, n_layers=16 und radius_factor=16.
Dies ist ein operatives Fail-Closed-Limit, **keine wissenschaftliche Modellgrenze**;
Erhöhungen benötigen Speicherbenchmark und versionierten Review.

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

Der vollstaendige Referenz-Block mit Kontext steht in `src/hexantk_v3.py:44-68`.

- **[Li23]** Z. Li, N. Kovachki, C. Choy et al., *Geometry-Informed Neural Operator for
  Large-Scale 3D PDEs*, NeurIPS 2023, arXiv:2309.00583. — Ursprung von **GINO**.
- **[GINOT25]** Q. Liu et al., *Geometry-Informed Neural Operator Transformer*,
  ScienceDirect S0045782525009405 (Dez. 2025).
- **[GAOT25]** S. Wen et al., *Geometry Aware Operator Transformer*, arXiv:2505.18781.
  Die 2D-Default-Hyperparameter in `GAOTConfig` sind aus dieser Arbeit uebernommen;
  fuer 3D gibt es dort **keinen** validierten Default (siehe `src/hexantk_v3.py:847-855`).
- **[Sarkar25]** S. Sarkar et al., *Physics- and geometry-aware spatio-spectral graph
  neural operator* (piG-Sp2GNO / mGNO), arXiv Aug. 2025.
- Hex-FV-Stencil: Lee & Tien 2014 / Lee 2017 (`src/hexantk_v3.py:659`).

Die Herkunft jeder Datei (inkl. der in-repo gehaerteten Ableitungen) ist in `SOURCES.md`
mit SHA-256 gepinnt und mit `scripts/verify_sources.py` nachpruefbar.

## Verwandte Repos

`coworker-dde` haelt eine weitere Kopie der v3-Engine (die Kopien sind auseinandergelaufen;
welche kanonisch ist, entscheidet Issue #74);
`HQST`, `hex-hqst`, `u2-dualspec`, `u6-bifurcation` gehoeren zur selben Serie.
**Diese Repos sind privat und von aussen nicht erreichbar.** Oeffentlich zugaenglich ist
aus der Serie nur [`phi-hex`](https://github.com/marcohost33-maker/phi-hex).

---
*Coworker Research / Coworkerz | Repo-Anlage 2026-06-04*
