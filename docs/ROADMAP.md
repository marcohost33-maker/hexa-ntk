# ROADMAP — HexaNTK

Ehrliche Liste der naechsten Schritte und offenen Schuld. Reihenfolge = Prioritaet.
Status-Konvention: `[x]` erledigt (mit Evidenz) · `[ ]` offen · `[~]` bewusst zurueckgestellt.

## Erledigt (verifiziert)

- [x] **Endlichkeitsvertrag `dt` beim Gruppieren (Issue #39, 2026-09-26).**
  `dataset_split_hex_v3`: mit `group_by_dt=True` wurde ein NaN-`dt` Teil des Gruppenschluessels;
  NaN != NaN, also bildete jedes Sample mit frischem NaN (`tensor.item()`, Arithmetik) eine
  eigene Gruppe und eine Trajektorie landete in Train UND Val (Repro auf `cdb81b2`: 50/50 Seeds).
  `_extract_dt_value` wirft jetzt fuer NaN/±inf (float, numpy float64, Skalar-Tensor) den
  Vertrags-`ValueError` `Sample 'dt' must be finite, got ...`, bevor der Schluessel gruppiert
  wird; ohne `group_by_dt` wird `dt` weiterhin nicht gelesen. Ein geteiltes NaN-Objekt verdeckt
  den Fehler (Identitaetsvergleich im Tupel) - die Tests erzeugen deshalb je Sample ein neues.
  _Evidenz: Rot 3 Methoden failures=36 -> 275/275 OK (skipped=2), Diskriminierung 8/8,
  `results/2026-09-26_issue39_local-gate.log`._

- [x] **Eingabevertrag `trajectory_id` + Zustandsreihenfolge Best-Checkpoint (Issues #36, #37,
  2026-09-25).** `dataset_split_hex_v3`: `int(trajectory_id)` warf fuer `inf` einen rohen
  `OverflowError` und schnitt nicht-ganzzahlige IDs still ab (1.5 und 1.7 → Gruppe 1, zwei
  Trajektorien in einer Split-Gruppe) → Vertrags-`ValueError` (`from exc`), exakte Integer
  unveraendert. `checkpoint_integration_hex_v3`: NaN wurde als Bestwert gespeichert und
  blockierte danach jede Verbesserung; `best_metric_value` wurde vor Konversion und Write
  gesetzt → jetzt validieren (NaN/±inf/10**400 → `ValueError`) → schreiben → committen.
  `runner_hex_v3` ueberspringt eine NaN/inf-Validierungsmetrik mit FailureSignature
  `NaN_or_Inf_Metric` statt abzubrechen (Muster Lightning `ModelCheckpoint`: NaN wird nie Best).
  _Evidenz: Rot 25 Tests failures=33 → 269/269 OK, Mutationsprobe 11/11, Differential alt/neu
  600/600 bitgleich, `results/2026-09-25_issues36-37_local-gate.log`._

- [x] **SOURCES.md vervollstaendigt + Provenance-Gate CI-enforced** — die 16 (jetzt 17)
  `src/toolkit/`-Eintraege sind erfasst (PR #5, LF-Normalisierung PR #6); `verify_sources.py`
  laeuft seit 2026-07-03 als blocking Step im CI-`gate`-Job (vorher existierte das fail-closed
  Skript, wurde aber nirgends automatisch ausgefuehrt — Integritaetsvertrag ohne Enforcement).
  _Evidenz: 17/17 PASS, `results/2026-07-03_coverage-baseline_gate.log`._
- [x] **Coverage gemessen + Gate gesetzt** — Branch-Coverage via `coverage.py` (7.15.0, gepinnt),
  Konfig in `pyproject.toml`, blocking im CI-`tests (cpu)`-Job. Ist-Wert **68%** ueber die 10
  Toolkit-Module (Engine wie beim Lint-Gate ausgenommen, Pflege via coworker-dde);
  `fail_under=65` als Ratchet knapp unter Ist. Zielkorridor 80–90% (Scientific-Python-Guide;
  Ratchet-Praxis: Schwelle folgt dem Ist-Wert nach oben, nie Wunschwert ueber Ist).
  _Evidenz: `results/2026-07-03_coverage-baseline_gate.log`._
- [x] **Testluecke `dataset_split_hex_v3.py` geschlossen** — einziges Toolkit-Modul ohne
  jeden Test (0% Coverage, wurde nicht einmal importiert), obwohl Leakage-Safety das
  Kernversprechen des Moduls ist. Neu: `test_dataset_split_hex_v3.py` (34 Tests: kein
  Trajectory-Overlap, Partition vollstaendig+disjunkt, Seed-Determinismus, group_by_dt +
  dt-Rundung, Override-Pfade, Subset-Wrapper, Fehlerpfade) → Modul **95%**, Suite 112→146
  Tests. SOURCES.md-Eintrag append-only ergaenzt (SHA-First).
  _Evidenz: 146/146 PASS, `results/2026-07-03_coverage-baseline_gate.log`._
- [x] **Atomic-Checkpoint-Writer cross-platform-Bug gefixt** — `AtomicCheckpointWriter.
  save_torch_payload_atomic` (`checkpoint_integration_hex_v3.py`) fsync'te (a) den Temp-File ueber
  einen read-only (`"rb"`) Handle und (b) unbedingt das Parent-Directory. Beides scheitert auf
  Windows mit `OSError(EBADF)` bzw. `OSError(EACCES)` → der "fuer Cluster-Preemption kritische"
  durable Write war auf nicht-POSIX komplett funktionsunfaehig (15/110 Toolkit-Tests rot). Fix:
  Temp-File ueber `"rb+"` flush+fsync (Pattern python-atomicwrites: fsync des *Write*-Handles);
  Directory-fsync POSIX-only gegated (auf Windows liefert `os.replace`/MoveFileEx die Durability).
  +2 Regressionstests. CI (ubuntu) war gruen, aber der Cross-Platform-Korrektheitsfehler blieb.
  _Toolkit-Modul (kein Engine-Kanon → kein coworker-dde-Sync). Evidenz: 112/112 Toolkit-Tests +
  unabhaengiges Oracle (atomar + byte-faithful + nonzero), Windows `os.name=nt`._
- [x] **Lint-Baseline gesetzt** — `ruff check` von non-blocking auf blocking, ruff==0.15.8
  gepinnt, Konfig in `pyproject.toml`. 16 Toolkit-Defekte gefixt (14 unused imports, 1
  Redefinition, 1 toter Block). _Evidenz: `results/2026-06-18_lint-baseline_gate.log`._
- [x] **CI-Test-Gate** — neuer `tests (cpu)`-Job: CPU-torch + Engine-Doctests/Selftests
  (`scripts/engine_gate.py`, exit-korrekt) + 110 Toolkit-Unit-Tests. _Evidenz: s.o._
- [x] **Pre-Commit-Hooks** — `.pre-commit-config.yaml` (ruff lint + Hygiene/Secret-Hooks).
- [x] **Engine-Lint-Schuld + `__main__` exit-Code (coworker-dde-Sync).** In der kanonischen
  Quelle `coworker-dde/src/hexantk_v3.py` gefixt (F401 `field` Z.76, F541 5× f-string ohne
  Platzhalter, `__main__` fail-closed via `sys.exit`) und byte-identisch hierher gesynct
  (git-Blob `3dd0b2af55bedd0b7385e9f6569ce84a0f89cc7c`, verifiziert `git hash-object` beide
  Repos). Ruff (E/F/W, line-length 120) auf der Engine jetzt clean (6→0). `src/hexantk_v3.py`
  bleibt bewusst in `extend-exclude` (Pflege via coworker-dde, kein hiesiger Lint-Enforce —
  verhindert, dass ein coworker-dde-Edit die HexaNTK-CI bricht); volle Lint-Enforcement wäre
  optional und ist ein separater Entscheid. _Evidenz: coworker-dde-Fix-PR + Engine-Gate PASS._

## Offen

- [ ] **Coverage-Ratchet Richtung 80–90%.** Ist 68% (Branch, 2026-07-03). Schwaechste Module,
  in Prioritaetsreihenfolge des Nutzens: `runner_hex_v3.py` **25%** (191 Stmts offen — groesster
  Hebel), `warmstart_utils_hex_v3.py` **43%**, `ema_hex_v3.py` **56%**, `resume_utils_hex_v3.py`
  **73%**. Pro geschlossener Luecke `fail_under` in `pyproject.toml` nachziehen.
  _Nachtrag 2026-09-25 (gemessen, py3.12/torch 2.6.0):_ Ist **83%** gesamt, `fail_under = 79`;
  `runner_hex_v3.py` 80%, `warmstart_utils_hex_v3.py` **43%**, `ema_hex_v3.py` **56%**,
  `resume_utils_hex_v3.py` 82%. Groesster Hebel jetzt warmstart/ema; Draft-PR #20 enthaelt
  einen aelteren Ratchet-Versuch (79→91), der gegen den aktuellen `main` neu zu pruefen ist.
- [ ] **Nachbarn der float→int-Klasse (P3, bewusst nicht in #36/#37).** `int(epoch)` /
  `int(global_step)` in `checkpoint_integration_hex_v3.py` (save_last/save_resume) und
  `resume_utils_hex_v3.py` werfen fuer `float('inf')` einen rohen `OverflowError` statt eines
  Vertrags-Fehlers. Schleifenzaehler, kein Datensatz-Metadatum; kein Zustand wird vorher
  veraendert. _Quelle: `results/2026-09-25_issues36-37_local-gate.log` [6/6]._
- [ ] **Nachbarn im Split-Modul (P3, bewusst nicht in #39).** (a) `group_by_dt=True`
  schluesselt nach exaktem `dt`: mischt eine Trajektorie `0.1` (float) und `tensor(0.1)`
  (float32 -> 0.10000000149...), zerfaellt sie in zwei Gruppen und landet in beiden Splits
  (gemessen 41/50 Seeds; mit `dt_round_ndigits=6` 0/50). Das ist dieselbe Mechanik wie
  `test_group_by_dt_separates_dt_variants` (mehrere `dt` je Trajektorie = gewollt getrennt) und
  widerspricht dem Modul-Docstring "No trajectory appears in both" - Produktentscheid offen.
  (b) `_normalize_dt_value`: `round(dt, negative ndigits)` nahe binary64-Max wirft rohen
  `OverflowError`; `dt_round_ndigits` wird nicht validiert (1.5 -> roher `TypeError`).
  _Quelle: `results/2026-09-26_issue39_local-gate.log` [5/5]._


## Wave 7 — PR #57 branch status (2026-10-05)

- [~] **Runner-/Checkpoint-Evidence-Semantik gehaertet.** Auf PR #57 emittiert
  `_validate` nur noch `val_loss`; ein echtes nRMSE bleibt dem Benchmark-Modul
  vorbehalten. Die konfigurierte Best-Metrik muss im Validation-Report vorhanden
  sein (kein stiller Fallback), und ein leerer Validation-Loader bricht fail-closed ab.
  Validation nutzt fuer Raw-/EMA-Gewichte denselben `OperatorAdapterV3`-Vertrag
  wie Training. _PR-CI: 355/355 Toolkit-Tests PASS._
- [~] **Resume-Restore-Reihenfolge korrigiert.** Scheduler-State wird nach
  Scheduler-Konstruktion, aber vor `optimizer.load_state_dict` restauriert;
  ein expliziter Regressionstest bindet die Reihenfolge.
- [~] **Coverage-Ratchet 79 -> 88 vorbereitet.** PR #57 misst unter
  Python 3.12.14 / torch 2.6.0+cpu **90 % Branch-Coverage**; neue gezielte
  Testmodule decken Resume- und Warmstart-Vertraege ab, EMA-Legacy/Fallback-Pfade
  sind erweitert. Erst nach Merge als Main-Evidenz markieren.
- [ ] **Nach PR #57:** Best-Selection-State beim Resume persistieren und
  EMA-validierte Best-Gewichte als eindeutig primären Selection-State abbilden.
- [ ] **MetricSpec-Migration:** korrigierte per-sample nRMSE-Semantik versionieren,
  Legacy-Reports nicht still umetikettieren; Nigin-Bridge entsprechend migrieren.
- [ ] **Stateful Loader / Distributed Recovery:** torchdata-StatefulDataLoader
  separat gegen committed/consumed Cursor falsifizieren; DCP/FSDP/2-rank als
  eigene Evidence-Linie, nicht als allgemeiner Exact-Resume-Claim.

## Zurueckgestellt (bewusst, mit Begruendung)

- [~] **Typing-Modernisierung (PEP 585/604).** Ruff `UP`-Regeln finden ~210 Stellen
  (`Dict`->`dict`, `Optional[X]`->`X | None`) in den gehaerteten Modulen. Wertvoll, aber grosse
  Aenderung — erst nach gruenem `tests (cpu)`-Gate als separater, einzeln verifizierter PR
  (Reality-Anchor). `select` in `pyproject.toml` dann um `UP`,`I` erweitern.
- [~] **`ruff format` durchsetzen.** Alle 16 Module sind nicht ruff-format-konform; ein
  Format-Gate wuerde grossflaechig reformatieren. Erst nach Typing-PR sinnvoll.
