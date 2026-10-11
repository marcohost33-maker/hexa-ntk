# ROADMAP — HexaNTK

Ehrliche Liste der naechsten Schritte und offenen Schuld. Reihenfolge = Prioritaet.
Status-Konvention: `[x]` erledigt (mit Evidenz) · `[ ]` offen · `[~]` bewusst zurueckgestellt.

## Erledigt (verifiziert)

- [x] **Issue #59 Follow-ups nach #58 (2026-10-11, PR aus `claude/hexa-ntk-next-steps-wwdwtz`).**
  Die drei offenen Befunde aus #58-Codex-Runde 3, jeder mit eigenem Test:
  _P2 Durability (`scripts/oracles/distributed_checkpoint_bundle_v1.py`):_ `finalize_bundle`
  machte nur die Verzeichnis-Eintraege der Shards durabel, nicht deren **Inhalt** — nach einem
  Power-Loss konnte der `COMMITTED.json`-Marker ueberleben, waehrend ein Shard verloren/abgeschnitten
  war. Neu `fsync_file` (POSIX-gated wie `fsync_directory`); `finalize_bundle` fsynct jetzt in der
  Reihenfolge Inhalt → Verzeichnis → Marker. `TestFinalizeDurabilityOrder` pinnt die volle
  Ereignisfolge (ein Mutant, der den Content-Flush auslaesst, verliert die `("file", …)`-Events).
  _P2 API (`benchmark_hex_v3.py`):_ `metric_semantics_label` ist Teil des v4-Vertrags
  (`spec/METRIC_SPEC_v4.md`), fehlte aber in `__all__` — ergaenzt; neuer Test pinnt alle vier
  oeffentlichen v4-Symbole gegen Spec-Text und `__all__` (fail-closed bei Drop).
  _P3 Quantisierung (`dataset_split_hex_v3.py`):_ der Precision-Twin-Guard verglich mit
  `math.isclose(…, abs_tol=0.0)`, sodass `0.0` und `1e-46` nie gleich waren, obwohl beide float32 `0.0`
  sind. Jetzt float32-exakter Vergleich (`_as_float32` via `struct`, Ueberlauf → ±inf): zwei dt-Keys
  einer Trajektorie sind derselbe Schritt gdw. sie auf denselben binary32-Wert runden. Der
  Multi-dt-Property-Pool verliert `5e-324` (teilt float32 `0.0` mit `0.0` → waere jetzt ein Twin).
  _Evidenz:_ Item 1 lokal verifiziert (torch-frei: DCP-Modul rot→gruen, 14/14 OK; rot = fehlendes
  `fsync_file` am alten Oracle), `ruff`/`compileall`/`verify_sources` rc 0 lokal;
  Items 2+3 beruehren torch-Module (`benchmark`, `dataset_split`) → im CI-`tests (cpu)`-Lauf
  verifiziert (`torch==2.6.0` CPU hier proxy-geblockt). Gate-Log
  `results/2026-10-11_issue59_local-gate.log`. README-Testzahl 528 → 530.

- [x] **Codex-Review #46 nach Merge von `main` (2026-09-28).**
  _EMA-Decay (PRRT_kwDOSwvSC86m2eei):_ die oeffentlichen Resume-Helfer nahmen `ema_state_dict`
  ohne `ema_decay` an, und der Loader uebersprang den Vergleich, sobald eine Seite fehlte (stilles
  Hybrid-EMA). Jetzt Pflicht beim Speichern (auch im Manager, auf jedem Rang) und beim
  Wiederherstellen, Gleichheit gefordert, `EMADecayError` vor jedem Restore. Entscheid: Alt-Checkpoints
  ohne Decay-Metadaten scheitern mit EMA (fail-closed), Ausweg Resume ohne EMA.
  _Zaehlangaben (PRRT_kwDOSwvSC86m2eeU):_ README-Zahlen aus eigenem Lauf, am Baum festgenagelt
  (`TestDocsContract`). Mutationsprobe: Decay-Mutant auf den neuen Code, drei Vertrags-Mutanten neu.
  _Evidenz:_ `results/2026-09-28_codex46-review_local-gate.log`.

- [x] **Review-Haertung 2026-09-27: Issues #40 und #42 geschlossen, Resume vollstaendig.**
  Alle Befunde vor dem Fix reproduziert (`results/2026-09-27_review-hardening_local-gate.log`).
  _Resume/Checkpoint:_ Der Resume-Checkpoint traegt jetzt den EMA-Zustand und den Best-Wert
  (`checkpoint_manager_state`); vorher startete das EMA nach Resume neu und `best_metric_value`
  bei `None`, sodass die erste Epoche ein besseres `best.pt` ueberschrieb. Uebernommen wird der
  Wert aus `best.pt`, wenn es dieselbe Policy, Lauf-ID (`run_id`) und denselben Metriknamen traegt
  und mindestens so gut ist wie der notierte Wert; ein anderes `ema_decay` beim Resume -> `ValueError`
  (Codex-Review auf #46). `CheckpointDurabilityError` (#40 P3-3): scheitert
  nach `os.replace` nur noch der Verzeichnis-fsync, wird der Best-Wert trotzdem committet (Speicher
  == Datei). `epoch`/`global_step` ueber `as_step_count` (inf/NaN/1.5/-1 -> `ValueError`, bool ->
  `TypeError`), in `load_resume_checkpoint` **vor** dem Ueberschreiben von Modell/Optimizer.
  `min_delta` NaN/inf -> `ValueError`. P3-2 per Test gepinnt.
  _Runner:_ Best-Checkpoint jede Epoche (vorher nur an `save_every_n_epochs`-Epochen), last/resume
  zusaetzlich nach der letzten Epoche; leere Validierung -> NaN + FailureSignature
  `Empty_Validation` statt Bestwert 0.0; Epoche ohne gesunden Schritt -> `mean_loss` NaN statt 0.0;
  `val_nrmse` = gepoolter relativer L2-Fehler statt MSE-Kopie; `RunnerConfig` validiert
  (`save_every_n_epochs=0` war ZeroDivisionError); `TrainingAnomalyDetector(window_size=1)` war
  StatisticsError, NaN-`spike_multiplier` schaltete die Erkennung ab.
  _Split (#42, #40 P3-1/5/6):_ Owner-Entscheid E0927-03 (Option B, woertlich in #42): die Invariante
  "keine Trajektorie in train UND val" gilt IMMER, unabhaengig von dt-Typen und `group_by_dt`.
  Umsetzung: die Split-Einheit ist die Trajektorie; unter `group_by_dt=True` bleiben die
  (trajectory_id, dt)-Gruppen fuer die Auswertung getrennt, landen aber immer auf derselben Seite
  (vorher 28/50 Seeds mit Trajektorie auf beiden Seiten bei dt 0.1+0.2); `val_fraction` und
  `min_val_groups` zaehlen Trajektorien; ein `val_group_keys_override`, der nur einen Teil der
  dt-Gruppen einer Trajektorie nennt -> `ValueError`. Bei einer Schrittweite je Trajektorie ist der
  Split je Seed unveraendert (gepinnt). dt-Schluessel einer Trajektorie, die sich nur unterhalb
  float32-Aufloesung unterscheiden (0.1 vs. `tensor(0.1)`), werfen weiterhin `ValueError` mit
  Hinweis auf `dt_round_ndigits` (Metadaten-Konsistenz). `trajectory_id`: 0-d
  numpy-Werte werden entpackt (1.5 -> Fehler statt 1), bool/bytes und nicht-kanonische Strings
  (`"012"`, `"+12"`, `" 12"`) abgelehnt, kanonische (`"12"`) bleiben gueltig; `Decimal('sNaN')`
  war bereits korrekt, jetzt gepinnt. `dt_round_ndigits` typvalidiert, Rundungs-Overflow ->
  `ValueError`.
  _Warmstart/EMA:_ Prefix-Transformationen scheitern bei Schluessel-Kollision (vorher still ein
  Tensor verworfen); optional `skip_shape_mismatch` mit Bericht; EMA-`decay` ausserhalb [0, 1]
  -> `ValueError`; `load_ema_state` respektiert `strict` auch im Full-State-Pfad.
  _Evidenz:_ neue Tests gegen die alten Modul-Bytes (HEAD 9538e87): failures=52, errors=16 (2
  Testmodule scheitern schon am Import neuer Namen; Shim-Lauf dieser 2 Module mit alter Logik:
  50 rot) -> mit Fix 372/372 OK, nach Codex-Review 378/378;
  Mutationsprobe 19/19 (mit Baseline-Pflicht) (`scripts/mutation_probe.py`, im Repo); Coverage 83 % -> **89,6 %**,
  `fail_under` 79 -> 87; warmstart 43 % -> 94 %, ema 56 % -> 88 %.

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

- [x] ~~**Coverage-Ratchet Richtung 80–90%.**~~ Erledigt 2026-09-27: 89,6 % Branch-Coverage,
  `fail_under = 87` (siehe oben). Schwaechste Module jetzt `runner_hex_v3.py` / `stress_hex_v3.py`
  (je 84 %). Draft-PR #20 (aelterer Ratchet-Versuch 79->91 + resume_from-Fix) ist damit inhaltlich
  ueberholt: resume_from kam mit PR #31, der Ratchet hiermit — Schliessen empfohlen.
- [x] ~~**Nachbarn der float→int-Klasse (P3).**~~ Erledigt 2026-09-27 (`as_step_count`, siehe oben).
- [x] ~~**Nachbarn im Split-Modul (P3, #42).**~~ Erledigt 2026-09-27 (siehe oben).
- [ ] **Evidenz-Nachtrag zu frueheren Eintraegen (#40 P3-4).** Die Angaben "Mutationsprobe 11/11"
  und "Differential alt/neu 600/600 bitgleich" (Eintrag #36/#37) stammen aus Skripten einer
  externen Sitzung, die nie ins Repo kamen; sie sind **aus dem Repo nicht reproduzierbar** und
  gelten hier nur als Bericht, nicht als Evidenz. Neue Mutationsbehauptungen nur noch mit
  Skript im Repo (Beispiel: `scripts/mutation_probe.py`).
- [ ] **`tests (cpu)` als Required Check.** Laut externem Bericht (2026-09-21) ist nur `gate`
  Required Check; ein roter Testjob blockiert den Merge also nicht. Nur der Repo-Owner kann das
  im Branch-Schutz aendern.
- [ ] **Mid-Epoch-Resume.** `dataloader_state_dict` wird gespeichert/geladen, der Runner nutzt es
  aber nicht (Resume nur an Epochengrenzen). Fuer lange Epochen auf Preemptible-Clustern relevant
  (Muster: `torchdata.StatefulDataLoader`).
- [ ] **DDP-Integration des Runners.** Nur `is_main_process`-Guard; kein `DistributedSampler.
  set_epoch`, kein Barrier nach dem Checkpoint-Write, Metriken nicht ueber Ranks reduziert.
- [ ] **CI-Reproduzierbarkeit:** `numpy` ist im Testjob ungepinnt (`pip install numpy`); ein
  numpy-Major-Sprung aendert die CI ohne Commit. Pin in `requirements-dev.txt` erwaegen.

## Zurueckgestellt (bewusst, mit Begruendung)

- [~] **Typing-Modernisierung (PEP 585/604).** Ruff `UP`-Regeln finden ~210 Stellen
  (`Dict`->`dict`, `Optional[X]`->`X | None`) in den gehaerteten Modulen. Wertvoll, aber grosse
  Aenderung — erst nach gruenem `tests (cpu)`-Gate als separater, einzeln verifizierter PR
  (Reality-Anchor). `select` in `pyproject.toml` dann um `UP`,`I` erweitern.
- [~] **`ruff format` durchsetzen.** Alle 16 Module sind nicht ruff-format-konform; ein
  Format-Gate wuerde grossflaechig reformatieren. Erst nach Typing-PR sinnvoll.
