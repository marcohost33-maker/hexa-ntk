# ROADMAP — HexaNTK

Ehrliche Liste der naechsten Schritte und offenen Schuld. Reihenfolge = Prioritaet.
Status-Konvention: `[x]` erledigt (mit Evidenz) · `[ ]` offen · `[~]` bewusst zurueckgestellt.

## Blockiert

- [ ] **BLOCKED (2026-10-12), historisch (V1): Phase 0b I1, `hex-mms-tri`.** Bleibt als
  Negativ-Befund stehen; gilt fuer den V1-Vertrag weiter (Test `TestMmsI1V1Historical`).
  Abgeloest fuer den geltenden Vertrag durch I1 V2 (unter Erledigt). Text vom
  2026-10-12 vor E1012-07, unveraendert:
  Seit der CG-Loeser seine Stoppregel
  am wahren Residuum `||b - A x||` entscheidet (PR #61, `spec/HEX_FV_SOLVER_v1.md` Abschnitt 5),
  verfehlt `hex-mms-tri` I1 auf N=192: das wahre Residuum stagniert bei 2.34e-13 > 1e-13. Die
  Loeser-Decke von `hex-mms-tri` steht damit auf `comparison_only`, nicht mehr auf
  `verified_numerical_reference`; drei Tests sind absichtlich rot. **Phase 0 ist nicht gruen**:
  kein Lern-Backend und kein Phase-1-Lauf, bevor I1 entschieden ist. Die Schwelle 1e-13 ist
  unveraendert; eine V2 mit anderer Schwelle oder Residuumsberechnung ist ein Owner-Entscheid
  (E1012-07). Gemessen und verworfen: Residual Replacement mit beibehaltener Suchrichtung statt
  Neustart (N=96 und N=192 laufen dann bis maxiter, Spec Abschnitt 5).
  _Evidenz:_ `results/2026-10-12_pr61-hardening_local-gate.log`.

## Erledigt (verifiziert)

- [x] **NTK-Trajektorie (#60 Punkt 2, 2026-10-22).** `PARAM_NTK_v1` (seit #71) misst die
  Parameter-NTK an einem Parameterpunkt; der Statuskommentar auf #60 vom 2026-10-20 nannte
  Punkt 2 deshalb noch offen. Neu: `spec/NTK_TRAJECTORY_v1.md` (PROPOSED),
  `src/toolkit/ntk_trajectory_hex_v3.py`: dieselbe Messung an einer vorab deklarierten
  Checkpoint-Folge eines Laufs (GPT-6-Roadmap 2026-10-19, Schritt G2). Relative Drift,
  Kernel-Distanz/-Geschwindigkeit (Fort et al. 2020), Modenkopplung in der Basis, effektiver
  Rang, optional Kernel-Target-Alignment; Null-Kerne ergeben `null`, nie `inf`. Bindung: die
  Digests rechnet das Modul selbst aus den gemessenen Bytes (State-Dicts, Eingabe, Basis,
  Parametersignatur, Kerne, Record); fremde Hash-Strings werden nicht angenommen (Kritikpunkt
  am GPT-6-Parallelmodul vom 2026-10-19). Befund am echten Backend: neuraloperator 2.0.0 legt
  Konstruktor-Metadaten (`_metadata`, mit `F.gelu` und der Klasse `SpectralConv`) in den
  State-Dict und entfernt sie in `load_state_dict` per `pop` aus dem uebergebenen dict. Der
  Vertrag bindet sie kanonisch (Callables per qualifiziertem Namen, Lambdas/gebundene Methoden
  verweigert), verlangt Gleichheit mit dem Modell (sonst andere Architektur) und laedt aus
  einer Kopie. Mutationsprobe: der Mutant "Schluesselnamen nicht gehasht" ueberlebte zuerst
  (der Split-Test deckte ihn nicht) -> Test fuer umbenannte Schluessel ergaenzt.
  _Evidenz:_ `results/2026-10-22_ntk-trajectory_local-gate.log`.
  Offen: welche Laeufe/Checkpoints/Eingaben in Phase 1 gemessen werden (Owner, Vorregistrierung);
  Laufzeitbudget fuer GINO-Groesse ungemessen.

- [x] **CI rot auf #61 (`tests (cpu)`, Head 6d219d9): CG-V2-Tests hingen am BLAS-Kernel
  (2026-10-14).** `test_at_rounding_floor_where_v1_stagnates` und
  `test_acceptance_hangs_on_true_comparisons` setzten voraus, dass V1 `rtol=1e-16` auf
  `_hex_system(12)` nie erreicht. Das erreichbare wahre Residuum liegt selbst bei ~1e-16 und haengt
  von der Summationsreihenfolge des OpenBLAS-`ddot`-Kernels ab: mit `OPENBLAS_CORETYPE` =
  Haswell/Zen/Prescott konvergiert V1 (9.9e-17 / 9.8e-17), mit SkylakeX/Sandybridge stagniert es
  (lokal reproduziert; der CI-Lauf scheitert genauso; numpy 2.5.4 ist NICHT die Ursache). Auf Zen waren beide
  Tests unabhaengig vom Code rot (null Trennschaerfe); zwei weitere Tests (Skalierungs-Invarianz
  n=12, nicht-endlicher Floor) liefen dort still den Konvergenz- statt den Floor-Pfad.
  Fix (nur Tests): `_RTOL_BELOW_ROUNDING = 1e-17` (~u/11) fuer jede "rtol unerreichbar"-Praemisse;
  jeder Kernel stagniert (bestes Residuum 9.4e-17 .. 3.8e-16, <= 683/2000 Iterationen); der
  Skalierungstest pinnt den Status je Fall. CI-Schritt "FV solver + reference case across OpenBLAS
  kernels" faehrt die torch-freien FV/Reference-Suites unter Prescott/Sandybridge/Haswell/Zen (+SkylakeX
  bei AVX-512) und faellt mit den alten Tests sofort. Mutanten der V2-Akzeptanz (NaN akzeptiert,
  nie akzeptieren, Floor >= 1 akzeptiert) auf Zen und SkylakeX getoetet.
  _Phase-0-Evidenz kernelunabhaengig im Urteil:_ `hex_phase0_reference_oracle.py` unter allen fuenf
  Kerneln: 55/55 Urteilsfelder gleich (Kriterien inkl. I1, Ceilings); p und GCI_fine identisch auf
  4 Stellen, physikalische Groessen (GCI, Fehler) <= 5e-11 relativ verschieden; nur
  Rundungsdiagnosen (Eigen-Residuum ~1e-12, CG-Residuum ~1e-16, 7 Iterationsfelder) schwanken
  kernelabhaengig. _Evidenz:_ `results/2026-10-14_cg-v2-blas-kernel_local-gate.log`.

- [x] **Phase 0b I1 V2 (E1012-07 = A, 2026-10-12).** Marco hat entschieden: I1 bekommt eine
  versionierte V2 mit erreichbarer Schwelle; V1 bleibt als historischer, ehrlich roter Vertrag
  stehen (Blockiert, oben). `spec/HEX_FV_SOLVER_v2.md`: Annahme bei
  `||b - A x|| <= tol_eff ||b||`, `tol_eff = max(rtol, 8 u (|| |A| |x| || + ||b||) / ||b||)`,
  `u = 2^-53` (c = 8: Zeilen mit 7 Termen plus die Subtraktion von b, gamma_8-Schranke der
  Residuumsauswertung), konsultiert erst, wo V1 aufgibt; Status `converged` /
  `at_rounding_floor`, Rueckwaertsfehler `eta` berichtet. Ergebnis @d4acb27: N=24/48/96
  `converged`, N=192 `at_rounding_floor` (2.344e-13 > 1e-13, <= tol_eff 4.437e-12,
  eta 4.69e-17; 1e-13 dort NICHT erreicht); Loeser-Decke `verified_numerical_reference` unter
  I1 V2. Lokal Windows, CI offen.
  _Evidenz:_ `results/2026-10-12_phase0_reference_i1-v2.json`,
  `results/2026-10-12_pr61-i1-v2_local-gate.log`.

- [x] **Phase-1-Instrumente (#60, 2026-10-07).** Ohne Modell und ohne Training:
  Crank-Nicolson mit warm gestartetem CG im Loeser (exakter Verstaerkungsfaktor auf
  Eigenvektoren, Ordnung 2 in der Zeit, unbedingt L2-stabil, massenerhaltend auf dem Torus);
  Aufgabe T1 `hex-heat-operator` (`spec/HEAT_OPERATOR_TASK_v1.md`): exakte Wahrheit aus der
  Eigenfunktionsbasis ((m,n)/(n,m) als ein Eigenraum per Gram-Rang geprueft), Seeds je
  `sample_id`, Manifest mit SHA-256, Integritaet vs. Reproduzierbarkeit getrennt, Split ueber
  `dataset_split_hex_v3` (Integrationsbefund: der Splitter verlangt das `samples`-Attribut),
  Metriken rel. L2/H1(Energieform)/Max, Baseline zweiter Ordnung. Vorregistrierung als DRAFT.
  _Evidenz:_ `results/2026-10-07_phase1-instruments_local-gate.log`.

- [x] **Reality Anchor Phase 0b: `hex-heat-torus` + `hex-mms-tri`, Loeserschicht (#60, W1-K08,
  2026-10-07).** Damit ist Phase 0 komplett (0a + 0b). Neue mechanistische Schicht
  `hex_fv_solver_hex_v3.py` (`spec/HEX_FV_SOLVER_v1.md`): Dreieck-Dirichlet- und periodisches
  Torus-Gitter, Two-Point-Flux mit exakt symmetrischen Mittelpunkt-Leitfaehigkeiten,
  Positivitaets-Gate (jetzt auch variables kappa), explizites Euler, CG (SPD, fail-closed); die
  0a-Logik ist bitgleich umgezogen (0a-`study_sha256` unveraendert 6a366cdb...).
  `spec/REFERENCE_CASES_HEX_PHASE0B_v1.md`: Torus mit ebenen Wellen (exakt, Erhaltung C10,
  Isotropie in drei nicht aequivalenten Richtungen), MMS mit variablem kappa + Quelle
  (Orakel O1-O3 inkl. unabhaengiger FD-Pruefung der Quelle, Loeser M1-M3, Celik C5-C9 auf
  dem Zentroid-Wert, Iterationsfehler I1); zwei ReferenceSpecs fuer MMS (Orakel
  `mathematical_oracle`, Loeser `verified_numerical_reference`). Diskriminierung u.a.: kappa am
  Knoten (nur Ordnung 1), Quelle ohne Gradiententerm, vorzeitiger CG-Abbruch, nicht-konservativer
  Operator; das arithmetische Mittel der Knoten-kappa (zweiter Ordnung) besteht.
  _Evidenz:_ `results/2026-10-07_phase0b_local-gate.log`, `results/2026-10-07_phase0_reference.json`.
  **Ueberholt 2026-10-12 (PR #61):** "Phase 0 komplett" und die Loeser-Decke
  `verified_numerical_reference` fuer `hex-mms-tri` gelten nicht mehr. Die Evidenz von 2026-10-07
  beruhte auf dem rekursiven CG-Residuum; mit dem wahren Residuum ist I1 rot und die Decke
  `comparison_only` (siehe Blockiert oben). Der Eintrag bleibt als Lineage stehen.
  **Nachtrag 2026-10-12 (E1012-07):** unter I1 V2 ist die Decke wieder
  `verified_numerical_reference` (Eintrag "Phase 0b I1 V2" oben); unter I1 V1 bleibt sie
  `comparison_only`.

- [x] **Reality Anchor Phase 0a: Referenzfall `hex-heat-tri` + ReferenceSpec v1 (#60, #33,
  W1-K08, 2026-10-07).** ReferenceSpec v1 aus Draft #33 per Merge-Commit uebernommen (Commits
  erhalten; dessen CI-Rot: fehlender SOURCES-Eintrag fuer `spec/REFERENCE_SPEC_v1.md` und Pins
  vor den letzten Haertungs-Commits -> append-only nachgezogen). Neuer Vertrag
  `spec/REFERENCE_CASE_HEX_HEAT_TRI_v1.md`, Modul `reference_case_hex_v3.py`: Lame-Moden (m, n,
  sin/cos) sind exakte diskrete Eigenvektoren des 7-Punkt-Hex-Laplace (Engine-Gewicht) auf dem
  Dirichlet-Dreieck; Studie N = 16..128 reproduziert die Zahlen aus #60 (lambda_h 51.9649 ...
  52.6273, Celik p 1.998 / 1.993), Waermeloesung Ordnung 2.00, GCI_fine 2.51e-4 >= exakter
  Fehler 2.01e-4, alle Kriterien C1-C9 gruen, abgeleitete Decke `verified_numerical_reference`
  nur fuer diesen Context of Use. Positivitaets-Gate auf dem assemblierten Operator (#60 Punkt
  1): Vorzeichenstruktur + `dt <= h^2/(4 kappa)`, gegen `hex_positivity_max_dt` geprueft.
  Diskriminierung: 1-%-Gewichtsfehler, Gewichtsfehler erster Ordnung, Neumann-artige Slots und
  ein zu kleiner GCI-Sicherheitsfaktor fallen durch; Mutationsprobe 34/34.
  _Evidenz:_ `results/2026-10-07_phase0-hex-heat-tri_local-gate.log`,
  `results/2026-10-07_hex-heat-tri_reference.json` (study_sha256 6a366cdb..., Quelle d23dec8).

- [x] **Issue #59: Folgepunkte aus PR #58 (2026-10-07).**
  _DCP-Bundle (P2):_ `finalize_bundle` fsynct den Inhalt jeder im Manifest gelisteten Datei
  (`fsync_file`, `rb+`-Handle, Windows-tauglich), dann die Shard-Verzeichnisse, dann Manifest
  und Marker. Vorher waren nur die Verzeichniseintraege dauerhaft. DCPs `FileSystemWriter`
  synct per Default (`sync_files=True`); das Bundle-Protokoll haengt davon nicht mehr ab.
  _MetricSpec v4 (P2):_ `metric_semantics_label` steht in `__all__`; `spec/METRIC_SPEC_v4.md`
  Abschnitt 0 listet die oeffentlichen Namen, ein Test liest genau diese Liste.
  _dt-Vergleich (P3):_ zwei dt-Schluessel einer Trajektorie sind dieselbe Schrittweite, wenn
  sie auf dasselbe endliche float32 runden oder relativ innerhalb `2**-20` liegen (vorher nur
  relativ: 0.0 vs 1e-46 nie erkannt). Testpraemisse des Property-Tests korrigiert (0.0/5e-324
  sind float32-Zwillinge; Pool nutzt jetzt `2**-149`).
  _Evidenz:_ rot gegen die alten Bytes failures=5, errors=2 -> 536/536 OK; Mutationsprobe
  27/27 (M23-M27 neu); DCP-Oracles lokal exit 0; `results/2026-10-07_issue59_local-gate.log`.
  Weiter offen aus #59: Reality Anchor (#33/#60), dt-Gruppierung als deklarierter
  Quantisierungsvertrag (`dt_round_ndigits=-1` wird akzeptiert), Positivitaets-Hinweis
  (`safety=0.85` ist nicht positivitaetserhaltend; dokumentiert, kein Default verspricht es).

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

- [ ] **Reality Anchor Phase 1 (#60): Owner-Freigabe der Vorregistrierung.**
  `spec/PREREGISTRATION_PHASE1_v1.md` ist ein DRAFT mit den Entscheidungen D1-D8 (Aufgaben-Konfig,
  Seeds, Stichprobengroessen, GINO-Raster/Budget, Hardware, BC-Achse nur periodisch in v1,
  optionale Abhaengigkeit `neuraloperator==2.0.0`, Lizenzanfrage GAOT/GINOT). Erst nach Freigabe:
  Datensaetze mit Manifest erzeugen, GINO-Raster auf Validierung, 5 Seeds, Testsets einmal,
  Kosten-Genauigkeits-Kurve der Baseline, Bericht mit IQM + Bootstrap-KI. Neumann-Achse braeuchte
  ein Neumann-Gitter im Loeser (v2).
- [ ] **ReferenceSpec v1 (#33), Review-Befund.** `ConvergenceStudy.observed_order` wird nur auf
  endlich/positiv geprueft, nicht gegen die Werte der Levels nachgerechnet; ein Aufrufer kann eine
  Ordnung behaupten, die die Daten nicht tragen. Vorschlag: bei konstantem r und >= 3 Levels
  Konsistenz gegen die Celik-Ordnung verlangen (Vertragsaenderung -> Owner-Entscheid).
  `reference_spec_hex_v3.py` hat 82 % Branch-Coverage (Gesamt-Gate 87 % gilt global).
- [ ] **dt-Gruppierung als deklarierter Quantisierungsvertrag (#59).** `dt_round_ndigits=-1` wird
  akzeptiert; Vertrag fuer zulaessige Werte und Semantik fehlt.

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
  im Branch-Schutz aendern. Nachtrag 2026-10-07: ein Session-Handoff vom 2026-10-05 (Drive)
  meldet "Required gate + tests (cpu)" als gesetzt; aus dem Repo/der Agent-Sitzung nicht
  pruefbar (keine Branch-Protection-API) -> Owner bestaetigt, dann diesen Punkt schliessen.
- [ ] **Mid-Epoch-Resume.** `dataloader_state_dict` wird gespeichert/geladen, der Runner nutzt es
  aber nicht (Resume nur an Epochengrenzen). Fuer lange Epochen auf Preemptible-Clustern relevant
  (Muster: `torchdata.StatefulDataLoader`).
- [ ] **DDP-Integration des Runners.** Nur `is_main_process`-Guard; kein `DistributedSampler.
  set_epoch`, kein Barrier nach dem Checkpoint-Write, Metriken nicht ueber Ranks reduziert.
- [x] ~~**CI-Reproduzierbarkeit:** `numpy` ist im Testjob ungepinnt.~~ Erledigt 2026-10-14
  (siehe oben): `numpy==2.5.3` im Testjob, wie in den Evidence-Lanes; inline wie `torch==2.6.0`,
  weil `requirements-dev.txt` laut Kopf nur Dev-Tools pinnt.

## Zurueckgestellt (bewusst, mit Begruendung)

- [~] **Typing-Modernisierung (PEP 585/604).** Ruff `UP`-Regeln finden ~210 Stellen
  (`Dict`->`dict`, `Optional[X]`->`X | None`) in den gehaerteten Modulen. Wertvoll, aber grosse
  Aenderung — erst nach gruenem `tests (cpu)`-Gate als separater, einzeln verifizierter PR
  (Reality-Anchor). `select` in `pyproject.toml` dann um `UP`,`I` erweitern.
- [~] **`ruff format` durchsetzen.** Alle 16 Module sind nicht ruff-format-konform; ein
  Format-Gate wuerde grossflaechig reformatieren. Erst nach Typing-PR sinnvoll.
