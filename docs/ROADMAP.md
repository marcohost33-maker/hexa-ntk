# ROADMAP — HexaNTK

Ehrliche Liste der naechsten Schritte und offenen Schuld. Reihenfolge = Prioritaet.
Status-Konvention: `[x]` erledigt (mit Evidenz) · `[ ]` offen · `[~]` bewusst zurueckgestellt.

## Erledigt (verifiziert)

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

- [x] **Coverage-Ratchet 80–90% erreicht (Ist 90%, fail_under 79→87).** Die drei
  schwaechsten Module per dedizierter Tests geschlossen: `warmstart_utils_hex_v3.py`
  **43%→94%** (neu `test_warmstart_utils_hex_v3.py`, 27 Tests: safe-load-Pfad inkl.
  numpy-safe_globals-Fallback + fail-closed bei nicht-whitelistbaren Objekten,
  Prefix-Strip/-Map, dtype-Toleranz, diagnose_checkpoint_safety), `resume_utils_hex_v3.py`
  **76%→96%** (neu `test_resume_utils_hex_v3.py`, 21 Tests: Payload-Komposition,
  requires_trusted_load-Vertrag, RNG-Roundtrips, safe-vs-trusted-Load, Validierungspfade),
  `ema_hex_v3.py` **56%→100%** (test_ema-Erweiterung: PyTorch<2.0-Fallback mit
  EMA-Mathe-Check, Legacy-Load-Pfade). Suite 166→218 Tests. SOURCES.md append-only
  ergaenzt (2 neue Testfiles, SHA-First); test_ema-Zeile aktualisiert (Praezedenz PR #12).
  _Evidenz: 218/218 PASS, 90% Branch, `results/2026-08-08_coverage-ratchet-87_gate.log`._

- [x] **Runner-Coverage 79%->94% + `resume_from`-Footgun gefixt (fail_under 87->91).**
  Selbstkritik-Runde auf der 08-08-Arbeit: (a) **Bug gefixt** — `maybe_resume()` prueft
  Existenz von `config.resume_from`, lud aber stumm den manager-eigenen `resume_path`
  (fremder Pfad wurde ignoriert); `load_resume` hat jetzt einen expliziten
  `path`-Override, der Runner reicht `resume_from` durch (+2 Regressionstests).
  (b) Runner-Luecken ehrlich getestet: EMA-AttributeError-Fallback, AMP-Scaler-Pfad
  (via auf CPU disabled `GradScaler("cuda")` — funktional derselbe Codepfad),
  Adapter-Dispatch in train/validate (graph + gino + defensiver else-Zweig),
  OOM-Capsule-Dump-Fehlpfad, Warmup-`_base_lr`-Cache, Save-Kadenz/No-EMA.
  (c) Test-Praezisierung: breite `assertRaises(Exception)` -> konkrete Typen
  (empirisch verifiziert: `pickle.UnpicklingError`; korrupte Legacy-Streams koennen
  auch `IndexError` werfen -> beide Re-Raise-Pfade getestet), `assertLogs` statt
  Log-Rauschen, numpy-RNG (nicht python-RNG) als tatsaechlicher trusted-Ausloeser
  dokumentiert. (d) `exclude_also` (coverage.py-Doku "Excluding code") fuer auf
  CPU-CI prinzipiell unerreichbare Zweige (CUDA-only, `except ImportError` bei
  gepinntem torch 2.6) statt Schein-Tests. Suite 218->231 Tests.
  _Evidenz: 231/231 PASS, 94% Branch, `results/2026-08-08_selfreview-ratchet-91_gate.log`._

## Offen

- [ ] **Rest-Coverage: Reports-/Diagnose-Zweige in `operator_adapter` (89%),
  `checkpoint_integration` (89%), `dataset_split` (95%), `benchmark` (96%),
  `failure_signature` (93%), `warmstart` (96%), `resume` (98%), `runner` (94%).**
  Verbleibende Misses sind ueberwiegend defensive Fehlerzweige einzelner
  Validierungspfade; Nutzen gegen Aufwand abwaegen, `fail_under` (91) bei
  Schliessung weiter nachziehen.

## Zurueckgestellt (bewusst, mit Begruendung)

- [~] **Typing-Modernisierung (PEP 585/604).** Ruff `UP`-Regeln finden ~210 Stellen
  (`Dict`->`dict`, `Optional[X]`->`X | None`) in den gehaerteten Modulen. Wertvoll, aber grosse
  Aenderung — erst nach gruenem `tests (cpu)`-Gate als separater, einzeln verifizierter PR
  (Reality-Anchor). `select` in `pyproject.toml` dann um `UP`,`I` erweitern.
- [~] **`ruff format` durchsetzen.** Alle 16 Module sind nicht ruff-format-konform; ein
  Format-Gate wuerde grossflaechig reformatieren. Erst nach Typing-PR sinnvoll.
