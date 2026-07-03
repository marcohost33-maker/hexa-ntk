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

## Offen

- [ ] **Engine-Lint-Schuld (coworker-dde-Sync).** `src/hexantk_v3.py` ist im hiesigen Lint-Gate
  via `extend-exclude` ausgenommen (kanonisch, Sync-Disziplin). Ruff (E/F/W) findet dort 6
  echte, triviale Defekte, die in der kanonischen Quelle `coworker-dde/src` gefixt und dann
  hierher gesynct werden sollten — **nicht** hier direkt editieren (sonst Divergenz):
  - `F401` `dataclasses.field` unused — Z. 76
  - `F541` f-string ohne Platzhalter — Z. 1037, 1427, 1480, 1521, 1570
  Nach dem Sync kann `src/hexantk_v3.py` aus `extend-exclude` entfernt und voll lint-enforced
  werden.
- [ ] **Engine `__main__` exit-Code.** `python src/hexantk_v3.py` exitet immer 0 (Rueckgabewert
  von `run_all_tests()` ungenutzt). In coworker-dde fixen: `sys.exit(0 if run_all_tests() and
  doctests_ok else 1)`. Bis dahin deckt `scripts/engine_gate.py` das Gate ab.
- [ ] **Coverage-Ratchet Richtung 80–90%.** Ist 68% (Branch, 2026-07-03). Schwaechste Module,
  in Prioritaetsreihenfolge des Nutzens: `runner_hex_v3.py` **25%** (191 Stmts offen — groesster
  Hebel), `warmstart_utils_hex_v3.py` **43%**, `ema_hex_v3.py` **56%**, `resume_utils_hex_v3.py`
  **73%**. Pro geschlossener Luecke `fail_under` in `pyproject.toml` nachziehen.

## Zurueckgestellt (bewusst, mit Begruendung)

- [~] **Typing-Modernisierung (PEP 585/604).** Ruff `UP`-Regeln finden ~210 Stellen
  (`Dict`->`dict`, `Optional[X]`->`X | None`) in den gehaerteten Modulen. Wertvoll, aber grosse
  Aenderung — erst nach gruenem `tests (cpu)`-Gate als separater, einzeln verifizierter PR
  (Reality-Anchor). `select` in `pyproject.toml` dann um `UP`,`I` erweitern.
- [~] **`ruff format` durchsetzen.** Alle 16 Module sind nicht ruff-format-konform; ein
  Format-Gate wuerde grossflaechig reformatieren. Erst nach Typing-PR sinnvoll.
