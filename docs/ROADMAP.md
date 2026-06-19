# ROADMAP — HexaNTK

Ehrliche Liste der naechsten Schritte und offenen Schuld. Reihenfolge = Prioritaet.
Status-Konvention: `[x]` erledigt (mit Evidenz) · `[ ]` offen · `[~]` bewusst zurueckgestellt.

## Erledigt (verifiziert)

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
- [ ] **SOURCES.md vervollstaendigen.** Nur Engine/Spec/Log sind erfasst; die 16
  `src/toolkit/`-Module fehlen in der Provenance-Tabelle (append-only ergaenzen, SHA-256 + mtime).
- [ ] **Coverage messen.** `pytest --cov` ueber Toolkit + Engine; Zielkorridor dokumentieren
  (Scientific-Python-Guide: hohe Abdeckung als Kernkriterium).

## Zurueckgestellt (bewusst, mit Begruendung)

- [~] **Typing-Modernisierung (PEP 585/604).** Ruff `UP`-Regeln finden ~210 Stellen
  (`Dict`->`dict`, `Optional[X]`->`X | None`) in den gehaerteten Modulen. Wertvoll, aber grosse
  Aenderung — erst nach gruenem `tests (cpu)`-Gate als separater, einzeln verifizierter PR
  (Reality-Anchor). `select` in `pyproject.toml` dann um `UP`,`I` erweitern.
- [~] **`ruff format` durchsetzen.** Alle 16 Module sind nicht ruff-format-konform; ein
  Format-Gate wuerde grossflaechig reformatieren. Erst nach Typing-PR sinnvoll.
