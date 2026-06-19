#!/usr/bin/env python3
"""engine_gate.py — exit-korrektes Gate fuer die kanonische Engine.

Hintergrund: `python src/hexantk_v3.py` exitet IMMER mit 0, weil der
__main__-Block den Rueckgabewert von run_all_tests() nicht auswertet. Fuer ein
echtes CI-Gate (Exit != 0 bei Fehlern) wird die Engine hier als Modul importiert
und Doctests + Selftests werden ausgewertet — ohne die kanonische Datei zu
editieren (AGENTS.md §Project: Engine wird ueber coworker-dde gepflegt).

Lauf: python scripts/engine_gate.py   (aus dem Repo-Root)
"""
from __future__ import annotations

import doctest
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hexantk_v3 as engine  # noqa: E402  (Pfad muss vor dem Import gesetzt sein)

doctest_result = doctest.testmod(engine, verbose=False)
selftests_ok = engine.run_all_tests()
doctests_ok = doctest_result.failed == 0
overall_ok = selftests_ok and doctests_ok

attempted = doctest_result.attempted
passed = attempted - doctest_result.failed
print(
    f"\n[ENGINE-GATE] doctests {passed}/{attempted} | "
    f"selftests_ok={selftests_ok} | overall={'PASS' if overall_ok else 'FAIL'}"
)
sys.exit(0 if overall_ok else 1)
