#!/usr/bin/env python3
"""verify_sources.py — fail-closed Provenance-Gate fuer SOURCES.md.

Prueft, dass die in SOURCES.md fuer ``src/toolkit`` eingetragenen SHA-256-Praefixe
mit den AKTUELLEN Repo-Dateien uebereinstimmen UND dass JEDES Toolkit-Modul (ausser
``__init__.py``) genau einen Eintrag hat. Damit ist SOURCES.md kein statischer Text
mehr, sondern ein re-verifizierbarer Integritaets-Vertrag (AGENTS.md: "SOURCES.md ist
die Integritaets-Wahrheit"; SLSA-Praxis: Artefakt-Digest gegen Manifest pruefen).

SHA-Konvention (identisch zu SOURCES.md): ``sha256(<repo-Datei>)[:16].upper()``.

Deterministisch + offline. Exit-Codes:
  0  alle Toolkit-Module erfasst und SHA stimmt
  1  Drift (SHA mismatch) oder Eintrag ohne Datei
  2  Toolkit-Modul ohne SOURCES.md-Eintrag (Luecke)
  3  SOURCES.md fehlt / nicht parsebar

Lauf: python scripts/verify_sources.py   (aus dem Repo-Root)
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = ROOT / "SOURCES.md"
TOOLKIT = ROOT / "src" / "toolkit"

# Zeilen der Form: | <SHA16> | <mtime> | src/toolkit | ...\<datei>.py[ (...)] |
ROW = re.compile(
    r"^\|\s*([0-9A-Fa-f]{16})\s*\|[^|]*\|\s*src/toolkit\s*\|\s*(.+?)\s*\|\s*$"
)
# Dateiname am Ende des Quell-Pfads (vor optionalem " (in-repo ...)"-Suffix).
FILENAME = re.compile(r"([^\\/ ]+\.py)(?:\s*\(.*\))?\s*$")


def sha16(path: pathlib.Path) -> str:
    # LF-normalisiert: CRLF-Checkout (Windows) und LF-Checkout (CI/Linux) ergeben
    # denselben Digest -> kein Fail-closed-Fehlalarm auf sauberem Checkout (Codex-P1).
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16].upper()


def parse_toolkit_rows(text: str) -> dict[str, str]:
    """filename -> erwarteter SHA16 (nur src/toolkit-Zeilen)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        sha, src = m.group(1).upper(), m.group(2)
        fn = FILENAME.search(src)
        if not fn:
            print(f"[FAIL] SOURCES-Zeile ohne erkennbaren Dateinamen: {line.strip()}")
            sys.exit(3)
        out[fn.group(1)] = sha
    return out


def main() -> int:
    if not SOURCES.exists():
        print(f"[FAIL] SOURCES.md fehlt: {SOURCES}")
        return 3
    expected = parse_toolkit_rows(SOURCES.read_text(encoding="utf-8"))
    if not expected:
        print("[FAIL] keine src/toolkit-Eintraege in SOURCES.md gefunden")
        return 3

    actual_files = {
        p.name for p in TOOLKIT.glob("*.py") if p.name != "__init__.py"
    }

    ok = True
    # 1) Luecke: Modul ohne Eintrag.
    missing = sorted(actual_files - expected.keys())
    for fn in missing:
        print(f"[GAP ] {fn}: Toolkit-Modul ohne SOURCES.md-Eintrag")
        ok = False
    if missing:
        return 2

    # 2) Drift: Eintrag ohne Datei oder SHA mismatch.
    drift = False
    for fn, exp in sorted(expected.items()):
        path = TOOLKIT / fn
        if not path.exists():
            print(f"[FAIL] SOURCES-Eintrag {fn}: Datei fehlt im Repo")
            drift = True
            continue
        got = sha16(path)
        if got != exp:
            print(f"[FAIL] {fn}: SHA {got} != SOURCES {exp}")
            drift = True
        else:
            print(f"[ OK ] {fn}: {got}")
    if drift:
        ok = False

    print(
        f"\n[VERIFY-SOURCES] {len(expected)} Toolkit-Eintraege | "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
