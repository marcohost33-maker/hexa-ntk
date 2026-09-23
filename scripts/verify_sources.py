#!/usr/bin/env python3
"""verify_sources.py — fail-closed Provenance-Gate fuer SOURCES.md.

Prueft, dass die in SOURCES.md fuer ``src/toolkit`` UND ``spec`` eingetragenen
SHA-256-Praefixe mit den AKTUELLEN Repo-Dateien uebereinstimmen UND dass JEDE
Datei dieser Ziele einen Eintrag hat:

* ``src/toolkit``: jedes ``*.py`` ausser ``__init__.py``;
* ``spec``: jedes ``*.md`` (Vertrags-Quelle laut AGENTS.md "Specs in spec/ sind
  Vertrags-Quelle; Code folgt der Spec" — ein ungepinnter Vertrag war vorher
  fuer dieses Gate unsichtbar, PR #32 F32-2).

Damit ist SOURCES.md kein statischer Text mehr, sondern ein re-verifizierbarer
Integritaets-Vertrag (AGENTS.md: "SOURCES.md ist die Integritaets-Wahrheit";
SLSA-Praxis: Artefakt-Digest gegen Manifest pruefen). Mehrere Zeilen fuer
denselben Dateinamen sind Lineage (append-only); massgeblich ist die LETZTE.

Nicht geprueft (bewusst): Ziele ``src`` (Engine) und ``results`` — deren Zeilen
sind Import-Locatoren des Originals, keine Pins der aktuellen Repo-Bytes.

SHA-Konvention (identisch zu SOURCES.md): ``sha256(<repo-Datei>)[:16].upper()``,
LF-normalisiert.

Deterministisch + offline. Exit-Codes:
  0  alle Dateien erfasst und SHA stimmt
  1  Drift (SHA mismatch) oder Eintrag ohne Datei
  2  Datei ohne SOURCES.md-Eintrag (Luecke)
  3  SOURCES.md fehlt / nicht parsebar / keine src/toolkit-Eintraege

Lauf: python scripts/verify_sources.py   (aus dem Repo-Root)
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Ziel-Spalte -> (Verzeichnis relativ zum Repo-Root, Dateiendung, ausgenommen)
TARGETS: dict[str, tuple[str, str, frozenset[str]]] = {
    "src/toolkit": ("src/toolkit", ".py", frozenset({"__init__.py"})),
    "spec": ("spec", ".md", frozenset()),
}

# Zeilen der Form: | <SHA16> | <mtime> | <ziel> | ...\<datei>[ (...)] |
ROW = re.compile(
    r"^\|\s*([0-9A-Fa-f]{16})\s*\|[^|]*\|\s*(src/toolkit|spec)\s*\|\s*(.+?)\s*\|\s*$"
)
# Dateiname am Ende des Quell-Pfads (vor optionalem " (in-repo ...)"-Suffix).
FILENAME = re.compile(r"([^\\/ ]+\.[A-Za-z0-9]+)(?:\s*\(.*\))?\s*$")


class SourcesParseError(Exception):
    """SOURCES.md row that names a target but no usable file name."""


def sha16(path: pathlib.Path) -> str:
    # LF-normalisiert: CRLF-Checkout (Windows) und LF-Checkout (CI/Linux) ergeben
    # denselben Digest -> kein Fail-closed-Fehlalarm auf sauberem Checkout (Codex-P1).
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16].upper()


def parse_rows(text: str) -> dict[str, dict[str, str]]:
    """target -> {filename -> erwarteter SHA16}; letzte Zeile je Datei gewinnt."""
    out: dict[str, dict[str, str]] = {target: {} for target in TARGETS}
    for line in text.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        sha, target, src = m.group(1).upper(), m.group(2), m.group(3)
        suffix = TARGETS[target][1]
        fn = FILENAME.search(src)
        if not fn or not fn.group(1).endswith(suffix):
            raise SourcesParseError(
                f"SOURCES-Zeile ({target}) ohne erkennbaren {suffix}-Dateinamen: "
                f"{line.strip()}"
            )
        out[target][fn.group(1)] = sha
    return out


def parse_toolkit_rows(text: str) -> dict[str, str]:
    """filename -> erwarteter SHA16 (nur src/toolkit-Zeilen; Rueckwaerts-API)."""
    return parse_rows(text)["src/toolkit"]


def main(root: pathlib.Path = ROOT) -> int:
    sources = root / "SOURCES.md"
    if not sources.exists():
        print(f"[FAIL] SOURCES.md fehlt: {sources}")
        return 3
    try:
        expected = parse_rows(sources.read_text(encoding="utf-8"))
    except SourcesParseError as exc:
        print(f"[FAIL] {exc}")
        return 3
    if not expected["src/toolkit"]:
        print("[FAIL] keine src/toolkit-Eintraege in SOURCES.md gefunden")
        return 3

    # 1) Luecke: Datei ohne Eintrag (alle Ziele pruefen, dann gesammelt melden).
    gap = False
    for target, (rel_dir, suffix, excluded) in TARGETS.items():
        directory = root / rel_dir
        actual = {
            p.name
            for p in directory.glob(f"*{suffix}")
            if p.is_file() and p.name not in excluded
        }
        for fn in sorted(actual - expected[target].keys()):
            print(f"[GAP ] {target}/{fn}: Datei ohne SOURCES.md-Eintrag")
            gap = True
        # SOURCES-Zeilen tragen nur den Dateinamen: eine Datei in einem
        # Unterordner waere fuer dieses Gate unsichtbar -> fail-closed melden
        # statt still ungeprueft lassen.
        for p in sorted(directory.rglob(f"*{suffix}")):
            if p.is_file() and p.parent != directory and "__pycache__" not in p.parts:
                rel = p.relative_to(root).as_posix()
                print(f"[GAP ] {rel}: verschachtelte Datei, vom Gate nicht abgedeckt")
                gap = True
    if gap:
        return 2

    # 2) Drift: Eintrag ohne Datei oder SHA mismatch.
    drift = False
    for target, (rel_dir, _suffix, _excluded) in TARGETS.items():
        for fn, exp in sorted(expected[target].items()):
            path = root / rel_dir / fn
            if not path.is_file():
                print(f"[FAIL] SOURCES-Eintrag {target}/{fn}: Datei fehlt im Repo")
                drift = True
                continue
            got = sha16(path)
            if got != exp:
                print(f"[FAIL] {target}/{fn}: SHA {got} != SOURCES {exp}")
                drift = True
            else:
                print(f"[ OK ] {target}/{fn}: {got}")

    print(
        f"\n[VERIFY-SOURCES] {len(expected['src/toolkit'])} Toolkit-Eintraege + "
        f"{len(expected['spec'])} Spec-Eintraege | "
        f"{'FAIL' if drift else 'PASS'}"
    )
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
