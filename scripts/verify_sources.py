#!/usr/bin/env python3
"""verify_sources.py — fail-closed Provenance-Gate fuer SOURCES.md.

Vertrag (AGENTS.md "Provenance"): JEDE Datei unter ``src/toolkit/``, ``spec/``,
``scripts/`` und ``scripts/oracles/`` (seit 2026-10-05, E5) braucht in SOURCES.md
eine Zeile mit dem 16-hex SHA-256-Prefix ihrer aktuellen,
LF-normalisierten Bytes. Mehrere Zeilen fuer denselben Dateinamen sind Lineage
(append-only); massgeblich ist die LETZTE.

Welche Dateien zaehlen (PR #32, E7 — vorher Endungsfilter *.py / *.md, womit z.B.
eine ungepinnte ``spec/schema.json`` still PASS ergab):

* In einem Git-Arbeitsbaum: ``git ls-files --cached --others --exclude-standard``
  fuer beide Verzeichnisse, also alle getrackten Dateien JEDER Endung plus
  ungetrackte, nicht ignorierte Dateien (eine neue Datei faellt schon vor
  ``git add`` auf).
* Ohne Git (z.B. entpacktes Archiv): alle Dateien im Dateisystem, jede Endung.

Dokumentierte Ausnahmen (und NUR diese):
  - ``src/toolkit/__init__.py`` (leerer Paket-Marker, falls vorhanden);
  - Verzeichnisse ``__pycache__`` (in beiden Modi; auch absichtlich/getrackt
    aufgenommene Cache-Artefakte sind keine Provenance-Quellen).

Fail-closed statt still ungeprueft:
  - Datei in einem Unterordner  -> Luecke (SOURCES-Zeilen tragen nur den Dateinamen);
  - symbolischer Link (Git-Modus 120000 oder Dateisystem-Symlink) -> Luecke
    (der Pin wuerde fremde Bytes beglaubigen);
  - Dateiname ohne Endung oder mit Leerzeichen -> Luecke (in einer Zeile nicht
    ausdrueckbar);
  - Gross-/Kleinschreibung: Zeile und Datei muessen exakt uebereinstimmen, sonst
    Luecke (unabhaengig davon, ob das Dateisystem Gross/Klein unterscheidet).

Nicht geprueft (bewusst): Ziele ``src`` (Engine) und ``results`` — deren Zeilen
sind Import-Locatoren des Originals.

Deterministisch + offline. Exit-Codes:
  0  alle Dateien erfasst und SHA stimmt
  1  Drift (SHA mismatch) oder Eintrag ohne Datei
  2  Datei ohne gueltigen SOURCES.md-Eintrag (Luecke)
  3  SOURCES.md fehlt / nicht parsebar / keine src/toolkit-Eintraege / git-Fehler

Lauf: python scripts/verify_sources.py   (aus dem Repo-Root)
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Ziel-Spalte in SOURCES.md -> Verzeichnis relativ zum Repo-Root
TARGETS: dict[str, str] = {
    "src/toolkit": "src/toolkit",
    "spec": "spec",
    # E5 (2026-10-05): gate and evidence scripts are provenance sources too.
    # scripts/oracles is its own target, so a SOURCES row names the file
    # relative to exactly one directory (the longest matching target wins).
    "scripts": "scripts",
    "scripts/oracles": "scripts/oracles",
}
# Dokumentierte Ausnahmen (repo-relative POSIX-Pfade).
EXCLUDED: frozenset[str] = frozenset({"src/toolkit/__init__.py"})

# Zeilen der Form: | <SHA16> | <mtime> | <ziel> | ...\<datei>[ (...)] |
_TARGET_ALTERNATION = "|".join(
    re.escape(t) for t in sorted(TARGETS, key=len, reverse=True)
)
ROW = re.compile(
    r"^\|\s*([0-9A-Fa-f]{16})\s*\|[^|]*\|\s*("
    + _TARGET_ALTERNATION
    + r")\s*\|\s*(.+?)\s*\|\s*$"
)
# Dateiname am Ende des Quell-Pfads (vor optionalem " (in-repo ...)"-Suffix).
FILENAME = re.compile(r"([^\\/ ]+\.[A-Za-z0-9]+)(?:\s*\(.*\))?\s*$")


class SourcesParseError(Exception):
    """SOURCES.md row that names a target but no usable file name."""


class EnumerationError(Exception):
    """The set of files under a target could not be determined."""


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
        fn = FILENAME.search(src)
        if not fn:
            raise SourcesParseError(
                f"SOURCES-Zeile ({target}) ohne erkennbaren Dateinamen: {line.strip()}"
            )
        out[target][fn.group(1)] = sha
    return out


def parse_toolkit_rows(text: str) -> dict[str, str]:
    """filename -> erwarteter SHA16 (nur src/toolkit-Zeilen; Rueckwaerts-API)."""
    return parse_rows(text)["src/toolkit"]


def _git(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    # stdin=DEVNULL: an inherited, invalid stdin handle (e.g. under pytest on
    # Windows) otherwise makes Popen fail with OSError before git even runs.
    return subprocess.run(
        ["git", "-C", str(root), *args],
        stdin=subprocess.DEVNULL, capture_output=True, check=False,
    )


def _is_git_worktree(root: pathlib.Path) -> bool:
    """Detect Git mode without silently downgrading a broken checkout.

    A genuine archive has no .git marker and may use filesystem enumeration.
    If a marker is present, inability to query Git is an integrity failure:
    filesystem walking cannot account for indexed-but-missing paths.
    """
    git_marker = os.path.lexists(root / ".git")
    try:
        probe = _git(root, "rev-parse", "--show-toplevel")
    except FileNotFoundError as exc:
        if git_marker:
            raise EnumerationError(
                "git metadata present but git executable is unavailable"
            ) from exc
        return False
    if probe.returncode != 0:
        if git_marker:
            detail = probe.stderr.decode("utf-8", "replace").strip()
            raise EnumerationError(
                "git metadata present but rev-parse failed"
                + (f": {detail}" if detail else "")
            )
        return False
    top_text = probe.stdout.decode("utf-8", "replace").strip()
    if not top_text:
        if git_marker:
            raise EnumerationError("git rev-parse returned an empty toplevel")
        return False
    top = pathlib.Path(top_text)
    if top.resolve() != root.resolve():
        if git_marker:
            raise EnumerationError(
                f"git toplevel {top} does not match repository root {root}"
            )
        return False
    return True


def enumerate_files(root: pathlib.Path) -> tuple[str, dict[str, bool]]:
    """Return (mode, {repo-relative POSIX path -> is_symlink}) for all targets."""
    dirs = list(TARGETS.values())
    if _is_git_worktree(root):
        files: dict[str, bool] = {}
        staged = _git(root, "ls-files", "-s", "-z", "--", *dirs)
        if staged.returncode != 0:
            raise EnumerationError(staged.stderr.decode("utf-8", "replace").strip())
        for entry in staged.stdout.split(b"\0"):
            if not entry:
                continue
            meta, _, path = entry.partition(b"\t")
            files[path.decode("utf-8")] = meta.split(b" ")[0] == b"120000"
        others = _git(
            root, "ls-files", "--others", "--exclude-standard", "-z", "--", *dirs
        )
        if others.returncode != 0:
            raise EnumerationError(others.stderr.decode("utf-8", "replace").strip())
        for path in others.stdout.split(b"\0"):
            if path:
                rel = path.decode("utf-8")
                files.setdefault(rel, os.path.islink(root / rel))
        return "git", files

    files = {}

    def _walk_error(exc: OSError) -> None:
        raise EnumerationError(f"filesystem traversal failed: {exc}") from exc

    for rel_dir in dirs:
        base = root / rel_dir
        if base.is_symlink():
            # A target-root symlink would otherwise be followed by is_dir()/walk
            # and could authenticate mutable bytes outside the archive root.
            files[rel_dir] = True
            continue
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(
            base, followlinks=False, onerror=_walk_error
        ):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            here = pathlib.Path(dirpath)
            for d in list(dirnames):
                if (here / d).is_symlink():  # directory symlink: report, do not follow
                    files[(here / d).relative_to(root).as_posix()] = True
                    dirnames.remove(d)
            for name in filenames:
                p = here / name
                files[p.relative_to(root).as_posix()] = p.is_symlink()
    return "filesystem", files


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
    try:
        mode, files = enumerate_files(root)
    except EnumerationError as exc:
        print(f"[FAIL] Dateiliste nicht bestimmbar: {exc}")
        return 3

    # 1) Luecke: jede Datei der Ziele braucht einen gueltigen, exakt passenden Eintrag.
    gap = False
    for rel, is_link in sorted(files.items()):
        # Die dokumentierte Verzeichnis-Ausnahme gilt in beiden Modi. Ein
        # force-getracktes __pycache__ darf nicht zur Provenance-Quelle werden.
        if rel in EXCLUDED or "__pycache__" in pathlib.PurePosixPath(rel).parts:
            continue
        if rel in TARGETS.values() and is_link:
            print(f"[GAP ] {rel}: Zielverzeichnis ist symbolischer Link")
            gap = True
            continue
        target = max(
            (t for t, d in TARGETS.items() if rel.startswith(d + "/")),
            key=lambda t: len(TARGETS[t]),
            default=None,
        )
        if target is None:  # defensive: git returned a path outside the targets
            print(f"[GAP ] {rel}: ausserhalb der Ziele gelistet")
            gap = True
            continue
        name = rel[len(TARGETS[target]) + 1:]
        if "/" in name:
            print(f"[GAP ] {rel}: verschachtelte Datei, vom Gate nicht abgedeckt")
            gap = True
        elif is_link:
            print(f"[GAP ] {rel}: symbolischer Link, nicht pinnbar")
            gap = True
        elif not FILENAME.fullmatch(name):
            print(f"[GAP ] {rel}: Dateiname in SOURCES.md nicht ausdrueckbar")
            gap = True
        elif name not in expected[target]:
            print(f"[GAP ] {rel}: Datei ohne SOURCES.md-Eintrag")
            gap = True
    if gap:
        return 2

    # 2) Drift: Eintrag ohne (exakt benannte) Datei oder SHA mismatch.
    listed = {rel for rel in files}
    drift = False
    for target, rel_dir in TARGETS.items():
        for fn, exp in sorted(expected[target].items()):
            rel = f"{rel_dir}/{fn}"
            path = root / rel
            if rel not in listed or not path.is_file():
                print(f"[FAIL] SOURCES-Eintrag {rel}: Datei fehlt im Repo")
                drift = True
                continue
            got = sha16(path)
            if got != exp:
                print(f"[FAIL] {rel}: SHA {got} != SOURCES {exp}")
                drift = True
            else:
                print(f"[ OK ] {rel}: {got}")

    print(
        f"\n[VERIFY-SOURCES] {len(expected['src/toolkit'])} Toolkit-Eintraege + "
        f"{len(expected['spec'])} Spec-Eintraege + "
        f"{len(expected['scripts']) + len(expected['scripts/oracles'])} Script-Eintraege "
        f"| Dateiliste: {mode} "
        f"({len(files)} Dateien) | {'FAIL' if drift else 'PASS'}"
    )
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
