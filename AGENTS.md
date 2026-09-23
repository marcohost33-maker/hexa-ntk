---
format: agents.md
version: "1.0"
project: HexaNTK
---

# AGENTS.md — HexaNTK

KI-Coding-Agent-Anweisungen. Reihenfolge: §1 Working agreements > §2 Conventions > §3 Don't > §4 When stuck.

## Project
- **Was:** Trainings-/Experiment-Infrastruktur fuer Neural-Operator-Laeufe auf hexagonalen Gittern: Checkpoint/EMA/DDP-Integration, Dataset-Split, Failure-Signatures, Benchmark. **Die Operatoren GINO/GINOT/GAOT sind NICHT implementiert** — nur Config-Vertrag + Shape-Validierung (src/hexantk_v3.py:717-720). Keine Paper-Ergebnisse nachgemessen.
- **Stack:** Python >= 3.12 (py3.10/3.11 gedroppt, Commit 7e06ad9; CI faehrt 3.12 mit torch==2.6.0 CPU-Wheel).
- **Provenance:** `SOURCES.md` ist der Provenance-/Integritaets-Index — Quelle + mtime + ein
  16-hex **SHA-256-Prefix** (64 bit, LF-normalisiert) der **aktuellen Repo-Bytes** jeder
  Datei unter `src/toolkit/` und `spec/`. `scripts/verify_sources.py` (CI-Job `gate`) prueft
  ihn fail-closed: Drift, Luecke oder verschachtelte Datei -> exit != 0. Append-only: eine
  In-Repo-Aenderung haengt eine neue Zeile an (massgeblich ist die letzte Zeile je
  Dateiname), Zeilen auf `main` werden nie geaendert und bleiben als Lineage stehen.
  Zeilen mit Ziel `src` (Engine) und `results` bleiben Import-Locatoren des Originals und
  werden nicht gate-geprueft. (Praezisiert 2026-09-23, PR #32: der Text beschrieb vorher
  einen reinen Original-Locator [u6 Option A / k-theory #13], das Gate (Skript seit
  PR #5, LF-Normalisierung PR #6, CI-blocking seit 2026-07-03 laut `docs/ROADMAP.md`) prueft
  die aktuellen Bytes — der Text folgt jetzt der gelebten Praxis.)
  Kanonische v3-Engines leben zusaetzlich in `coworker-dde/src` — Aenderungen dort und
  hier synchron halten (Byte-Mirror, per git-Blob verifiziert).

## Working agreements
1. **Reality-Anchor.** "Tests N/N PASS (verified <date>)" statt "sollte gehen". Gate-Logs in `results/` sind die Evidenz.
2. **Lineage ehrlich.** Vorgaenger nach `archive/`, RETRACTED-Material klar markieren, nie still ersetzen.
3. **SHA-First.** Neue Quelldateien nur mit SOURCES.md-Eintrag (Quelle + SHA-256 + mtime).

## Conventions
- Specs in `spec/` sind Vertrags-Quelle; Code folgt der Spec.
- Negativ-Results (NEG*/NR*) sind Buerger erster Klasse — nie loeschen.

## Branch & PR conventions (agents)
- **One agent = one branch prefix:** `claude/<task>` (Claude Code), `codex/<task>` (OpenAI Codex), `bot/<task>` (CI/automation). Human-led: `feat|fix|docs/<task>`.
- **Agent output opens as a Draft PR**; ready erst nach verifizierter Definition-of-Done.
- **Label agent PRs:** `agent:claude` / `agent:codex` / `agent:bot`.
- **Auto-merge over manual merge** (`gh pr merge --auto --squash`); zweiter paralleler PR rebased.
- **No concurrent agent pushes** auf dieses Repo — sequenziell ordnen.

## Don't
- Don't merge mit rotem `compile`-Gate.
- Don't commit or leak secrets, tokens, `.env` files or credentials — auch nicht in Logs, Commit-Messages oder Test-Fixtures.
- Don't claim physics results without a gate-log in `results/`.
- Don't delete or rewrite `SOURCES.md`-Eintraege (append-only Lineage).

## When stuck
- `README.md` (Struktur/Status) -> `SOURCES.md` (Herkunft) -> Engine-Header (Selftest-Anleitung).
- Eskalation nach 3 Fehlversuchen am selben Schritt: Draft-PR/Issue statt Loop.
