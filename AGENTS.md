---
format: agents.md
version: "1.0"
project: HexaNTK
---

# AGENTS.md — HexaNTK

KI-Coding-Agent-Anweisungen. Reihenfolge: §1 Working agreements > §2 Conventions > §3 Don't > §4 When stuck.

## Project
- **Was:** Neural-Operator-Toolkit (GINO/GAOT-Familie) fuer physics-informed ML auf hexagonalen Gittern: Checkpoint/EMA/DDP-Integration, Dataset-Split, Failure-Signatures, Benchmark.
- **Stack:** Python >= 3.10 (numpy/scipy/torch fuer volle Laeufe; CI = Syntax + Lint).
- **Provenance:** `SOURCES.md` ist der Provenance-/Lineage-Index — Quelle + mtime + ein
  16-hex **SHA-256-Prefix-Locator** (64 bit) des *importierten Originals*, **kein**
  Voll-Integritaets-Digest der aktuellen Repo-Bytes (Eintraege append-only, werden bei
  spaeteren In-Repo-Aenderungen nicht neu geschrieben; vgl. u6 Option A / k-theory #13).
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
