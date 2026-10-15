# ADR-002 — Optionales GINO-Backend im Required CI Gate (Entscheidung ausstehend)

**Status: PROPOSED / NOT APPROVED** · 2026-10-11 · betrifft Draft-Preregistration D7 und `spec/GINO_ADAPTER_PHASE1_v1.md`.

## Konflikt und Evidenz

Die vorhandenen Spezifikationen verlangen eine getrennte optionale `neuraloperator==2.0.0`-Abhaengigkeit, und die Preregistration sagt `CI stays torch-only`; der geschuetzte GitHub-Job `ci / gate` war aber zunaechst der einzige als required dokumentierte Check. Ein separater `gino-smoke.yml`-Workflow allein erzwingt damit keine Backend-Kompatibilitaet vor einem Merge (siehe GitHub Docs zu required checks und SKIP).

PR #63/#64 fuehrten deshalb in `ci / gate` eine optionale Abhaengigkeit und einen echten GINO-Forward-/Backward-/Optimizer-Smoketest hinzu. PR #64 CI bestaetigte Funktion, aber **der schriftliche Preregistrierungs-/GINO-Vertrag wurde NICHT autorisiert geaendert**. Dies ist ein offener Spezifikationskonflikt, nicht stillschweigende Compliance.

## Entscheidungsvorschlag (Review noetig)

1. Bevorzugt: `ci / gate` bleibt torch-only fuer Grundfunktionen; ein eigener **required aggregate**-Check erfordert ausdruecklich sowohl Core-Gate als auch separaten GINO-Job. Repository-Branch-Schutz muss den neuen Check am richtigen Commit verpflichtend setzen. Ein skipped Check allein gilt laut GitHub nicht als hinreichende Absicherung. GitHub Rulesets und Schutzkonfiguration vor Integration inspizieren.
2. Alternativ: Die Preregistration D7 und GINO-Adapter-Spec **versioniert und explizit** durch den Owner um eine CI-only Ausnahme ergaenzen, waehrend `requirements-dev.txt` weiterhin kein NeuralOperator enthaelt und normale `tests (cpu)` torch-only bleiben. Die Implementierung ist dann ein zulassiger Gate-Schritt, nicht eine Trainingserlaubnis.
3. Bis zu dieser Entscheidung ist die PR-Kette **nicht releasebereit**. Keine D1-D8-Entscheidung, keine Preregistration-Freigabe und keine Phase-1-Training- oder wissenschaftliche Claim-Ableitung.

## Verifikation

- Auf unveraendertem PR #64 HEAD: https://github.com/marcohost33-maker/hexa-ntk/actions/runs/38105548067 sowie https://github.com/marcohost33-maker/hexa-ntk/actions/runs/38105548091
- GitHub Docs: https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks

Dieses ADR ist eine **Entscheidungsvorlage**, kein Nachweis, dass der Vertrag bereits genehmigt wurde.
