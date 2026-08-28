# Security Policy

## Melde-Kanal

Verwundbarkeiten bitte **NICHT als oeffentliches Issue** melden, sondern ueber
**Private Vulnerability Reporting** (Reiter "Security" -> "Report a vulnerability").
Falls der Kanal einmal nicht erreichbar ist: ein leeres Issue mit Label `security`
und der Bitte um einen privaten Kanal eroeffnen — **ohne** Details, Repro-Schritte
oder Payloads.

Integritaets-Probleme **ohne** Sicherheitsbezug (z.B. `SOURCES.md`-Abweichungen,
manipulierte oder nicht reproduzierbare Evidenz-Logs) duerfen weiterhin direkt als
GitHub-Issue mit Label `security` gemeldet werden.

Bitte angeben: betroffener Commit/Version, Repro-Schritte, Auswirkung, bekannter
Workaround. Keine echten Secrets und keine personenbezogenen Daten im Report.

## Geltungsbereich

Forschungs-Repo (Coworker Research / Coworkerz). Der Code ist **Trainings-/
Experiment-Infrastruktur**, kein betriebener Dienst: es gibt keinen Server, keine
Nutzerdaten und keine Netzwerk-Endpunkte. Relevante Klassen sind daher vor allem
Deserialisierungs-/Checkpoint-Pfade (`torch.load`), Pfad-Behandlung beim
Checkpoint-Schreiben und die Workflow-Supply-Chain.

## Supply-Chain-Haertung

Actions sind SHA-gepinnt, `zizmor` auditiert die Workflows, Dependabot laeuft mit
7-Tage-Cooldown. `scripts/verify_sources.py` ist ein fail-closed Provenance-Gate
gegen `SOURCES.md`.
