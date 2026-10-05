# MetricSpec v4 — Vertrag (verbindlich)

Stand 2026-10-05. Vertrags-Quelle fuer `MetricSpecV4`, `evaluate_rollout_v4`,
`metric_semantics_label` und `preserve_legacy_report_without_relabeling` in
`src/toolkit/benchmark_hex_v3.py` (eingefuehrt mit PR #52, Codex-Befund #52 T0:
die Semantik stand vorher nur im Code). Ergaenzt §3.4 von
`HexaNTK_Review_Gehaertet.md`, ersetzt ihn nicht: die Legacy-Metriken
(`nrmse_mean`, `divergence_step`, H1) bleiben unveraendert gueltig.

## 1. Identitaet

| Feld | Wert | Regel |
|---|---|---|
| `name` | `"mean_nrmse"` | fest; jeder andere Wert -> `ValueError` |
| `exclude_initial_state` | `bool` | nur echtes `bool`; `"false"`, `1` u.ae. -> `TypeError` |
| `version` | `"4.0"` | fest; jeder andere Wert -> `ValueError` |
| `metric_spec_digest` | SHA-256 ueber `to_dict()` (JSON, sortierte Schluessel, kompakt) | identifiziert die Konfiguration |

Ein Bericht darf keine andere Metrik oder Version deklarieren als die, die er
berechnet.

## 2. Stichproben-Reduktion (`sample_reduction`)

Nur `"mean_of_ratios"`: je Zeitschritt `t` und Stichprobe `b`. Eingabeformen
(praezisiert 2026-10-05, Codex PR #58): `[T, DOF]` = eine Stichprobe (identisch
zu `[T, 1, DOF]`); `[T, N, C]` = eine Stichprobe mit `N*C` Freiheitsgraden;
`[T, B, N, C]` = `B` Stichproben.

    nRMSE[t, b] = ||pred[t, b] - target[t, b]||_2 / ||target[t, b]||_2

dann Mittel ueber `b`. Das ist der Mittelwert der Stichproben-Verhaeltnisse
(PR #52 bezeichnet ihn als APEBench Eq. 31), NICHT das Verhaeltnis der ueber den
Batch aggregierten RMS-Werte (Legacy).

## 3. Null-Referenz (`zero_reference_policy`)

- `"error"` (Default): `||target[t, b]|| == 0` -> `ZeroDivisionError`.
- `"zero_if_both_zero"`: Verhaeltnis 0, wenn auch der Fehler 0 ist; sonst
  `ZeroDivisionError`.

## 4. Zeit-Aggregation

`time_aggregation` in `{"geometric_mean", "arithmetic_mean", "max"}`.
Das geometrische Mittel ist 0, sobald ein Schritt 0 ist.

`exclude_initial_state=True` (Default) verwirft Index 0. Zulaessig NUR, wenn
Index 0 der gemeinsame Anfangszustand ist: `evaluate_rollout_v4` verlangt
`pred[0] == target[0]` exakt und wirft sonst `ValueError` (fail-closed,
Codex #52 T1). Reine Vorhersage-Rollouts setzen `exclude_initial_state=False`.

## 5. Semantik-Label

`metric_semantics` wird aus der Spec abgeleitet, nie hart kodiert:

    APEBench_Eq31_mean_of_sample_ratios; <Eq32_|not_Eq32_><aggregation>_<forecast_only|all_steps>_time_aggregation

Das Praefix `Eq32_` steht nur bei `geometric_mean` UND `exclude_initial_state=True`
(geometrisches Mittel ueber die Vorhersageschritte); alle anderen Kombinationen
tragen `not_Eq32_` (Codex #52 T5).

## 6. Schwellen-Ueberschreitung

`threshold_crossing_step` = erster Index mit `nRMSE > threshold` (strikt),
`None` wenn keiner. Kein Sentinel-Wert (Legacy `divergence_step` lieferte `T`).

## 7. Legacy-Erhalt ohne Umetikettierung

Legacy-Berichte sind ohne Rohdaten je Stichprobe nicht nach v4 umrechenbar.
`preserve_legacy_report_without_relabeling` uebernimmt die Legacy-Werte mit
`legacy_`-Praefix, setzt die v4-Felder auf `None` und traegt die zur Deutung
noetigen Parameter mit (Codex #52 T3): `legacy_num_steps`,
`legacy_aggregation_method`, `legacy_divergence_threshold`,
`legacy_divergence_step_is_no_crossing_sentinel` (`divergence_step == num_steps`).

## 8. Anzeige

`BenchmarkReport.summary()` erkennt v4-Ergebnisse an `aggregated_mean_nrmse` und
zeigt `mean_nRMSE_v<version>` und `crossing_step` (Codex #52 T4).

## Claim-Grenze

Die Zuordnung zu APEBench Eq. 31/32 stammt aus PR #52 und dem Codex-Review;
die Gleichungsnummern sind in diesem Repo nicht gegen das Paper nachgeprueft.
