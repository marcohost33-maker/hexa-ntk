# ADR-003 — Phase-1-Baseline: Raumverfeinerung (D-T1-2) und linearer Vergleichsmassstab (D-T1-1) (Entscheidung ausstehend)

**Status: PROPOSED / NOT APPROVED** · 2026-11-05, zweite Fassung nach Selbstpruefung · betrifft
`spec/PREREGISTRATION_PHASE1_v1.md` Abschnitte 4-6 und 8 (DRAFT) sowie den Freigabekandidaten
Draft-PR #91. Entscheider: Owner `[M]`.

Dieses ADR ist eine **Entscheidungsvorlage**. Es aendert die Vorregistrierung nicht, und nichts
hier erlaubt einen Fit auf T1-Daten. Die Instrumente sind gebaut und getestet; ob und wie sie
in den registrierten Vertrag kommen, entscheidet der Owner.

## Kontext

Der Gesamtbericht vom 2026-11-05 (Drive, `HexaNTK_Gesamtbericht_2026-11-05.pdf`, Kapitel 9, 11,
18, 20) nennt als roten blinden Fleck: PR #91 friert die T1-Baseline-Kurve **ohne
Raumverfeinerung** (D-T1-2) und **ohne linearen Vergleichsmassstab** (D-T1-1) ein; beides gehoert
vor die Freigabe in die Abschnitte 4-7. Beide Punkte stammen aus der Gegenpruefung vom
2026-10-28 und waren weder auf `main` (cd5f6dc) noch in PR #91 (e21e596) umgesetzt.

- **D-T1-2.** Die registrierte Kurve variiert nur die Zeitschritte auf dem Auswertungsgitter.
  Ihr Fehler bleibt beim Raumfehler von `N` stehen.
- **D-T1-1.** T1 ist exakt linear (Rang 15 auf dem Dreieck, 40 auf dem Torus). Ein Modell, das
  eine lineare Abbildung dieses Rangs darstellen kann, lernt sie aus 15 bzw. 40 generischen
  Paaren. Ein Sieg gegen Crank-Nicolson sagt dann wenig ueber neuronale Operatoren.

## Selbstpruefung der ersten Fassung (2026-11-05): ein blinder Fleck

Die erste Fassung dieses ADR (Commit 520702d) startete den verfeinerten Loeser von der
**exakten** feinen Anfangsbedingung aus den Modenkoeffizienten, ungetimt, und nannte das
"Problemdaten". Das ist Information, die das Modell nie sieht: es bekommt `u0` auf `N`. Ein
Praktiker mit `u0` auf `N` muss auf `k N` **prolongieren**, und das kostet Zeit und Genauigkeit.
Der dort gemessene Faktor 16 ist deshalb eine **Orakel-Obergrenze**, nicht die faire Baseline.
Dieselbe Herkunft hat vermutlich der Befund vom 2026-10-28 (CN "mit k = 2/4 5.7e-4 / 1.2e-4"),
auf den sich der Gesamtbericht stuetzt: er entspricht der Orakel-Kurve (zweite Ordnung je
Stufe), nicht einer Kurve aus `u0` auf `N`.

## Was gebaut ist (Instrumente, keine Entscheidung)

| Baustein | Ort | Vertrag |
|---|---|---|
| `nested_node_index(coarse, fine)` | `src/toolkit/hex_fv_solver_hex_v3.py` | `spec/HEX_FV_SOLVER_v1.md` §2, Nesting: Knoten `(i, j)` von `N` ist `(k i, k j)` von `k N`, `k` Zweierpotenz, Koordinaten **bitgleich** (geprueft) |
| `prolongate_linear(u, coarse, fine)` | `hex_fv_solver_hex_v3.py` | §2, Prolongation: P1-Interpolant auf der groben Triangulation (Rand 0), bitgleich mit einem unabhaengigen Paritaets-Orakel, zweite Ordnung; kennt keine Basis der Aufgabe |
| `baseline_cost_accuracy_curve(..., refinements, prolongation, coefficients, richardson)` | `src/toolkit/phase1_protocol_hex_v3.py` | `spec/PHASE1_PROTOCOL_v1.md` §2a: Loesung auf `k N`; `prolongation="linear"` (fair, **im getimten Aufruf**) oder `"exact"` (Orakel, ungetimt, Identitaet geprueft); Injektion auf die Auswertungsknoten; `richardson=True` fuegt `(4 u_2k - u_k)/3` als `scheme="cn_richardson_h"` hinzu (beide Loesungen in einem getimten Aufruf); Felder `n_solve`, `prolongation`, `scheme` |
| Bindung in der Entscheidung | `decide_preregistered` | §5a: nur `_REGISTERED_REFINEMENTS = (1,)`, `scheme = "cn"`, Prolongation `none` fuer `k = 1`, registriert `"linear"` darueber; `"exact"` und Richardson werden abgelehnt |
| `exact_linear_map(config, lattice)` | `phase1_protocol_hex_v3.py` | §2b: die T1-Wahrheit als Faktoren `L (R u0)`; Orakel aus der Basis, kein Fit |
| `PODRidgeBaseline.factors()` | `src/toolkit/linear_baseline_hex_v3.py` | `G = L @ R` bitgleich `matrix()` |
| `linear_comparator_point(...)` | `phase1_protocol_hex_v3.py` | §2b: Latenz und Fehler einer linearen Abbildung mit demselben Timing-Protokoll; Leakage-Guard vor jeder Zeitmessung; von keiner Regel gelesen |
| Pilotmessung v2 | `scripts/oracles/t1_baseline_refinement_pilot.py` | Pilot-IDs `900100..900149`; fair und Orakel, mit Richardson; Fehlerbudget der Prolongation; Entscheidungskarten; `--threads` pinnt BLAS/torch; kein Modell |

## Messung r2 (Pilot, Linux-Container, 4 Threads gepinnt, Zeiten dieser Maschine, nicht D5)

Registrierte Aufgabe [D1], registriertes Timing (3 + 10), `steps` 1..64, `N = 32`, 50
Pilot-Samples, davon 10 getimt. Evidenz: `results/2026-11-05_t1-baseline-refinement-pilot_r2.json`
(gebunden an Commit {{CODE_SHA}}, Exit 0) und `results/2026-11-05_d-t1-fair-prolongation_local-gate.log`.
Die Fehler sind deterministisch; die Zeiten sind Medianwerte dieses Containers.

### Boden der Kosten-Genauigkeits-Kurve (bester Punkt der Pareto-Front, `mean rel_l2`)

| Fall | registriert (`k = 1`) | fair (`linear`, `k = 1, 2, 4`) | fair + Richardson | Orakel (`exact`) | Orakel + Richardson | exakte lineare Abbildung |
|---|---|---|---|---|---|---|
| Dreieck `N = 32` | {{TRI_K1}} | {{TRI_FAIR}} (x{{TRI_FAIR_G}}) | {{TRI_FAIR_R}} (x{{TRI_FAIR_R_G}}) | {{TRI_EX}} (x{{TRI_EX_G}}) | {{TRI_EX_R}} (x{{TRI_EX_R_G}}) | {{TRI_MAP}} in {{TRI_MAP_T}} |
| Torus `N = 32` | {{TOR_K1}} | {{TOR_FAIR}} (x{{TOR_FAIR_G}}) | {{TOR_FAIR_R}} (x{{TOR_FAIR_R_G}}) | {{TOR_EX}} (x{{TOR_EX_G}}) | {{TOR_EX_R}} (x{{TOR_EX_R_G}}) | {{TOR_MAP}} in {{TOR_MAP_T}} |

Fehlerbudget der linearen Prolongation allein (`u0` von `N` auf `2N` / `4N`, relativer
L2-Fehler gegen das exakte feine Feld): Dreieck {{TRI_BUD2}} / {{TRI_BUD4}}, Torus {{TOR_BUD2}} /
{{TOR_BUD4}}; Ordnung `N -> 2N` gegen `2N -> 4N`: Verhaeltnis {{TRI_ORD}} (Dreieck), {{TOR_ORD}}
(Torus). Zweite Ordnung des Orakels im Raum bei `steps = 64`: Faktoren {{TRI_O2}} (Dreieck),
{{TOR_O2}} (Torus).

### Befund (Falsifikation der ersten Fassung)

- **Fair** bringt die Verfeinerung {{FAIR_SUMMARY}}. Der Mechanismus: der Interpolationsfehler
  der Anfangsbedingung ist `O(h_N^2)`, unabhaengig vom Loesungsgitter, und ersetzt den
  Diskretisierungsfehler `O(h_N^2)`, statt ihn zu beseitigen. Richardson hebt den
  Diskretisierungsterm auf und laesst genau den Interpolationsfehler stehen.
- **Orakel** gewinnt ~4 je Stufe, mit Richardson weit mehr. Die Differenz zwischen beiden ist
  die Information in der Anfangsbedingung, die das Modell nicht hat.
- Die exakte lineare Abbildung dominiert mit ~`5e-16` in Mikrosekunden jede CN-Front: D-T1-1 in
  Zahlen.

### Was die faire Verfeinerung an der Entscheidung aendert

Entscheidungskarte des Oracles (hypothetische Ausgaenge, kein Modell: 41 Latenzen von 0.1 ms
bis 1 s x 46 obere IQM-Schranken von 1e-5 bis 10^-0.5, Regeln aus `_rules`), jeweils gegen die
registrierte Kurve:

| Fall | Kandidat | gleiches Urteil | undecidable/H0 -> advantage | advantage -> H0 | "baseline floor" wird ausgewertet | Gleich-Laufzeit wird auswertbar |
|---|---|---|---|---|---|---|
{{MAP_ROWS}}

- Die registrierte Kurve ist **nicht GINO-beguenstigend, sondern strukturell H0-lastig**: ihre
  Front endet nach wenigen Millisekunden, und ausserhalb gibt es keine Gleich-Laufzeit-Aussage
  (keine Extrapolation, Abschnitt 6). Das Beispiel des Gesamtberichts ("1e-3 bei 20 ms gewinnt")
  ergibt mit `k = 1` **H0** mit der Begruendung "baseline floor" — formal kein Claim, aber eine
  Formulierung, die wie ein Genauigkeitsvorsprung gelesen wird.
- Die faire Verfeinerung {{FAIR_MAP_SUMMARY}}.
- Befund dieser Maschine: auf D5 ist die Karte neu zu rechnen.

## Entscheidungsvorschlag

**E1 (D-T1-2), empfohlen: {{E1_REC}}.**
- **A** Registrieren mit der fairen Prolongation: die Kurve laeuft fuer jedes Auswertungsgitter
  mit `N_solve = k N`, `k in (1, 2)`, Anfangsbedingung per `prolongate_linear` im getimten
  Aufruf, Fehler auf `N` nach Injektion (`spec/PHASE1_PROTOCOL_v1.md` §2a). `k = 4` bringt fair
  nichts mehr (Messung) und wuerde fuer `N_eval = 128` (`N_solve = 512`) nur Kosten erzeugen.
  Code-Folge: `_REGISTERED_REFINEMENTS` und der Text aendern sich zusammen, in einem PR mit
  Abweichungsvermerk.
- **B** Nicht registrieren (Stand heute). Dann muss der Bericht jedes "baseline floor" als
  "nicht entschieden, Baseline nicht verfeinert" ausweisen, nicht als Genauigkeitsvorsprung, und
  die Orakel-Kurve als Kontext nennen.
- **C** (nicht empfohlen) Die Orakel-Anfangsbedingung registrieren: eine Baseline mit
  Information, die das Modell nicht hat, ist kein fairer Vergleich; die Entscheidung lehnt
  `"exact"` ab.

**E1b (Richardson, Orakel-Obergrenze), empfohlen: berichten, nicht registrieren.** Mit fairer
Prolongation ohne Nutzen; mit Orakel eine Obergrenze dessen, was ein Loeser mit vollstaendiger
Anfangsinformation erreicht — nuetzlich als Einordnung jedes T1-Ergebnisses.

**E1c (offenes Instrument, Owner-Frage):** eine generische Prolongation hoeherer Ordnung
(spektral: Zero-Padding in der Eigenbasis des Gitters, auf dem Torus Standard-FFT-Interpolation,
auf dem Dreieck eine Lame-Moden-Sinustransformation) wuerde den Abstand zum Orakel fuer
bandbegrenzte Eingaben schliessen. Ob ein solcher Loeser als "Praktiker-Methode" gilt, ist ein
Fairness-Entscheid; das Instrument ist nicht gebaut.

**E2 (D-T1-1), empfohlen: A.**
- **A** Der gelernte lineare Massstab (POD-Ridge, `LinearBaselineConfig()` Defaults, nur auf dem
  Train-Split von [D3] gefittet, nach Freigabe) wird als `LinearComparatorPoint` neben jeder
  Kurve berichtet, ebenso das Orakel `exact_linear_map`. Jede Aussage "GINO vs. CN" nennt im
  selben Satz GINOs Stand gegen diesen Punkt. Keine neue Entscheidungsregel: T1 bleibt die
  Kalibrierstudie (Owner 2026-10-13), H0/advantage bleibt gegen CN definiert.
- **B** Den linearen Massstab zur Entscheidungs-Baseline machen. Auf T1 waere H0 damit praktisch
  vorentschieden (die Abbildung ist exakt und schneller); sinnvoll erst auf T2 (`hex-darcy`,
  dort laesst ein linearer Ersatz 6.7-9.4 % Fehler, Kandidat vom 2026-10-28).

### Textbausteine fuer die Vorregistrierung (bei E1 A und E2 A, Abschnitt 4 und 5)

> **Cost-accuracy curve.** For each evaluation lattice `N_eval` in `16, 32, 64, 128`, the
> baseline solves on `N_solve = k N_eval` for `k` in `1, 2`, with `steps` in
> `1, 2, 4, 8, 16, 32, 64`, on the same test inputs; for `k = 2` the initial condition is the
> piecewise-linear prolongation of `u0` from `N_eval`, inside the timed call, and the solution is
> injected onto the evaluation nodes (`spec/PHASE1_PROTOCOL_v1.md` section 2a). Per point it
> records the mean `rel_l2` on `N_eval` and the median wall time per sample.

> **Linear comparator (reported, not a decision baseline).** The POD-ridge map fitted on the
> training split only, and the exact linear map of T1, are timed with the same protocol and
> reported as points beside each curve (`spec/PHASE1_PROTOCOL_v1.md` section 2b). Every
> statement about GINO against the baseline names GINO's standing against these points. The
> oracle curve (exact initial condition, Richardson) is reported as an upper bound of the solver.

## Konsequenzen

+ Die Phase-1-Aussage wird dort entscheidbar, wo ein CPU-GINO liegen wird, gegen eine faire,
  staerkere Baseline (McGreivy & Hakim, Regel 2), und ein T1-Ergebnis wird gegen den linearen
  Massstab und die Orakel-Obergrenze eingeordnet.
+ Die Instrumente sind fertig: der Owner-Entscheid ist eine Textaenderung plus zwei Konstanten.
- Der faire Gewinn der Verfeinerung ist klein; der grosse Gewinn der ersten Fassung war ein
  Orakel-Artefakt. Das ist ein Negativbefund erster Klasse, nicht ein Rueckschlag.
- Die Baseline-Kurve wird mit `k = 2` etwa 4-5x teurer je Punkt (Messung oben).
- PR #91 und dieses ADR beruehren die Vorregistrierung erst bei der Freigabe; dieser PR aendert
  sie nicht, damit es keinen Konflikt mit #91 gibt.
- Format: `CostAccuracyPoint` und der Entscheidungs-Record haben die neuen Felder `n_solve`,
  `prolongation`, `scheme` (Bruch, `spec/PHASE1_PROTOCOL_v1.md` §5a); es gibt noch keinen
  Phase-1-Record.

## Verifikation

- Tests, Mutationsprobe und Gate: `results/2026-11-05_d-t1-fair-prolongation_local-gate.log`
  (erste Runde: `results/2026-11-05_d-t1-instruments_local-gate.log`).
- Pilotmessung r2 (gebunden an den Commit, Exit 0):
  `results/2026-11-05_t1-baseline-refinement-pilot_r2.json`; erste Runde (Orakel allein):
  `results/2026-11-05_t1-baseline-refinement-pilot.json`.
