# ADR-003 — Phase-1-Baseline: Raumverfeinerung (D-T1-2) und linearer Vergleichsmassstab (D-T1-1) (Entscheidung ausstehend)

**Status: PROPOSED / NOT APPROVED** · 2026-11-05 · betrifft `spec/PREREGISTRATION_PHASE1_v1.md`
Abschnitte 4-6 und 8 (DRAFT) sowie den Freigabekandidaten Draft-PR #91. Entscheider: Owner `[M]`.

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

## Was gebaut ist (Instrumente, keine Entscheidung)

| Baustein | Ort | Vertrag |
|---|---|---|
| `nested_node_index(coarse, fine)` | `src/toolkit/hex_fv_solver_hex_v3.py` | `spec/HEX_FV_SOLVER_v1.md` §2, Nesting: Knoten `(i, j)` von `N` ist `(k i, k j)` von `k N`, `k` Zweierpotenz, Koordinaten **bitgleich** (geprueft) |
| `baseline_cost_accuracy_curve(..., refinements, coefficients)` | `src/toolkit/phase1_protocol_hex_v3.py` | `spec/PHASE1_PROTOCOL_v1.md` §2a: Loesung auf `k N`, Injektion auf die Auswertungsknoten, Fehler auf `N`; neues Feld `CostAccuracyPoint.n_solve` |
| Bindung in der Entscheidung | `decide_preregistered` | §5a: nur `_REGISTERED_REFINEMENTS = (1,)`; ein Punkt mit `n_solve != n` wird abgelehnt, bis der Owner die Verfeinerung registriert |
| `exact_linear_map(config, lattice)` | `phase1_protocol_hex_v3.py` | §2b: die T1-Wahrheit als Faktoren `L (R u0)`; Orakel aus der Basis, kein Fit |
| `PODRidgeBaseline.factors()` | `src/toolkit/linear_baseline_hex_v3.py` | `G = L @ R` bitgleich `matrix()` |
| `linear_comparator_point(...)` | `phase1_protocol_hex_v3.py` | §2b: Latenz und Fehler einer linearen Abbildung mit demselben Timing-Protokoll; Leakage-Guard vor jeder Zeitmessung; von keiner Regel gelesen |
| Pilotmessung | `scripts/oracles/t1_baseline_refinement_pilot.py` | Pilot-IDs `900100..900149` ausserhalb aller registrierten Bereiche; kein Modell |

## Messung (Pilot, Linux-Container, Zeiten dieser Maschine, nicht D5)

Registrierte Aufgabe [D1], registriertes Timing (3 + 10), `steps` 1..64, `N = 32`, 50 Pilot-Samples,
davon 10 getimt. Evidenz: `results/2026-11-05_t1-baseline-refinement-pilot.json` und
`results/2026-11-05_d-t1-instruments_local-gate.log`. Die Fehler sind deterministisch; die Zeiten
sind Medianwerte dieses Containers (4 CPUs, Threads nicht gepinnt).

| Fall | registrierte Kurve (k = 1): Boden, Zeitbereich der Front | k = 1, 2, 4: Boden, Zeitbereich | Faktor | exakte lineare Abbildung: Fehler, Latenz |
|---|---|---|---|---|
| Dreieck `N = 32` | 3.47e-3 (steps 8), Front 0.68-3.46 ms | 2.15e-4 (`N_solve` 128, steps 32), Front bis 195 ms | 16.2 | 4.1e-16, 3.3 µs |
| Torus `N = 32` | 2.09e-3 (steps 8), Front 1.39-5.24 ms | 1.30e-4 (`N_solve` 128, steps 32), Front bis 468 ms | 16.1 | 6.3e-16, 9.5 µs |

Zweite Ordnung im Raum bestaetigt (steps 64, Fehler je Halbierung von `h`: Dreieck 5.39e-3 ->
1.32e-3 -> 3.04e-4, Torus 2.99e-3 -> 7.32e-4 -> 1.72e-4; Faktoren 4.1 / 4.3 bzw. 4.1 / 4.3).

### Was die Verfeinerung an der Entscheidung aendert (Korrektur der Begruendung)

Der Gesamtbericht begruendet D-T1-2 damit, ein Modell mit `1e-3` bei `>= 20 ms` koennte ohne
Verfeinerung "genauer bei gleicher Laufzeit" gewinnen. **Unter den registrierten Regeln stimmt
das nicht:** die registrierte Front endet mit ihrem Boden-Punkt (`steps = 8`) bei 3.46 ms (Dreieck)
bzw. 5.24 ms (Torus), und ausserhalb ihres Zeitbereichs gibt es keine Gleich-Laufzeit-Aussage (keine
Extrapolation, Abschnitt 6). Mit `k = 1` lautet das Urteil fuer `(20 ms, 1e-3)` deshalb **H0**, mit
der Begruendung "baseline floor: die gemessene Baseline erreicht 1e-3 nicht" — formal kein Claim,
aber eine Formulierung, die wie ein Genauigkeitsvorsprung von GINO gelesen wird. Mit `k = 1, 2, 4`
ist es ebenfalls H0, aber begruendet: die Baseline erreicht bei 20 ms `8.97e-4` (Torus `8.36e-4`).

Die Entscheidungskarte des Oracles (hypothetische Ausgaenge, kein Modell: 41 Latenzen von
0.1 ms bis 1 s x 46 obere IQM-Schranken von 1e-5 bis 10^-0.5, die Regeln aus `_rules`) zeigt den
eigentlichen Effekt:

| Fall | Zellen | gleiches Urteil | H0 -> advantage | advantage -> H0 | "baseline floor" wird ausgewertet | Gleich-Laufzeit wird auswertbar |
|---|---|---|---|---|---|---|
| Dreieck | 1886 | 1442 | 444 | **0** | 492 | 782 |
| Torus | 1886 | 1412 | 474 | **0** | 492 | 874 |

- Die registrierte Kurve ist **nicht GINO-beguenstigend, sondern strukturell H0-lastig**: jede
  GINO-Latenz oberhalb weniger Millisekunden ist bei gleicher Laufzeit nicht vergleichbar, jeder
  Fehler unterhalb des Bodens endet im "baseline floor".
- Die Verfeinerung nimmt auf dem Pilotgitter **keinen** `k = 1`-Claim weg (0 Zellen): alle
  verfeinerten Front-Punkte liegen zeitlich hinter der registrierten Front (der erste ist
  `N_solve = 64`, `steps = 8`). Der einzige verfeinerte Punkt im registrierten Zeitbereich
  (Dreieck, `N_solve = 64`, `steps = 1`, 3.37 ms, Fehler 0.37) ist dominiert und aendert die Front
  nicht. Das ist ein Befund dieser Maschine, kein Satz: auf einer anderen Maschine (D5) muss die
  Karte neu gerechnet werden.
- Sie macht den Vergleich dort **entscheidbar**, wo ein GINO auf der CPU realistisch liegt
  (Latenz Millisekunden bis Zehntelsekunden, Fehler `1e-4 .. 3e-3`), gegen eine 16-mal staerkere
  Baseline. Ein dort gewonnener Claim ist aussagekraeftiger als jeder unter `k = 1` moegliche.

Die exakte lineare Abbildung liegt mit `4.1e-16` / `6.3e-16` in 3.3 / 9.5 µs links unterhalb jeder
CN-Front: auf T1 dominiert sie jede Baseline und jedes Modell. Das ist D-T1-1 in Zahlen.

## Entscheidungsvorschlag

**E1 (D-T1-2), empfohlen: A.**
- **A** Registrieren: die Kurve laeuft fuer jedes Auswertungsgitter mit `N_solve = k N`,
  `k in (1, 2, 4)`; Fehler auf `N` nach Injektion (`spec/PHASE1_PROTOCOL_v1.md` §2a). Fuer
  `N_eval = 128` ist `k = 4` (`N_solve = 512`, rund 130 000 Knoten) auf der D5-Maschine teuer;
  Unterentscheid: dort `k in (1, 2)` und das im Bericht nennen, oder `timed_samples` klein halten.
  Code-Folge: `_REGISTERED_REFINEMENTS` (dann je Gitter) und der Text aendern sich zusammen,
  in einem PR mit Abweichungsvermerk.
- **B** Nicht registrieren (Stand heute). Dann muss der Bericht jedes "baseline floor" als
  "nicht entschieden, Baseline nicht verfeinert" ausweisen, nicht als Genauigkeitsvorsprung.

**E2 (D-T1-1), empfohlen: A.**
- **A** Der gelernte lineare Massstab (POD-Ridge, `LinearBaselineConfig()` Defaults, nur auf dem
  Train-Split von [D3] gefittet, nach Freigabe) wird als `LinearComparatorPoint` neben jeder
  Kurve berichtet, ebenso das Orakel `exact_linear_map`. Jede Aussage "GINO vs. CN" nennt im
  selben Satz GINOs Stand gegen diesen Punkt. Keine neue Entscheidungsregel: T1 bleibt die
  Kalibrierstudie (Owner 2026-10-13), H0/advantage bleibt gegen CN definiert.
- **B** Den linearen Massstab zur Entscheidungs-Baseline machen. Auf T1 waere H0 damit praktisch
  vorentschieden (die Abbildung ist exakt und schneller); sinnvoll erst auf T2 (`hex-darcy`,
  dort laesst ein linearer Ersatz 6.7-9.4 % Fehler, Kandidat vom 2026-10-28).

### Textbausteine fuer die Vorregistrierung (bei A/A, Abschnitt 4 und 5)

> **Cost-accuracy curve.** For each evaluation lattice `N_eval` in `16, 32, 64, 128`, the
> baseline solves on `N_solve = k N_eval` for `k` in `1, 2, 4` (`k` in `1, 2` for `N_eval = 128`),
> with `steps` in `1, 2, 4, 8, 16, 32, 64`, on the same test inputs; the solution is injected
> onto the evaluation nodes (`spec/PHASE1_PROTOCOL_v1.md` section 2a). Per point it records the
> mean `rel_l2` on `N_eval` and the median wall time per sample, including the injection.

> **Linear comparator (reported, not a decision baseline).** The POD-ridge map fitted on the
> training split only, and the exact linear map of T1, are timed with the same protocol and
> reported as points beside each curve (`spec/PHASE1_PROTOCOL_v1.md` section 2b). Every
> statement about GINO against the baseline names GINO's standing against these points.

## Konsequenzen

+ Die Phase-1-Aussage wird im realistischen GINO-Bereich entscheidbar, gegen eine starke Baseline
  (McGreivy & Hakim, Regel 2), und ein T1-Ergebnis wird gegen den linearen Massstab eingeordnet.
+ Die Instrumente sind fertig: der Owner-Entscheid ist eine Textaenderung plus eine Konstante.
- Die Baseline-Kurve wird teurer (Messung oben; `N_eval = 128` mit `k = 4` dominiert).
- PR #91 und dieses ADR beruehren beide die Vorregistrierung erst bei der Freigabe; dieser PR
  aendert die Vorregistrierung nicht, damit es keinen Konflikt mit #91 gibt.
- Format: `CostAccuracyPoint` und der Entscheidungs-Record haben ein neues Feld `n_solve`
  (Bruch, `spec/PHASE1_PROTOCOL_v1.md` §5a); es gibt noch keinen Phase-1-Record.

## Verifikation

- Tests, Mutationsprobe und Gate: `results/2026-11-05_d-t1-instruments_local-gate.log`.
- Pilotmessung (gebunden an den Commit, Exit 0): `results/2026-11-05_t1-baseline-refinement-pilot.json`.
