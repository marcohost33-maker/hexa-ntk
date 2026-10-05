# HexaNTK — Gehärtetes Expert Review & Lückenanalyse

**Datum:** 2026-04-06
**Scope:** Alle drei HexaNTK-Dokumente (HexaNTK_2, 3HexaNTK2_Gem, 3HexaNTK)
**Ziel:** Überprüfung, Lückenidentifikation, Härtung auf höchstes Qualitätsniveau

---

## 1. Gesamtbewertung

### 1.1 Stärken (bestätigt durch Web-Recherche)

Die drei Dokumente bilden zusammen eine solide Grundlage für eine hexagonale PDE-Operator-Learning-Pipeline. Die Kernarchitektur-Entscheidungen sind fachlich korrekt:

- **Trajektorienbasierter Dataset-Split** statt sample-basiertem `random_split` ist methodisch zwingend für zeitabhängige PDE-Daten. Blockbasiertes Splitting bei abhängigen Beobachtungen ist etablierte Best Practice (vgl. Roberts et al., Kapoor & Narayanan 2023 zu Reproduzierbarkeitskrise durch Leakage in 290+ ML-Papers).

- **Safe/Trusted-Trennung beim Checkpoint-Loading** entspricht exakt dem PyTorch-2.6-Standard: `weights_only=True` ist seit PyTorch 2.6 der Default. Optimizer-Zustände, die z.B. `numpy.core.multiarray.scalar` enthalten, erfordern entweder `safe_globals`-Whitelisting oder den expliziten `trusted`-Pfad mit `weights_only=False`.

- **Atomisches Checkpoint-Schreiben** via `tempfile` + `os.replace` ist der korrekte POSIX-Ansatz für Preemption-Sicherheit auf Cluster-Systemen.

- **GINO-Architektur mit SDF + shared geometry** passt zur GINO-Spezifikation (Li et al., NeurIPS 2023): GINO nutzt GNO für die Transformation von irregulären zu regulären Gittern und FNO im Latent Space. Die Einschränkung auf `shared_geometry=True` im Batch ist für hexagonale Gitter mit fester Topologie korrekt.

- **Hex-Laplacian** mit ∇²ₕu₀ = 2/(3h²) Σⱼ₌₁⁶(uⱼ − u₀) und CFL-Grenze Δt_max = h²/(3Dᵥ) stimmt mit der Literatur zu FV-Schemata auf hexagonalen Gittern überein (Lee & Tien 2014, Lee 2017).

### 1.2 Gesamtarchitektur-Qualität

| Komponente | Status | Qualitätsstufe |
|---|---|---|
| `dataset_split_hex_v3.py` V2 | Gehärtet | ★★★★☆ |
| `checkpoint_integration_hex_v3.py` | Funktional, Lücken | ★★★☆☆ |
| `resume_utils_hex_v3.py` | Gehärtet | ★★★★☆ |
| `operator_adapter_hex_v3.py` | Gut strukturiert | ★★★★☆ |
| `test_dataset_split_hex_v3.py` V2 | Gehärtet | ★★★★☆ |
| `test_resume_utils_hex_v3.py` | Umfassend | ★★★★☆ |
| Runner/Anomaly-Detection | Konzeptionell stark, Code-Lücken | ★★★☆☆ |

---

## 2. Kritische Lücken & Korrekturen

### 2.1 LÜCKE: Zwei konkurrierende checkpoint_integration Designs

**Problem:** Doc2 (Gemini) und Doc1 definieren beide eine `checkpoint_integration_hex_v3.py`, aber mit fundamental unterschiedlichen APIs:

| Aspekt | Doc1 (Claude-Version) | Doc2 (Gemini-Version) |
|---|---|---|
| Architektur | Klasse `CheckpointIntegrationManager` | Freie Funktionen |
| EMA-Support | ❌ Fehlt | ✅ `ema_model_state` Parameter |
| DDP/FSDP | ❌ Fehlt | ✅ Erwähnt, nicht implementiert |
| `FailureSignature` | ❌ Fehlt | ✅ Verwendet |
| Atomik-Implementierung | Via `AtomicCheckpointWriter` Klasse | Via `_save_atomic` Funktion |
| Best-Policy mit `min_delta` | ✅ Ja | ❌ Fehlt |
| `checkpoint_semantics` Feld | ✅ Ja | ❌ Fehlt |

**Korrektur:** Die beiden Designs müssen zu einem einzigen kanonischen Design zusammengeführt werden. Empfohlene Synthese:

```python
# Kanonisches Design: Klasse mit allen Features beider Versionen
class CheckpointIntegrationManager:
    # Von Doc1: Klassen-Architektur, BestCheckpointPolicy, min_delta, checkpoint_semantics
    # Von Doc2: EMA-Support, FailureSignature-Integration, is_main_process-Guard
    # Von beiden: Atomisches Schreiben (vereinheitlicht via AtomicCheckpointWriter)
    
    def save_best_model_checkpoint(
        self, *, model, metric_value, metric_name, epoch,
        global_step=None, extra=None,
        ema_model_state=None,          # NEU: aus Doc2
        is_main_process: bool = True,  # NEU: DDP-Guard
    ) -> Dict[str, Any]: ...
```

### 2.2 LÜCKE: `FailureSignature` wird verwendet aber nie definiert

**Problem:** Doc2 importiert und verwendet `FailureSignature` aus `checkpoint_integration_hex_v3`, aber weder Doc1 noch Doc3 definieren diese Klasse. Der Runner-Code (Doc2) wird crashen.

**Korrektur:**

```python
@dataclass
class FailureSignature:
    """Structured failure record for debugging and post-mortem analysis."""
    kind: str              # z.B. "NaN_or_Inf_Loss", "Loss_Spike", "CUDA_OOM"
    step: int
    locus: str             # z.B. "train_step_forward_or_loss"
    observable: str        # z.B. "Loss value: nan"
    replay_capsule: Optional[str] = None  # Pfad zur Replay-Datei
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            from datetime import datetime, timezone
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def register(self, output_dir: Path) -> str:
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"failure_{self.step}_{self.kind}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return str(path)
```

### 2.3 LÜCKE: `copy_dict` in test_resume_utils fehlplatziert

**Problem:** Die Funktion `copy_dict` wird in `test_resume_utils_hex_v3.py` (Doc3) am Ende der Datei definiert, aber schon vorher in `_save_resume_fixture` aufgerufen. Python führt Top-Level-Definitionen sequentiell aus, aber innerhalb von Klassen/Funktionen erst bei Aufruf — das funktioniert nur, weil `copy_dict` auf Modulebene ist und der Test-Runner alle Definitionen erst einliest. Trotzdem ist es schlechter Stil und in manchen Kontexten fehleranfällig.

**Korrektur:** `copy_dict` vor die Testklasse verschieben oder durch `dict(x) if x else None` inlinen.

### 2.4 LÜCKE: `build_resume_payload` wird erwähnt aber nie implementiert

**Problem:** Doc1 schlägt vor, `save_resume_checkpoint` in `build_resume_payload` + `save_resume_checkpoint` aufzuteilen. Doc3 implementiert die finale `resume_utils_hex_v3.py`, aber ohne diese Trennung. Die Checkpoint-Integration in Doc1 referenziert `build_resume_payload` direkt.

**Korrektur:** Entweder:
- (a) `build_resume_payload` als separate Funktion in `resume_utils_hex_v3.py` extrahieren, oder
- (b) die Checkpoint-Integration anpassen, den Payload intern zu bauen und `AtomicCheckpointWriter.save_torch_payload_atomic` direkt zu nutzen

Option (a) ist sauberer für die Architektur.

### 2.5 LÜCKE: Runner-Code (Doc2) hat potentiellen Bug im OOM-Handler

**Problem:** Im gehärteten Runner (Doc2) wird im `except`-Block:

```python
del pred
del loss
gc.collect()
torch.cuda.empty_cache()
```

Aber `pred` und `loss` könnten in bestimmten Code-Pfaden gar nicht definiert sein (z.B. wenn der Fehler VOR der Zuweisung von `pred` auftritt). `del` auf eine undefinierte Variable wirft `NameError`.

**Korrektur:**

```python
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        # Sichere Referenz-Zerstörung
        for var_name in ('pred', 'loss'):
            if var_name in locals():
                del locals()[var_name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ...
```

Besser noch: `pred = loss = None` am Anfang initialisieren (wie in Doc2 V2 schon angedeutet), dann `del` sicher.

### 2.6 LÜCKE: Scheduler-Restore-Reihenfolge inkonsistent

**Problem:** Doc3 dokumentiert die Reihenfolge Optimizer vor Scheduler laden; die Implementierung in `load_resume_checkpoint` setzt das um. Aber Doc1 und Doc2 beachten diese Reihenfolge nicht in ihren Checkpoint-Integration-Designs. *(Korrigiert 2026-10-05, E4: der Satz lautete vorher "Scheduler vor Optimizer laden" und widersprach der Korrektur unten und §8.3.)*

**Korrektur:** In der kanonischen `CheckpointIntegrationManager.load_resume` muss die Reihenfolge verbindlich sein:

```
1. model.load_state_dict(...)
2. optimizer.load_state_dict(...)
3. scheduler.load_state_dict(...)  # NACH optimizer (PyTorch-Konvention)
4. grad_scaler.load_state_dict(...)
5. RNG-States
```

Dies entspricht der Standard-PyTorch-Konvention für General Checkpoints.

---

## 3. Fehlende Komponenten (Gap Analysis)

### 3.1 Fehlend: `warmstart_utils_hex_v3.py`

Doc1 importiert `load_partial_checkpoint` aus `warmstart_utils_hex_v3`, aber diese Datei wird in keinem der drei Dokumente implementiert. Die Warmstart-Funktionalität in Doc2 und Doc3 ist jeweils inline in der checkpoint_integration oder als Standalone-Funktion implementiert.

**Empfohlene kanonische Implementierung:**

```python
# warmstart_utils_hex_v3.py
def load_partial_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strip_prefixes: Optional[list[str]] = None,
    prefix_map: Optional[dict[str, str]] = None,
    allow_dtype_mismatch: bool = True,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load only model weights (partial or full) from any checkpoint type.
    
    NIEMALS Optimizer-State laden.
    NIEMALS als Resume interpretieren.
    
    Unterstützt:
    - strip_prefixes: z.B. ["module."] für DDP-Checkpoints
    - prefix_map: z.B. {"encoder.": "backbone."} für Architektur-Migration
    - allow_dtype_mismatch: Cast bei float16/float32-Differenzen
    """
    # SoTA: weights_only=True als Default
    # Fallback via safe_globals für Legacy-Checkpoints
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as e:
        if "Weights only load failed" in str(e):
            import logging
            logging.warning("Legacy checkpoint: versuche safe_globals Whitelisting...")
            try:
                import numpy as np
                from torch.serialization import safe_globals
                with safe_globals([np.core.multiarray.scalar, np.dtype, 
                                   type(np.dtype(np.float64))]):
                    payload = torch.load(checkpoint_path, map_location=map_location, 
                                        weights_only=True)
            except (ImportError, Exception):
                raise e
        else:
            raise e
    
    # state_dict extrahieren
    if isinstance(payload, dict):
        state_dict = payload.get("model_state_dict", payload)
    else:
        raise TypeError("Checkpoint payload must be a dict")
    
    # Prefix-Transformation
    if strip_prefixes:
        for prefix in strip_prefixes:
            state_dict = {
                k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in state_dict.items()
            }
    
    if prefix_map:
        for old_prefix, new_prefix in prefix_map.items():
            state_dict = {
                new_prefix + k[len(old_prefix):] if k.startswith(old_prefix) else k: v
                for k, v in state_dict.items()
            }
    
    # Dtype-Mismatch-Handling
    if allow_dtype_mismatch:
        model_sd = model.state_dict()
        for k in state_dict:
            if k in model_sd and state_dict[k].dtype != model_sd[k].dtype:
                state_dict[k] = state_dict[k].to(model_sd[k].dtype)
    
    incompat = model.load_state_dict(state_dict, strict=strict)
    
    return {
        "checkpoint_path": str(checkpoint_path),
        "action": "warmstart",
        "checkpoint_semantics": "warmstart",
        "load_path_type": "safe",
        "strict_mode_used": strict,
        "missing_keys": list(incompat.missing_keys),
        "unexpected_keys": list(incompat.unexpected_keys),
        "warning": "Optimizer and scheduler states were INTENTIONALLY dropped for warmstart.",
    }
```

### 3.2 Fehlend: `test_checkpoint_integration_hex_v3.py`

Wird in Doc1 als nächster Schritt angekündigt, aber nie implementiert. Dies ist eine kritische Lücke.

**Mindest-Testfälle:**

```
- test_save_and_load_best_model_roundtrip
- test_best_model_min_delta_policy
- test_save_and_load_last_model
- test_save_and_load_resume_roundtrip
- test_warmstart_does_not_restore_optimizer
- test_atomic_write_survives_simulated_crash (tmpfile cleanup)
- test_checkpoint_semantics_field_present
- test_best_model_loads_with_weights_only_true
- test_ema_state_dict_roundtrip (wenn EMA implementiert)
```

### 3.3 Fehlend: `test_operator_adapter_hex_v3.py`

Wird am Ende von Doc1 angekündigt, aber nie implementiert.

**Mindest-Testfälle:**

```
- test_graph_operator_valid_call
- test_gino_valid_call
- test_missing_edge_index_raises_for_graph_operator
- test_missing_coords_raises_for_gino
- test_shared_geometry_false_raises_for_gino
- test_wrong_x_shape_raises
- test_output_shape_mismatch_raises
- test_unsupported_model_type_raises
- test_sdf_shape_validation
- test_boundary_mask_shape_validation
```

### 3.4 Fehlend: `benchmark_hex_v3.py` mit kanonischen Metriken

Wird in der P2-Roadmap erwähnt, aber nicht implementiert. Die kanonischen Metriken sollten sein:

```python
# Pflicht-Metriken (aus Review-Dokument):
nRMSE_t     # normalized RMSE als Funktion von t (Rollout-Metrik)
h1_rmse_mean  # H¹-Sobolev-RMSE (Gradient-Fehler)
h1_nrmse_mean # H¹-normalisierter RMSE
divergence_step  # Erster Zeitschritt mit nRMSE > threshold
```

---

## 4. Code-Qualitätsprobleme

### 4.1 Inkonsistente Import-Pfade

| Datei | Importiert | Definiert in |
|---|---|---|
| `checkpoint_integration_hex_v3.py` (Doc1) | `warmstart_utils_hex_v3` | ❌ Nirgends |
| `checkpoint_integration_hex_v3.py` (Doc1) | `resume_utils_hex_v3` | Doc3 |
| Runner (Doc2) | `FailureSignature` aus `checkpoint_integration_hex_v3` | ❌ Nirgends |
| `test_resume_utils_hex_v3.py` (Doc3) | `gino_hex_v3_v2.GINOHexOperatorV3V2` | ❌ Nicht in Docs |

**Risiko:** Die Tests sind nicht standalone ausführbar, da `GINOHexOperatorV3V2` nicht in den Dokumenten enthalten ist. Für isolierte Tests sollte ein Dummy-Modell verwendet werden.

### 4.2 Type-Annotation-Inkonsistenzen

Doc1 mischt `str | Path` (Python 3.10+) mit `Optional[...]` und `Dict[str, Any]`. Für Konsistenz sollte entweder durchgängig `from __future__ import annotations` (schon vorhanden) oder die explizite `Union`/`Optional` Form verwendet werden.

### 4.3 Fehlende `__all__` in mehreren Modulen

`resume_utils_hex_v3.py` (Doc3) hat `__all__`, aber `warmstart_utils_hex_v3` existiert nicht, und der Runner-Code hat kein `__all__`.

---

## 5. Numerische/Physikalische Korrektheit

### 5.1 Hex-Laplacian: BESTÄTIGT KORREKT

Der Standard-7-Punkt-FV-Stencil auf regulären Hexagonen:

∇²ₕu₀ = (2/(3h²)) Σⱼ₌₁⁶ (uⱼ − u₀)

ist korrekt dokumentiert. Die Literatur (Lee & Tien 2014, Lee 2017) bestätigt diesen Skalierungsfaktor für das Standard-Finite-Volumen-Schema auf regulären hexagonalen Gittern. Der Faktor 2/3 kommt aus der Geometrie: der Abstand zwischen Hexagon-Zentren ist h, aber die Voronoi-Fläche und die gemeinsamen Kantenlängen ergeben den Korrekturfaktor.

### 5.2 CFL-Grenze: BESTÄTIGT KORREKT

Für Forward-Euler mit dem obigen Stencil:

Δt_max ≤ h²/(3Dᵥ)

Der Skalierungsfaktor 4 (beim Standard-FD auf regulärem Gitter) wird durch den Hex-Stencil-Faktor 2/3 × 6 = 4 ersetzt, was zu 1/(2·2) = 1/4 führt — aber mit dem korrekten Hex-FV-Faktor ergibt sich 3 im Nenner. Die frühere, zu konservative Variante h²/(6Dᵥ) ist dokumentiert als falsch.

**Empfehlung:** Im Produktionscode einen Sicherheitsfaktor von 0.8–0.9 verwenden:

```python
dt_max = safety_factor * h**2 / (3 * D_v)  # safety_factor ∈ [0.8, 0.9]
```

### 5.2a Positivitäts-/DMP-Grenze (Nachtrag 2026-09-27, Vertrag zu PR #45)

L²-Stabilität (5.2) impliziert **keine** Positivität. Für Forward-Euler mit dem
obigen 7-Punkt-Stencil ist der Zentrums-Koeffizient 1 − 4Dᵥ·Δt/h², die sechs
Nachbar-Koeffizienten sind je (2/3)·Dᵥ·Δt/h² ≥ 0. Alle Koeffizienten sind genau
dann nicht-negativ, wenn

Δt ≤ h²/(4Dᵥ)

Dann ist der Schritt eine Konvexkombination: Positivität, diskretes
Maximumprinzip und Monotonie gelten — nur für dieses reguläre, lineare Schema mit
konstantem isotropem Dᵥ, nicht allgemein für variable/anisotrope Diffusion,
irreguläre Zellen oder andere Rand-/Operator-Diskretisierungen.

Implementierungs-Vertrag (`src/hexantk_v3.py`):

- `hex_cfl_max_dt(h, D_v, safety=0.85)` = safety · h²/(3Dᵥ) — nur L²-Stabilität;
  der historische Default 0.85 ist für nicht-negative Zustände **keine**
  Positivitätsgarantie (Zentrums-Koeffizient bei h = Dᵥ = 1: −2/15).
- `hex_positivity_max_dt(h, D_v, safety=1.0)` = safety · h²/(4Dᵥ) —
  Positivität/DMP; für Dichten, Konzentrationen, Wahrscheinlichkeiten verwenden.
- Beide verlangen endliche h, Dᵥ > 0 und 0 < safety ≤ 1, rechnen exakt und runden
  das Ergebnis **Richtung null** (der Rückgabewert liegt nie über der exakten
  Grenze); nicht als endliches positives binary64 darstellbare Ergebnisse
  (Überlauf, Unterlauf auf 0) werden mit `ValueError` abgewiesen.

### 5.3 FHN L²-Monotonie: KORREKT ALS FALSCH MARKIERT

Die Dokumente warnen richtigerweise, dass für das FitzHugh-Nagumo-System keine monotone L²-Energieabnahme behauptet werden darf. FHN hat aktivatorische Dynamik (Excitability), die Energie erzeugen kann. Stattdessen sollten Bounds-Guards und observable drift monitoring verwendet werden.

---

## 6. Architektur-Empfehlungen (SoTA-Abgleich)

### 6.1 GINO-Erweiterungen (aus aktueller Literatur)

Basierend auf aktueller GINO-Forschung (Stand 2025/2026):

1. **GINO Transformer (GINOT):** Liu et al. (Apr 2025) erweitern GINO mit Transformer-Attention. Euer Adapter-Code hat `gino_hex_v3_v2` als unterstützten Typ — GINOT-Erweiterung sollte als `gino_hex_v3_v3` oder `ginot_hex_v3` vorbereitet werden.

2. **Time-dependent extensions:** Aktuelle GINO-Literatur (Sarkar et al. Aug 2025, Ramezankhani et al. Jun 2025) betont, dass zeitabhängige Modelle auto-regressive oder causal-attention-Erweiterungen brauchen. Euer history-aware operator path (P3) ist genau das.

3. **Adaptive Sampling:** Nair et al. (2024), Liu et al. (Apr 2025) empfehlen adaptive Sampling-Strategien für scharfe Features. Für FHN-Spiralwellen relevant.

### 6.2 Checkpoint-Architektur (PyTorch 2.6+ SoTA)

Die aktuelle Best Practice laut PyTorch-Dokumentation und Community-Erfahrung:

1. **`torch.serialization.get_unsafe_globals_in_checkpoint(path)`** — Neue API (PyTorch 2.6+) zum statischen Prüfen, welche Globals in einem Checkpoint blockiert werden. Sollte in die Warmstart-Logik integriert werden.

2. **`torch.serialization.safe_globals` als Context Manager** — Bevorzugt gegenüber `add_safe_globals` (globaler State vs. scoped).

3. **`TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`** — Environment-Variable für globalen Safe-Load-Default. Nützlich für CI/Test-Environments.

### 6.3 Datenarchitektur (Best Practices)

1. **Grouped/Block-Split ist Pflicht** — Die Literatur bestätigt: bei abhängigen Beobachtungen (Trajektorien, Zeitreihen) muss auf Gruppenebene gesplittet werden. Euer `split_dataset_by_trajectory` ist der korrekte Ansatz.

2. **StatefulDataLoader** für Mid-Epoch-Resume — Korrekt als Anschlusspunkt identifiziert, noch nicht implementiert. Niedrige Priorität, aber Architektur sollte anschlussfähig bleiben.

3. **`worker_init_fn` + `generator`** für DataLoader-Reproduzierbarkeit — Im Resume-Kontext wichtig, aber in keinem der Dokumente implementiert.

---

## 7. Priorisierte Aufgabenliste

### P0 — Sofort (Blocker für Lauffähigkeit)

| # | Aufgabe | Grund |
|---|---|---|
| P0.1 | `FailureSignature` Dataclass definieren | Runner crasht ohne Definition |
| P0.2 | `warmstart_utils_hex_v3.py` implementieren | Import-Fehler in checkpoint_integration |
| P0.3 | checkpoint_integration Doc1/Doc2 mergen | Zwei inkonsistente Designs |
| P0.4 | `build_resume_payload` implementieren oder Referenzen entfernen | Import-Fehler |
| P0.5 | OOM-Handler Bug fixen (`del` auf undefinierte Variable) | Potentieller NameError |

### P1 — Direkt danach (Qualitätssicherung)

| # | Aufgabe | Grund |
|---|---|---|
| P1.1 | `test_checkpoint_integration_hex_v3.py` schreiben | Ungetestete Kernkomponente |
| P1.2 | `test_operator_adapter_hex_v3.py` schreiben | Ungetestete Validierungsschicht |
| P1.3 | Scheduler-Reihenfolge in allen Load-Pfaden vereinheitlichen | PyTorch-Warnung |
| P1.4 | `copy_dict` Position in Tests korrigieren | Code-Qualität |
| P1.5 | `gino_hex_v3_v2`-Import in Tests durch Dummy ersetzen | Standalone-Testbarkeit |

### P2 — Danach (SoTA-Härtung)

| # | Aufgabe | Grund |
|---|---|---|
| P2.1 | EMA-Support in kanonische checkpoint_integration | SoTA für PDE-Rollouts |
| P2.2 | `is_main_process`-Guard für DDP | Multi-GPU-Readiness |
| P2.3 | `benchmark_hex_v3.py` mit nRMSE(t), H1-Metriken | Evaluation-Qualität |
| P2.4 | `get_unsafe_globals_in_checkpoint` in Warmstart | Diagnostik |
| P2.5 | `worker_init_fn` + Generator für DataLoader-Resume | Reproduzierbarkeit |

### P3 — Später (Erweiterungen)

| # | Aufgabe | Grund |
|---|---|---|
| P3.1 | GINOT-Branch im OperatorAdapter | Architektur-Evolution |
| P3.2 | StatefulDataLoader-Integration | Mid-Epoch-Resume |
| P3.3 | Stratifizierter Gruppen-Split | Regime-Balance |
| P3.4 | EdgeCacheManager | Performance |
| P3.5 | Zero-shot Spatial Transfer | Generalisierung |
| P3.6 | Spectral Normalization als optionaler Schalter | Rollout-Stabilität |

---

## 8. Kanonische Checkpoint-Policy (gehärtet)

### 8.1 Vier Checkpoint-Typen

| Typ | `checkpoint_semantics` | `weights_only` | Inhalt |
|---|---|---|---|
| Best-Model | `"best_model"` | `True` (immer) | model_state_dict, metrics, optional EMA |
| Last-Model | `"last_model_only"` | `True` | model_state_dict, epoch |
| Resume | `"resume_full"` | `True` oder `False` (je nach RNG) | model, optimizer, scheduler, grad_scaler, RNG, epoch, global_step |
| Warmstart | `"warmstart"` | `True` (immer) | model_state_dict (partiell OK) |

### 8.2 Verbindliche Regeln

1. **last.pt ≠ resume_last.pt** — `last.pt` darf NIEMALS als Resume interpretiert werden.
2. **resume_last.pt ≠ best.pt** — Resume darf NIEMALS als reiner Best-Model-Checkpoint behandelt werden.
3. **Warmstart ≠ Resume** — Warmstart lädt NIEMALS Optimizer-State.
4. **Jeder Checkpoint trägt `checkpoint_type` und `checkpoint_semantics`** — Bei Load validieren.
5. **Atomisches Schreiben ist Pflicht** für alle Checkpoint-Typen.
6. **`requires_trusted_load`-Flag** im Save-Report setzen, wenn Python-/NumPy-RNG gespeichert wird.

### 8.3 Restore-Reihenfolge (verbindlich)

```
1. model.load_state_dict(...)
2. optimizer.load_state_dict(...)
3. scheduler.load_state_dict(...)  ← NACH Optimizer (PyTorch-Konvention)
4. grad_scaler.load_state_dict(...)
5. RNG-States (Python, NumPy, PyTorch, CUDA)
6. DataLoader-State (optional)
```

**Vorbedingung (präzisiert 2026-10-05, E4):** Optimizer UND Scheduler sind
konstruiert, BEVOR irgendein State geladen wird. Der Scheduler-Konstruktor ruft
einen ersten `step()` auf und schreibt dabei die Lernrate in die
Optimizer-Gruppen; wird der Scheduler erst nach `optimizer.load_state_dict`
konstruiert, überschreibt er die wiederhergestellte Lernrate. Das ist der
eigentliche Defekt, gegen den diese Regel schützt.

Für die PyTorch-Standard-Scheduler ist die Reihenfolge der beiden *Loads*
zustandsgleich: `LRScheduler.load_state_dict` aktualisiert nur das eigene
`__dict__` (torch 2.6.0, `lr_scheduler.py:166-173`), `Optimizer.load_state_dict`
übernimmt die gespeicherten Gruppenwerte ausser `params` (`optimizer.py:925-933`).
Die Reihenfolge Optimizer → Scheduler bleibt trotzdem verbindlich, damit eigene
Scheduler, deren `load_state_dict` Optimizer-Gruppen anfasst, das letzte Wort
haben.

Prüforakel: Verhalten, nicht Aufruf-Reihenfolge — nach Resume ist die Lernrate
gleich der im Checkpoint, und Lernraten-Verlauf und Gewichte sind bitgleich zu
einem ununterbrochenen Lauf (mehrere Scheduler-Typen); die Negativkontrolle
"Scheduler nach `optimizer.load_state_dict` konstruiert" muss dabei auffallen
(`TestRestoreOrderBehaviour` in `src/toolkit/test_resume_utils_hex_v3.py`).

---

## 9. Kanonische Augmentation-Priorisierung

Basierend auf Doc3 und aktueller Literatur:

```
Tier 1 (JETZT aktiv):
  ├── D₆-Symmetrien (hexagonale Rotationen/Spiegelungen)
  ├── Multi-Δt / Temporal Subsampling (k ∈ {1, 2, 4} mit CFL-Check)
  └── Parameter-Sampling

Tier 2 (NACH Warmstart, mit Gates):
  ├── Moderates physik-konsistentes Noise
  └── Inverse Evolution (NUR hinter warmstart_ok==True + Stabilitäts-Gates)

Tier 3 (SPÄTER):
  ├── Regime-aware Sampling (wenn Labels/Regime-Schätzer verfügbar)
  ├── Spectral Normalization
  └── History-aware Operator / erweiterte Markov-Ordnung
```

---

## 10. Zusammenfassung

### Was gut ist
- Trajektorienbasierter Split mit `dt_round_ndigits` und `val_group_keys_override`
- Safe/Trusted-Trennung beim Checkpoint-Loading
- Atomisches Schreiben mit POSIX `os.replace`
- OperatorAdapter mit explizitem Dispatch und `validate_context`
- Korrekte Hex-Numerik (Laplacian, CFL)
- Umfassende Test-Suite für Resume-Utils

### Was fehlt oder gebrochen ist
- Zwei inkonsistente checkpoint_integration-Designs (Doc1 vs Doc2)
- `FailureSignature` verwendet aber nie definiert
- `warmstart_utils_hex_v3.py` importiert aber nicht implementiert
- `build_resume_payload` referenziert aber nicht implementiert
- OOM-Handler Bug (del auf potentiell undefinierte Variable)
- Fehlende Tests für checkpoint_integration und operator_adapter
- EMA-Support nicht in der Klassen-Architektur
- DDP/FSDP-Guards fehlen
- Scheduler-Restore-Reihenfolge nicht überall konsistent

### Nächster konkreter Schritt
**P0.1–P0.5 abarbeiten**, damit der Code überhaupt lauffähig ist. Dann P1.1–P1.2 (Tests), dann P2 (SoTA-Härtung).
