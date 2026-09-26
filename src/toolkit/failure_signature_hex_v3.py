"""
failure_signature_hex_v3.py
===========================
Structured failure tracking for HexaNTK training pipeline.

Used by:
- runner_hex_v3.py (anomaly detection, OOM rescue)
- checkpoint_integration_hex_v3.py (failure registration)

Design:
- Immutable dataclass for thread-safety
- JSON-serializable for post-mortem analysis
- Replay capsule reference for deterministic reproduction
"""
from __future__ import annotations

__all__ = [
    "FailureSignature",
]

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FailureSignature:
    """
    Structured failure record for debugging and post-mortem analysis.

    Attributes
    ----------
    kind :
        Failure category. Canonical values:
        - "NaN_or_Inf_Loss"
        - "NaN_or_Inf_Metric" (validation metric; best checkpoint skipped)
        - "Loss_Spike"
        - "CUDA_OOM"
        - "Runtime_Exception"
        - "Gradient_Explosion"
    step :
        Global training step at which the failure occurred.
    locus :
        Code location identifier, e.g. "train_step_forward_or_loss".
    observable :
        Human-readable description of the observed anomaly.
    replay_capsule :
        Optional path to a saved .pt file containing the batch tensors
        and model state_dict for deterministic reproduction.
    timestamp :
        ISO 8601 timestamp (UTC). Auto-generated if not provided.
    extra :
        Optional structured metadata (config hash, env fingerprint, etc.).
    """

    kind: str
    step: int
    locus: str
    observable: str
    replay_capsule: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for JSON serialization."""
        return asdict(self)

    def register(self, output_dir: str | Path) -> str:
        """
        Write this failure signature as a JSON file to output_dir.

        Returns
        -------
        str
            Path to the written JSON file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize kind for filename
        safe_kind = self.kind.replace("/", "_").replace(" ", "_")[:40]
        filename = f"failure_step{self.step}_{safe_kind}.json"
        path = output_dir / filename

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            logger.info("FailureSignature registered: %s", path)
        except OSError as exc:
            logger.error("Failed to write FailureSignature to %s: %s", path, exc)
            raise

        return str(path)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "FailureSignature":
        """Load a FailureSignature from a previously registered JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Remove unknown keys gracefully
        known_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
