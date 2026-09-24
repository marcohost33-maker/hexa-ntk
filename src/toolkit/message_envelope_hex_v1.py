"""
message_envelope_hex_v1.py
==========================

Immutable message metadata for HexaNTK cluster information flow.

The design borrows interoperability principles from CloudEvents and distributed trace
context (stable event identity, explicit type/schema, trace correlation) without claiming
wire-level conformance to either standard. Payload bytes remain outside this object; only
their digest and size are carried here.

This module validates transport/orchestration metadata only. It does not validate the
scientific meaning or truth of a payload.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re

from hex_topology_v1 import FlowKind, FlowPlan


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_payload(payload: bytes) -> str:
    """Return lowercase SHA-256 for a payload."""
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MessageEnvelope:
    message_id: str
    task_id: str
    trace_id: str
    source: str
    target: str
    kind: FlowKind
    schema_version: str
    sequence_no: int
    ttl_hops: int
    payload_digest: str
    payload_bytes: int
    causation_source: str | None = None
    causation_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "message_id",
            "task_id",
            "trace_id",
            "source",
            "target",
            "schema_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if isinstance(self.sequence_no, bool) or not isinstance(self.sequence_no, int):
            raise ValueError("sequence_no must be an integer")
        if self.sequence_no < 0:
            raise ValueError("sequence_no must be >= 0")
        if isinstance(self.ttl_hops, bool) or not isinstance(self.ttl_hops, int):
            raise ValueError("ttl_hops must be an integer")
        if self.ttl_hops < 0:
            raise ValueError("ttl_hops must be >= 0")
        if isinstance(self.payload_bytes, bool) or not isinstance(self.payload_bytes, int):
            raise ValueError("payload_bytes must be an integer")
        if self.payload_bytes < 0:
            raise ValueError("payload_bytes must be >= 0")
        if not isinstance(self.payload_digest, str) or not _SHA256_RE.fullmatch(
            self.payload_digest
        ):
            raise ValueError("payload_digest must be a lowercase 64-hex SHA-256")
        if (self.causation_source is None) != (self.causation_id is None):
            raise ValueError(
                "causation_source and causation_id must be provided together"
            )
        if self.causation_source is not None and self.causation_id is not None:
            if (
                not isinstance(self.causation_source, str)
                or not self.causation_source.strip()
                or not isinstance(self.causation_id, str)
                or not self.causation_id.strip()
            ):
                raise ValueError(
                    "causation_source and causation_id must be non-empty strings"
                )
            if (
                self.causation_source == self.source
                and self.causation_id == self.message_id
            ):
                raise ValueError("message cannot cause itself")

    @property
    def dedup_key(self) -> tuple[str, str]:
        """Producer-scoped identity, analogous to source+id event deduplication."""
        return self.source, self.message_id

    @property
    def ordering_key(self) -> tuple[str, str, int]:
        """Per-task, per-producer ordering key; not a global total order."""
        return self.task_id, self.source, self.sequence_no

    def consume_hop(self) -> "MessageEnvelope":
        """Return a copy with one hop consumed, failing closed at TTL zero."""
        if self.ttl_hops <= 0:
            raise ValueError("TTL_EXHAUSTED")
        return replace(self, ttl_hops=self.ttl_hops - 1)


def validate_envelope_for_flow(
    envelope: MessageEnvelope,
    plan: FlowPlan,
) -> None:
    """Fail closed if envelope metadata does not match the planned information lane."""
    if envelope.kind is not plan.kind:
        raise ValueError("FLOW_KIND_MISMATCH")
    if not plan.path:
        raise ValueError("EMPTY_FLOW_PATH")
    if envelope.source != plan.path[0]:
        raise ValueError("FLOW_SOURCE_MISMATCH")
    if envelope.target != plan.path[-1]:
        raise ValueError("FLOW_TARGET_MISMATCH")
    if envelope.ttl_hops < plan.hop_count:
        raise ValueError("TTL_INSUFFICIENT_FOR_FLOW")


def verify_payload(envelope: MessageEnvelope, payload: bytes) -> None:
    """Verify payload size and digest against immutable envelope metadata."""
    if len(payload) != envelope.payload_bytes:
        raise ValueError("PAYLOAD_SIZE_MISMATCH")
    if sha256_payload(payload) != envelope.payload_digest:
        raise ValueError("PAYLOAD_DIGEST_MISMATCH")
