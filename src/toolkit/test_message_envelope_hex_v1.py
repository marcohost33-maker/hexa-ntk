"""Tests for immutable HexaNTK message envelopes."""
from __future__ import annotations

import unittest

from hex_topology_v1 import FlowKind, HexExpertTopology
from message_envelope_hex_v1 import (
    MessageEnvelope,
    sha256_payload,
    validate_envelope_for_flow,
    verify_payload,
)


IDS = ("ade", "sciforge", "qforge", "mrt", "pade", "liouscope")


def envelope(
    *,
    source: str = "ade",
    target: str = "sciforge",
    kind: FlowKind = FlowKind.LOCAL_STATE,
    ttl_hops: int = 1,
    payload: bytes = b"state",
    **kwargs: object,
) -> MessageEnvelope:
    defaults = dict(
        message_id="m-1",
        task_id="task-1",
        trace_id="trace-1",
        source=source,
        target=target,
        kind=kind,
        schema_version="hex.message.v1",
        sequence_no=1,
        ttl_hops=ttl_hops,
        payload_digest=sha256_payload(payload),
        payload_bytes=len(payload),
        causation_source=None,
        causation_id=None,
    )
    defaults.update(kwargs)
    return MessageEnvelope(**defaults)


class TestEnvelopeValidation(unittest.TestCase):
    def test_identity_and_ordering_keys(self) -> None:
        msg = envelope()
        self.assertEqual(msg.dedup_key, ("ade", "m-1"))
        self.assertEqual(msg.ordering_key, ("task-1", "ade", 1))

    def test_invalid_digest_rejected(self) -> None:
        with self.assertRaises(ValueError):
            envelope(payload_digest="abc")

    def test_negative_sequence_ttl_and_size_rejected(self) -> None:
        for field, value in (
            ("sequence_no", -1),
            ("ttl_hops", -1),
            ("payload_bytes", -1),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    envelope(**{field: value})

    def test_incomplete_causation_identity_rejected(self) -> None:
        with self.assertRaises(ValueError):
            envelope(causation_id="parent")

    def test_self_causation_rejected(self) -> None:
        with self.assertRaises(ValueError):
            envelope(
                message_id="same",
                causation_source="ade",
                causation_id="same",
            )

    def test_cross_producer_same_local_id_is_not_self_causation(self) -> None:
        msg = envelope(
            message_id="same",
            causation_source="other-producer",
            causation_id="same",
        )
        self.assertEqual(
            (msg.causation_source, msg.causation_id),
            ("other-producer", "same"),
        )

    def test_consume_hop_is_immutable(self) -> None:
        msg = envelope(ttl_hops=2)
        forwarded = msg.consume_hop()
        self.assertEqual(msg.ttl_hops, 2)
        self.assertEqual(forwarded.ttl_hops, 1)

    def test_consume_hop_fails_at_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "TTL_EXHAUSTED"):
            envelope(ttl_hops=0).consume_hop()


class TestFlowBinding(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = HexExpertTopology.canonical_six(IDS)

    def test_local_envelope_matches_local_plan(self) -> None:
        plan = self.topology.plan_flow(
            "ade", kind=FlowKind.LOCAL_STATE, target="sciforge"
        )
        validate_envelope_for_flow(envelope(), plan)

    def test_kind_mismatch_rejected(self) -> None:
        plan = self.topology.plan_flow(
            "ade", kind=FlowKind.LOCAL_STATE, target="sciforge"
        )
        with self.assertRaisesRegex(ValueError, "FLOW_KIND_MISMATCH"):
            validate_envelope_for_flow(
                envelope(kind=FlowKind.CONTROL), plan
            )

    def test_source_and_target_mismatch_rejected(self) -> None:
        plan = self.topology.plan_flow(
            "ade", kind=FlowKind.LOCAL_STATE, target="sciforge"
        )
        with self.assertRaisesRegex(ValueError, "FLOW_SOURCE_MISMATCH"):
            validate_envelope_for_flow(
                envelope(source="pade"), plan
            )
        with self.assertRaisesRegex(ValueError, "FLOW_TARGET_MISMATCH"):
            validate_envelope_for_flow(
                envelope(target="pade"), plan
            )

    def test_ttl_must_cover_planned_hops(self) -> None:
        plan = self.topology.plan_flow(
            "ade", kind=FlowKind.GLOBAL_SUMMARY, target="pade"
        )
        msg = envelope(
            target="pade",
            kind=FlowKind.GLOBAL_SUMMARY,
            ttl_hops=1,
        )
        with self.assertRaisesRegex(ValueError, "TTL_INSUFFICIENT_FOR_FLOW"):
            validate_envelope_for_flow(msg, plan)


class TestPayloadIntegrity(unittest.TestCase):
    def test_payload_digest_and_size_roundtrip(self) -> None:
        payload = b"immutable research payload"
        msg = envelope(payload=payload)
        verify_payload(msg, payload)

    def test_payload_size_mismatch_rejected_first(self) -> None:
        msg = envelope(payload=b"abc")
        with self.assertRaisesRegex(ValueError, "PAYLOAD_SIZE_MISMATCH"):
            verify_payload(msg, b"abcd")

    def test_payload_digest_mismatch_rejected(self) -> None:
        msg = envelope(payload=b"abc")
        with self.assertRaisesRegex(ValueError, "PAYLOAD_DIGEST_MISMATCH"):
            verify_payload(msg, b"abd")


if __name__ == "__main__":
    unittest.main()
