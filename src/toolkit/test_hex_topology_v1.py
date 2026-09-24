"""Tests for deterministic hex topology and typed information flow."""
from __future__ import annotations

import unittest

from hex_topology_v1 import (
    HEX_DIRECTIONS,
    ORIGIN,
    VIRTUAL_ASSURANCE,
    VIRTUAL_CONTROL,
    VIRTUAL_RELAY,
    FlowKind,
    HexCoord,
    HexExpertTopology,
    hex_distance,
    hex_ring,
    physical_neighbors,
)


IDS = ("ade", "sciforge", "liouscope", "pade", "mrt", "qforge")


class TestHexGeometry(unittest.TestCase):
    def test_cube_constraint_is_implicit(self) -> None:
        coord = HexCoord(3, -5)
        self.assertEqual(coord.q + coord.r + coord.s, 0)

    def test_six_direction_vectors_are_unit_distance(self) -> None:
        self.assertEqual(len(HEX_DIRECTIONS), 6)
        self.assertEqual(len(set(HEX_DIRECTIONS)), 6)
        for direction in HEX_DIRECTIONS:
            self.assertEqual(hex_distance(ORIGIN, direction), 1)

    def test_physical_site_has_six_neighbours(self) -> None:
        neighbours = physical_neighbors(HexCoord(4, -2))
        self.assertEqual(len(neighbours), 6)
        self.assertEqual(len(set(neighbours)), 6)
        self.assertTrue(all(hex_distance(HexCoord(4, -2), n) == 1 for n in neighbours))

    def test_radius_two_ring_has_twelve_sites(self) -> None:
        ring = hex_ring(2)
        self.assertEqual(len(ring), 12)
        self.assertTrue(all(hex_distance(ORIGIN, coord) == 2 for coord in ring))

    def test_radius_zero_ring_is_origin(self) -> None:
        self.assertEqual(hex_ring(0), (ORIGIN,))

    def test_negative_ring_radius_rejected(self) -> None:
        with self.assertRaises(ValueError):
            hex_ring(-1)


class TestCanonicalExpertHexagon(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = HexExpertTopology.canonical_six(IDS)

    def test_exactly_six_unique_project_positions(self) -> None:
        placements = self.topology.placements
        self.assertEqual(set(placements), set(IDS))
        self.assertEqual(set(placements.values()), set(HEX_DIRECTIONS))
        self.assertNotIn(ORIGIN, placements.values())

    def test_each_project_has_two_local_project_neighbours(self) -> None:
        self.assertEqual(
            {expert_id: len(self.topology.neighbors(expert_id)) for expert_id in IDS},
            {expert_id: 2 for expert_id in IDS},
        )

    def test_ring_has_six_local_edges(self) -> None:
        self.assertEqual(self.topology.local_edge_count(), 6)

    def test_ring_local_diameter_is_three(self) -> None:
        self.assertEqual(self.topology.local_diameter(), 3)

    def test_opposite_pairs_exist(self) -> None:
        for expert_id in IDS:
            opposite = self.topology.opposite(expert_id)
            self.assertIsNotNone(opposite)
            self.assertEqual(self.topology.opposite(opposite), expert_id)

    def test_normalized_baseline_weights_sum_to_one(self) -> None:
        weights = self.topology.normalized_local_weights("ade")
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(set(weights.values()), {0.5})

    def test_origin_is_reserved_for_virtual_plane(self) -> None:
        with self.assertRaises(ValueError):
            HexExpertTopology({"bad": ORIGIN})

    def test_duplicate_coordinates_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HexExpertTopology({"a": HexCoord(1, 0), "b": HexCoord(1, 0)})

    def test_canonical_cell_requires_six_unique_ids(self) -> None:
        with self.assertRaises(ValueError):
            HexExpertTopology.canonical_six(("a", "b"))
        with self.assertRaises(ValueError):
            HexExpertTopology.canonical_six(("a", "a", "b", "c", "d", "e"))


class TestInformationFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = HexExpertTopology.canonical_six(IDS)

    def test_local_state_is_exactly_one_semantic_hop(self) -> None:
        plan = self.topology.plan_flow(
            "ade", kind=FlowKind.LOCAL_STATE, target="sciforge"
        )
        self.assertEqual(plan.kind, FlowKind.LOCAL_STATE)
        self.assertEqual(plan.path, ("ade", "sciforge"))
        self.assertEqual(plan.hop_count, 1)
        self.assertNotIn(VIRTUAL_RELAY, plan.path)
        self.assertNotIn(VIRTUAL_CONTROL, plan.path)
        self.assertTrue(plan.may_mutate_compute_state)

    def test_local_state_rejects_hidden_multihop_tunnelling(self) -> None:
        with self.assertRaisesRegex(ValueError, "one semantic hop"):
            self.topology.plan_flow(
                "ade", kind=FlowKind.LOCAL_STATE, target="pade"
            )

    def test_global_summary_uses_virtual_relay(self) -> None:
        plan = self.topology.plan_flow(
            "ade", kind=FlowKind.GLOBAL_SUMMARY, target="pade"
        )
        self.assertEqual(plan.path, ("ade", VIRTUAL_RELAY, "pade"))
        self.assertEqual(plan.hop_count, 2)
        self.assertFalse(plan.may_mutate_compute_state)
        self.assertEqual(plan.payload_policy, "summary-only")

    def test_control_uses_distinct_virtual_control_channel(self) -> None:
        plan = self.topology.plan_flow(
            "ade", kind=FlowKind.CONTROL, target="pade"
        )
        self.assertEqual(plan.path, ("ade", VIRTUAL_CONTROL, "pade"))
        self.assertNotIn(VIRTUAL_RELAY, plan.path)
        self.assertFalse(plan.may_mutate_compute_state)

    def test_evidence_is_one_way_to_assurance_boundary(self) -> None:
        plan = self.topology.plan_flow("pade", kind=FlowKind.EVIDENCE)
        self.assertEqual(plan.path, ("pade", VIRTUAL_ASSURANCE))
        self.assertEqual(plan.payload_policy, "immutable-reference-only")
        self.assertFalse(plan.may_mutate_compute_state)

    def test_evidence_cannot_be_peer_payload(self) -> None:
        with self.assertRaises(ValueError):
            self.topology.plan_flow(
                "pade", kind=FlowKind.EVIDENCE, target="liouscope"
            )

    def test_global_channels_require_known_project_target(self) -> None:
        for kind in (FlowKind.GLOBAL_SUMMARY, FlowKind.CONTROL):
            with self.subTest(kind=kind):
                with self.assertRaises(KeyError):
                    self.topology.plan_flow("ade", kind=kind, target="missing")

    def test_local_channel_requires_target(self) -> None:
        with self.assertRaises(ValueError):
            self.topology.plan_flow("ade", kind=FlowKind.LOCAL_STATE)

    def test_unknown_source_fails_closed(self) -> None:
        with self.assertRaises(KeyError):
            self.topology.plan_flow(
                "missing", kind=FlowKind.GLOBAL_SUMMARY, target="ade"
            )


class TestNonCanonicalTopologies(unittest.TestCase):
    def test_disconnected_local_topology_fails_for_local_path(self) -> None:
        topology = HexExpertTopology(
            {"a": HexCoord(1, 0), "b": HexCoord(4, 0)}
        )
        with self.assertRaises(ValueError):
            topology.shortest_local_path("a", "b")

    def test_global_relay_can_bridge_disconnected_projects(self) -> None:
        topology = HexExpertTopology(
            {"a": HexCoord(1, 0), "b": HexCoord(4, 0)}
        )
        plan = topology.plan_flow("a", kind=FlowKind.GLOBAL_SUMMARY, target="b")
        self.assertEqual(plan.path, ("a", VIRTUAL_RELAY, "b"))


if __name__ == "__main__":
    unittest.main()
