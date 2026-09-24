"""Tests for structural coordination topology baselines."""
from __future__ import annotations

import unittest

from coordination_topology_hex_v1 import (
    TopologyKind,
    build_topology,
    measure_standard_six,
    measure_topology,
)


IDS = ("a", "b", "c", "d", "e", "f")


class TestTopologyBuilders(unittest.TestCase):
    def test_independent_has_no_edges(self) -> None:
        graph = build_topology(TopologyKind.INDEPENDENT, IDS)
        self.assertTrue(all(not neighbours for neighbours in graph.values()))

    def test_centralized_uses_one_virtual_orchestrator(self) -> None:
        graph = build_topology(TopologyKind.CENTRALIZED, IDS)
        self.assertIn("@orchestrator", graph)
        self.assertEqual(len(graph["@orchestrator"]), 6)
        self.assertTrue(all("@orchestrator" in graph[item] for item in IDS))

    def test_hex_ring_has_degree_two(self) -> None:
        graph = build_topology(TopologyKind.HEX_RING, IDS)
        self.assertTrue(all(len(graph[item]) == 2 for item in IDS))

    def test_hex_ring_requires_exactly_six_experts(self) -> None:
        with self.assertRaises(ValueError):
            build_topology(TopologyKind.HEX_RING, ("a", "b", "c"))

    def test_all_to_all_has_degree_five(self) -> None:
        graph = build_topology(TopologyKind.ALL_TO_ALL, IDS)
        self.assertTrue(all(len(graph[item]) == 5 for item in IDS))

    def test_duplicate_expert_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_topology(
                TopologyKind.ALL_TO_ALL,
                ("a", "a"),
            )


class TestTopologyMetrics(unittest.TestCase):
    def test_six_ring_metrics(self) -> None:
        metrics = measure_topology(TopologyKind.HEX_RING, IDS)
        self.assertEqual(metrics.transport_edge_count, 6)
        self.assertEqual(metrics.n_transport_nodes, 6)
        self.assertEqual(metrics.expert_pair_connected_fraction, 1.0)
        self.assertAlmostEqual(metrics.mean_expert_pair_distance, 1.8)
        self.assertEqual(metrics.expert_pair_diameter, 3)
        self.assertEqual(metrics.max_transport_degree, 2)
        self.assertAlmostEqual(metrics.mean_transport_degree, 2.0)

    def test_centralized_metrics(self) -> None:
        metrics = measure_topology(TopologyKind.CENTRALIZED, IDS)
        self.assertEqual(metrics.transport_edge_count, 6)
        self.assertEqual(metrics.n_transport_nodes, 7)
        self.assertEqual(metrics.expert_pair_connected_fraction, 1.0)
        self.assertAlmostEqual(metrics.mean_expert_pair_distance, 2.0)
        self.assertEqual(metrics.expert_pair_diameter, 2)
        self.assertEqual(metrics.max_transport_degree, 6)

    def test_all_to_all_metrics(self) -> None:
        metrics = measure_topology(TopologyKind.ALL_TO_ALL, IDS)
        self.assertEqual(metrics.transport_edge_count, 15)
        self.assertEqual(metrics.n_transport_nodes, 6)
        self.assertEqual(metrics.expert_pair_connected_fraction, 1.0)
        self.assertAlmostEqual(metrics.mean_expert_pair_distance, 1.0)
        self.assertEqual(metrics.expert_pair_diameter, 1)
        self.assertEqual(metrics.max_transport_degree, 5)

    def test_independent_metrics_have_no_pair_distance(self) -> None:
        metrics = measure_topology(TopologyKind.INDEPENDENT, IDS)
        self.assertEqual(metrics.transport_edge_count, 0)
        self.assertEqual(metrics.expert_pair_connected_fraction, 0.0)
        self.assertIsNone(metrics.mean_expert_pair_distance)
        self.assertIsNone(metrics.expert_pair_diameter)

    def test_standard_six_returns_every_baseline(self) -> None:
        metrics = measure_standard_six()
        self.assertEqual(set(metrics), set(TopologyKind))


if __name__ == "__main__":
    unittest.main()
