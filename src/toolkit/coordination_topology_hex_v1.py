"""
coordination_topology_hex_v1.py
==============================

Structural coordination-topology baselines for six-expert HexaFabric ablations.

These metrics describe graph structure only. They do NOT predict task quality and must
not be used to rank architectures without matched-budget empirical task measurements.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import fmean
from typing import Mapping


class TopologyKind(str, Enum):
    INDEPENDENT = "independent"
    CENTRALIZED = "centralized"
    HEX_RING = "hex_ring"
    ALL_TO_ALL = "all_to_all"


@dataclass(frozen=True)
class TopologyMetrics:
    kind: TopologyKind
    n_experts: int
    n_transport_nodes: int
    transport_edge_count: int
    expert_pair_connected_fraction: float
    mean_expert_pair_distance: float | None
    expert_pair_diameter: int | None
    max_transport_degree: int
    mean_transport_degree: float


def _add_edge(graph: dict[str, set[str]], left: str, right: str) -> None:
    graph.setdefault(left, set()).add(right)
    graph.setdefault(right, set()).add(left)


def build_topology(
    kind: TopologyKind,
    expert_ids: tuple[str, ...],
) -> dict[str, frozenset[str]]:
    if len(expert_ids) < 1 or len(set(expert_ids)) != len(expert_ids):
        raise ValueError("expert_ids must be non-empty and unique")
    graph: dict[str, set[str]] = {expert_id: set() for expert_id in expert_ids}

    if kind is TopologyKind.INDEPENDENT:
        pass
    elif kind is TopologyKind.CENTRALIZED:
        relay = "@orchestrator"
        graph[relay] = set()
        for expert_id in expert_ids:
            _add_edge(graph, relay, expert_id)
    elif kind is TopologyKind.HEX_RING:
        if len(expert_ids) != 6:
            raise ValueError("HEX_RING baseline requires exactly six experts")
        for index, expert_id in enumerate(expert_ids):
            _add_edge(graph, expert_id, expert_ids[(index + 1) % len(expert_ids)])
    elif kind is TopologyKind.ALL_TO_ALL:
        for index, left in enumerate(expert_ids):
            for right in expert_ids[index + 1 :]:
                _add_edge(graph, left, right)
    else:
        raise ValueError(f"unsupported topology kind: {kind}")

    return {
        node: frozenset(sorted(neighbours))
        for node, neighbours in sorted(graph.items())
    }


def _shortest_distance(
    graph: Mapping[str, frozenset[str]],
    source: str,
    target: str,
) -> int | None:
    if source == target:
        return 0
    queue: deque[tuple[str, int]] = deque([(source, 0)])
    visited = {source}
    while queue:
        current, distance = queue.popleft()
        for neighbour in graph[current]:
            if neighbour in visited:
                continue
            if neighbour == target:
                return distance + 1
            visited.add(neighbour)
            queue.append((neighbour, distance + 1))
    return None


def measure_topology(
    kind: TopologyKind,
    expert_ids: tuple[str, ...],
) -> TopologyMetrics:
    graph = build_topology(kind, expert_ids)
    distances: list[int] = []
    n_pairs = 0
    for index, left in enumerate(expert_ids):
        for right in expert_ids[index + 1 :]:
            n_pairs += 1
            distance = _shortest_distance(graph, left, right)
            if distance is not None:
                distances.append(distance)

    degrees = [len(neighbours) for neighbours in graph.values()]
    edge_count = sum(degrees) // 2
    connected_fraction = len(distances) / n_pairs if n_pairs else 1.0
    return TopologyMetrics(
        kind=kind,
        n_experts=len(expert_ids),
        n_transport_nodes=len(graph),
        transport_edge_count=edge_count,
        expert_pair_connected_fraction=connected_fraction,
        mean_expert_pair_distance=fmean(distances) if distances else None,
        expert_pair_diameter=max(distances) if distances else None,
        max_transport_degree=max(degrees, default=0),
        mean_transport_degree=fmean(degrees) if degrees else 0.0,
    )


def measure_standard_six() -> dict[TopologyKind, TopologyMetrics]:
    ids = ("e0", "e1", "e2", "e3", "e4", "e5")
    return {kind: measure_topology(kind, ids) for kind in TopologyKind}
