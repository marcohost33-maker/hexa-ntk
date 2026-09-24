"""
hex_topology_v1.py
==================

Deterministic topology and information-flow primitives for HexaNTK cluster cells.

Two different geometries are deliberately separated:
1. physical hex-centre point lattice (triangular Bravais lattice): every interior
   site has six equidistant nearest neighbours; this is not the degree-3 honeycomb
   vertex graph;
2. six-project expert cell: six project experts occupy the radius-1 ring around a
   *virtual* centre. Each project therefore has two project neighbours; the centre
   is not a seventh expert and may only be used as a typed relay/control endpoint.

This module contains no learned router and makes no scientific-validity claim.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Iterable, Mapping


@dataclass(frozen=True, order=True)
class HexCoord:
    """Axial hex coordinate (q, r), with implicit cube coordinate s=-q-r."""

    q: int
    r: int

    @property
    def s(self) -> int:
        return -self.q - self.r

    def __add__(self, other: "HexCoord") -> "HexCoord":
        return HexCoord(self.q + other.q, self.r + other.r)

    def scale(self, factor: int) -> "HexCoord":
        return HexCoord(self.q * factor, self.r * factor)


# Axial basis used by axial_to_cartesian below. With Cartesian +y upward,
# this tuple runs clockwise: E, SE, SW, W, NW, NE.
HEX_DIRECTIONS: tuple[HexCoord, ...] = (
    HexCoord(+1, 0),
    HexCoord(+1, -1),
    HexCoord(0, -1),
    HexCoord(-1, 0),
    HexCoord(-1, +1),
    HexCoord(0, +1),
)

ORIGIN = HexCoord(0, 0)
VIRTUAL_RELAY = "@relay"
VIRTUAL_CONTROL = "@control"
VIRTUAL_ASSURANCE = "@assurance"


def hex_distance(a: HexCoord, b: HexCoord) -> int:
    """Shortest-path distance on an infinite regular hex lattice."""
    dq = a.q - b.q
    dr = a.r - b.r
    ds = a.s - b.s
    return max(abs(dq), abs(dr), abs(ds))


def axial_to_cartesian(coord: HexCoord, h: float = 1.0) -> tuple[float, float]:
    """Map axial lattice coordinates to Cartesian coordinates.

    ``h`` is the nearest-neighbour centre spacing, matching the spacing used by
    the canonical 7-point hex-grid Laplacian in ``src/hexantk_v3.py``.
    """
    if h <= 0:
        raise ValueError("h must be > 0")
    x = h * (coord.q + 0.5 * coord.r)
    y = h * (sqrt(3.0) / 2.0) * coord.r
    return x, y


def physical_neighbors(coord: HexCoord) -> tuple[HexCoord, ...]:
    """The six nearest neighbours of a physical hex-lattice site."""
    return tuple(coord + direction for direction in HEX_DIRECTIONS)


def hex_ring(radius: int) -> tuple[HexCoord, ...]:
    """All lattice coordinates at exactly radius from the origin.

    Returned in deterministic lexicographic order; callers must not infer angular
    orientation from this function. Use HEX_DIRECTIONS for radius-1 orientation.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if radius == 0:
        return (ORIGIN,)
    coords: list[HexCoord] = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            coord = HexCoord(q, r)
            if hex_distance(coord, ORIGIN) == radius:
                coords.append(coord)
    return tuple(sorted(coords))


class FlowKind(str, Enum):
    """Typed information lanes with different semantics and trust boundaries."""

    LOCAL_STATE = "local_state"
    GLOBAL_SUMMARY = "global_summary"
    CONTROL = "control"
    EVIDENCE = "evidence"


@dataclass(frozen=True)
class FlowPlan:
    kind: FlowKind
    path: tuple[str, ...]
    payload_policy: str
    may_mutate_compute_state: bool

    @property
    def hop_count(self) -> int:
        return max(0, len(self.path) - 1)


class HexExpertTopology:
    """Logical expert topology embedded in axial hex coordinates."""

    def __init__(self, placements: Mapping[str, HexCoord]) -> None:
        if not placements:
            raise ValueError("placements must be non-empty")
        if any(not expert_id for expert_id in placements):
            raise ValueError("expert ids must be non-empty")
        if len(set(placements.values())) != len(placements):
            raise ValueError("two experts cannot occupy the same hex coordinate")
        if ORIGIN in placements.values():
            raise ValueError(
                "origin is reserved for the virtual relay/control plane, not an expert"
            )
        self._placements = dict(placements)
        self._adjacency = self._build_local_adjacency()

    @classmethod
    def canonical_six(cls, expert_ids: Iterable[str]) -> "HexExpertTopology":
        ids = tuple(expert_ids)
        if len(ids) != 6:
            raise ValueError("canonical six-project cell requires exactly 6 experts")
        if len(set(ids)) != 6:
            raise ValueError("expert ids must be unique")
        return cls(dict(zip(ids, HEX_DIRECTIONS, strict=True)))

    @property
    def placements(self) -> dict[str, HexCoord]:
        return dict(self._placements)

    def coord(self, expert_id: str) -> HexCoord:
        try:
            return self._placements[expert_id]
        except KeyError as exc:
            raise KeyError(f"unknown expert_id: {expert_id}") from exc

    def neighbors(self, expert_id: str) -> tuple[str, ...]:
        self.coord(expert_id)
        return self._adjacency[expert_id]

    def opposite(self, expert_id: str) -> str | None:
        coord = self.coord(expert_id)
        target = HexCoord(-coord.q, -coord.r)
        for candidate_id, candidate_coord in self._placements.items():
            if candidate_coord == target:
                return candidate_id
        return None

    def local_edge_count(self) -> int:
        return sum(len(neigh) for neigh in self._adjacency.values()) // 2

    def shortest_local_path(self, source: str, target: str) -> tuple[str, ...]:
        self.coord(source)
        self.coord(target)
        if source == target:
            return (source,)

        queue: deque[tuple[str, tuple[str, ...]]] = deque([(source, (source,))])
        visited = {source}
        while queue:
            current, path = queue.popleft()
            for neighbour in self._adjacency[current]:
                if neighbour in visited:
                    continue
                next_path = path + (neighbour,)
                if neighbour == target:
                    return next_path
                visited.add(neighbour)
                queue.append((neighbour, next_path))
        raise ValueError(f"no local path between {source!r} and {target!r}")

    def local_diameter(self) -> int:
        ids = tuple(sorted(self._placements))
        diameter = 0
        for idx, source in enumerate(ids):
            for target in ids[idx + 1 :]:
                diameter = max(
                    diameter, len(self.shortest_local_path(source, target)) - 1
                )
        return diameter

    def normalized_local_weights(self, expert_id: str) -> dict[str, float]:
        """Uniform topology-only baseline weights over current local neighbours.

        These are *not* physical quadrature or neural attention weights. They exist
        only as a deterministic baseline for expert-to-expert message scheduling.
        """
        neighbours = self.neighbors(expert_id)
        if not neighbours:
            return {}
        weight = 1.0 / len(neighbours)
        return {neighbour: weight for neighbour in neighbours}

    def plan_flow(
        self,
        source: str,
        *,
        kind: FlowKind,
        target: str | None = None,
    ) -> FlowPlan:
        """Return a typed, deterministic information path.

        LOCAL_STATE uses only project-to-project edges.
        GLOBAL_SUMMARY and CONTROL use distinct virtual relays to avoid overloading
        the local ring and to keep semantic channels separate.
        EVIDENCE is one-way from a project to the virtual assurance boundary.
        """
        self.coord(source)

        if kind is FlowKind.LOCAL_STATE:
            if target is None:
                raise ValueError("LOCAL_STATE requires a project target")
            self.coord(target)
            if source != target and target not in self.neighbors(source):
                raise ValueError(
                    "LOCAL_STATE is one semantic hop only; use explicit update rounds "
                    "or GLOBAL_SUMMARY for non-neighbour context"
                )
            path = (source,) if source == target else (source, target)
            return FlowPlan(kind, path, "bounded-delta-or-feature", True)

        if kind is FlowKind.GLOBAL_SUMMARY:
            if target is None:
                raise ValueError("GLOBAL_SUMMARY requires a project target")
            self.coord(target)
            path = (source,) if source == target else (source, VIRTUAL_RELAY, target)
            return FlowPlan(kind, path, "summary-only", False)

        if kind is FlowKind.CONTROL:
            if target is None:
                raise ValueError("CONTROL requires a project target")
            self.coord(target)
            path = (source,) if source == target else (source, VIRTUAL_CONTROL, target)
            return FlowPlan(kind, path, "schema-only-no-scientific-payload", False)

        if kind is FlowKind.EVIDENCE:
            if target is not None and target != VIRTUAL_ASSURANCE:
                raise ValueError("EVIDENCE may only target the assurance boundary")
            return FlowPlan(
                kind,
                (source, VIRTUAL_ASSURANCE),
                "immutable-reference-only",
                False,
            )

        raise ValueError(f"unsupported flow kind: {kind}")

    def _build_local_adjacency(self) -> dict[str, tuple[str, ...]]:
        adjacency: dict[str, list[str]] = {
            expert_id: [] for expert_id in self._placements
        }
        ids = tuple(sorted(self._placements))
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                if hex_distance(self._placements[left], self._placements[right]) == 1:
                    adjacency[left].append(right)
                    adjacency[right].append(left)
        return {
            expert_id: tuple(sorted(neighbours))
            for expert_id, neighbours in adjacency.items()
        }
