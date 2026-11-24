"""Build bounded mutual contacts using public producer types, never private inventories."""

from __future__ import annotations

import math
import random


def build_neighbor_graph(
    specializations: dict[int, str], locations: dict[int, list[float]], maximum_neighbors: int,
    required_resources: list[str], seed: int, previous_edges: set[tuple[int, int]],
) -> dict[int, set[int]]:
    """Prioritize producer access, then fill spare contacts while favoring existing nearby links."""
    identifiers = sorted(specializations)
    limit = min(maximum_neighbors, max(0, len(identifiers) - 1))
    generator = random.Random(seed)
    best_graph: dict[int, set[int]] = {identifier: set() for identifier in identifiers}
    best_score: tuple[int, int, int, float] | None = None
    for _ in range(64):
        graph: dict[int, set[int]] = {identifier: set() for identifier in identifiers}
        priorities = {identifier: generator.random() for identifier in identifiers}

        def connect(source: int, candidates: list[int]) -> bool:
            """Use a free mutual contact, balancing producer load before geographic preferences."""
            available = [target for target in candidates if target != source and target not in graph[source]
                         and len(graph[target]) < limit and len(graph[source]) < limit]
            if not available:
                return False
            target = min(available, key=lambda candidate: (
                len(graph[candidate]), tuple(sorted((source, candidate))) not in previous_edges,
                priorities[candidate], math.dist(locations[source], locations[candidate]),
            ))
            graph[source].add(target)
            graph[target].add(source)
            return True

        for resource in required_resources:
            producers = [identifier for identifier in identifiers if specializations[identifier] == resource]
            generator.shuffle(producers)
            # Producers also need a distinct neighbor of their own resource type.
            for source in producers:
                if not any(specializations[target] == resource for target in graph[source]):
                    connect(source, producers)
            others = [identifier for identifier in identifiers if specializations[identifier] != resource]
            others.sort(key=lambda identifier: (-len(graph[identifier]), priorities[identifier]))
            for source in others:
                if not any(specializations[target] == resource for target in graph[source]):
                    connect(source, producers)
        while True:
            available = [identifier for identifier in identifiers if len(graph[identifier]) < limit]
            available.sort(key=lambda identifier: (len(graph[identifier]), priorities[identifier]))
            if not any(connect(source, available) for source in available):
                break
        missing = sum(not any(specializations[target] == resource for target in graph[source])
                      for source in identifiers for resource in required_resources)
        edges = {(source, target) for source in identifiers for target in graph[source] if source < target}
        score = (-missing, len(edges), len(edges & previous_edges),
                 -sum(math.dist(locations[source], locations[target]) for source, target in edges))
        if best_score is None or score > best_score:
            best_graph, best_score = graph, score
        if missing == 0 and len(edges) == len(identifiers) * limit // 2 and previous_edges <= edges:
            break
    return best_graph
