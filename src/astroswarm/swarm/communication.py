
from __future__ import annotations

import numpy as np


def find_neighbors(agents, comms_radius: float) -> dict[int, list[int]]:
    
    positions = {a.agent_id: np.asarray(a.position, dtype=float) for a in agents}
    neighbors: dict[int, list[int]] = {aid: [] for aid in positions}
    ids = list(positions)
    for i, ai in enumerate(ids):
        for aj in ids[i + 1:]:
            if np.linalg.norm(positions[ai] - positions[aj]) <= comms_radius:
                neighbors[ai].append(aj)
                neighbors[aj].append(ai)
    return neighbors


def gather_local_state(agent) -> dict:
    
    return {
        "agent_id": getattr(agent, "agent_id", None),
        "position": np.asarray(agent.position, dtype=float),
        "delta_v": float(getattr(agent, "delta_v", 0.0)),
    }


def share_state(agents, neighbors) -> dict[int, list[dict]]:
    
    states = {a.agent_id: gather_local_state(a) for a in agents}
    by_id = {a.agent_id: a for a in agents}
    shared: dict[int, list[dict]] = {}
    for aid, nbr_ids in neighbors.items():
        shared[aid] = [states[n] for n in nbr_ids]
        by_id[aid].neighbor_states = shared[aid]
    return shared


def connectivity(agents, comms_radius: float) -> float:
    
    ids = [a.agent_id for a in agents]
    n = len(ids)
    if n <= 1:
        return 1.0
    adj = find_neighbors(agents, comms_radius)

    seen: set[int] = set()
    connected_pairs = 0
    for start in ids:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        connected_pairs += size * (size - 1)

    return connected_pairs / (n * (n - 1))
