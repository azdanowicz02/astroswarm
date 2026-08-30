
from __future__ import annotations

import copy

import numpy as np



_DEFAULT_WEIGHTS = {
    "w_coverage": 1.0,
    "w_quality": 0.5,
    "w_pheromone": 0.8,
    "w_cost": 0.2,
    "w_neighbor": 0.3,
}



_DEFAULT_LOOKAHEAD = {"lookahead_steps": 200, "lookahead_stride": 5}


def weights_from_config(config) -> dict:
    
    w = dict(_DEFAULT_WEIGHTS)
    w.update((config or {}).get("decision_weights", {}) or {})
    return {k: float(v) for k, v in w.items()}


def _lookahead_params(context, horizon_steps, stride) -> tuple[int, int]:
    
    cfg = getattr(context, "config", None) or {}
    m = dict(_DEFAULT_LOOKAHEAD)
    m.update(cfg.get("maneuvers", {}) or {})
    horizon = int(m["lookahead_steps"] if horizon_steps is None else horizon_steps)
    stride = max(1, int(m["lookahead_stride"] if stride is None else stride))
    return horizon, stride


def _comms_radius(context) -> float | None:
    
    cfg = getattr(context, "config", None) or {}
    r = (cfg.get("swarm", {}) or {}).get("comms_radius")
    return float(r) if r is not None else None


def _rollout_terms(orbit, context, pheromone, horizon_steps, stride):
    
    sensor = context.sensor
    surface_map = context.surface_map
    dt = context.dt

    orb = copy.copy(orbit)                
    ast = copy.copy(context.asteroid)     
    predicted_cov = surface_map.coverage.copy()
    predicted_qual = surface_map.quality.copy()
    counted = np.zeros(surface_map.n_cells, dtype=bool)   

    cov_gain = 0.0
    qual_gain = 0.0
    phero_mass = 0.0
    n_samples = max(1, horizon_steps // stride)
    for _ in range(n_samples):
        pos = orb.position
        cells = sensor.footprint_cells(pos, ast)
        if cells.size:
            q = sensor.quality_at(pos, ast)
            if q > 0.0:
                
                cap = 1.0 - predicted_cov[cells]
                add = np.clip(np.minimum(q, cap), 0.0, None)
                cov_gain += float(add.sum())
                predicted_cov[cells] += add
                
                qgain = np.clip(q - predicted_qual[cells], 0.0, None)
                qual_gain += float(qgain.sum())
                predicted_qual[cells] = np.maximum(predicted_qual[cells], q)
                
                if pheromone is not None:
                    fresh = cells[~counted[cells]]
                    if fresh.size:
                        phero_mass += float(pheromone.at(fresh).sum())
                    counted[cells] = True
        for _s in range(stride):
            orb.propagate(dt)
            ast.update(dt)

    return cov_gain, qual_gain, phero_mass


def neighbour_separation(orbit, neighbor_states, context) -> float:
    
    if not neighbor_states:
        return 0.0
    pos = np.asarray(orbit.position, dtype=float)
    dists = np.array([
        np.linalg.norm(pos - np.asarray(s["position"], dtype=float))
        for s in neighbor_states
    ])
    r = _comms_radius(context)
    if r and r > 0.0:
        return float(np.clip(dists / r, 0.0, 1.0).mean())
    scale = float(dists.max())
    if scale <= 0.0:
        return 0.0
    return float((dists / scale).mean())


def score_action(action, context, pheromone=None, neighbor_states=None,
                 weights=None, *, horizon_steps=None, stride=None) -> float:
    
    _label, orbit, delta_v = action
    if weights is None:
        weights = weights_from_config(getattr(context, "config", None))
    horizon, stride = _lookahead_params(context, horizon_steps, stride)

    cov_gain, qual_gain, phero_mass = _rollout_terms(
        orbit, context, pheromone, horizon, stride
    )
    nbr = neighbour_separation(orbit, neighbor_states, context)

    return (
        weights["w_coverage"] * cov_gain
        + weights["w_quality"] * qual_gain
        - weights["w_pheromone"] * phero_mass
        - weights["w_cost"] * float(delta_v)
        + weights["w_neighbor"] * nbr
    )


def choose_action(candidate_actions, context, pheromone=None, neighbor_states=None,
                  weights=None, *, horizon_steps=None, stride=None):
    
    candidate_actions = list(candidate_actions)
    if not candidate_actions:
        return None
    scores = [
        score_action(a, context, pheromone, neighbor_states, weights,
                     horizon_steps=horizon_steps, stride=stride)
        for a in candidate_actions
    ]
    best = int(np.argmax(scores))
    return candidate_actions[best]
