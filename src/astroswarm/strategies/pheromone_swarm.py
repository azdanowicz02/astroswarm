
from __future__ import annotations

import numpy as np

from .base import Strategy
from .candidates import enumerate_candidate_maneuvers
from ..swarm.decision import choose_action, weights_from_config
from ..swarm.pheromone import PheromoneMap


class _SharedPheromone:
    

    def __init__(self):
        self.field: PheromoneMap | None = None
        self.last_t: float | None = None

    def ensure(self, context) -> PheromoneMap:
        
        if self.field is None:
            self.field = PheromoneMap.from_config(
                context.surface_map.n_cells, getattr(context, "config", None))
            self.last_t = context.t
        return self.field

    def evaporate_to(self, t: float, dt: float) -> None:
        
        if self.last_t is None:
            self.last_t = t
            return
        steps = int(round((t - self.last_t) / dt)) if dt > 0 else 0
        if steps > 0:
            self.field.evaporate(steps)
            self.last_t = t


class PheromoneSwarm(Strategy):
    

    def __init__(self, shared: _SharedPheromone | None = None, rng=None, **params):
        super().__init__(rng=rng, **params)
        self.shared = shared if shared is not None else _SharedPheromone()

    @classmethod
    def shared_factory(cls, rng=None):
        
        shared = _SharedPheromone()
        return lambda i: cls(shared=shared, rng=rng)

    @property
    def pheromone(self) -> PheromoneMap | None:
        
        return self.shared.field

    def _neighbor_states(self, agent, context):
        
        graph = getattr(context, "neighbours", None)
        if not graph:
            return None
        by_id = {a.agent_id: a for a in context.agents}
        states = []
        for nid in graph.get(agent.agent_id, []):
            nb = by_id.get(nid)
            if nb is not None:
                states.append({"agent_id": nid,
                               "position": np.asarray(nb.position, dtype=float)})
        return states or None

    def select_action(self, agent, context):
        field = self.shared.ensure(context)
        
        self.shared.evaporate_to(context.t, context.dt)
        
        candidates = enumerate_candidate_maneuvers(agent.orbit, context)
        weights = weights_from_config(getattr(context, "config", None))
        neighbors = self._neighbor_states(agent, context)
        chosen = choose_action(candidates, context, pheromone=field,
                               neighbor_states=neighbors, weights=weights)
        
        
        cells = context.sensor.footprint_cells(agent.position, context.asteroid)
        if cells.size:
            q = context.sensor.quality_at(agent.position, context.asteroid)
            if q > 0.0:
                field.deposit(cells, weight=q)
        
        if chosen is None:
            return None
        label, new_orbit, dv = chosen
        if label == "stay":
            return None                      
        return (new_orbit, dv)

    def reset(self) -> None:
        
        self.shared.field = None
        self.shared.last_t = None
