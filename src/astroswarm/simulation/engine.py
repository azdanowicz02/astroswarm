
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ..environment import Asteroid, Sensor, SurfaceMap
from .metrics import MetricsRecorder


@dataclass
class DecisionContext:
    
    t: float
    dt: float
    asteroid: Asteroid
    sensor: Sensor
    surface_map: SurfaceMap
    agents: list
    pheromone: object = None
    neighbours: object = None
    config: object = None


class Simulation:
    

    def __init__(
        self,
        asteroid: Asteroid,
        sensor: Sensor,
        surface_map: SurfaceMap,
        agents: Iterable,
        dt: float,
        metrics: MetricsRecorder | None = None,
        config: Optional[dict] = None,
    ):
        self.asteroid = asteroid
        self.sensor = sensor
        self.surface_map = surface_map
        self.agents = list(agents)
        self.dt = float(dt)
        self.metrics = metrics or MetricsRecorder()
        self.config = config
        self.t = 0.0

    def _total_delta_v(self) -> float:
        return sum(getattr(a, "delta_v", 0.0) for a in self.agents)

    def _comms_radius(self) -> float | None:
        
        cfg = self.config or {}
        swarm = cfg.get("swarm", {}) or {}
        r = swarm.get("comms_radius")
        return float(r) if r is not None else None

    def _connectivity(self) -> float | None:
        
        r = self._comms_radius()
        if r is None or not all(hasattr(a, "agent_id") for a in self.agents):
            return None
        from ..swarm.communication import connectivity
        return connectivity(self.agents, r)

    def _neighbours(self):
        
        r = self._comms_radius()
        if r is None or not all(hasattr(a, "agent_id") for a in self.agents):
            return None
        from ..swarm.communication import find_neighbors
        return find_neighbors(self.agents, r)

    def _make_context(self) -> DecisionContext:
        
        return DecisionContext(
            t=self.t,
            dt=self.dt,
            asteroid=self.asteroid,
            sensor=self.sensor,
            surface_map=self.surface_map,
            agents=self.agents,
            neighbours=self._neighbours(),
            config=self.config,
        )

    def step(self) -> None:
        
        self.asteroid.update(self.dt)
        for agent in self.agents:
            agent.propagate(self.dt)
            self.sensor.observe(agent.position, self.asteroid, self.surface_map)
        self.t += self.dt

        
        
        context = None
        for agent in self.agents:
            decide = getattr(agent, "decide", None)
            if decide is None:
                continue
            if context is None:
                context = self._make_context()
            decide(context)

    def run(
        self,
        t_end: float,
        record_every: int = 1,
        stop_at_target: float | None = None,
    ) -> MetricsRecorder:
        
        
        self.metrics.record(self.t, self.surface_map, self._total_delta_v(), self._connectivity())
        n_steps = int(round(t_end / self.dt))
        for k in range(1, n_steps + 1):
            self.step()
            if k % record_every == 0:
                self.metrics.record(self.t, self.surface_map, self._total_delta_v(), self._connectivity())
            if stop_at_target is not None and self.surface_map.is_target_reached(stop_at_target):
                self.metrics.record(self.t, self.surface_map, self._total_delta_v(), self._connectivity())
                break
        return self.metrics
