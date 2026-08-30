
from __future__ import annotations

import numpy as np

from ..environment import Asteroid, Sensor, SurfaceMap
from ..simulation import MetricsRecorder, Simulation
from ..swarm import build_swarm_from_config
from ..utils.seeding import make_rng
from .evaluate import STRATEGY_FACTORIES


def _altitudes(agents, radius: float) -> list[float]:
    return [float(np.linalg.norm(a.position) - radius) for a in agents]


def run_with_traces(cfg: dict, strategy: str, seed: int = 0,
                    stop_at: float | None = None, early_stop: bool = True,
                    record_every: int = 20) -> dict:
    
    if strategy not in STRATEGY_FACTORIES:
        raise KeyError(f"unknown strategy {strategy!r}")
    rng = make_rng(seed)
    asteroid = Asteroid(**cfg["asteroid"])
    sensor = Sensor(**cfg["sensor"])
    surface_map = SurfaceMap(asteroid.n_cells)
    init_rng = make_rng(1_000_003 + int(seed))
    agents = build_swarm_from_config(cfg, strategy_factory=STRATEGY_FACTORIES[strategy](rng),
                                     rng=init_rng)

    sim_cfg = cfg["simulation"]
    dt = float(sim_cfg["dt"])
    metrics = MetricsRecorder(target_coverage=sim_cfg["target_coverage"])
    sim = Simulation(asteroid, sensor, surface_map, agents, dt=dt,
                     metrics=metrics, config=cfg)

    if not early_stop:
        stop = None
    elif stop_at is not None:
        stop = stop_at
    else:
        stop = sim_cfg["target_coverage"]

    times = [sim.t]
    alt_rows = [_altitudes(agents, asteroid.R)]
    metrics.record(sim.t, surface_map, sim._total_delta_v(), sim._connectivity())

    n_steps = int(round(float(sim_cfg["t_end"]) / dt))
    for k in range(1, n_steps + 1):
        sim.step()
        reached = stop is not None and surface_map.is_target_reached(stop)
        if k % record_every == 0 or reached:
            times.append(sim.t)
            alt_rows.append(_altitudes(agents, asteroid.R))
            metrics.record(sim.t, surface_map, sim._total_delta_v(), sim._connectivity())
        if reached:
            break

    alt_arr = np.asarray(alt_rows)                      
    altitudes = [alt_arr[:, i] for i in range(alt_arr.shape[1])]

    pheromone = None
    ph = getattr(agents[0].strategy, "pheromone", None)
    if ph is not None:
        pheromone = ph.value.copy()

    return {
        "time": np.asarray(times, dtype=float),
        "altitudes": altitudes,
        "surface_map": surface_map,
        "asteroid": asteroid,
        "agents": agents,
        "metrics": metrics,
        "pheromone": pheromone,
    }
