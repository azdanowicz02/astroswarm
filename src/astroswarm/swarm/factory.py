
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..dynamics import KeplerOrbit
from .spacecraft import Spacecraft

_TWO_PI = 2.0 * np.pi
_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))   



StrategyFactory = Callable[[int], object]


def build_swarm(
    n: int,
    mu: float,
    radius: float,
    altitude: float,
    inclination: float = 1.2,
    *,
    strategy_factory: Optional[StrategyFactory] = None,
    decision_interval: int = 1,
    raan0: float = 0.0,
    eccentricity: float = 0.0,
    inclination_spread: float = 0.0,
    rng=None,
    init_jitter: float = 0.0,
) -> list[Spacecraft]:
    
    if n < 1:
        raise ValueError(f"swarm size must be >= 1, got {n}")

    a = float(radius) + float(altitude)
    idx = np.arange(n)

    raans = raan0 + _TWO_PI * idx / n                 
    nus = (idx * _GOLDEN_ANGLE) % _TWO_PI             
    
    
    
    if init_jitter and rng is not None:
        raans = raans + rng.uniform(-init_jitter, init_jitter, n)
        nus = (nus + rng.uniform(-init_jitter, init_jitter, n)) % _TWO_PI
    if inclination_spread and n > 1:
        incs = inclination + inclination_spread * (idx / (n - 1) - 0.5)
    else:
        incs = np.full(n, float(inclination))

    agents: list[Spacecraft] = []
    for i in range(n):
        orbit = KeplerOrbit.from_elements(
            mu=float(mu),
            a=a,
            e=float(eccentricity),
            inclination=float(incs[i]),
            raan=float(raans[i]),
            nu=float(nus[i]),
        )
        strategy = strategy_factory(i) if strategy_factory is not None else None
        agents.append(
            Spacecraft(orbit, strategy=strategy, agent_id=i,
                       decision_interval=decision_interval)
        )
    return agents


def build_swarm_from_config(cfg: dict, strategy_factory: Optional[StrategyFactory] = None,
                            rng=None):
    
    ast = cfg["asteroid"]
    swarm = cfg.get("swarm", {}) or {}
    orbit = cfg.get("orbit", {}) or {}
    return build_swarm(
        n=int(swarm.get("n_spacecraft", 8)),
        mu=float(ast["mu"]),
        radius=float(ast["radius"]),
        altitude=float(orbit.get("altitude", 400.0)),
        inclination=float(orbit.get("inclination", 1.2)),
        strategy_factory=strategy_factory,
        decision_interval=int(swarm.get("decision_interval", 20)),
        raan0=float(orbit.get("raan", 0.0)),
        inclination_spread=float(swarm.get("inclination_spread", 0.0)),
        rng=rng,
        init_jitter=float(swarm.get("init_jitter", 0.0)),
    )
