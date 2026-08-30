
import numpy as np
import pytest

from astroswarm.environment import Asteroid, Sensor, SurfaceMap
from astroswarm.simulation import MetricsRecorder, Simulation
from astroswarm.strategies.base import Strategy
from astroswarm.swarm import build_swarm
from astroswarm.swarm.communication import (
    connectivity,
    find_neighbors,
    share_state,
)

MU = 3.5e-5
R = 500.0


class _Agent:
    def __init__(self, agent_id, pos):
        self.agent_id = agent_id
        self.position = np.asarray(pos, dtype=float)
        self.delta_v = 0.0


def _brute_force_neighbors(agents, radius):
    
    out = {a.agent_id: set() for a in agents}
    for a in agents:
        for b in agents:
            if a.agent_id == b.agent_id:
                continue
            if np.linalg.norm(a.position - b.position) <= radius:
                out[a.agent_id].add(b.agent_id)
    return out



def test_find_neighbors_matches_brute_force_random():
    rng = np.random.default_rng(0)
    agents = [_Agent(i, rng.normal(size=3) * 500.0) for i in range(12)]
    for radius in (100.0, 400.0, 800.0, 2000.0):
        got = {k: set(v) for k, v in find_neighbors(agents, radius).items()}
        assert got == _brute_force_neighbors(agents, radius)


def test_find_neighbors_symmetric_and_excludes_self():
    agents = [_Agent(0, [0, 0, 0]), _Agent(1, [100, 0, 0]), _Agent(2, [5000, 0, 0])]
    nbrs = find_neighbors(agents, comms_radius=200.0)
    assert 1 in nbrs[0] and 0 in nbrs[1]      
    assert 0 not in nbrs[0]                    
    assert nbrs[2] == []                       


def test_find_neighbors_boundary_is_inclusive():
    agents = [_Agent(0, [0, 0, 0]), _Agent(1, [100, 0, 0])]
    assert find_neighbors(agents, 100.0)[0] == [1]   
    assert find_neighbors(agents, 99.999)[0] == []   



def test_share_state_is_range_limited():
    agents = [_Agent(0, [0, 0, 0]), _Agent(1, [50, 0, 0]),
              _Agent(2, [5000, 0, 0])]
    nbrs = find_neighbors(agents, comms_radius=100.0)
    shared = share_state(agents, nbrs)
    
    assert {s["agent_id"] for s in shared[0]} == {1}
    assert {s["agent_id"] for s in shared[1]} == {0}
    assert shared[2] == []
    
    assert {s["agent_id"] for s in agents[0].neighbor_states} == {1}
    
    assert np.allclose(shared[1][0]["position"], [0, 0, 0])



def test_connectivity_tracks_a_splitting_swarm():
    close = [_Agent(0, [0, 0, 0]), _Agent(1, [10, 0, 0])]
    assert connectivity(close, 100.0) == pytest.approx(1.0)
    far = [_Agent(0, [0, 0, 0]), _Agent(1, [10_000, 0, 0])]
    assert connectivity(far, 100.0) == pytest.approx(0.0)



class _SpyStrategy(Strategy):
    
    def __init__(self):
        super().__init__()
        self.seen_neighbours = None

    def select_action(self, agent, context):
        self.seen_neighbours = context.neighbours
        return None


def test_engine_populates_context_neighbours():
    ast = Asteroid(radius=R, spin_rate=2e-4, mu=MU, n_cells=300)
    sensor = Sensor(half_angle=0.5, quality_ref_alt=200.0, max_alt=3000.0)
    sm = SurfaceMap(ast.n_cells)
    spies = []

    def make_spy(i):
        s = _SpyStrategy()
        spies.append(s)
        return s

    agents = build_swarm(n=4, mu=MU, radius=R, altitude=400.0,
                         strategy_factory=make_spy, decision_interval=1)
    cfg = {"swarm": {"comms_radius": 5000.0}}
    sim = Simulation(ast, sensor, sm, agents, dt=10.0,
                     metrics=MetricsRecorder(), config=cfg)
    sim.run(20.0, record_every=1)

    
    for s in spies:
        assert isinstance(s.seen_neighbours, dict)
        assert set(s.seen_neighbours.keys()) == {0, 1, 2, 3}
