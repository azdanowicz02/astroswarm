
import numpy as np
import pytest

from astroswarm.dynamics import raise_orbit
from astroswarm.environment import Asteroid, Sensor, SurfaceMap
from astroswarm.simulation import MetricsRecorder, Simulation
from astroswarm.strategies import Greedy, NoOp
from astroswarm.strategies.base import Strategy
from astroswarm.swarm import build_swarm, build_swarm_from_config

MU = 3.5e-5
R = 500.0


def make_world(n_cells=600):
    ast = Asteroid(radius=R, spin_rate=2e-4, mu=MU, n_cells=n_cells)
    sensor = Sensor(half_angle=0.5, quality_ref_alt=200.0, max_alt=3000.0)
    sm = SurfaceMap(ast.n_cells)
    return ast, sensor, sm


class AlwaysRaise(Strategy):
    

    def select_action(self, agent, context):
        return raise_orbit(agent.orbit, 1.0)       



def test_build_swarm_creates_n_distinct_agents():
    agents = build_swarm(n=8, mu=MU, radius=R, altitude=400.0,
                         strategy_factory=lambda i: NoOp())
    assert len(agents) == 8
    assert [a.agent_id for a in agents] == list(range(8))

    
    positions = np.array([a.position for a in agents])
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            assert not np.allclose(positions[i], positions[j], atol=1.0)

    
    assert len({id(a.strategy) for a in agents}) == 8

    
    sma = {round(a.orbit.semi_major_axis, 6) for a in agents}
    assert len(sma) == 1


def test_build_swarm_rejects_empty():
    with pytest.raises(ValueError):
        build_swarm(n=0, mu=MU, radius=R, altitude=400.0)


def test_build_swarm_without_strategy_gives_fixed_agents():
    agents = build_swarm(n=4, mu=MU, radius=R, altitude=400.0)
    assert all(a.strategy is None for a in agents)
    assert all(a.delta_v == 0.0 for a in agents)


def test_build_swarm_from_config_reads_swarm_block():
    cfg = {
        "asteroid": {"radius": R, "mu": MU},
        "orbit": {"altitude": 400.0, "inclination": 1.2, "raan": 0.0},
        "swarm": {"n_spacecraft": 6, "decision_interval": 10},
    }
    agents = build_swarm_from_config(cfg, strategy_factory=lambda i: NoOp())
    assert len(agents) == 6
    assert all(a.decision_interval == 10 for a in agents)



def test_eight_agents_run_and_delta_v_aggregates():
    ast, sensor, sm = make_world()
    agents = build_swarm(n=8, mu=MU, radius=R, altitude=400.0,
                         strategy_factory=lambda i: AlwaysRaise(),
                         decision_interval=5)
    metrics = MetricsRecorder(target_coverage=0.95)
    sim = Simulation(ast, sensor, sm, agents, dt=10.0, metrics=metrics)
    sim.run(2000.0, record_every=10)

    per_agent = [a.delta_v for a in agents]
    assert all(dv > 0.0 for dv in per_agent)                 
    
    assert sim._total_delta_v() == pytest.approx(sum(per_agent), rel=1e-12)
    assert metrics.summary()["total_delta_v"] == pytest.approx(sum(per_agent), rel=1e-12)
    
    assert sm.coverage_fraction() > 0.0


def test_greedy_swarm_runs_through_engine():
    ast, sensor, sm = make_world(n_cells=400)
    agents = build_swarm(n=8, mu=MU, radius=R, altitude=400.0,
                         strategy_factory=lambda i: Greedy(),
                         decision_interval=25)
    sim = Simulation(ast, sensor, sm, agents, dt=10.0, metrics=MetricsRecorder())
    sim.run(3000.0, record_every=25)
    assert sm.coverage_fraction() > 0.0
    assert sim._total_delta_v() >= 0.0


def test_swarm_maps_more_than_single_agent_same_time():
    
    t_end, dt = 3000.0, 10.0

    ast1, sensor1, sm1 = make_world(n_cells=600)
    one = build_swarm(n=1, mu=MU, radius=R, altitude=400.0,
                      strategy_factory=lambda i: NoOp())
    Simulation(ast1, sensor1, sm1, one, dt=dt, metrics=MetricsRecorder()).run(t_end, record_every=10)

    ast8, sensor8, sm8 = make_world(n_cells=600)
    many = build_swarm(n=8, mu=MU, radius=R, altitude=400.0,
                       strategy_factory=lambda i: NoOp())
    Simulation(ast8, sensor8, sm8, many, dt=dt, metrics=MetricsRecorder()).run(t_end, record_every=10)

    assert sm8.coverage_fraction() > sm1.coverage_fraction()
