
import numpy as np
import pytest

from astroswarm.dynamics import KeplerOrbit, raise_orbit
from astroswarm.environment import Asteroid, Sensor, SurfaceMap
from astroswarm.simulation import Simulation, MetricsRecorder
from astroswarm.strategies import NoOp
from astroswarm.strategies.base import Strategy
from astroswarm.swarm.spacecraft import Spacecraft

MU = 3.5e-5


def make_orbit():
    return KeplerOrbit.from_elements(mu=MU, a=900.0, e=0.0, inclination=1.2, raan=0.4, nu=0.0)


def make_world(n_cells=800):
    ast = Asteroid(radius=500.0, spin_rate=2e-4, mu=MU, n_cells=n_cells)
    sensor = Sensor(half_angle=0.5, quality_ref_alt=200.0, max_alt=3000.0)
    sm = SurfaceMap(ast.n_cells)
    return ast, sensor, sm



def test_noop_spacecraft_matches_bare_orbit_trajectory():
    sc = Spacecraft(make_orbit(), strategy=None)     
    bare = make_orbit()
    for _ in range(50):
        sc.propagate(10.0)
        bare.propagate(10.0)
        assert np.allclose(sc.position, bare.position, atol=1e-9)
    assert sc.delta_v == 0.0


def test_noop_strategy_is_zero_delta_v():
    sc = Spacecraft(make_orbit(), strategy=NoOp(), decision_interval=1)
    bare = make_orbit()

    class Ctx:  
        pass

    for _ in range(30):
        sc.propagate(10.0)
        sc.decide(Ctx())
        bare.propagate(10.0)
        assert np.allclose(sc.position, bare.position, atol=1e-9)
    assert sc.delta_v == 0.0


def test_noop_through_engine_logs_zero_delta_v_and_maps():
    ast, sensor, sm = make_world()
    agent = Spacecraft(make_orbit(), strategy=NoOp(), decision_interval=5)
    metrics = MetricsRecorder(target_coverage=0.95)
    sim = Simulation(ast, sensor, sm, [agent], dt=10.0, metrics=metrics)
    sim.run(2000.0, record_every=10)
    assert agent.delta_v == 0.0
    assert sim._total_delta_v() == 0.0
    assert sm.coverage_fraction() > 0.0            



class CountingRaise(Strategy):
    

    def __init__(self):
        super().__init__()
        self.calls = 0

    def select_action(self, agent, context):
        self.calls += 1
        return raise_orbit(agent.orbit, 1.0)       


def test_decisions_respect_interval():
    strat = CountingRaise()
    sc = Spacecraft(make_orbit(), strategy=strat, decision_interval=5)

    class Ctx:
        pass

    for _ in range(20):                            
        sc.propagate(10.0)
        sc.decide(Ctx())
    assert strat.calls == 4                         
    assert sc.delta_v > 0.0


def test_maneuvering_agent_accumulates_delta_v_in_engine():
    ast, sensor, sm = make_world()
    strat = CountingRaise()
    agent = Spacecraft(make_orbit(), strategy=strat, decision_interval=10)
    metrics = MetricsRecorder(target_coverage=0.95)
    sim = Simulation(ast, sensor, sm, [agent], dt=10.0, metrics=metrics)
    sim.run(1000.0, record_every=10)               
    assert strat.calls == 10
    assert agent.delta_v > 0.0
    assert sim._total_delta_v() == pytest.approx(agent.delta_v, rel=1e-12)


def test_delta_v_mirrors_orbit():
    sc = Spacecraft(make_orbit(), strategy=CountingRaise(), decision_interval=1)

    class Ctx:
        pass

    sc.propagate(10.0)
    sc.decide(Ctx())
    assert sc.delta_v == pytest.approx(sc.orbit.delta_v, rel=1e-12)
    assert sc.delta_v > 0.0
