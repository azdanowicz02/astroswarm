
import numpy as np
import pytest

from astroswarm.environment import Asteroid, Sensor, SurfaceMap
from astroswarm.simulation import MetricsRecorder, Simulation, DecisionContext
from astroswarm.strategies import Greedy, RandomWalk
from astroswarm.strategies.candidates import enumerate_candidate_maneuvers
from astroswarm.swarm import build_swarm
from astroswarm.utils.seeding import make_rng

MU = 3.5e-5
R = 500.0


def make_world(n_cells=600):
    ast = Asteroid(radius=R, spin_rate=2e-4, mu=MU, n_cells=n_cells)
    sensor = Sensor(half_angle=0.5, quality_ref_alt=200.0, max_alt=3000.0)
    sm = SurfaceMap(ast.n_cells)
    return ast, sensor, sm


def make_context(ast, sensor, sm, config=None):
    return DecisionContext(t=0.0, dt=10.0, asteroid=ast, sensor=sensor,
                           surface_map=sm, agents=[], config=config)


def circular(a):
    from astroswarm.dynamics import KeplerOrbit
    return KeplerOrbit.from_elements(mu=MU, a=a, e=0.0, inclination=1.2, raan=0.4, nu=0.0)


class _Agent:
    
    def __init__(self, orbit):
        self.orbit = orbit



def _orbit_signature(o):
    
    g = lambda name: round(float(getattr(o, name, 0.0)), 6)
    return (round(o.a, 3), g("e"), g("inclination"), g("raan"), g("argp"), g("nu"))


def test_randomwalk_returns_valid_menu_action():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    orbit = circular(ast.R + 400.0)
    
    
    menu = enumerate_candidate_maneuvers(orbit, ctx)
    menu_sigs = {_orbit_signature(o) for (_l, o, _d) in menu}

    strat = RandomWalk(rng=make_rng(0))
    agent = _Agent(orbit)
    seen_move = False
    for _ in range(50):
        action = strat.select_action(agent, ctx)
        if action is None:
            continue                          
        new_orbit, dv = action
        assert dv >= 0.0
        assert _orbit_signature(new_orbit) in menu_sigs   
        seen_move = True
    assert seen_move                          



def test_randomwalk_is_reproducible():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    orbit = circular(ast.R + 400.0)

    def run_seed(seed):
        strat = RandomWalk(rng=make_rng(seed))
        agent = _Agent(orbit)
        return [strat.select_action(agent, ctx) is None for _ in range(30)]

    assert run_seed(42) == run_seed(42)        
    
    assert run_seed(1) != run_seed(2)



def test_random_swarm_runs_and_spends_delta_v():
    ast, sensor, sm = make_world()
    master = make_rng(0)
    agents = build_swarm(
        n=8, mu=MU, radius=R, altitude=400.0,
        strategy_factory=lambda i: RandomWalk(rng=make_rng(int(master.integers(2**32)))),
        decision_interval=5,
    )
    sim = Simulation(ast, sensor, sm, agents, dt=10.0, metrics=MetricsRecorder())
    sim.run(2000.0, record_every=10)

    assert sim._total_delta_v() == pytest.approx(sum(a.delta_v for a in agents), rel=1e-12)
    assert sim._total_delta_v() > 0.0          
    assert sm.coverage_fraction() > 0.0



def test_greedy_not_below_random_on_coverage():
    t_end, dt = 4000.0, 10.0

    ast_r, sensor_r, sm_r = make_world(n_cells=500)
    master = make_rng(7)
    rand = build_swarm(
        n=6, mu=MU, radius=R, altitude=400.0,
        strategy_factory=lambda i: RandomWalk(rng=make_rng(int(master.integers(2**32)))),
        decision_interval=20,
    )
    Simulation(ast_r, sensor_r, sm_r, rand, dt=dt, metrics=MetricsRecorder()).run(t_end, record_every=10)

    ast_g, sensor_g, sm_g = make_world(n_cells=500)
    greedy = build_swarm(
        n=6, mu=MU, radius=R, altitude=400.0,
        strategy_factory=lambda i: Greedy(),
        decision_interval=20,
    )
    Simulation(ast_g, sensor_g, sm_g, greedy, dt=dt, metrics=MetricsRecorder()).run(t_end, record_every=10)

    
    
    
    
    assert sm_g.coverage_fraction() >= sm_r.coverage_fraction() - 0.05
