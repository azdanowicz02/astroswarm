
import numpy as np
import pytest

from astroswarm.environment import Asteroid, Sensor, SurfaceMap
from astroswarm.simulation import DecisionContext, MetricsRecorder, Simulation
from astroswarm.strategies import Greedy, PheromoneSwarm
from astroswarm.strategies.pheromone_swarm import _SharedPheromone
from astroswarm.swarm import build_swarm
from astroswarm.swarm.communication import find_neighbors
from astroswarm.swarm.pheromone import PheromoneMap
from astroswarm.utils.seeding import make_rng

MU = 3.5e-5
R = 500.0

CONFIG = {
    "swarm": {"comms_radius": 5000.0, "min_altitude": 150.0, "max_altitude": 1200.0},
    "decision_weights": {
        "w_coverage": 1.0, "w_quality": 0.5, "w_pheromone": 0.8,
        "w_cost": 0.2, "w_neighbor": 0.3,
    },
    "pheromone": {"deposit": 1.0, "evaporation": 0.05, "diffusion": 0.0},
    "maneuvers": {"altitude_step": 150.0, "phase_step": 0.5, "inclination_step": 0.1,
                  "lookahead_steps": 40, "lookahead_stride": 5},
}


def make_world(n_cells=400):
    ast = Asteroid(radius=R, spin_rate=2e-4, mu=MU, n_cells=n_cells)
    sensor = Sensor(half_angle=0.5, quality_ref_alt=200.0, max_alt=3000.0)
    sm = SurfaceMap(ast.n_cells)
    return ast, sensor, sm


def make_swarm(n=4, decision_interval=5, seed=0):
    return build_swarm(
        n=n, mu=MU, radius=R, altitude=400.0,
        strategy_factory=PheromoneSwarm.shared_factory(rng=make_rng(seed)),
        decision_interval=decision_interval,
    )



def test_swarm_runs_and_builds_a_shared_field():
    ast, sensor, sm = make_world()
    agents = make_swarm(n=4)
    sim = Simulation(ast, sensor, sm, agents, dt=10.0,
                     metrics=MetricsRecorder(), config=CONFIG)
    sim.run(300.0, record_every=5)

    
    assert sm.coverage_fraction() > 0.0

    
    field = agents[0].strategy.pheromone
    assert field is not None
    assert all(a.strategy.pheromone is field for a in agents)

    
    assert field.total() > 0.0


def test_action_shape_matches_engine_contract():
    ast, sensor, sm = make_world()
    agents = make_swarm(n=4)
    ctx = DecisionContext(t=0.0, dt=10.0, asteroid=ast, sensor=sensor,
                          surface_map=sm, agents=agents,
                          neighbours=find_neighbors(agents, 5000.0), config=CONFIG)
    action = agents[0].strategy.select_action(agents[0], ctx)
    
    assert action is None or (isinstance(action, tuple) and len(action) == 2)
    if action is not None:
        _new_orbit, dv = action
        assert float(dv) >= 0.0
    
    assert agents[0].strategy.pheromone is not None


def test_is_a_drop_in_for_a_baseline_factory():
    
    common = dict(n=3, mu=MU, radius=R, altitude=400.0, decision_interval=5)
    greedy = build_swarm(strategy_factory=lambda i: Greedy(), **common)
    pher = build_swarm(strategy_factory=PheromoneSwarm.shared_factory(), **common)
    assert len(greedy) == len(pher) == 3
    assert all(isinstance(a.strategy, PheromoneSwarm) for a in pher)



def test_neighbor_states_read_from_context_graph():
    ast, sensor, sm = make_world()
    agents = make_swarm(n=3)
    strat = agents[0].strategy
    graph = {0: [1], 1: [0], 2: []}
    ctx = DecisionContext(t=0.0, dt=10.0, asteroid=ast, sensor=sensor,
                          surface_map=sm, agents=agents, neighbours=graph, config=CONFIG)

    s0 = strat._neighbor_states(agents[0], ctx)
    assert {d["agent_id"] for d in s0} == {1}
    assert np.allclose(s0[0]["position"], agents[1].position)

    
    assert strat._neighbor_states(agents[2], ctx) is None
    ctx.neighbours = None
    assert strat._neighbor_states(agents[0], ctx) is None



def test_shared_field_evaporates_geometrically_once_per_step():
    sp = _SharedPheromone()
    sp.field = PheromoneMap(5, deposit=1.0, evaporation=0.1)
    sp.field.value[:] = 1.0
    sp.last_t = 0.0

    sp.evaporate_to(0.0, 10.0)                     
    assert sp.field.value[0] == pytest.approx(1.0)

    sp.evaporate_to(30.0, 10.0)                    
    assert sp.field.value[0] == pytest.approx((1.0 - 0.1) ** 3)

    frozen = sp.field.value.copy()
    sp.evaporate_to(30.0, 10.0)                    
    assert np.array_equal(sp.field.value, frozen)


def test_reset_clears_shared_field():
    ast, sensor, sm = make_world()
    agents = make_swarm(n=2)
    ctx = DecisionContext(t=0.0, dt=10.0, asteroid=ast, sensor=sensor,
                          surface_map=sm, agents=agents,
                          neighbours=find_neighbors(agents, 5000.0), config=CONFIG)
    agents[0].strategy.select_action(agents[0], ctx)
    assert agents[0].strategy.pheromone is not None
    agents[0].strategy.reset()
    assert agents[0].strategy.pheromone is None
