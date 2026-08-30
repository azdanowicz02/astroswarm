
import numpy as np
import pytest

from astroswarm.dynamics import KeplerOrbit
from astroswarm.environment import Asteroid, Sensor, SurfaceMap
from astroswarm.simulation import DecisionContext
from astroswarm.strategies.candidates import enumerate_candidate_maneuvers
from astroswarm.swarm.decision import (
    choose_action,
    neighbour_separation,
    score_action,
    weights_from_config,
)
from astroswarm.swarm.pheromone import PheromoneMap

MU = 3.5e-5
R = 500.0
COMMS = 1000.0



CONFIG = {
    "swarm": {"comms_radius": COMMS, "min_altitude": 150.0, "max_altitude": 1200.0},
    "decision_weights": {
        "w_coverage": 1.0, "w_quality": 0.5, "w_pheromone": 0.8,
        "w_cost": 0.2, "w_neighbor": 0.3,
    },
    "maneuvers": {"lookahead_steps": 120, "lookahead_stride": 5},
}


def make_world(n_cells=800):
    ast = Asteroid(radius=R, spin_rate=2e-4, mu=MU, n_cells=n_cells)
    sensor = Sensor(half_angle=0.5, quality_ref_alt=200.0, max_alt=3000.0)
    sm = SurfaceMap(ast.n_cells)
    return ast, sensor, sm


def make_context(ast, sensor, sm, config=CONFIG):
    return DecisionContext(t=0.0, dt=10.0, asteroid=ast, sensor=sensor,
                           surface_map=sm, agents=[], config=config)


def circular(a, nu=0.0):
    return KeplerOrbit.from_elements(mu=MU, a=a, e=0.0, inclination=1.2,
                                     raan=0.4, nu=nu)


def only(weight_name):
    
    w = {k: 0.0 for k in
         ("w_coverage", "w_quality", "w_pheromone", "w_cost", "w_neighbor")}
    w[weight_name] = 1.0
    return w


def action(orbit, dv=0.0, label="cand"):
    return (label, orbit, dv)



def test_weights_from_config_reads_block_and_falls_back():
    w = weights_from_config(CONFIG)
    assert w["w_coverage"] == 1.0 and w["w_pheromone"] == 0.8
    
    d = weights_from_config({})
    assert d["w_coverage"] == 1.0 and d["w_quality"] == 0.5
    
    p = weights_from_config({"decision_weights": {"w_cost": 5.0}})
    assert p["w_cost"] == 5.0 and p["w_neighbor"] == 0.3


def test_score_defaults_weights_from_context_config():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    a = action(circular(R + 400.0), dv=1.0)
    s_default = score_action(a, ctx)                              
    s_explicit = score_action(a, ctx, weights=weights_from_config(CONFIG))
    assert s_default == pytest.approx(s_explicit)



def test_coverage_term_zeroes_once_mapped():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    a = action(circular(R + 400.0))
    fresh = score_action(a, ctx, weights=only("w_coverage"))
    assert fresh > 0.0                          
    sm.coverage[:] = 1.0                        
    mapped = score_action(a, ctx, weights=only("w_coverage"))
    assert mapped == pytest.approx(0.0, abs=1e-9)



def test_quality_term_zeroes_once_quality_maxed():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    a = action(circular(R + 400.0))
    fresh = score_action(a, ctx, weights=only("w_quality"))
    assert fresh > 0.0                          
    sm.quality[:] = 1.0                         
    maxed = score_action(a, ctx, weights=only("w_quality"))
    assert maxed == pytest.approx(0.0, abs=1e-9)



def test_pheromone_penalty_scales_with_swept_pheromone():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    a = action(circular(R + 400.0))

    
    assert score_action(a, ctx, pheromone=None, weights=only("w_pheromone")) == 0.0

    
    
    light = PheromoneMap(sm.n_cells, deposit=1.0, evaporation=0.0)
    light.value[:] = 1.0
    heavy = PheromoneMap(sm.n_cells, deposit=1.0, evaporation=0.0)
    heavy.value[:] = 3.0
    s_light = score_action(a, ctx, pheromone=light, weights=only("w_pheromone"))
    s_heavy = score_action(a, ctx, pheromone=heavy, weights=only("w_pheromone"))
    assert s_light < 0.0
    assert s_heavy < s_light
    
    assert s_heavy == pytest.approx(3.0 * s_light)



def test_cost_penalty_is_proportional_to_delta_v():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    orb = circular(R + 400.0)
    free = score_action(action(orb, dv=0.0), ctx, weights=only("w_cost"))
    burn = score_action(action(orb, dv=5.0), ctx, weights=only("w_cost"))
    assert free == pytest.approx(0.0)
    assert burn == pytest.approx(-5.0)



def test_neighbour_separation_is_zero_without_neighbours():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    orb = circular(R + 400.0)
    assert neighbour_separation(orb, None, ctx) == 0.0
    assert neighbour_separation(orb, [], ctx) == 0.0


def test_neighbour_separation_grows_with_distance():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    orb = circular(R + 400.0)
    pos = np.asarray(orb.position, dtype=float)

    on_top = [{"agent_id": 1, "position": pos}]
    far = [{"agent_id": 1, "position": pos + np.array([2 * COMMS, 0.0, 0.0])}]
    assert neighbour_separation(orb, on_top, ctx) == pytest.approx(0.0)
    
    assert neighbour_separation(orb, far, ctx) == pytest.approx(1.0)
    
    s = score_action(action(orb), ctx, neighbor_states=far, weights=only("w_neighbor"))
    assert s == pytest.approx(1.0)



def test_choose_action_returns_scored_argmax():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    menu = enumerate_candidate_maneuvers(circular(R + 400.0), ctx)
    ph = PheromoneMap.from_config(sm.n_cells, CONFIG)

    scores = [score_action(a, ctx, pheromone=ph) for a in menu]
    expected = menu[int(np.argmax(scores))]
    chosen = choose_action(menu, ctx, pheromone=ph)
    assert chosen == expected
    
    assert chosen in menu and len(chosen) == 3


def test_choose_action_empty_menu_returns_none():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    assert choose_action([], ctx) is None



def test_scoring_does_not_mutate_the_world():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    ph = PheromoneMap(sm.n_cells, deposit=1.0, evaporation=0.0)
    ph.value[:] = 2.0

    att0 = ast.attitude
    cov0 = sm.coverage.copy()
    qual0 = sm.quality.copy()
    phero0 = ph.value.copy()

    score_action(action(circular(R + 400.0), dv=3.0), ctx,
                 pheromone=ph, neighbor_states=[{"agent_id": 1, "position": [0, 0, 0]}])

    assert ast.attitude == att0                         
    assert np.array_equal(sm.coverage, cov0)            
    assert np.array_equal(sm.quality, qual0)            
    assert np.array_equal(ph.value, phero0)             