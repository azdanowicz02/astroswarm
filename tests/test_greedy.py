
import numpy as np
import pytest

from astroswarm.dynamics import CircularOrbit, KeplerOrbit
from astroswarm.environment import Asteroid, Sensor, SurfaceMap
from astroswarm.simulation import Simulation, MetricsRecorder, DecisionContext
from astroswarm.strategies import Greedy
from astroswarm.strategies.candidates import (
    enumerate_candidate_maneuvers,
    estimate_new_coverage,
)
from astroswarm.swarm.spacecraft import Spacecraft

MU = 3.5e-5


def make_world(n_cells=800):
    ast = Asteroid(radius=500.0, spin_rate=2e-4, mu=MU, n_cells=n_cells)
    sensor = Sensor(half_angle=0.5, quality_ref_alt=200.0, max_alt=3000.0)
    sm = SurfaceMap(ast.n_cells)
    return ast, sensor, sm


def make_context(ast, sensor, sm, config=None):
    return DecisionContext(t=0.0, dt=10.0, asteroid=ast, sensor=sensor,
                           surface_map=sm, agents=[], config=config)


def circular(a):
    return KeplerOrbit.from_elements(mu=MU, a=a, e=0.0, inclination=1.2, raan=0.4, nu=0.0)



def test_menu_includes_stay_first_and_expected_labels():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    cands = enumerate_candidate_maneuvers(circular(ast.R + 400.0), ctx)
    labels = [c[0] for c in cands]
    assert labels[0] == "stay"
    for expected in ("raise", "lower", "phase+", "phase-", "incl+", "incl-"):
        assert expected in labels
    
    assert cands[0][2] == 0.0
    assert all(c[2] >= 0.0 for c in cands)


def test_menu_clamps_to_altitude_band():
    ast, sensor, sm = make_world()
    cfg = {"swarm": {"min_altitude": 150.0, "max_altitude": 1200.0}}
    ctx = make_context(ast, sensor, sm, config=cfg)
    
    high = enumerate_candidate_maneuvers(circular(ast.R + 1150.0), ctx)
    assert "raise" not in [c[0] for c in high]
    
    low = enumerate_candidate_maneuvers(circular(ast.R + 200.0), ctx)
    assert "lower" not in [c[0] for c in low]



def test_estimate_new_coverage_is_read_only_and_nonnegative():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    att0 = ast.attitude
    cov0 = sm.coverage.copy()
    g = estimate_new_coverage(circular(ast.R + 400.0), ctx, horizon_steps=100, stride=5)
    assert g >= 0.0
    assert ast.attitude == att0                       
    assert np.array_equal(sm.coverage, cov0)          


def test_estimate_reflects_existing_coverage():
    ast, sensor, sm = make_world()
    ctx = make_context(ast, sensor, sm)
    orb = circular(ast.R + 400.0)
    g_fresh = estimate_new_coverage(orb, ctx, horizon_steps=100, stride=5)
    sm.coverage[:] = 1.0                               
    g_full = estimate_new_coverage(orb, ctx, horizon_steps=100, stride=5)
    assert g_full == pytest.approx(0.0, abs=1e-9)
    assert g_fresh > g_full



def test_greedy_beats_fixed_orbit_on_coverage():
    t_end, dt = 4000.0, 10.0
    a0 = 900.0  

    
    ast_f, sensor_f, sm_f = make_world()
    fixed = CircularOrbit(mu=MU, radius=ast_f.R, altitude=a0 - ast_f.R,
                          inclination=1.2, raan=0.4, phase0=0.0)
    Simulation(ast_f, sensor_f, sm_f, [fixed], dt=dt,
               metrics=MetricsRecorder()).run(t_end, record_every=10)
    cov_fixed = sm_f.coverage_fraction()

    
    ast_g, sensor_g, sm_g = make_world()
    agent = Spacecraft(circular(a0), strategy=Greedy(), decision_interval=20)
    Simulation(ast_g, sensor_g, sm_g, [agent], dt=dt,
               metrics=MetricsRecorder()).run(t_end, record_every=10)
    cov_greedy = sm_g.coverage_fraction()

    assert cov_greedy > cov_fixed
