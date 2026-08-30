
import numpy as np
import pytest

from astroswarm.environment import Asteroid, Sensor, SurfaceMap
from astroswarm.simulation import MetricsRecorder, Simulation
from astroswarm.strategies import Greedy
from astroswarm.swarm import build_swarm

MU = 3.5e-5
R = 500.0


def test_observations_count_useful_looks_only():
    sm = SurfaceMap(5)
    sm.observe([0, 1], 0.5)                 
    assert sm.observations[0] == 1 and sm.observations[1] == 1
    sm.observe([0], 0.0)                     
    assert sm.observations[0] == 1
    sm.observe([0], 0.3)                     
    assert sm.observations[0] == 2


def test_observations_accumulate_duplicates():
    sm = SurfaceMap(5)
    sm.observe([2, 2, 3], 1.0)               
    assert sm.observations[2] == 2
    assert sm.observations[3] == 1


def test_mean_reobservations_math():
    sm = SurfaceMap(4)
    assert sm.mean_reobservations() == 0.0                  
    sm.observe([0], 1.0)                                    
    assert sm.mean_reobservations() == pytest.approx(0.0)   
    sm.observe([0], 0.5)
    sm.observe([1], 0.5)                                    
    
    assert sm.mean_reobservations() == pytest.approx(0.5)
    assert sm.total_observations() == 3


def test_metrics_track_redundancy_series_and_summary():
    sm = SurfaceMap(4)
    rec = MetricsRecorder(target_coverage=0.95)
    rec.record(0.0, sm)                                     
    sm.observe([0, 1], 1.0)
    sm.observe([0], 1.0)                                    
    rec.record(1.0, sm)

    assert rec.redundancy[-1] == pytest.approx(0.5)
    assert rec.summary()["final_redundancy"] == pytest.approx(0.5)
    arrs = rec.as_arrays()
    assert "redundancy" in arrs and arrs["redundancy"].shape == (2,)


def test_redundancy_populated_in_a_swarm_run():
    ast = Asteroid(radius=R, spin_rate=2e-4, mu=MU, n_cells=300)
    sensor = Sensor(half_angle=0.5, quality_ref_alt=200.0, max_alt=3000.0)
    sm = SurfaceMap(ast.n_cells)
    agents = build_swarm(n=3, mu=MU, radius=R, altitude=400.0,
                         strategy_factory=lambda i: Greedy(), decision_interval=5)
    rec = MetricsRecorder()
    Simulation(ast, sensor, sm, agents, dt=10.0, metrics=rec,
               config={"swarm": {"comms_radius": 5000.0}}).run(200.0, record_every=5)

    assert len(rec.redundancy) == len(rec.coverage)          
    assert rec.summary()["final_redundancy"] >= 0.0
    assert sm.total_observations() > 0
