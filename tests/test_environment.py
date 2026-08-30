
import numpy as np
import pytest

from astroswarm.environment import Asteroid, Sensor, SurfaceMap
from astroswarm.utils.geometry import fibonacci_sphere


def make_asteroid(n_cells=500):
    return Asteroid(radius=500.0, spin_rate=2e-4, mu=3.5e-5, n_cells=n_cells)



def test_fibonacci_sphere_unit_vectors():
    pts = fibonacci_sphere(1000)
    norms = np.linalg.norm(pts, axis=1)
    assert pts.shape == (1000, 3)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_asteroid_rotation_advances_and_wraps():
    ast = make_asteroid()
    ast.update(100.0)
    assert ast.attitude == pytest.approx(2e-4 * 100.0)
    
    for _ in range(100000):
        ast.update(100.0)
    assert 0.0 <= ast.attitude < 2 * np.pi


def test_cell_positions_lie_on_surface():
    ast = make_asteroid()
    ast.update(1234.0)
    pos = ast.cell_positions_inertial()
    assert np.allclose(np.linalg.norm(pos, axis=1), ast.R, atol=1e-6)


def test_visible_cells_are_roughly_half():
    ast = make_asteroid(n_cells=2000)
    sc = np.array([2000.0, 0.0, 0.0])
    vis = ast.visible_cells(sc)
    frac = len(vis) / ast.n_cells
    
    assert 0.3 < frac < 0.7



def test_surface_map_monotonic_and_saturating():
    sm = SurfaceMap(10)
    sm.observe([0, 1], [0.4, 0.4])
    sm.observe([0], [0.9])          
    assert sm.quality[0] == pytest.approx(0.9)
    assert sm.coverage[0] == pytest.approx(1.0)   
    assert sm.coverage[1] == pytest.approx(0.4)
    assert 0.0 <= sm.coverage_fraction() <= 1.0


def test_surface_map_target_and_empty_observe():
    sm = SurfaceMap(4)
    sm.observe([], [])              
    assert sm.coverage_fraction() == 0.0
    sm.observe([0, 1, 2, 3], 1.0)   
    assert sm.is_target_reached(0.95)



def test_sensor_quality_decreases_with_altitude():
    ast = make_asteroid()
    sensor = Sensor(half_angle=0.35, quality_ref_alt=200.0, max_alt=3000.0)
    low = sensor.quality_at(np.array([ast.R + 200.0, 0, 0]), ast)
    high = sensor.quality_at(np.array([ast.R + 1000.0, 0, 0]), ast)
    too_high = sensor.quality_at(np.array([ast.R + 5000.0, 0, 0]), ast)
    assert low == pytest.approx(1.0)
    assert 0.0 < high < low
    assert too_high == 0.0


def test_sensor_observe_increases_coverage():
    ast = make_asteroid(n_cells=2000)
    sensor = Sensor(half_angle=0.5, quality_ref_alt=200.0, max_alt=3000.0)
    sm = SurfaceMap(ast.n_cells)
    sc = np.array([ast.R + 300.0, 0.0, 0.0])
    cells, q = sensor.observe(sc, ast, sm)
    assert len(cells) > 0
    assert q > 0.0
    assert sm.coverage_fraction() > 0.0
    
    assert np.count_nonzero(sm.coverage) == len(cells)
