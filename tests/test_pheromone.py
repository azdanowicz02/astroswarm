
import numpy as np
import pytest

from astroswarm.swarm.pheromone import PheromoneMap


def test_deposit_raises_only_the_observed_cell():
    ph = PheromoneMap(n_cells=10, deposit=1.0, evaporation=0.02)
    ph.deposit([3])
    assert ph.value[3] == pytest.approx(1.0)
    assert ph.value.sum() == pytest.approx(1.0)      


def test_deposit_accumulates_and_handles_duplicates():
    ph = PheromoneMap(n_cells=10, deposit=2.0)
    ph.deposit([1, 1, 4])                             
    assert ph.value[1] == pytest.approx(4.0)
    assert ph.value[4] == pytest.approx(2.0)
    ph.deposit([1])                                  
    assert ph.value[1] == pytest.approx(6.0)


def test_deposit_weighted_by_quality():
    ph = PheromoneMap(n_cells=5, deposit=1.0)
    ph.deposit([0, 1], weight=0.25)                  
    assert ph.value[0] == pytest.approx(0.25)
    ph.deposit([2, 3], weight=[0.5, 1.0])            
    assert ph.value[2] == pytest.approx(0.5)
    assert ph.value[3] == pytest.approx(1.0)


def test_empty_deposit_is_noop():
    ph = PheromoneMap(n_cells=5)
    ph.deposit([])
    assert ph.value.sum() == 0.0


def test_evaporation_is_geometric():
    ph = PheromoneMap(n_cells=4, deposit=1.0, evaporation=0.1)
    ph.deposit([2])                                  
    for k in range(1, 6):
        ph.evaporate()
        assert ph.value[2] == pytest.approx((1.0 - 0.1) ** k)


def test_evaporation_multi_step_matches_repeated():
    a = PheromoneMap(n_cells=3, deposit=1.0, evaporation=0.05)
    b = PheromoneMap(n_cells=3, deposit=1.0, evaporation=0.05)
    a.deposit([0]); b.deposit([0])
    a.evaporate(steps=7)
    for _ in range(7):
        b.evaporate()
    assert a.value[0] == pytest.approx(b.value[0])


def test_zero_evaporation_holds_value():
    ph = PheromoneMap(n_cells=3, deposit=1.0, evaporation=0.0)
    ph.deposit([0])
    ph.evaporate(steps=100)
    assert ph.value[0] == pytest.approx(1.0)


def test_diffusion_off_is_noop():
    ph = PheromoneMap(n_cells=4, deposit=1.0, evaporation=0.0, diffusion=0.0)
    ph.deposit([1])
    before = ph.value.copy()
    ph.diffuse()                                     
    ph.step()                                        
    assert np.array_equal(ph.value, before)


def test_from_config_reads_pheromone_block():
    cfg = {"pheromone": {"deposit": 3.0, "evaporation": 0.2, "diffusion": 0.0}}
    ph = PheromoneMap.from_config(8, cfg)
    assert ph.n_cells == 8
    assert ph.deposit_amount == 3.0
    assert ph.evaporation == 0.2
    
    ph2 = PheromoneMap.from_config(8, {})
    assert ph2.deposit_amount == 1.0 and ph2.evaporation == 0.02


def test_reset_and_total():
    ph = PheromoneMap(n_cells=5, deposit=1.0)
    ph.deposit([0, 1, 2])
    assert ph.total() == pytest.approx(3.0)
    assert np.allclose(ph.at([0, 1]), [1.0, 1.0])
    ph.reset()
    assert ph.total() == 0.0
