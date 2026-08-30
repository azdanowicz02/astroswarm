
import numpy as np
import pytest

from astroswarm.optimization.evolution import (
    WEIGHT_KEYS,
    benchmark,
    decode,
    default_genome,
    encode,
    evolve,
    fitness,
)
from astroswarm.analysis.plots import plot_evolution_history, plot_evolved_vs_manual

CFG = {
    "seed": 0,
    "asteroid": {"radius": 500.0, "spin_rate": 2e-4, "mu": 3.5e-5, "n_cells": 120},
    "sensor": {"half_angle": 0.5, "quality_ref_alt": 200.0, "max_alt": 3000.0},
    "simulation": {"dt": 10.0, "t_end": 800.0, "target_coverage": 0.2},
    "orbit": {"altitude": 400.0, "inclination": 1.2, "raan": 0.0},
    "swarm": {"n_spacecraft": 3, "comms_radius": 5000.0, "decision_interval": 1,
              "min_altitude": 150.0, "max_altitude": 1200.0},
    "decision_weights": {"w_coverage": 1.0, "w_quality": 0.5, "w_pheromone": 0.8,
                         "w_cost": 0.2, "w_neighbor": 0.3},
    "pheromone": {"deposit": 1.0, "evaporation": 0.05, "diffusion": 0.0},
    "maneuvers": {"altitude_step": 150.0, "phase_step": 0.5, "inclination_step": 0.1,
                  "lookahead_steps": 20, "lookahead_stride": 5},
}


def test_encode_decode_roundtrip():
    g = encode(CFG["decision_weights"])
    assert list(g) == [1.0, 0.5, 0.8, 0.2, 0.3]
    assert decode(g) == pytest.approx(CFG["decision_weights"])
    assert np.allclose(default_genome(CFG), g)


def test_fitness_is_finite_and_bounded():
    g = default_genome(CFG)
    f = fitness(g, CFG, seeds=[0], stop_at=0.2)
    assert isinstance(f, float)
    assert -10.0 < f <= 1.0                      
    
    assert f == pytest.approx(fitness(g, CFG, seeds=[0], stop_at=0.2))


def test_evolve_history_is_monotone_and_reproducible():
    kw = dict(seeds=[0], generations=2, pop_size=3, sigma=0.3, elite=2,
              seed=0, stop_at=0.2)
    res = evolve(CFG, **kw)
    assert set(res["best_weights"]) == set(WEIGHT_KEYS)
    
    h = res["history"]
    assert len(h) == 3                            
    assert all(h[i + 1] >= h[i] - 1e-12 for i in range(len(h) - 1))
    assert res["best_fitness"] == pytest.approx(max(h))
    
    assert res["best_fitness"] >= fitness(default_genome(CFG), CFG, seeds=[0], stop_at=0.2) - 1e-9
    
    res2 = evolve(CFG, **kw)
    assert np.allclose(res["best_genome"], res2["best_genome"])
    assert res["history"] == pytest.approx(res2["history"])


def test_benchmark_reports_manual_and_evolved():
    b = benchmark(CFG, evolved_weights=CFG["decision_weights"],
                  seeds=[0], stop_at=0.2)
    for side in ("manual", "evolved"):
        assert "fitness" in b[side] and "aggregate" in b[side]
        assert set(b[side]["weights"]) == set(WEIGHT_KEYS)
    
    assert b["manual"]["fitness"] == pytest.approx(b["evolved"]["fitness"])


def test_evolution_figures_write(tmp_path):
    res = evolve(CFG, seeds=[0], generations=2, pop_size=3, seed=0, stop_at=0.2)
    b = benchmark(CFG, res["best_weights"], seeds=[0], stop_at=0.2)
    h = plot_evolution_history(res["history"], out_path=tmp_path / "hist.png")
    c = plot_evolved_vs_manual(b, out_path=tmp_path / "cmp.png")
    assert h.exists() and h.stat().st_size > 0
    assert c.exists() and c.stat().st_size > 0
