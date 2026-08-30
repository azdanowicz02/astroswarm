
import numpy as np
import pytest

from astroswarm.analysis.emergent import run_with_traces
from astroswarm.analysis.plots import (
    plot_coverage_map,
    plot_orbit_bands,
    plot_pheromone_map,
)

CFG = {
    "seed": 0,
    "asteroid": {"radius": 500.0, "spin_rate": 2e-4, "mu": 3.5e-5, "n_cells": 200},
    "sensor": {"half_angle": 0.5, "quality_ref_alt": 200.0, "max_alt": 3000.0},
    "simulation": {"dt": 10.0, "t_end": 1500.0, "target_coverage": 0.2},
    "orbit": {"altitude": 400.0, "inclination": 1.2, "raan": 0.0},
    "swarm": {"n_spacecraft": 3, "comms_radius": 5000.0, "decision_interval": 1,
              "min_altitude": 150.0, "max_altitude": 1200.0},
    "decision_weights": {"w_coverage": 1.0, "w_quality": 0.5, "w_pheromone": 0.8,
                         "w_cost": 0.2, "w_neighbor": 0.3},
    "pheromone": {"deposit": 1.0, "evaporation": 0.05, "diffusion": 0.0},
    "maneuvers": {"altitude_step": 150.0, "phase_step": 0.5, "inclination_step": 0.1,
                  "lookahead_steps": 30, "lookahead_stride": 5},
}


def test_traces_have_one_altitude_series_per_agent():
    tr = run_with_traces(CFG, "pheromone", seed=0, stop_at=0.2, record_every=5)
    assert len(tr["altitudes"]) == CFG["swarm"]["n_spacecraft"]
    T = len(tr["time"])
    assert all(len(a) == T for a in tr["altitudes"])
    assert tr["surface_map"].coverage_fraction() > 0.0
    assert np.all(np.asarray(tr["altitudes"][0]) > 0.0)


def test_pheromone_field_present_only_for_pheromone_strategy():
    tr_p = run_with_traces(CFG, "pheromone", seed=0, stop_at=0.2, record_every=5)
    assert tr_p["pheromone"] is not None
    assert tr_p["pheromone"].shape == (CFG["asteroid"]["n_cells"],)
    assert float(tr_p["pheromone"].sum()) > 0.0

    tr_g = run_with_traces(CFG, "greedy", seed=0, stop_at=0.2, record_every=5)
    assert tr_g["pheromone"] is None


def test_emergent_figures_write(tmp_path):
    tr = run_with_traces(CFG, "pheromone", seed=0, stop_at=0.2, record_every=5)
    b = plot_orbit_bands(tr["time"], tr["altitudes"], out_path=tmp_path / "bands.png")
    c = plot_coverage_map(tr["asteroid"], tr["surface_map"], out_path=tmp_path / "cov.png")
    p = plot_pheromone_map(tr["asteroid"], tr["pheromone"], out_path=tmp_path / "phero.png")
    for f in (b, c, p):
        assert f.exists() and f.stat().st_size > 0
