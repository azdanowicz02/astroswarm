
import numpy as np
import pytest

from astroswarm.analysis.evaluate import (
    parameter_sweep,
    sensitivity_ranking,
    set_by_path,
    sweep_curve,
    sweep_effect,
)
from astroswarm.analysis.plots import plot_parameter_sweep

CFG = {
    "seed": 0,
    "asteroid": {"radius": 500.0, "spin_rate": 2e-4, "mu": 3.5e-5, "n_cells": 200},
    "sensor": {"half_angle": 0.5, "quality_ref_alt": 200.0, "max_alt": 3000.0},
    "simulation": {"dt": 10.0, "t_end": 1500.0, "target_coverage": 0.2},
    "orbit": {"altitude": 400.0, "inclination": 1.2, "raan": 0.0},
    "swarm": {"n_spacecraft": 3, "comms_radius": 5000.0, "decision_interval": 5,
              "min_altitude": 150.0, "max_altitude": 1200.0},
    "decision_weights": {"w_coverage": 1.0, "w_quality": 0.5, "w_pheromone": 0.8,
                         "w_cost": 0.2, "w_neighbor": 0.3},
    "pheromone": {"deposit": 1.0, "evaporation": 0.05, "diffusion": 0.0},
    "maneuvers": {"altitude_step": 150.0, "phase_step": 0.5, "inclination_step": 0.1,
                  "lookahead_steps": 30, "lookahead_stride": 5},
}


def test_set_by_path_sets_nested_without_mutating_base():
    out = set_by_path(CFG, "decision_weights.w_pheromone", 1.6)
    assert out["decision_weights"]["w_pheromone"] == 1.6
    assert CFG["decision_weights"]["w_pheromone"] == 0.8


def test_set_by_path_creates_missing_intermediate_keys():
    out = set_by_path({"a": 1}, "x.y.z", 5)
    assert out["x"]["y"]["z"] == 5
    assert out["a"] == 1


def test_parameter_sweep_produces_one_result_per_value():
    sw = parameter_sweep(CFG, "swarm.comms_radius", [1000.0, 5000.0],
                         "pheromone", seeds=[0, 1], stop_at=0.2)
    assert [r["value"] for r in sw] == [1000.0, 5000.0]
    assert all(r["param"] == "swarm.comms_radius" for r in sw)
    assert all(len(r["summaries"]) == 2 for r in sw)

    vals, means, stds = sweep_curve(sw, "final_coverage")
    assert vals == [1000.0, 5000.0]
    assert all(m >= 0.0 for m in means)
    assert len(stds) == 2


def test_sweep_effect_measures_spread():
    fake = [
        {"param": "swarm.n_spacecraft", "value": 4,
         "aggregate": {"time_to_target": {"mean": 100.0, "std": 1.0, "n": 3}}},
        {"param": "swarm.n_spacecraft", "value": 8,
         "aggregate": {"time_to_target": {"mean": 60.0, "std": 1.0, "n": 3}}},
    ]
    eff = sweep_effect(fake, "time_to_target")
    assert eff["param"] == "swarm.n_spacecraft"
    assert eff["min"] == 60.0 and eff["max"] == 100.0
    assert eff["range"] == pytest.approx(40.0)
    assert eff["rel_range"] == pytest.approx(40.0 / 80.0)


def test_sensitivity_ranking_orders_by_relative_range():
    strong = [
        {"param": "A", "value": 1, "aggregate": {"m": {"mean": 100.0, "n": 2}}},
        {"param": "A", "value": 2, "aggregate": {"m": {"mean": 20.0, "n": 2}}},
    ]
    weak = [
        {"param": "B", "value": 1, "aggregate": {"m": {"mean": 50.0, "n": 2}}},
        {"param": "B", "value": 2, "aggregate": {"m": {"mean": 48.0, "n": 2}}},
    ]
    ranked = sensitivity_ranking([weak, strong], "m")
    assert [e["param"] for e in ranked] == ["A", "B"]


def test_plot_parameter_sweep_writes_file(tmp_path):
    out = tmp_path / "sweep.png"
    p = plot_parameter_sweep([1e3, 5e3, 1e4], [20.0, 25.0, 30.0], [1.0, 2.0, 3.0],
                             "swarm.comms_radius", "time_to_target",
                             out_path=out, logx=True)
    assert p.exists() and p.stat().st_size > 0
