
import math

import numpy as np
import pytest

from astroswarm.analysis.evaluate import (
    aggregate_runs,
    format_aggregate_table,
    run_seeds,
)


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



def test_aggregate_mean_std_and_n():
    summaries = [{"a": 1.0}, {"a": 3.0}, {"a": 5.0}]
    agg = aggregate_runs(summaries)
    assert agg["a"]["mean"] == pytest.approx(3.0)
    assert agg["a"]["std"] == pytest.approx(2.0)          
    assert agg["a"]["n"] == 3
    assert agg["a"]["min"] == 1.0 and agg["a"]["max"] == 5.0


def test_aggregate_drops_none_and_nan():
    summaries = [
        {"time_to_target": None, "x": 1.0},
        {"time_to_target": 10.0, "x": float("nan")},
        {"time_to_target": 20.0, "x": 3.0},
    ]
    agg = aggregate_runs(summaries)
    
    assert agg["time_to_target"]["n"] == 2
    assert agg["time_to_target"]["mean"] == pytest.approx(15.0)
    
    assert agg["x"]["n"] == 2
    assert agg["x"]["mean"] == pytest.approx(2.0)


def test_aggregate_single_run_has_zero_std():
    agg = aggregate_runs([{"a": 7.0}])
    assert agg["a"]["n"] == 1
    assert agg["a"]["std"] == 0.0


def test_aggregate_all_missing_metric_is_empty():
    agg = aggregate_runs([{"a": None}, {"a": None}])
    assert agg["a"]["n"] == 0 and agg["a"]["mean"] is None



def test_run_seeds_aggregates_over_seeds():
    summaries = run_seeds(CFG, "greedy", [0, 1])
    assert len(summaries) == 2
    assert {s["seed"] for s in summaries} == {0, 1}
    for s in summaries:
        assert s["final_coverage"] >= 0.0
        assert s["aggregate_dv"] >= 0.0

    agg = aggregate_runs(summaries)
    assert agg["aggregate_dv"]["n"] == 2
    assert agg["final_coverage"]["mean"] >= 0.0
    
    table = format_aggregate_table(agg)
    assert "final_coverage" in table and "mean" in table


def test_run_seeds_is_reproducible():
    a = run_seeds(CFG, "greedy", [7])[0]
    b = run_seeds(CFG, "greedy", [7])[0]
    assert a["final_coverage"] == pytest.approx(b["final_coverage"])
    assert a["aggregate_dv"] == pytest.approx(b["aggregate_dv"])
