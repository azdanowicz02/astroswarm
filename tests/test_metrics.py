
import json

import numpy as np
import pytest

from astroswarm.environment import Asteroid, Sensor, SurfaceMap
from astroswarm.simulation import MetricsRecorder, Simulation
from astroswarm.simulation.logger import RunLogger
from astroswarm.strategies import NoOp
from astroswarm.swarm import build_swarm
from astroswarm.swarm.communication import connectivity

MU = 3.5e-5
R = 500.0


class _FakeAgent:
    def __init__(self, agent_id, pos):
        self.agent_id = agent_id
        self.position = np.asarray(pos, dtype=float)


class _FakeMap:
    def __init__(self, cov=0.5, qual=0.3):
        self._cov, self._qual = cov, qual
    def coverage_fraction(self):
        return self._cov
    def mean_quality(self):
        return self._qual
    def mean_reobservations(self):
        return 0.0



def test_connectivity_extremes_and_partial():
    
    agents = [_FakeAgent(0, [0, 0, 0]), _FakeAgent(1, [10, 0, 0]),
              _FakeAgent(2, [1000, 0, 0]), _FakeAgent(3, [1010, 0, 0])]
    
    assert connectivity(agents, comms_radius=50.0) == pytest.approx(1.0 / 3.0)
    
    assert connectivity(agents, comms_radius=1e6) == pytest.approx(1.0)
    
    assert connectivity(agents, comms_radius=1.0) == pytest.approx(0.0)
    
    assert connectivity([_FakeAgent(0, [0, 0, 0])], comms_radius=1.0) == 1.0



def test_recorder_tracks_connectivity_summary():
    m = MetricsRecorder(target_coverage=0.95)
    m.record(0.0, _FakeMap(), 0.0, connectivity=1.0)
    m.record(10.0, _FakeMap(), 5.0, connectivity=0.5)
    s = m.summary()
    assert s["mean_connectivity"] == pytest.approx(0.75)
    assert s["final_connectivity"] == pytest.approx(0.5)
    assert m.as_arrays()["connectivity"].shape == (2,)


def test_recorder_connectivity_none_is_nan_and_summary_none():
    m = MetricsRecorder()
    m.record(0.0, _FakeMap(0.1, 0.0), 0.0, connectivity=None)
    assert np.isnan(m.as_arrays()["connectivity"][0])
    assert m.summary()["mean_connectivity"] is None
    assert m.summary()["final_connectivity"] is None



def test_saved_run_json_has_all_metrics_and_reloads(tmp_path):
    ast = Asteroid(radius=R, spin_rate=2e-4, mu=MU, n_cells=400)
    sensor = Sensor(half_angle=0.5, quality_ref_alt=200.0, max_alt=3000.0)
    sm = SurfaceMap(ast.n_cells)
    cfg = {
        "seed": 0,
        "swarm": {"comms_radius": 3000.0},
        "simulation": {"target_coverage": 0.95},
    }
    agents = build_swarm(n=8, mu=MU, radius=R, altitude=400.0,
                         strategy_factory=lambda i: NoOp(), decision_interval=5)
    metrics = MetricsRecorder(target_coverage=0.95)
    sim = Simulation(ast, sensor, sm, agents, dt=10.0, metrics=metrics, config=cfg)
    sim.run(1000.0, record_every=10)

    path = RunLogger("test_run", cfg, out_dir=tmp_path).save(metrics)
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)

    for key in ("final_coverage", "final_quality", "total_delta_v",
                "mean_connectivity", "final_connectivity",
                "time_to_target", "target_coverage"):
        assert key in rec["summary"], f"missing summary metric: {key}"

    for key in ("time", "coverage", "quality", "total_delta_v", "connectivity"):
        assert key in rec["series"], f"missing series: {key}"
        assert len(rec["series"][key]) == len(rec["series"]["time"])

    
    conn = rec["series"]["connectivity"]
    assert all(0.0 <= c <= 1.0 for c in conn)
    assert rec["summary"]["mean_connectivity"] is not None
    assert rec["config"] == cfg and rec["seed"] == 0
