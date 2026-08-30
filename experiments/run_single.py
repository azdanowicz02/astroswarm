
from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astroswarm.analysis.plots import plot_coverage_curve, plot_coverage_map
from astroswarm.dynamics import CircularOrbit
from astroswarm.environment import Asteroid, Sensor, SurfaceMap
from astroswarm.simulation import MetricsRecorder, Simulation
from astroswarm.simulation.logger import RunLogger
from astroswarm.utils.config import load_config
from astroswarm.utils.seeding import make_rng


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-agent mapping baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--name", default="single_baseline")
    args = parser.parse_args()

    cfg = load_config(args.config)
    make_rng(cfg.get("seed", 0))  

    ast_cfg, sen_cfg = cfg["asteroid"], cfg["sensor"]
    sim_cfg, orb_cfg = cfg["simulation"], cfg["orbit"]

    asteroid = Asteroid(**ast_cfg)
    sensor = Sensor(**sen_cfg)
    surface_map = SurfaceMap(asteroid.n_cells)

    agent = CircularOrbit(
        mu=asteroid.mu,
        radius=asteroid.R,
        altitude=orb_cfg["altitude"],
        inclination=orb_cfg["inclination"],
        raan=orb_cfg["raan"],
        phase0=orb_cfg["phase0"],
    )

    metrics = MetricsRecorder(target_coverage=sim_cfg["target_coverage"])
    sim = Simulation(asteroid, sensor, surface_map, [agent], dt=sim_cfg["dt"],
                     metrics=metrics)
    sim.run(sim_cfg["t_end"], record_every=10)

    summary = metrics.summary()
    print("Run summary:")
    for k, v in summary.items():
        print(f"  {k:>16}: {v}")

    curve = plot_coverage_curve(metrics)
    cmap = plot_coverage_map(asteroid, surface_map)
    record = RunLogger(args.name, cfg).save(metrics)
    print(f"\nSaved: {curve}\n       {cmap}\n       {record}")


if __name__ == "__main__":
    main()
