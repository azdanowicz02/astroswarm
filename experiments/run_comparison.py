
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astroswarm.analysis.plots import (
    plot_coverage_comparison,
    plot_coverage_curve,
    plot_redundancy_comparison,
)
from astroswarm.environment import Asteroid, Sensor, SurfaceMap
from astroswarm.simulation import MetricsRecorder, Simulation
from astroswarm.simulation.logger import RunLogger
from astroswarm.strategies import Greedy, NoOp, PheromoneSwarm, RandomWalk
from astroswarm.swarm import build_swarm_from_config
from astroswarm.utils.config import load_config
from astroswarm.utils.seeding import make_rng

STRATEGIES = {
    "noop": lambda rng: (lambda i: NoOp()),
    "greedy": lambda rng: (lambda i: Greedy(rng=rng)),
    "random": lambda rng: (lambda i: RandomWalk(rng=make_rng(int(rng.integers(2**32))))),
    "pheromone": lambda rng: PheromoneSwarm.shared_factory(rng=rng),
}


DEFAULT_COMPARE = ["noop", "random", "greedy", "pheromone"]


def run_swarm(cfg: dict, strategy_name: str, seed: int, stop_at: float | None):
    """Build the environment + swarm, run (stopping at ``stop_at``), return the pieces."""
    rng = make_rng(seed)

    asteroid = Asteroid(**cfg["asteroid"])
    sensor = Sensor(**cfg["sensor"])
    surface_map = SurfaceMap(asteroid.n_cells)

    strategy_factory = STRATEGIES[strategy_name](rng)
    init_rng = make_rng(1_000_003 + int(seed))
    agents = build_swarm_from_config(cfg, strategy_factory=strategy_factory, rng=init_rng)

    sim_cfg = cfg["simulation"]
    metrics = MetricsRecorder(target_coverage=sim_cfg["target_coverage"])
    sim = Simulation(asteroid, sensor, surface_map, agents, dt=sim_cfg["dt"],
                     metrics=metrics, config=cfg)
    sim.run(sim_cfg["t_end"], record_every=10, stop_at_target=stop_at)
    return agents, metrics, asteroid, surface_map


def _fmt(x, spec):
    return "  n/a  " if x is None else format(x, spec)


def _print_single(agents, metrics, strategy, seed, end_t, stopped_early):
    total_dv = sum(a.delta_v for a in agents)
    print(f"Swarm baseline: {len(agents)} agents | strategy={strategy} | seed={seed}")
    print(f"Stopped at t={end_t:.0f}s "
          f"({'target coverage reached' if stopped_early else 'reached t_end'})")
    print("Per-agent delta-v [m/s]:")
    for a in agents:
        print(f"  agent {a.agent_id:>2}: {a.delta_v:12.6f}")
    print(f"Aggregate swarm delta-v: {total_dv:.6f} m/s")
    print("Run summary:")
    for k, v in metrics.summary().items():
        print(f"  {k:>18}: {v}")


def _print_comparison_table(rows):
    print("\nComparison (same seed):")
    hdr = f"  {'strategy':<8} {'t_to_95% [s]':>14} {'agg dv [m/s]':>14} " \
          f"{'redund':>9} {'final_qual':>11} {'mean_conn':>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['strategy']:<8} {_fmt(r['time_to_target'], '14.0f')} "
              f"{r['aggregate_dv']:14.1f} {r['final_redundancy']:9.2f} "
              f"{r['final_quality']:11.3f} "
              f"{_fmt(r['mean_connectivity'], '11.3f')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Swarm baseline / comparison runner.")
    parser.add_argument("--config", nargs="+",
                        default=["configs/default.yaml", "configs/swarm.yaml"],
                        help="one or more YAML files, deep-merged left-to-right")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="greedy")
    parser.add_argument("--compare", nargs="*", metavar="STRATEGY", default=None,
                        help="run several strategies on the same seed and overlay their "
                             f"coverage curves (default: {' '.join(DEFAULT_COMPARE)})")
    parser.add_argument("--n", type=int, default=None,
                        help="override swarm size (configs/swarm.yaml: n_spacecraft)")
    parser.add_argument("--seed", type=int, default=None, help="override master seed")
    parser.add_argument("--stop-at", type=float, default=None,
                        help="coverage fraction at which to stop early "
                             "(default: simulation.target_coverage)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="run the full t_end even after target coverage is reached")
    parser.add_argument("--name", default=None, help="run name for the JSON record")
    args = parser.parse_args()

    cfg = load_config(*args.config)
    if args.n is not None:
        cfg.setdefault("swarm", {})["n_spacecraft"] = args.n
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed

    if args.no_early_stop:
        stop_at = None
    elif args.stop_at is not None:
        stop_at = args.stop_at
    else:
        stop_at = cfg["simulation"]["target_coverage"]
    target = cfg["simulation"]["target_coverage"]

    if args.compare is not None:
        strategies = args.compare or DEFAULT_COMPARE
        unknown = [s for s in strategies if s not in STRATEGIES]
        if unknown:
            parser.error(f"unknown strategy/strategies: {', '.join(unknown)}")

        runs = {}
        rows = []
        print(f"Comparing {strategies} on seed {seed} "
              f"({cfg['swarm']['n_spacecraft']} agents)...")
        for st in strategies:
            agents, metrics, _ast, _sm = run_swarm(cfg, st, seed, stop_at)
            runs[st] = metrics
            summ = metrics.summary()
            agg_dv = sum(a.delta_v for a in agents)
            rows.append({"strategy": st, "aggregate_dv": agg_dv, **summ})
            RunLogger(f"swarm_{st}", cfg).save(
                metrics,
                extra={"strategy": st, "n_agents": len(agents), "aggregate_delta_v": agg_dv},
            )
            print(f"  {st:<8} done: t_to_target={summ['time_to_target']}, "
                  f"agg_dv={agg_dv:.1f}")

        _print_comparison_table(rows)
        fig = plot_coverage_comparison(
            runs, out_path=Path("results/figures") / "comparison_coverage.png",
            target=target, title="Uncoordinated baselines (same seed)")
        fig_r = plot_redundancy_comparison(
            rows, out_path=Path("results/figures") / "redundancy_comparison.png")
        print(f"\nSaved figures: {fig}\n               {fig_r}")
        return

    # ---- single-strategy mode -------------------------------------------
    name = args.name or f"swarm_{args.strategy}"
    agents, metrics, _asteroid, surface_map = run_swarm(cfg, args.strategy, seed, stop_at)
    end_t = metrics.time[-1] if metrics.time else 0.0
    stopped_early = stop_at is not None and surface_map.coverage_fraction() >= stop_at
    _print_single(agents, metrics, args.strategy, seed, end_t, stopped_early)

    total_dv = sum(a.delta_v for a in agents)
    curve = plot_coverage_curve(
        metrics, out_path=Path("results/figures") / f"{name}_coverage.png")
    record = RunLogger(name, cfg).save(
        metrics,
        extra={
            "strategy": args.strategy,
            "n_agents": len(agents),
            "aggregate_delta_v": total_dv,
            "per_agent_delta_v": {a.agent_id: a.delta_v for a in agents},
            "stopped_early": stopped_early,
        },
    )
    print(f"\nSaved: {curve}\n       {record}")


if __name__ == "__main__":
    main()
