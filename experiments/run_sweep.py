
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astroswarm.analysis.evaluate import (
    STRATEGY_FACTORIES,
    parameter_sweep,
    sensitivity_ranking,
    sweep_curve,
    sweep_effect,
)
from astroswarm.analysis.plots import plot_parameter_sweep
from astroswarm.utils.config import load_config


STANDARD_SWEEPS = [
    ("swarm.n_spacecraft", [4, 8, 12, 16], False),
    ("pheromone.evaporation", [0.01, 0.05, 0.1], False),
    ("decision_weights.w_pheromone", [0.0, 0.8, 1.6], False),
    ("sensor.half_angle", [0.15, 0.20, 0.30], False),
]


def _fig_name(param: str, metric: str) -> Path:
    return Path("results/figures") / f"sweep_{param.replace('.', '_')}_{metric}.png"


def _sweep_run_path(param: str, strategy: str) -> Path:
    return Path("results/runs") / f"sweep_{param.replace('.', '_')}_{strategy}.json"


def _save_sweep(param, strategy, seeds, stop_at, early_stop, results) -> Path:
    out_dir = Path("results/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "param": param,
        "strategy": strategy,
        "seeds": list(seeds),
        "stop_at": stop_at,
        "early_stop": early_stop,
        "results": results,
    }
    path = _sweep_run_path(param, strategy)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return path


def _run_one_sweep(cfg, param, values, strategy, seeds, metric, stop_at, early_stop, logx):
    results = parameter_sweep(cfg, param, values, strategy, seeds=seeds,
                              stop_at=stop_at, early_stop=early_stop)
    saved = _save_sweep(param, strategy, seeds, stop_at, early_stop, results)
    vals, means, stds = sweep_curve(results, metric)
    fig = plot_parameter_sweep(vals, means, stds, param, metric,
                               out_path=_fig_name(param, metric), logx=logx)
    print(f"  {param:<28} -> {fig}  (cached: {saved})")
    for v, m, s in zip(vals, means, stds):
        print(f"      {v!s:>10}: {metric}={m:.4g} ± {s:.4g}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sweeps (T5.2/T5.3).")
    parser.add_argument("--config", nargs="+",
                        default=["configs/default.yaml", "configs/swarm.yaml"])
    parser.add_argument("--strategy", choices=sorted(STRATEGY_FACTORIES), default="pheromone")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--metric", default="time_to_target",
                        help="summary metric to plot / rank (e.g. time_to_target, "
                             "final_redundancy, aggregate_dv, final_quality)")
    parser.add_argument("--stop-at", type=float, default=None)
    parser.add_argument("--no-early-stop", action="store_true")

    parser.add_argument("--param", default=None, help="dotted config key to sweep")
    parser.add_argument("--values", nargs="*", type=float, default=None)
    parser.add_argument("--logx", action="store_true")

    parser.add_argument("--preset", choices=["standard"], default=None)
    args = parser.parse_args()

    cfg = load_config(*args.config)
    early = not args.no_early_stop

    if args.preset == "standard":
        print(f"Standard sweep set | strategy={args.strategy} | metric={args.metric} "
              f"| seeds={args.seeds}")
        sweeps = []
        for param, values, logx in STANDARD_SWEEPS:
            results = _run_one_sweep(cfg, param, values, args.strategy, args.seeds,
                                     args.metric, args.stop_at, early, logx)
            sweeps.append(results)
        print(f"\nSensitivity ranking for '{args.metric}' (most -> least sensitive):")
        print(f"  {'parameter':<28}{'range':>12}{'rel_range':>12}")
        print("  " + "-" * 50)
        for e in sensitivity_ranking(sweeps, args.metric):
            rng = "n/a" if e["range"] is None else f"{e['range']:.4g}"
            rel = "n/a" if e["rel_range"] is None else f"{e['rel_range']:.3f}"
            print(f"  {e['param']:<28}{rng:>12}{rel:>12}")
        return

    if not args.param or not args.values:
        parser.error("single-sweep mode needs --param and --values (or use --preset)")
    print(f"Sweep {args.param} over {args.values} | strategy={args.strategy} "
          f"| metric={args.metric} | seeds={args.seeds}")
    results = _run_one_sweep(cfg, args.param, args.values, args.strategy, args.seeds,
                             args.metric, args.stop_at, early, args.logx)
    print(f"\nEffect: {sweep_effect(results, args.metric)}")


if __name__ == "__main__":
    main()
