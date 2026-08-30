
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astroswarm.analysis.evaluate import sensitivity_ranking, sweep_curve
from astroswarm.analysis.plots import plot_parameter_sweep

STANDARD_PARAMS = [
    "swarm.n_spacecraft",
    "pheromone.evaporation",
    "decision_weights.w_pheromone",
    "sensor.half_angle",
]


def _fig_name(param: str, metric: str) -> Path:
    return Path("results/figures") / f"sweep_{param.replace('.', '_')}_{metric}.png"


def _sweep_run_path(runs_dir: Path, param: str, strategy: str) -> Path:
    return runs_dir / f"sweep_{param.replace('.', '_')}_{strategy}.json"


def _load_sweep(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path} -- run experiments/run_sweep.py for this "
                         f"param/strategy at least once (it now caches raw results "
                         f"there); after that this script can replot from the cache.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _replot(record: dict, metric: str, logx: bool, out: Path | None) -> Path:
    results = record["results"]
    vals, means, stds = sweep_curve(results, metric)
    out_path = out or _fig_name(record["param"], metric)
    fig = plot_parameter_sweep(vals, means, stds, record["param"], metric,
                               out_path=out_path, logx=logx)
    print(f"  {record['param']:<28} -> {fig}")
    for v, m, s in zip(vals, means, stds):
        print(f"      {v!s:>10}: {metric}={m:.4g} ± {s:.4g}")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-plot cached parameter-sweep results (T5.2/T5.3) without "
                    "rerunning the simulation sweep. Reads results/runs/sweep_*.json "
                    "written by experiments/run_sweep.py.")
    parser.add_argument("--strategy", default="pheromone")
    parser.add_argument("--metric", default="time_to_target",
                        help="any metric present in the cached summaries, e.g. "
                             "time_to_target, final_redundancy, aggregate_dv, final_quality")
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--param", default=None,
                        help="dotted config key; omit and pass --preset standard "
                             "to replot all four standard sweeps at once")
    parser.add_argument("--out", default=None)
    parser.add_argument("--logx", action="store_true")
    parser.add_argument("--preset", choices=["standard"], default=None)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    if args.preset == "standard":
        records = []
        for param in STANDARD_PARAMS:
            path = _sweep_run_path(runs_dir, param, args.strategy)
            record = _load_sweep(path)
            print(f"  loaded {path}")
            records.append(record)
        for record in records:
            _replot(record, args.metric, args.logx, None)
        print(f"\nSensitivity ranking for '{args.metric}' (most -> least sensitive):")
        print(f"  {'parameter':<28}{'range':>12}{'rel_range':>12}")
        print("  " + "-" * 50)
        for e in sensitivity_ranking([r["results"] for r in records], args.metric):
            rng = "n/a" if e["range"] is None else f"{e['range']:.4g}"
            rel = "n/a" if e["rel_range"] is None else f"{e['rel_range']:.3f}"
            print(f"  {e['param']:<28}{rng:>12}{rel:>12}")
        return

    if not args.param:
        parser.error("need --param (or use --preset standard)")
    path = _sweep_run_path(runs_dir, args.param, args.strategy)
    record = _load_sweep(path)
    print(f"  loaded {path}")
    out = Path(args.out) if args.out else None
    _replot(record, args.metric, args.logx, out)


if __name__ == "__main__":
    main()
