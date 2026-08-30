
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astroswarm.analysis.evaluate import (
    STRATEGY_FACTORIES,
    aggregate_runs,
    format_aggregate_table,
    run_seeds,
)
from astroswarm.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed aggregation (T5.1).")
    parser.add_argument("--config", nargs="+",
                        default=["configs/default.yaml", "configs/swarm.yaml"],
                        help="one or more YAML files, deep-merged left-to-right")
    parser.add_argument("--strategy", choices=sorted(STRATEGY_FACTORIES), default="greedy")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4],
                        help="seeds to run (default: 0..4)")
    parser.add_argument("--n", type=int, default=None,
                        help="override swarm size (swarm.n_spacecraft)")
    parser.add_argument("--stop-at", type=float, default=None,
                        help="coverage fraction to stop each run early "
                             "(default: simulation.target_coverage)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="run every seed to t_end instead of stopping at target")
    parser.add_argument("--name", default=None, help="run name for the JSON record")
    args = parser.parse_args()

    cfg = load_config(*args.config)
    if args.n is not None:
        cfg.setdefault("swarm", {})["n_spacecraft"] = args.n

    print(f"Running strategy={args.strategy} over seeds {args.seeds} "
          f"({cfg['swarm']['n_spacecraft']} agents)...")
    summaries = run_seeds(cfg, args.strategy, args.seeds,
                          stop_at=args.stop_at, early_stop=not args.no_early_stop)

    
    for s in summaries:
        tt = s.get("time_to_target")
        tt_s = f"{tt:.0f}" if tt is not None else "  n/a "
        print(f"  seed {s['seed']:>2}: t_to_target={tt_s:>8}  "
              f"agg_dv={s['aggregate_dv']:.1f}  "
              f"redund={s['final_redundancy']:.2f}  "
              f"final_cov={s['final_coverage']:.3f}")

    agg = aggregate_runs(summaries)
    print()
    print(format_aggregate_table(agg,
          title=f"{args.strategy}: {len(summaries)} seeds (mean +/- std)"))

    name = args.name or f"multiseed_{args.strategy}"
    out_dir = Path("results/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_name": name,
        "strategy": args.strategy,
        "seeds": list(args.seeds),
        "config": cfg,
        "summaries": summaries,
        "aggregate": agg,
    }
    path = out_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
