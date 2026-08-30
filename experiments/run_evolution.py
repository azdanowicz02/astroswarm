
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astroswarm.analysis.plots import plot_evolution_history, plot_evolved_vs_manual
from astroswarm.optimization.evolution import WEIGHT_KEYS, benchmark, evolve
from astroswarm.utils.config import load_config


def _fmt_weights(w):
    return "{" + ", ".join(f"{k.replace('w_', '')}={w[k]:.3f}" for k in WEIGHT_KEYS) + "}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve decision weights (Phase 6).")
    parser.add_argument("--config", nargs="+",
                        default=["configs/default.yaml", "configs/swarm.yaml"])
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                        help="training seeds the fitness averages over")
    parser.add_argument("--eval-seeds", nargs="*", type=int, default=[10, 11, 12],
                        help="held-out seeds for the final benchmark")
    parser.add_argument("--evo-seed", type=int, default=0, help="RNG seed for evolution")
    parser.add_argument("--stop-at", type=float, default=None)
    parser.add_argument("--name", default="evolution")
    args = parser.parse_args()

    cfg = load_config(*args.config)

    print(f"Evolving weights | gens={args.generations} pop={args.pop_size} "
          f"train_seeds={args.seeds} ...")
    res = evolve(cfg, seeds=args.seeds, generations=args.generations,
                 pop_size=args.pop_size, sigma=args.sigma, elite=args.elite,
                 seed=args.evo_seed, stop_at=args.stop_at, verbose=True)
    print(f"\nBest fitness {res['best_fitness']:.4f} after {res['n_evaluations']} evals")
    print(f"  hand-tuned start: {_fmt_weights(res['init_weights'])}")
    print(f"  evolved weights : {_fmt_weights(res['best_weights'])}")

    fig_hist = plot_evolution_history(res["history"],
                                      out_path=Path("results/figures") / f"{args.name}_history.png")

    print(f"\nBenchmarking on held-out seeds {args.eval_seeds} ...")
    bench = benchmark(cfg, res["best_weights"], seeds=args.eval_seeds, stop_at=args.stop_at)
    man, evo = bench["manual"], bench["evolved"]
    print(f"  {'':<10}{'fitness':>10}{'t_to_target':>14}{'agg_dv':>12}{'redund':>10}")
    for label, r in (("manual", man), ("evolved", evo)):
        agg = r["aggregate"]
        tt = agg.get("time_to_target", {}).get("mean")
        tt_s = f"{tt:.0f}" if tt is not None else "n/a"
        print(f"  {label:<10}{r['fitness']:>10.4f}{tt_s:>14}"
              f"{agg.get('aggregate_dv', {}).get('mean', 0.0):>12.1f}"
              f"{agg.get('final_redundancy', {}).get('mean', 0.0):>10.2f}")

    fig_cmp = plot_evolved_vs_manual(bench,
                                     out_path=Path("results/figures") / f"{args.name}_vs_manual.png")

    out = Path("results/runs") / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"evolution": res, "benchmark": bench, "config": cfg}, fh, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nSaved figures: {fig_hist}\n               {fig_cmp}\nSaved record: {out}")


if __name__ == "__main__":
    main()
