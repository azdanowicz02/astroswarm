"""Quick illustrative manual-vs-evolved comparison (deadline version).

NOT a statistically rigorous benchmark: truncated coverage target instead
of the full 0.95 report target, because the full evolution run (T6.3)
does not finish in time. The "evolved" weights below are copied straight
from the evolution console log, where the best genome was unchanged
across generations 4, 5 and 6 (only sigma kept shrinking) -- treated
here as converged.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astroswarm.analysis.evaluate import aggregate_runs, run_seeds
from astroswarm.analysis.plots import plot_evolved_vs_manual
from astroswarm.optimization.evolution import WEIGHT_KEYS, _fitness_params
from astroswarm.utils.config import load_config

EVOLVED_WEIGHTS = {
    "w_coverage": 0.0,
    "w_quality": 0.273,
    "w_pheromone": 0.749,
    "w_cost": 1.432,
    "w_neighbor": 0.852,
}

SEEDS = [0] 
STOP_AT = 0.6     
NAME = "evolution_quicklook"


def _evaluate(cfg: dict, weights: dict, seeds, stop_at) -> dict:
    """One simulation pass per seed (not two -- evolution.benchmark() re-simulates
    a second time for its own fitness() call, which we can't afford here)."""
    cfg_w = copy.deepcopy(cfg)
    cfg_w["decision_weights"] = dict(weights)
    summaries = run_seeds(cfg_w, "pheromone", seeds, stop_at=stop_at, early_stop=True)

    p = _fitness_params(cfg)
    t_end = float(cfg["simulation"]["t_end"])
    scores = []
    for s in summaries:
        cov = float(s.get("final_coverage") or 0.0)
        tt = s.get("time_to_target")
        tt = float(tt) if tt is not None else t_end
        dv = float(s.get("aggregate_dv") or 0.0)
        scores.append(cov - p["k_time"] * (tt / t_end) - p["k_dv"] * (dv / p["dv_ref"]))

    return {
        "weights": dict(weights),
        "aggregate": aggregate_runs(summaries),
        "fitness": float(sum(scores) / len(scores)) if scores else float("-inf"),
    }


def main() -> None:
    cfg = load_config("configs/default.yaml", "configs/swarm.yaml")
    manual_weights = dict(cfg["decision_weights"])

    print("Manual weights :", {k: manual_weights[k] for k in WEIGHT_KEYS})
    print("Evolved weights:", EVOLVED_WEIGHTS)
    print(f"Benchmarking on seeds {SEEDS}, stop_at={STOP_AT} (quicklook, not held-out eval seeds)...")

    bench = {
        "seeds": list(SEEDS),
        "manual": _evaluate(cfg, manual_weights, SEEDS, STOP_AT),
        "evolved": _evaluate(cfg, EVOLVED_WEIGHTS, SEEDS, STOP_AT),
    }

    man, evo = bench["manual"], bench["evolved"]
    print(f"  {'':<10}{'fitness':>10}{'t_to_target':>14}{'agg_dv':>12}{'redund':>10}")
    for label, r in (("manual", man), ("evolved", evo)):
        agg = r["aggregate"]
        tt = agg.get("time_to_target", {}).get("mean")
        tt_s = f"{tt:.0f}" if tt is not None else "n/a"
        print(f"  {label:<10}{r['fitness']:>10.4f}{tt_s:>14}"
              f"{agg.get('aggregate_dv', {}).get('mean', 0.0):>12.1f}"
              f"{agg.get('final_redundancy', {}).get('mean', 0.0):>10.2f}")

    fig = plot_evolved_vs_manual(bench, out_path=Path("results/figures") / f"{NAME}_vs_manual.png")

    out = Path("results/runs") / f"{NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"benchmark": bench, "config": cfg,
                   "note": (f"quicklook: seeds {SEEDS}, stop_at={STOP_AT}, evolved weights taken "
                            "from converged generation (unchanged across gens 4-6) of the "
                            "in-progress full evolution run -- not a held-out, separately-tuned "
                            "eval-seed verdict, treat as illustrative")},
                  fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nSaved figure: {fig}")
    print(f"Saved record: {out}")


if __name__ == "__main__":
    main()
