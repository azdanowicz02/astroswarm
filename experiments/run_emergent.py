
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astroswarm.analysis.emergent import run_with_traces
from astroswarm.analysis.evaluate import STRATEGY_FACTORIES
from astroswarm.analysis.plots import (
    plot_coverage_map,
    plot_orbit_bands,
    plot_pheromone_map,
)
from astroswarm.utils.config import load_config

_FIG = Path("results/figures")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emergent-behaviour figures (T5.4).")
    parser.add_argument("--config", nargs="+",
                        default=["configs/default.yaml", "configs/swarm.yaml"])
    parser.add_argument("--strategy", choices=sorted(STRATEGY_FACTORIES), default="pheromone")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-at", type=float, default=None)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--record-every", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(*args.config)
    print(f"Tracing strategy={args.strategy} seed={args.seed} ...")
    tr = run_with_traces(cfg, args.strategy, seed=args.seed, stop_at=args.stop_at,
                         early_stop=not args.no_early_stop, record_every=args.record_every)

    tag = f"{args.strategy}_seed{args.seed}"
    figs = [
        plot_orbit_bands(tr["time"], tr["altitudes"],
                         out_path=_FIG / f"emergent_bands_{tag}.png"),
        plot_coverage_map(tr["asteroid"], tr["surface_map"],
                          out_path=_FIG / f"emergent_coverage_{tag}.png"),
    ]
    if tr["pheromone"] is not None:
        figs.append(plot_pheromone_map(tr["asteroid"], tr["pheromone"],
                                       out_path=_FIG / f"emergent_pheromone_{tag}.png"))

    summ = tr["metrics"].summary()
    print(f"final_coverage={summ['final_coverage']:.3f} "
          f"final_redundancy={summ['final_redundancy']:.2f} "
          f"time_to_target={summ['time_to_target']}")
    print("Saved figures:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
