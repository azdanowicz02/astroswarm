from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astroswarm.analysis.plots import plot_baseline_pareto

DEFAULT_STRATEGIES = ["random", "greedy", "pheromone"]
DEFAULT_LABELS = {"random": "Random walk", "greedy": "Greedy", "pheromone": "Pheromone swarm"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Pareto plot (report Fig. 3.1).")
    parser.add_argument("--strategies", nargs="*", default=DEFAULT_STRATEGIES,
                        help="strategy names, matching results/runs/multiseed_<name>.json")
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--out", default="results/figures/baseline_pareto.png")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    aggregates = {}
    for strategy in args.strategies:
        path = runs_dir / f"multiseed_{strategy}.json"
        if not path.exists():
            parser.error(f"missing {path} -- run experiments/run_multiseed.py "
                        f"--strategy {strategy} --seeds 0 1 2 3 4 first")
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
        label = DEFAULT_LABELS.get(strategy, strategy)
        aggregates[label] = record["aggregate"]
        print(f"  loaded {path} ({len(record['seeds'])} seeds)")

    fig = plot_baseline_pareto(aggregates, out_path=args.out)
    print(f"\nSaved: {fig}")


if __name__ == "__main__":
    main()
