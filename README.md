# AstroSwarm

AstroSwarm is a research framework for studying bio-inspired, decentralized
spacecraft swarms for asteroid surface mapping. It combines orbital dynamics,
local communication, stigmergic coordination, and optional evolutionary
optimization of swarm decision policies.

The project supports single-spacecraft and swarm simulations, reproducible
strategy comparisons, multi-seed evaluation, parameter sweeps, and figures for
coverage, redundancy, orbital behavior, and pheromone fields.

## Features

- Circular two-body orbital dynamics and impulsive maneuvers
- Discretized rotating asteroid surface and quality-aware sensing
- Configurable multi-spacecraft swarms with local communication
- No-op, random-walk, greedy, and pheromone-based strategies
- Coverage, mapping quality, redundancy, connectivity, and delta-v metrics
- Reproducible multi-seed experiments and sensitivity analysis
- Optional evolutionary optimization of decision weights

## Requirements

- Python 3.10 or newer
- NumPy, PyYAML, and Matplotlib
- pytest for development and testing
- NetworkX for optional communication-connectivity metrics
- CMA-ES (`cma`) for optional evolutionary optimization

## Installation

From the repository root, create and activate a virtual environment, then
install the package in editable mode:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux: good luck 

## Quick start

Run a single-spacecraft mapping baseline:

```bash
python experiments/run_single.py
```

Run the default eight-spacecraft swarm with the greedy strategy:

```bash
python experiments/run_comparison.py --strategy greedy
```

Compare all implemented strategies using the same seed:

```bash
python experiments/run_comparison.py --compare
```

The experiment scripts write plots to `results/figures/` and JSON run records
to `results/runs/`. These directories are created automatically.

> **Note:** The default configuration runs to `400000` simulated seconds and
> uses `8000` surface cells. For a quick smoke test, copy the YAML configuration
> and reduce `simulation.t_end` and `asteroid.n_cells`.

## Configuration

The base environment and orbit settings live in `configs/default.yaml`; swarm,
decision, pheromone, and maneuver settings live in `configs/swarm.yaml`.
Experiment scripts deep-merge configuration files from left to right, so later
files override earlier ones:

```bash
python experiments/run_comparison.py \
  --config configs/default.yaml configs/swarm.yaml my_config.yaml \
  --strategy pheromone --seed 42 --n 12
```

Angles are expressed in radians, distances in metres, time in seconds, and
delta-v in metres per second.

Key configuration groups are:

| Group | Purpose |
| --- | --- |
| `asteroid` | Radius, spin rate, gravitational parameter, and surface resolution |
| `sensor` | Field of view, reference altitude, and maximum sensing altitude |
| `simulation` | Time step, end time, and target coverage |
| `orbit` | Initial altitude, inclination, RAAN, and phase |
| `swarm` | Spacecraft count, communication range, limits, and decision cadence |
| `decision_weights` | Coverage, quality, pheromone, cost, and neighbor weights |
| `pheromone` | Deposit, evaporation, and diffusion parameters |
| `maneuvers` | Candidate maneuver sizes and look-ahead settings |

## Experiments

### Strategy comparison

Run selected strategies on identical initial conditions:

```bash
python experiments/run_comparison.py --compare random greedy pheromone --seed 7
```

Use `--no-early-stop` to continue to `simulation.t_end`; otherwise swarm runs
stop when `simulation.target_coverage` is reached.

### Multi-seed evaluation

Aggregate a strategy across several random seeds:

```bash
python experiments/run_multiseed.py --strategy pheromone --seeds 0 1 2 3 4
```

After generating multi-seed records for the desired strategies, create a
baseline Pareto plot with:

```bash
python experiments/plot_baseline.py
```

### Parameter sweeps

Sweep one dotted configuration key:

```bash
python experiments/run_sweep.py \
  --param swarm.n_spacecraft --values 4 8 12 16 \
  --strategy pheromone --seeds 0 1 2
```

Run the built-in sweep set and print a sensitivity ranking:

```bash
python experiments/run_sweep.py --preset standard
```

### Emergent behavior

Generate orbital-band, coverage-map, and (where applicable) pheromone-map
figures:

```bash
python experiments/run_emergent.py --strategy pheromone --seed 0
```

### Evolutionary optimization

Install the `evolution` extra, then optimize the decision weights and benchmark
them on held-out seeds:

```bash
python experiments/run_evolution.py \
  --generations 8 --pop-size 8 \
  --seeds 0 1 2 --eval-seeds 10 11 12
```

## Testing

Run the full test suite from the repository root:

```bash
python -m pytest
```

Run a focused test module with:

```bash
python -m pytest tests/test_decision.py
```

## Repository layout

```text
astroswarm/
|-- configs/                 YAML simulation and swarm configurations
|-- docs/                    Development notes and pseudocode
|-- experiments/             Executable experiment and plotting scripts
|-- src/astroswarm/
|   |-- analysis/            Evaluation and plotting utilities
|   |-- dynamics/            Orbit propagation and maneuvers
|   |-- environment/         Asteroid, sensor, and surface-map models
|   |-- optimization/        Evolutionary weight optimization
|   |-- simulation/          Simulation engine, metrics, and logging
|   |-- strategies/          Swarm decision strategies
|   |-- swarm/               Agents, communication, and pheromones
|   `-- utils/               Configuration, geometry, and seeding helpers
|-- tests/                   pytest test suite
|-- pyproject.toml           Package metadata and dependencies
`-- README.md
```

## Reproducibility

Experiment entry points expose seed arguments and save their merged
configuration with the results. Use the same configuration, strategy, and seed
to reproduce a run. For comparisons, `run_comparison.py` deliberately uses the
same seed and initial swarm construction across strategies.
