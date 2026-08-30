
from __future__ import annotations

import copy
import math
from typing import Callable, Iterable

import numpy as np

from ..environment import Asteroid, Sensor, SurfaceMap
from ..simulation import MetricsRecorder, Simulation
from ..strategies import Greedy, NoOp, PheromoneSwarm, RandomWalk
from ..swarm import build_swarm_from_config
from ..utils.seeding import make_rng



STRATEGY_FACTORIES: dict[str, Callable] = {
    "noop": lambda rng: (lambda i: NoOp()),
    "greedy": lambda rng: (lambda i: Greedy(rng=rng)),
    
    "random": lambda rng: (lambda i: RandomWalk(rng=make_rng(int(rng.integers(2**32))))),
    "pheromone": lambda rng: PheromoneSwarm.shared_factory(rng=rng),
}


METRIC_ORDER = [
    "time_to_target", "final_coverage", "final_quality",
    "aggregate_dv", "final_redundancy", "mean_connectivity",
]


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)) \
        and not isinstance(x, bool)


def aggregate_runs(summaries: list[dict]) -> dict:
    
    keys: list[str] = []
    for s in summaries:
        for k in s:
            if k not in keys:
                keys.append(k)

    out: dict[str, dict] = {}
    for k in keys:
        vals = [s[k] for s in summaries if k in s and _is_number(s[k])]
        if not vals:
            out[k] = {"mean": None, "std": None, "n": 0, "min": None,
                      "max": None, "values": []}
            continue
        arr = np.asarray(vals, dtype=float)
        out[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "n": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "values": [float(v) for v in vals],
        }
    return out


def _run_single(cfg: dict, strategy_name: str, seed: int, stop_at: float | None):
    
    if strategy_name not in STRATEGY_FACTORIES:
        raise KeyError(f"unknown strategy {strategy_name!r}; "
                       f"choose from {sorted(STRATEGY_FACTORIES)}")
    rng = make_rng(seed)
    asteroid = Asteroid(**cfg["asteroid"])
    sensor = Sensor(**cfg["sensor"])
    surface_map = SurfaceMap(asteroid.n_cells)

    factory = STRATEGY_FACTORIES[strategy_name](rng)
    
    
    init_rng = make_rng(1_000_003 + int(seed))
    agents = build_swarm_from_config(cfg, strategy_factory=factory, rng=init_rng)

    sim_cfg = cfg["simulation"]
    metrics = MetricsRecorder(target_coverage=sim_cfg["target_coverage"])
    sim = Simulation(asteroid, sensor, surface_map, agents, dt=sim_cfg["dt"],
                     metrics=metrics, config=cfg)
    sim.run(sim_cfg["t_end"], record_every=10, stop_at_target=stop_at)
    return agents, metrics, surface_map


def run_seeds(cfg: dict, strategy_name: str, seeds: Iterable[int],
              stop_at: float | None = None, early_stop: bool = True) -> list[dict]:
    
    summaries: list[dict] = []
    for seed in seeds:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed["seed"] = int(seed)
        if not early_stop:
            stop = None
        elif stop_at is not None:
            stop = stop_at
        else:
            stop = cfg_seed["simulation"]["target_coverage"]
        agents, metrics, _sm = _run_single(cfg_seed, strategy_name, int(seed), stop)
        summ = dict(metrics.summary())
        summ["seed"] = int(seed)
        summ["aggregate_dv"] = float(sum(a.delta_v for a in agents))
        summaries.append(summ)
    return summaries


def format_aggregate_table(agg: dict, order: list[str] | None = None,
                           title: str = "Multi-seed aggregate (mean +/- std)") -> str:
    
    
    
    cols = [c for c in (order if order is not None else METRIC_ORDER) if c in agg]
    lines = [title, f"  {'metric':<20}{'mean':>16}{'std':>16}{'n':>5}"]
    lines.append("  " + "-" * 57)
    for k in cols:
        st = agg.get(k)
        if not st or st["mean"] is None:
            continue
        lines.append(f"  {k:<20}{st['mean']:>16.4g}{st['std']:>16.4g}{st['n']:>5d}")
    return "\n".join(lines)


def set_by_path(cfg: dict, param_path: str, value) -> dict:
    
    out = copy.deepcopy(cfg)
    keys = param_path.split(".")
    node = out
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            node[k] = nxt
        node = nxt
    node[keys[-1]] = value
    return out


def parameter_sweep(base_config: dict, param_path: str, values: list,
                    strategy: str, seeds: Iterable[int] = (0, 1, 2),
                    stop_at: float | None = None, early_stop: bool = True) -> list[dict]:
    
    seeds = list(seeds)
    results = []
    for v in values:
        cfg_v = set_by_path(base_config, param_path, v)
        summaries = run_seeds(cfg_v, strategy, seeds,
                              stop_at=stop_at, early_stop=early_stop)
        results.append({
            "param": param_path,
            "value": v,
            "summaries": summaries,
            "aggregate": aggregate_runs(summaries),
        })
    return results


def sweep_curve(sweep_results: list[dict], metric: str):
    
    values, means, stds = [], [], []
    for r in sweep_results:
        st = r["aggregate"].get(metric, {})
        values.append(r["value"])
        means.append(st.get("mean") if st.get("mean") is not None else float("nan"))
        stds.append(st.get("std") if st.get("std") is not None else float("nan"))
    return values, means, stds


def sweep_effect(sweep_results: list[dict], metric: str) -> dict:
    
    _values, means, _stds = sweep_curve(sweep_results, metric)
    arr = np.asarray([m for m in means if not math.isnan(m)], dtype=float)
    if arr.size == 0:
        return {"param": sweep_results[0]["param"] if sweep_results else None,
                "metric": metric, "min": None, "max": None, "range": None,
                "rel_range": None, "n_values": 0}
    lo, hi = float(arr.min()), float(arr.max())
    centre = float(np.abs(arr).mean())
    return {
        "param": sweep_results[0]["param"],
        "metric": metric,
        "min": lo, "max": hi, "range": hi - lo,
        "rel_range": (hi - lo) / centre if centre > 0 else float("inf"),
        "n_values": int(arr.size),
    }


def sensitivity_ranking(sweeps: list[list[dict]], metric: str) -> list[dict]:
    
    effects = [sweep_effect(s, metric) for s in sweeps if s]
    return sorted(
        effects,
        key=lambda e: (e["rel_range"] if e["rel_range"] is not None else -1.0),
        reverse=True,
    )
