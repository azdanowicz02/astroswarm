
from __future__ import annotations

import copy

import numpy as np

from ..analysis.evaluate import aggregate_runs, run_seeds


WEIGHT_KEYS = ["w_coverage", "w_quality", "w_pheromone", "w_cost", "w_neighbor"]



_FITNESS_DEFAULTS = {
    "k_time": 1.0,      
    "k_dv": 0.5,        
    "dv_ref": 10000.0,  
}


def encode(decision_weights: dict) -> np.ndarray:
    
    return np.array([decision_weights[k] for k in WEIGHT_KEYS], dtype=float)


def decode(genome) -> dict:
    
    return {k: float(v) for k, v in zip(WEIGHT_KEYS, genome)}


def _fitness_params(base_config: dict) -> dict:
    p = dict(_FITNESS_DEFAULTS)
    p.update((base_config or {}).get("evolution", {}).get("fitness", {}) or {})
    return p


def _config_with_weights(base_config: dict, genome) -> dict:
    
    cfg = copy.deepcopy(base_config)
    cfg["decision_weights"] = decode(genome)
    return cfg


def fitness(genome, base_config: dict, seeds=(0, 1, 2), strategy: str = "pheromone",
            stop_at: float | None = None) -> float:
    
    p = _fitness_params(base_config)
    t_end = float(base_config["simulation"]["t_end"])
    cfg = _config_with_weights(base_config, genome)
    summaries = run_seeds(cfg, strategy, seeds, stop_at=stop_at, early_stop=True)

    scores = []
    for s in summaries:
        cov = float(s.get("final_coverage") or 0.0)
        tt = s.get("time_to_target")
        tt = float(tt) if tt is not None else t_end          
        dv = float(s.get("aggregate_dv") or 0.0)
        scores.append(cov
                      - p["k_time"] * (tt / t_end)
                      - p["k_dv"] * (dv / p["dv_ref"]))
    return float(np.mean(scores)) if scores else float("-inf")


def default_genome(base_config: dict) -> np.ndarray:
    
    dw = (base_config or {}).get("decision_weights", {})
    return np.array([float(dw.get(k, 0.0)) for k in WEIGHT_KEYS], dtype=float)


def evolve(base_config: dict, seeds=(0, 1, 2), generations: int = 10,
           pop_size: int = 8, sigma: float = 0.3, elite: int = 2, seed: int = 0,
           strategy: str = "pheromone", stop_at: float | None = None,
           bounds: tuple = (0.0, 3.0), init_weights=None, sigma_decay: float = 0.9,
           verbose: bool = False) -> dict:
    
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    x0 = np.asarray(init_weights, dtype=float) if init_weights is not None \
        else default_genome(base_config)
    x0 = np.clip(x0, lo, hi)
    dim = x0.size

    def evaluate(genome):
        return fitness(genome, base_config, seeds, strategy=strategy, stop_at=stop_at)

    
    population = [x0.copy()]
    while len(population) < pop_size:
        population.append(np.clip(x0 + rng.normal(0.0, sigma, dim), lo, hi))
    scored = [(g, evaluate(g)) for g in population]
    n_eval = len(scored)

    history = []
    best_genome, best_fit = max(scored, key=lambda t: t[1])
    history.append(best_fit)
    if verbose:
        print(f"gen 0: best fitness {best_fit:.4f}  weights {np.round(best_genome,3)}")

    s = sigma
    for gen in range(1, generations + 1):
        scored.sort(key=lambda t: t[1], reverse=True)
        elites = [g for g, _f in scored[:elite]]
        
        offspring = []
        while len(offspring) < pop_size - elite:
            parent = elites[int(rng.integers(len(elites)))]
            child = np.clip(parent + rng.normal(0.0, s, dim), lo, hi)
            offspring.append(child)
        new_scored = [(g, scored[i][1]) for i, g in enumerate(elites)]  
        for g in offspring:
            new_scored.append((g, evaluate(g)))
            n_eval += 1
        scored = new_scored
        gen_best_g, gen_best_f = max(scored, key=lambda t: t[1])
        if gen_best_f > best_fit:
            best_fit, best_genome = gen_best_f, gen_best_g.copy()
        history.append(best_fit)
        s *= sigma_decay
        if verbose:
            print(f"gen {gen}: best fitness {best_fit:.4f}  "
                  f"weights {np.round(best_genome,3)}  sigma {s:.3f}")

    return {
        "best_genome": best_genome,
        "best_weights": decode(best_genome),
        "best_fitness": float(best_fit),
        "history": [float(h) for h in history],
        "n_evaluations": int(n_eval),
        "seeds": list(seeds),
        "generations": generations,
        "pop_size": pop_size,
        "init_weights": decode(x0),
    }


def benchmark(base_config: dict, evolved_weights: dict, manual_weights: dict | None = None,
              seeds=(10, 11, 12), strategy: str = "pheromone",
              stop_at: float | None = None) -> dict:
    
    if manual_weights is None:
        manual_weights = dict(base_config.get("decision_weights", {}))

    def evaluate(weights):
        genome = encode({k: weights[k] for k in WEIGHT_KEYS})
        cfg = _config_with_weights(base_config, genome)
        summaries = run_seeds(cfg, strategy, seeds, stop_at=stop_at, early_stop=True)
        return {
            "weights": decode(genome),
            "aggregate": aggregate_runs(summaries),
            "fitness": fitness(genome, base_config, seeds, strategy=strategy, stop_at=stop_at),
        }

    return {
        "seeds": list(seeds),
        "manual": evaluate(manual_weights),
        "evolved": evaluate(evolved_weights),
    }
