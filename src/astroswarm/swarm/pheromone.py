
from __future__ import annotations

import numpy as np


class PheromoneMap:
    

    def __init__(self, n_cells: int, deposit: float = 1.0,
                 evaporation: float = 0.02, diffusion: float = 0.0):
        self.n_cells = int(n_cells)
        self.value = np.zeros(self.n_cells)
        self.deposit_amount = float(deposit)
        self.evaporation = float(evaporation)
        self.diffusion = float(diffusion)

    @classmethod
    def from_config(cls, n_cells: int, config: dict | None) -> "PheromoneMap":
        
        p = (config or {}).get("pheromone", {}) or {}
        return cls(
            n_cells,
            deposit=float(p.get("deposit", 1.0)),
            evaporation=float(p.get("evaporation", 0.02)),
            diffusion=float(p.get("diffusion", 0.0)),
        )

    def deposit(self, cell_indices, weight=None) -> None:
        
        idx = np.asarray(cell_indices, dtype=int)
        if idx.size == 0:
            return
        if weight is None:
            amount = self.deposit_amount
        else:
            amount = self.deposit_amount * np.asarray(weight, dtype=float)
        np.add.at(self.value, idx, amount)

    def evaporate(self, steps: int = 1) -> None:
        
        if self.evaporation > 0.0 and steps > 0:
            self.value *= (1.0 - self.evaporation) ** steps

    def diffuse(self, adjacency=None) -> None:
        
        if self.diffusion <= 0.0 or adjacency is None:
            return
        
        

    def step(self, steps: int = 1, adjacency=None) -> None:
        
        self.evaporate(steps)
        self.diffuse(adjacency)

    
    def at(self, cell_indices) -> np.ndarray:
        
        return self.value[np.asarray(cell_indices, dtype=int)]

    def total(self) -> float:
        
        return float(self.value.sum())

    def reset(self) -> None:
        
        self.value[:] = 0.0
