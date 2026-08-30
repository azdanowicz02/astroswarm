
from __future__ import annotations

import numpy as np


class SurfaceMap:
    

    def __init__(self, n_cells: int, coverage_epsilon: float = 1e-3):
        self.n_cells = int(n_cells)
        self.coverage = np.zeros(self.n_cells)      
        self.quality = np.zeros(self.n_cells)       
        self.observations = np.zeros(self.n_cells, dtype=np.int64)  
        self.coverage_epsilon = float(coverage_epsilon)

    def observe(self, cell_indices, quality_values) -> None:
        
        idx = np.asarray(cell_indices, dtype=int)
        if idx.size == 0:
            return
        q = np.asarray(quality_values, dtype=float)
        if q.ndim == 0:
            q = np.full(idx.shape, float(q))
        self.quality[idx] = np.maximum(self.quality[idx], q)
        self.coverage[idx] = np.minimum(1.0, self.coverage[idx] + q)
        useful = q > 0.0
        if useful.any():
            np.add.at(self.observations, idx[useful], 1)

    
    def coverage_fraction(self) -> float:
        
        return float(self.coverage.mean())

    def mean_quality(self) -> float:
        
        return float(self.quality.mean())

    def mean_reobservations(self) -> float:
        
        seen = self.observations > 0
        if not seen.any():
            return 0.0
        return float(self.observations[seen].mean() - 1.0)

    def total_observations(self) -> int:
        
        return int(self.observations.sum())

    def is_target_reached(self, threshold: float) -> bool:
        
        return self.coverage_fraction() >= threshold

    def unexplored_mask(self) -> np.ndarray:
        
        return self.coverage < self.coverage_epsilon
