
from __future__ import annotations

import numpy as np

from ..utils.geometry import normalize
from .asteroid import Asteroid
from .surface_map import SurfaceMap


class Sensor:
    

    def __init__(self, half_angle: float, quality_ref_alt: float, max_alt: float):
        self.half_angle = float(half_angle)          
        self.quality_ref_alt = float(quality_ref_alt)  
        self.max_alt = float(max_alt)                

    def altitude(self, sc_position: np.ndarray, asteroid: Asteroid) -> float:
        
        return float(np.linalg.norm(sc_position) - asteroid.R)

    def footprint_cells(self, sc_position: np.ndarray, asteroid: Asteroid) -> np.ndarray:
        
        sc = np.asarray(sc_position, dtype=float)
        candidates = asteroid.visible_cells(sc)
        if candidates.size == 0:
            return candidates
        positions = asteroid.cell_positions_inertial()[candidates]  
        directions = positions - sc[None, :]
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        nadir = normalize(-sc)
        cos_angle = directions @ nadir
        inside = cos_angle >= np.cos(self.half_angle)
        return candidates[inside]

    def quality_at(self, sc_position: np.ndarray, asteroid: Asteroid) -> float:
        
        alt = self.altitude(sc_position, asteroid)
        if alt > self.max_alt or alt <= 0.0:
            return 0.0
        return float(np.clip(self.quality_ref_alt / alt, 0.0, 1.0))

    def observe(self, sc_position, asteroid: Asteroid, surface_map: SurfaceMap):
        
        cells = self.footprint_cells(sc_position, asteroid)
        q = self.quality_at(sc_position, asteroid)
        surface_map.observe(cells, q)
        return cells, q
