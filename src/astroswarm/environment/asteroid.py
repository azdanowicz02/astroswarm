
from __future__ import annotations

import numpy as np

from ..utils.geometry import fibonacci_sphere, normalize, rotate_z


class Asteroid:
    

    def __init__(self, radius: float, spin_rate: float, mu: float, n_cells: int):
        self.R = float(radius)
        self.omega = float(spin_rate)      
        self.mu = float(mu)                
        self.cells = fibonacci_sphere(n_cells)   
        self.n_cells = self.cells.shape[0]
        self.cell_area = 4.0 * np.pi * self.R ** 2 / self.n_cells
        self.attitude = 0.0                

    
    def update(self, dt: float) -> None:
        
        self.attitude = (self.attitude + self.omega * dt) % (2.0 * np.pi)

    
    def cell_positions_inertial(self) -> np.ndarray:
        
        return rotate_z(self.cells, self.attitude) * self.R

    def cell_position_inertial(self, i: int) -> np.ndarray:
        
        return rotate_z(self.cells[i], self.attitude) * self.R

    def cell_normals_inertial(self) -> np.ndarray:
        
        return rotate_z(self.cells, self.attitude)

    def cell_normal_inertial(self, i: int) -> np.ndarray:
        
        return normalize(self.cell_position_inertial(i))

    
    def visible_cells(self, sc_position: np.ndarray) -> np.ndarray:
        
        sc = np.asarray(sc_position, dtype=float)
        positions = self.cell_positions_inertial()   
        normals = self.cell_normals_inertial()        
        los = sc[None, :] - positions                 
        facing = np.einsum("ij,ij->i", normals, los)  
        return np.nonzero(facing > 0.0)[0]
