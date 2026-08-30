
from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def latlon_to_unit(lat: float, lon: float) -> np.ndarray:
    
    return np.array(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]
    )


def rotate_z(v: np.ndarray, angle: float) -> np.ndarray:
    
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return v @ r.T


def fibonacci_sphere(n: int) -> np.ndarray:
    
    idx = np.arange(n)
    
    z = 1.0 - (2.0 * idx + 1.0) / n
    radius = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    golden = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden * idx
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y, z])
