
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MetricsRecorder:
    

    target_coverage: float = 0.95

    time: list[float] = field(default_factory=list)
    coverage: list[float] = field(default_factory=list)
    quality: list[float] = field(default_factory=list)
    total_delta_v: list[float] = field(default_factory=list)
    connectivity: list[float] = field(default_factory=list)
    redundancy: list[float] = field(default_factory=list)

    _time_to_target: float | None = None

    def record(self, t: float, surface_map, delta_v: float = 0.0,
               connectivity: float | None = None) -> None:
        
        cov = surface_map.coverage_fraction()
        self.time.append(float(t))
        self.coverage.append(cov)
        self.quality.append(surface_map.mean_quality())
        self.total_delta_v.append(float(delta_v))
        self.connectivity.append(float(connectivity) if connectivity is not None
                                 else float("nan"))
        self.redundancy.append(surface_map.mean_reobservations())
        if self._time_to_target is None and cov >= self.target_coverage:
            self._time_to_target = float(t)

    def _mean_connectivity(self) -> float | None:
        conn = np.asarray(self.connectivity, dtype=float)
        if conn.size == 0 or np.all(np.isnan(conn)):
            return None
        return float(np.nanmean(conn))

    def summary(self) -> dict:
        
        conn = np.asarray(self.connectivity, dtype=float)
        final_conn = (float(conn[-1]) if conn.size and not np.isnan(conn[-1])
                      else None)
        return {
            "final_coverage": self.coverage[-1] if self.coverage else 0.0,
            "final_quality": self.quality[-1] if self.quality else 0.0,
            "total_delta_v": self.total_delta_v[-1] if self.total_delta_v else 0.0,
            "mean_connectivity": self._mean_connectivity(),
            "final_connectivity": final_conn,
            "final_redundancy": self.redundancy[-1] if self.redundancy else 0.0,
            "time_to_target": self._time_to_target,
            "target_coverage": self.target_coverage,
        }

    def as_arrays(self) -> dict[str, np.ndarray]:
        
        return {
            "time": np.asarray(self.time),
            "coverage": np.asarray(self.coverage),
            "quality": np.asarray(self.quality),
            "total_delta_v": np.asarray(self.total_delta_v),
            "connectivity": np.asarray(self.connectivity),
            "redundancy": np.asarray(self.redundancy),
        }
