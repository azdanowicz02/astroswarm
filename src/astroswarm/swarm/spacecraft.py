
from __future__ import annotations

import numpy as np


class Spacecraft:
    

    def __init__(self, orbit, strategy=None, agent_id: int = 0, decision_interval: int = 1):
        self.orbit = orbit
        self.strategy = strategy
        self.agent_id = int(agent_id)
        self.decision_interval = max(1, int(decision_interval))
        self._steps_since_decision = 0

    
    @property
    def position(self) -> np.ndarray:
        
        return self.orbit.position

    @property
    def velocity(self) -> np.ndarray:
        
        return self.orbit.velocity

    @property
    def delta_v(self) -> float:
        
        return float(getattr(self.orbit, "delta_v", 0.0))

    def propagate(self, dt: float) -> None:
        
        self.orbit.propagate(dt)
        self._steps_since_decision += 1

    
    def due_for_decision(self) -> bool:
        
        return self.strategy is not None and \
            self._steps_since_decision >= self.decision_interval

    def decide(self, context) -> None:
        
        if not self.due_for_decision():
            return
        self._steps_since_decision = 0
        action = self.strategy.select_action(self, context)
        self.apply_action(action)

    def apply_action(self, action) -> None:
        
        if action is None:
            return
        new_orbit, _dv = action
        self.orbit = new_orbit
