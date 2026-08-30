
from __future__ import annotations

import numpy as np

from .base import Strategy
from .candidates import enumerate_candidate_maneuvers, estimate_new_coverage


class NoOp(Strategy):
    

    def select_action(self, agent, context):
        return None


class RandomWalk(Strategy):
    

    def select_action(self, agent, context):
        candidates = enumerate_candidate_maneuvers(agent.orbit, context)
        rng = self.rng if self.rng is not None else np.random.default_rng()
        choice = int(rng.integers(len(candidates)))
        label, new_orbit, dv = candidates[choice]
        if label == "stay":
            return None                      
        return (new_orbit, dv)


class Greedy(Strategy):
    

    def select_action(self, agent, context):
        candidates = enumerate_candidate_maneuvers(agent.orbit, context)
        scores = [estimate_new_coverage(orb, context) for (_label, orb, _dv) in candidates]
        best = int(np.argmax(scores))
        label, new_orbit, dv = candidates[best]
        if label == "stay":
            return None                      
        return (new_orbit, dv)
