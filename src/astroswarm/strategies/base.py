
from __future__ import annotations

from abc import ABC, abstractmethod


class Strategy(ABC):
    

    def __init__(self, rng=None, **params):
        self.rng = rng
        self.params = params

    @abstractmethod
    def select_action(self, agent, context):
        
        raise NotImplementedError

    def reset(self) -> None:
        
        pass
