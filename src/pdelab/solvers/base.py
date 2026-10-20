import numpy as np
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    @abstractmethod
    def solve(self, u0, dt, num_steps):
        pass

    @abstractmethod
    def validate(self):
        pass
