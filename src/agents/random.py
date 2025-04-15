import numpy as np

from .agent import AbstractAgent, ActIndex


class RandomAgent(AbstractAgent):
    """Random policy"""

    def policy(self, chars: np.ndarray) -> ActIndex:
        return int(self.action_space.sample())
