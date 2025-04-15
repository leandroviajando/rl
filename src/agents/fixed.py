import numpy as np

from src import minihack_envs

from .agent import AbstractAgent, ActIndex


class FixedAgent(AbstractAgent):
    """Hardcoded policy"""

    def policy(self, chars: np.ndarray) -> ActIndex:
        agent_position = np.where(chars == minihack_envs.AGENT)
        if agent_position[0].size == 0 or agent_position[1].size == 0:
            raise ValueError(f"Agent not found in the observation: {chars}.")
        agent_row, agent_col = agent_position[0][0], agent_position[1][0]

        if not (agent_row == chars.shape[0] - 1 or chars[agent_row + 1, agent_col] != minihack_envs.FREE):
            return minihack_envs.ACTIONS.index(minihack_envs.ACTION.S)
        return minihack_envs.ACTIONS.index(minihack_envs.ACTION.E)
