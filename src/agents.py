import random
from collections import defaultdict

import gymnasium as gym
import numpy as np

from src.minihack_envs import ActIndex, HashableState, State, hashable

Reward = float


class AbstractAgent:
    """An abstract interface for an agent.

    :param id: a str-unique identifier for the agent
    :param action_space: the actions that an agent can take
    :param max_episode_steps: the maximum number of steps in an episode
    :param alpha: the learning rate
    :param gamma: the discount factor
    :param epsilon: initial value for epsilon-greedy action selection
    :param num_planning_steps: the number of planning steps for model-based planning agents
    """

    def __init__(
        self,
        id: str,
        action_space: gym.spaces.Discrete,
        max_episode_steps: int = 50,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon: float = 0.05,
        num_planning_steps: int = 10,
    ) -> None:
        self.id = id
        self.action_space = action_space
        self.max_episode_steps = max_episode_steps

        self.learning: bool = True
        """Flag that you can change for distinguishing whether the agent is used for learning or for testing.
        You may want to disable some behaviour when not learning (e.g. no update rule, no exploration eps = 0, etc.)"""
        self.reset()

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q: defaultdict[tuple[HashableState, ActIndex], float] = defaultdict(float)
        self.state_action_count: defaultdict[tuple[HashableState, ActIndex], int] = defaultdict(int)

        self.num_planning_steps = num_planning_steps

    def reset(self) -> None:
        self.S: list[HashableState] = []
        self.A: list[ActIndex] = []
        self.R: list[Reward] = []
        self.model: dict[tuple[HashableState, ActIndex], tuple[Reward, HashableState]] = {}

    def __str__(self) -> str:
        return self.__class__.__name__

    def act(self, state: State) -> ActIndex:
        """Epsilon-greedy policy for action selection.

        :param state: the state on which to act
        :return action: the action to take in the given state
        """
        S_t = hashable(state)
        A = range(self.action_space.n)

        Q = [self.Q[(S_t, a)] for a in A]
        argmax_a = random.choice([a for a, q in enumerate(Q) if q == max(Q)])

        if self.learning and random.random() >= (1 - self.epsilon + self.epsilon / self.action_space.n):
            action = random.choice([a for a in A if a != argmax_a])
        else:
            action = argmax_a

        return action

    def on_step_end(self, t: int) -> None:
        """This function can be exploited to allow the agent to perform some internal process (e.g. learning-related)
        at the end of a step.

        :param t: the time step
        """
        pass

    def on_episode_end(self, k: int) -> None:
        """This function can be exploited to allow the agent to perform some internal process (e.g. learning-related)
        at the end of an episode.

        :param k: the episode
        """
        self.reset()


class RandomAgent(AbstractAgent):
    """Random Policy"""

    def act(self, state: State) -> ActIndex:
        return int(self.action_space.sample())


class FixedAgent(AbstractAgent):
    """Hardcoded Policy"""

    def act(self, state: State) -> ActIndex:
        from src import minihack_envs

        observation, _ = state
        chars = minihack_envs.get_crop_chars_from_observation(observation)

        agent_position = np.where(chars == minihack_envs.AGENT)
        if agent_position[0].size == 0 or agent_position[1].size == 0:
            raise ValueError(f"Agent not found in the observation: {chars}.")
        agent_row, agent_col = agent_position[0][0], agent_position[1][0]

        if not (agent_row == chars.shape[0] - 1 or chars[agent_row + 1, agent_col] != minihack_envs.FREE):
            return minihack_envs.ACTIONS.index(minihack_envs.ACTION.S)
        return minihack_envs.ACTIONS.index(minihack_envs.ACTION.E)


class MCAgent(AbstractAgent):
    """Monte Carlo On-Policy"""

    def on_episode_end(self, k: int) -> None:
        if not (len(self.S) == len(self.A) == len(self.R)):
            raise ValueError(
                f"The lengths of S ({len(self.S)}), A ({len(self.A)}), and R ({len(self.R)}) must be equal at the end of an episode."
            )
        t = len(self.S) - 1
        G_t = 0.0

        for t in range(t - 2, -1, -1):
            S_t, A_t = self.S[t], self.A[t]
            G_t = self.gamma * G_t + self.R[t + 1]

            if not self._already_occurred(S_t, A_t, t):
                self.Q[(S_t, A_t)] = self._incremental_avg(self.Q[(S_t, A_t)], G_t, self.state_action_count[(S_t, A_t)])
                self.state_action_count[(S_t, A_t)] += 1

        if k % 9999999 == 0:
            self.epsilon *= 0.5

        self.reset()

    def _already_occurred(self, state: HashableState, action: ActIndex, t: int) -> bool:
        for t_prev in range(0, t):
            if self.S[t_prev] == state and self.A[t_prev] == action:
                return True
        return False

    @staticmethod
    def _incremental_avg(prev_avg: float, curr_val: float, n: int) -> float:
        return prev_avg * (n) / (n + 1) + curr_val / (n + 1)


class SARSAAgent(AbstractAgent):
    """Temporal Difference On-Policy SARSA"""

    def on_step_end(self, t: int) -> None:
        self.Q[(self.S[t], self.A[t])] += self.alpha * (
            self.R[t] + self.gamma * self.Q[(self.S[t + 1], self.A[t + 1])] - self.Q[(self.S[t], self.A[t])]
        )


class QAgent(AbstractAgent):
    """Temporal Difference Off-Policy Q-learning"""

    def on_step_end(self, t: int) -> None:
        A = range(self.action_space.n)

        self.Q[(self.S[t], self.A[t])] += self.alpha * (
            self.R[t] + self.gamma * max([self.Q[(self.S[t + 1], a)] for a in A]) - self.Q[(self.S[t], self.A[t])]
        )


class DynaQAgent(AbstractAgent):
    """Dyna-Q with Background Model-Based Planning Strategy"""

    def on_step_end(self, t: int) -> None:
        S_t, A_t = self.S[t], self.A[t]
        R_t, S_t_plus_1 = self.R[t], self.S[t + 1]
        A = range(self.action_space.n)

        self.Q[(S_t, A_t)] += self.alpha * (
            R_t + self.gamma * max([self.Q[(S_t_plus_1, a)] for a in A]) - self.Q[(S_t, A_t)]
        )
        self.model[(S_t, A_t)] = (R_t, S_t_plus_1)

        for _ in range(self.num_planning_steps):
            S_t, A_t = random.choice(list(self.model.keys()))
            R_t, S_t_plus_1 = self.model[(S_t, A_t)]
            A = range(self.action_space.n)

            self.Q[(S_t, A_t)] += self.alpha * (
                R_t + self.gamma * max([self.Q[(S_t_plus_1, a)] for a in A]) - self.Q[(S_t, A_t)]
            )
