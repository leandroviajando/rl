from typing import Optional, SupportsFloat

import gymnasium as gym
import numpy as np

from src.minihack_envs import (
    ActIndex,
    InfoType,
    ObsType,
    get_crop_chars_from_observation,
)

State = bytes
Reward = SupportsFloat


class AbstractAgent:
    """An abstract interface for an agent.

    :param id: a str-unique identifier for the agent
    :param action_space: the actions that an agent can take
    :param max_episode_steps: the maximum number of steps in an episode
    :param alpha: the learning rate
    :param gamma: the discount factor
    :param epsilon: initial value for epsilon-greedy action selection
    """

    def __init__(
        self,
        id: str,
        action_space: gym.spaces.Discrete,
        max_episode_steps: int = 50,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon: float = 0.05,
    ) -> None:
        self.id = id
        self.action_space = action_space
        self.max_episode_steps = max_episode_steps

        # Flag that you can change for distinguishing whether the agent is used for learning or for testing.
        # You may want to disable some behaviour when not learning (e.g. no update rule, no exploration eps = 0, etc.)
        self.learning: bool = True
        self.reset()

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def reset(self) -> None:
        self.S: list[Optional[State]] = []
        self.A: list[Optional[ActIndex]] = []
        self.R: list[Reward] = []

    def __str__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def hash(chars: np.ndarray) -> State:
        return chars.data.tobytes()

    def act(self, state: tuple[ObsType, InfoType], reward: Reward = 0.0) -> ActIndex:
        """This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.

        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action).
        :return action: the action to take in the given state
        """
        observation, info = state
        chars = get_crop_chars_from_observation(observation)
        self.S.append(self.hash(chars))

        action = self.policy(chars)
        self.A.append(action)
        return action

    def policy(self, chars: np.ndarray) -> ActIndex:
        raise NotImplementedError

    def on_step_end(self, t: int, next_state: tuple[ObsType, InfoType]) -> None:
        """This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of a step.

        :param t: the time step
        """
        pass

    def on_episode_end(self, t: int) -> None:
        """This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.

        :param t: the time step
        """
        self.reset()
