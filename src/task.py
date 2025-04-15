from typing import Callable, Optional, cast

import gymnasium as gym
import tqdm

from .agents import AbstractAgent, InfoType, ObsType, Reward
from .returns import calculate_episodic_return


class RLTask:
    """This class abstracts the concept of an agent interacting with an environment.

    :param env: the environment to interact with (e.g. a gym.Env)
    :param agent: the interacting agent
    """

    def __init__(self, env: gym.Env, agent: AbstractAgent) -> None:
        self.env = env
        self.agent = agent

    def interact(self, num_episodes: int) -> list[float]:
        """This function executes num_episodes of interaction between the agent and the environment.

        :param num_episodes: the number of episodes of the interaction
        :return: a list of episode average returns
        """
        avg_returns: list[float] = []
        """.. math:: \hat{G}_k = \frac{1}{k+1}\sum_{i=0}^k{G_i}"""
        episodic_returns: list[float] = []

        for k in tqdm.tqdm(range(num_episodes)):
            self.env.reset()
            self.agent.reset()
            t = 0
            s_t = cast(tuple[ObsType, InfoType], self.env.reset())
            r_t: Reward = 0.0
            done = False

            while not done and t < self.agent.max_episode_steps:
                a_t = self.agent.act(s_t, r_t)
                observation, r_t, terminated, truncated, info = self.env.step(a_t)

                done = terminated or truncated
                self.agent.R.append(r_t)

                s_t_plus_1 = cast(tuple[ObsType, InfoType], (observation, info))
                if self.agent.learning:
                    self.agent.on_step_end(t, s_t_plus_1)

                s_t = s_t_plus_1
                t += 1

            episodic_returns.append(calculate_episodic_return(self.agent.R, self.agent.gamma))
            avg_returns.append(sum(episodic_returns) / (k + 1))

            if self.agent.learning:
                self.agent.on_episode_end(t)

        return avg_returns

    def visualize_episode(
        self,
        max_num_steps: Optional[int] = None,
        *,
        custom_callback: Optional[Callable[[tuple[ObsType, InfoType]], None]] = None,
    ) -> None:
        """This function executes and plot an episode (or a fixed number 'max_num_steps' steps).

        :param max_num_steps: Optional, maximum number of steps to plot.
        """
        self.env.reset()
        self.agent.reset()
        t = 0
        done = False
        s_t = cast(tuple[ObsType, InfoType], self.env.reset())

        print(f"Step {t}:")
        custom_callback(s_t) if custom_callback else self.env.render()

        while not done and (max_num_steps is None or t < max_num_steps):
            t += 1
            a_t = self.agent.act(s_t)
            observation, _, terminated, truncated, info = self.env.step(a_t)
            s_t = cast(tuple[ObsType, InfoType], (observation, info))
            done = terminated or truncated

            print(f"Step {t}:")
            custom_callback(s_t) if custom_callback else self.env.render()
