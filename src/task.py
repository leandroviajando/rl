from typing import Callable, Optional, cast

import gymnasium as gym
import tqdm

from .agents import AbstractAgent, Reward, State
from .minihack_envs import hashable
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

            state = cast(State, self.env.reset())
            self.agent.S.append(hashable(state))
            action = self.agent.act(state)
            self.agent.A.append(action)
            reward: Reward = 0.0
            done = False

            for t in range(self.agent.max_episode_steps):
                observation, reward, terminated, truncated, info = self.env.step(action)  # type: ignore[assignment]
                self.agent.R.append(reward)

                done = terminated or truncated
                if done:
                    break

                state = cast(State, (observation, info))
                self.agent.S.append(hashable(state))
                action = self.agent.act(state)
                self.agent.A.append(action)

                self.agent.on_step_end(t)

            episodic_returns.append(calculate_episodic_return(self.agent.R, self.agent.gamma))
            avg_returns.append(sum(episodic_returns) / (k + 1))
            self.agent.on_episode_end(k)

        return avg_returns

    def visualize_episode(
        self,
        max_num_steps: Optional[int] = None,
        *,
        custom_callback: Optional[Callable[[State], None]] = None,
    ) -> None:
        """This function executes and plot an episode (or a fixed number 'max_num_steps' steps).

        :param max_num_steps: Optional, maximum number of steps to plot.
        :param custom_callback: Optional, a function to call at each step instead of rendering the environment.
        """
        self.env.reset()
        self.agent.reset()
        self.agent.learning = False

        t = 0
        done = False
        state = cast(State, self.env.reset())

        print(f"Step {t}:")
        custom_callback(state) if custom_callback else self.env.render()

        while not done and (max_num_steps is None or t < max_num_steps):
            t += 1
            action = self.agent.act(state)
            observation, _, terminated, truncated, info = self.env.step(action)
            state = cast(State, (observation, info))
            done = terminated or truncated

            print(f"Step {t}:")
            custom_callback(state) if custom_callback else self.env.render()
