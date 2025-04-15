import os
from typing import cast

import matplotlib.pyplot as plt

from src import agents, grid_world_goal_finding_env, minihack_envs, task


def grid_world_goal_finding_world(*, num_episodes: int, max_num_steps: int) -> None:
    grid_world_goal_finding_task = task.RLTask(
        grid_world_goal_finding_env.Env(5, 5),
        agents.RandomAgent(
            "grid-world-goal-finding_RandomAgent", action_space=grid_world_goal_finding_env.action_space
        ),
    )

    plt.figure()
    plt.plot(grid_world_goal_finding_task.interact(num_episodes))
    plt.title("Average Returns over Episodes: grid-world-goal-finding env, RandomAgent")
    plt.xlabel("Episodes")
    plt.ylabel("Average Return")
    savefig_path = "tex/assets/grid-world-goal-finding_RandomAgent.png"
    os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
    plt.savefig(savefig_path)
    plt.show()
    grid_world_goal_finding_task.visualize_episode(max_num_steps)


def minihack_worlds(*, max_num_steps: int) -> None:
    for env_id in [minihack_envs.EMPTY_ROOM, minihack_envs.ROOM_WITH_LAVA]:
        task.RLTask(
            minihack_envs.get_env(id=env_id, add_pixels=True),
            agents.FixedAgent(f"{env_id}_FixedAgent", action_space=minihack_envs.action_space),
        ).visualize_episode(max_num_steps, custom_callback=minihack_envs.plot_observations)


def learning(*, num_episodes: int, max_num_steps: int) -> None:
    for env_id in [
        minihack_envs.EMPTY_ROOM,
        # minihack_envs.ROOM_WITH_LAVA,
        # minihack_envs.ROOM_WITH_MONSTER,
        # minihack_envs.ROOM_WITH_MULTIPLE_MONSTERS,
    ]:
        for Agent in cast(
            list[type[agents.AbstractAgent]],
            [
                agents.MCAgent,
                agents.SARSAAgent,
                agents.QAgent,
                agents.DynaQAgent,
            ],
        ):
            agent = Agent(f"{env_id}_{str(Agent)}", action_space=minihack_envs.action_space)

            learning_env = minihack_envs.get_env(env_id, add_pixels=False)
            avg_returns = task.RLTask(learning_env, agent).interact(num_episodes)

            plt.figure()
            plt.plot(avg_returns)
            plt.title(f"Average Returns over Episodes: {env_id} env, {str(agent)}")
            plt.xlabel("Episodes")
            plt.ylabel("Average Return")
            savefig_path = f"tex/assets/{env_id}_{str(agent)}.png"
            os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
            plt.savefig(savefig_path)
            plt.show()

            eval_env = minihack_envs.get_env(env_id, add_pixels=True)
            task.RLTask(eval_env, agent).visualize_episode(
                max_num_steps, custom_callback=minihack_envs.plot_observations
            )


if __name__ == "__main__":
    grid_world_goal_finding_world(num_episodes=10_000, max_num_steps=10)
    minihack_worlds(max_num_steps=10)
    learning(num_episodes=700, max_num_steps=10)
