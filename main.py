import os

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


if __name__ == "__main__":
    grid_world_goal_finding_world(num_episodes=10_000, max_num_steps=10)
    minihack_worlds(max_num_steps=10)
