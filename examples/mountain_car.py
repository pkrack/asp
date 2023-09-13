import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.env_util import make_vec_env
import pathlib
from stable_baselines3.common.logger import configure
import multiprocessing
from stable_baselines3.common.vec_env import SubprocVecEnv

from asp import ASP
from asp.env import (
    AliceRewardParameters,
    AliceWrapper,
    BobReward,
    BobRewardParameters,
    BobWrapper,
    Counter,
    GoalDefinition,
)


def extract_goal(obs):
    return obs[:1]


def get_iterator(goal_obs):
    return iter((goal_obs,))


def valid(initial_obs, goal_obs):
    return not np.isclose(initial_obs, goal_obs, atol=0.01)


def out_of_zone(_):
    return False


def in_position(element_obs, goal_obs):
    return np.isclose(element_obs, goal_obs, atol=0.01)


goal_definition: GoalDefinition = GoalDefinition(
    extract_goal=extract_goal,
    get_iterator=get_iterator,
    valid=valid,
    out_of_zone=out_of_zone,
    in_position=in_position,
)


def load_data(filename):
    with open(filename) as f:
        return list(csv.DictReader(f))[2:]


def plot(data):
    a_rews = tuple(tuple(float(row["rollout/avg_alice_reward"]) for row in rundata) for rundata in data)
    b_rews = tuple(tuple(float(row["rollout/avg_bob_reward"]) for row in rundata) for rundata in data)
    successes = tuple(tuple(float(row["rollout/bob_success_rate"]) for row in rundata) for rundata in data)
    x_axis = [int(row["time/total_timesteps"]) for row in min(data, key=len)]

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(x_axis, a_rews[0], label="Alice reward")
    ax[1].plot(x_axis, b_rews[0], label="Bob reward")
    ax[2].plot(x_axis, successes[0], label="Bob success rate")

    ax[0].set_xlabel("Time step")
    ax[1].set_xlabel("Time step")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()


if __name__ == "__main__":
    # Configuration
    num_workers = multiprocessing.cpu_count()
    seed = 0
    steps_per_goal = 30
    goals_per_episode = 5
    alice_reward = AliceRewardParameters(goal_valid=1.0, bob_failed=5.0, out_of_zone_penalty=0.0)
    bob_reward = BobRewardParameters(per_object=1.0, success=5.0, in_position=in_position)
    # Create vectorized environments
    alice_env = make_vec_env(
        env_id="MountainCar-v0",
        n_envs=num_workers,
        seed=0,
        start_index=0,
        wrapper_class=AliceWrapper,
        wrapper_kwargs={
            "counter": Counter(steps_per_goal=steps_per_goal, goals_per_episode=goals_per_episode),
            "goal_definition": goal_definition,
        },
        vec_env_cls=SubprocVecEnv,
    )
    bob_env = make_vec_env(
        env_id="MountainCar-v0",
        n_envs=num_workers,
        seed=0,
        start_index=num_workers,
        wrapper_class=BobWrapper,
        wrapper_kwargs={
            "counter": Counter(steps_per_goal=steps_per_goal * 2, goals_per_episode=goals_per_episode),
            "goal_definition": goal_definition,
            "reward": BobReward(per_object=1.0, success=5.0, in_position=in_position),
        },
        vec_env_cls=SubprocVecEnv,
    )
    model = ASP(
        alice_env=alice_env,
        bob_env=bob_env,
        alice_reward_parameters=alice_reward,
        alice_ppo={"policy": "MlpPolicy"},
        bob_ppo={"policy": "MlpPolicy"},
        seed=0,
    )
    model.set_logger(logger=configure(os.path.dirname(__file__), ["csv"]))
    model.learn(total_timesteps=500_000, progress_bar=True)
    plot((load_data(pathlib.Path(os.path.dirname(__file__)) / "progress.csv"),))
