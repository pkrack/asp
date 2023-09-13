"""ASP algorithm implementation."""
import logging
from collections import deque
from copy import deepcopy
from functools import partial
from itertools import chain, zip_longest
from types import SimpleNamespace
from typing import TypeVar

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import PPO

from asp.env.reward import AliceRewardParameters
from asp.ppo_patched import PPO as PatchedPPO
from asp.rollout_buffer import ASPRolloutBuffer, Step
from asp.stats import EpisodeStats, RolloutStats

logger = logging.getLogger(__name__)
# We know when a goal is done if these keys are present in the info dict
ALICE_DONE_KEY = "goal"
BOB_DONE_KEY = "success"


SelfASP = TypeVar("SelfASP", bound="ASP")


class ASP(PatchedPPO):
    """ASP algorithm, derived from PPO. Has an internal PPO trained to generate
    goals."""

    def __init__(
        self,
        alice_env: VecEnv,
        bob_env: VecEnv,
        alice_reward_parameters: AliceRewardParameters,
        alice_ppo: dict,
        bob_ppo: dict,
        seed: int,
        bc_coef: float = 1.0,
        n_steps: int = 2048,
        batch_size: int = 64,
    ):
        self.alice_env = alice_env
        self.bob_env = bob_env
        self.alice_reward_parameters = alice_reward_parameters
        super().__init__(
            env=self.bob_env,
            seed=seed,
            n_steps=n_steps,
            batch_size=batch_size,
            **bob_ppo,
        )
        self.bc_coef = bc_coef
        self.alice = PPO(
            env=self.alice_env,
            seed=seed,
            n_steps=n_steps,
            batch_size=batch_size,
            **alice_ppo,
        )
        bob_rollout_buffer = self.rollout_buffer
        self.rollout_buffer = ASPRolloutBuffer(
            self.alice.rollout_buffer,
            bob_rollout_buffer,
            self.policy,
        )
        self._last_observation = SimpleNamespace(alice=None, bob=None)
        self._first_step = SimpleNamespace(alice=True, bob=True)
        self.step_queues = SimpleNamespace(
            alice=tuple(deque() for _ in range(alice_env.num_envs)),
            bob=tuple(deque() for _ in range(alice_env.num_envs)),
            bc=tuple(deque() for _ in range(alice_env.num_envs)),
        )
        self.episode_queues = SimpleNamespace(alice=deque(), bob=deque(), bc=deque())
        self._callback = None
        self._rollout_stats: RolloutStats = RolloutStats(self.alice_env.num_envs)
        self._episode_stats: EpisodeStats = EpisodeStats(self.alice_env.num_envs)

    def train(self) -> None:
        self.logger.info("Training alice")
        self.alice.train()
        self.alice.logger.dump()
        self.logger.info("Training bob")
        super().train()

    def learn(
        self: SelfASP,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "ASP",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfASP:
        self.alice._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name + "_alice",
            False,
        )
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def collect_rollouts(self, env, callback: BaseCallback, _, n_rollout_steps):
        self.policy.set_training_mode(False)
        self.alice.policy.set_training_mode(False)
        self._callback = callback
        self.rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()
        self._rollout_stats.reset()
        while not self.alice.rollout_buffer.full:
            self._generate_new_episodes()
            self._add_batch_to_alice_buffer()
        self._fill_bob_and_bc_buffers()
        policy, buffer = self.alice.policy, self.rollout_buffer.alice_buffer
        self._compute_returns_and_advantage(policy, buffer)
        policy, buffer = self.policy, self.rollout_buffer.bob_buffer
        self._compute_returns_and_advantage(policy, buffer)
        self.num_timesteps += self.rollout_buffer.bob_buffer.pos * self.bob_env.num_envs
        callback.on_rollout_end()
        self._clear_queues()
        self._rollout_stats.log(self.logger)

    def _generate_new_episodes(self):
        while any(not queue for queue in self.step_queues.alice):
            if not self.episode_queues.alice:
                alice_episodes, bob_episodes, bc_episodes = self._run_asp_episode()
                self.episode_queues.alice.extend(alice_episodes)
                self.episode_queues.bob.extend(bob_episodes)
                self.episode_queues.bc.extend(bc_episodes)
            shortest_step_queue = min(self.step_queues.alice, key=len)
            steps = (step for episode in self.episode_queues.alice.popleft() for step in episode)
            shortest_step_queue.extend(steps)

    def _run_asp_episode(self):
        alice_episodes = [[] for _ in range(self.alice_env.num_envs)]
        bob_episodes = [[] for _ in range(self.bob_env.num_envs)]
        bc_episodes = [[] for _ in range(self.bob_env.num_envs)]
        self._reset_environments()
        self._episode_stats.reset()
        while True:
            alice_trajectories, goal = self._gen_alice()
            bob_trajectories, success = self._gen_bob(goal)
            self._update_alice_reward(alice_trajectories, goal, success)
            bc_trajectories = self._gen_bc(alice_trajectories, success, goal)
            self.add_trajectories(alice_episodes, alice_trajectories)
            self.add_trajectories(bob_episodes, bob_trajectories)
            self.add_trajectories(bc_episodes, bc_trajectories)
            if self.all_alices_are_done(alice_trajectories):
                break
        self._rollout_stats.add(self._episode_stats)
        return alice_episodes, bob_episodes, bc_episodes

    def _reset_environments(self):
        self._last_observation.alice = self.alice_env.reset()
        self._last_observation.bob = self.bob_env.reset()
        self._first_step.alice = True
        self._first_step.bob = True

    def _gen_alice(self):
        alice_trajectory = [[] for _ in range(self.alice_env.num_envs)]
        goals = [{} for _ in range(self.alice_env.num_envs)]
        self.alice_env.env_method("start_goal")
        while not all(goal for goal in goals):
            step = self._execute_alice_step()
            self._update_alice_trajectory(alice_trajectory, goals, step)
            self._first_step.alice = False
        return alice_trajectory, goals

    def _execute_alice_step(self):
        action, value, log_prob = self._alice_action_value_log_prob(self._last_observation.alice)
        self._callback.update_locals(locals())
        if self._callback.on_step() is False:
            return False
        new_obs, reward, _, infos = self.alice_env.step(action)
        self._episode_stats.alice.total_reward += reward.sum()
        last_obs = self._last_observation.alice
        self._last_observation.alice = new_obs
        return action, value, log_prob, last_obs, reward, infos

    def _alice_action_value_log_prob(self, obs):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.device)
            actions, values, log_probs = self.alice.policy(obs_tensor)
        actions = actions.cpu().numpy()
        clipped_actions = actions
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                actions,
                self.alice.action_space.low,
                self.alice.action_space.high,
            )
        return clipped_actions, values, log_probs

    # pylint: disable=too-many-locals
    def _update_alice_trajectory(self, alice_trajectory, goals, step_data):
        action, value, log_prob, obs, reward, info = step_data
        step_params = (obs, action, reward, [], value, log_prob)
        steps = zip_longest(*step_params, fillvalue=self._first_step.alice)
        steps = (Step(*step) for step in steps)
        gen = zip(alice_trajectory, steps, info, goals, strict=True)
        gen = ((t, s, i, g) for t, s, i, g in gen if not g)
        for trajectory, step, info, goal in gen:
            trajectory.append(step)
            if info["dummy"]:
                trajectory.clear()
            if ALICE_DONE_KEY in info:
                keys = ("goal", "goal_valid", "goal_out_of_zone")
                self._episode_stats.alice.goals += 1
                if info["goal_valid"]:
                    self._episode_stats.alice.valid_goals += 1
                    if info["goal_out_of_zone"]:
                        self._episode_stats.alice.out_of_zone_goals += 1
                goal.update((k, info[k]) for k in keys)

    def _gen_bob(self, alice_goal):
        bob_trajectory = [[] for _ in range(self.bob_env.num_envs)]
        successes = [None for _ in range(self.bob_env.num_envs)]
        self._start_bob_goals(alice_goal)
        while any(success is None for success in successes):
            steps = self._execute_bob_step()
            self._update_bob_trajectory(bob_trajectory, successes, steps)
            self._first_step.bob = False
        return bob_trajectory, successes

    def _start_bob_goals(self, alice_goal):
        goals = ((goal["goal"], goal["goal_valid"]) for goal in alice_goal)
        start_goal = partial(self.bob_env.env_method, "start_goal")
        for idx, (goal, valid) in enumerate(goals):
            [obs] = start_goal(goal, valid, indices=idx)
            self._last_observation.bob[idx] = obs

    def _execute_bob_step(self):
        action, value, log_prob = self._bob_action_value_log_prob(self._last_observation.bob)
        new_obs, reward, _, info = self.bob_env.step(action)
        self._episode_stats.bob.total_reward += reward.sum()
        last_obs = self._last_observation.bob
        self._last_observation.bob = new_obs
        return action, value, log_prob, last_obs, reward, info

    def _bob_action_value_log_prob(self, obs):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.device)
            actions, values, log_probs = self.policy(obs_tensor)
        actions = actions.cpu().numpy()
        clipped_actions = actions
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return clipped_actions, values, log_probs

    def _update_bob_trajectory(self, bob_trajectory, successes, steps):
        action, value, log_prob, obs, reward, info = steps
        step_params = (obs, action, reward, [], value, log_prob)
        steps = zip_longest(*step_params, fillvalue=self._first_step.bob)
        steps = (Step(*step) for step in steps)
        gen = zip(steps, info, successes, strict=True)
        gen = ((idx, s, i, succ) for idx, (s, i, succ) in enumerate(gen))
        gen = ((idx, s, i) for idx, s, i, succ in gen if succ is None)
        for idx, step, info in gen:
            if not info["dummy"]:
                bob_trajectory[idx].append(step)
            if BOB_DONE_KEY in info:
                if not info["dummy"]:
                    self._episode_stats.bob.goals += 1
                    if info["success"]:
                        self._episode_stats.bob.successes += 1
                successes[idx] = info["success"]

    def _update_alice_reward(self, trajectory, goals, successes):
        gen = zip(trajectory, goals, successes)
        gen = ((t, g, s) for t, g, s in gen if t)
        gen = ((t, g, s) for t, g, s in gen if g["goal_valid"])
        gen = ((t[-1], g, s) for t, g, s in gen)
        gen = ((t, g["goal_out_of_zone"], s) for t, g, s in gen)
        for last_step, out_of_zone, success in gen:
            last_step.reward = self._compute_alice_reward(out_of_zone, success)
            self._episode_stats.alice.total_reward += last_step.reward

    def _compute_alice_reward(self, out_of_zone, success):
        reward = self.alice_reward_parameters.goal_valid
        if out_of_zone:
            reward -= self.alice_reward_parameters.out_of_zone_penalty
        if not success:
            reward += self.alice_reward_parameters.bob_failed
        return reward

    def _gen_bc(self, trajectories, success, goals):
        bc_trajectories = [[] for _ in range(self.alice_env.num_envs)]
        gen = zip(bc_trajectories, trajectories, success, goals, strict=True)
        gen = ((bc, a, s, g) for bc, a, s, g in gen if g["goal_valid"])
        gen = ((bc, a, g) for bc, a, s, g in gen if not s)
        gen = ((bc, a, g["goal"]) for bc, a, g in gen)
        gen = ((bc, step, g) for bc, a, g in gen for step in a)
        for bc_trajectory, step, goal in gen:
            bc_step = deepcopy(step)
            bc_step.reward = 0
            bc_step.obs = torch.cat((torch.from_numpy(step.obs), torch.from_numpy(goal)))
            bc_step.action = (
                torch.from_numpy(step.action) if isinstance(step.action, np.ndarray) else torch.tensor([step.action])
            )
            with torch.no_grad():
                _, bc_step.log_prob, _ = self.policy.evaluate_actions(
                    bc_step.obs.unsqueeze(0).to(self.device), bc_step.action.unsqueeze(0).to(self.device)
                )
            bc_trajectory.append(bc_step)
        return bc_trajectories

    @staticmethod
    def add_trajectories(episodes, trajectories):
        """Add the trajectories to the episodes if the trajectory is not empty."""
        for env_idx, episode in enumerate(trajectories):
            if episode:
                episodes[env_idx].append(episode)

    @staticmethod
    def all_alices_are_done(trajectories):
        """Return True if all the episodes are done, i.e. when the last trajectories are
        empty."""
        return all(not traj for traj in trajectories)

    def _add_batch_to_alice_buffer(self):
        batch = (queue.popleft() for queue in self.step_queues.alice)
        self.rollout_buffer.add_alice(batch)

    def _fill_bob_and_bc_buffers(self):
        self._fill_step_queues(self.episode_queues.bob, self.step_queues.bob)
        self._fill_step_queues(self.episode_queues.bc, self.step_queues.bc)
        self._consume_step_queues(self.rollout_buffer.add_bob, self.step_queues.bob)
        self._consume_step_queues(self.rollout_buffer.add_bc, self.step_queues.bc)

    @staticmethod
    def _fill_step_queues(episodes, queues):
        non_empty_episodes = (episode for episode in episodes if episode)
        for episode in non_empty_episodes:
            shortest_queue = min(queues, key=len)
            steps = (step for goal in episode for step in goal)
            for step in steps:
                shortest_queue.append(step)

    @staticmethod
    def _consume_step_queues(add_method, queues):
        while True:
            try:
                add_method(queue.popleft() for queue in queues)
            except IndexError:
                break

    def _compute_returns_and_advantage(self, policy, rollout_buffer):
        if len(rollout_buffer.observations):
            with torch.no_grad():
                obs = rollout_buffer.observations[-1]
                obs = obs_as_tensor(obs, self.device)
                values = policy.predict_values(obs)
            dones = rollout_buffer.episode_starts[-1]
            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    def _clear_queues(self):
        to_clear = chain(
            self.step_queues.alice,
            self.step_queues.bob,
            self.step_queues.bc,
            (
                self.episode_queues.alice,
                self.episode_queues.bob,
                self.episode_queues.bc,
            ),
        )
        for queue in to_clear:
            queue.clear()

    def _excluded_save_params(self):
        return [
            *super()._excluded_save_params(),
            "alice_env",
            "bob_env",
            "alice",
            "_callback",
            "_step_queues",
            "episode_queues",
        ]
