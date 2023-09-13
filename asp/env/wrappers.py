"""Wrappers to transform a gym Env into an Alice/Bob Env"""
from dataclasses import astuple, dataclass
from typing import NamedTuple

import gymnasium as gym
import numpy as np

from asp.env.counter import Counter
from asp.env.goal import GoalDefinition
from asp.env.reward import BobReward


@dataclass
class Step:
    """Information returned by each step"""

    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    infos: dict


@dataclass
class AliceState:
    """The state of an Alice environment"""

    goal_done: bool = True
    episode_done: bool = True
    first_obs: np.ndarray | None = None
    last_obs: np.ndarray | None = None


class AliceWrapperParameters(NamedTuple):
    """Parameters for the AliceWrapper"""

    counter: Counter
    goal_def: GoalDefinition


class AliceWrapper(gym.Wrapper):
    """Transforms a gym Env into an Alice Env"""

    def __init__(self, env: gym.Env, counter: Counter, goal_definition: GoalDefinition):
        super().__init__(env)
        self.counter = counter
        self.goal_def = goal_definition
        self.state: AliceState = AliceState()
        self.dummy_step: Step | None = None

    def step(self, action):
        assert self.dummy_step is not None and self.state.first_obs is not None
        if self.state.episode_done or self.state.goal_done:
            return astuple(self.dummy_step)
        step = Step(*self.env.step(action))
        self.state.last_obs = step.obs
        if step.terminated:
            self.state.goal_done = self.state.episode_done = True
            return astuple(self.dummy_step)
        if self.counter.last_step_of_goal:
            self._handle_last_step(step)
        self.counter.step()
        step.infos.update(dummy=False)
        return astuple(step)

    def _handle_last_step(self, step):
        valid = self.goal_def.valid(
            self.goal_def.extract_goal(self.state.first_obs),
            self.goal_def.extract_goal(step.obs),
        )
        self.state.episode_done = not valid or self.counter.last_step_of_episode
        self.state.goal_done = True
        step.infos.update(
            dummy=False,
            goal=self.goal_def.extract_goal(step.obs),
            goal_valid=valid,
            goal_out_of_zone=self.goal_def.out_of_zone(self.goal_def.extract_goal(step.obs)),
        )

    def reset(self, *_, **kwargs):
        self.counter.reset()
        first_obs, reset_infos = self.env.reset()
        self._reset_state(first_obs)
        self._reset_dummy_step(first_obs)
        return first_obs, reset_infos

    def _reset_state(self, first_obs):
        self.state.goal_done = True
        self.state.episode_done = False
        self.state.last_obs = first_obs
        self.state.first_obs = None

    def _reset_dummy_step(self, first_obs):
        obs = np.zeros_like(first_obs)
        goal_obs = self.goal_def.extract_goal(obs)
        infos = {
            "dummy": True,
            "goal": goal_obs,
            "goal_valid": False,
            "goal_out_of_zone": False,
        }
        self.dummy_step = Step(obs, 0.0, False, False, infos)

    def start_goal(self):
        """Start the next goal, if the state allows it."""
        assert self.state.last_obs is not None
        assert self.state.goal_done
        if not self.state.episode_done:
            self.state.goal_done = False
            self.state.first_obs = self.state.last_obs
            self.state.last_obs = None


@dataclass
class BobState:
    """The state of a Bob environment"""

    goal_done: bool = True
    episode_done: bool = True
    first_obs: np.ndarray | None = None
    last_obs: np.ndarray | None = None
    goal: np.ndarray | None = None


class BobWrapperParameters(NamedTuple):
    """Parameters for the BobWrapper"""

    counter: Counter
    goal_definition: GoalDefinition
    reward: BobReward


class BobWrapper(gym.Wrapper):
    """Transforms a gym Env into a Bob Env"""

    def __init__(self, env: gym.Env, counter: Counter, goal_definition: GoalDefinition, reward: BobReward):
        super().__init__(env)
        self.counter = counter
        self.goal_definition = goal_definition
        self.reward = reward
        self.state: BobState = BobState()
        self.dummy_step: Step | None = None
        self._extend_observation_space()

    def _extend_observation_space(self):
        goal_part = self.goal_definition.extract_goal(self.observation_space.low)
        low = self.goal_definition.augment(self.observation_space.low, goal_part)
        high = self.goal_definition.augment(self.observation_space.high, goal_part)
        assert (shape := len(low)) == len(high)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(shape,), dtype=np.float32)

    def step(self, action):
        assert self.dummy_step is not None
        if self.state.episode_done or self.state.goal_done:
            return astuple(self.dummy_step)
        step = Step(*self.env.step(action))
        if step.terminated:
            self.state.goal_done = True
            self.state.episode_done = True
            return astuple(self.dummy_step)
        goal_obs = self.goal_definition.extract_goal(step.obs)
        obs_object_iterator = self.goal_definition.get_iterator(goal_obs)
        goal_object_iterator = self.goal_definition.get_iterator(self.state.goal)
        step.reward = self.reward(obs_object_iterator, goal_object_iterator)
        self.state.last_obs = step.obs
        step.obs = np.append(step.obs, self.state.goal)
        step.infos.update(dummy=False)
        if self.reward.all_objects_in_place:
            self.state.goal_done = True
            step.infos.update(success=True)
            return astuple(step)
        if self.counter.last_step_of_goal:
            self.state.goal_done = self.state.episode_done = True
            step.infos.update(success=False)
            return astuple(step)
        self.counter.step()
        return astuple(step)

    def reset(self, *_, **kwargs):
        self.counter.reset()
        first_obs, reset_infos = self.env.reset()
        num_objects = self.goal_definition.count_objects(first_obs)
        self.reward.reset(num_objects)
        self._reset_state(first_obs)
        self._reset_dummy_step(first_obs)
        return self.goal_definition.augment(first_obs, self.goal_definition.extract_goal(first_obs)), reset_infos

    def _reset_state(self, first_obs):
        self.state.goal_done = True
        self.state.episode_done = False
        self.state.last_obs = first_obs

    def _reset_dummy_step(self, first_obs):
        obs = np.zeros_like(first_obs)
        goal = self.goal_definition.extract_goal(obs)
        obs = np.append(obs, goal)
        infos = {"success": False, "dummy": True}
        self.dummy_step = Step(obs, 0, False, False, infos)

    def start_goal(self, goal, valid):
        """Start the next goal, if the state allows it and the goal is valid."""
        assert self.state.last_obs is not None and self.state.goal_done and self.dummy_step is not None
        self.state.goal = goal
        if not valid:
            self.state.goal_done = self.state.episode_done = True
            return self.dummy_step.obs
        if not self.state.episode_done:
            num_objects = self.goal_definition.count_objects(self.state.last_obs)
            self.reward.reset(num_objects)
            self.counter.next_goal()
            obs = np.append(self.state.last_obs, self.state.goal)
            self.state.goal_done = False
            return obs
        return self.dummy_step.obs
