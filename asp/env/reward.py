"""Configurable classes to compute rewards for Alice and Bob"""
from collections.abc import Callable, Iterable
from typing import NamedTuple

import numpy as np


class AliceRewardParameters(NamedTuple):
    """Parameters for Alice's reward computation."""

    goal_valid: float
    bob_failed: float
    out_of_zone_penalty: float


class AliceReward:
    """Configurable class that can be called like a function to compute Alice's
    reward."""

    def __init__(self, goal_valid: float, bob_failed: float, out_of_zone_penalty: float):
        self.goal_valid = goal_valid
        self.bob_failed = bob_failed
        self.out_of_zone_penalty = out_of_zone_penalty

    def __call__(self, valid: bool, out_of_zone: bool, bob_success: bool) -> float:
        if not valid:
            return 0
        reward = self.goal_valid
        if out_of_zone:
            reward -= self.out_of_zone_penalty
        if not bob_success:
            reward += self.bob_failed
        return reward


class BobRewardParameters(NamedTuple):
    """Parameters for Bob's reward computation."""

    per_object: float
    success: float
    in_position: Callable[[np.ndarray], bool]


class BobReward:
    """Configurable class that can be called like a function to compute Bob's reward."""

    Observation = Iterable[np.ndarray]
    Goal = Iterable[np.ndarray]

    def __init__(self, per_object: float, success: float, in_position: Callable):
        self.per_object = per_object
        self.success = success
        self.in_position = in_position
        self._objects_in_place = []

    def reset(self, num_objects) -> None:
        """Reset this object by computing the maximum reward and setting that all
        objects are not in position."""
        self._objects_in_place = [False] * num_objects

    @property
    def all_objects_in_place(self):
        """Returns True when all objects are in place."""
        assert self._objects_in_place
        return all(self._objects_in_place)

    def __call__(self, observation: Observation, goal: Goal) -> float:
        assert self._objects_in_place
        reward = self._total_object_reward(observation, goal)
        if all(self._objects_in_place):
            reward += self.success
        return reward

    def _total_object_reward(self, observation: Observation, goal: Goal) -> float:
        reward = 0
        for object_index, (object_observation, goal_object) in enumerate(zip(observation, goal)):
            reward += self._single_object_reward(object_index, object_observation, goal_object)
        return reward

    def _single_object_reward(self, index, observation, goal):
        in_place = self.in_position(observation, goal)
        was_in_place = self._objects_in_place[index]
        if in_place and not was_in_place:
            self._objects_in_place[index] = True
            return self.per_object
        in_place = self.in_position(observation, goal)
        was_in_place = self._objects_in_place[index]
        if (not in_place) and was_in_place:
            self._objects_in_place[index] = False
            return -self.per_object
        return 0
