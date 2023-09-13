"""Utility functions for checking if a goal is valid and whether Bob succeeded"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class GoalDefinition:
    """A set of functions that together define the goal within an environment.
    `extract_goal` is a function that takes an observation as defined by the
    environment's observation space and returns the goal part of the observation.
    `get_iterator` is a function that takes a goal observation as defined by the
    `extract_goal` function and returns an iterator over the different objects inside
    that observation.
    `valid` is a function that defines whether a goal observation as defined by the
    `extract_goal` function is valid or not. The first parameter is a the initial goal
    observation, the second is the final goal observation.
    `out_of_zone` is a function that defines whether a goal observation as defined by
    the `extract_goal` function is out-of-zone or not (cf. ASP paper).
    `in_position` is a function that defines whether an object observation as defined
    by the `get_iterator` function is valid or not.
    """

    extract_goal: Callable[[np.ndarray], np.ndarray]
    get_iterator: Callable[[np.ndarray], np.ndarray]
    valid: Callable[[np.ndarray, np.ndarray], bool]
    out_of_zone: Callable[[np.ndarray], bool]
    in_position: Callable[[np.ndarray, np.ndarray], bool]

    def count_objects(self, obs):
        """Count the number of objects in a goal observation"""
        return sum(1 for _ in self.get_iterator(self.extract_goal(obs)))

    @staticmethod
    def augment(obs, goal_obs):
        """Augment an observation with the goal part"""
        return np.append(obs, goal_obs)
