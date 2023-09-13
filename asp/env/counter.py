"""Implements a simple counter class for internal use"""


class Counter:
    """Keeps track of the current episode and goal"""

    def __init__(self, steps_per_goal: int, goals_per_episode: int):
        self._steps_per_goal = steps_per_goal
        self._goals_per_episode = goals_per_episode
        self.current_goal: int = 0
        self.current_step: int = 0

    @property
    def first_step_of_goal(self):
        """Returns true this is the first step of the current goal"""
        return not self.current_step

    @property
    def last_step_of_goal(self):
        """Returns true if this is the las step of the current goal"""
        return self.current_step == self._steps_per_goal - 1

    @property
    def last_goal_of_episode(self):
        """Returns true if this is the last goal of the episode"""
        return self.current_goal == self._goals_per_episode - 1

    @property
    def last_step_of_episode(self):
        """Returns true if this is the last step of the episode"""
        return self.last_step_of_goal and self.last_goal_of_episode

    def step(self):
        """Step the counter"""
        self.current_step += 1
        self.current_step %= self._steps_per_goal
        if not self.current_step:
            self.current_goal += 1

    def next_goal(self):
        """Step the counter to the next goal"""
        self.current_step = 0
        self.current_goal += 1

    def reset(self):
        """Reset the counter"""
        self.current_goal = 0
        self.current_step = 0
