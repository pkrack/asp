"""Rollout statistics."""
from dataclasses import dataclass, field


@dataclass
class AliceStats:
    """Stats for a single episode for Alice."""
    goals: int = 0
    valid_goals: int = 0
    out_of_zone_goals: int = 0
    total_reward: float = 0

    def reset(self):
        """Reset the stats."""
        self.goals = 0
        self.valid_goals = 0
        self.out_of_zone_goals = 0
        self.total_reward = 0


@dataclass
class BobStats:
    """Stats for a single episode for Bob."""
    goals: int = 0
    successes: int = 0
    total_reward: float = 0

    def reset(self):
        """Reset the stats."""
        self.goals = 0
        self.successes = 0
        self.total_reward = 0


@dataclass
class EpisodeStats:
    """Stats for a single episode."""
    num_envs: int
    alice: AliceStats = field(default_factory=AliceStats)
    bob: BobStats = field(default_factory=BobStats)

    def reset(self):
        """Reset the stats."""
        self.alice.reset()
        self.bob.reset()


@dataclass
class RolloutStats:
    """Rollout statistics."""
    num_envs: int
    num_episodes: int = 0
    alice: AliceStats = field(default_factory=AliceStats)
    bob: BobStats = field(default_factory=BobStats)

    def add(self, episode_stats: EpisodeStats):
        """Add episode stats to the rollout stats."""
        self.num_episodes += episode_stats.num_envs
        self.alice.goals += episode_stats.alice.goals
        self.alice.valid_goals += episode_stats.alice.valid_goals
        self.alice.out_of_zone_goals += episode_stats.alice.out_of_zone_goals
        self.alice.total_reward += episode_stats.alice.total_reward
        self.bob.goals += episode_stats.bob.goals
        self.bob.successes += episode_stats.bob.successes
        self.bob.total_reward += episode_stats.bob.total_reward

    def reset(self):
        """Reset the rollout stats."""
        self.num_episodes = 0
        self.alice.reset()
        self.bob.reset()

    def log(self, logger):
        """Log the given rollout stats."""
        avg_alice_reward = self.alice.total_reward / self.alice.goals
        valid_goal_rate = self.alice.valid_goals / self.alice.goals
        out_of_zone_goal_rate = self.alice.out_of_zone_goals / self.alice.goals
        logger.record("rollout/goals_generated", self.alice.goals)
        logger.record("rollout/num_episodes", self.num_episodes)
        logger.record("rollout/avg_alice_reward", avg_alice_reward)
        logger.record("rollout/valid_goal_rate", valid_goal_rate)
        logger.record("rollout/out_of_zone_goal_rate", out_of_zone_goal_rate)
        logger.record("rollout/goals_tried", self.bob.goals)
        if self.bob.goals:
            avg_bob_reward = self.bob.total_reward / self.bob.goals
            bob_success_rate = self.bob.successes / self.bob.goals
            logger.record("rollout/avg_bob_reward", avg_bob_reward)
            logger.record("rollout/bob_success_rate", bob_success_rate)
