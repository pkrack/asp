"""Rollout buffer for ASP."""
import warnings
from dataclasses import astuple, dataclass
from itertools import zip_longest

import numpy as np
import torch
from stable_baselines3.common.buffers import (
    DictRolloutBuffer,
    RolloutBuffer,
)
from stable_baselines3.common.utils import get_device


@dataclass
class Step:
    """Step data for the policy."""

    # NOTE: The asp rollout buffer depends on the names of these fields.
    # It checks if the field is named "value" or "log_prob" to determine
    # whether to stack the values into a numpy array or a torch tensor.
    obs: np.ndarray
    action: torch.Tensor
    reward: float
    episode_start: bool
    value: torch.Tensor
    log_prob: torch.Tensor

    def __post_init__(self):
        if isinstance(self.value, torch.Tensor):
            self.value = self.value.cpu()
        if isinstance(self.log_prob, torch.Tensor):
            self.log_prob = self.log_prob.cpu()


class ASPRolloutBuffer:
    """Rollout buffer for ASP"""

    def __init__(
        self,
        alice_buffer: RolloutBuffer | DictRolloutBuffer,
        bob_buffer: RolloutBuffer | DictRolloutBuffer,
        policy,
        device: torch.device | str = "auto",
    ):
        self.alice_buffer = alice_buffer
        self.bob_buffer = bob_buffer
        self.bc_steps = tuple([] for _ in range(alice_buffer.n_envs))
        self.policy = policy
        self.device = get_device(device)
        self.generator_ready = False

    @staticmethod
    def _add(buffer, steps):
        batch = zip(*(astuple(step) for step in steps))
        batch = (np.stack(x) for x in batch)
        batch = (torch.from_numpy(x) for x in batch)
        buffer.add(*batch)

    def add_alice(self, steps):
        """Add steps to the alice buffer."""
        self._add(self.alice_buffer, steps)

    def add_bob(self, steps):
        """Add steps to the bob buffer."""
        self._add(self.bob_buffer, steps)

    def add_bc(self, steps):
        """Add steps to the bc buffer."""
        for step, step_list in zip(steps, self.bc_steps, strict=True):
            step.log_prob = step.log_prob.cpu()
            step_list.append(step)

    def get(self, batch_size):
        """Get a batch of data from the buffers. Returns batches for bob and bc. Either
        may be None but not both at the same time."""
        self.bob_buffer.full = True
        self.bob_buffer.buffer_size = self.bob_buffer.pos
        bob_generator = self.bob_buffer.get(batch_size)
        bob_generator = (
            batch for batch in bob_generator if len(batch.observations) == batch_size
        )
        bc_generator = self._bc_generator(batch_size)
        return zip_longest(bob_generator, bc_generator, fillvalue=None)

    @property
    def values(self):
        """Get the values from the bob buffer."""
        return self.bob_buffer.values

    @property
    def returns(self):
        """Get the returns from the bob buffer."""
        return self.bob_buffer.returns

    def reset(self):
        """Reset the buffers."""
        self.alice_buffer.reset()
        self.bob_buffer.reset()
        self.bc_steps = tuple([] for _ in range(self.alice_buffer.n_envs))
        self.generator_ready = False

    def _bc_generator(self, batch_size):
        if not self.generator_ready:
            self.bc_steps = tuple(step for steplist in self.bc_steps for step in steplist)
            self.generator_ready = True
        n_steps = len(self.bc_steps)
        indices = np.random.permutation(n_steps)
        steps = (self.bc_steps[i] for i in indices)
        # batched: Generator[tuple[Step * batch_size]]
        batched = zip(*(iter(steps),) * batch_size)
        # batched: Generator[Step]
        for batch in batched:
            step = Step(
                obs=self._to_torch(np.stack(tuple(step.obs for step in batch))),
                action=self._to_torch(np.stack(tuple(step.action for step in batch))),
                log_prob=np.stack(tuple(step.log_prob for step in batch)),
                reward=None,
                episode_start=None,
                value=self._to_torch(torch.zeros(0)),
            )
            step.log_prob = self._to_torch(step.log_prob)
            yield step

    def _to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        if copy:
            with warnings.catch_warnings():
                # NOTE: suppress UserWarning about using clone().detach() instead of
                # torch.Tensor. We do the same as is done in stable-baselines3.
                # cf.ghttps://github.com/DLR-RM/stable-baselines3/blob/4fcda6b2d453a179c615162a36624d654e3456b2/stable_baselines3/common/buffers.py#L124
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                )
                return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)
