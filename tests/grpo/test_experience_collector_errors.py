import torch
import pytest

from grpo.experience_collector import ExperienceCollector
from grpo.policy_base import PolicyModel
from grpo.reward_model_base import RewardModel


class BadPolicyWrongGroup(PolicyModel):
    """Policy that returns an incorrect number of actions per state to trigger validation error."""

    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size

    def generate_actions(self, states, num_actions_per_state, **generation_kwargs):
        # Return one fewer action than expected
        return [[f"bad_a{j}" for j in range(num_actions_per_state - 1)] for _ in states]

    def get_log_probabilities(self, states, actions):
        return torch.zeros(len(actions))

    # Stub implementations
    def get_parameters(self):
        return {}

    def train(self):
        pass

    def eval(self):
        pass

    def to(self, device):
        return self


class BadRewardWrongShape(RewardModel):
    """Reward model that returns a 2-D tensor instead of the required 1-D."""

    def compute_rewards(self, states, actions):
        # Intentionally return shape (N, 1)
        return torch.ones(len(actions), 1)

    def to(self, device):
        return self

    def eval(self):
        pass

    def train(self):
        pass


class DummyGoodPolicy(PolicyModel):
    """Minimal correct policy used for reward-shape test."""

    def generate_actions(self, states, num_actions_per_state, **generation_kwargs):
        actions = []
        for idx, _ in enumerate(states):
            actions.append([f"s{idx}_a{j}" for j in range(num_actions_per_state)])
        return actions

    def get_log_probabilities(self, states, actions):
        return torch.zeros(len(actions))

    def get_parameters(self):
        return {}

    def train(self):
        pass

    def eval(self):
        pass

    def to(self, device):
        return self


class DummyGoodReward(RewardModel):
    """Reward model that returns correct 1-D tensor."""

    def compute_rewards(self, states, actions):
        return torch.ones(len(actions))

    def to(self, device):
        return self

    def eval(self):
        pass

    def train(self):
        pass


# -----------------------------------------------------------------------------
# Actual tests
# -----------------------------------------------------------------------------

def test_group_size_validation():
    """Collector should raise when policy emits wrong group size."""
    collector = ExperienceCollector(BadPolicyWrongGroup(group_size=3), DummyGoodReward(), group_size=3)
    with pytest.raises(ValueError, match="Expected 3 actions per state"):
        collector.collect_experiences(["state_0", "state_1"])


def test_reward_shape_validation():
    """Collector should raise when reward tensor has incorrect shape."""
    collector = ExperienceCollector(DummyGoodPolicy(), BadRewardWrongShape(), group_size=2)
    with pytest.raises(ValueError, match="rewards.*1-D tensor"):
        collector.collect_experiences(["state"])


def test_buffer_cleanup():
    """_flat_states / _flat_actions buffers must be cleared after collection."""
    collector = ExperienceCollector(DummyGoodPolicy(), DummyGoodReward(), group_size=2)
    collector.collect_experiences(["state_a", "state_b"])
    assert collector._flat_states is None
    assert collector._flat_actions is None 