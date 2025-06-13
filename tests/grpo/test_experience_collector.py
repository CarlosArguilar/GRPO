import torch
import pytest

from grpo.experience_collector import ExperienceCollector
from grpo.policy_base import PolicyModel
from grpo.reward_model_base import RewardModel


class DummyPolicy(PolicyModel):
    """A minimal deterministic policy for testing ExperienceCollector."""

    def generate_actions(self, states, num_actions_per_state, **generation_kwargs):
        # Produce simple string actions based on state index
        actions = []
        for idx, _ in enumerate(states):
            actions.append([f"s{idx}_a{j}" for j in range(num_actions_per_state)])
        return actions

    def get_log_probabilities(self, states, actions):
        # Return zeros for simplicity
        return torch.zeros(len(actions))

    # The remaining abstract methods are no-ops for the dummy
    def get_parameters(self):
        return {}

    def train(self):
        pass

    def eval(self):
        pass

    def to(self, device):
        return self


class DummyReward(RewardModel):
    """A reward model that returns 1.0 for every (state, action)."""

    def compute_rewards(self, states, actions):
        return torch.ones(len(actions))

    def to(self, device):
        return self

    def eval(self):
        pass

    def train(self):
        pass


def test_collect_experiences():
    policy = DummyPolicy()
    reward_model = DummyReward()
    collector = ExperienceCollector(policy, reward_model, group_size=3)

    states = ["prompt_0", "prompt_1"]
    batch = collector.collect_experiences(states)

    # Basic shape checks
    assert batch.states == states
    assert len(batch.action_groups) == 2
    assert len(batch.action_groups[0]) == 3
    assert len(batch.reward_groups) == 2
    assert len(batch.reward_groups[0]) == 3
    assert len(batch.log_prob_old_groups) == 2
    assert len(batch.log_prob_old_groups[0]) == 3

    # Reward and log-prob values
    assert all(r == 1.0 for grp in batch.reward_groups for r in grp)
    assert all(lp == 0.0 for grp in batch.log_prob_old_groups for lp in grp) 