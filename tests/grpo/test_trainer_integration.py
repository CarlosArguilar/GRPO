import torch
import pytest

from torch.optim import Adam

from grpo.trainer import GRPOConfig, GRPOTrainer
from grpo.policy_base import PolicyModel
from grpo.reward_model_base import RewardModel


class TinyPolicy(PolicyModel, torch.nn.Module):
    """Minimal trainable policy that outputs constant log-probs = learnable param."""

    def __init__(self):
        super().__init__()
        torch.nn.Module.__init__(self)
        self.logit = torch.nn.Parameter(torch.zeros(1))

    # Policy API
    def generate_actions(self, states, num_actions_per_state, **kwargs):
        return [[f"act_{i}_{j}" for j in range(num_actions_per_state)] for i, _ in enumerate(states)]

    def get_log_probabilities(self, states, actions):
        return self.logit.expand(len(actions))

    def get_parameters(self):
        return {"logit": self.logit}

    # Override to match nn.Module signature (mode: bool = True)
    def train(self, mode: bool = True):  # type: ignore[override]
        return torch.nn.Module.train(self, mode)

    def eval(self):
        torch.nn.Module.eval(self)

    def to(self, device):
        torch.nn.Module.to(self, device)
        return self


class ConstantReward(RewardModel):
    def compute_rewards(self, states, actions):
        return torch.ones(len(actions))

    def to(self, device):
        return self

    def eval(self):
        pass

    def train(self):
        pass


def test_trainer_single_step():
    policy = TinyPolicy()
    reward_model = ConstantReward()
    optimizer = Adam(policy.parameters(), lr=1e-3)
    config = GRPOConfig(group_size=2, batch_size=4, normalize_advantages=False)

    trainer = GRPOTrainer(policy, reward_model, optimizer, config)

    states = ["s0", "s1"]
    metrics = trainer.train_step(states)

    # Basic checks – we expect epoch_0_* keys now
    expected_suffixes = {"total_loss", "policy_loss", "kl_penalty", "entropy_bonus"}
    for suffix in expected_suffixes:
        assert any(k.endswith(suffix) for k in metrics.keys())

    # Ensure gradients updated param
    before = policy.logit.detach().clone()
    trainer.train_step(states)
    after = policy.logit.detach().clone()
    assert not torch.allclose(before, after)


def _dummy_loader(states_list):
    """Simple iterable data loader cycling through provided list."""
    while True:
        for s in states_list:
            yield s


def test_train_loop_and_checkpoint(tmp_path):
    policy = TinyPolicy()
    reward_model = ConstantReward()
    optimizer = Adam(policy.parameters(), lr=1e-3)
    cfg = GRPOConfig(group_size=2, batch_size=4, reference_update_freq=1, checkpoint_freq=2)

    trainer = GRPOTrainer(policy, reward_model, optimizer, cfg)

    loader = _dummy_loader([["a", "b"]])

    history = trainer.train(loader, total_steps=3, checkpoint_dir=tmp_path)

    # Ensure history length matches steps
    assert len(history) == 3

    # Check that checkpoint file created
    ckpt_file = tmp_path / "final_checkpoint.pt"
    assert ckpt_file.exists()

    # Test loading does not raise
    trainer.load_checkpoint(str(ckpt_file)) 