import torch

from grpo.grpo_objective import GRPOObjective
from grpo.experience_collector import ExperienceBatch
from grpo.policy_base import PolicyModel


class ConstantPolicy(PolicyModel):
    """Policy that returns a constant log-probability for any (state, action)."""

    def __init__(self, log_prob_value: float):
        super().__init__()
        self._val = torch.tensor(float(log_prob_value))

    # Only get_log_probabilities is required for these tests
    def get_log_probabilities(self, states, actions):  # noqa: D401
        return self._val.repeat(len(actions))

    def generate_actions(self, states, num_actions_per_state, **kwargs):
        raise NotImplementedError

    def get_parameters(self):
        return {}

    def train(self):
        pass

    def eval(self):
        pass

    def to(self, device):
        return self


def _make_dummy_batch() -> ExperienceBatch:
    states = ["s0"]
    action_groups = [["a0", "a1"]]
    reward_groups = [[1.0, 2.0]]
    log_prob_old_groups = [[0.0, 0.0]]  # old policy probability = 1.0  (exp(0))
    return ExperienceBatch(states, action_groups, reward_groups, log_prob_old_groups)


def test_policy_loss_with_clipping():
    batch = _make_dummy_batch()
    advantages = torch.tensor([1.0, 1.0])  # already computed elsewhere

    # Current policy makes log-prob = 1 (ratio = e ≈ 2.718 > 1+0.2)
    current = ConstantPolicy(log_prob_value=1.0)
    objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.0, entropy_coeff=0.0)

    loss_dict = objective.compute_loss(current, None, batch, advantages)

    clipped_ratio = 1.0 + 0.2
    expected_policy_loss = -(clipped_ratio * advantages).mean()

    assert torch.allclose(loss_dict["policy_loss"], expected_policy_loss)
    assert torch.allclose(loss_dict["total_loss"], expected_policy_loss)


def test_kl_and_entropy_terms():
    batch = _make_dummy_batch()
    advantages = torch.tensor([0.0, 0.0])  # zero so policy loss is zero

    current = ConstantPolicy(log_prob_value=0.0)
    reference = ConstantPolicy(log_prob_value=-1.0)

    objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.5, entropy_coeff=0.1)
    result = objective.compute_loss(current, reference, batch, advantages)

    # Expected KL = mean(log_cur - log_ref) = 1 (since 0 - (-1) = 1)
    expected_kl = 0.5 * 1.0
    # Entropy bonus = coeff * (-mean(log_cur)) = 0.1 * (-0) = 0
    expected_entropy = 0.0

    assert torch.isclose(result["policy_loss"], torch.tensor(0.0))
    assert torch.isclose(result["kl_penalty"], torch.tensor(expected_kl))
    assert torch.isclose(result["entropy_bonus"], torch.tensor(expected_entropy))
    assert torch.isclose(result["total_loss"], torch.tensor(expected_kl)) 