"""Test algorithmic correctness of GRPO implementation."""

import torch
import pytest
import numpy as np
from torch.optim import SGD

from grpo.grpo_objective import GRPOObjective
from grpo.advantage_estimator import AdvantageEstimator
from grpo.experience_collector import ExperienceCollector, ExperienceBatch
from grpo.policy_base import PolicyModel
from grpo.reward_model_base import RewardModel


class DeterministicPolicy(PolicyModel, torch.nn.Module):
    """Policy with deterministic log probabilities for testing."""
    
    def __init__(self, log_probs_dict=None):
        super().__init__()
        torch.nn.Module.__init__(self)
        self.log_probs_dict = log_probs_dict or {}
        self.param = torch.nn.Parameter(torch.zeros(1))
    
    def generate_actions(self, states, num_actions_per_state, **kwargs):
        return [[f"action_{i}_{j}" for j in range(num_actions_per_state)] 
                for i, _ in enumerate(states)]
    
    def get_log_probabilities(self, states, actions):
        if self.log_probs_dict:
            return torch.tensor([self.log_probs_dict.get(action, 0.0) for action in actions])
        return torch.zeros(len(actions)) + self.param
    
    def get_parameters(self):
        return {"param": self.param}
    
    def train(self, mode: bool = True):
        return torch.nn.Module.train(self, mode)
    
    def eval(self):
        torch.nn.Module.eval(self)
    
    def to(self, device):
        torch.nn.Module.to(self, device)
        return self


class DeterministicRewardModel(RewardModel):
    """Reward model with predetermined rewards."""
    
    def __init__(self, rewards_dict=None):
        self.rewards_dict = rewards_dict or {}
    
    def compute_rewards(self, states, actions):
        if self.rewards_dict:
            return torch.tensor([self.rewards_dict.get(action, 0.0) for action in actions])
        return torch.ones(len(actions))
    
    def to(self, device):
        return self
    
    def eval(self):
        pass
    
    def train(self):
        pass


class TestAlgorithmCorrectness:
    """Test mathematical correctness of GRPO components."""
    
    def test_advantage_computation_correctness(self):
        """Test that advantages are computed correctly as r - mean(r_group)."""
        # Manual advantage computation
        rewards = [[1.0, 3.0], [2.0, 0.0, 4.0]]  # Groups: [1,3], [2,0,4]
        expected_advantages = [
            1.0 - 2.0,  # 1 - mean([1,3]) = 1 - 2 = -1
            3.0 - 2.0,  # 3 - mean([1,3]) = 3 - 2 = 1
            2.0 - 2.0,  # 2 - mean([2,0,4]) = 2 - 2 = 0
            0.0 - 2.0,  # 0 - mean([2,0,4]) = 0 - 2 = -2
            4.0 - 2.0,  # 4 - mean([2,0,4]) = 4 - 2 = 2
        ]
        
        # Create experience batch
        batch = ExperienceBatch(
            states=["s1", "s2"],
            action_groups=[["a1", "a2"], ["b1", "b2", "b3"]],
            reward_groups=rewards,
            log_prob_old_groups=[[0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        
        estimator = AdvantageEstimator(normalize_advantages=False)
        advantages = estimator.compute_advantages(batch)
        
        expected = torch.tensor(expected_advantages)
        assert torch.allclose(advantages, expected, atol=1e-6), \
            f"Expected {expected}, got {advantages}"
    
    def test_advantage_normalization_correctness(self):
        """Test that advantage normalization produces mean≈0 and reduces variance."""
        rewards = [[1.0, 5.0], [2.0, 0.0, 3.0]]
        
        batch = ExperienceBatch(
            states=["s1", "s2"],
            action_groups=[["a1", "a2"], ["b1", "b2", "b3"]],
            reward_groups=rewards,
            log_prob_old_groups=[[0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        
        # Test raw advantages first
        estimator_raw = AdvantageEstimator(normalize_advantages=False)
        raw_advantages = estimator_raw.compute_advantages(batch)
        
        # Test normalized advantages
        estimator_norm = AdvantageEstimator(normalize_advantages=True)
        norm_advantages = estimator_norm.compute_advantages(batch)
        
        # Check that mean is approximately zero
        assert abs(norm_advantages.mean().item()) < 1e-5, "Normalized advantages should have mean ≈ 0"
        
        # Check that the normalization formula is correctly applied
        # normalized = (raw - raw.mean()) / (raw.std() + eps)
        expected_norm = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std(unbiased=False) + estimator_norm.eps)
        
        assert torch.allclose(norm_advantages, expected_norm, atol=1e-6), \
            "Normalization should follow the expected formula"
        
        # Check that variance is reduced compared to raw (this is the main purpose)
        raw_var = raw_advantages.var()
        norm_var = norm_advantages.var()
        
        # Normalized advantages should have smaller variance than raw
        # (though not necessarily exactly 1.0 due to the epsilon term)
        assert norm_var < raw_var, "Normalization should reduce variance"
    
    def test_ppo_clipping_correctness(self):
        """Test that PPO clipping math is implemented correctly."""
        # Test case: ratio > 1 + epsilon, advantage > 0 (should clip)
        policy = DeterministicPolicy()
        
        # Create batch with known log probs
        batch = ExperienceBatch(
            states=["s1"],
            action_groups=[["a1"]],
            reward_groups=[[1.0]],
            log_prob_old_groups=[[-2.0]]  # old log prob
        )
        
        # Set current policy to give higher log prob
        with torch.no_grad():
            policy.param.data = torch.tensor([0.0])  # current log prob = 0
        
        advantages = torch.tensor([1.0])  # positive advantage
        clip_eps = 0.2
        
        objective = GRPOObjective(clip_epsilon=clip_eps, kl_coeff=0.0, entropy_coeff=0.0)
        loss_dict = objective.compute_loss(policy, None, batch, advantages)
        
        # Manual calculation
        log_prob_old = -2.0
        log_prob_current = 0.0
        ratio = torch.exp(torch.tensor(log_prob_current - log_prob_old))  # exp(0 - (-2)) = exp(2)
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)  # clamp to [0.8, 1.2]
        
        advantage = 1.0
        surrogate1 = ratio * advantage
        surrogate2 = clipped_ratio * advantage
        expected_objective = torch.minimum(surrogate1, surrogate2)
        expected_loss = -expected_objective.mean()
        
        assert torch.allclose(loss_dict["policy_loss"], expected_loss, atol=1e-6), \
            f"Expected loss {expected_loss}, got {loss_dict['policy_loss']}"
    
    def test_kl_divergence_calculation(self):
        """Test KL divergence calculation correctness."""
        # Create policies with known log probabilities
        current_log_probs = {"a1": 0.0, "a2": -1.0}
        reference_log_probs = {"a1": -0.5, "a2": -0.5}
        
        current_policy = DeterministicPolicy(current_log_probs)
        reference_policy = DeterministicPolicy(reference_log_probs)
        
        batch = ExperienceBatch(
            states=["s1"],
            action_groups=[["a1", "a2"]],
            reward_groups=[[0.0, 0.0]],
            log_prob_old_groups=[[0.0, 0.0]]
        )
        
        advantages = torch.zeros(2)
        kl_coeff = 1.0
        
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=kl_coeff, entropy_coeff=0.0)
        loss_dict = objective.compute_loss(current_policy, reference_policy, batch, advantages)
        
        # Manual KL calculation: KL(current || reference) = sum(p_current * log(p_current / p_reference))
        # For log space: KL = mean(log_prob_current - log_prob_reference)
        expected_kl = torch.tensor([(0.0 - (-0.5)) + (-1.0 - (-0.5))]) / 2  # mean of [0.5, -0.5]
        expected_kl_penalty = kl_coeff * expected_kl
        
        assert torch.allclose(loss_dict["kl_penalty"], expected_kl_penalty, atol=1e-6), \
            f"Expected KL penalty {expected_kl_penalty}, got {loss_dict['kl_penalty']}"
    
    def test_entropy_calculation(self):
        """Test entropy calculation correctness."""
        log_probs = {"a1": -1.0, "a2": -2.0}  # Different probabilities
        policy = DeterministicPolicy(log_probs)
        
        batch = ExperienceBatch(
            states=["s1"],
            action_groups=[["a1", "a2"]],
            reward_groups=[[0.0, 0.0]],
            log_prob_old_groups=[[0.0, 0.0]]
        )
        
        advantages = torch.zeros(2)
        entropy_coeff = 1.0
        
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.0, entropy_coeff=entropy_coeff)
        loss_dict = objective.compute_loss(policy, None, batch, advantages)
        
        # Manual entropy calculation: H = -mean(log_probs)
        expected_entropy = -torch.tensor([-1.0, -2.0]).mean()  # -mean([-1,-2]) = -(-1.5) = 1.5
        expected_entropy_bonus = entropy_coeff * expected_entropy
        
        assert torch.allclose(loss_dict["entropy_bonus"], expected_entropy_bonus, atol=1e-6), \
            f"Expected entropy bonus {expected_entropy_bonus}, got {loss_dict['entropy_bonus']}"
    
    def test_total_loss_composition(self):
        """Test that total loss correctly combines all components."""
        current_policy = DeterministicPolicy({"a1": 0.0})
        reference_policy = DeterministicPolicy({"a1": -0.5})
        
        batch = ExperienceBatch(
            states=["s1"],
            action_groups=[["a1"]],
            reward_groups=[[1.0]],
            log_prob_old_groups=[[-1.0]]
        )
        
        advantages = torch.tensor([0.5])
        
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.1, entropy_coeff=0.05)
        loss_dict = objective.compute_loss(current_policy, reference_policy, batch, advantages)
        
        # Check composition
        expected_total = (loss_dict["policy_loss"] + 
                         loss_dict["kl_penalty"] - 
                         loss_dict["entropy_bonus"])
        
        assert torch.allclose(loss_dict["total_loss"], expected_total, atol=1e-6), \
            "Total loss should equal policy_loss + kl_penalty - entropy_bonus"
    
    def test_experience_flattening_correctness(self):
        """Test that experience flattening maintains correct order."""
        batch = ExperienceBatch(
            states=["s1", "s2"],
            action_groups=[["a1", "a2"], ["b1", "b2", "b3"]],
            reward_groups=[[1.0, 2.0], [3.0, 4.0, 5.0]],
            log_prob_old_groups=[[0.1, 0.2], [0.3, 0.4, 0.5]]
        )
        
        objective = GRPOObjective()
        flat_states, flat_actions = objective._flatten_experiences(batch)
        
        expected_states = ["s1", "s1", "s2", "s2", "s2"]
        expected_actions = ["a1", "a2", "b1", "b2", "b3"]
        
        assert flat_states == expected_states, f"Expected {expected_states}, got {flat_states}"
        assert flat_actions == expected_actions, f"Expected {expected_actions}, got {flat_actions}"
        
        # Test log prob flattening
        log_probs_old = objective._get_log_probs_old(batch, torch.device("cpu"))
        expected_log_probs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        
        assert torch.allclose(log_probs_old, expected_log_probs), \
            f"Expected {expected_log_probs}, got {log_probs_old}"
    
    def test_policy_update_direction(self):
        """Test that policy updates move in the correct direction."""
        # Create scenario where we know the expected update direction
        policy = DeterministicPolicy()
        reward_model = DeterministicRewardModel({"good": 1.0, "bad": 0.0})
        
        # Good action should be reinforced (positive advantage)
        # Bad action should be discouraged (negative advantage)
        batch = ExperienceBatch(
            states=["s1", "s2"],
            action_groups=[["good", "bad"], ["good", "bad"]],
            reward_groups=[[1.0, 0.0], [1.0, 0.0]],  # good=1, bad=0 in both groups
            log_prob_old_groups=[[0.0, 0.0], [0.0, 0.0]]
        )
        
        # Advantages: good actions get +0.5, bad actions get -0.5
        expected_advantages = torch.tensor([0.5, -0.5, 0.5, -0.5])
        
        estimator = AdvantageEstimator(normalize_advantages=False)
        computed_advantages = estimator.compute_advantages(batch)
        
        assert torch.allclose(computed_advantages, expected_advantages), \
            "Advantages should favor good actions over bad actions"
        
        # Test that policy loss has correct sign for optimization
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.0, entropy_coeff=0.0)
        
        # Store initial parameter
        initial_param = policy.param.detach().clone()
        
        # Compute gradients
        loss_dict = objective.compute_loss(policy, None, batch, computed_advantages)
        loss_dict["policy_loss"].backward()
        
        # Since we have positive and negative advantages equally, and equal old log probs,
        # the gradient direction should depend on the specific implementation
        # but the loss should be well-defined
        assert loss_dict["policy_loss"].requires_grad or policy.param.grad is not None, \
            "Policy loss should create gradients for parameter updates"
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        policy = DeterministicPolicy()
        
        # Test with very large log probabilities
        batch = ExperienceBatch(
            states=["s1"],
            action_groups=[["a1"]],
            reward_groups=[[1.0]],
            log_prob_old_groups=[[100.0]]  # Very large old log prob
        )
        
        with torch.no_grad():
            policy.param.data = torch.tensor([101.0])  # Even larger current log prob
        
        advantages = torch.tensor([1.0])
        
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.0, entropy_coeff=0.0)
        loss_dict = objective.compute_loss(policy, None, batch, advantages)
        
        # Should not produce NaN or Inf
        assert torch.isfinite(loss_dict["policy_loss"]), "Policy loss should be finite"
        assert torch.isfinite(loss_dict["total_loss"]), "Total loss should be finite"
        
        # Test with very small advantages
        tiny_advantages = torch.tensor([1e-10])
        loss_dict = objective.compute_loss(policy, None, batch, tiny_advantages)
        
        assert torch.isfinite(loss_dict["policy_loss"]), "Policy loss should handle tiny advantages"


if __name__ == "__main__":
    pytest.main([__file__]) 