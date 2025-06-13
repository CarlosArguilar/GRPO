"""Comprehensive gradient flow tests for GRPO components."""

import torch
import pytest
from torch.optim import SGD

from grpo.grpo_objective import GRPOObjective
from grpo.policy_base import PolicyModel
from grpo.reward_model_base import RewardModel
from grpo.experience_collector import ExperienceCollector, ExperienceBatch
from grpo.trainer import GRPOTrainer, GRPOConfig


class GradientTestPolicy(PolicyModel, torch.nn.Module):
    """Policy with multiple parameters for gradient testing."""
    
    def __init__(self, num_params=3):
        super().__init__()
        torch.nn.Module.__init__(self)
        # Multiple parameters to test gradient flow
        self.params = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(1) * 0.1) for _ in range(num_params)
        ])
        self.linear = torch.nn.Linear(1, 1)
    
    def generate_actions(self, states, num_actions_per_state, **kwargs):
        return [[f"action_{i}_{j}" for j in range(num_actions_per_state)] 
                for i, _ in enumerate(states)]
    
    def get_log_probabilities(self, states, actions):
        # Create log probs that depend on all parameters
        base_logit = sum(self.params) + self.linear.weight.sum() + self.linear.bias
        return base_logit.expand(len(actions))
    
    def get_parameters(self):
        return {f"param_{i}": p for i, p in enumerate(self.params)}
    
    def train(self, mode: bool = True):
        return torch.nn.Module.train(self, mode)
    
    def eval(self):
        torch.nn.Module.eval(self)
    
    def to(self, device):
        torch.nn.Module.to(self, device)
        return self


class VariableRewardModel(RewardModel):
    """Reward model that creates non-zero advantages."""
    
    def compute_rewards(self, states, actions):
        rewards = []
        for i, action in enumerate(actions):
            # Create varying rewards based on action index
            rewards.append(float(i % 3))  # 0, 1, 2, 0, 1, 2, ...
        return torch.tensor(rewards, dtype=torch.float32)
    
    def to(self, device):
        return self
    
    def eval(self):
        pass
    
    def train(self):
        pass


class TestGradientFlow:
    """Test gradient flow through GRPO components."""
    
    def test_policy_loss_gradients(self):
        """Test that policy loss gradients flow correctly."""
        policy = GradientTestPolicy()
        reward_model = VariableRewardModel()
        
        # Create experience batch
        collector = ExperienceCollector(policy, reward_model, group_size=2)
        batch = collector.collect_experiences(["state1", "state2"])
        
        # Create advantages (non-zero for gradient flow)
        advantages = torch.tensor([0.5, -0.3, 0.2, -0.1])
        
        # Compute loss
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.0, entropy_coeff=0.0)
        loss_dict = objective.compute_loss(
            current_policy=policy,
            reference_policy=None,
            experience_batch=batch,
            advantages=advantages
        )
        
        # Zero gradients and backward
        for param in policy.parameters():
            param.grad = None
        
        loss_dict["policy_loss"].backward()
        
        # Check that all parameters have gradients
        for name, param in policy.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                f"Parameter {name} has zero gradient"
    
    def test_kl_penalty_gradients(self):
        """Test that KL penalty gradients flow correctly."""
        policy = GradientTestPolicy()
        reference_policy = GradientTestPolicy()
        reward_model = VariableRewardModel()
        
        # Freeze reference policy
        for param in reference_policy.parameters():
            param.requires_grad_(False)
        
        # Make policies different to create non-zero KL
        with torch.no_grad():
            for ref_p, curr_p in zip(reference_policy.parameters(), policy.parameters()):
                ref_p.data = curr_p.data + 0.5
        
        collector = ExperienceCollector(policy, reward_model, group_size=2)
        batch = collector.collect_experiences(["state1", "state2"])
        advantages = torch.zeros(4)  # Zero advantages to isolate KL
        
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.1, entropy_coeff=0.0)
        loss_dict = objective.compute_loss(
            current_policy=policy,
            reference_policy=reference_policy,
            experience_batch=batch,
            advantages=advantages
        )
        
        # Zero gradients and backward
        for param in policy.parameters():
            param.grad = None
        
        loss_dict["kl_penalty"].backward()
        
        # Check KL penalty creates gradients
        for name, param in policy.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient from KL penalty"
            # KL should create non-zero gradients when policies differ
            if "param" in name or "weight" in name or "bias" in name:
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                    f"Parameter {name} has zero gradient from KL penalty"
    
    def test_entropy_bonus_gradients(self):
        """Test that entropy bonus gradients flow correctly."""
        policy = GradientTestPolicy()
        reward_model = VariableRewardModel()
        
        collector = ExperienceCollector(policy, reward_model, group_size=2)
        batch = collector.collect_experiences(["state1", "state2"])
        advantages = torch.zeros(4)  # Zero advantages to isolate entropy
        
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.0, entropy_coeff=0.1)
        loss_dict = objective.compute_loss(
            current_policy=policy,
            reference_policy=None,
            experience_batch=batch,
            advantages=advantages
        )
        
        # Zero gradients and backward
        for param in policy.parameters():
            param.grad = None
        
        loss_dict["entropy_bonus"].backward()
        
        # Check entropy creates gradients
        for name, param in policy.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient from entropy"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                f"Parameter {name} has zero gradient from entropy"
    
    def test_total_loss_gradient_composition(self):
        """Test that total loss gradients are composition of components."""
        policy = GradientTestPolicy()
        reference_policy = GradientTestPolicy()
        reward_model = VariableRewardModel()
        
        # Make reference different
        with torch.no_grad():
            for ref_p, curr_p in zip(reference_policy.parameters(), policy.parameters()):
                ref_p.data = curr_p.data + 0.3
                ref_p.requires_grad_(False)
        
        collector = ExperienceCollector(policy, reward_model, group_size=2)
        batch = collector.collect_experiences(["state1", "state2"])
        advantages = torch.tensor([0.5, -0.3, 0.2, -0.1])
        
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.05, entropy_coeff=0.02)
        
        # Compute individual component gradients
        components = ["policy_loss", "kl_penalty", "entropy_bonus"]
        component_grads = {}
        
        for component in components:
            # Fresh policy for each component
            test_policy = GradientTestPolicy()
            test_policy.load_state_dict(policy.state_dict())
            
            # Zero gradients
            for param in test_policy.parameters():
                param.grad = None
            
            # Compute loss with only this component
            if component == "policy_loss":
                test_obj = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.0, entropy_coeff=0.0)
                loss_dict = test_obj.compute_loss(test_policy, None, batch, advantages)
                loss_dict[component].backward()
            elif component == "kl_penalty":
                test_obj = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.05, entropy_coeff=0.0)
                loss_dict = test_obj.compute_loss(test_policy, reference_policy, batch, 
                                                torch.zeros_like(advantages))
                loss_dict[component].backward()
            else:  # entropy_bonus
                test_obj = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.0, entropy_coeff=0.02)
                loss_dict = test_obj.compute_loss(test_policy, None, batch, torch.zeros_like(advantages))
                loss_dict[component].backward()
            
            # Store gradients
            component_grads[component] = {
                name: param.grad.clone() for name, param in test_policy.named_parameters()
            }
        
        # Compute total loss gradients
        for param in policy.parameters():
            param.grad = None
        
        loss_dict = objective.compute_loss(policy, reference_policy, batch, advantages)
        loss_dict["total_loss"].backward()
        
        # Check that total gradients approximately equal sum of components
        for name, param in policy.named_parameters():
            total_grad = param.grad
            expected_grad = (component_grads["policy_loss"][name] + 
                           component_grads["kl_penalty"][name] - 
                           component_grads["entropy_bonus"][name])  # minus because entropy is bonus
            
            assert torch.allclose(total_grad, expected_grad, atol=1e-6), \
                f"Total gradient for {name} doesn't match sum of components"
    
    def test_trainer_gradient_accumulation(self):
        """Test that trainer properly accumulates and applies gradients."""
        policy = GradientTestPolicy()
        reward_model = VariableRewardModel()
        optimizer = SGD(policy.parameters(), lr=0.1)
        config = GRPOConfig(group_size=2, batch_size=4, kl_coeff=0.1)
        
        trainer = GRPOTrainer(policy, reward_model, optimizer, config)
        
        # Store initial parameters
        initial_params = {name: param.detach().clone() 
                         for name, param in policy.named_parameters()}
        
        # Training step
        states = ["state1", "state2"]
        metrics = trainer.train_step(states)
        
        # Check parameters changed
        for name, param in policy.named_parameters():
            assert not torch.allclose(param, initial_params[name]), \
                f"Parameter {name} was not updated by trainer"
        
        # Check that expected metrics are present (trainer uses epoch_0_* format)
        assert "step_loss" in metrics, "Should have computed step loss"
        assert "grad_norm" in metrics, "Should have computed gradient norm"
        # Check for epoch-specific metrics
        epoch_loss_keys = [k for k in metrics.keys() if k.startswith("epoch_0_") and "loss" in k]
        assert len(epoch_loss_keys) > 0, f"Should have epoch loss metrics, got keys: {list(metrics.keys())}"
    
    def test_gradient_clipping_effect(self):
        """Test that gradient clipping works correctly."""
        policy = GradientTestPolicy()
        reward_model = VariableRewardModel()
        
        # Create large gradients by using large advantages
        collector = ExperienceCollector(policy, reward_model, group_size=2)
        batch = collector.collect_experiences(["state1", "state2"])
        large_advantages = torch.tensor([10.0, -10.0, 5.0, -5.0])
        
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.0, entropy_coeff=0.0)
        
        # Compute loss and gradients
        for param in policy.parameters():
            param.grad = None
        
        loss_dict = objective.compute_loss(policy, None, batch, large_advantages)
        loss_dict["policy_loss"].backward()
        
        # Check gradient norm before clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'))
        
        # Create a fresh policy with same parameters for second test
        policy2 = GradientTestPolicy()
        policy2.load_state_dict(policy.state_dict())
        
        # Compute gradients again on fresh policy
        for param in policy2.parameters():
            param.grad = None
        
        loss_dict2 = objective.compute_loss(policy2, None, batch, large_advantages)
        loss_dict2["policy_loss"].backward()
        
        # Apply clipping
        max_norm = 1.0
        grad_norm_after = torch.nn.utils.clip_grad_norm_(policy2.parameters(), max_norm)
        
        # Check clipping worked
        if grad_norm_before > max_norm:
            assert grad_norm_after <= max_norm + 1e-6, "Gradient clipping failed"
            
            # Verify individual gradients are scaled
            total_norm = 0.0
            for param in policy2.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            assert abs(total_norm - min(grad_norm_before, max_norm)) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__]) 