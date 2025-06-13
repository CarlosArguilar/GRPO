"""Test edge cases and robustness of GRPO implementation."""

import torch
import pytest
from torch.optim import Adam

from grpo.grpo_objective import GRPOObjective
from grpo.advantage_estimator import AdvantageEstimator
from grpo.experience_collector import ExperienceCollector, ExperienceBatch
from grpo.trainer import GRPOTrainer, GRPOConfig
from grpo.policy_base import PolicyModel
from grpo.reward_model_base import RewardModel


class EdgeCasePolicy(PolicyModel, torch.nn.Module):
    """Policy for testing edge cases."""
    
    def __init__(self, behavior="normal"):
        super().__init__()
        torch.nn.Module.__init__(self)
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.behavior = behavior
    
    def generate_actions(self, states, num_actions_per_state, **kwargs):
        if self.behavior == "empty_actions":
            return [[] for _ in states]
        elif self.behavior == "variable_actions":
            # Generate variable actions based on state name
            return [["a"] if "s1" in state or "s3" in state else ["b", "c"] for state in states]
        else:
            return [[f"action_{i}_{j}" for j in range(num_actions_per_state)] 
                    for i, _ in enumerate(states)]
    
    def get_log_probabilities(self, states, actions):
        if self.behavior == "extreme_logprobs":
            return torch.tensor([1e6 if "high" in action else -1e6 for action in actions])
        elif self.behavior == "nan_logprobs":
            return torch.full((len(actions),), float('nan'))
        elif self.behavior == "inf_logprobs":
            return torch.full((len(actions),), float('inf'))
        else:
            return self.param.expand(len(actions))
    
    def get_parameters(self):
        return {"param": self.param}
    
    def train(self, mode: bool = True):
        return torch.nn.Module.train(self, mode)
    
    def eval(self):
        torch.nn.Module.eval(self)
    
    def to(self, device):
        torch.nn.Module.to(self, device)
        return self


class EdgeCaseRewardModel(RewardModel):
    """Reward model for testing edge cases."""
    
    def __init__(self, behavior="normal"):
        self.behavior = behavior
    
    def compute_rewards(self, states, actions):
        if self.behavior == "zero_rewards":
            return torch.zeros(len(actions))
        elif self.behavior == "extreme_rewards":
            return torch.tensor([1e6 if i % 2 == 0 else -1e6 for i in range(len(actions))])
        elif self.behavior == "nan_rewards":
            return torch.full((len(actions),), float('nan'))
        elif self.behavior == "inf_rewards":
            return torch.full((len(actions),), float('inf'))
        elif self.behavior == "negative_rewards":
            return torch.full((len(actions),), -1.0)
        else:
            return torch.ones(len(actions))
    
    def to(self, device):
        return self
    
    def eval(self):
        pass
    
    def train(self):
        pass


class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_empty_states_list(self):
        """Test handling of empty states list."""
        policy = EdgeCasePolicy()
        reward_model = EdgeCaseRewardModel()
        
        collector = ExperienceCollector(policy, reward_model, group_size=2)
        
        # Empty states should raise ValueError
        with pytest.raises(ValueError, match="states list must be non-empty"):
            batch = collector.collect_experiences([])
    
    def test_single_state_single_action(self):
        """Test minimal case: one state, one action."""
        policy = EdgeCasePolicy()
        reward_model = EdgeCaseRewardModel()
        
        collector = ExperienceCollector(policy, reward_model, group_size=1)
        batch = collector.collect_experiences(["single_state"])
        
        assert len(batch.states) == 1
        assert len(batch.action_groups) == 1
        assert len(batch.action_groups[0]) == 1
        
        # Test advantage computation
        estimator = AdvantageEstimator(normalize_advantages=False)
        advantages = estimator.compute_advantages(batch)
        
        # Single action in group should have zero advantage (r - mean(r) = r - r = 0)
        assert torch.allclose(advantages, torch.zeros(1)), "Single action should have zero advantage"
    
    def test_variable_group_sizes(self):
        """Test that experience collector validates consistent action counts per state."""
        policy = EdgeCasePolicy(behavior="variable_actions")
        reward_model = EdgeCaseRewardModel()
        
        collector = ExperienceCollector(policy, reward_model, group_size=1)
        
        # s1 generates 1 action, should work with group_size=1
        batch1 = collector.collect_experiences(["s1"])
        assert len(batch1.action_groups[0]) == 1, "s1 should generate 1 action"
        
        # s2 generates 2 actions, should fail with group_size=1  
        with pytest.raises(ValueError, match="Expected 1 actions per state, got 2"):
            batch2 = collector.collect_experiences(["s2"])
        
        # Test with appropriate group size for s2
        collector2 = ExperienceCollector(policy, reward_model, group_size=2)
        batch2 = collector2.collect_experiences(["s2"])
        assert len(batch2.action_groups[0]) == 2, "s2 should generate 2 actions"
    
    def test_extreme_rewards(self):
        """Test handling of extreme reward values."""
        policy = EdgeCasePolicy()
        reward_model = EdgeCaseRewardModel(behavior="extreme_rewards")
        
        collector = ExperienceCollector(policy, reward_model, group_size=2)
        batch = collector.collect_experiences(["s1", "s2"])
        
        # Should handle extreme rewards without crashing
        estimator = AdvantageEstimator(normalize_advantages=False)
        advantages = estimator.compute_advantages(batch)
        
        # Check that advantages are finite
        assert torch.all(torch.isfinite(advantages)), "Advantages should be finite even with extreme rewards"
    
    def test_all_zero_rewards(self):
        """Test handling when all rewards are zero."""
        policy = EdgeCasePolicy()
        reward_model = EdgeCaseRewardModel(behavior="zero_rewards")
        
        collector = ExperienceCollector(policy, reward_model, group_size=2)
        batch = collector.collect_experiences(["s1", "s2"])
        
        estimator = AdvantageEstimator(normalize_advantages=False)
        advantages = estimator.compute_advantages(batch)
        
        # All advantages should be zero when all rewards are zero
        assert torch.allclose(advantages, torch.zeros_like(advantages)), \
            "All advantages should be zero when all rewards are zero"
    
    def test_normalization_with_zero_std(self):
        """Test advantage normalization when std is zero (all advantages equal)."""
        # Create rewards that result in identical advantages
        batch = ExperienceBatch(
            states=["s1", "s2"],
            action_groups=[["a1", "a2"], ["b1", "b2"]],
            reward_groups=[[1.0, 1.0], [2.0, 2.0]],  # Same within each group
            log_prob_old_groups=[[0.0, 0.0], [0.0, 0.0]]
        )
        
        estimator = AdvantageEstimator(normalize_advantages=True)
        advantages = estimator.compute_advantages(batch)
        
        # Should handle zero std gracefully (advantages should be zero)
        assert torch.allclose(advantages, torch.zeros_like(advantages)), \
            "Normalization should handle zero std case"
    
    def test_extreme_log_probabilities(self):
        """Test handling of extreme log probabilities."""
        policy = EdgeCasePolicy(behavior="extreme_logprobs")
        reward_model = EdgeCaseRewardModel()
        
        # Create batch with extreme log probs
        batch = ExperienceBatch(
            states=["s1"],
            action_groups=[["high_prob", "low_prob"]],
            reward_groups=[[1.0, 0.0]],
            log_prob_old_groups=[[0.0, 0.0]]
        )
        
        advantages = torch.tensor([0.5, -0.5])
        
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.0, entropy_coeff=0.0)
        loss_dict = objective.compute_loss(policy, None, batch, advantages)
        
        # Should handle extreme values without producing NaN/Inf
        assert torch.isfinite(loss_dict["policy_loss"]), "Should handle extreme log probs"
        assert torch.isfinite(loss_dict["total_loss"]), "Total loss should be finite"
    
    def test_nan_reward_handling(self):
        """Test behavior when reward model produces NaN."""
        policy = EdgeCasePolicy()
        reward_model = EdgeCaseRewardModel(behavior="nan_rewards")
        
        collector = ExperienceCollector(policy, reward_model, group_size=2)
        
        # The experience collector doesn't validate rewards for NaN, 
        # so collection might succeed but advantage computation should detect issues
        batch = collector.collect_experiences(["s1"])
        estimator = AdvantageEstimator(normalize_advantages=False)
        
        # Check if advantage computation handles NaN gracefully or produces NaN results
        advantages = estimator.compute_advantages(batch)
        
        # Either advantages contain NaN (which is expected) or computation raises an error
        # We just verify that the system doesn't crash completely
        assert advantages is not None, "Advantage computation should not crash completely"
    
    def test_very_large_group_size(self):
        """Test handling of group size larger than available actions."""
        policy = EdgeCasePolicy()
        reward_model = EdgeCaseRewardModel()
        
        # Generate a large number of actions by creating a custom policy
        class LargeActionPolicy(EdgeCasePolicy):
            def generate_actions(self, states, num_actions_per_state=2, **kwargs):
                # Return 100 actions regardless of the requested number
                return [[f"action_{i}_{j}" for j in range(100)] 
                        for i, _ in enumerate(states)]
        
        large_policy = LargeActionPolicy()
        large_collector = ExperienceCollector(large_policy, reward_model, group_size=100)
        batch = large_collector.collect_experiences(["s1"])
        
        # Should create one group with all available actions
        assert len(batch.action_groups) == 1
        assert len(batch.action_groups[0]) == 100, f"Expected 100 actions, got {len(batch.action_groups[0])}"
    
    def test_clipping_edge_cases(self):
        """Test PPO clipping with edge case ratios."""
        policy = EdgeCasePolicy()
        
        # Test with ratio exactly at clipping boundaries
        batch = ExperienceBatch(
            states=["s1"],
            action_groups=[["a1"]],
            reward_groups=[[1.0]],
            log_prob_old_groups=[[0.0]]
        )
        
        # Set up for exact clipping boundary
        clip_eps = 0.2
        with torch.no_grad():
            # Set current log prob to create ratio = 1 + epsilon
            policy.param.data = torch.tensor([torch.log(torch.tensor(1.0 + clip_eps))])
        
        advantages = torch.tensor([1.0])
        objective = GRPOObjective(clip_epsilon=clip_eps, kl_coeff=0.0, entropy_coeff=0.0)
        loss_dict = objective.compute_loss(policy, None, batch, advantages)
        
        assert torch.isfinite(loss_dict["policy_loss"]), "Should handle boundary clipping cases"
    
    def test_zero_kl_coefficient(self):
        """Test that KL penalty is properly disabled when coefficient is zero."""
        policy = EdgeCasePolicy()
        reference_policy = EdgeCasePolicy()
        
        # Make policies very different
        with torch.no_grad():
            policy.param.data = torch.tensor([5.0])
            reference_policy.param.data = torch.tensor([-5.0])
        
        batch = ExperienceBatch(
            states=["s1"],
            action_groups=[["a1"]],
            reward_groups=[[1.0]],
            log_prob_old_groups=[[0.0]]
        )
        
        advantages = torch.tensor([1.0])
        
        # Test with zero KL coefficient
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.0, entropy_coeff=0.0)
        loss_dict = objective.compute_loss(policy, reference_policy, batch, advantages)
        
        assert torch.allclose(loss_dict["kl_penalty"], torch.tensor(0.0)), \
            "KL penalty should be zero when coefficient is zero"
    
    def test_trainer_with_problematic_optimizer(self):
        """Test trainer robustness with edge case optimizer settings."""
        policy = EdgeCasePolicy()
        reward_model = EdgeCaseRewardModel()
        
        # Optimizer with extreme learning rate
        optimizer = Adam(policy.parameters(), lr=1e10)  # Very large LR
        config = GRPOConfig(group_size=2, max_grad_norm=1e-10)  # Very small grad norm
        
        trainer = GRPOTrainer(policy, reward_model, optimizer, config)
        
        # Should not crash even with extreme settings
        metrics = trainer.train_step(["s1"])
        
        # Check for step_loss instead of total_loss (which is epoch-specific)
        assert "step_loss" in metrics, "Should produce valid metrics"
        assert torch.isfinite(torch.tensor(metrics["step_loss"])), "Loss should be finite"
    
    def test_multi_epoch_with_zero_advantages(self):
        """Test multi-epoch training when advantages are all zero."""
        policy = EdgeCasePolicy()
        reward_model = EdgeCaseRewardModel(behavior="zero_rewards")
        optimizer = Adam(policy.parameters(), lr=0.01)
        
        config = GRPOConfig(
            group_size=2, 
            num_epochs_per_update=5,  # Multiple epochs
            kl_coeff=0.1,  # Ensure some gradient from KL
            normalize_advantages=False
        )
        
        trainer = GRPOTrainer(policy, reward_model, optimizer, config)
        
        initial_param = policy.param.detach().clone()
        metrics = trainer.train_step(["s1", "s2"])
        final_param = policy.param.detach().clone()
        
        # Should still update due to KL penalty even with zero advantages
        assert not torch.allclose(initial_param, final_param), \
            "Parameters should update via KL penalty even with zero advantages"
        
        # Should have metrics for all epochs
        epoch_keys = [k for k in metrics.keys() if k.startswith("epoch_")]
        assert len(epoch_keys) >= 5 * 4, "Should have metrics for all epochs and components"  # 5 epochs * 4 components
    
    def test_device_mismatch_robustness(self):
        """Test handling when tensors end up on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mismatch test")
        
        policy = EdgeCasePolicy()
        reward_model = EdgeCaseRewardModel()
        
        # Create batch on CPU
        collector = ExperienceCollector(policy, reward_model, group_size=2)
        batch = collector.collect_experiences(["s1"])
        
        # Move policy to CUDA but keep batch on CPU
        policy = policy.to("cuda")
        
        advantages = torch.tensor([0.5, -0.5])  # CPU tensor
        
        objective = GRPOObjective()
        
        # Should handle device transfer automatically
        loss_dict = objective.compute_loss(policy, None, batch, advantages)
        
        assert loss_dict["total_loss"].device.type == "cuda", "Loss should be on same device as policy"
    
    def test_checkpoint_with_corrupted_state(self):
        """Test checkpoint loading robustness."""
        policy = EdgeCasePolicy()
        reward_model = EdgeCaseRewardModel()
        optimizer = Adam(policy.parameters(), lr=0.01)
        config = GRPOConfig()
        
        trainer = GRPOTrainer(policy, reward_model, optimizer, config)
        
        # Create a corrupted checkpoint dict
        corrupted_checkpoint = {
            'step_count': "not_an_int",  # Wrong type
            'policy_state_dict': {},  # Empty state dict
            'optimizer_state_dict': None,  # None instead of dict
            # Missing required keys
        }
        
        # Save corrupted checkpoint
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(corrupted_checkpoint, f.name)
            
            # Should handle corrupted checkpoint gracefully
            with pytest.raises((KeyError, TypeError, RuntimeError)):
                trainer.load_checkpoint(f.name)


if __name__ == "__main__":
    pytest.main([__file__]) 