"""Performance and stress tests for GRPO implementation."""

import time
import torch
import pytest
from torch.optim import Adam

from grpo.grpo_objective import GRPOObjective
from grpo.advantage_estimator import AdvantageEstimator
from grpo.experience_collector import ExperienceCollector
from grpo.trainer import GRPOTrainer, GRPOConfig
from grpo.policy_base import PolicyModel
from grpo.reward_model_base import RewardModel


class BenchmarkPolicy(PolicyModel, torch.nn.Module):
    """Policy for performance testing."""
    
    def __init__(self, complexity="medium"):
        super().__init__()
        torch.nn.Module.__init__(self)
        
        if complexity == "simple":
            self.net = torch.nn.Linear(1, 1)
        elif complexity == "medium":
            self.net = torch.nn.Sequential(
                torch.nn.Linear(10, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1)
            )
        else:  # complex
            self.net = torch.nn.Sequential(
                torch.nn.Linear(100, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            )
    
    def generate_actions(self, states, num_actions_per_state, **kwargs):
        return [[f"action_{i}_{j}" for j in range(num_actions_per_state)] 
                for i, _ in enumerate(states)]
    
    def get_log_probabilities(self, states, actions):
        # Simulate computation based on network complexity
        if hasattr(self.net, '__len__'):  # Sequential
            input_size = self.net[0].in_features
        else:  # Linear
            input_size = self.net.in_features
        
        batch_input = torch.randn(len(actions), input_size)
        output = self.net(batch_input)
        return output.squeeze()
    
    def get_parameters(self):
        return {name: param for name, param in self.named_parameters()}
    
    def train(self, mode: bool = True):
        return torch.nn.Module.train(self, mode)
    
    def eval(self):
        torch.nn.Module.eval(self)
    
    def to(self, device):
        torch.nn.Module.to(self, device)
        return self


class BenchmarkRewardModel(RewardModel):
    """Reward model for performance testing."""
    
    def __init__(self, complexity="medium"):
        self.complexity = complexity
    
    def compute_rewards(self, states, actions):
        if self.complexity == "simple":
            return torch.ones(len(actions))
        elif self.complexity == "medium":
            # Simulate some computation
            rewards = []
            for i, action in enumerate(actions):
                reward = hash(action) % 1000 / 1000.0  # Deterministic but varying
                rewards.append(reward)
            return torch.tensor(rewards)
        else:  # complex
            # Simulate expensive reward computation
            rewards = torch.randn(len(actions))
            for _ in range(10):  # Multiple iterations
                rewards = torch.sin(rewards * 3.14159) + torch.cos(rewards * 2.71828)
            return rewards
    
    def to(self, device):
        return self
    
    def eval(self):
        pass
    
    def train(self):
        pass


class TestPerformanceBenchmarks:
    """Performance and stress tests."""
    
    @pytest.mark.parametrize("group_size", [4, 16, 64, 256])
    def test_scaling_with_group_size(self, group_size):
        """Test performance scaling with different group sizes."""
        policy = BenchmarkPolicy("medium")
        reward_model = BenchmarkRewardModel("medium")
        
        collector = ExperienceCollector(policy, reward_model, group_size=group_size)
        
        # Generate enough states to create multiple groups
        num_states = group_size * 3
        states = [f"state_{i}" for i in range(num_states)]
        
        start_time = time.time()
        batch = collector.collect_experiences(states)
        collection_time = time.time() - start_time
        
        # Test advantage computation
        estimator = AdvantageEstimator(normalize_advantages=True)
        start_time = time.time()
        advantages = estimator.compute_advantages(batch)
        advantage_time = time.time() - start_time
        
        # Test objective computation
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.04, entropy_coeff=0.01)
        start_time = time.time()
        loss_dict = objective.compute_loss(policy, policy, batch, advantages)
        objective_time = time.time() - start_time
        
        # Performance assertions (these are loose bounds)
        assert collection_time < 5.0, f"Collection took too long: {collection_time:.3f}s"
        assert advantage_time < 1.0, f"Advantage computation took too long: {advantage_time:.3f}s"
        assert objective_time < 2.0, f"Objective computation took too long: {objective_time:.3f}s"
        
        # Correctness checks
        total_actions = sum(len(group) for group in batch.action_groups)
        assert advantages.shape[0] == total_actions
        assert torch.isfinite(loss_dict["total_loss"])
    
    @pytest.mark.parametrize("num_states", [10, 50, 200, 500])
    def test_scaling_with_batch_size(self, num_states):
        """Test performance scaling with batch size."""
        policy = BenchmarkPolicy("medium")
        reward_model = BenchmarkRewardModel("medium")
        
        collector = ExperienceCollector(policy, reward_model, group_size=16)
        states = [f"state_{i}" for i in range(num_states)]
        
        start_time = time.time()
        batch = collector.collect_experiences(states)
        collection_time = time.time() - start_time
        
        estimator = AdvantageEstimator(normalize_advantages=True)
        start_time = time.time()
        advantages = estimator.compute_advantages(batch)
        advantage_time = time.time() - start_time
        
        # Time should scale roughly linearly (with some overhead)
        expected_max_time = num_states * 0.01 + 1.0  # 10ms per state + 1s overhead
        
        assert collection_time < expected_max_time, \
            f"Collection time {collection_time:.3f}s exceeded expected {expected_max_time:.3f}s"
        assert advantage_time < 2.0, f"Advantage time too long: {advantage_time:.3f}s"
    
    def test_memory_efficiency(self):
        """Test memory usage doesn't grow excessively."""
        policy = BenchmarkPolicy("medium")
        reward_model = BenchmarkRewardModel("medium")
        optimizer = Adam(policy.parameters(), lr=0.001)
        config = GRPOConfig(group_size=32, batch_size=128)
        
        trainer = GRPOTrainer(policy, reward_model, optimizer, config)
        
        # Monitor memory usage across multiple steps
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        states = [f"state_{i}" for i in range(100)]
        
        # Run multiple training steps
        for step in range(10):
            trainer.train_step(states)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory shouldn't grow excessively (allow 500MB increase)
            assert memory_increase < 500, \
                f"Memory increased by {memory_increase:.1f}MB at step {step}"
    
    def test_gradient_computation_efficiency(self):
        """Test that gradient computation is efficient."""
        policy = BenchmarkPolicy("complex")  # Larger model
        reward_model = BenchmarkRewardModel("simple")  # Simple rewards
        
        collector = ExperienceCollector(policy, reward_model, group_size=64)
        batch = collector.collect_experiences([f"state_{i}" for i in range(200)])
        
        estimator = AdvantageEstimator(normalize_advantages=True)
        advantages = estimator.compute_advantages(batch)
        
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.04, entropy_coeff=0.01)
        
        # Time gradient computation
        start_time = time.time()
        
        # Zero gradients
        for param in policy.parameters():
            param.grad = None
        
        loss_dict = objective.compute_loss(policy, policy, batch, advantages)
        loss_dict["total_loss"].backward()
        
        gradient_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert gradient_time < 5.0, f"Gradient computation took too long: {gradient_time:.3f}s"
        
        # Check all parameters have gradients
        param_count = 0
        grad_count = 0
        for param in policy.parameters():
            param_count += 1
            if param.grad is not None:
                grad_count += 1
        
        assert grad_count == param_count, "Not all parameters received gradients"
    
    def test_multi_epoch_performance(self):
        """Test performance with multiple optimization epochs."""
        policy = BenchmarkPolicy("medium")
        reward_model = BenchmarkRewardModel("medium")
        optimizer = Adam(policy.parameters(), lr=0.001)
        
        config = GRPOConfig(
            group_size=32,
            num_epochs_per_update=10,  # Multiple epochs
            kl_coeff=0.04,
            entropy_coeff=0.01
        )
        
        trainer = GRPOTrainer(policy, reward_model, optimizer, config)
        states = [f"state_{i}" for i in range(100)]
        
        start_time = time.time()
        metrics = trainer.train_step(states)
        total_time = time.time() - start_time
        
        # Should handle multiple epochs efficiently
        assert total_time < 10.0, f"Multi-epoch training took too long: {total_time:.3f}s"
        
        # Check we got metrics for all epochs
        epoch_metrics = [k for k in metrics.keys() if k.startswith("epoch_")]
        assert len(epoch_metrics) >= 10 * 4, "Missing epoch metrics"  # 10 epochs * 4 loss components
    
    def test_device_transfer_performance(self):
        """Test performance with GPU/CPU transfers."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device transfer test")
        
        # Test CPU -> GPU transfer efficiency
        policy_cpu = BenchmarkPolicy("medium")
        reward_model_cpu = BenchmarkRewardModel("medium")
        
        collector = ExperienceCollector(policy_cpu, reward_model_cpu, group_size=32)
        batch = collector.collect_experiences([f"state_{i}" for i in range(100)])
        
        # Move to GPU
        policy_gpu = policy_cpu.to("cuda")
        
        estimator = AdvantageEstimator(normalize_advantages=True)
        advantages = estimator.compute_advantages(batch)
        
        objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.04, entropy_coeff=0.01)
        
        # Time GPU computation
        start_time = time.time()
        loss_dict = objective.compute_loss(policy_gpu, policy_gpu, batch, advantages)
        gpu_time = time.time() - start_time
        
        # Should handle device transfers without major slowdown
        assert gpu_time < 3.0, f"GPU computation took too long: {gpu_time:.3f}s"
        assert loss_dict["total_loss"].device.type == "cuda"
    
    def test_large_vocabulary_handling(self):
        """Test performance with large action vocabularies."""
        class LargeVocabPolicy(BenchmarkPolicy):
            def generate_actions(self, states, num_actions_per_state=2, **kwargs):
                # Generate many actions per state, ignore the requested number
                return [[f"action_{i}_{j}" for j in range(100)] 
                        for i, _ in enumerate(states)]
        
        policy = LargeVocabPolicy("medium")
        reward_model = BenchmarkRewardModel("medium")
        
        collector = ExperienceCollector(policy, reward_model, group_size=100)  # Match the action count
        
        start_time = time.time()
        batch = collector.collect_experiences(["s1", "s2"])
        collection_time = time.time() - start_time
        
        # Should handle large vocabularies efficiently
        assert collection_time < 5.0, f"Large vocab collection took too long: {collection_time:.3f}s"
        
        # Verify we got the expected number of actions
        total_actions = sum(len(group) for group in batch.action_groups)
        assert total_actions == 200, f"Expected 200 actions, got {total_actions}"
    
    def test_stress_test_continuous_training(self):
        """Stress test with continuous training steps."""
        policy = BenchmarkPolicy("medium")
        reward_model = BenchmarkRewardModel("medium")
        optimizer = Adam(policy.parameters(), lr=0.001)
        config = GRPOConfig(group_size=16, num_epochs_per_update=3)
        
        trainer = GRPOTrainer(policy, reward_model, optimizer, config)
        states = [f"state_{i}" for i in range(50)]
        
        # Run many training steps
        start_time = time.time()
        
        for step in range(20):
            metrics = trainer.train_step(states)
            
            # Verify metrics are reasonable
            assert torch.isfinite(torch.tensor(metrics["step_loss"])), f"Invalid loss at step {step}"
            assert "total_samples" in metrics, f"Missing total_samples at step {step}"
        
        total_time = time.time() - start_time
        avg_time_per_step = total_time / 20
        
        # Each step should be reasonably fast
        assert avg_time_per_step < 2.0, f"Average step time too slow: {avg_time_per_step:.3f}s"
        
        # Verify training history
        assert len(trainer.history) == 20, "Training history length mismatch"
    
    def test_memory_leak_detection(self):
        """Test for potential memory leaks during training."""
        policy = BenchmarkPolicy("medium")
        reward_model = BenchmarkRewardModel("medium")
        optimizer = Adam(policy.parameters(), lr=0.001)
        config = GRPOConfig(group_size=32)
        
        trainer = GRPOTrainer(policy, reward_model, optimizer, config)
        states = [f"state_{i}" for i in range(100)]
        
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run training and force garbage collection between steps
        for step in range(15):
            trainer.train_step(states)
            
            # Clear history periodically to avoid legitimate accumulation
            if step % 5 == 0:
                trainer.history = trainer.history[-1:]  # Keep only last metric
            
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Allow some memory increase but detect major leaks
        assert memory_increase < 200, \
            f"Potential memory leak detected: {memory_increase:.1f}MB increase"


if __name__ == "__main__":
    pytest.main([__file__]) 