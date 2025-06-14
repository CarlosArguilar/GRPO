"""Optimized GRPO Trainer implementation."""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Callable, Iterable
import time
import logging
import copy

import torch
from torch import Tensor, nn
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader

from .policy_base import PolicyModel
from .reward_model_base import RewardModel
from .experience_collector import ExperienceCollector
from .advantage_estimator import AdvantageEstimator
from .grpo_objective import GRPOObjective

# Import validation functions from base modules
from .policy_base import check_policy_model
from .reward_model_base import check_reward_model


class GRPOConfig:
    """Configuration class for GRPO training with validation."""
    
    def __init__(
        self,
        group_size: int = 64,
        batch_size: int = 1024,
        learning_rate: float = 1e-6,
        num_epochs_per_update: int = 1,
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.04,
        entropy_coeff: float = 0.0,
        normalize_advantages: bool = True,
        max_grad_norm: Optional[float] = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        reference_update_freq: int = 1,
        checkpoint_freq: int = 100
    ) -> None:
        # Parameter validation
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if num_epochs_per_update <= 0:
            raise ValueError("num_epochs_per_update must be positive")
        if clip_epsilon <= 0:
            raise ValueError("clip_epsilon must be positive")
        if kl_coeff < 0:
            raise ValueError("kl_coeff cannot be negative")
        if entropy_coeff < 0:
            raise ValueError("entropy_coeff cannot be negative")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if reference_update_freq <= 0:
            raise ValueError("reference_update_freq must be positive")
        if checkpoint_freq <= 0:
            raise ValueError("checkpoint_freq must be positive")

        # Assign validated parameters
        self.group_size = group_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs_per_update = num_epochs_per_update
        self.clip_epsilon = clip_epsilon
        self.kl_coeff = kl_coeff
        self.entropy_coeff = entropy_coeff
        self.normalize_advantages = normalize_advantages
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.reference_update_freq = reference_update_freq
        self.checkpoint_freq = checkpoint_freq


class GRPOTrainer:
    """Optimized GRPO trainer with multi-epoch support and checkpointing."""
    
    def __init__(
        self,
        policy_model: PolicyModel,
        reward_model: RewardModel,
        optimizer: Optional[Optimizer] = None,
        config: Optional[GRPOConfig] = None,
        reference_policy: Optional[PolicyModel] = None
    ) -> None:
        # Validate models implement required protocols
        check_policy_model(policy_model, raise_on_invalid=True)
        check_reward_model(reward_model, raise_on_invalid=True)
        if reference_policy is not None:
            check_policy_model(reference_policy, raise_on_invalid=True)
        
        # Default config if not provided
        self.config = config or GRPOConfig()
        
        # Device setup
        self.device = torch.device(self.config.device)
        self.policy = policy_model.to(self.device)
        self.reward_model = reward_model.to(self.device)
        
        # Optimizer setup
        if optimizer is None:
            self.optimizer = AdamW(self.policy.parameters(), lr=self.config.learning_rate)
        else:
            self.optimizer = optimizer
        
        # Reference policy handling
        self.reference_policy = reference_policy
        if self.reference_policy is None:
            self._init_reference_policy()
        else:
            self.reference_policy = self.reference_policy.to(self.device)
            for p in self.reference_policy.parameters():
                p.requires_grad_(False)
        
        # Helper modules
        self.collector = ExperienceCollector(
            policy_model=self.policy,
            reward_model=self.reward_model,
            group_size=self.config.group_size,
        )
        
        self.advantage_estimator = AdvantageEstimator(
            normalize_advantages=self.config.normalize_advantages
        )
        
        self.objective = GRPOObjective(
            clip_epsilon=self.config.clip_epsilon,
            kl_coeff=self.config.kl_coeff,
            entropy_coeff=self.config.entropy_coeff,
        )
        
        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.history: List[Dict[str, float]] = []
        self.best_loss = float('inf')
        
        # Logging setup
        self.logger = logging.getLogger("GRPOTrainer")
        self.logger.setLevel(logging.INFO)
        
    def _init_reference_policy(self) -> None:
        """Initialize reference policy with current policy weights."""
        self.reference_policy = copy.deepcopy(self.policy)
        self.reference_policy = self.reference_policy.to(self.device)
        self.reference_policy.eval()
        for p in self.reference_policy.parameters():
            p.requires_grad_(False)
    
    def train_step(
        self, 
        states: List[Any],
        **generation_kwargs: Any
    ) -> Dict[str, float]:
        """Optimized training step with multi-epoch support."""
        # Timing and metrics
        step_start = time.time()
        metrics = {"step": self.step_count}
        
        # ------------------------------------------------------------------
        # 1. Experience Collection (no gradients)
        # ------------------------------------------------------------------
        collection_start = time.time()
        with torch.no_grad():
            batch = self.collector.collect_experiences(states, **generation_kwargs)
        metrics["collection_time"] = time.time() - collection_start
        
        # ------------------------------------------------------------------
        # 2. Advantage Computation (no gradients)
        # ------------------------------------------------------------------
        advantage_start = time.time()
        advantages = self.advantage_estimator.compute_advantages(batch)
        metrics["advantage_time"] = time.time() - advantage_start
        
        # ------------------------------------------------------------------
        # 3. Multi-epoch Optimization Loop
        # ------------------------------------------------------------------
        total_samples = sum(len(g) for g in batch.action_groups)
        metrics["total_samples"] = total_samples
        
        for epoch in range(self.config.num_epochs_per_update):
            self.epoch_count += 1
            
            # Forward pass and loss computation
            self.optimizer.zero_grad(set_to_none=True)
            
            loss_dict = self.objective.compute_loss(
                current_policy=self.policy,
                reference_policy=self.reference_policy,
                experience_batch=batch,
                advantages=advantages,
            )
            
            # Backpropagation
            loss_dict["total_loss"].backward()
            
            # Gradient clipping
            if self.config.max_grad_norm is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.config.max_grad_norm
                )
                metrics["grad_norm"] = grad_norm.item()
            
            # Optimization step
            self.optimizer.step()
            
            # Update metrics
            for k, v in loss_dict.items():
                metrics[f"epoch_{epoch}_{k}"] = v.item()
            
            # Update best loss tracking
            current_loss = loss_dict["total_loss"].item()
            if current_loss < self.best_loss:
                self.best_loss = current_loss
        
        # ------------------------------------------------------------------
        # 4. Reference Policy Update (periodic)
        # ------------------------------------------------------------------
        if self.step_count % self.config.reference_update_freq == 0:
            self._update_reference_policy()
            metrics["reference_updated"] = 1.0
        
        # ------------------------------------------------------------------
        # 5. Additional Metrics
        # ------------------------------------------------------------------
        metrics["step_time"] = time.time() - step_start
        metrics["advantage_mean"] = advantages.mean().item()
        metrics["advantage_std"] = advantages.std().item()
        metrics["step_loss"] = current_loss
        
        # Store in history and return
        self.history.append(metrics)
        self.step_count += 1
        return metrics
    
    def train(
        self,
        data_loader: Iterable[List[Any]],
        total_steps: int,
        log_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        checkpoint_dir: Optional[str] = None
    ) -> List[Dict[str, float]]:
        """
        Run full GRPO training loop with checkpointing.
        
        Args:
            data_loader: Data loader providing batches of states
            total_steps: Total training steps to perform
            log_callback: Optional callback for logging metrics
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            List of training metrics for each step
        """
        self.logger.info("Starting GRPO training for %d steps", total_steps)
        start_time = time.time()
        
        step = 0
        while step < total_steps:
            try:
                states_batch = next(data_loader)
                if not states_batch:
                    continue
                
                metrics = self.train_step(states_batch)
                
                # Logging
                if log_callback:
                    log_callback(step, metrics)
                else:
                    self._log_metrics(step, metrics)
                
                # Checkpointing
                if checkpoint_dir and (step % self.config.checkpoint_freq == 0):
                    self.save_checkpoint(
                        f"{checkpoint_dir}/checkpoint_step{step}.pt"
                    )
                
                step += 1
                
            except StopIteration:
                self.logger.info("Data loader exhausted. Resetting...")
                data_loader = iter(data_loader)
        
        # Final checkpoint
        if checkpoint_dir:
            self.save_checkpoint(f"{checkpoint_dir}/final_checkpoint.pt")
        
        self.logger.info(
            "Training completed in %.2f seconds", 
            time.time() - start_time
        )
        return self.history
    
    def _update_reference_policy(self) -> None:
        """Efficient reference policy update without full copy."""
        # Update reference policy parameters in-place
        with torch.no_grad():
            for ref_param, current_param in zip(
                self.reference_policy.parameters(), 
                self.policy.parameters()
            ):
                ref_param.copy_(current_param)
    
    def _log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Default metric logging implementation."""
        log_str = f"Step {step}: "
        log_str += f"Loss = {metrics.get('step_loss', 0.0):.4f}, "
        log_str += f"Time = {metrics.get('step_time', 0.0):.2f}s, "
        log_str += f"Grad Norm = {metrics.get('grad_norm', 0.0):.2f}"
        self.logger.info(log_str)
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save comprehensive training checkpoint."""
        checkpoint = {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'policy_state_dict': self.policy.state_dict(),
            'reference_state_dict': self.reference_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'config': self.config.__dict__,
        }
        torch.save(checkpoint, filepath)
        self.logger.info("Saved checkpoint to %s", filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint with validation."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Load state
        self.step_count = checkpoint['step_count']
        self.epoch_count = checkpoint['epoch_count']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        
        # Load models
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.reference_policy.load_state_dict(checkpoint['reference_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Move models to current device
        self.policy = self.policy.to(self.device)
        self.reference_policy = self.reference_policy.to(self.device)
        
        self.logger.info("Loaded checkpoint from %s", filepath)