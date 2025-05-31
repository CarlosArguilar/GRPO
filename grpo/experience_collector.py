from typing import List, Dict, Any, Tuple, NamedTuple
import torch
from torch import Tensor
from .policy_base import PolicyModel
from .reward_model_base import RewardModel


class Experience(NamedTuple):
    """Single experience tuple for GRPO."""
    state: Any
    action: Any
    reward: float
    log_prob_old: float


class ExperienceBatch(NamedTuple):
    """Batch of experiences grouped by state."""
    states: List[Any]
    action_groups: List[List[Any]]  # Groups of actions per state
    reward_groups: List[List[float]]  # Groups of rewards per state
    log_prob_old_groups: List[List[float]]  # Groups of old log probs per state


class ExperienceCollector:
    """Collects experiences for GRPO training by sampling actions and computing rewards."""
    
    def __init__(
        self,
        policy_model: PolicyModel,
        reward_model: RewardModel,
        group_size: int = 64
    ) -> None:
        """
        Initialize the experience collector.
        
        Args:
            policy_model: The policy model to collect experiences from
            reward_model: The reward model to evaluate actions
            group_size: Number of actions to sample per state
        """
        pass
    
    def collect_experiences(
        self, 
        states: List[Any],
        **generation_kwargs: Any
    ) -> ExperienceBatch:
        """
        Collect a batch of experiences for the given states.
        
        Args:
            states: List of input states to collect experiences for
            **generation_kwargs: Additional parameters for action generation
            
        Returns:
            ExperienceBatch containing grouped experiences
        """
        pass
    
    def _sample_actions_and_log_probs(
        self, 
        states: List[Any],
        **generation_kwargs: Any
    ) -> Tuple[List[List[Any]], List[List[float]]]:
        """Sample actions and compute their log probabilities under current policy."""
        pass
    
    def _compute_rewards(
        self, 
        states: List[Any], 
        action_groups: List[List[Any]]
    ) -> List[List[float]]:
        """Compute rewards for all state-action pairs."""
        pass 