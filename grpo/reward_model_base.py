from abc import ABC, abstractmethod
from typing import List, Any, Union
import torch
from torch import Tensor


class RewardModel(ABC):
    """Abstract base class for reward models used in GRPO training - model agnostic."""
    
    @abstractmethod
    def compute_rewards(
        self, 
        states: List[Any], 
        actions: List[Any]
    ) -> Tensor:
        """
        Compute rewards for state-action pairs.
        
        Args:
            states: List of input states/observations
            actions: List of corresponding actions
            
        Returns:
            Tensor of reward scores for each (state, action) pair
        """
        raise NotImplementedError
    
    @abstractmethod
    def to(self, device: torch.device) -> 'RewardModel':
        """Move model to specified device."""
        raise NotImplementedError
    
    @abstractmethod
    def eval(self) -> None:
        """Set model to evaluation mode."""
        raise NotImplementedError 
    
    @abstractmethod
    def train(self) -> None:
        """Set model to training mode."""
        raise NotImplementedError 