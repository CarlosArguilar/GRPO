from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
import torch
from torch import Tensor


class PolicyModel(ABC):
    """Abstract base class for policy models used in GRPO training - model agnostic."""
    
    @abstractmethod
    def generate_actions(
        self, 
        states: List[Any], 
        num_actions_per_state: int,
        **generation_kwargs: Any
    ) -> List[List[Any]]:
        """
        Generate multiple actions for each state/input.
        
        Args:
            states: List of input states/observations
            num_actions_per_state: Number of actions to generate per state
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of lists, where each inner list contains actions for one state
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_log_probabilities(
        self, 
        states: List[Any], 
        actions: List[Any]
    ) -> Tensor:
        """
        Calculate log probabilities of actions given states.
        
        Args:
            states: List of input states/observations
            actions: List of corresponding actions
            
        Returns:
            Tensor of log probabilities for each (state, action) pair
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Tensor]:
        """Get model parameters for optimization."""
        raise NotImplementedError
    
    @abstractmethod
    def train(self) -> None:
        """Set model to training mode."""
        raise NotImplementedError
    
    @abstractmethod
    def eval(self) -> None:
        """Set model to evaluation mode."""
        raise NotImplementedError
    
    @abstractmethod
    def to(self, device: torch.device) -> 'PolicyModel':
        """Move model to specified device."""
        raise NotImplementedError
