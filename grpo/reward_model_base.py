from typing import List, Any, Union, Protocol, Tuple
import torch
from torch import Tensor


class RewardModel(Protocol):
    """Protocol for reward models used in GRPO training - model agnostic.
    
    This protocol defines the interface that reward models should implement.
    Models don't need to inherit from this class - they just need to implement
    these methods (duck typing).
    """
    
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
        ...
    
    def to(self, device: torch.device) -> 'RewardModel':
        """Move model to specified device."""
        ...
    
    def eval(self) -> None:
        """Set model to evaluation mode."""
        ...
    
    def train(self) -> None:
        """Set model to training mode."""
        ...


def validate_reward_model(model: Any) -> Tuple[bool, List[str]]:
    """
    Validate that a model implements the RewardModel protocol.
    
    Args:
        model: The model to validate
        
    Returns:
        Tuple of (is_valid, list_of_missing_methods)
    """
    required_methods = [
        'compute_rewards',
        'to',
        'eval', 
        'train'
    ]
    
    missing_methods = []
    for method_name in required_methods:
        if not hasattr(model, method_name):
            missing_methods.append(method_name)
        elif not callable(getattr(model, method_name)):
            missing_methods.append(f"{method_name} (not callable)")
    
    return len(missing_methods) == 0, missing_methods


def check_reward_model(model: Any, raise_on_invalid: bool = True) -> bool:
    """
    Check if a model implements RewardModel protocol, with helpful error messages.
    
    Args:
        model: The model to check
        raise_on_invalid: Whether to raise an exception if invalid
        
    Returns:
        True if valid, False otherwise (if raise_on_invalid=False)
        
    Raises:
        TypeError: If model is invalid and raise_on_invalid=True
    """
    is_valid, missing_methods = validate_reward_model(model)
    
    if not is_valid:
        error_msg = (
            f"Model {type(model).__name__} does not implement RewardModel protocol.\n"
            f"Missing methods: {', '.join(missing_methods)}\n"
            f"Required methods: compute_rewards, to, eval, train"
        )
        if raise_on_invalid:
            raise TypeError(error_msg)
        else:
            print(f"Warning: {error_msg}")
            return False
    
    return True 