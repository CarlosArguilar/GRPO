from typing import List, Dict, Any, Tuple, Optional, Union, Protocol
import torch
from torch import Tensor


class PolicyModel(Protocol):
    """Protocol for policy models used in GRPO training - model agnostic.
    
    This protocol defines the interface that policy models should implement.
    Models don't need to inherit from this class - they just need to implement
    these methods (duck typing).
    """
    
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
        ...
    
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
        ...
    
    def get_parameters(self) -> Dict[str, Tensor]:
        """Get model parameters for optimization."""
        ...
    
    def train(self) -> None:
        """Set model to training mode."""
        ...
    
    def eval(self) -> None:
        """Set model to evaluation mode."""
        ...
    
    def to(self, device: torch.device) -> 'PolicyModel':
        """Move model to specified device."""
        ...


def validate_policy_model(model: Any) -> Tuple[bool, List[str]]:
    """
    Validate that a model implements the PolicyModel protocol.
    
    Args:
        model: The model to validate
        
    Returns:
        Tuple of (is_valid, list_of_missing_methods)
    """
    required_methods = [
        'generate_actions',
        'get_log_probabilities', 
        'get_parameters',
        'train',
        'eval',
        'to'
    ]
    
    missing_methods = []
    for method_name in required_methods:
        if not hasattr(model, method_name):
            missing_methods.append(method_name)
        elif not callable(getattr(model, method_name)):
            missing_methods.append(f"{method_name} (not callable)")
    
    return len(missing_methods) == 0, missing_methods


def check_policy_model(model: Any, raise_on_invalid: bool = True) -> bool:
    """
    Check if a model implements PolicyModel protocol, with helpful error messages.
    
    Args:
        model: The model to check
        raise_on_invalid: Whether to raise an exception if invalid
        
    Returns:
        True if valid, False otherwise (if raise_on_invalid=False)
        
    Raises:
        TypeError: If model is invalid and raise_on_invalid=True
    """
    is_valid, missing_methods = validate_policy_model(model)
    
    if not is_valid:
        error_msg = (
            f"Model {type(model).__name__} does not implement PolicyModel protocol.\n"
            f"Missing methods: {', '.join(missing_methods)}\n"
            f"Required methods: generate_actions, get_log_probabilities, get_parameters, train, eval, to"
        )
        if raise_on_invalid:
            raise TypeError(error_msg)
        else:
            print(f"Warning: {error_msg}")
            return False
    
    return True
