from typing import List, Dict, Any, Optional, Union
import torch
import pickle
import json
from pathlib import Path


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested list.
    
    Args:
        nested_list: List of lists to flatten
        
    Returns:
        Flattened list
    """
    pass


def batch_list(data: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list into batches.
    
    Args:
        data: List to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    pass


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    pass


def save_object(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object to file using pickle.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    pass


def load_object(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    pass


def save_config(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save configuration dictionary to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save file
    """
    pass


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Configuration dictionary
    """
    pass


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        PyTorch device
    """
    pass


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    pass 