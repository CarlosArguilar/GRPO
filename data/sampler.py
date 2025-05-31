from abc import ABC, abstractmethod
from typing import List, Any, Iterator, Optional
import torch


class DataSampler(ABC):
    """Abstract base class for data samplers that provide states for GRPO training."""
    
    @abstractmethod
    def sample_batch(self, batch_size: int) -> List[Any]:
        """
        Sample a batch of states.
        
        Args:
            batch_size: Number of states to sample
            
        Returns:
            List of sampled states
        """
        raise NotImplementedError
    
    @abstractmethod
    def __iter__(self) -> Iterator[List[Any]]:
        """Return iterator over batches of states."""
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        """Return total number of available states."""
        raise NotImplementedError


class SimpleDataSampler(DataSampler):
    """Simple implementation of DataSampler for basic use cases."""
    
    def __init__(
        self, 
        data: List[Any], 
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the simple data sampler.
        
        Args:
            data: List of states/inputs
            batch_size: Default batch size
            shuffle: Whether to shuffle data
            seed: Random seed for reproducibility
        """
        pass
    
    def sample_batch(self, batch_size: int) -> List[Any]:
        """Sample a batch of states."""
        pass
    
    def __iter__(self) -> Iterator[List[Any]]:
        """Return iterator over batches of states."""
        pass
    
    def __len__(self) -> int:
        """Return total number of available states."""
        pass
    
    def reset(self) -> None:
        """Reset the sampler state."""
        pass 