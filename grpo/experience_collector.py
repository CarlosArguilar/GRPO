from typing import List, Any, Tuple, NamedTuple, Optional
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
        if group_size <= 0:
            raise ValueError("group_size must be positive")

        self.policy_model = policy_model
        self.reward_model = reward_model
        self.group_size = group_size

        # Internal buffers reused between method calls to minimise allocations
        self._flat_states: Optional[List[Any]] = None
        self._flat_actions: Optional[List[Any]] = None
    
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
        if len(states) == 0:
            raise ValueError("states list must be non-empty")

        # Main computation is done under a single no_grad context for memory efficiency
        with torch.no_grad():
            # 1. Generate candidate actions for each state using the current policy
            action_groups = self.policy_model.generate_actions(
                states, num_actions_per_state=self.group_size, **generation_kwargs
            )

            # 2. Flatten states/actions to leverage batched model APIs in one pass
            self._prepare_flattened_inputs(states, action_groups)

            # 3. Compute log-probabilities and rewards for every (state, action) pair
            log_probs_flat: Tensor = self.policy_model.get_log_probabilities(
                self._flat_states, self._flat_actions
            )
            rewards_flat: Tensor = self.reward_model.compute_rewards(
                self._flat_states, self._flat_actions
            )

            # 4. Validate returned tensor shapes before further processing
            self._validate_tensor(log_probs_flat, "log_probabilities")
            self._validate_tensor(rewards_flat, "rewards")

            # 5. Convert flat tensors back into per-state Python lists
            group_lengths = [len(g) for g in action_groups]
            log_prob_old_groups, reward_groups = self._unflatten_results(
                log_probs_flat, rewards_flat, group_lengths
            )

        # Clear internal buffers to prevent unnecessary memory retention
        self._flat_states = None
        self._flat_actions = None

        return ExperienceBatch(
            states=states,
            action_groups=action_groups,
            reward_groups=reward_groups,
            log_prob_old_groups=log_prob_old_groups,
        )
    
    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    def _prepare_flattened_inputs(
        self,
        states: List[Any],
        action_groups: List[List[Any]],
    ) -> None:
        """Create flattened lists of states & actions for batched model evaluation."""
        self._flat_states = []
        self._flat_actions = []

        for state, actions in zip(states, action_groups):
            if len(actions) != self.group_size:
                raise ValueError(
                    f"Expected {self.group_size} actions per state, got {len(actions)}"
                )
            self._flat_states.extend([state] * len(actions))
            self._flat_actions.extend(actions)

    def _validate_tensor(self, tensor: Tensor, name: str) -> None:
        """Ensure the given tensor is 1-D and matches the number of flat actions."""
        if tensor.ndim != 1 or tensor.shape[0] != len(self._flat_actions):
            raise ValueError(
                f"{name} must return 1-D tensor with length equal to number of actions "
                f"({len(self._flat_actions)}), got shape {tensor.shape}"
            )

    def _unflatten_results(
        self,
        log_probs_flat: Tensor,
        rewards_flat: Tensor,
        group_lengths: List[int],
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Convert flat tensors back to per-state group lists."""
        log_prob_groups: List[List[float]] = []
        reward_groups: List[List[float]] = []
        start = 0

        for length in group_lengths:
            end = start + length
            log_prob_groups.append(log_probs_flat[start:end].tolist())
            reward_groups.append(rewards_flat[start:end].tolist())
            start = end

        return log_prob_groups, reward_groups 