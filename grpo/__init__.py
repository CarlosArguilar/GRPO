"""
GRPO (Group Relative Policy Optimization) Framework

A model-agnostic implementation of Group Relative Policy Optimization
for reinforcement learning training.
"""

from .policy_base import PolicyModel, validate_policy_model, check_policy_model
from .reward_model_base import RewardModel, validate_reward_model, check_reward_model
from .experience_collector import Experience, ExperienceBatch, ExperienceCollector
from .advantage_estimator import AdvantageEstimator
from .grpo_objective import GRPOObjective
from .trainer import GRPOConfig, GRPOTrainer

__all__ = [
    "PolicyModel",
    "RewardModel", 
    "Experience",
    "ExperienceBatch",
    "ExperienceCollector",
    "AdvantageEstimator",
    "GRPOObjective",
    "GRPOConfig",
    "GRPOTrainer",
    "validate_policy_model",
    "validate_reward_model", 
    "check_policy_model",
    "check_reward_model",
]

__version__ = "0.1.0" 