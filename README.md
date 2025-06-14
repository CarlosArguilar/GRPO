# GRPO: A General, Model-Agnostic Implementation of Group-Relative Policy Optimization

GRPO (Group-Relative Policy Optimization) is a powerful reinforcement learning algorithm that extends Proximal Policy Optimization (PPO) with a novel advantage estimation mechanism. This repository provides a model-agnostic implementation of GRPO that decouples the algorithm core from specific model architectures, enabling seamless application to diverse domains.

## Overview

Recent advancements in reinforcement learning have led to powerful algorithms like GRPO, which have been successfully applied to enhance the capabilities of Large Language Models (LLMs). However, many existing implementations of GRPO are tightly coupled with specific LLM frameworks. This makes it sometimes non-direct to apply these powerful algorithms to other domains, such as robotics, game development, or scientific research, without significant modifications.

This project was created to address this gap. It offers a **decoupled, model-agnostic framework** for GRPO that can be easily adapted to any custom policy and reward models. By adhering to a simple set of `Protocol` interfaces, you can integrate your own models and leverage the GRPO training pipeline with minimal effort.

## What is Group-Relative Policy Optimization (GRPO)?

Group-Relative Policy Optimization is a variant of Proximal Policy Optimization (PPO). Instead of relying on a learned value function (critic) to estimate the advantage of an action, GRPO computes the advantage based on the *relative performance* of multiple actions sampled for the same state (or prompt).

For a given state, the policy generates a group of candidate actions. A reward model then scores each action. The core idea is to normalize these rewards within the group to derive an advantage signal. This approach is particularly effective when only preference-based or relative feedback is available.

### Mathematical Formulation

The GRPO objective aims to maximize a clipped surrogate objective function, regularized by a KL-divergence term to prevent the policy from deviating too far from a reference policy. The objective function to be minimized is:

$$L = -\mathbf{E}[\min(\rho A, \hat{\rho} A)] + \lambda_{\text{KL}} \text{KL}(\pi_\theta \parallel \pi_{\text{ref}}) - \beta_{\text{ent}} \mathcal{H}(\pi_\theta)$$

Where:
-   $A_{i,j} = r_{i,j} - \bar{r}_i$ is the **group-relative advantage** for action $j$ in state $i$, where $\bar{r}_i$ is the mean reward for the group of actions sampled for state $i$.
-   $\rho = \exp(\ell - \ell_0)$ is the likelihood ratio between the current policy $\pi_\theta$ and the old policy $\pi_{\text{old}}$.
-   $\hat{\rho} = \text{clip}(\rho, 1-\varepsilon_{\text{clip}}, 1+\varepsilon_{\text{clip}})$ is the clipped likelihood ratio.
-   $\text{KL}(\pi_\theta \parallel \pi_{\text{ref}})$ is the KL divergence between the current policy and a reference policy.
-   $\mathcal{H}(\pi_\theta)$ is the entropy bonus to encourage exploration.

This method was introduced in the following paper, which we recommend for a deeper understanding:

> Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv preprint arXiv:2402.03300.

## Project Structure

The core logic of the GRPO implementation is located in the `grpo/` directory. The structure is designed to be modular and easy to understand:

-   `policy_base.py`: Defines the `PolicyModel` protocol. Your custom policy model must implement this interface.
-   `reward_model_base.py`: Defines the `RewardModel` protocol. Your custom reward model must implement this interface.
-   `experience_collector.py`: Handles the rollout phase, collecting actions from the policy and rewards from the reward model.
-   `advantage_estimator.py`: Computes the group-relative advantages from the collected experiences.
-   `grpo_objective.py`: Implements the GRPO loss function.
-   `trainer.py`: Orchestrates the entire training process, tying all the components together.

For a more in-depth explanation of each component, please refer to the detailed `README.md` inside the `grpo/` directory.

## Getting Started: A Minimalistic Example

Here is a minimalistic example of how to use this framework.

### 1. Define your Policy and Reward Models

First, create your own policy and reward models. They don't need to inherit from any specific class, but they must implement the methods defined in the `PolicyModel` and `RewardModel` protocols.

```python
import torch
from typing import List, Any, Dict

# A dummy Policy Model
class MyPolicy:
    def generate_actions(self, states: List[Any], num_actions_per_state: int, **kwargs) -> List[List[Any]]:
        # In a real scenario, this would generate actions based on states
        return [[f"action_{i}" for i in range(num_actions_per_state)] for _ in states]

    def get_log_probabilities(self, states: List[Any], actions: List[Any]) -> torch.Tensor:
        # Dummy log probabilities
        return torch.randn(len(states))

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        # In a real scenario, this would return the model's parameters
        return {}
    
    def train(self): pass
    def eval(self): pass
    def to(self, device): return self

# A dummy Reward Model
class MyRewardModel:
    def compute_rewards(self, states: List[Any], actions: List[Any]) -> torch.Tensor:
        # Dummy rewards
        return torch.randn(len(states))

    def train(self): pass
    def eval(self): pass
    def to(self, device): return self

```

### 2. Set up and Run the Trainer

Instantiate your models and the `GRPOTrainer`, then start the training loop.

```python
import torch
import copy
from grpo import GRPOTrainer, GRPOConfig

# 1. Instantiate your models
policy = MyPolicy()
reward_model = MyRewardModel()
# The reference policy is often a snapshot of the initial policy
reference_policy = copy.deepcopy(policy)

# 2. Define training configuration
config = GRPOConfig(
    group_size=4,
    batch_size=2,
    num_epochs_per_update=2
)

# 3. Instantiate the trainer
trainer = GRPOTrainer(
    config=config,
    policy=policy,
    reference_policy=reference_policy,
    reward_model=reward_model,
    optimizer=None # Pass a torch.optim.Optimizer in a real use case
)

# 4. Prepare some dummy data
states = ["state_1", "state_2", "state_3", "state_4"]

# 5. Run a training step
# In a real training loop, you would call this repeatedly
metrics = trainer.step(states)

print("Training step completed. Metrics:")
print(metrics)
```

This example demonstrates the core workflow. You provide the states, and the `GRPOTrainer` handles the experience collection, advantage estimation, and optimization.

## Examples

This repository includes a practical example demonstrating the use of GRPO in a dynamic environment.

-   **Dynamic Arm Bandit**: The `examples/grpo_dynamic_bandit_demo.ipynb` notebook shows how to use GRPO to solve a multi-armed bandit problem where the rewards from the arms change over time.

## References

-   Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv preprint arXiv:2402.03300.
