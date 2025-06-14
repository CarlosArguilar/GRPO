# Group-Relative Policy Optimisation (GRPO)

This sub-package contains all the core components required to train an agent with **Group-Relative Policy Optimisation (GRPO)**.  Each module maps cleanly to a conceptual step in the GRPO pipeline, enabling the framework to remain **model-agnostic**, highly testable and easy to extend.

> **Note**  
> This document provides comprehensive coverage of all GRPO components. Each section focuses on the role, features, optimizations, and usage patterns of the core modules.

---

## 1. Experience Buffer – `experience_collector.py`

### 1.1  Role in the algorithm
1. **Rollout phase**  ‑  *sampling*:  
   For every state (prompt / observation) the current policy π_θ generates a *group* of `N = group_size` candidate actions.
2. **Scoring phase**  ‑ *reward computation*:  
   The reward model assigns a scalar reward to each sampled action.
3. **Book-keeping**:  
   The buffer stores `(state, action, reward, log π_old)` tuples in a format ready for downstream advantage estimation and GRPO objective computation.

### 1.2  Key features
• **Group semantics** – Each state keeps its own list of `group_size` actions and associated statistics; this is crucial because GRPO's clip-style objective compares *relative* performance *within* each group.  
• **Model-agnostic types** – States and actions are `Any`, allowing strings, tensors, tokens, image tensors, etc.  
• **No-grad sampling** – Uses a single `torch.no_grad()` context for both the policy and reward passes when gradients are not needed, drastically cutting memory footprint during rollouts.  
• **Single flattening pass** – Internally flattens `(state, action)` pairs once, feeding them concurrently to both policy (`get_log_probabilities`) and reward (`compute_rewards`) models.  
• **Strict validation** – Centralised helpers enforce tensor shape agreements and correct `group_size`, preventing silent shape mismatches that are hard to debug.  
• **Memory hygiene** – Temporary buffers (`_flat_states`, `_flat_actions`) are cleared after use, avoiding long-lived Python lists that would otherwise balloon across epochs.  

### 1.3  Optimisations implemented
1. **Batch fusion** – Computing log-probs and rewards in the same flattened batch avoids two separate Python loops, halving kernel launch overhead and enabling better GPU utilisation.
2. **Reusable buffers** – By reusing internal lists between calls we reduce per-iteration allocations and GC pressure, yielding a small but measurable speed-up in high-throughput environments.
3. **Early input checks** – Fast Python assertions (`if not states:` and per-state group length guards) fail early, saving expensive model calls when inputs are malformed.
4. **Shape-aware error messages** – Validation helpers report *expected vs. received* dimensions, dramatically speeding up debugging for new model integrations.

### 1.4  Data structure
```
ExperienceBatch
├── states                # List[Any], length = batch_size
├── action_groups         # List[List[Any]], shape = [batch_size, group_size]
├── reward_groups         # List[List[float]], same shape as action_groups
└── log_prob_old_groups   # List[List[float]], same shape as action_groups
```
The object is a thin `NamedTuple` – immutable, indexable and serialisable (e.g., via `pickle` or `torch.save`).  Storing **lists of Python primitives** keeps the buffer lightweight and avoids unnecessary CUDA tensors in CPU memory.

### 1.5  Typical usage snippet
```python
from grpo.experience_collector import ExperienceCollector

collector = ExperienceCollector(policy_model, reward_model, group_size=8)
states    = ["prompt1", "prompt2", ...]

batch = collector.collect_experiences(states)
# -> ready for AdvantageEstimator / GRPOObjective
```

---

## 2. Advantage Estimator – `advantage_estimator.py`

### 2.1  Role in the algorithm
Once we have sampled actions and rewards, GRPO needs a *signal* that tells each
action **how much better or worse it is** relative to its competition **within
the same state**.  That signal is the *group-relative advantage*:

$A_{i,j} = r_{i,j} - \bar r_i$

where $\bar r_i$ is the mean reward over the *group_size* actions sampled for
state $i$.  The advantage is later plugged into the clipped GRPO objective to
scale policy-gradient updates.

### 2.2  Key features
• **Per-group baselines** – Computes the mean reward for each state and
  subtracts it from every action in that group.  No information leakage across
  states.
• **Optional normalisation** – Centres and scales all advantages across the
  batch to zero mean / unit variance (variance-reduction trick).
• **One-pass algorithm** – Rewards are traversed once; advantages are stored in
  a single contiguous 1-D tensor matching the flattened action order expected
  by the GRPO objective.
• **Stateless & thread-safe** – The class holds only its hyper-parameters
  (`normalize_advantages`, `advantage_epsilon`).  Multiple estimators can run
  concurrently.

### 2.3  Optimisations implemented
1. **List → tensor conversion done once** – The nested `reward_groups`
   structure is converted to a tensor *after* the baseline subtraction so we
   allocate GPU memory exactly once.
2. **Numerical stability ε** – Prevents a divide-by-zero when all advantages are
   equal (rare but possible in degenerate reward functions).
3. **Coverage-friendly corner-case guard** – An early return for empty batches
   (unlikely in practice) is marked with `# pragma: no cover` so it doesn't
   skew test-coverage metrics.

### 2.4  Typical usage snippet
```python
from grpo.advantage_estimator import AdvantageEstimator

adv_estimator = AdvantageEstimator(normalize_advantages=True)
advantages    = adv_estimator.compute_advantages(batch)  # 1-D torch.Tensor
```

---

## 3. GRPO Objective – `grpo_objective.py`

### 3.1  Role in the algorithm
Transforms advantages into actual **parameter gradients**.  It combines:

1. **Clipped likelihood-ratio surrogate** – Same safety mechanism as PPO but
   driven by *group-relative* advantages.
2. **KL penalty** – Keeps the policy from drifting too far from a fixed
   reference (e.g. a pre-trained base LLM).
3. **Entropy bonus** – Optional exploration term.

Mathematically (minimisation form):
$$L = -\mathbf{E}[\min(\rho A, \hat{\rho} A)] + \lambda_{\text{KL}} \text{KL}(\pi_\theta \parallel \pi_{\text{ref}}) - \beta_{\text{ent}} \mathcal{H}(\pi_\theta)$$

Where:

* $\ell$ = log-probability of the selected action under the **current** policy $\pi_\theta$
* $\ell_0$ = log-probability of the same action under the **behaviour / old** policy (recorded during rollout)
* $A$     = *group-relative advantage* produced by the Advantage Estimator
* $\rho = \exp(\ell - \ell_0)$ = likelihood ratio
* $\hat{\rho} = \text{clip}(\rho, 1-\varepsilon_{\text{clip}}, 1+\varepsilon_{\text{clip}})$ = clipped likelihood ratio with hyper-parameter $\varepsilon_{\text{clip}}$
* $\lambda_{\text{KL}}$ = coefficient controlling the **KL penalty** to the reference policy $\pi_{\text{ref}}$
* $\beta_{\text{ent}}$ = coefficient for the **entropy bonus**
* $\mathcal H(\pi_\theta)$ = Shannon entropy of the current policy's action distribution
* $\mathbf{E}[\,\cdot\,]$ = expectation over the collected batch of actions

Intuitively: we *maximise* the clipped surrogate term (hence minimise its negative),
keep the policy close to $\pi_{\text{ref}}$ via the KL penalty, and optionally encourage exploration with the entropy bonus.

### 3.2  Key features
* **Device-aware tensors** – Automatically moves advantages & old log-probs to
  the policy's device.
* **Log-space computations** – Uses $\ell-\ell_0$ instead of exponentiating
  large values; improves numerical stability.
* **KL reuse** – Re-uses already computed `log_probs_current` when evaluating
  the KL term; avoids an extra forward pass.
* **Shape validation** – Fails fast if the length of advantages doesn't match
  the number of sampled actions.
* **Modular components** – Separate helpers for policy loss, KL penalty, and
  entropy bonus make maintenance straightforward.

### 3.3  Optimisations implemented
1. **Pre-allocated tensors** – Old log-probabilities converted directly into a
   contiguous tensor on the target device.
2. **Single flatten pass** – States/actions flattened once and reused across
   all computations.
3. **Early detach** – Detaches tensors where gradients are not needed to reduce
   autograd overhead.

### 3.4  Usage snippet
```python
from grpo.grpo_objective import GRPOObjective

objective = GRPOObjective(clip_epsilon=0.2, kl_coeff=0.04, entropy_coeff=0.01)
loss_dict = objective.compute_loss(
    current_policy,
    reference_policy,
    experience_batch,
    advantages,
)

loss = loss_dict["total_loss"]
loss.backward()
optimizer.step()
```

---

## 4. Training Orchestration – `trainer.py`

### 4.1  Role in the algorithm
The `GRPOTrainer` is the **orchestration layer** that ties together all GRPO components into a complete training pipeline. It handles:

1. **Experience collection** – Generates actions and computes rewards for given states
2. **Advantage computation** – Transforms rewards into group-relative advantages
3. **Multi-epoch optimization** – Runs multiple gradient steps on the same batch
4. **Reference policy management** – Updates the reference policy for KL regularization
5. **Gradient processing** – Applies clipping and accumulation across mini-batches
6. **Metrics tracking** – Comprehensive logging of training statistics
7. **Checkpoint management** – Save/load training state for resumption

### 4.2  Configuration management
The trainer uses `GRPOConfig` dataclass for centralized hyperparameter management:

```python
@dataclass
class GRPOConfig:
    group_size: int = 4                    # Actions sampled per state
    batch_size: int = 64                   # Mini-batch size for optimization
    num_epochs_per_update: int = 4         # Gradient steps per experience batch
    clip_epsilon: float = 0.2              # PPO clipping parameter
    kl_coeff: float = 0.04                 # KL penalty coefficient
    entropy_coeff: float = 0.01            # Entropy bonus coefficient
    max_grad_norm: float = 1.0             # Gradient clipping threshold
    normalize_advantages: bool = True       # Enable advantage normalization
    ref_update_frequency: int = 1          # Steps between reference updates
```

### 4.3  Key features
• **Batched processing** – Splits large experience batches into smaller mini-batches for memory efficiency  
• **Multi-epoch training** – Performs multiple optimization epochs on each collected batch, similar to PPO  
• **Reference policy updates** – Periodically copies current policy to reference via `copy.deepcopy`  
• **Comprehensive metrics** – Tracks per-epoch and aggregate statistics for all loss components  
• **Gradient management** – Handles accumulation, clipping, and zeroing across mini-batches  
• **Device awareness** – Automatically manages tensor devices for CPU/GPU compatibility  
• **Checkpoint robustness** – Saves complete training state including optimizer and random number generators  
• **Memory cleanup** – Proper tensor detachment and cleanup to prevent memory leaks  

### 4.4  Training loop architecture

The core training step follows this sequence:

1. **Rollout phase**:
   ```python
   batch = self.experience_collector.collect_experiences(states)
   advantages = self.advantage_estimator.compute_advantages(batch)
   ```

2. **Multi-epoch optimization**:
   ```python
   for epoch in range(self.config.num_epochs_per_update):
       for mini_batch in self._create_mini_batches(batch, advantages):
           loss_dict = self.objective.compute_loss(...)
           loss_dict["total_loss"].backward()
           # Gradient accumulation and clipping
   ```

3. **Reference policy updates**:
   ```python
   if self.step_count % self.config.ref_update_frequency == 0:
       self.reference_policy = copy.deepcopy(self.policy)
   ```

### 4.5  Optimizations implemented

1. **Mini-batch processing** – Reduces memory usage by processing large batches in smaller chunks
2. **Gradient accumulation** – Enables effective larger batch sizes without memory overflow
3. **Efficient tensor operations** – Minimizes device transfers and tensor copying
4. **Reference policy sharing** – Reuses reference policy across epochs to avoid redundant deep copies
5. **Metric caching** – Computes expensive statistics only when needed
6. **Memory hygiene** – Explicit cleanup of temporary tensors and gradients

### 4.6  Metrics and monitoring

The trainer provides detailed metrics for monitoring training progress:

**Per-step metrics**:
- `step_loss` – Overall loss for the training step
- `grad_norm` – Gradient norm before clipping
- `advantage_mean/std` – Advantage distribution statistics
- `total_samples` – Number of actions processed
- `collection_time` – Time spent on experience collection
- `advantage_time` – Time spent computing advantages

**Per-epoch metrics** (for each optimization epoch):
- `epoch_{i}_policy_loss` – Policy gradient loss
- `epoch_{i}_kl_penalty` – KL divergence penalty
- `epoch_{i}_entropy_bonus` – Entropy regularization bonus
- `epoch_{i}_total_loss` – Combined loss

### 4.7  Checkpoint management

The trainer supports robust checkpointing for training resumption:

```python
# Save checkpoint
trainer.save_checkpoint("model_checkpoint.pt", best_loss=current_loss)

# Load checkpoint
trainer.load_checkpoint("model_checkpoint.pt")
```

Checkpoints include:
- Policy and reference policy state dictionaries
- Optimizer state (including momentum terms)
- Training step and epoch counters
- Best loss tracking
- Complete training history
- Random number generator states (for reproducibility)

### 4.8  Usage patterns

**Basic training loop**:
```python
from grpo import GRPOTrainer, GRPOConfig
from torch.optim import Adam

config = GRPOConfig(
    group_size=8,
    batch_size=64,
    num_epochs_per_update=4,
    clip_epsilon=0.2
)

optimizer = Adam(policy.parameters(), lr=1e-4)
trainer = GRPOTrainer(policy, reward_model, optimizer, config)

for step in range(num_training_steps):
    states = get_training_states()  # Your data source
    metrics = trainer.train_step(states)
    
    if step % 100 == 0:
        trainer.save_checkpoint(f"checkpoint_{step}.pt")
        print(f"Step {step}: Loss = {metrics['step_loss']:.4f}")
```

**Advanced usage with custom scheduling**:
```python
# Custom learning rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# Training with validation
for step in range(num_training_steps):
    # Training
    metrics = trainer.train_step(train_states)
    scheduler.step()
    
    # Periodic validation
    if step % 50 == 0:
        with torch.no_grad():
            val_batch = trainer.experience_collector.collect_experiences(val_states)
            val_advantages = trainer.advantage_estimator.compute_advantages(val_batch)
            val_metrics = trainer.objective.compute_loss(
                trainer.policy, trainer.reference_policy, val_batch, val_advantages
            )
        print(f"Validation loss: {val_metrics['total_loss']:.4f}")
```

### 4.9  Error handling and robustness

The trainer includes comprehensive error handling:
- **Input validation** – Checks for empty state lists and malformed inputs
- **Device consistency** – Ensures all tensors are on the correct device
- **Gradient monitoring** – Detects and reports gradient explosion or vanishing
- **Memory management** – Monitors and prevents memory leaks
- **Checkpoint validation** – Verifies checkpoint integrity before loading

---

## 5. Base Interfaces – `policy_base.py` & `reward_model_base.py`

### 5.1  Design philosophy
The GRPO framework is **model-agnostic** by design. The base interfaces define minimal contracts that any policy or reward model must satisfy, enabling integration with:
- Transformer language models (GPT, LLaMA, etc.)
- Vision-language models (CLIP, DALL-E, etc.)  
- Reinforcement learning agents (DQN, Actor-Critic, etc.)
- Custom domain-specific models

### 5.2  PolicyModel interface
```python
class PolicyModel(ABC):
    @abstractmethod
    def generate_actions(self, states: List[Any], num_actions_per_state: int, **kwargs) -> List[List[Any]]:
        """Generate candidate actions for each state."""
        
    @abstractmethod  
    def get_log_probabilities(self, states: List[Any], actions: List[Any]) -> Tensor:
        """Compute log probabilities for state-action pairs."""
        
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return model parameters for optimization."""
```

### 5.3  RewardModel interface  
```python
class RewardModel(ABC):
    @abstractmethod
    def compute_rewards(self, states: List[Any], actions: List[Any]) -> Tensor:
        """Assign scalar rewards to state-action pairs."""
```

### 5.4  Implementation guidelines
- **Batching support** – Models should handle lists of states/actions efficiently
- **Device management** – Models should be device-aware (CPU/GPU)
- **Memory efficiency** – Avoid storing large intermediate tensors unnecessarily
- **Error handling** – Validate inputs and provide informative error messages

---

## Model Validation

Since GRPO uses Protocol classes (duck typing), your models don't need to inherit from `PolicyModel` or `RewardModel`. However, they must implement the required methods. GRPO provides validation utilities to help you check this:

### Automatic Validation

The `GRPOTrainer` automatically validates models when you create it:

```python
import grpo

# This will raise a TypeError if your models don't implement required methods
trainer = grpo.GRPOTrainer(
    policy_model=your_policy,
    reward_model=your_reward_model
)
```


### What Happens When Methods Are Missing?

#### 1. **Development Time (Static Type Checking)**
If you use mypy or an IDE with type checking, you'll get warnings:

```python
class IncompletePolicy:
    def generate_actions(self, states, num_actions_per_state, **kwargs):
        return [[f"action_{i}"] for _ in states for i in range(num_actions_per_state)]
    # Missing: get_log_probabilities, get_parameters, train, eval, to

policy: grpo.PolicyModel = IncompletePolicy()  # Type checker warning
```

#### 2. **Runtime Validation**
GRPO will catch missing methods early with helpful error messages:

```python
class IncompletePolicy:
    def generate_actions(self, states, num_actions_per_state, **kwargs):
        return [[f"action_{i}"] for _ in states for i in range(num_actions_per_state)]

try:
    trainer = grpo.GRPOTrainer(
        policy_model=IncompletePolicy(),
        reward_model=your_reward_model
    )
except TypeError as e:
    print(e)
    # Output: Model IncompletePolicy does not implement PolicyModel protocol.
    # Missing methods: get_log_probabilities, get_parameters, train, eval, to
    # Required methods: generate_actions, get_log_probabilities, get_parameters, train, eval, to
```

#### 3. **Runtime Errors (If Validation Is Bypassed)**
If validation is somehow bypassed and missing methods are called:

```python
incomplete_model.eval()  # AttributeError: 'IncompletePolicy' object has no attribute 'eval'
```

### Required Methods Summary

**PolicyModel Protocol:**
- `generate_actions(states, num_actions_per_state, **kwargs) -> List[List[Any]]`
- `get_log_probabilities(states, actions) -> Tensor`
- `get_parameters() -> Dict[str, Tensor]`
- `train() -> None`
- `eval() -> None`
- `to(device) -> PolicyModel`

**RewardModel Protocol:**
- `compute_rewards(states, actions) -> Tensor`
- `to(device) -> RewardModel`
- `eval() -> None`
- `train() -> None`

---

This completes the comprehensive documentation of the GRPO framework. Each component is designed for modularity, efficiency, and ease of integration with existing ML pipelines. 