# Group-Relative Policy Optimisation (GRPO)

This sub-package contains all the core components required to train an agent with **Group-Relative Policy Optimisation (GRPO)**.  Each module maps cleanly to a conceptual step in the GRPO pipeline, enabling the framework to remain **model-agnostic**, highly testable and easy to extend.

> **Note**  
> This document will be progressively expanded.  For the moment it focuses on the *experience buffer* because this is the linchpin that connects the rollout and optimisation phases.  Other modules will be documented in upcoming iterations.

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

with $\rho = \exp(\ell - \ell_0)$ and $\hat{\rho} = \text{clip}(\rho, 1-\varepsilon, 1+\varepsilon)$.

### 3.2  Key features
• **Device-aware tensors** – Automatically moves advantages & old log-probs to
  the policy's device.
• **Log-space computations** – Uses \(\ell-\ell_0\) instead of exponentiating
  large values; improves numerical stability.
• **KL reuse** – Re-uses already computed `log_probs_current` when evaluating
  the KL term; avoids an extra forward pass.
• **Shape validation** – Fails fast if the length of advantages doesn't match
  the number of sampled actions.
• **Modular components** – Separate helpers for policy loss, KL penalty, and
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

*Future sections will document:*  
• Training loop orchestration (`trainer.py`)  
• Base interfaces (`policy_base.py`, `reward_model_base.py`)  
• Utility helpers (`utils/`) and logging facilities. 