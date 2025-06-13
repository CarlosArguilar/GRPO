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

*Future sections will document:*  
• Advantage estimation (`advantage_estimator.py`)  
• GRPO clipped objective (`grpo_objective.py`)  
• Training loop orchestration (`trainer.py`)  
• Base interfaces (`policy_base.py`, `reward_model_base.py`)  
• Utility helpers (`utils/`) and logging facilities. 