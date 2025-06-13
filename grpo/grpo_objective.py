"""GRPO objective implementation.

This module provides the **Group-Relative Policy Optimisation (GRPO)** loss
computation, which extends the standard PPO surrogate with

1. *Group-relative advantages* – supplied externally by
   ``AdvantageEstimator`` (already mean-centered per state).
2. *Clipped likelihood ratio* – identical to PPO to bound policy updates.
3. Optional *KL divergence* penalty that keeps the current policy close to a
   fixed reference policy (e.g. a pre-trained base model).
4. Optional *entropy bonus* to maintain exploration.

Mathematical form
-----------------

Let

    r̂    : computed group-relative advantage (scalar)
    ℓ₀   : log-probability under the *behaviour/old* policy
    ℓ    : log-probability under the *current* policy

    ρ = exp(ℓ − ℓ₀)                # likelihood ratio
    ρ̂ = clip(ρ, 1 − ε, 1 + ε)    # clipped ratio

The clipped surrogate objective is

    L_policy = E[min(ρ · r̂, ρ̂ · r̂)]

Total loss (to **minimise**) adds penalties/bonuses:

    L_total = -L_policy            # maximise surrogate → minimise negative
              + λ_kl · KL(π║π_ref)  (if reference provided)
              - β_ent · Entropy(π)  (optional)

The implementation operates on *flattened* tensors aligned with the order used
by ``ExperienceCollector`` and ``AdvantageEstimator``.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional, Any

import torch
from torch import Tensor

from .policy_base import PolicyModel
from .experience_collector import ExperienceBatch


class GRPOObjective:
    """Compute the GRPO loss consisting of surrogate, KL and entropy terms."""

    def __init__(
        self,
        *,
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.04,
        entropy_coeff: float = 0.0,
    ) -> None:
        self.clip_eps = float(clip_epsilon)
        self.kl_coeff = float(kl_coeff)
        self.entropy_coeff = float(entropy_coeff)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        current_policy: PolicyModel,
        reference_policy: Optional[PolicyModel],
        experience_batch: ExperienceBatch,
        advantages: Tensor,
    ) -> Dict[str, Tensor]:
        """Return a dict with `total_loss` and individual components.

        The returned *losses are already reduced* (mean over samples) and have
        the correct sign for **minimisation**.
        """

        # ------------------------------------------------------------------
        # Prepare data – flatten experiences & validate inputs
        # ------------------------------------------------------------------
        states, actions = self._flatten_experiences(experience_batch)

        total_actions = len(actions)
        if advantages.shape != (total_actions,):
            raise ValueError(
                "Advantages shape {advantages.shape} doesn't match total actions "
                f"({total_actions})"
            )

        # 1. Current log-probabilities (gives us device as well)
        log_probs_current: Tensor = current_policy.get_log_probabilities(states, actions)
        device = log_probs_current.device

        # 2. Move/prepare other tensors on same device & detach where gradients not needed
        advantages = advantages.detach().to(device)
        log_probs_old = self._get_log_probs_old(experience_batch, device)

        # 3. Surrogate (policy) loss
        policy_loss = self._compute_policy_loss(log_probs_current, log_probs_old, advantages)

        # 2. KL penalty (optional)
        if reference_policy is not None and self.kl_coeff > 0.0:
            kl_penalty = self._compute_kl_penalty(current_policy, reference_policy, states, actions, log_probs_current)
        else:
            kl_penalty = torch.tensor(0.0, dtype=torch.float32, device=log_probs_current.device)

        # 3. Entropy bonus (optional)
        if self.entropy_coeff != 0.0:
            entropy_bonus = self._compute_entropy_bonus(log_probs_current)
        else:
            entropy_bonus = torch.tensor(0.0, dtype=torch.float32, device=log_probs_current.device)

        total_loss = policy_loss + kl_penalty - entropy_bonus  # minus because bonus improves obj

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "kl_penalty": kl_penalty,
            "entropy_bonus": entropy_bonus,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_policy_loss(
        self, log_probs_current: Tensor, log_probs_old: Tensor, advantages: Tensor
    ) -> Tensor:
        """PPO-style clipped surrogate *loss* (negative of objective)."""

        # Likelihood ratio
        ratios = torch.exp(log_probs_current - log_probs_old)

        # Clipped ratios
        clipped = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

        # Surrogate
        surrogate1 = ratios * advantages
        surrogate2 = clipped * advantages
        objective = torch.minimum(surrogate1, surrogate2)

        # We **maximise** objective → minimise negative
        return -objective.mean()

    def _compute_kl_penalty(
        self,
        current_policy: PolicyModel,
        reference_policy: PolicyModel,
        states: List[Any],
        actions: List[Any],
        log_probs_current: Optional[Tensor] = None,
    ) -> Tensor:
        """Mean forward KL: KL(current || reference).

        If ``log_probs_current`` is provided we reuse it to avoid duplicate
        model passes; otherwise we query the policy again (needed for grad).
        """

        if log_probs_current is None:
            log_probs_cur = current_policy.get_log_probabilities(states, actions)
        else:
            # Use the provided log_probs_current directly (with gradients)
            log_probs_cur = log_probs_current

        with torch.no_grad():
            log_probs_ref = reference_policy.get_log_probabilities(states, actions)

        kl = (log_probs_cur - log_probs_ref).mean()
        return self.kl_coeff * kl

    def _compute_entropy_bonus(self, log_probs_current: Tensor) -> Tensor:
        """Simple entropy estimator using -log p (works for categorical sampling)."""

        entropy_est = -log_probs_current.mean()
        return self.entropy_coeff * entropy_est

    @staticmethod
    def _flatten_experiences(
        experience_batch: ExperienceBatch,
    ) -> Tuple[List[Any], List[Any]]:
        """Return (flat_states, flat_actions) matching collection/advantage order."""

        flat_states: List[Any] = []
        flat_actions: List[Any] = []

        for state, acts in zip(experience_batch.states, experience_batch.action_groups):
            flat_states.extend([state] * len(acts))
            flat_actions.extend(acts)

        return flat_states, flat_actions

    def _get_log_probs_old(self, experience_batch: ExperienceBatch, device: torch.device) -> Tensor:
        """Return log-probabilities from the old policy."""
        log_probs_old = torch.tensor(
            [lp for grp in experience_batch.log_prob_old_groups for lp in grp],
            dtype=torch.float32,
            device=device
        )
        return log_probs_old 