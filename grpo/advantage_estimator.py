"""Advantage estimation utilities for Group-Relative Policy Optimisation (GRPO).

The estimator computes *group-relative* advantages, i.e. for every sampled
action the baseline is the **mean reward within its own state-level group**.
This is the quantity required by GRPO's clipped objective where each action
only competes against the other actions drawn from the *same* state.

Equations
---------
Let 𝑖 index environment states (size = *batch_size*)
and 𝑗 index the *group_size* actions drawn for that state.

    rᵢⱼ      : scalar reward of action j in state i
    bᵢ       : baseline for state i =  meanⱼ(rᵢⱼ)
    Aᵢⱼ      : advantage = rᵢⱼ − bᵢ

After computing the raw advantages, we optionally **normalise** them across the
entire batch to have zero mean and unit variance – a well-known variance
reduction trick that does *not* alter the gradient expectation but often speeds
up optimisation.

Implementation details
----------------------
1.  Rewards arrive as nested Python lists via ``ExperienceBatch.reward_groups``.
2.  A single pass computes per-group means (baselines) and raw advantages.
3.  Advantages are stored in a 1-D ``torch.Tensor`` whose order matches the
    flattened *(state, action)* order used elsewhere in the pipeline.
4.  When normalisation is enabled, the tensor is centred and scaled in place
    using unbiased=False variance for numerical stability.

The estimator is intentionally **stateless** beyond its hyper-parameters; this
keeps it cheap to instantiate and multi-thread safe.
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from .experience_collector import ExperienceBatch


class AdvantageEstimator:
    """Compute *group-relative* advantages required by GRPO.

    Parameters
    ----------
    normalize_advantages:
        If *True* the returned tensor is centred and scaled to have zero mean
        and unit variance.  Recommended when used together with first-order
        optimisers such as Adam.
    advantage_epsilon:
        Small constant added to the denominator during normalisation to avoid
        division by zero in degenerate cases (e.g. all rewards identical).
    """

    def __init__(
        self, *, normalize_advantages: bool = True, advantage_epsilon: float = 1e-8
    ) -> None:
        self.normalize = normalize_advantages
        self.eps = float(advantage_epsilon)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_advantages(self, experience_batch: ExperienceBatch) -> Tensor:
        """Return a 1-D tensor of advantages aligned with the flattened action order.

        Parameters
        ----------
        experience_batch:
            The collected roll-out data.  Only ``reward_groups`` is required but
            we keep the full object for type clarity.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size * group_size,)`` containing **raw** or
            **normalised** advantages depending on initialisation.
        """

        reward_groups: List[List[float]] = experience_batch.reward_groups

        # 1. Compute per-group baselines (mean reward)
        baselines = self._compute_group_baselines(reward_groups)

        # 2. Compute raw advantages – vectorised across groups
        #    We'll allocate a single 1-D list then convert to tensor once.
        raw_adv: List[float] = []
        for baseline, rewards in zip(baselines, reward_groups):
            raw_adv.extend([r - baseline for r in rewards])

        advantages = torch.tensor(raw_adv, dtype=torch.float32)

        # 3. Optional normalisation
        if self.normalize:
            advantages = self._normalize_advantages(advantages)

        return advantages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_group_baselines(reward_groups: List[List[float]]) -> List[float]:
        """Return the mean reward of each state-level group."""

        return [float(sum(group) / len(group)) if group else 0.0 for group in reward_groups]

    def _normalize_advantages(self, advantages: Tensor) -> Tensor:
        """Center & scale advantages: :math:`(A - \\mu) / (\\sigma + \\varepsilon)`"""

        if advantages.numel() == 0:  # pragma: no cover
            return advantages  # avoid NaNs on empty batches

        mean = advantages.mean()
        std = advantages.std(unbiased=False)
        # Avoid division by zero (all advantages identical)
        return (advantages - mean) / (std + self.eps) 