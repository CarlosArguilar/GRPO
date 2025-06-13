from grpo.advantage_estimator import AdvantageEstimator
from grpo.experience_collector import ExperienceBatch
import torch


def _make_dummy_batch() -> ExperienceBatch:
    """Create a hand-crafted ExperienceBatch for deterministic testing."""
    states = ["s0", "s1"]
    # two groups, three actions each
    action_groups = [["a0", "a1", "a2"], ["b0", "b1", "b2"]]
    reward_groups = [[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]]
    log_prob_old_groups = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    return ExperienceBatch(
        states=states,
        action_groups=action_groups,
        reward_groups=reward_groups,
        log_prob_old_groups=log_prob_old_groups,
    )


def test_raw_advantage_values():
    batch = _make_dummy_batch()
    est = AdvantageEstimator(normalize_advantages=False)
    adv = est.compute_advantages(batch)

    expected = torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(adv, expected)


def test_normalised_advantages_mean_std():
    batch = _make_dummy_batch()
    est = AdvantageEstimator(normalize_advantages=True)
    adv = est.compute_advantages(batch)

    # Numerical tolerance due to fp32 rounding
    assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(adv.std(unbiased=False), torch.tensor(1.0), atol=1e-6) 