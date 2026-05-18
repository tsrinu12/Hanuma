"""Unit tests for LinUCB core math."""

import numpy as np
import pytest

from app.bandit import LinUCB


def test_dim_validation():
    b = LinUCB(context_dim=4, alpha=1.0)
    with pytest.raises(ValueError):
        b.score(0, np.zeros(3))
    with pytest.raises(ValueError):
        b.update(0, np.zeros(5), 1.0)


def test_initial_score_is_pure_exploration_for_zero_context():
    b = LinUCB(context_dim=4, alpha=1.0)
    score, exploit, explore = b.score(arm_id=0, context=np.zeros(4))
    # Before any updates: theta=0, A=I, A_inv=I.
    # exploit = 0, explore = alpha * sqrt(0) = 0.
    assert exploit == 0.0
    assert explore == 0.0
    assert score == 0.0


def test_initial_score_explores_for_nonzero_context():
    b = LinUCB(context_dim=4, alpha=2.0)
    x = np.array([1.0, 0.0, 0.0, 0.0])
    score, exploit, explore = b.score(0, x)
    # With A=I: explore = alpha * sqrt(x^T x) = alpha * 1 = 2.0
    assert exploit == 0.0
    assert explore == pytest.approx(2.0)
    assert score == pytest.approx(2.0)


def test_update_increases_pulls_and_arm_count():
    b = LinUCB(context_dim=2, alpha=1.0)
    assert b.arm_count() == 0
    b.update(7, np.array([1.0, 0.0]), 1.0)
    assert b.arm_count() == 1
    assert b.total_pulls() == 1
    b.update(7, np.array([0.0, 1.0]), 0.5)
    assert b.total_pulls() == 2


def test_theta_converges_toward_reward_direction():
    """If we always reward when context = e1, theta should point toward e1."""
    b = LinUCB(context_dim=3, alpha=0.0)  # alpha=0 → pure exploit, simpler check
    x = np.array([1.0, 0.0, 0.0])
    for _ in range(50):
        b.update(0, x, 1.0)
    score_aligned, _, _ = b.score(0, x)
    score_orthog, _, _ = b.score(0, np.array([0.0, 1.0, 0.0]))
    assert score_aligned > score_orthog
    assert score_aligned > 0.5  # learned something


def test_exploration_shrinks_as_pulls_accumulate():
    """For the same context, the explore term should decay with more updates."""
    b = LinUCB(context_dim=3, alpha=1.0)
    x = np.array([1.0, 0.0, 0.0])
    _, _, explore_0 = b.score(0, x)
    for _ in range(100):
        b.update(0, x, 1.0)
    _, _, explore_n = b.score(0, x)
    assert explore_n < explore_0


def test_alpha_override_scales_explore():
    b = LinUCB(context_dim=3, alpha=1.0)
    x = np.array([1.0, 0.0, 0.0])
    _, _, e_default = b.score(0, x)
    _, _, e_big = b.score(0, x, alpha=4.0)
    assert e_big == pytest.approx(4.0 * e_default)


def test_arms_are_independent():
    b = LinUCB(context_dim=2, alpha=1.0)
    x = np.array([1.0, 0.0])
    b.update(0, x, 1.0)
    b.update(0, x, 1.0)
    score_0, _, _ = b.score(0, x)
    score_1, _, _ = b.score(1, x)  # untouched arm
    assert score_0 != score_1
