"""A large temporal-difference error must not produce a large weight step.

With a squared loss the gradient grows without bound with the error, so one
surprising transition can wreck the network.  Measured on this project the
greedy policy degraded monotonically across five evaluations - 59.9% to 35.7%
coverage - while the exploration rate fell and the learned policy took over:
the signature of a diverging Q-function.  Huber loss, gradient clipping and
bounded rewards are the standard mitigations, and this pins them.
"""

import pytest

from turtleboot3_autonomous_nav.dqn import DQNPolicy, ReplayBuffer
from turtleboot3_autonomous_nav.dqn_trainer import (
    TrainerConfig,
    episode_reward,
    optimize_replay,
)


def _policy_and_replay(reward):
    policy = DQNPolicy(3, 2, hidden_sizes=(4, 4), seed=11)
    replay = ReplayBuffer(10)
    for _ in range(4):
        replay.add([0.5, -0.25, 1.0], 0, reward, [0.1, 0.2, 0.3], True)
    return policy, replay


def _weight_change(reward):
    policy, replay = _policy_and_replay(reward)
    before = [row[:] for row in policy.state_dict()['layers'][0]['weights']]
    optimize_replay(policy, replay, 4, 0.99, 0.01)
    after = policy.state_dict()['layers'][0]['weights']
    return max(
        abs(a - b)
        for row_a, row_b in zip(after, before)
        for a, b in zip(row_a, row_b)
    )


def test_a_huge_error_does_not_scale_the_weight_step_with_it():
    """A thousandfold error must not move the weights a thousand times more."""
    modest = _weight_change(1.0)
    huge = _weight_change(1000.0)
    assert huge < modest * 10.0


def test_the_reported_loss_is_bounded_for_a_huge_error():
    """A squared loss would report a million here; Huber grows linearly."""
    policy, replay = _policy_and_replay(1000.0)
    assert optimize_replay(policy, replay, 4, 0.99, 0.01) < 1500.0


def test_a_modest_error_still_moves_the_weights():
    """Bounding the step must not stop learning on ordinary transitions."""
    assert _weight_change(1.0) > 0.0


def test_gradient_clipping_caps_the_step_for_a_large_input():
    """Large activations must not bypass the bound on the error."""
    policy = DQNPolicy(3, 2, hidden_sizes=(4, 4), seed=11)
    replay = ReplayBuffer(10)
    for _ in range(4):
        replay.add([500.0, -500.0, 500.0], 0, 5.0, [1.0, 1.0, 1.0], True)
    before = [row[:] for row in policy.state_dict()['layers'][0]['weights']]
    optimize_replay(policy, replay, 4, 0.99, 0.01)
    after = policy.state_dict()['layers'][0]['weights']
    change = max(
        abs(a - b)
        for row_a, row_b in zip(after, before)
        for a, b in zip(row_a, row_b)
    )
    assert change < 1.0


def test_a_step_reward_is_clipped_to_the_configured_bound():
    """A burst of newly seen cells must not dwarf every other transition."""
    config = TrainerConfig(new_cell_reward=1.0, reward_clip=1.0)
    assert episode_reward(500, False, False, False, False, config) == pytest.approx(1.0)


def test_a_large_penalty_is_clipped_symmetrically():
    config = TrainerConfig(intervention_penalty=50.0, reward_clip=1.0)
    assert episode_reward(0, False, True, False, False, config) == pytest.approx(-1.0)


def test_clipping_is_disabled_by_a_non_positive_bound():
    """Keeping the raw scale available makes the bound an explicit choice."""
    config = TrainerConfig(new_cell_reward=1.0, reward_clip=0.0, step_penalty=0.0)
    assert episode_reward(500, False, False, False, False, config) == pytest.approx(500.0)
