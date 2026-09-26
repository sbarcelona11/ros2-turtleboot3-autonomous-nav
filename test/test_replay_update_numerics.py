"""Pin the exact arithmetic of one replay update.

The update is the learning contract: a rewrite that changes its numerics
silently changes what the agent learns, so any optimisation of the inner
loops has to reproduce these values rather than merely "train something".

The values were regenerated once, deliberately, when the loss changed from
squared error to Huber: with a squared loss the gradient grew without bound
with the temporal-difference error and the policy diverged, losing 24 points
of coverage across five evaluations.  See `test_update_stability`.
"""

import pytest

from turtleboot3_autonomous_nav.dqn import DQNPolicy, ReplayBuffer
from turtleboot3_autonomous_nav.dqn_trainer import optimize_replay


def _fixture():
    """A deterministic policy and replay small enough to reason about."""
    policy = DQNPolicy(3, 2, hidden_sizes=(2, 2), seed=5)
    replay = ReplayBuffer(10)
    replay.add([0.5, -0.25, 1.0], 0, 1.5, [0.1, 0.2, 0.3], False)
    replay.add([-1.0, 0.75, 0.0], 1, -0.5, [0.4, -0.1, 0.2], True)
    replay.add([0.2, 0.2, 0.2], 1, 0.25, [0.0, 0.0, 0.1], False)
    return policy, replay


def _update(policy, replay):
    return optimize_replay(
        policy, replay, batch_size=3, gamma=0.9, learning_rate=0.05
    )


def test_one_update_returns_the_reference_mean_huber_loss():
    policy, replay = _fixture()
    assert _update(policy, replay) == pytest.approx(0.6741452241902105, rel=1e-9)


def test_one_update_produces_the_reference_q_values():
    """The whole point of the update is where it moves the Q-values."""
    policy, replay = _fixture()
    _update(policy, replay)
    assert policy.q_values([0.5, -0.25, 1.0]) == pytest.approx(
        (-0.9710971593953728, -0.65611658797798), rel=1e-9
    )


def test_one_update_produces_the_reference_input_layer_weights():
    """Gradients must reach the first layer, not only the output head."""
    policy, replay = _fixture()
    _update(policy, replay)
    expected = [
        [0.25492594842675553, 0.5363863812086721, 0.6284722652488282],
        [0.9661040019085411, 0.5266571266480143, 0.9191305835707119],
    ]
    weights = policy.state_dict()['layers'][0]['weights']
    for row, expected_row in zip(weights, expected):
        assert row == pytest.approx(expected_row, rel=1e-9)


def test_one_update_produces_the_reference_output_bias():
    policy, replay = _fixture()
    _update(policy, replay)
    assert policy.state_dict()['layers'][2]['bias'] == pytest.approx(
        [0.016666666666666666, 0.001379768440940815], rel=1e-9
    )


def test_the_target_network_is_left_untouched_by_an_update():
    """Bootstrapping stays stable only if the target copy is not trained."""
    policy, replay = _fixture()
    before = policy.state_dict()['target_layers']
    _update(policy, replay)
    assert policy.state_dict()['target_layers'] == before


def test_an_update_is_skipped_until_the_replay_holds_a_full_batch():
    policy, replay = _fixture()
    assert optimize_replay(policy, replay, 4, 0.9, 0.05) is None
