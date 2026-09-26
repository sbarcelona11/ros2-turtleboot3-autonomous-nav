"""A campaign must survive a simulator crash.

Gazebo's physics aborted mid-campaign with an ODE assertion on a body's
bounding box, killing three hours of training.  The crash is upstream and
outside this package's control, so what is in our control is not losing the
work: the trainer writes its progress after every episode and picks it up on
the next launch.

Replay memory is deliberately not persisted - it is large and the checkpoint
format already excludes it - so a resume continues from the learned weights
and refills experience.
"""

import math

import pytest

from turtleboot3_autonomous_nav.dqn import DQNPolicy
from turtleboot3_autonomous_nav.dqn_trainer import (
    load_training_state,
    save_training_state,
)
from turtleboot3_autonomous_nav.observation import OBSERVATION_CONTRACT


def _progress(**overrides):
    progress = {
        'training_episodes': 24,
        'total_episodes': 30,
        'global_steps': 14_000,
        'best_mean_coverage': 0.517,
        'is_evaluation': False,
        'evaluation_coverages': [],
    }
    progress.update(overrides)
    return progress


def test_a_saved_campaign_reloads_its_progress(tmp_path):
    path = tmp_path / 'training_state.pt'
    save_training_state(
        path, DQNPolicy(6, 3, seed=2),
        {'observation_contract': OBSERVATION_CONTRACT}, _progress(),
    )

    _, progress = load_training_state(path, 6, 3)

    assert progress['training_episodes'] == 24
    assert progress['total_episodes'] == 30
    assert progress['global_steps'] == 14_000
    assert progress['best_mean_coverage'] == pytest.approx(0.517)


def test_a_resumed_policy_keeps_the_learned_weights(tmp_path):
    """Resuming from scratch weights would discard the whole campaign."""
    path = tmp_path / 'training_state.pt'
    policy = DQNPolicy(6, 3, seed=2)
    observation = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
    expected = policy.q_values(observation)
    save_training_state(
        path, policy, {'observation_contract': OBSERVATION_CONTRACT}, _progress()
    )

    resumed, _ = load_training_state(path, 6, 3)

    assert resumed.q_values(observation) == pytest.approx(expected)


def test_an_unfinished_evaluation_block_is_restored(tmp_path):
    """Otherwise a crash mid-evaluation would corrupt the next checkpoint."""
    path = tmp_path / 'training_state.pt'
    save_training_state(
        path,
        DQNPolicy(6, 3, seed=2),
        {'observation_contract': OBSERVATION_CONTRACT},
        _progress(is_evaluation=True, evaluation_coverages=[0.41, 0.52]),
    )

    _, progress = load_training_state(path, 6, 3)

    assert progress['is_evaluation'] is True
    assert progress['evaluation_coverages'] == pytest.approx([0.41, 0.52])


def test_a_campaign_with_no_evaluation_yet_round_trips_its_empty_best(tmp_path):
    """The first campaign has no best coverage; -inf must survive the trip."""
    path = tmp_path / 'training_state.pt'
    save_training_state(
        path, DQNPolicy(6, 3, seed=2),
        {'observation_contract': OBSERVATION_CONTRACT},
        _progress(best_mean_coverage=-math.inf)
    )

    _, progress = load_training_state(path, 6, 3)

    assert progress['best_mean_coverage'] == -math.inf


def test_a_missing_state_starts_a_fresh_campaign(tmp_path):
    """A first run must not need a state file to exist."""
    assert load_training_state(tmp_path / 'absent.pt', 6, 3) is None


def test_a_state_from_a_different_network_is_rejected(tmp_path):
    """Silently training a mismatched network would waste the campaign."""
    path = tmp_path / 'training_state.pt'
    save_training_state(
        path, DQNPolicy(6, 3, seed=2),
        {'observation_contract': OBSERVATION_CONTRACT}, _progress(),
    )

    with pytest.raises(ValueError):
        load_training_state(path, 90, 5)


def test_training_state_rejects_an_old_observation_contract(tmp_path):
    """Resume must not apply map-semantic weights to the new observations."""
    state = tmp_path / 'training_state.pt'
    save_training_state(state, DQNPolicy(90, 5), {}, _progress())

    with pytest.raises(ValueError, match='observation contract'):
        load_training_state(state, 90, 5)
