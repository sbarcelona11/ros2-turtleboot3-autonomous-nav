import json
import random
from types import SimpleNamespace

import pytest

from turtleboot3_autonomous_nav.dqn import (
    DQNPolicy,
    ReplayBuffer,
    load_checkpoint,
    save_checkpoint,
)
from turtleboot3_autonomous_nav.dqn_explorer import greedy_action_for_observation
from turtleboot3_autonomous_nav.dqn_trainer import (
    EpisodeResetGate,
    TrainerConfig,
    configure_reset_all,
    episode_end_reason,
    episode_reward,
    optimize_replay,
    parse_reset_provenance,
    save_if_improved,
)
from turtleboot3_autonomous_nav.observation import OBSERVATION_CONTRACT


def test_checkpoint_round_trip_preserves_action_space(tmp_path):
    """A saved model retains its discrete action-space metadata."""
    policy = DQNPolicy(observation_size=10, action_count=6)
    observation = [0.25] * 10
    expected_values = policy.q_values(observation)

    save_checkpoint(
        tmp_path / "best.pt",
        policy,
        {"epsilon": 0.0},
        {"coverage": 0.5},
    )

    loaded, _, metrics = load_checkpoint(tmp_path / "best.pt")

    assert loaded.action_count == 6
    assert metrics["coverage"] == 0.5
    assert loaded.q_values(observation) == expected_values


def test_trainer_config_defaults_to_the_monitoring_observation_contract():
    """New campaign snapshots must identify the observation semantics."""
    assert TrainerConfig().observation_contract == OBSERVATION_CONTRACT


def test_trainer_config_rejects_a_stale_observation_contract():
    """A trainer cannot create checkpoints that inference must reject."""
    with pytest.raises(ValueError, match='observation contract'):
        TrainerConfig(observation_contract='legacy')


def test_replay_buffer_retains_only_its_most_recent_capacity():
    """Old experiences must be evicted instead of growing replay without bound."""
    replay = ReplayBuffer(capacity=2)

    for action in range(3):
        replay.add([action], action, float(action), [action + 1], False)

    retained_actions = {transition.action for transition in replay.sample(2)}

    assert len(replay) == 2
    assert retained_actions == {1, 2}


def test_training_selection_uses_epsilon_to_choose_exploration_or_greedy_action():
    """Epsilon zero is greedy, while epsilon one uses the supplied random action."""
    policy = DQNPolicy(observation_size=3, action_count=4)
    observation = [0.1, -0.2, 0.3]
    greedy_action = policy.select_action(observation)
    exploratory_action = (greedy_action + 1) % policy.action_count

    class ForcedExploration:
        def random(self):
            return 0.0

        def randrange(self, stop):
            assert stop == policy.action_count
            return exploratory_action

    assert policy.select_training_action(observation, epsilon=0.0) == greedy_action
    assert (
        policy.select_training_action(observation, epsilon=1.0, rng=ForcedExploration())
        == exploratory_action
    )


def test_copy_target_network_replaces_target_weights_with_online_weights():
    """Target synchronization must discard stale target-network values."""
    policy = DQNPolicy(observation_size=2, action_count=2)
    observation = [1.0, -1.0]
    state = policy.state_dict()
    state["target_layers"][-1]["bias"][0] += 100.0
    policy.load_state_dict(state)

    assert policy.target_q_values(observation) != policy.q_values(observation)

    policy.copy_target_network()

    assert policy.target_q_values(observation) == policy.q_values(observation)


def test_inference_rejects_an_observation_with_the_wrong_dimension():
    """A malformed observation must not reach the network's first layer."""
    policy = DQNPolicy(observation_size=10, action_count=6)

    with pytest.raises(ValueError, match="expected observation size 10"):
        policy.select_action([0.0] * 9)


def test_explorer_adapter_uses_greedy_policy_inference():
    """Exploration deployment must not accidentally retain training epsilon."""
    policy = DQNPolicy(observation_size=3, action_count=4)
    observation = [0.2, 0.1, -0.5]

    assert greedy_action_for_observation(policy, observation) == policy.select_action(
        observation
    )


def test_reward_favors_new_coverage_and_penalizes_intervention():
    """Coverage progress must outweigh normal step cost, unsafe progress must not."""
    assert episode_reward(12, True, False, False) > 0
    assert episode_reward(0, False, True, True) < 0


def test_episode_ends_at_the_first_configured_terminal_condition():
    """A trainer must not run past its episode, coverage, or stall bounds."""
    config = TrainerConfig(max_steps=5, target_coverage=0.8, stall_seconds=3.0)

    assert episode_end_reason(5, 0.1, 0, config) == "max_steps"
    assert episode_end_reason(2, 0.8, 0, config) == "target_coverage"
    assert episode_end_reason(2, 0.1, 3, config) == "stalled"
    assert episode_end_reason(2, 0.1, 2, config) is None


def test_reset_request_uses_gazebo_reset_all_flag():
    """Episode resets must request Gazebo's full world reset, not a time reset."""
    request = SimpleNamespace(
        world_control=SimpleNamespace(reset=SimpleNamespace(all=False))
    )

    configure_reset_all(request)

    assert request.world_control.reset.all is True


def test_checkpoint_is_saved_only_when_mean_coverage_improved(tmp_path):
    """A worse evaluation must never overwrite the deployment checkpoint."""
    policy = DQNPolicy(observation_size=2, action_count=2, seed=7)
    checkpoint = tmp_path / "best.pt"
    metrics = tmp_path / "best.metrics.json"
    episodes = [
        {'reason': 'target_coverage', 'reachable_coverage': 0.8,
         'grid_coverage': 0.6, 'interventions': 1, 'policy_actions': 10,
         'turn_actions': 2, 'turn_ratio': 0.2},
        {'reason': 'time_limit', 'reachable_coverage': 0.6,
         'grid_coverage': 0.5, 'interventions': 3, 'policy_actions': 20,
         'turn_actions': 10, 'turn_ratio': 0.5},
    ]

    saved = save_if_improved(
        checkpoint,
        metrics,
        policy,
        {"epsilon": 0.0},
        {"mean_coverage": 0.7, "episode": 4, "episodes": episodes},
        best_mean_coverage=0.6,
    )
    rejected = save_if_improved(
        checkpoint,
        metrics,
        policy,
        {"epsilon": 0.0},
        {"mean_coverage": 0.6, "episode": 5},
        best_mean_coverage=0.7,
    )

    assert saved is True
    assert rejected is False
    assert checkpoint.exists()
    stored = json.loads(metrics.read_text())['metrics']
    assert stored['mean_coverage'] == 0.7
    assert stored['episodes'] == episodes
    assert load_checkpoint(checkpoint)[2]['episodes'] == episodes


def test_replay_optimization_increases_the_value_of_a_rewarded_action():
    """A terminal positive reward must move its chosen Q-value upward."""
    policy = DQNPolicy(observation_size=2, action_count=2, hidden_sizes=(1, 1), seed=3)
    replay = ReplayBuffer(capacity=2)
    observation = [1.0, 0.0]
    replay.add(observation, 0, 10.0, [0.0, 0.0], True)
    before = policy.q_values(observation)[0]

    optimize_replay(policy, replay, batch_size=1, gamma=0.99, learning_rate=0.01)

    assert policy.q_values(observation)[0] > before


def test_episode_reset_gate_rejects_delayed_pre_reset_sensor_timestamps():
    """Queued pre-reset callbacks must never arm a new episode."""
    gate = EpisodeResetGate()

    gate.begin_reset()
    assert gate.record_sensor("odom", 1) is False
    gate.world_reset_succeeded()
    assert gate.record_sensor("scan", 1) is False

    gate.mapper_reset_succeeded(cutoff_ns=1_000, epoch=7)
    assert gate.record_sensor("odom", 999) is False
    assert gate.record_sensor("scan", 999) is False
    assert gate.record_sensor("odom", 1_000) is False
    assert gate.ready is False
    assert gate.record_sensor("odom", 1_001) is True
    assert gate.record_sensor("scan", 1_001) is True
    assert gate.ready is True

    gate.start_episode()
    assert gate.accepts_timestamp(999) is False
    assert gate.accepts_timestamp(1_002) is True
    gate.begin_reset()
    assert gate.accepts_timestamp(1_003) is False


@pytest.mark.parametrize(
    "message",
    [
        "null",
        "[]",
        "{}",
        '{"cutoff_ns": true, "epoch": 1}',
        '{"cutoff_ns": 1, "epoch": 0}',
    ],
)
def test_invalid_reset_acknowledgement_cannot_arm_an_episode(message):
    with pytest.raises(ValueError):
        parse_reset_provenance(message)


def test_mapper_reset_provenance_is_parsed_from_the_reset_acknowledgement():
    """The trainer must use the mapper-issued cutoff, not local delivery order."""
    assert parse_reset_provenance('{"cutoff_ns": 123, "epoch": 4}') == (123, 4)
