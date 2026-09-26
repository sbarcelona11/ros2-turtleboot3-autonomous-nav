"""Recovery is a controller safety behaviour, not a navigation action.

Selecting it locks the robot into ``recovery_duration`` seconds of turning in
place, while every other action lasts only until the next one arrives.  Left
in the policy's action space it dominates the time budget: an untrained agent
picking it one time in six spends most of the episode frozen, and the mission
looks stationary no matter how well the policy is trained.
"""

from pathlib import Path

import pytest
import yaml

from turtleboot3_autonomous_nav.control import (
    POLICY_ACTION_COUNT,
    RECOVER,
    action_commands_translation,
)
from turtleboot3_autonomous_nav.dqn_trainer import TrainerConfig

CONFIG_ROOT = Path(__file__).resolve().parents[1] / 'config'


def test_recovery_is_outside_the_policy_action_space():
    """The policy must not be able to command a multi-second freeze."""
    assert RECOVER >= POLICY_ACTION_COUNT


def test_most_policy_actions_move_the_robot():
    """A random policy has to translate often enough to explore at all."""
    translating = [
        action
        for action in range(POLICY_ACTION_COUNT)
        if action_commands_translation(action)
    ]
    assert len(translating) > POLICY_ACTION_COUNT / 2


def test_the_trainer_and_the_shipped_config_agree_on_the_action_space():
    """A mismatch would train a head whose extra action nothing can execute."""
    config = yaml.safe_load(
        (CONFIG_ROOT / 'training.yaml').read_text()
    )['dqn_trainer']['ros__parameters']
    assert TrainerConfig().action_count == POLICY_ACTION_COUNT
    assert config['action_count'] == POLICY_ACTION_COUNT


@pytest.mark.parametrize('action', range(POLICY_ACTION_COUNT))
def test_every_policy_action_maps_to_a_command(action):
    """An action the controller cannot execute would publish nothing."""
    from turtleboot3_autonomous_nav.control import ControlConfig, _action_twist

    assert _action_twist(action, ControlConfig()) is not None
