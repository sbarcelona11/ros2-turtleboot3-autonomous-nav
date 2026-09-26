"""A stall means the robot was told to move and did not.

Turning in place is a legitimate command that never translates the robot, so
counting it as lack of progress traps the controller: recovery only rotates,
rotation produces no translation, and the stall condition that triggered the
recovery is therefore renewed by the recovery itself.
"""

import pytest

from turtleboot3_autonomous_nav.control import (
    FORWARD,
    LEFT,
    RECOVER,
    RIGHT,
    SOFT_LEFT,
    SOFT_RIGHT,
    action_commands_translation,
)


@pytest.mark.parametrize('action', [FORWARD, SOFT_LEFT, SOFT_RIGHT])
def test_actions_that_drive_forward_are_expected_to_translate(action):
    assert action_commands_translation(action)


@pytest.mark.parametrize('action', [LEFT, RIGHT, RECOVER])
def test_turning_in_place_is_not_expected_to_translate(action):
    """These actions command zero linear velocity, so no motion is expected."""
    assert not action_commands_translation(action)


def test_an_unknown_action_is_not_expected_to_translate():
    """An unmapped action publishes no motion, so it cannot stall."""
    assert not action_commands_translation(99)
