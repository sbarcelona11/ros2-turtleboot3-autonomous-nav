"""Slow down for obstacles beside the robot, not only ahead of it.

The speed taper only looked forward, but the two moving cylinders reach the
robot from the side, where it was still travelling at full speed.  With this
physics engine a side impact tips the robot: measured repeatedly at roll 41
and pitch 32 degrees within a minute of starting, which ends the mission.

A side obstacle slows the robot rather than stopping it - a corridor has walls
on both sides, and braking to a halt there would strand the mission.
"""

import numpy as np
import pytest

from turtleboot3_autonomous_nav.control import (
    FORWARD,
    ControlConfig,
    safe_twist,
    side_speed_scale,
)


def _ranges(front=3.0, left=3.0, right=3.0, rear=3.0):
    return np.asarray([front, left, right, rear], dtype=float)


def test_open_space_beside_the_robot_allows_full_speed():
    assert side_speed_scale(_ranges(), ControlConfig()) == pytest.approx(1.0)


def test_an_obstacle_alongside_slows_the_robot():
    config = ControlConfig()
    assert side_speed_scale(_ranges(left=0.2), config) < 1.0


def test_a_side_obstacle_never_brings_the_robot_to_a_halt():
    """A corridor has walls on both sides; stopping there ends the mission."""
    config = ControlConfig()
    scale = side_speed_scale(_ranges(left=0.01, right=0.01), config)
    assert scale >= config.side_speed_floor
    assert scale > 0.0


def test_the_closer_side_decides():
    config = ControlConfig()
    assert side_speed_scale(_ranges(left=0.2, right=3.0), config) == pytest.approx(
        side_speed_scale(_ranges(left=3.0, right=0.2), config)
    )


def test_an_unreadable_side_is_treated_as_close():
    """An unknown clearance must not authorise full speed past it."""
    config = ControlConfig()
    assert side_speed_scale(_ranges(left=float('nan')), config) < 1.0


def test_a_forward_command_is_slowed_by_an_obstacle_alongside():
    """The taper only helps if the emitted twist carries it."""
    config = ControlConfig()
    open_space = safe_twist(FORWARD, _ranges(), False, config)
    alongside = safe_twist(FORWARD, _ranges(left=0.2), False, config)
    assert 0.0 < alongside.linear_x < open_space.linear_x


def test_the_nominal_speed_leaves_room_against_this_physics_engine():
    """Momentum is what tips the robot when a moving obstacle reaches it."""
    assert ControlConfig().linear_speed <= 0.10
