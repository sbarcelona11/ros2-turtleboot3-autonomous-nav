"""Reduce the energy of contacts, and notice when one has ended the mission.

Driving at full speed until 0.20 m and then stopping dead puts the robot into
walls hard enough to slip its wheels - which corrupts odometry - and, with
this physics engine, hard enough to tip it over: measured on the robot,
roll 94.7 and pitch 90.0 degrees, lying on its side while the planner kept
issuing turns for minutes.
"""

import math

import pytest

from turtleboot3_autonomous_nav.control import (
    ControlConfig,
    approach_speed_scale,
    is_tipped,
)


def test_full_speed_is_allowed_with_open_space_ahead():
    assert approach_speed_scale(3.0, ControlConfig()) == pytest.approx(1.0)


def test_speed_falls_to_zero_at_the_braking_threshold():
    config = ControlConfig()
    assert approach_speed_scale(config.stop_distance, config) == pytest.approx(0.0)


def test_speed_tapers_between_the_two_distances():
    """A step change is what slams the robot into the wall."""
    config = ControlConfig()
    middle = (config.stop_distance + config.slow_distance) / 2.0
    scale = approach_speed_scale(middle, config)
    assert 0.0 < scale < 1.0


def test_closer_means_slower():
    config = ControlConfig()
    near = approach_speed_scale(config.stop_distance + 0.05, config)
    far = approach_speed_scale(config.stop_distance + 0.25, config)
    assert near < far


def test_an_unreadable_range_is_treated_as_blocked():
    """An unknown clearance must not authorise full speed."""
    assert approach_speed_scale(math.nan, ControlConfig()) == pytest.approx(0.0)


def test_a_forward_command_is_slowed_by_a_near_obstacle():
    """The taper only helps if the emitted twist actually carries it."""
    from turtleboot3_autonomous_nav.control import FORWARD, safe_twist

    config = ControlConfig()
    import numpy as np

    open_space = safe_twist(FORWARD, np.asarray([3.0, 3.0, 3.0]), False, config)
    closing = safe_twist(
        FORWARD,
        np.asarray([config.stop_distance + 0.05, 3.0, 3.0]),
        False,
        config,
    )
    assert closing.linear_x < open_space.linear_x
    assert closing.linear_x > 0.0


def test_an_upright_robot_is_not_tipped():
    assert not is_tipped(0.0, 0.0)
    assert not is_tipped(0.1, -0.1)


def test_a_robot_on_its_side_is_tipped():
    assert is_tipped(math.radians(94.7), math.radians(90.0))


def test_either_axis_alone_is_enough_to_report_a_tip():
    assert is_tipped(math.radians(80.0), 0.0)
    assert is_tipped(0.0, math.radians(-80.0))


@pytest.mark.parametrize('angle', [math.nan, math.inf])
def test_an_unreadable_attitude_is_reported_as_tipped(angle):
    """Losing the attitude reading must stop the robot, not free it."""
    assert is_tipped(angle, 0.0)
