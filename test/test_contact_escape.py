"""Turning in place against an obstacle climbs it; backing out does not.

Recorded from the simulator in the seconds before a flip, with the robot
pinned and the safety controller in charge of the command:

    t -2.52s  (-0.75,-0.12) z=+0.023  cmd=(+0.000,-1.000)  laser=0.12m@+24
    t -2.45s  (-0.74,-0.12) z=+0.029  cmd=(+0.000,-1.000)  laser=0.12m@+24

The braking is correct - no translation is commanded - and the laser is not
blind: it clips cleanly at its 0.12 m minimum and the front sector reads it.
What follows is a full-rate turn in place with the nose resting on the
obstacle, and z climbing steadily from its 0.010 m spawn height.  The wheels
lever the robot up onto whatever it is touching until it goes over: 174
degrees of roll in one run, 88 in another, and the mission ends on
robot_tipped every time.

Backing out is already implemented, but only reachable through `stalled` or an
explicit RECOVER, and the stall timer needs 20 seconds to fire.  The climb
takes under three.  Contact ahead has to trigger the escape by itself.
"""

import numpy as np
import pytest

from turtleboot3_autonomous_nav.control import (
    FORWARD,
    LEFT,
    RIGHT,
    ControlConfig,
    safe_twist,
)


def _sectors(front, left=1.0, right=1.0, rear=1.0):
    return np.asarray([front, left, right, rear], dtype=float)


def test_contact_is_closer_than_the_braking_threshold():
    """Contact must name touching, not merely being close enough to brake."""
    config = ControlConfig()
    assert 0.0 < config.contact_distance < config.stop_distance


def test_a_robot_touching_something_ahead_backs_out_rather_than_turning():
    """A turn in place with the nose loaded is what lifts the robot."""
    config = ControlConfig()
    decision = safe_twist(LEFT, _sectors(0.12, rear=1.0), False, config)
    assert decision.linear_x < 0.0


def test_backing_out_of_a_contact_is_reported_as_recovery():
    """The mission has to see the escape, not read it as ordinary driving."""
    config = ControlConfig()
    decision = safe_twist(LEFT, _sectors(0.12, rear=1.0), False, config)
    assert decision.recovery
    assert decision.intervention


def test_a_robot_that_cannot_back_out_turns_gently_instead_of_at_full_rate():
    """Halving the rate halves the torque levering the robot onto the obstacle."""
    config = ControlConfig()
    decision = safe_twist(LEFT, _sectors(0.12, rear=0.10), False, config)
    assert decision.linear_x == 0.0
    assert 0.0 < abs(decision.angular_z) <= config.soft_turn_speed
    assert abs(decision.angular_z) < config.turn_speed


def test_an_ordinary_turn_in_open_space_is_left_alone():
    """The escape must not slow down turning where there is nothing to climb."""
    config = ControlConfig()
    decision = safe_twist(LEFT, _sectors(1.5), False, config)
    assert decision.linear_x == 0.0
    assert decision.angular_z == pytest.approx(config.turn_speed)
    assert not decision.recovery


def test_an_obstacle_within_braking_range_but_not_touching_still_turns_freely():
    """Turning away is the right move while there is still room to do it."""
    config = ControlConfig()
    decision = safe_twist(RIGHT, _sectors(0.18), False, config)
    assert decision.linear_x == 0.0
    assert abs(decision.angular_z) == pytest.approx(config.turn_speed)


def test_driving_into_a_contact_backs_out_too():
    """A forward action against a touched obstacle must not spin either."""
    config = ControlConfig()
    decision = safe_twist(FORWARD, _sectors(0.12, rear=1.0), False, config)
    assert decision.linear_x < 0.0
    assert decision.recovery


def test_the_escape_never_reverses_into_something_behind():
    """Trading a contact ahead for one behind is not an escape."""
    config = ControlConfig()
    decision = safe_twist(FORWARD, _sectors(0.12, rear=0.10), False, config)
    assert decision.linear_x == 0.0


def test_the_reverse_respects_the_configured_reverse_speed():
    """Backing out blind at driving speed trades one collision for another."""
    config = ControlConfig()
    decision = safe_twist(LEFT, _sectors(0.12, rear=1.0), False, config)
    assert abs(decision.linear_x) == pytest.approx(config.reverse_speed)
