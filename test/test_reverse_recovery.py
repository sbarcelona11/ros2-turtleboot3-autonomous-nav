"""Recovery has to be able to move the robot, not only rotate it.

Three times in this project a recovery mechanism could not produce the motion
needed to recover.  The last one pinned the robot against an inner wall for
ninety seconds with 147 safety interventions and no forward progress: the
planner aimed at a frontier behind the wall, the robot pushed, the emergency
turn rotated it, and the cycle repeated.  Nothing in the system could command
reverse, so nothing could unstick it.

The LiDAR sweeps 360 degrees, so the space behind the robot is measured and
backing out can be done safely.
"""

import numpy as np
import pytest

from turtleboot3_autonomous_nav.control import (
    FORWARD,
    ControlConfig,
    control_sector_ranges,
    safe_twist,
)


def _scan(front, left, right, rear, count=360):
    """Build a scan whose four safety sectors hold the given clearances.

    Each band is narrower than its sector so no reading sits on a boundary:
    the node recomputes the angles from ``angle_min`` and the increment, and
    the rounding difference is enough to move an edge cell into the
    neighbouring sector and change its minimum.
    """
    angles = np.linspace(-np.pi, np.pi, count, endpoint=False)
    ranges = np.full(count, 10.0)
    ranges[np.abs(angles) <= np.pi / 8.0] = front
    ranges[(angles > np.pi / 4.0) & (angles <= 2.0 * np.pi / 3.0)] = left
    ranges[(angles < -np.pi / 4.0) & (angles >= -2.0 * np.pi / 3.0)] = right
    ranges[np.abs(angles) > 7.0 * np.pi / 8.0] = rear
    return ranges, float(angles[0]), float(angles[1] - angles[0])


def test_the_scan_reduces_to_four_safety_sectors():
    """The rear was never measured, so backing out could not be made safe."""
    ranges, angle_min, increment = _scan(0.5, 1.0, 2.0, 3.0)
    sectors = control_sector_ranges(ranges, angle_min, increment, 10.0)
    assert sectors.size == 4
    assert sectors[0] == pytest.approx(0.5)
    assert sectors[3] == pytest.approx(3.0)


def test_a_stalled_robot_with_clear_space_behind_backs_out():
    config = ControlConfig()
    sectors = np.asarray([0.1, 1.0, 1.0, 2.0])
    decision = safe_twist(FORWARD, sectors, True, config)
    assert decision.recovery
    assert decision.linear_x < 0.0


def test_the_reverse_stays_within_the_speed_limit():
    """Backing out blind at full speed would trade one collision for another."""
    config = ControlConfig()
    decision = safe_twist(FORWARD, np.asarray([0.1, 1.0, 1.0, 2.0]), True, config)
    assert abs(decision.linear_x) <= config.max_linear_speed


def test_a_stalled_robot_boxed_in_behind_only_turns():
    """Reversing into a wall must never be the answer."""
    config = ControlConfig()
    decision = safe_twist(FORWARD, np.asarray([0.1, 1.0, 1.0, 0.05]), True, config)
    assert decision.recovery
    assert decision.linear_x == pytest.approx(0.0)


def test_a_missing_rear_reading_is_treated_as_blocked():
    """An unmeasured rear sector reads zero, which must not authorise reverse."""
    config = ControlConfig()
    decision = safe_twist(FORWARD, np.asarray([0.1, 1.0, 1.0, 0.0]), True, config)
    assert decision.linear_x == pytest.approx(0.0)


def test_recovery_still_turns_towards_the_clearer_side_while_reversing():
    """Backing straight out leaves the robot facing the same obstacle."""
    config = ControlConfig()
    decision = safe_twist(FORWARD, np.asarray([0.1, 2.0, 0.3, 2.0]), True, config)
    assert decision.angular_z > 0.0
