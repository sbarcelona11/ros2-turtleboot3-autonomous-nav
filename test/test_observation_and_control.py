import numpy as np

from turtleboot3_autonomous_nav.observation import LIDAR_MAX_RANGE, build_observation
from turtleboot3_autonomous_nav.control import (
    FORWARD,
    SOFT_LEFT,
    ControlConfig,
    control_sector_ranges,
    safe_twist,
    stale_twist,
)


def test_front_obstacle_overrides_forward_action():
    """A forward command must never pass a LiDAR stop-distance obstacle."""
    result = safe_twist(
        FORWARD,
        np.array([0.15, 2.0, 2.0]),
        False,
        ControlConfig(stop_distance=0.20),
    )

    assert result.linear_x == 0.0
    assert result.intervention is True


def test_stop_distance_boundary_overrides_forward_action():
    """A return exactly at the stop distance is not safe clearance."""
    result = safe_twist(
        FORWARD,
        np.array([0.20, 2.0, 2.0]),
        False,
        ControlConfig(stop_distance=0.20),
    )

    assert result.linear_x == 0.0
    assert result.intervention is True


def test_invalid_front_scan_sector_remains_blocked_despite_range_max():
    """A fresh scan with no positive front return cannot imply open space."""
    sectors = control_sector_ranges(
        np.array([2.0, np.nan, 0.0, np.nan, 2.0]),
        angle_min=-np.pi / 2.0,
        angle_increment=np.pi / 4.0,
        range_max=3.5,
    )
    result = safe_twist(FORWARD, sectors, False, ControlConfig(stop_distance=0.20))

    assert sectors[0] == 0.0
    assert result.linear_x == 0.0
    assert result.intervention is True


def test_emergency_stop_turns_toward_the_clearer_side():
    """A stopped robot turns right when its right LiDAR sector has more room."""
    result = safe_twist(
        FORWARD,
        np.array([0.15, 0.4, 1.2]),
        False,
        ControlConfig(stop_distance=0.20),
    )

    assert result.angular_z < 0.0
    assert result.intervention is True


def test_stalled_progress_forces_recovery_toward_clearer_side():
    """Stall detection supersedes forward travel with a marked recovery turn."""
    result = safe_twist(
        FORWARD,
        np.array([1.0, 1.4, 0.4]),
        True,
        ControlConfig(stop_distance=0.20),
    )

    assert result.linear_x == 0.0
    assert result.angular_z > 0.0
    assert result.recovery is True


def test_action_speeds_are_clamped_to_configured_limits():
    """A nominal soft-left command cannot exceed either configured limit."""
    result = safe_twist(
        SOFT_LEFT,
        np.array([2.0, 2.0, 2.0]),
        False,
        ControlConfig(
            linear_speed=1.0,
            soft_turn_speed=1.0,
            max_linear_speed=0.20,
            max_angular_speed=0.30,
        ),
    )

    assert result.linear_x == 0.20
    assert result.angular_z == 0.30
    assert result.intervention is False


def test_a_scan_of_any_length_reduces_to_twelve_sector_minima():
    """The LiDAR field is fixed width whatever the simulated scan resolution."""
    observation = build_observation(
        np.arange(1.0, 25.0),
        np.full((5, 7), -1, dtype=np.int8),
        (2, 3),
        0.0,
        linear_velocity=0.12,
        angular_velocity=-0.34,
    )

    assert np.allclose(
        observation[:12], np.clip(np.arange(1.0, 24.0, 2.0) / LIDAR_MAX_RANGE, 0.0, 1.0)
    )
    assert np.allclose(observation[12:20], np.ones(8))
    assert np.allclose(observation[-2:], np.array([0.12, -0.34]))


def test_stale_sensor_data_publishes_a_zero_intervention_decision():
    """A stale scan is a fail-safe stop, not a reuse of the last velocity."""
    result = stale_twist()

    assert result.linear_x == 0.0
    assert result.angular_z == 0.0
    assert result.intervention is True
    assert result.recovery is False
