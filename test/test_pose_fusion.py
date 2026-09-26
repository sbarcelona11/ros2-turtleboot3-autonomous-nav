"""Heading comes from the gyroscope, not from the wheels.

Wheel odometry integrates encoder counts, so every wall graze that makes the
wheels slip corrupts the heading permanently.  Measured against Gazebo's true
pose mid-mission: odometry was 65.1 degrees off while the IMU was 10.2 degrees
off.  A 65 degree error places every LiDAR ray in the wrong direction, which
is why the map appeared to rotate as the mission went on.
"""

import math

import pytest

from turtleboot3_autonomous_nav.pose_fusion import fused_heading, yaw_from_quaternion


def test_an_identity_quaternion_points_along_x():
    assert yaw_from_quaternion(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)


def test_a_quarter_turn_about_z_reads_as_ninety_degrees():
    half = math.sqrt(0.5)
    assert yaw_from_quaternion(0.0, 0.0, half, half) == pytest.approx(math.pi / 2.0)


def test_the_gyroscope_heading_is_preferred_over_the_wheels():
    """The wheels are the drifting source, so they lose."""
    assert fused_heading(odom_yaw=1.0, imu_yaw=0.2) == pytest.approx(0.2)


def test_the_wheels_are_used_while_no_gyroscope_reading_has_arrived():
    """Control must not wait for the first IMU message to steer."""
    assert fused_heading(odom_yaw=1.0, imu_yaw=None) == pytest.approx(1.0)


@pytest.mark.parametrize('imu_yaw', [math.nan, math.inf])
def test_an_unusable_gyroscope_reading_falls_back_to_the_wheels(imu_yaw):
    """A faulty sensor must degrade the heading, not stop the robot."""
    assert fused_heading(odom_yaw=0.7, imu_yaw=imu_yaw) == pytest.approx(0.7)


def test_a_non_finite_odometry_heading_is_rejected():
    with pytest.raises(ValueError):
        fused_heading(odom_yaw=math.nan, imu_yaw=None)
