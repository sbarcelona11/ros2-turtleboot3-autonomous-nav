"""Heading estimation shared by the mapper, the observation and the planner.

Wheel odometry integrates encoder counts, so a wheel that slips against a wall
corrupts the heading permanently.  Measured against Gazebo's true pose in the
middle of a mission, odometry was 65.1 degrees off while the IMU was 10.2
degrees off.  Since every LiDAR ray is placed using this angle, a heading
error rotates and smears the whole map, which is what a drifting mission looks
like on screen.

Position still comes from odometry: correcting it needs a second position
source, while the heading only needs the gyroscope the robot already carries.
"""

from __future__ import annotations

import math


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Return the rotation about the vertical axis carried by a quaternion."""
    return math.atan2(
        2.0 * (float(w) * float(z) + float(x) * float(y)),
        1.0 - 2.0 * (float(y) * float(y) + float(z) * float(z)),
    )


def roll_pitch_from_quaternion(
    x: float, y: float, z: float, w: float
) -> tuple[float, float]:
    """Return the attitude off the horizontal carried by a quaternion.

    A differential-drive robot on flat ground stays near zero on both axes,
    so a large value means it has tipped and can no longer execute actions.
    """
    roll = math.atan2(
        2.0 * (float(w) * float(x) + float(y) * float(z)),
        1.0 - 2.0 * (float(x) * float(x) + float(y) * float(y)),
    )
    sine = 2.0 * (float(w) * float(y) - float(z) * float(x))
    pitch = math.asin(max(-1.0, min(1.0, sine)))
    return roll, pitch


def fused_heading(odom_yaw: float, imu_yaw: float | None) -> float:
    """Return the heading to use, preferring the gyroscope over the wheels.

    Odometry remains the fallback so control still works before the first IMU
    message arrives, or if the sensor starts reporting unusable values.
    """
    odom = float(odom_yaw)
    if not math.isfinite(odom):
        raise ValueError('odometry heading must be finite')
    if imu_yaw is None:
        return odom
    imu = float(imu_yaw)
    return imu if math.isfinite(imu) else odom
