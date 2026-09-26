"""ROS adapter that publishes fixed DQN observations from map, scan, and odometry."""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger

from turtleboot3_autonomous_nav.observation import build_observation, observation_label
from turtleboot3_autonomous_nav.pose_fusion import fused_heading, yaw_from_quaternion
from turtleboot3_autonomous_nav.reset_provenance import (
    is_post_reset_timestamp,
    source_timestamp_ns,
)


def should_publish_observation(
    now_ns: int, last_ns: int | None, min_period_ns: int
) -> bool:
    """Whether to publish now, given when the last observation went out.

    The policy acts once per observation, so this sets the decision rate.  A
    backwards clock jump, which an episode reset produces, always publishes so
    control resumes immediately.
    """
    if int(min_period_ns) <= 0 or last_ns is None:
        return True
    elapsed = int(now_ns) - int(last_ns)
    return elapsed < 0 or elapsed >= int(min_period_ns)


class ObservationBuilder(Node):
    """Publish an observation for each scan once matching map and odometry exist."""

    def __init__(self) -> None:
        super().__init__('observation_builder')
        self._latest_scan: np.ndarray | None = None
        self._map: OccupancyGrid | None = None
        self._position: tuple[float, float] | None = None
        self._heading = 0.0
        self._imu_yaw: float | None = None
        self._linear_velocity = 0.0
        self._angular_velocity = 0.0
        self._reset_cutoff_ns: int | None = None
        self._reset_epoch = 0
        # One observation is one decision: the simulated LiDAR runs far faster
        # than a robot at 0.15 m/s can act on, and deciding at sensor rate
        # fills the replay buffer with near-duplicate transitions.
        rate = float(self.declare_parameter('max_observation_rate', 10.0).value)
        self._min_publish_period_ns = int(1_000_000_000 / rate) if rate > 0.0 else 0
        self._last_publish_ns: int | None = None
        self._publisher = self.create_publisher(Float32MultiArray, '/dqn_observation', 10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 10)
        self.create_subscription(OccupancyGrid, '/monitoring_map', self._on_map, 10)
        self.create_subscription(Odometry, '/odom', self._on_odometry, 10)
        # The gyroscope heading survives wheel slip; odometry's does not.
        self.create_subscription(Imu, '/imu', self._on_imu, 10)
        self.create_service(Trigger, '/observation_builder/reset', self._on_reset)

    def _on_reset(self, request: Trigger.Request, response: Trigger.Response):
        """Clear cached inputs after the mapper reset and establish an epoch."""
        self._latest_scan = self._map = self._position = None
        self._last_publish_ns = None
        self._heading = 0.0
        self._linear_velocity = self._angular_velocity = 0.0
        self._reset_epoch += 1
        self._reset_cutoff_ns = time.time_ns()
        response.success = True
        response.message = json.dumps({
            'cutoff_ns': self._reset_cutoff_ns, 'epoch': self._reset_epoch,
        })
        return response

    def _accepts(self, info: Any) -> bool:
        return is_post_reset_timestamp(source_timestamp_ns(info), self._reset_cutoff_ns)

    def _on_scan(self, message: LaserScan, info: Any) -> None:
        if not self._accepts(info):
            return
        self._latest_scan = np.asarray(message.ranges, dtype=float)
        self._publish_if_ready()

    def _on_map(self, message: OccupancyGrid, info: Any) -> None:
        if not self._accepts(info):
            return
        self._map = message

    def _on_imu(self, message: Imu, info: Any) -> None:
        if not self._accepts(info):
            return
        orientation = message.orientation
        self._imu_yaw = yaw_from_quaternion(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

    def _on_odometry(self, message: Odometry, info: Any) -> None:
        if not self._accepts(info):
            return
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        self._position = (position.x, position.y)
        self._heading = fused_heading(
            yaw_from_quaternion(
                orientation.x, orientation.y, orientation.z, orientation.w
            ),
            self._imu_yaw,
        )
        self._linear_velocity = message.twist.twist.linear.x
        self._angular_velocity = message.twist.twist.angular.z

    def _publish_if_ready(self) -> None:
        if self._latest_scan is None or self._map is None or self._position is None:
            return
        # Simulated time, not wall time: the decision rate must describe how
        # far the robot drives between decisions.  Headless Gazebo runs many
        # times faster than real time, so a wall-clock limit would leave the
        # robot driving blind for over a simulated second between actions.
        now_ns = self.get_clock().now().nanoseconds
        if not should_publish_observation(
            now_ns, self._last_publish_ns, self._min_publish_period_ns
        ):
            return
        located = self._grid_and_cell(self._map, self._position)
        if located is None:
            return
        self._last_publish_ns = now_ns
        grid, cell = located
        observation = build_observation(
            self._latest_scan,
            grid,
            cell,
            self._heading,
            self._linear_velocity,
            self._angular_velocity,
        )
        message = Float32MultiArray(data=observation.tolist())
        message.layout.dim = [MultiArrayDimension(
            label=observation_label(self._reset_epoch),
            size=len(observation),
            stride=len(observation),
        )]
        self._publisher.publish(message)

    @staticmethod
    def _grid_and_cell(
        message: OccupancyGrid, position: tuple[float, float]
    ) -> tuple[np.ndarray, tuple[int, int]] | None:
        """Return the whole map and the cell holding the odometric pose.

        The cell is clamped to the map because odometry drift can place the
        robot just outside the grid, which must degrade the pose input rather
        than drop the observation the policy needs.
        """
        width = int(message.info.width)
        height = int(message.info.height)
        resolution = float(message.info.resolution)
        if width <= 0 or height <= 0 or resolution <= 0.0:
            return None
        map_data = np.asarray(message.data, dtype=np.int8)
        if map_data.size != width * height:
            return None
        origin = message.info.origin.position
        column = int(np.floor((position[0] - origin.x) / resolution))
        row = int(np.floor((position[1] - origin.y) / resolution))
        return map_data.reshape(height, width), (
            min(max(row, 0), height - 1),
            min(max(column, 0), width - 1),
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ObservationBuilder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
