"""ROS 2 adapter for deterministic frontier exploration.

It publishes the same `/exploration_action` interface as the learned policy,
so the safety controller, the mapper and the rest of the mission are
unchanged and `/cmd_vel` still has a single publisher.  The difference is
only in how the action is chosen: a search over the map instead of a network.
"""

from __future__ import annotations

import math
import time


def main(args: list[str] | None = None) -> None:
    """Drive a bounded mission from map-based frontier planning."""
    import numpy as np
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Odometry
    from sensor_msgs.msg import Imu
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32, Int32
    from std_srvs.srv import SetBool

    from turtleboot3_autonomous_nav.control import ControlConfig
    from turtleboot3_autonomous_nav.pose_fusion import (
        fused_heading,
        yaw_from_quaternion,
    )
    from turtleboot3_autonomous_nav.frontier import (
        action_for_heading,
        clearance_cells_for,
        nearest_frontier_step,
    )

    class FrontierExplorer(Node):
        """Steer towards the closest reachable unknown cell on every map update."""

        def __init__(self) -> None:
            super().__init__('frontier_explorer')
            self.declare_parameter('max_steps', 500)
            self.declare_parameter('target_coverage', 0.98)
            self.declare_parameter('control_timeout_seconds', 10.0)
            # The Burger is about 0.105 m in radius; at 0.05 m per cell three
            # cells keep a planned path off the walls the controller brakes for.
            # Derived from the distance the safety controller brakes at, so a
            # planned path never enters the range that triggers it.
            self.declare_parameter('stop_distance', ControlConfig().stop_distance)
            self._stop_distance = float(self.get_parameter('stop_distance').value)
            # Aiming a cell ahead makes the heading swing on every map update
            # and the robot pivots instead of driving; 0.4 m of lookahead at
            # 0.05 m per cell keeps the target steady.
            self.declare_parameter('lookahead_cells', 8)
            self._lookahead = max(1, int(self.get_parameter('lookahead_cells').value))
            self._max_steps = int(self.get_parameter('max_steps').value)
            self._target_coverage = float(self.get_parameter('target_coverage').value)
            if self._max_steps <= 0 or not 0.0 < self._target_coverage <= 1.0:
                raise ValueError(
                    'mission max_steps and target_coverage must be positive and bounded'
                )
            self._control_timeout = float(
                self.get_parameter('control_timeout_seconds').value
            )
            if not math.isfinite(self._control_timeout) or self._control_timeout <= 0:
                raise ValueError('control_timeout_seconds must be finite and positive')

            self._pose: tuple[float, float, float] | None = None
            self._imu_yaw: float | None = None
            self._steps = 0
            self._finished = False
            self._enabled = False
            self._mission_start_ns: int | None = None
            self._reachable_coverage = 0.0
            self._last_reported_coverage_percent = -1
            self._control_future = None
            self._control_request = True
            self._control_deadline = time.monotonic() + self._control_timeout
            self._controller_enable = self.create_client(
                SetBool, '/safe_motion_controller/enable'
            )
            self._publisher = self.create_publisher(Int32, '/exploration_action', 10)
            # The monitoring map, not the occupancy map: a cell resolved by a
            # distant ray is already known but has not been monitored, and it
            # is the monitoring metric the mission is scored on.
            self.create_subscription(OccupancyGrid, '/monitoring_map', self._on_map, 10)
            self.create_subscription(Odometry, '/odom', self._on_odometry, 10)
            # Steering on a slipped odometric heading aims at the wrong cell.
            self.create_subscription(Imu, '/imu', self._on_imu, 10)
            self.create_subscription(
                Float32, '/coverage_reachable', self._on_coverage, 10
            )
            self.create_subscription(
                Float32, '/coverage_metrics', self._on_map_coverage, 10
            )
            self.create_subscription(Bool, '/robot_tipped', self._on_tipped, 10)
            self.create_timer(
                0.1, self._poll_control, clock=Clock(clock_type=ClockType.STEADY_TIME)
            )

        def _on_imu(self, message) -> None:
            orientation = message.orientation
            self._imu_yaw = yaw_from_quaternion(
                orientation.x, orientation.y, orientation.z, orientation.w
            )

        def _on_odometry(self, message: Odometry) -> None:
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            self._pose = (
                position.x,
                position.y,
                fused_heading(
                    yaw_from_quaternion(
                        orientation.x, orientation.y, orientation.z, orientation.w
                    ),
                    self._imu_yaw,
                ),
            )

        def _on_map(self, message: OccupancyGrid) -> None:
            if self._finished or not self._enabled or self._pose is None:
                return
            width = int(message.info.width)
            height = int(message.info.height)
            resolution = float(message.info.resolution)
            if width <= 0 or height <= 0 or resolution <= 0.0:
                return
            data = np.asarray(message.data, dtype=np.int8)
            if data.size != width * height:
                return
            grid = data.reshape(height, width)
            origin = message.info.origin.position
            pose_x, pose_y, yaw = self._pose
            # Odometry drift can place the robot just outside the grid; clamping
            # keeps planning available instead of dropping the mission.
            column = min(max(int((pose_x - origin.x) / resolution), 0), width - 1)
            row = min(max(int((pose_y - origin.y) / resolution), 0), height - 1)

            step = nearest_frontier_step(
                grid,
                (row, column),
                clearance_cells_for(self._stop_distance, resolution),
                self._lookahead,
            )
            if step is None:
                self._finish('coverage_complete')
                return
            target_x = origin.x + (step[1] + 0.5) * resolution
            target_y = origin.y + (step[0] + 0.5) * resolution
            desired = math.atan2(target_y - pose_y, target_x - pose_x)
            self._publisher.publish(Int32(data=action_for_heading(yaw, desired)))
            self._steps += 1
            if self._steps >= self._max_steps:
                self._finish('max_steps')

        def _on_tipped(self, message) -> None:
            if message.data:
                self._finish('robot_tipped')

        def _on_coverage(self, message: Float32) -> None:
            if not math.isfinite(message.data):
                return
            self._reachable_coverage = min(max(float(message.data), 0.0), 1.0)
            if self._reachable_coverage >= self._target_coverage:
                self._finish('target_coverage')

        def _on_map_coverage(self, message: Float32) -> None:
            if not math.isfinite(message.data):
                return
            percentage = int(message.data * 100.0 + 1e-6)
            if percentage > self._last_reported_coverage_percent:
                self._last_reported_coverage_percent = percentage
                self.get_logger().info(f'Mapa descubierto: {percentage}%')

        def _finish(self, reason: str) -> None:
            if self._finished:
                return
            self._finished = True
            self._enabled = False
            start_ns = self._mission_start_ns
            elapsed = 0.0 if start_ns is None else max(
                0.0, (self.get_clock().now().nanoseconds - start_ns) / 1_000_000_000.0
            )
            self.get_logger().info(
                f'Mission ended: {reason}; '
                f'reachable_coverage={self._reachable_coverage:.2%}; '
                f'simulated_time={elapsed:.2f}s; steps={self._steps}.'
            )
            future, self._control_future = self._control_future, None
            if future is not None:
                future.cancel()
            self._control_request = False
            self._control_deadline = time.monotonic() + self._control_timeout
            self._poll_control()

        def _poll_control(self) -> None:
            if self._control_request is None and self._control_future is None:
                return
            if time.monotonic() >= self._control_deadline:
                self.get_logger().error(
                    'Controller enable/stop acknowledgement timed out.'
                )
                if not self._finished:
                    self._finish('controller unavailable')
                else:
                    self._control_request = None
                    future, self._control_future = self._control_future, None
                    if future is not None:
                        future.cancel()
                return
            if (
                self._control_future is not None
                or not self._controller_enable.service_is_ready()
            ):
                return
            enabled = self._control_request
            self._control_request = None
            self._control_future = self._controller_enable.call_async(
                SetBool.Request(data=bool(enabled))
            )
            self._control_future.add_done_callback(
                lambda future: self._on_control_reply(future, bool(enabled))
            )

        def _on_control_reply(self, future, enabled: bool) -> None:
            self._control_future = None
            try:
                response = future.result()
                if response is None or not response.success:
                    raise RuntimeError('controller rejected the request')
            except Exception as error:
                self.get_logger().error(f'Controller request failed: {error}')
                if not self._finished:
                    self._finish('controller unavailable')
                return
            self._enabled = enabled and not self._finished
            if self._enabled and self._mission_start_ns is None:
                self._mission_start_ns = self.get_clock().now().nanoseconds

    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
