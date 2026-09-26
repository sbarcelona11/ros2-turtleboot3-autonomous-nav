"""ROS 2 adapter that turns current odometry and LiDAR into a coverage map."""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np
import rclpy
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

from turtleboot3_autonomous_nav.grid_mapping import OccupancyGridModel
from turtleboot3_autonomous_nav.pose_fusion import fused_heading, yaw_from_quaternion
from turtleboot3_autonomous_nav.reset_provenance import (
    is_post_reset_timestamp,
    source_timestamp_ns,
)


def should_process_scan(
    now_ns: int, last_ns: int | None, min_period_ns: int
) -> bool:
    """Whether to integrate this scan, given when the last one was integrated.

    Ray tracing costs tens of milliseconds and scans arrive far faster than
    the map needs updating.  On a single-threaded executor an unthrottled
    callback starves the reset service, which aborts training.  A backwards
    clock jump, which a simulation reset produces, always integrates.
    """
    if int(min_period_ns) <= 0 or last_ns is None:
        return True
    elapsed = int(now_ns) - int(last_ns)
    return elapsed < 0 or elapsed >= int(min_period_ns)


class CoverageMapper(Node):
    """Publish an odometric occupancy map only after scan and odometry arrive."""

    def __init__(self) -> None:
        super().__init__("coverage_mapper")
        self.declare_parameter("map_width", 400)
        self.declare_parameter("map_height", 400)
        self.declare_parameter("map_resolution", 0.05)
        self.declare_parameter("map_origin_x", -10.0)
        self.declare_parameter("map_origin_y", -10.0)
        self.declare_parameter("occupied_threshold", 1)
        # A cell counts as monitored only when observed from within this many
        # metres; a non-positive value falls back to plain line of sight.
        self.declare_parameter("inspection_radius", 1.5)
        # Scans arrive far faster than the map needs updating; integrating
        # every one starves the reset service on a single-threaded executor.
        self.declare_parameter("max_scan_rate", 8.0)

        inspection_radius = float(self.get_parameter("inspection_radius").value)
        self._grid = OccupancyGridModel(
            self.get_parameter("map_width").value,
            self.get_parameter("map_height").value,
            self.get_parameter("map_resolution").value,
            (
                self.get_parameter("map_origin_x").value,
                self.get_parameter("map_origin_y").value,
            ),
            self.get_parameter("occupied_threshold").value,
            inspection_radius=inspection_radius if inspection_radius > 0.0 else None,
        )
        self._latest_pose: tuple[float, float, float] | None = None
        self._imu_yaw: float | None = None
        self._reset_cutoff_ns: int | None = None
        self._reset_epoch = 0
        max_scan_rate = float(self.get_parameter("max_scan_rate").value)
        self._min_scan_period_ns = (
            int(1_000_000_000 / max_scan_rate) if max_scan_rate > 0.0 else 0
        )
        self._last_scan_ns: int | None = None
        self._map_publisher = self.create_publisher(OccupancyGrid, "/coverage_map", 10)
        self._metrics_publisher = self.create_publisher(
            Float32, "/coverage_metrics", 10
        )
        # Discovery reward needs the stable whole-grid fraction; mission
        # progress needs the fraction of what the robot can still reach.
        self._reachable_publisher = self.create_publisher(
            Float32, "/coverage_reachable", 10
        )
        # What still needs monitoring, as its own map: a frontier search over
        # plain occupancy runs out of targets while much of the arena has
        # never been approached.
        self._monitoring_publisher = self.create_publisher(
            OccupancyGrid, "/monitoring_map", 10
        )
        self.create_subscription(Odometry, "/odom", self._on_odometry, 10)
        self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        # Heading comes from the gyroscope: wheels that slip against a wall
        # corrupt the odometric angle, and every ray is placed with it.
        self.create_subscription(Imu, "/imu", self._on_imu, 10)
        self.create_service(Trigger, "/coverage_mapper/reset", self._on_reset)

    def _on_reset(
        self, _: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Clear map state and acknowledge the temporal cutoff for the episode."""
        self._grid.reset()
        self._latest_pose = None
        self._last_scan_ns = None
        self._reset_epoch += 1
        self._metrics_publisher.publish(Float32(data=0.0))
        self._reachable_publisher.publish(Float32(data=0.0))
        # This successful response establishes the boundary, after all reset
        # work. Sensor headers cannot serve as provenance: reset.all rewinds them.
        self._reset_cutoff_ns = time.time_ns()
        response.success = True
        response.message = json.dumps(
            {"cutoff_ns": self._reset_cutoff_ns, "epoch": self._reset_epoch},
            sort_keys=True,
        )
        return response

    def _on_imu(self, message: Imu, info: Any) -> None:
        if not self._is_post_reset_message(info):
            return
        orientation = message.orientation
        self._imu_yaw = yaw_from_quaternion(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

    def _on_odometry(self, message: Odometry, info: Any) -> None:
        if not self._is_post_reset_message(info):
            return
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        self._latest_pose = (
            position.x,
            position.y,
            fused_heading(
                yaw_from_quaternion(
                    orientation.x, orientation.y, orientation.z, orientation.w
                ),
                self._imu_yaw,
            ),
        )

    def _on_scan(self, message: LaserScan, info: Any) -> None:
        if not self._is_post_reset_message(info):
            return
        if self._latest_pose is None:
            return
        # Simulated time, so the map integrates at a rate tied to how far the
        # robot has actually moved rather than to the simulator's speed.
        now_ns = self.get_clock().now().nanoseconds
        if not should_process_scan(
            now_ns, self._last_scan_ns, self._min_scan_period_ns
        ):
            return
        self._last_scan_ns = now_ns

        self._grid.update_scan(
            self._latest_pose,
            np.asarray(message.ranges, dtype=float),
            message.angle_min,
            message.angle_increment,
            message.range_max,
        )
        self._map_publisher.publish(self._grid.to_message(message.header.stamp))
        monitoring = self._grid.to_message(message.header.stamp)
        monitoring.data = self._grid.monitoring_grid().reshape(-1).tolist()
        self._monitoring_publisher.publish(monitoring)
        self._metrics_publisher.publish(Float32(data=self._grid.coverage_fraction()))
        self._reachable_publisher.publish(
            Float32(data=self._grid.reachable_coverage_fraction())
        )

    def _is_post_reset_message(self, info: Any) -> bool:
        """Reject a delayed message emitted before the mapper reset cutoff."""
        return is_post_reset_timestamp(source_timestamp_ns(info), self._reset_cutoff_ns)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CoverageMapper()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
