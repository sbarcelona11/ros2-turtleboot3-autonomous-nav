"""ROS safety wrapper and sole ``/cmd_vel`` publisher for exploration actions."""

from __future__ import annotations

import math
import json
import time
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float32, Int32
from std_srvs.srv import SetBool

from turtleboot3_autonomous_nav.reset_provenance import (
    is_post_reset_timestamp,
    source_timestamp_ns,
)

from turtleboot3_autonomous_nav.pose_fusion import roll_pitch_from_quaternion
from turtleboot3_autonomous_nav.control import (
    RECOVER,
    ControlConfig,
    TwistDecision,
    action_commands_translation,
    control_sector_ranges,
    is_tipped,
    safe_twist,
    stale_twist,
)


class SafeMotionController(Node):
    """Turn policy actions into fail-safe velocity commands at a fixed rate."""

    def __init__(self) -> None:
        super().__init__('safe_motion_controller')
        self.declare_parameter('stop_distance', 0.20)
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('soft_turn_speed', 0.5)
        self.declare_parameter('turn_speed', 1.0)
        self.declare_parameter('emergency_turn_speed', 1.0)
        self.declare_parameter('recovery_turn_speed', 1.2)
        self.declare_parameter('max_linear_speed', 0.22)
        self.declare_parameter('max_angular_speed', 1.5)
        self.declare_parameter('sensor_timeout', 0.5)
        self.declare_parameter('action_timeout', 1.0)
        self.declare_parameter('progress_distance', 0.03)
        self.declare_parameter('progress_timeout', 3.0)
        self.declare_parameter('recovery_duration', 2.0)
        self.declare_parameter('coverage_delta', 0.0001)
        self.declare_parameter('control_rate', 10.0)

        self._config = ControlConfig(
            stop_distance=float(self.get_parameter('stop_distance').value),
            linear_speed=float(self.get_parameter('linear_speed').value),
            soft_turn_speed=float(self.get_parameter('soft_turn_speed').value),
            turn_speed=float(self.get_parameter('turn_speed').value),
            emergency_turn_speed=float(
                self.get_parameter('emergency_turn_speed').value
            ),
            recovery_turn_speed=float(
                self.get_parameter('recovery_turn_speed').value
            ),
            max_linear_speed=float(self.get_parameter('max_linear_speed').value),
            max_angular_speed=float(self.get_parameter('max_angular_speed').value),
        )
        self._sensor_timeout_ns = int(
            float(self.get_parameter('sensor_timeout').value) * 1_000_000_000
        )
        self._action_timeout_ns = int(
            float(self.get_parameter('action_timeout').value) * 1_000_000_000
        )
        self._progress_distance = float(self.get_parameter('progress_distance').value)
        self._progress_timeout_ns = int(
            float(self.get_parameter('progress_timeout').value) * 1_000_000_000
        )
        self._coverage_delta = float(self.get_parameter('coverage_delta').value)
        self._recovery_duration_ns = int(
            float(self.get_parameter('recovery_duration').value) * 1_000_000_000
        )
        if self._recovery_duration_ns <= 0:
            raise ValueError('recovery_duration must be positive')
        now = self._now_ns()
        self._enabled = False
        self._reset_cutoff_ns = time.time_ns()
        self._reset_epoch = 0
        self._scan: LaserScan | None = None
        self._scan_time_ns: int | None = None
        self._action = RECOVER
        self._action_time_ns: int | None = None
        self._odom_time_ns: int | None = None
        self._receipt_times: dict[str, int] = {}
        self._last_position: tuple[float, float] | None = None
        self._last_coverage: float | None = None
        self._last_progress_ns = now
        self._tipped = False
        self._recovery_started_ns: int | None = None
        self._policy_recovery_exhausted = False

        self._cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self._intervention_publisher = self.create_publisher(
            Bool, '/safety_intervention', 10
        )
        self._recovery_publisher = self.create_publisher(Bool, '/recovery_active', 10)
        # Nothing downstream can see the attitude, so a fallen robot would
        # keep receiving actions it cannot execute.
        self._tipped_publisher = self.create_publisher(Bool, '/robot_tipped', 10)
        self.create_subscription(Int32, '/exploration_action', self._on_action, 10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 10)
        self.create_subscription(Odometry, '/odom', self._on_odometry, 10)
        self.create_subscription(Float32, '/coverage_metrics', self._on_coverage, 10)
        # A tipped robot cannot execute any action; without this the
        # mission keeps steering a robot lying on its side.
        self.create_subscription(Imu, '/imu', self._on_imu, 10)
        self.create_service(SetBool, '/safe_motion_controller/enable', self._on_enable)
        control_rate = float(self.get_parameter('control_rate').value)
        self.create_timer(
            1.0 / max(control_rate, 1.0), self._on_control_timer,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

    def _on_enable(self, request: SetBool.Request, response: SetBool.Response):
        """Acknowledge zero motion and discard every prior control input."""
        self._enabled = bool(request.data)
        self._scan = None
        self._scan_time_ns = self._odom_time_ns = self._action_time_ns = None
        self._receipt_times.clear()
        self._last_position = self._last_coverage = None
        self._last_progress_ns = self._now_ns()
        self._tipped = False
        self._recovery_started_ns = None
        self._policy_recovery_exhausted = False
        self.publish_stop()
        self._reset_cutoff_ns = time.time_ns()
        self._reset_epoch += 1
        response.success = True
        response.message = json.dumps({
            'cutoff_ns': self._reset_cutoff_ns, 'epoch': self._reset_epoch,
        })
        return response

    def _accepts(self, info: Any) -> bool:
        return self._enabled and is_post_reset_timestamp(
            source_timestamp_ns(info), self._reset_cutoff_ns
        )

    def _on_action(self, message: Int32, info: Any) -> None:
        if not self._accepts(info):
            return
        self._action = int(message.data)
        self._action_time_ns = self._now_ns()
        self._receipt_times['action'] = time.monotonic_ns()
        if self._action != RECOVER:
            self._policy_recovery_exhausted = False

    def _hold_progress_clock_for_recovery(self, now: int) -> None:
        """Stop the stall clock while recovery is the thing doing the turning.

        Recovery only turns, so counting its turn as lack of progress would let
        it renew the very condition that triggered it.  The hold has to be tied
        to recovery running, though, and not to any action that fails to
        translate: LEFT and RIGHT do not translate either, so a policy free to
        pick them could pin this clock at zero and keep the automatic recovery
        asleep indefinitely.  The trained agent did exactly that - 63 % of its
        decisions were turns in place - and sat against a wall for a full 3000
        step mission without recovery ever arming.
        """
        recovering = self._recovery_started_ns is not None or self._action == RECOVER
        if recovering and not action_commands_translation(self._action):
            self._last_progress_ns = now

    def _on_imu(self, message: Imu, info: Any) -> None:
        if not self._accepts(info):
            return
        orientation = message.orientation
        roll, pitch = roll_pitch_from_quaternion(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        tipped = is_tipped(roll, pitch, self._config.tip_limit)
        if tipped and not self._tipped:
            self.get_logger().error(
                f'Robot is no longer upright (roll {math.degrees(roll):.0f} deg, '
                f'pitch {math.degrees(pitch):.0f} deg); motion is stopped.'
            )
        self._tipped = tipped
        self._tipped_publisher.publish(Bool(data=bool(tipped)))

    def _on_scan(self, message: LaserScan, info: Any) -> None:
        if not self._accepts(info):
            return
        self._scan = message
        self._scan_time_ns = self._now_ns()
        self._receipt_times['scan'] = time.monotonic_ns()

    def _on_odometry(self, message: Odometry, info: Any) -> None:
        if not self._accepts(info):
            return
        self._odom_time_ns = self._now_ns()
        self._receipt_times['odom'] = time.monotonic_ns()
        position = message.pose.pose.position
        current = (position.x, position.y)
        if self._last_position is None:
            self._last_position = current
            return
        if math.dist(current, self._last_position) >= self._progress_distance:
            self._last_progress_ns = self._now_ns()
            self._last_position = current

    def _on_coverage(self, message: Float32, info: Any) -> None:
        if not self._accepts(info):
            return
        coverage = float(message.data)
        if self._last_coverage is None:
            self._last_coverage = coverage
            return
        if coverage - self._last_coverage >= self._coverage_delta:
            self._last_progress_ns = self._now_ns()
        self._last_coverage = coverage

    def _on_control_timer(self) -> None:
        now = self._now_ns()
        if self._tipped:
            self._publish_decision(stale_twist())
            return
        if self._data_is_stale(now):
            self._publish_decision(stale_twist())
            return
        assert self._scan is not None
        if self._recovery_started_ns is not None and (
            now - self._recovery_started_ns >= self._recovery_duration_ns
        ):
            self._recovery_started_ns = None
            self._last_progress_ns = now
            self._policy_recovery_exhausted = self._action == RECOVER
        if self._action == RECOVER and self._policy_recovery_exhausted:
            self._publish_decision(stale_twist())
            return
        self._hold_progress_clock_for_recovery(now)
        stalled = now - self._last_progress_ns >= self._progress_timeout_ns
        if self._recovery_started_ns is None and (stalled or self._action == RECOVER):
            self._recovery_started_ns = now
        decision = safe_twist(
            self._action,
            _control_sector_ranges(self._scan),
            self._recovery_started_ns is not None,
            self._config,
        )
        self._publish_decision(decision)

    def _data_is_stale(self, now_ns: int) -> bool:
        if not self._enabled or self._scan is None:
            return True
        wall_now = time.monotonic_ns()
        for name, stamp, timeout in (
            ('scan', self._scan_time_ns, self._sensor_timeout_ns),
            ('odom', self._odom_time_ns, self._sensor_timeout_ns),
            ('action', self._action_time_ns, self._action_timeout_ns),
        ):
            receipt = self._receipt_times.get(name)
            if stamp is None or receipt is None or not 0 <= now_ns - stamp <= timeout:
                return True
            if wall_now - receipt > timeout:
                return True
        return False

    def _publish_decision(self, decision: TwistDecision) -> None:
        command = Twist()
        command.linear.x = decision.linear_x
        command.angular.z = decision.angular_z
        self._cmd_publisher.publish(command)
        self._intervention_publisher.publish(Bool(data=decision.intervention))
        self._recovery_publisher.publish(Bool(data=decision.recovery))

    def publish_stop(self) -> None:
        """Issue a final zero command before node shutdown."""
        self._publish_decision(stale_twist())

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds


def _control_sector_ranges(message: LaserScan) -> np.ndarray:
    """Return minimum ``[front, left, right]`` clearances from a scan."""
    return control_sector_ranges(
        np.asarray(message.ranges, dtype=float),
        message.angle_min,
        message.angle_increment,
        message.range_max,
    )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SafeMotionController()
    try:
        rclpy.spin(node)
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()
