"""ROS 2 adapter for greedy DQN exploration inference."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import math
import time

from turtleboot3_autonomous_nav.dqn import DQNPolicy, load_checkpoint
from turtleboot3_autonomous_nav.observation import (
    has_observation_contract,
    require_observation_contract,
)


def greedy_action_for_observation(
    policy: DQNPolicy, observation: Sequence[float]
) -> int:
    """Select the deployment action without applying training exploration."""
    return policy.select_action(observation)


def main(args: list[str] | None = None) -> None:
    """Run bounded mission inference only after loading a trained checkpoint."""
    import rclpy
    from rclpy.node import Node
    from rclpy.clock import Clock, ClockType
    from std_msgs.msg import Float32, Float32MultiArray, Int32
    from std_srvs.srv import SetBool

    class DQNExplorer(Node):
        """Convert each complete DQN observation into one discrete action."""

        def __init__(self) -> None:
            super().__init__('dqn_explorer')
            self.declare_parameter('model_path', '')
            self.declare_parameter('observation_size', 90)
            self.declare_parameter('action_count', 5)
            self.declare_parameter('max_steps', 500)
            self.declare_parameter('target_coverage', 0.75)
            self.declare_parameter('control_timeout_seconds', 10.0)
            observation_size = int(self.get_parameter('observation_size').value)
            action_count = int(self.get_parameter('action_count').value)
            model_path = str(self.get_parameter('model_path').value)
            self._policy = self._load_policy(
                model_path, observation_size, action_count
            )
            self._max_steps = int(self.get_parameter('max_steps').value)
            self._target_coverage = float(self.get_parameter('target_coverage').value)
            if self._max_steps <= 0 or not 0.0 < self._target_coverage <= 1.0:
                raise ValueError('mission max_steps and target_coverage must be positive and bounded')
            self._control_timeout = float(self.get_parameter('control_timeout_seconds').value)
            if not math.isfinite(self._control_timeout) or self._control_timeout <= 0:
                raise ValueError('control_timeout_seconds must be finite and positive')
            self._steps = 0
            self._finished = False
            self._enabled = False
            self._mission_start_ns: int | None = None
            self._reachable_coverage = 0.0
            self._last_reported_coverage_percent = -1
            self._control_future = None
            self._control_request = True
            self._control_deadline = time.monotonic() + self._control_timeout
            self._controller_enable = self.create_client(SetBool, '/safe_motion_controller/enable')
            self._publisher = self.create_publisher(Int32, '/exploration_action', 10)
            self.create_subscription(
                Float32MultiArray, '/dqn_observation', self._on_observation, 10
            )
            self.create_subscription(
                Float32, '/coverage_metrics', self._on_map_coverage, 10
            )
            self.create_subscription(
                Float32, '/coverage_reachable', self._on_reachable_coverage, 10
            )
            self.create_timer(0.1, self._poll_control, clock=Clock(clock_type=ClockType.STEADY_TIME))

        def _load_policy(
            self, model_path: str, observation_size: int, action_count: int
        ) -> DQNPolicy:
            if not model_path.strip():
                raise ValueError('model_path must name a trained checkpoint; mission is disabled')

            policy, config, _ = load_checkpoint(Path(model_path))
            require_observation_contract(config)
            if policy.observation_size != observation_size:
                raise ValueError(
                    'checkpoint observation size does not match observation_size '
                    'parameter'
                )
            if policy.action_count != action_count:
                raise ValueError(
                    'checkpoint action count does not match action_count parameter'
                )
            return policy

        def _on_observation(self, message: Float32MultiArray) -> None:
            if self._finished or not self._enabled:
                return
            if (
                not message.layout.dim
                or not has_observation_contract(message.layout.dim[0].label)
            ):
                self.get_logger().error('Rejected DQN observation: incompatible observation contract')
                return
            try:
                action = greedy_action_for_observation(self._policy, message.data)
            except ValueError as error:
                self.get_logger().error(f'Rejected DQN observation: {error}')
                return
            self._publisher.publish(Int32(data=action))
            self._steps += 1
            if self._steps >= self._max_steps:
                self._finish('max_steps')

        def _on_map_coverage(self, message: Float32) -> None:
            if not math.isfinite(message.data):
                return
            percentage = int(message.data * 100.0 + 1e-6)
            if percentage > self._last_reported_coverage_percent:
                self._last_reported_coverage_percent = percentage
                self.get_logger().info(f'Mapa descubierto: {percentage}%')

        def _on_reachable_coverage(self, message: Float32) -> None:
            if not math.isfinite(message.data):
                return
            self._reachable_coverage = min(max(float(message.data), 0.0), 1.0)
            if self._reachable_coverage >= self._target_coverage:
                self._finish('target_coverage')

        def _on_coverage(self, message: Float32) -> None:
            """Preserve the mission-termination callback contract."""
            self._on_reachable_coverage(message)

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
                self.get_logger().error('Controller enable/stop acknowledgement timed out.')
                if not self._finished:
                    self._finish('controller unavailable')
                else:
                    self._control_request = None
                    future, self._control_future = self._control_future, None
                    if future is not None:
                        future.cancel()
                return
            if self._control_future is not None or not self._controller_enable.service_is_ready():
                return
            enabled = self._control_request
            self._control_request = None
            self._control_deadline = time.monotonic() + self._control_timeout
            self._control_future = self._controller_enable.call_async(SetBool.Request(data=enabled))
            self._control_future.add_done_callback(lambda result: self._on_control_reply(result, enabled))

        def _on_control_reply(self, future, enabled) -> None:
            if future is not self._control_future:
                return
            self._control_future = None
            try:
                response = future.result()
                if response is None or not response.success:
                    raise RuntimeError(getattr(response, 'message', 'request rejected'))
                self._enabled = enabled and not self._finished
                if self._enabled and self._mission_start_ns is None:
                    self._mission_start_ns = self.get_clock().now().nanoseconds
            except Exception as error:
                self.get_logger().error(f'Controller request failed: {error}')
                self._finish('controller request failed')

    rclpy.init(args=args)
    node = None
    try:
        node = DQNExplorer()
        rclpy.spin(node)
    finally:
        if node is not None:
            if rclpy.ok() and not node._finished:
                node._finish('shutdown')
                if node._control_future is not None:
                    rclpy.spin_until_future_complete(node, node._control_future, timeout_sec=1.0)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
