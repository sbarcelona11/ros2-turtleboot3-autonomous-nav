"""Read-only console monitor for a running training or mission session.

The trainer only logs one line when an episode ends, which hides the progress
inside an episode.  This node subscribes to the published metrics without
publishing anything, so it can be started and stopped at any time during a run
without disturbing the episode lifecycle.

It leads with reachable coverage - the observed fraction of the region the
robot can still drive to - because that is what says how much of the mission
remains.  The raw grid fraction follows, since that is what the discovery
reward is computed from.
"""

from __future__ import annotations

import sys
import time


LAUNCHER_UPDATE_INTERVAL = 2.0
"""Seconds between console updates when stdout is not an interactive terminal."""


def should_emit(is_tty: bool, elapsed: float, interval: float) -> bool:
    """Whether to print now, given where the output goes.

    A terminal redraws one line, so every update is cheap.  Under a launcher
    each write becomes its own prefixed log line, so updates are throttled.
    """
    if not interval > 0.0:
        raise ValueError("interval must be positive")
    return True if is_tty else float(elapsed) >= float(interval)


def format_line(status: str, is_tty: bool) -> str:
    """Return the console payload for one status update."""
    return f"\r{status}" if is_tty else f"{status}\n"


def format_status(metrics: tuple[float, ...], steps_per_second: float) -> str:
    """Render one status line from the latest training metrics.

    Older trainers with only six core values or seven including reachable
    coverage still render. Intervention and turn diagnostics require both
    optional fields.
    """
    if len(metrics) < 6:
        raise ValueError("training metrics must contain six values")
    episode, step, coverage, reward, epsilon, best = metrics[:6]
    reachable = f"{metrics[6]:6.1%}" if len(metrics) > 6 else "   n/a"
    diagnostics = (
        f" | intervenciones {metrics[7]:.0f} | giros {metrics[8]:.1%}"
        if len(metrics) >= 9 else ""
    )
    return (
        f"ep {episode:4.0f} | paso {step:4.0f} | alcanzable {reachable} | "
        f"grid {coverage:8.3%} | recompensa {reward:9.1f} | "
        f"eps {epsilon:5.3f} | mejor {best:8.3%} | {steps_per_second:4.1f} pasos/s"
        f"{diagnostics}"
    )


def main(args: list[str] | None = None) -> None:
    """Print a live status line while a training or mission session runs."""
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray

    class TrainingMonitor(Node):
        """Subscribe to the metrics and print without publishing anything."""

        def __init__(self) -> None:
            super().__init__("training_monitor")
            self._interval = float(
                self.declare_parameter(
                    "update_interval", LAUNCHER_UPDATE_INTERVAL
                ).value
            )
            self._is_tty = sys.stdout.isatty()
            self._last_emit = 0.0
            self._episode = 0.0
            self._last_step = 0.0
            self._last_step_time = time.monotonic()
            self._steps_per_second = 0.0
            self.create_subscription(
                Float32MultiArray, "/training_metrics", self._on_metrics, 10
            )
            self.get_logger().info(
                "Monitor de solo lectura activo. Ctrl-C para salir sin afectar la corrida."
            )

        def _on_metrics(self, message: Float32MultiArray) -> None:
            metrics = tuple(float(value) for value in message.data)
            if len(metrics) < 6:
                return
            self._update_rate(metrics[1])
            now = time.monotonic()
            new_episode = metrics[0] != self._episode
            if not new_episode and not should_emit(
                self._is_tty, now - self._last_emit, self._interval
            ):
                return
            self._last_emit = now
            status = format_status(metrics, self._steps_per_second)
            if new_episode and self._is_tty:
                # Keep the finished episode's last line visible above the new one.
                sys.stdout.write("\n")
            self._episode = metrics[0]
            sys.stdout.write(format_line(status, self._is_tty))
            sys.stdout.flush()

        def _update_rate(self, step: float) -> None:
            now = time.monotonic()
            elapsed = now - self._last_step_time
            if step > self._last_step and elapsed > 0.0:
                self._steps_per_second = (step - self._last_step) / elapsed
            self._last_step = step
            self._last_step_time = now

    rclpy.init(args=args)
    node = TrainingMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Leaving the monitor must never look like a training failure.
        sys.stdout.write("\n")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
