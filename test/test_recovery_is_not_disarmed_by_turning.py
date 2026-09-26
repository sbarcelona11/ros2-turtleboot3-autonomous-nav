"""A policy that turns in place must not switch off its own rescue.

The controller resets its progress clock on any action that does not command
translation.  The reason is sound: recovery itself only turns, so counting a
recovery turn as lack of progress would let recovery renew the very condition
that triggered it.

The side effect was not intended.  A policy is free to choose LEFT or RIGHT,
which also do not translate, so a policy that keeps choosing them keeps the
clock pinned at zero forever and the automatic recovery never arms.  The
trained agent found exactly that: 63 % of its decisions were turns in place,
and a mission sat motionless against a wall for the whole of its 3000 steps
without recovery ever firing.

The clock has to be held only while recovery is actually running, which keeps
the original protection and closes the loophole.
"""

import pytest

rclpy = pytest.importorskip('rclpy')

from turtleboot3_autonomous_nav.control import (  # noqa: E402
    FORWARD,
    LEFT,
    RECOVER,
)
from turtleboot3_autonomous_nav.safe_motion_controller import (  # noqa: E402
    SafeMotionController,
)


@pytest.fixture
def controller():
    """Match the lifecycle the other ROS tests use, so the context stays clean."""
    rclpy.init()
    node = SafeMotionController()
    node._enabled = True
    yield node
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_a_policy_turn_lets_the_stall_clock_run(controller):
    """This is the loophole: the policy could disarm recovery by spinning."""
    start = controller._last_progress_ns
    controller._action = LEFT
    controller._recovery_started_ns = None
    controller._hold_progress_clock_for_recovery(start + 1_000_000_000)
    assert controller._last_progress_ns == start, (
        'a policy turn must not reset the progress clock'
    )


def test_a_recovery_turn_still_holds_the_clock(controller):
    """Without this, recovery renews the condition that triggered it."""
    controller._action = LEFT
    controller._recovery_started_ns = 5
    now = controller._last_progress_ns + 1_000_000_000
    controller._hold_progress_clock_for_recovery(now)
    assert controller._last_progress_ns == now


def test_an_explicit_recover_action_also_holds_the_clock(controller):
    """RECOVER is the policy asking for the same turning motion."""
    controller._action = RECOVER
    controller._recovery_started_ns = None
    now = controller._last_progress_ns + 1_000_000_000
    controller._hold_progress_clock_for_recovery(now)
    assert controller._last_progress_ns == now


def test_a_translating_action_never_holds_the_clock(controller):
    """Driving forward and making no ground is exactly what stalling means."""
    start = controller._last_progress_ns
    controller._action = FORWARD
    controller._recovery_started_ns = 5
    controller._hold_progress_clock_for_recovery(start + 1_000_000_000)
    assert controller._last_progress_ns == start
