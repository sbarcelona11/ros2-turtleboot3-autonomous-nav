"""The controller must actually stop when the robot is no longer upright.

The pure predicate is covered by `test_contact_safety`; what matters here is
that the node reads the IMU and acts on it.  A detector that is never
consulted protects nothing - twice in this project a rate limit and a
throttle passed their unit tests while the wiring was broken.
"""

import math

import pytest

rclpy = pytest.importorskip('rclpy')
from sensor_msgs.msg import Imu
from types import SimpleNamespace

from turtleboot3_autonomous_nav.safe_motion_controller import SafeMotionController


def delivery(stamp_ns):
    return SimpleNamespace(
        source_timestamp=stamp_ns,
        get_rmw_message_info=lambda: {'source_timestamp': stamp_ns},
    )


@pytest.fixture
def controller():
    rclpy.init()
    node = SafeMotionController()
    node._enabled = True
    yield node
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _imu(roll=0.0, pitch=0.0):
    message = Imu()
    half_roll, half_pitch = roll / 2.0, pitch / 2.0
    message.orientation.w = math.cos(half_roll) * math.cos(half_pitch)
    message.orientation.x = math.sin(half_roll) * math.cos(half_pitch)
    message.orientation.y = math.cos(half_roll) * math.sin(half_pitch)
    message.orientation.z = -math.sin(half_roll) * math.sin(half_pitch)
    return message


def test_an_upright_reading_leaves_the_controller_running(controller):
    controller._on_imu(_imu(), delivery(controller._reset_cutoff_ns + 1))
    assert not controller._tipped


def test_a_tipped_reading_stops_the_controller(controller):
    """Roll of 95 degrees is what was measured on the fallen robot."""
    controller._on_imu(
        _imu(roll=math.radians(95.0)), delivery(controller._reset_cutoff_ns + 1)
    )
    assert controller._tipped


def test_a_tipped_controller_publishes_no_motion(controller):
    published = []
    controller._cmd_publisher.publish = published.append

    controller._on_imu(
        _imu(roll=math.radians(95.0)), delivery(controller._reset_cutoff_ns + 1)
    )
    controller._on_control_timer()

    assert published
    assert published[-1].linear.x == 0.0
    assert published[-1].angular.z == 0.0


def test_enabling_a_new_run_clears_the_tipped_state(controller):
    """A reset places the robot upright again; the flag must not persist."""
    from std_srvs.srv import SetBool

    controller._tipped = True
    controller._on_enable(SetBool.Request(data=True), SetBool.Response())
    assert not controller._tipped
