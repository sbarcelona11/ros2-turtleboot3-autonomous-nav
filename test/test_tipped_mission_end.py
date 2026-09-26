"""A tipped robot ends the mission instead of burning its step budget.

The controller already stops the motors when the robot leaves an upright
attitude - measured once at roll 94.7 and pitch 90.0 degrees, lying on its
side.  But the explorer kept issuing actions to it for minutes, so the run
looked alive while nothing could happen.  Reporting the state ends the
mission with a reason that says what went wrong.
"""

import math

import pytest

rclpy = pytest.importorskip('rclpy')
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
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


def _imu(roll=0.0):
    message = Imu()
    message.orientation.w = math.cos(roll / 2.0)
    message.orientation.x = math.sin(roll / 2.0)
    return message


def test_the_controller_reports_an_upright_robot(controller):
    published = []
    controller._tipped_publisher.publish = published.append

    controller._on_imu(_imu(), delivery(controller._reset_cutoff_ns + 1))

    assert published and published[-1].data is False


def test_the_controller_reports_a_tipped_robot(controller):
    """Nothing downstream can see the attitude; the controller has to say so."""
    published = []
    controller._tipped_publisher.publish = published.append

    controller._on_imu(
        _imu(roll=math.radians(95.0)), delivery(controller._reset_cutoff_ns + 1)
    )

    assert published and published[-1].data is True


def test_the_mission_ends_when_the_robot_tips(monkeypatch):
    """Otherwise the run spends its whole step budget on a fallen robot."""
    from turtleboot3_autonomous_nav import frontier_explorer

    ended = []

    def exercise(node):
        node._finish = lambda reason: ended.append(reason)
        node._on_tipped(Bool(data=True))

    monkeypatch.setattr(rclpy, 'spin', exercise)
    frontier_explorer.main([])

    assert ended == ['robot_tipped']


def test_an_upright_report_does_not_end_the_mission(monkeypatch):
    from turtleboot3_autonomous_nav import frontier_explorer

    ended = []

    def exercise(node):
        node._finish = lambda reason: ended.append(reason)
        node._on_tipped(Bool(data=False))

    monkeypatch.setattr(rclpy, 'spin', exercise)
    frontier_explorer.main([])

    assert ended == []
