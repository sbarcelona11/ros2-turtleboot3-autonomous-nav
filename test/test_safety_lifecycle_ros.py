"""ROS adapter safety regressions with controlled clocks and message delivery."""

import json
import time
from types import SimpleNamespace

import pytest

rclpy = pytest.importorskip('rclpy')
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Int32
from std_srvs.srv import SetBool, Trigger

from turtleboot3_autonomous_nav.safe_motion_controller import SafeMotionController
from turtleboot3_autonomous_nav.observation import observation_label
from turtleboot3_autonomous_nav.observation_builder import ObservationBuilder


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def info(stamp):
    return {'source_timestamp': stamp}


def scan():
    return LaserScan(ranges=[2.0] * 360, angle_increment=0.0174533, range_max=3.5)


@pytest.fixture
def controller(ros_context, monkeypatch):
    node = SafeMotionController()
    clock = [10_000_000_000]
    decisions = []
    monkeypatch.setattr(node, '_now_ns', lambda: clock[0])
    monkeypatch.setattr(node, '_publish_decision', decisions.append)
    yield node, clock, decisions
    node.destroy_node()


def enable(node, value=True):
    ack = node._on_enable(SetBool.Request(data=value), SetBool.Response())
    assert ack.success
    return json.loads(ack.message)['cutoff_ns']


def feed_controller(node, stamp, action=0):
    node._on_odometry(Odometry(), info(stamp))
    node._on_scan(scan(), info(stamp))
    node._on_action(Int32(data=action), info(stamp))


def test_controller_refuses_missing_odometry(controller):
    node, clock, decisions = controller
    cutoff = enable(node)
    node._on_scan(scan(), info(cutoff + 1))
    node._on_action(Int32(data=0), info(cutoff + 1))
    assert node._data_is_stale(clock[0])


def test_controller_rejects_negative_simulated_ages(controller):
    node, clock, decisions = controller
    cutoff = enable(node)
    feed_controller(node, cutoff + 1)
    assert node._data_is_stale(clock[0] - 1)


def test_controller_wall_watchdog_stops_when_simulation_clock_is_frozen(controller, monkeypatch):
    node, clock, decisions = controller
    wall_time = [time.monotonic_ns()]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: wall_time[0])
    cutoff = enable(node)
    feed_controller(node, cutoff + 1)
    node._on_control_timer()
    assert decisions[-1].linear_x > 0
    wall_time[0] += 600_000_000
    node._on_control_timer()
    assert decisions[-1].linear_x == decisions[-1].angular_z == 0


def test_controller_disable_and_enable_require_new_publications(controller):
    node, clock, decisions = controller
    cutoff = enable(node)
    feed_controller(node, cutoff + 1)
    node._on_control_timer()
    assert decisions[-1].linear_x > 0

    enable(node, False)
    assert decisions[-1].linear_x == decisions[-1].angular_z == 0
    feed_controller(node, time.time_ns() + 1)
    node._on_control_timer()
    assert decisions[-1].linear_x == decisions[-1].angular_z == 0

    cutoff = enable(node)
    # A scan/action emitted before reset but delivered later stays invalid.
    feed_controller(node, cutoff)
    node._on_control_timer()
    assert decisions[-1].linear_x == decisions[-1].angular_z == 0
    feed_controller(node, cutoff + 1)
    node._on_control_timer()
    assert decisions[-1].linear_x > 0


def test_controller_stops_when_odometry_expires(controller):
    node, clock, decisions = controller
    cutoff = enable(node)
    feed_controller(node, cutoff + 1)
    clock[0] += 600_000_000
    node._on_scan(scan(), info(cutoff + 2))
    node._on_action(Int32(data=0), info(cutoff + 2))
    node._on_control_timer()
    assert decisions[-1].linear_x == decisions[-1].angular_z == 0
    assert decisions[-1].intervention


def test_stall_recovery_finishes_and_allows_forward_progress(controller):
    node, clock, decisions = controller
    cutoff = enable(node)
    feed_controller(node, cutoff + 1)
    clock[0] += 4_000_000_000
    feed_controller(node, cutoff + 2)
    node._on_control_timer()
    assert decisions[-1].recovery
    clock[0] += 3_000_000_000
    feed_controller(node, cutoff + 3)
    node._on_control_timer()
    assert not decisions[-1].recovery
    assert decisions[-1].linear_x > 0


def test_repeated_policy_recovery_is_bounded(controller):
    node, clock, decisions = controller
    cutoff = enable(node)
    feed_controller(node, cutoff + 1, action=5)
    node._on_control_timer()
    assert decisions[-1].recovery
    clock[0] += 3_000_000_000
    feed_controller(node, cutoff + 2, action=5)
    node._on_control_timer()
    assert decisions[-1].linear_x == decisions[-1].angular_z == 0
    assert not decisions[-1].recovery


def test_builder_reset_rejects_cached_and_queued_previous_episode_inputs(ros_context):
    node = ObservationBuilder()
    publications = []
    node._publisher = SimpleNamespace(publish=publications.append)
    grid = OccupancyGrid()
    grid.info.width = grid.info.height = 8
    grid.info.resolution = 1.0
    grid.data = [0] * 64
    try:
        before = time.time_ns()
        node._on_map(grid, info(before))
        node._on_odometry(Odometry(), info(before))
        node._on_scan(scan(), info(before))
        assert len(publications) == 1
        ack = node._on_reset(Trigger.Request(), Trigger.Response())
        cutoff = json.loads(ack.message)['cutoff_ns']
        node._on_map(grid, info(before))
        node._on_odometry(Odometry(), info(before))
        node._on_scan(scan(), info(cutoff + 1))
        assert len(publications) == 1
        node._on_map(grid, info(cutoff + 2))
        node._on_scan(scan(), info(cutoff + 3))
        assert len(publications) == 1
        node._on_odometry(Odometry(), info(cutoff + 4))
        node._on_scan(scan(), info(cutoff + 5))
        assert len(publications) == 2
        # The reset epoch is carried by the observation consumed by the trainer.
        assert publications[-1].layout.dim[0].label == observation_label(1)
    finally:
        node.destroy_node()
