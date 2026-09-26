"""Drive the observation adapter with real ROS messages.

The pure builder is covered elsewhere; what matters here is that the node
feeds it the whole map and the robot's cell, so a published observation
matches the width the policy and trainer expect.
"""

from types import SimpleNamespace

import pytest

rclpy = pytest.importorskip('rclpy')
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan

from turtleboot3_autonomous_nav.observation import OBSERVATION_SIZE, observation_label
from turtleboot3_autonomous_nav.observation_builder import ObservationBuilder


def delivery(stamp_ns):
    return SimpleNamespace(
        source_timestamp=stamp_ns,
        get_rmw_message_info=lambda: {'source_timestamp': stamp_ns},
    )


@pytest.fixture
def node():
    rclpy.init()
    builder = ObservationBuilder()
    yield builder
    builder.destroy_node()
    rclpy.shutdown()


def _arena_map(width=100, height=100):
    message = OccupancyGrid()
    message.info.width = width
    message.info.height = height
    message.info.resolution = 0.05
    message.info.origin.position.x = -2.5
    message.info.origin.position.y = -2.5
    message.data = [-1] * (width * height)
    return message


def _odometry(x=0.0, y=0.0):
    message = Odometry()
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    message.pose.pose.orientation.w = 1.0
    return message


def _scan():
    message = LaserScan()
    message.ranges = [1.0] * 360
    return message


def test_the_published_observation_matches_the_expected_width(node):
    """A width mismatch would make the trainer reject every observation."""
    published = []
    node._publisher.publish = published.append

    request = SimpleNamespace()
    node._on_reset(request, SimpleNamespace(success=False, message=''))
    node._on_map(_arena_map(), delivery(node._reset_cutoff_ns + 1))
    node._on_odometry(_odometry(), delivery(node._reset_cutoff_ns + 2))
    node._on_scan(_scan(), delivery(node._reset_cutoff_ns + 3))

    assert len(published) == 1
    assert len(published[0].data) == OBSERVATION_SIZE
    assert published[0].layout.dim[0].label == observation_label(node._reset_epoch)


def test_a_pose_outside_the_map_still_produces_an_observation(node):
    """Odometry drift past the arena must degrade the pose, not stop control."""
    published = []
    node._publisher.publish = published.append

    node._on_reset(SimpleNamespace(), SimpleNamespace(success=False, message=''))
    node._on_map(_arena_map(), delivery(node._reset_cutoff_ns + 1))
    node._on_odometry(_odometry(x=99.0, y=99.0), delivery(node._reset_cutoff_ns + 2))
    node._on_scan(_scan(), delivery(node._reset_cutoff_ns + 3))

    assert len(published) == 1
    assert len(published[0].data) == OBSERVATION_SIZE


def test_scans_arriving_faster_than_the_decision_rate_publish_once(node):
    """The throttle only works if the node records when it last published."""
    published = []
    node._publisher.publish = published.append
    node._min_publish_period_ns = 10_000_000_000

    node._on_reset(SimpleNamespace(), SimpleNamespace(success=False, message=''))
    node._on_map(_arena_map(), delivery(node._reset_cutoff_ns + 1))
    node._on_odometry(_odometry(), delivery(node._reset_cutoff_ns + 2))
    for offset in range(3, 13):
        node._on_scan(_scan(), delivery(node._reset_cutoff_ns + offset))

    assert len(published) == 1


def test_no_observation_is_published_before_the_map_arrives(node):
    """Publishing without a map would feed the policy an empty scenario."""
    published = []
    node._publisher.publish = published.append

    node._on_reset(SimpleNamespace(), SimpleNamespace(success=False, message=''))
    node._on_odometry(_odometry(), delivery(node._reset_cutoff_ns + 1))
    node._on_scan(_scan(), delivery(node._reset_cutoff_ns + 2))

    assert published == []
