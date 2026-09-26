"""RViz status markers make exploration progress readable at a glance."""

import pytest

rclpy = pytest.importorskip('rclpy')
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker

from turtleboot3_autonomous_nav.exploration_visualizer import ExplorationVisualizer


@pytest.fixture
def visualizer():
    rclpy.init()
    node = ExplorationVisualizer()
    yield node
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_visualizer_draws_monitored_floor_and_robot_trajectory(visualizer):
    """The green overlay excludes unknown space and walls, like the reference."""
    grid = OccupancyGrid()
    grid.header.frame_id = 'odom'
    grid.info.resolution = 0.5
    grid.info.width = 3
    grid.info.height = 2
    grid.info.origin.orientation.w = 1.0
    grid.data = [-1, 0, 100, 0, -1, 0]
    visualizer._on_monitoring_map(grid)

    first = Odometry()
    first.pose.pose.position.x = 0.25
    first.pose.pose.position.y = 0.25
    second = Odometry()
    second.pose.pose.position.x = 0.75
    second.pose.pose.position.y = 0.25
    visualizer._on_odometry(first)
    visualizer._on_odometry(second)
    visualizer._on_coverage(Float32(data=0.342))

    published = []
    visualizer._publisher.publish = published.append
    visualizer._publish()
    markers = {marker.ns: marker for marker in published[-1].markers}

    area = markers['monitored_area']
    assert area.type == Marker.CUBE_LIST
    assert [(point.x, point.y) for point in area.points] == [
        (0.75, 0.25), (0.25, 0.75), (1.25, 0.75)
    ]
    assert area.color.g > 0.5 and area.color.b > 0.5 and area.color.a < 1.0

    trail = markers['robot_trajectory']
    assert trail.type == Marker.LINE_STRIP
    assert [(point.x, point.y) for point in trail.points] == [(0.25, 0.25), (0.75, 0.25)]
    assert trail.color.r > 0.9 and trail.color.g > 0.7 and trail.color.b < 0.2

    assert markers['coverage_metrics'].text.startswith('Cobertura: 34.2%')
