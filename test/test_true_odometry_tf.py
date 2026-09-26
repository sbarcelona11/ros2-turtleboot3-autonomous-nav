"""The transform tree must agree with the map the mission builds.

The map is now drawn from the simulator's pose and matches the arena to 8 cm,
but RViz places the robot and its laser through `odom -> base_footprint`,
which Gazebo still publishes from the wheels.  The result looks like a broken
system: a correct map with the scan drawn a metre away from it.

Suppressing the wheel transform and publishing the same one from the pose the
map already uses makes the picture consistent.  The rest of the chain,
`base_footprint -> base_link -> base_scan`, keeps coming from the robot state
publisher.
"""

from pathlib import Path

import pytest

rclpy = pytest.importorskip('rclpy')
from nav_msgs.msg import Odometry

from turtleboot3_autonomous_nav.simulation_localizer import transform_from_odometry


def _odometry(x=1.5, y=-0.25, w=1.0, z=0.0):
    message = Odometry()
    message.header.frame_id = 'odom'
    message.child_frame_id = 'base_footprint'
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    message.pose.pose.orientation.z = z
    message.pose.pose.orientation.w = w
    return message


def test_the_transform_carries_the_pose():
    transform = transform_from_odometry(_odometry())
    assert transform.transform.translation.x == pytest.approx(1.5)
    assert transform.transform.translation.y == pytest.approx(-0.25)


def test_the_transform_keeps_the_frames_of_the_odometry():
    """Renaming a frame here would detach the robot from the map."""
    transform = transform_from_odometry(_odometry())
    assert transform.header.frame_id == 'odom'
    assert transform.child_frame_id == 'base_footprint'


def test_the_rotation_is_carried_through():
    transform = transform_from_odometry(_odometry(w=0.7071, z=0.7071))
    assert transform.transform.rotation.z == pytest.approx(0.7071)
    assert transform.transform.rotation.w == pytest.approx(0.7071)


def test_the_stamp_is_taken_from_the_odometry():
    """A stale stamp makes the transform unusable for the scan's timestamp."""
    message = _odometry()
    message.header.stamp.sec = 42
    assert transform_from_odometry(message).header.stamp.sec == 42


def test_the_wheel_transform_is_diverted_away_from_the_tree(monkeypatch):
    """Two publishers of one transform would make the robot jitter."""
    pytest.importorskip('launch_ros')
    from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
    from launch import LaunchContext
    from launch.utilities import perform_substitutions
    from launch_ros.actions import Node
    from turtleboot3_autonomous_nav import stage4_launch

    try:
        share = Path(get_package_share_directory('turtlebot3_gazebo'))
    except PackageNotFoundError:
        pytest.skip('Official TurtleBot3 simulation sources are not installed')
    monkeypatch.setenv('TURTLEBOT3_MODEL', 'burger')
    context = LaunchContext()
    description = stage4_launch.InitialRobotBridgeSource(
        str(share / 'launch' / 'spawn_turtlebot3.launch.py')
    ).get_launch_description(context)
    bridges = [action for action in description.entities
               if isinstance(action, Node) and action.node_executable == 'parameter_bridge']
    assert bridges, 'the upstream spawn no longer runs a bridge'
    diverted = [
        (perform_substitutions(context, list(source)),
         perform_substitutions(context, list(target)))
        for bridge in bridges
        for source, target in (bridge._Node__remappings or ())
    ]
    assert ('/tf', '/tf_wheel') in diverted
