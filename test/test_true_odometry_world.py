"""The simulated robot must publish a pose that is not integrated from wheels.

Wheel odometry drifted 1.3 m and 163.8 degrees in this arena, measured against
Gazebo's own pose: the robot slips on every contact and nothing closes the
loop, so the map it produced bore no relation to the scenario.

Gazebo's odometry publisher derives the pose from the model's world state, so
it does not accumulate slip.  Attaching it to the robot in the derived world
gives a dedicated topic per model, which the bridge carries without needing
the model names that the pose-list conversion drops.
"""

from pathlib import Path

import pytest

TRUE_ODOMETRY_TOPIC = 'odom_truth'


def _derived_world(monkeypatch):
    import xml.etree.ElementTree as ET
    from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
    from launch import LaunchContext
    from launch.actions import IncludeLaunchDescription
    from turtleboot3_autonomous_nav import stage4_launch

    try:
        share = Path(get_package_share_directory('turtlebot3_gazebo'))
    except PackageNotFoundError:
        pytest.skip('Official TurtleBot3 simulation sources are not installed')
    monkeypatch.setenv('TURTLEBOT3_MODEL', 'burger')
    context = LaunchContext()
    context.launch_configurations['use_gui'] = 'false'
    description = stage4_launch.TrainingStage4LaunchSource(
        str(share / 'launch' / 'turtlebot3_dqn_stage4.launch.py')
    ).get_launch_description(context)
    includes = [action for action in description.entities
                if isinstance(action, IncludeLaunchDescription)]
    server = next(action for action in includes if 'gz_args' in dict(action.launch_arguments))
    generated = Path(dict(server.launch_arguments)['gz_args'][-1])
    return ET.parse(generated).getroot().find('world')


def test_the_robot_carries_an_odometry_publisher(monkeypatch):
    """Without it the only pose available is the one that drifts."""
    pytest.importorskip('launch_ros')
    world = _derived_world(monkeypatch)
    robot = world.findall('include')[-1]
    plugins = robot.findall('plugin')
    assert plugins, 'the robot in the derived world publishes no true odometry'
    assert any(
        plugin.get('name') == 'gz::sim::systems::OdometryPublisher'
        for plugin in plugins
    )


def test_the_odometry_goes_to_a_topic_of_its_own(monkeypatch):
    """A per-model topic survives the bridge; a pose list loses the names."""
    pytest.importorskip('launch_ros')
    world = _derived_world(monkeypatch)
    plugin = world.findall('include')[-1].findall('plugin')[0]
    assert plugin.findtext('odom_topic') == TRUE_ODOMETRY_TOPIC


def test_the_odometry_uses_the_frames_the_rest_of_the_stack_expects(monkeypatch):
    pytest.importorskip('launch_ros')
    world = _derived_world(monkeypatch)
    plugin = world.findall('include')[-1].findall('plugin')[0]
    assert plugin.findtext('odom_frame') == 'odom'
    assert plugin.findtext('robot_base_frame') == 'base_footprint'


def _mission_nodes(monkeypatch):
    from launch.launch_description_sources import get_launch_description_from_python_launch_file
    from launch_ros.actions import Node

    monkeypatch.setenv('TURTLEBOT3_MODEL', 'burger')
    description = get_launch_description_from_python_launch_file(
        str(Path(__file__).resolve().parents[1] / 'launch' / 'mission.launch.py'))
    return [entity for entity in description.entities if isinstance(entity, Node)]


def test_the_mission_bridges_the_true_odometry(monkeypatch):
    """The topic is useless until it reaches ROS under the expected name."""
    pytest.importorskip('launch_ros')
    bridged = [
        argument
        for node in _mission_nodes(monkeypatch)
        if node.node_executable == 'parameter_bridge'
        for argument in (node._Node__arguments or ())
        if TRUE_ODOMETRY_TOPIC in str(argument)
    ]
    assert bridged, 'the true odometry never reaches ROS'
    assert 'nav_msgs/msg/Odometry' in str(bridged[0])


def test_the_mission_feeds_the_stack_from_the_true_odometry(monkeypatch):
    """Mapping on the drifting pose is what made the map meaningless."""
    pytest.importorskip('launch_ros')
    from launch import LaunchContext
    from launch.utilities import perform_substitutions

    context = LaunchContext()
    remapped = {
        node.node_executable
        for node in _mission_nodes(monkeypatch)
        for source, target in (node._Node__remappings or ())
        if perform_substitutions(context, list(source)) == '/odom'
        and TRUE_ODOMETRY_TOPIC in perform_substitutions(context, list(target))
    }
    assert {'coverage_mapper', 'observation_builder', 'safe_motion_controller'} <= remapped
