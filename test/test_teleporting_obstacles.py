"""The moving obstacles are driven by teleporting, which launches the robot.

`Obstacle1Plugin` and `Obstacle2Plugin` move their cylinder with
`SetWorldPoseCmd`, warping it to a new pose every update rather than giving it
a velocity, and they compute that pose from `std::chrono::steady_clock` - wall
time, not simulated time.  When the simulation runs slower than real time the
cylinder jumps a long way between physics steps, and a jump that lands inside
the robot is a deep interpenetration the solver resolves explosively.

Measured with the simulator's own pose: the robot reached 8.1 m of altitude in
one sample and 16.9 m the next, 14 m outside a 4.85 m arena.  Bounding
`contact_max_correcting_vel` did not help, because a teleported body produces
constraint impulses rather than error correction.

Removing the plugins from the derived world leaves the cylinders in place as
obstacles - the shadows they cast are the point of the exploration task - but
stops them warping into the robot.  The upstream world is untouched.
"""

from pathlib import Path

import pytest


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
    server = next(action for action in description.entities
                  if isinstance(action, IncludeLaunchDescription)
                  and 'gz_args' in dict(action.launch_arguments))
    return ET.parse(Path(dict(server.launch_arguments)['gz_args'][-1])).getroot().find('world')


def test_no_obstacle_is_driven_by_teleporting(monkeypatch):
    """A warped body lands inside the robot and the solver ejects it."""
    pytest.importorskip('launch_ros')
    world = _derived_world(monkeypatch)
    plugins = [plugin.get('name') or ''
               for include in world.findall('include')
               for plugin in include.findall('plugin')]
    assert not [name for name in plugins if 'Obstacle' in name]


def test_the_obstacles_themselves_are_still_there(monkeypatch):
    """Their shadows are what makes the exploration task non-trivial."""
    pytest.importorskip('launch_ros')
    world = _derived_world(monkeypatch)
    uris = [include.findtext('uri') or '' for include in world.findall('include')]
    assert any('obstacle1' in uri for uri in uris)
    assert any('obstacle2' in uri for uri in uris)


def test_the_inner_walls_are_untouched(monkeypatch):
    """Only the teleporting plugins go; the scenario keeps its geometry."""
    pytest.importorskip('launch_ros')
    world = _derived_world(monkeypatch)
    uris = [include.findtext('uri') or '' for include in world.findall('include')]
    assert any('inner_walls' in uri for uri in uris)
