"""Let the simulation run ahead of the wall clock when the machine can.

Measured during headless training: the real-time factor sits at 0.97 to 0.99
and total CPU use is about 190 % of the 1200 % available - the Gazebo server at
65 %, the mapper at 43 %, the controller at 34 %, the observation builder at
29 % and the bridge at 18 %.  Nothing is saturated.  The simulation is not
short of machine, it is pinned to the wall clock by the world declaring a
real-time factor of one.

Raising it is safe for the experiment because every budget in this package is
denominated in simulated time: episode length, stall, the penalties charged per
simulated second, and the decision throttle.  The physics step stays at
0.001 s, so the solver sees exactly what it saw before; only the pace at which
wall time is consumed changes.

What the machine actually achieves is a separate question from what the world
asks for, and only measurement answers it.  These tests pin the request and the
untouched step size.
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


def test_the_derived_world_asks_to_run_faster_than_real_time(monkeypatch):
    """At a factor of one the campaign costs one wall second per simulated one."""
    pytest.importorskip('launch_ros')
    world = _derived_world(monkeypatch)
    assert float(world.findtext('physics/real_time_factor')) > 1.0


def test_the_physics_step_is_not_touched(monkeypatch):
    """Pace is not fidelity: a larger step would change what the solver sees."""
    pytest.importorskip('launch_ros')
    world = _derived_world(monkeypatch)
    assert world.findtext('physics/max_step_size').strip() == '0.001'


def test_the_solver_is_not_touched_either(monkeypatch):
    pytest.importorskip('launch_ros')
    world = _derived_world(monkeypatch)
    assert world.findtext('physics/ode/solver/iters').strip() == '150'
