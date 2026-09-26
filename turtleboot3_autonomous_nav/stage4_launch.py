"""Include the official stage4 description with optional Gazebo GUI."""

import shlex
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from launch import LaunchContext, LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.utilities import normalize_to_list_of_substitutions
from launch_ros.actions import Node


class Stage4LaunchSource(PythonLaunchDescriptionSource):
    """Gate the GUI-only include in the upstream Jazzy stage4 launcher.

    Upstream has no GUI launch argument. Preserve its world, spawn and bridge
    actions; only wrap the include whose literal gz_args selects GUI-only mode.
    Fail explicitly if upstream changes that contract.
    """

    def _get_launch_description(self, location):
        description = super()._get_launch_description(location)
        actions = []
        gui_count = 0
        for action in description.entities:
            arguments = dict(action.launch_arguments) if isinstance(
                action, IncludeLaunchDescription) else {}
            gz_args = arguments.get('gz_args')
            if isinstance(gz_args, str) and '-g' in shlex.split(gz_args):
                action = GroupAction(
                    actions=[action], condition=IfCondition(LaunchConfiguration('use_gui')))
                gui_count += 1
            actions.append(action)
        if gui_count != 1:
            raise RuntimeError('Official stage4 GUI layout changed; expected one GUI-only include')
        return LaunchDescription(actions)


class InitialRobotBridgeSource(PythonLaunchDescriptionSource):
    """Keep upstream robot bridges when Burger already exists in the world.

    The bridge also carries Gazebo's wheel-derived ``odom -> base_footprint``
    transform, which drifted 1.3 m in a mission and placed the robot and its
    laser away from the map every display draws.  Divert it so the transform
    published from the pose the map itself uses is the only one in the tree.
    """

    def _get_launch_description(self, location):
        description = super()._get_launch_description(location)
        spawners = [action for action in description.entities
                    if isinstance(action, Node) and action.node_executable == 'create']
        if len(spawners) != 1:
            raise RuntimeError('Official Burger spawn layout changed; expected one create process')
        actions = [action for action in description.entities if action is not spawners[0]]
        bridges = [action for action in actions
                   if isinstance(action, Node) and action.node_executable == 'parameter_bridge']
        if not bridges:
            raise RuntimeError('Official Burger spawn layout changed; expected a bridge process')
        diverted = (normalize_to_list_of_substitutions('/tf'),
                    normalize_to_list_of_substitutions('/tf_wheel'))
        for bridge in bridges:
            bridge._Node__remappings = list(bridge._Node__remappings or []) + [diverted]
        return LaunchDescription(actions)


TRUE_ODOMETRY_TOPIC = 'odom_truth'
"""Per-model topic carrying a pose that is not integrated from wheel counts.

Wheel odometry drifted 1.3 m and 163.8 degrees in this arena because the robot
slips on every contact and nothing closes the loop, and the map is drawn with
that pose.  A topic of its own is what makes this usable: bridging the
simulator's list of every model's pose loses the names, so nothing downstream
can tell the robot from an obstacle.
"""

_ROBOT_INCLUDE = (
    '<include><uri>model://turtlebot3_burger</uri><name>burger</name>'
    '<pose>0 0 0.01 0 0 0</pose>'
    '<plugin filename="gz-sim-odometry-publisher-system"'
    ' name="gz::sim::systems::OdometryPublisher">'
    '<odom_frame>odom</odom_frame>'
    '<robot_base_frame>base_footprint</robot_base_frame>'
    f'<odom_topic>{TRUE_ODOMETRY_TOPIC}</odom_topic>'
    '<dimensions>3</dimensions>'
    '</plugin></include>'
)


# The world's `<physics type="ode">` block, and the contact tuning inside it,
# are Gazebo Classic configuration that gz-sim ignores.  The server log settles
# it: `Loaded [gz::physics::dartsim::Plugin]`.  Bounding
# `contact_max_correcting_vel` there - tried at 100 and at 1 m/s - changed
# nothing, because nothing reads it.  The tipping was in our own control law:
# see `test_contact_escape.py`.


TELEPORTING_OBSTACLE_PLUGINS = ('Obstacle1Plugin', 'Obstacle2Plugin')
"""The two plugins that move a cylinder by warping it rather than driving it.

Both call ``SetWorldPoseCmd`` every update with a pose computed from
``std::chrono::steady_clock`` - wall time, not simulated time.  Whenever the
simulation runs slower than real time, which it does here, the cylinder covers
the elapsed wall seconds in a single physics step: it does not sweep through
the space between, it appears at the far end.  Landing inside the robot is a
deep interpenetration, and the solver resolves it with an impulse.

That is what the pose samples show.  The robot reached 8.1 m of altitude and
16.9 m the next sample, 14 m outside a 4.85 m arena, 0.5 m from anything it
could have driven into.  Bounding the contact correction from 2000 to 100 and
then to 1 m/s changed none of it, because a warped body generates constraint
impulses, not the error correction that value bounds.

Dropping the plugins leaves both cylinders standing where they spawn.  The
arena keeps its geometry and its occlusions - what the exploration has to work
around - and loses only the motion.  Static obstacles make the scenario
slightly easier than the official stage4, which is documented as a deviation.
"""


TARGET_REAL_TIME_FACTOR = 3.0
"""How much faster than the wall clock the simulation is allowed to run.

Measured during headless training: the achieved factor sat at 0.97 to 0.99 with
total CPU use around 190 % of the 1200 % available - the server at 65 %, the
mapper at 43 %, the controller at 34 %, the builder at 29 %, the bridge at 18 %.
Nothing was saturated.  The simulation was not short of machine; the world
pinned it to the wall clock by asking for a factor of one.

This is a request, not a guarantee: the server is the limit and what it
actually reaches has to be measured.  Asking for more than seems reachable
costs nothing, because the engine simply runs as fast as it can.

It is safe for the experiment because every budget here is denominated in
simulated time - episode length, stall, the penalties charged per simulated
second, the decision throttle - so the same behaviour costs the same whatever
the pace.  The physics step is untouched, so the solver sees what it saw.
"""


def _paced_faster_than_real_time(content: str) -> str:
    """Return the world with its real-time cap raised, and nothing else."""
    world = ET.fromstring(content)
    physics = world.find('world/physics')
    if physics is None:
        raise RuntimeError('Official stage4 physics layout changed; expected a physics block')
    factor = physics.find('real_time_factor')
    if factor is None:
        raise RuntimeError('Official stage4 physics no longer declares a real-time factor')
    factor.text = f'{TARGET_REAL_TIME_FACTOR:.1f}'
    return ET.tostring(world, encoding='unicode')


def _without_teleporting_obstacles(content: str) -> str:
    """Return the world with the warping obstacle plugins removed."""
    world = ET.fromstring(content)
    removed = 0
    for include in world.iter('include'):
        for plugin in list(include.findall('plugin')):
            if any(name in (plugin.get('name') or '') for name in TELEPORTING_OBSTACLE_PLUGINS):
                include.remove(plugin)
                removed += 1
    if removed != len(TELEPORTING_OBSTACLE_PLUGINS):
        raise RuntimeError(
            'Official stage4 obstacle layout changed; expected '
            f'{len(TELEPORTING_OBSTACLE_PLUGINS)} moving obstacle plugins, found {removed}')
    return ET.tostring(world, encoding='unicode')


class TrainingStage4LaunchSource(Stage4LaunchSource):
    """Make the official Burger part of Gazebo's initial reset snapshot.

    Gazebo reset.all deletes entities spawned through /create after startup.
    Derive a temporary world from the untouched upstream stage4 world with its
    standard Burger include, and retain the upstream state publisher/bridges.
    """

    def _get_launch_description(self, location):
        description = super()._get_launch_description(location)
        share = Path(location).parent.parent
        official_world = share / 'worlds' / 'turtlebot3_dqn_stage4.world'
        content = official_world.read_text(encoding='utf-8')
        world = ET.fromstring(content).find('world')
        if world is None or world.get('name') != 'dqn' or content.count('</world>') != 1:
            raise RuntimeError('Official stage4 world layout changed; expected one dqn world')
        temporary = tempfile.TemporaryDirectory(prefix='turtlebot3-training-')
        derived_world = Path(temporary.name) / official_world.name
        derived_world.write_text(
            _paced_faster_than_real_time(
                _without_teleporting_obstacles(
                    content.replace('</world>', f'{_ROBOT_INCLUDE}</world>'))),
            encoding='utf-8')
        actions = []
        server_count = spawn_count = 0
        for action in description.entities:
            if isinstance(action, IncludeLaunchDescription):
                arguments = dict(action.launch_arguments)
                gz_args = arguments.get('gz_args')
                if isinstance(gz_args, list) and len(gz_args) == 2 and gz_args[-1] == str(official_world):
                    arguments['gz_args'] = [gz_args[0], str(derived_world)]
                    action = IncludeLaunchDescription(action.launch_description_source,
                                                      launch_arguments=arguments.items())
                    server_count += 1
                elif set(arguments) == {'x_pose', 'y_pose'}:
                    action.launch_description_source.get_launch_description(LaunchContext())
                    if Path(action.launch_description_source.location).name != 'spawn_turtlebot3.launch.py':
                        raise RuntimeError('Official stage4 robot spawn include changed')
                    action = IncludeLaunchDescription(InitialRobotBridgeSource(
                        action.launch_description_source.location),
                        launch_arguments=arguments.items())
                    spawn_count += 1
            actions.append(action)
        if server_count != 1 or spawn_count != 1:
            temporary.cleanup()
            raise RuntimeError('Official stage4 launch layout changed; expected one server and spawn include')

        def cleanup(context):
            temporary.cleanup()

        actions.append(RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)])))
        return LaunchDescription(actions)
