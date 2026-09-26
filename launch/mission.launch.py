"""Autonomous exploration in the official TurtleBot3 stage4 world."""

from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import EqualsSubstitution, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackagePrefix, FindPackageShare

from turtleboot3_autonomous_nav.stage4_launch import (
    TRUE_ODOMETRY_TOPIC,
    TrainingStage4LaunchSource,
)


def generate_launch_description():
    package = 'turtleboot3_autonomous_nav'
    share = FindPackageShare(package)
    sim = {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}
    return LaunchDescription([
        # Empty by default so the frontier explorer needs no checkpoint; the
        # DQN explorer still refuses to run without one.
        DeclareLaunchArgument('model_path', default_value='',
                              description='Trusted trained checkpoint path (dqn explorer)'),
        DeclareLaunchArgument('explorer', default_value='dqn',
                              description='Decision source: dqn or frontier'),
        # The assignment asks for the whole scenario, so the run ends when the
        # arena is covered, not at an arbitrary fraction of it.  The step cap
        # is the safety net for a policy that gets stuck, and has to sit well
        # clear of the target: reaching three quarters already took 304 steps.
        DeclareLaunchArgument('max_steps', default_value='3000'),
        DeclareLaunchArgument('target_coverage', default_value='0.98'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_gui', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        AppendEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', PathJoinSubstitution([
            FindPackagePrefix('turtlebot3_gazebo'), 'lib', 'turtlebot3_gazebo'])),
        GroupAction(actions=[
            SetRemap(src='/cmd_vel', dst='/official_cmd_vel_stamped'),
            IncludeLaunchDescription(TrainingStage4LaunchSource(PathJoinSubstitution([
                FindPackageShare('turtlebot3_gazebo'), 'launch', 'turtlebot3_dqn_stage4.launch.py'])),
                launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()),
        ]),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='exploration_velocity_bridge',
             arguments=['/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist'], parameters=[sim]),
        # Wheel odometry drifts past a metre in this arena and the map is drawn
        # with that pose; this one comes from the model's world state.
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='true_odometry_bridge',
             arguments=[f'/{TRUE_ODOMETRY_TOPIC}@nav_msgs/msg/Odometry[gz.msgs.Odometry'],
             parameters=[sim]),
        # Publishes odom -> base_footprint from the same pose the map uses, so
        # the displays draw the robot and its laser on top of the map.
        Node(package=package, executable='simulation_localizer', name='simulation_localizer',
             parameters=[sim]),
        Node(package=package, executable='coverage_mapper', name='coverage_mapper',
             remappings=[('/odom', f'/{TRUE_ODOMETRY_TOPIC}')], parameters=[PathJoinSubstitution([share, 'config', 'exploration.yaml']), sim]),
        Node(package=package, executable='observation_builder', name='observation_builder',
             remappings=[('/odom', f'/{TRUE_ODOMETRY_TOPIC}')], parameters=[sim]),
        Node(package=package, executable='safe_motion_controller', name='safe_motion_controller',
             remappings=[('/odom', f'/{TRUE_ODOMETRY_TOPIC}')], parameters=[sim]),
        Node(package=package, executable='dqn_explorer', name='dqn_explorer',
             remappings=[('/odom', f'/{TRUE_ODOMETRY_TOPIC}')], parameters=[sim, {
            'model_path': ParameterValue(LaunchConfiguration('model_path'), value_type=str),
            'max_steps': ParameterValue(LaunchConfiguration('max_steps'), value_type=int),
            'target_coverage': ParameterValue(LaunchConfiguration('target_coverage'), value_type=float)}],
             condition=IfCondition(EqualsSubstitution(LaunchConfiguration('explorer'), 'dqn'))),
        Node(package=package, executable='frontier_explorer', name='frontier_explorer',
             remappings=[('/odom', f'/{TRUE_ODOMETRY_TOPIC}')], parameters=[sim, {
            'max_steps': ParameterValue(LaunchConfiguration('max_steps'), value_type=int),
            'target_coverage': ParameterValue(LaunchConfiguration('target_coverage'), value_type=float)}],
             condition=IfCondition(EqualsSubstitution(LaunchConfiguration('explorer'), 'frontier'))),
        Node(package=package, executable='exploration_visualizer', name='exploration_visualizer',
             parameters=[sim], condition=IfCondition(LaunchConfiguration('use_rviz'))),
        Node(package='rviz2', executable='rviz2', name='rviz2', parameters=[sim],
             arguments=['-d', PathJoinSubstitution([share, 'rviz', 'exploration.rviz'])],
             condition=IfCondition(LaunchConfiguration('use_rviz'))),
    ])
