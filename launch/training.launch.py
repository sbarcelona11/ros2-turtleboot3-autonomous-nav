"""Train a DQN with resettable episodes in the official stage4 world."""

from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackagePrefix, FindPackageShare

from turtleboot3_autonomous_nav.stage4_launch import TrainingStage4LaunchSource


def generate_launch_description():
    package = 'turtleboot3_autonomous_nav'
    share = FindPackageShare(package)
    sim = {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}
    return LaunchDescription([
        DeclareLaunchArgument('episodes', default_value='100'),
        DeclareLaunchArgument('use_gui', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('training_config', default_value=PathJoinSubstitution([share, 'config', 'training.yaml'])),
        DeclareLaunchArgument('model_directory', default_value='models'),
        DeclareLaunchArgument('resume', default_value='true'),
        DeclareLaunchArgument('monitor', default_value='true'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        AppendEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', PathJoinSubstitution([
            FindPackagePrefix('turtlebot3_gazebo'), 'lib', 'turtlebot3_gazebo'])),
        GroupAction(actions=[
            SetRemap(src='/cmd_vel', dst='/official_cmd_vel_stamped'),
            IncludeLaunchDescription(TrainingStage4LaunchSource(PathJoinSubstitution([
                FindPackageShare('turtlebot3_gazebo'), 'launch', 'turtlebot3_dqn_stage4.launch.py'])),
                launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()),
        ]),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='exploration_bridge',
             arguments=['/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                        '/world/dqn/control@ros_gz_interfaces/srv/ControlWorld'], parameters=[sim]),
        Node(package=package, executable='coverage_mapper', name='coverage_mapper',
             parameters=[PathJoinSubstitution([share, 'config', 'exploration.yaml']), sim]),
        Node(package=package, executable='observation_builder', name='observation_builder', parameters=[sim]),
        Node(package=package, executable='safe_motion_controller', name='safe_motion_controller', parameters=[sim]),
        Node(package=package, executable='training_monitor', name='training_monitor',
             output='screen', parameters=[sim],
             condition=IfCondition(LaunchConfiguration('monitor'))),
        Node(package=package, executable='dqn_trainer', name='dqn_trainer', parameters=[
            LaunchConfiguration('training_config'), sim, {
                'max_episodes': ParameterValue(LaunchConfiguration('episodes'), value_type=int),
                'model_directory': ParameterValue(LaunchConfiguration('model_directory'), value_type=str),
                'resume': ParameterValue(LaunchConfiguration('resume'), value_type=bool),
            }]),
    ])
