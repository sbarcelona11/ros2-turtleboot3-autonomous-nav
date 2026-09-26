from pathlib import Path

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_coverage_aligned_training_runbook_avoids_legacy_checkpoints():
    """Fresh campaigns must use the aligned model directory and diagnostics."""
    readme = (PACKAGE_ROOT / 'README.md').read_text()
    assert 'model_directory:=/ros2_ws/models/coverage-aligned-v1' in readme
    assert 'resume:=false' in readme
    assert '/coverage_reachable' in readme
    assert '/training_episode_diagnostics' in readme


def test_readme_labels_legacy_checkpoint_and_compares_explorers():
    """Historical DQN numbers must not be mistaken for aligned-run results."""
    readme = (PACKAGE_ROOT / 'README.md').read_text()
    assert 'models/best.pt' in readme
    assert 'not `monitoring-map-v1`' in readme
    assert '| strategy | source | reachable coverage |' in readme
    assert '| DQN v7 (legacy checkpoint) | episode 156, 3 greedy evaluations | 74.60 % |' in readme
    assert '| Frontier | current-scenario mission baseline | 97.97 % |' in readme
    assert (PACKAGE_ROOT / 'models' / 'best.pt').is_file()


def test_package_declares_runtime_dependencies():
    xml = (PACKAGE_ROOT / 'package.xml').read_text()
    for name in (
        'rclpy',
        'sensor_msgs',
        'nav_msgs',
        'geometry_msgs',
        'tf2_ros',
        'ros_gz_interfaces',
    ):
        assert f'<exec_depend>{name}</exec_depend>' in xml


def test_turtlebot3_dependency_manifest_pins_all_required_jazzy_sources():
    manifest = PACKAGE_ROOT / 'dependencies' / 'turtlebot3_jazzy.repos'
    data = yaml.safe_load(manifest.read_text())
    repositories = data['repositories']
    expected = {
        'turtlebot3': 'https://github.com/ROBOTIS-GIT/turtlebot3.git',
        'turtlebot3_msgs': 'https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git',
        'turtlebot3_simulations': 'https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git',
    }
    assert set(repositories) == set(expected)
    for name, url in expected.items():
        assert repositories[name]['type'] == 'git'
        assert repositories[name]['url'] == url
        assert repositories[name]['version'] == 'jazzy'


def test_dependency_installer_is_explicit_and_never_runs_during_colcon_build():
    script = PACKAGE_ROOT / 'turtleboot3_autonomous_nav' / 'dependency_installer.py'
    source = script.read_text()
    assert 'vcs import' in source
    assert 'rosdep install' in source
    assert 'colcon build' in source
    assert 'subprocess.run' in source
    assert 'setup.py' not in source


def test_dependency_installer_excludes_optional_cartographer_packages():
    """Stage 4 only needs Gazebo and its transitive TurtleBot3 packages."""
    script = PACKAGE_ROOT / 'turtleboot3_autonomous_nav' / 'dependency_installer.py'
    source = script.read_text()
    assert "'src/turtlebot3_msgs'" in source
    assert "'src/turtlebot3/turtlebot3_description'" in source
    assert "'src/turtlebot3_simulations/turtlebot3_gazebo'" in source
    assert "'--from-paths', 'src'" not in source
    assert "'--packages-up-to', 'turtlebot3_gazebo'" in source


def test_mission_uses_stage4_without_nav_or_slam():
    source = (PACKAGE_ROOT / 'launch' / 'mission.launch.py').read_text()
    assert 'turtlebot3_dqn_stage4.launch.py' in source
    assert 'nav2' not in source.lower()
    assert 'slam' not in source.lower()


@pytest.mark.parametrize('mode,policy', [('mission', 'dqn_explorer'), ('training', 'dqn_trainer')])
def test_launch_has_one_policy_and_controller(mode, policy, monkeypatch):
    pytest.importorskip('launch_ros')
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch.launch_description_sources import get_launch_description_from_python_launch_file
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters

    monkeypatch.setenv('TURTLEBOT3_MODEL', 'burger')
    description = get_launch_description_from_python_launch_file(
        str(PACKAGE_ROOT / 'launch' / f'{mode}.launch.py'))
    context = LaunchContext()
    context.launch_configurations['model_path'] = '/tmp/test-policy.pt'
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)
    nodes = [entity for entity in description.entities if isinstance(entity, Node)]
    executables = [node.node_executable for node in nodes]
    assert executables.count('safe_motion_controller') == 1
    assert executables.count(policy) == 1
    assert ('dqn_trainer' if mode == 'mission' else 'dqn_explorer') not in executables
    assert {'coverage_mapper', 'observation_builder'} <= set(executables)
    assert context.launch_configurations['use_sim_time'] == 'true'
    if mode == 'training':
        assert context.launch_configurations['use_gui'] == 'false'
        context.launch_configurations['episodes'] = '7'
        context.launch_configurations['resume'] = 'false'
        trainer = nodes[executables.index('dqn_trainer')]
        params = evaluate_parameters(context, trainer._Node__parameters)[-1]
        assert params['max_episodes'] == 7
        assert params['resume'] is False
    else:
        context.launch_configurations['max_steps'] = '12'
        context.launch_configurations['target_coverage'] = '0.4'
        explorer = nodes[executables.index('dqn_explorer')]
        params = evaluate_parameters(context, explorer._Node__parameters)[-1]
        assert params['max_steps'] == 12
        assert params['target_coverage'] == 0.4
        rviz = nodes[executables.index('rviz2')]
        context.launch_configurations['use_rviz'] = 'false'
        assert not rviz.condition.evaluate(context)


@pytest.mark.parametrize(('resume', 'expected'), [('false', False), ('true', True)])
def test_training_launch_passes_resume_as_boolean(monkeypatch, resume, expected):
    """The launcher must pass each requested resume mode to the trainer."""
    pytest.importorskip('launch_ros')
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch.launch_description_sources import get_launch_description_from_python_launch_file
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters

    monkeypatch.setenv('TURTLEBOT3_MODEL', 'burger')
    description = get_launch_description_from_python_launch_file(
        str(PACKAGE_ROOT / 'launch' / 'training.launch.py'))
    context = LaunchContext()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)
    context.launch_configurations['resume'] = resume
    trainer = next(
        node for node in description.entities
        if isinstance(node, Node) and node.node_executable == 'dqn_trainer')
    params = evaluate_parameters(context, trainer._Node__parameters)[-1]
    assert params['resume'] is expected


def test_the_mission_stops_only_once_the_arena_is_covered(monkeypatch):
    """A default of 0.75 ends the run with a quarter of the arena unseen.

    The assignment asks for the whole scenario, the trainer aims at 0.98 and
    the explorer node's own default is 0.98; only the launcher disagreed, and
    it is the launcher that wins.  A run stopped itself at 75.3 % reporting
    'Mission ended: target_coverage' - working exactly as configured, and
    nothing like what was asked for.  The step cap has to clear the coverage
    target too, or it becomes the real limit: that same run needed 304 steps
    just to reach three quarters.
    """
    pytest.importorskip('launch_ros')
    from launch.actions import DeclareLaunchArgument
    from launch.launch_description_sources import get_launch_description_from_python_launch_file

    monkeypatch.setenv('TURTLEBOT3_MODEL', 'burger')
    description = get_launch_description_from_python_launch_file(
        str(PACKAGE_ROOT / 'launch' / 'mission.launch.py'))
    defaults = {action.name: action.default_value[0].text
                for action in description.entities
                if isinstance(action, DeclareLaunchArgument)}
    assert float(defaults['target_coverage']) >= 0.98
    assert int(defaults['max_steps']) >= 2000


@pytest.mark.parametrize('explorer,expected', [
    ('dqn', 'dqn_explorer'), ('frontier', 'frontier_explorer')])
def test_mission_runs_exactly_one_policy_for_the_selected_explorer(
        monkeypatch, explorer, expected):
    """Two policies publishing actions would fight over the same robot."""
    pytest.importorskip('launch_ros')
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch.launch_description_sources import get_launch_description_from_python_launch_file
    from launch_ros.actions import Node

    monkeypatch.setenv('TURTLEBOT3_MODEL', 'burger')
    description = get_launch_description_from_python_launch_file(
        str(PACKAGE_ROOT / 'launch' / 'mission.launch.py'))
    context = LaunchContext()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)
    context.launch_configurations['explorer'] = explorer
    nodes = [entity for entity in description.entities if isinstance(entity, Node)]
    enabled = [
        node.node_executable for node in nodes
        if node.node_executable in ('dqn_explorer', 'frontier_explorer')
        and node.condition.evaluate(context)
    ]
    assert enabled == [expected]


def test_training_launch_runs_the_console_monitor_under_a_flag(monkeypatch):
    """Training must show live metrics without a second manual terminal."""
    pytest.importorskip('launch_ros')
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch.launch_description_sources import get_launch_description_from_python_launch_file
    from launch_ros.actions import Node

    monkeypatch.setenv('TURTLEBOT3_MODEL', 'burger')
    description = get_launch_description_from_python_launch_file(
        str(PACKAGE_ROOT / 'launch' / 'training.launch.py'))
    context = LaunchContext()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)
    nodes = [entity for entity in description.entities if isinstance(entity, Node)]
    executables = [node.node_executable for node in nodes]
    assert executables.count('training_monitor') == 1
    monitor = nodes[executables.index('training_monitor')]
    assert monitor.condition.evaluate(context)
    context.launch_configurations['monitor'] = 'false'
    assert not monitor.condition.evaluate(context)


def test_official_stage4_gui_is_gated_without_removing_server(monkeypatch):
    pytest.importorskip('launch_ros')
    from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
    from launch import LaunchContext
    from launch.actions import GroupAction, IncludeLaunchDescription
    from turtleboot3_autonomous_nav.stage4_launch import Stage4LaunchSource

    try:
        share = get_package_share_directory('turtlebot3_gazebo')
    except PackageNotFoundError:
        pytest.skip('Official TurtleBot3 simulation sources are not installed')
    monkeypatch.setenv('TURTLEBOT3_MODEL', 'burger')
    context = LaunchContext()
    context.launch_configurations['use_gui'] = 'false'
    description = Stage4LaunchSource(str(Path(share) / 'launch' /
        'turtlebot3_dqn_stage4.launch.py')).get_launch_description(context)
    gui = [action for action in description.entities if isinstance(action, GroupAction)]
    assert len(gui) == 1
    assert not gui[0].condition.evaluate(context)
    context.launch_configurations['use_gui'] = 'true'
    assert gui[0].condition.evaluate(context)
    # World server, robot state publisher, and robot spawn stay included.
    assert sum(isinstance(action, IncludeLaunchDescription)
               for action in description.entities) == 3


def test_training_robot_is_part_of_full_reset_world(monkeypatch):
    """A late /create robot is deleted by Gazebo reset.all; include it at load."""
    pytest.importorskip('launch_ros')
    import xml.etree.ElementTree as ET
    from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
    from launch import LaunchContext
    from launch.actions import IncludeLaunchDescription
    from launch_ros.actions import Node
    from turtleboot3_autonomous_nav import stage4_launch

    try:
        share = Path(get_package_share_directory('turtlebot3_gazebo'))
    except PackageNotFoundError:
        pytest.skip('Official TurtleBot3 simulation sources are not installed')
    monkeypatch.setenv('TURTLEBOT3_MODEL', 'burger')
    context = LaunchContext()
    context.launch_configurations['use_gui'] = 'false'
    source_type = getattr(stage4_launch, 'TrainingStage4LaunchSource', None)
    assert source_type is not None, 'Training needs a robot in the initial world'
    description = source_type(str(share / 'launch' /
        'turtlebot3_dqn_stage4.launch.py')).get_launch_description(context)
    includes = [action for action in description.entities if isinstance(action, IncludeLaunchDescription)]
    server = next(action for action in includes if 'gz_args' in dict(action.launch_arguments))
    generated_path = Path(dict(server.launch_arguments)['gz_args'][-1])
    assert generated_path != share / 'worlds' / 'turtlebot3_dqn_stage4.world'
    generated = ET.parse(generated_path).getroot().find('world')
    official = ET.parse(share / 'worlds' / 'turtlebot3_dqn_stage4.world').getroot().find('world')
    robot = generated.findall('include')[-1]
    assert robot.findtext('uri') == 'model://turtlebot3_burger'
    assert robot.findtext('name') == 'burger'
    assert robot.findtext('pose') == '0 0 0.01 0 0 0'
    generated.remove(robot)
    # Two things are changed on purpose: the plugins that warp the moving
    # obstacles into the robot are dropped (test_teleporting_obstacles.py) and
    # the real-time cap is raised (test_simulation_pacing.py).  Normalise both
    # away so this still proves nothing else was touched.
    for tree in (generated, official):
        tree.find('physics/real_time_factor').text = 'normalised'
        for include in tree.iter('include'):
            for plugin in list(include.findall('plugin')):
                if any(name in (plugin.get('name') or '')
                       for name in stage4_launch.TELEPORTING_OBSTACLE_PLUGINS):
                    include.remove(plugin)
    assert ET.tostring(generated).strip() == ET.tostring(official).strip()
    spawn = next(action for action in includes if isinstance(
        action.launch_description_source, stage4_launch.InitialRobotBridgeSource))
    spawn_description = spawn.launch_description_source.get_launch_description(context)
    executables = [action.node_executable for action in spawn_description.entities if isinstance(action, Node)]
    assert 'create' not in executables
    assert executables == ['parameter_bridge']
