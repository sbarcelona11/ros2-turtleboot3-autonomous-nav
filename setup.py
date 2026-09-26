from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'turtleboot3_autonomous_nav'


def resource_files(directory):
    files = [path for path in glob(os.path.join(directory, '*')) if os.path.isfile(path)]
    if not files:
        return []
    return [(os.path.join('share', package_name, directory), files)]


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        *resource_files('launch'),
        *resource_files('config'),
        *resource_files('dependencies'),
        *resource_files('rviz'),
        *resource_files('models'),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='turtleboot3_autonomous_nav maintainers',
    maintainer_email='maintainer@example.com',
    description='Autonomous TurtleBot3 coverage mapping and DQN exploration.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'coverage_mapper = turtleboot3_autonomous_nav.coverage_mapper:main',
            'observation_builder = turtleboot3_autonomous_nav.observation_builder:main',
            'safe_motion_controller = turtleboot3_autonomous_nav.safe_motion_controller:main',
            'dqn_explorer = turtleboot3_autonomous_nav.dqn_explorer:main',
            'frontier_explorer = turtleboot3_autonomous_nav.frontier_explorer:main',
            'simulation_localizer = turtleboot3_autonomous_nav.simulation_localizer:main',
            'dqn_trainer = turtleboot3_autonomous_nav.dqn_trainer:main',
            'exploration_visualizer = turtleboot3_autonomous_nav.exploration_visualizer:main',
            'training_monitor = turtleboot3_autonomous_nav.training_monitor:main',
            'install_turtlebot3_dependencies = turtleboot3_autonomous_nav.dependency_installer:main',
        ],
    },
)
