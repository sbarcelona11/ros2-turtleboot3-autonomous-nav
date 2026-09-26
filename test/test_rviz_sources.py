"""RViz must show the pose the rest of the stack uses.

The odometry display pointed at `/odom`, the wheel-integrated pose that drifts
past a metre, and kept fifty of them.  On screen that became a dense burst of
arrows piled up outside the arena while the map and the laser were drawn
correctly inside it - it looked like the mapping was broken when it was the
display reading a different pose.
"""

from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[1] / 'rviz' / 'exploration.rviz'


def _displays():
    config = yaml.safe_load(CONFIG.read_text())
    for panel in config['Visualization Manager']['Displays']:
        yield panel


def test_the_odometry_display_shows_the_pose_the_map_is_built_with():
    """Showing the drifting pose next to a correct map is misleading."""
    odometry = [d for d in _displays() if d['Class'].endswith('Odometry')]
    assert odometry, 'the odometry display disappeared'
    assert odometry[0]['Topic']['Value'] == '/odom_truth'


def test_no_display_reads_the_drifting_wheel_odometry():
    sources = {
        display['Topic']['Value']
        for display in _displays()
        if isinstance(display.get('Topic'), dict)
    }
    assert '/odom' not in sources


def test_the_fixed_frame_matches_the_frame_the_map_is_published_in():
    config = yaml.safe_load(CONFIG.read_text())
    assert config['Visualization Manager']['Global Options']['Fixed Frame'] == 'odom'


def test_the_odometry_display_does_not_pile_up_history():
    """Fifty retained arrows turn a small drift into a wall of clutter."""
    odometry = [d for d in _displays() if d['Class'].endswith('Odometry')][0]
    assert odometry['Keep'] <= 5
