"""Every input must reach the network on a comparable scale.

The LiDAR sectors arrived in metres, spanning 0 to 3.5, while the arena map
is -1/0/1 and the gains are 0 to 1.  A small network trained with plain SGD
lets the widest input dominate the first layer, which is one reason the policy
collapsed to a single action instead of learning where to drive.
"""

import numpy as np

from turtleboot3_autonomous_nav.observation import (
    LIDAR_MAX_RANGE,
    OBSERVATION_SIZE,
    build_observation,
)


def _grid():
    grid = np.full((100, 100), -1, dtype=np.int8)
    grid[3:97, 3:97] = 0
    return grid


def _observe(scan):
    return build_observation(scan, _grid(), (50, 50), 0.0, 0.0, 0.0)


def test_lidar_sectors_are_reported_as_a_fraction_of_the_maximum_range():
    observation = _observe(np.full(360, LIDAR_MAX_RANGE / 2.0))
    assert np.allclose(observation[:12], np.full(12, 0.5))


def test_a_reading_at_the_maximum_range_saturates_at_one():
    assert np.allclose(_observe(np.full(360, LIDAR_MAX_RANGE))[:12], np.ones(12))


def test_a_reading_beyond_the_maximum_range_is_clipped():
    """An overshooting reading must not push an input outside the shared scale."""
    assert np.allclose(_observe(np.full(360, 12.0))[:12], np.ones(12))


def test_every_input_stays_within_the_shared_range():
    """No field may dominate the first layer by sheer magnitude."""
    observation = _observe(np.linspace(0.05, 4.0, 360))
    assert observation.shape == (OBSERVATION_SIZE,)
    assert np.all(np.abs(observation[:-2]) <= 1.0)


def test_a_blocked_sector_still_reads_as_zero():
    """Zero must keep meaning "no clearance" after the rescaling."""
    assert _observe(np.full(360, 0.0))[0] == 0.0
