"""The observation must describe the whole scenario, not a 0.4 m neighbourhood.

A policy can only steer towards unexplored area that it can perceive.  These
tests pin the scenario-scale fields: an arena map in the fixed frame, unknown
area gains measured across that whole map, and the pose that locates the robot
inside it.
"""

from pathlib import Path

import numpy as np
import pytest

from turtleboot3_autonomous_nav.observation import (
    ARENA_MAP_SIZE,
    LIDAR_MAX_RANGE,
    OBSERVATION_SIZE,
    arena_map,
    build_observation,
    directional_unknown_gains,
    normalized_pose,
)

CONFIG_ROOT = Path(__file__).resolve().parents[1] / 'config'


def _unknown_grid(size=100):
    return np.full((size, size), -1, dtype=np.int8)


def test_arena_map_reports_one_cell_per_block_of_the_whole_grid():
    """The map summarises the entire grid, so its scale follows the scenario."""
    grid = _unknown_grid(100)
    grid[:, :] = 0
    assert arena_map(grid).shape == (ARENA_MAP_SIZE, ARENA_MAP_SIZE)
    assert np.all(arena_map(grid) == 0)


def test_a_block_holding_any_occupied_cell_reads_occupied():
    """Obstacles must survive downsampling or the agent would drive into them."""
    grid = np.zeros((100, 100), dtype=np.int8)
    grid[0, 0] = 100
    assert arena_map(grid)[0, 0] == 1


def test_a_block_with_free_and_unknown_cells_reads_free():
    """Partially observed blocks count as visited, not as pending frontier."""
    grid = _unknown_grid(100)
    grid[0, 0] = 0
    assert arena_map(grid)[0, 0] == 0


def test_a_fully_unknown_block_stays_unknown():
    """Unexplored area must remain visible as the target of exploration."""
    assert arena_map(_unknown_grid(100))[0, 0] == -1


def test_gains_point_towards_the_unknown_half_of_the_scenario():
    """Direction matters: an explored west half must not attract the robot."""
    grid = _unknown_grid(100)
    grid[:, :50] = 0
    gains = directional_unknown_gains(grid, (50, 50))
    east = gains[0]
    west = gains[4]
    assert east > west


def test_gains_are_measured_across_the_whole_grid_not_a_local_window():
    """Unknown area metres away must still register, unlike the old patch."""
    grid = np.zeros((100, 100), dtype=np.int8)
    grid[:, 90:] = -1
    gains = directional_unknown_gains(grid, (50, 50))
    assert gains[0] > 0.0


def test_gains_are_zero_for_a_fully_explored_scenario():
    """With nothing left unknown the exploration signal must vanish."""
    gains = directional_unknown_gains(np.zeros((100, 100), dtype=np.int8), (50, 50))
    assert np.allclose(gains, np.zeros(8))


def test_pose_is_normalized_to_the_grid_extent():
    """Normalising by the grid keeps the pose comparable across map sizes."""
    assert np.allclose(normalized_pose((0, 0), (100, 100)), [-1.0, -1.0])
    assert np.allclose(normalized_pose((99, 99), (100, 100)), [1.0, 1.0])
    assert np.allclose(normalized_pose((50, 50), (100, 100)), [0.0, 0.0], atol=0.03)


def test_observation_layout_carries_scenario_scale_fields():
    """The vector keeps lidar, gains, arena map, pose, heading and velocity."""
    grid = _unknown_grid(100)
    grid[:, :50] = 0
    observation = build_observation(
        np.full(360, 2.0), grid, (50, 50), 0.0, 0.12, -0.34
    )
    assert OBSERVATION_SIZE == 90
    assert observation.shape == (90,)
    assert np.allclose(observation[:12], np.full(12, 2.0 / LIDAR_MAX_RANGE))
    assert observation[12] > observation[16]
    assert set(np.unique(observation[20:84])).issubset({-1.0, 0.0, 1.0})
    assert np.allclose(observation[86:88], [0.0, 1.0])
    assert np.allclose(observation[-2:], [0.12, -0.34])


def test_observation_reports_heading_as_sine_and_cosine():
    """A wrapped angle would jump at pi; sine and cosine stay continuous."""
    observation = build_observation(
        np.full(360, 1.0), _unknown_grid(100), (50, 50), np.pi / 2.0, 0.0, 0.0
    )
    assert np.allclose(observation[86:88], [1.0, 0.0], atol=1e-6)


@pytest.mark.parametrize('cell', [(-1, 0), (0, -1), (100, 0), (0, 100)])
def test_a_pose_outside_the_grid_is_rejected(cell):
    with pytest.raises(ValueError):
        build_observation(np.full(360, 1.0), _unknown_grid(100), cell, 0.0, 0.0, 0.0)


def test_the_trainer_and_the_shipped_config_expect_the_built_vector():
    """A stale observation_size would make every observation be rejected."""
    import yaml

    from turtleboot3_autonomous_nav.dqn_trainer import TrainerConfig

    config = yaml.safe_load(
        (CONFIG_ROOT / 'training.yaml').read_text()
    )['dqn_trainer']['ros__parameters']
    assert TrainerConfig().observation_size == OBSERVATION_SIZE
    assert config['observation_size'] == OBSERVATION_SIZE
