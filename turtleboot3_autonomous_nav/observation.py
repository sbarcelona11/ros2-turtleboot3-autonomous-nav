"""Fixed-size DQN observation construction independent of ROS messages.

The observation describes the scenario, not only the robot's immediate
surroundings.  A policy cannot steer towards unexplored area it cannot
perceive, so the vector carries an arena-scale map, unknown-area gains
measured across that whole map, and the pose that locates the robot inside
it.  Fine obstacle detail comes from the LiDAR sectors instead.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


OBSERVATION_CONTRACT = 'monitoring-map-v1'


def observation_label(epoch: int) -> str:
    return f'{OBSERVATION_CONTRACT}:episode:{int(epoch)}'


def has_observation_contract(label: str) -> bool:
    return str(label).startswith(f'{OBSERVATION_CONTRACT}:episode:')


def require_observation_contract(config: Mapping[str, Any]) -> None:
    if config.get('observation_contract') != OBSERVATION_CONTRACT:
        raise ValueError(
            'checkpoint observation contract is incompatible; retrain with '
            f'{OBSERVATION_CONTRACT}'
        )


LIDAR_SECTOR_COUNT = 12
DIRECTIONAL_GAIN_COUNT = 8
ARENA_MAP_SIZE = 8
POSE_SIZE = 2
HEADING_SIZE = 2
OBSERVATION_SIZE = (
    LIDAR_SECTOR_COUNT
    + DIRECTIONAL_GAIN_COUNT
    + ARENA_MAP_SIZE * ARENA_MAP_SIZE
    + POSE_SIZE
    + HEADING_SIZE
    + 2
)

UNKNOWN = -1
"""Occupancy value of a cell that no LiDAR ray has resolved yet."""

LIDAR_MAX_RANGE = 3.5
"""Burger LiDAR range, used to bring the sectors onto the shared input scale.

The map is encoded as -1/0/1 and the gains span 0 to 1, so leaving the
sectors in metres would let them dominate the first layer of a small network
trained with plain gradient descent.
"""

_OCCUPIED_THRESHOLD = 50


def build_observation(
    scan_ranges: np.ndarray,
    coverage_grid: np.ndarray,
    robot_cell: tuple[int, int],
    heading: float,
    linear_velocity: float,
    angular_velocity: float,
) -> np.ndarray:
    """Build the observation vector from LiDAR, the map, the pose and motion.

    The layout is twelve LiDAR-sector minima, eight directional unknown-area
    gains over the whole map, an arena map in the fixed frame encoded as
    ``-1``/``0``/``1``, the normalized position, the heading as sine and
    cosine, then linear and angular velocity.
    """
    grid = _validated_grid(coverage_grid, robot_cell)
    return np.concatenate(
        (
            _lidar_sector_minima(scan_ranges),
            directional_unknown_gains(grid, robot_cell),
            arena_map(grid).reshape(-1).astype(np.float32),
            np.asarray(normalized_pose(robot_cell, grid.shape), dtype=np.float32),
            np.asarray(
                [math.sin(float(heading)), math.cos(float(heading))], dtype=np.float32
            ),
            np.asarray(
                [_finite_or_zero(linear_velocity), _finite_or_zero(angular_velocity)],
                dtype=np.float32,
            ),
        )
    ).astype(np.float32, copy=False)


def arena_map(coverage_grid: np.ndarray) -> np.ndarray:
    """Summarise the whole map as a fixed ``ARENA_MAP_SIZE`` square.

    A block reads occupied when it holds any occupied cell, so obstacles
    survive the reduction; it reads free when it holds any resolved cell, so
    only genuinely unvisited blocks stay unknown and attract exploration.
    """
    grid = np.asarray(coverage_grid)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError('coverage_grid must be a non-empty two-dimensional array')
    reduced = np.empty((ARENA_MAP_SIZE, ARENA_MAP_SIZE), dtype=np.float32)
    for row_index, row_block in enumerate(np.array_split(grid, ARENA_MAP_SIZE, axis=0)):
        for column_index, block in enumerate(
            np.array_split(row_block, ARENA_MAP_SIZE, axis=1)
        ):
            reduced[row_index, column_index] = _reduce_block(block)
    return reduced


def directional_unknown_gains(
    coverage_grid: np.ndarray, robot_cell: tuple[int, int]
) -> np.ndarray:
    """Return the unknown fraction of each 45 degree sector around the robot.

    The sectors span the entire map, so unexplored area metres away still
    produces a signal the policy can follow.
    """
    grid = _validated_grid(coverage_grid, robot_cell)
    rows, columns = np.indices(grid.shape)
    delta_y = rows - int(robot_cell[0])
    delta_x = columns - int(robot_cell[1])
    outside_centre = (delta_x != 0) | (delta_y != 0)
    sector = np.floor(
        (np.arctan2(delta_y, delta_x) + math.pi / 8.0) / (math.pi / 4.0)
    ).astype(int) % DIRECTIONAL_GAIN_COUNT
    totals = np.bincount(
        sector[outside_centre], minlength=DIRECTIONAL_GAIN_COUNT
    ).astype(np.float32)
    unknown = np.bincount(
        sector[outside_centre & (grid == UNKNOWN)], minlength=DIRECTIONAL_GAIN_COUNT
    ).astype(np.float32)
    return np.divide(
        unknown, totals, out=np.zeros_like(unknown), where=totals != 0
    ).astype(np.float32)


def normalized_pose(
    robot_cell: tuple[int, int], grid_shape: tuple[int, int]
) -> tuple[float, float]:
    """Return the robot position in ``[-1, 1]`` along each map axis."""
    height, width = int(grid_shape[0]), int(grid_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError('grid_shape must be positive')
    return (
        _normalized_axis(int(robot_cell[1]), width),
        _normalized_axis(int(robot_cell[0]), height),
    )


def _normalized_axis(index: int, size: int) -> float:
    if size == 1:
        return 0.0
    return 2.0 * float(index) / float(size - 1) - 1.0


def _reduce_block(block: np.ndarray) -> float:
    if np.any(block >= _OCCUPIED_THRESHOLD):
        return 1.0
    if np.any(block != UNKNOWN):
        return 0.0
    return -1.0


def _validated_grid(
    coverage_grid: np.ndarray, robot_cell: tuple[int, int]
) -> np.ndarray:
    grid = np.asarray(coverage_grid)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError('coverage_grid must be a non-empty two-dimensional array')
    row, column = int(robot_cell[0]), int(robot_cell[1])
    if not (0 <= row < grid.shape[0] and 0 <= column < grid.shape[1]):
        raise ValueError('robot_cell must lie inside the coverage grid')
    return grid


def _lidar_sector_minima(scan_ranges: np.ndarray) -> np.ndarray:
    ranges = np.asarray(scan_ranges, dtype=float).reshape(-1)
    minima = np.zeros(LIDAR_SECTOR_COUNT, dtype=np.float32)
    for index, sector in enumerate(np.array_split(ranges, LIDAR_SECTOR_COUNT)):
        valid = sector[np.isfinite(sector) & (sector > 0.0)]
        if valid.size:
            minima[index] = float(np.min(valid))
    return np.clip(minima / LIDAR_MAX_RANGE, 0.0, 1.0).astype(np.float32)


def _finite_or_zero(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0
