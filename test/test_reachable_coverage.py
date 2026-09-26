"""Coverage must be measured against what the robot can still reach.

Counting every grid cell makes a full mission look unfinished: the inside of
walls, the inside of obstacles and the space beyond them can never be
observed.  Measuring against the reachable region instead lets a completed
exploration read as complete, and keeps the shadows behind obstacles - which
are reachable - counted as pending work.
"""

import numpy as np

from turtleboot3_autonomous_nav.grid_mapping import OccupancyGridModel


UNKNOWN = OccupancyGridModel.UNKNOWN
FREE = OccupancyGridModel.FREE
OCCUPIED = OccupancyGridModel.OCCUPIED


def _model(grid):
    """Build a model whose every known cell also counts as monitored.

    These tests isolate the reachable-region rule, so proximity is taken as
    already satisfied; `test_proximity_monitoring` covers the radius itself.
    """
    model = OccupancyGridModel(grid.shape[1], grid.shape[0], 0.05, (0.0, 0.0))
    model._grid = np.asarray(grid, dtype=np.int8)
    model._monitored = model._grid != OccupancyGridModel.UNKNOWN
    return model


def test_an_unobserved_map_reports_no_coverage():
    """Before the first scan there is no reachable region to have covered."""
    assert _model(np.full((5, 5), UNKNOWN, dtype=np.int8)).reachable_coverage_fraction() == 0.0


def test_a_fully_observed_room_reports_complete_coverage():
    """Every reachable cell is known, so the mission is finished."""
    grid = np.full((5, 5), FREE, dtype=np.int8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = OCCUPIED
    assert _model(grid).reachable_coverage_fraction() == 1.0


def test_a_shadow_reachable_through_free_space_counts_as_pending():
    """Unknown cells the robot can still drive to must lower the fraction."""
    grid = np.full((5, 5), FREE, dtype=np.int8)
    grid[2, 2] = UNKNOWN
    fraction = _model(grid).reachable_coverage_fraction()
    assert fraction == 24 / 25


def test_unknown_space_sealed_behind_occupied_cells_is_excluded():
    """The inside of a wall is unobservable, so it must not count as pending."""
    grid = np.full((6, 6), UNKNOWN, dtype=np.int8)
    grid[1:5, 1:5] = OCCUPIED
    grid[2:4, 2:4] = UNKNOWN
    grid[0, 0] = FREE
    model = _model(grid)
    # The free corner is the only reachable known cell; the sealed pocket and
    # the cells outside the ring are what remain, and the pocket is excluded.
    assert model.reachable_coverage_fraction() < 1.0
    grid[0, :] = grid[:, 0] = grid[5, :] = grid[:, 5] = FREE
    assert _model(grid).reachable_coverage_fraction() == 1.0


def test_occupied_cells_are_not_counted_as_area_to_monitor():
    """Obstacles are not space the robot must observe the inside of."""
    grid = np.full((4, 4), FREE, dtype=np.int8)
    grid[1:3, 1:3] = OCCUPIED
    assert _model(grid).reachable_coverage_fraction() == 1.0


def test_coverage_rises_as_a_shadow_is_revealed():
    """The measure must reward clearing the reachable unknown region."""
    grid = np.full((5, 5), FREE, dtype=np.int8)
    grid[2, 2] = grid[2, 3] = UNKNOWN
    before = _model(grid).reachable_coverage_fraction()
    grid[2, 2] = FREE
    assert _model(grid).reachable_coverage_fraction() > before
