"""How far the robot is from the nearest thing left to discover.

The trained policy spends 63 % of its decisions turning in place, because
nothing in the reward distinguishes a robot heading for unexplored ground from
one spinning where it stands.  Discovery pays; travelling towards discovery
pays nothing.  A distance to the nearest frontier is the missing quantity: it
gives the reward something to shape against, and it tells the episode budget
that a robot crossing known ground is working rather than stalled.

The same breadth-first search already drives the deterministic explorer to
97.97 % coverage, so this is not a new idea about the scenario - it is the
signal that explorer uses, made available to the learner.
"""

import numpy as np
import pytest

from turtleboot3_autonomous_nav.frontier import (
    UNKNOWN,
    distance_to_nearest_frontier,
)

FREE = 0
OCCUPIED = 100


def test_standing_next_to_an_unknown_cell_is_one_step_away():
    grid = np.full((5, 5), FREE, dtype=int)
    grid[2, 3] = UNKNOWN
    assert distance_to_nearest_frontier(grid, (2, 2)) == 1


def test_distance_grows_with_the_gap_to_the_unknown_cell():
    grid = np.full((5, 9), FREE, dtype=int)
    grid[2, 8] = UNKNOWN
    assert distance_to_nearest_frontier(grid, (2, 2)) == 6


def test_the_route_bends_around_an_obstacle():
    """A straight-line distance would understate the work to get there."""
    grid = np.full((5, 5), FREE, dtype=int)
    grid[:, 2] = OCCUPIED
    grid[4, 2] = FREE
    grid[2, 4] = UNKNOWN
    # Straight across would be two cells; the only gap is at the bottom row.
    assert distance_to_nearest_frontier(grid, (2, 0)) > 2


def test_a_fully_explored_map_has_no_frontier():
    """`None` is what a finished exploration reports, not a zero distance."""
    grid = np.full((4, 4), FREE, dtype=int)
    assert distance_to_nearest_frontier(grid, (1, 1)) is None


def test_unknown_space_sealed_behind_obstacles_does_not_count():
    """The inside of a wall is not work; treating it as work never ends."""
    grid = np.full((7, 7), FREE, dtype=int)
    grid[2:5, 2:5] = OCCUPIED
    grid[3, 3] = UNKNOWN
    assert distance_to_nearest_frontier(grid, (0, 0)) is None


def test_standing_on_the_frontier_reads_as_zero():
    """A robot already on unknown ground has nothing left to travel."""
    grid = np.full((5, 5), FREE, dtype=int)
    grid[2, 2] = UNKNOWN
    assert distance_to_nearest_frontier(grid, (2, 2)) == 0


def test_a_robot_outside_the_grid_is_rejected():
    grid = np.full((4, 4), FREE, dtype=int)
    with pytest.raises(ValueError):
        distance_to_nearest_frontier(grid, (9, 9))


def test_clearance_does_not_hide_a_frontier_it_cannot_route_to():
    """Inflation may seal the only corridor; a sealed map must still report."""
    grid = np.full((5, 5), FREE, dtype=int)
    grid[2, 4] = UNKNOWN
    inflated = distance_to_nearest_frontier(grid, (2, 0), clearance_cells=3)
    assert inflated is not None
