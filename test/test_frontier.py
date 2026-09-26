"""Frontier exploration: always head for the nearest reachable unknown cell.

The learned policy harvests the easy area and then circles ground it has
already monitored, because crossing known space earns nothing on the way.  A
frontier search has no such blind spot: it plans through the map to the
closest unexplored cell, so it keeps making progress until nothing reachable
is left.
"""

import math

import numpy as np
import pytest

from turtleboot3_autonomous_nav.control import (
    FORWARD,
    LEFT,
    RIGHT,
    SOFT_LEFT,
    SOFT_RIGHT,
)
from turtleboot3_autonomous_nav.frontier import (
    action_for_heading,
    nearest_frontier_step,
)

UNKNOWN, FREE, OCCUPIED = -1, 0, 100


def _room(size=9):
    """A free room with an occupied border."""
    grid = np.full((size, size), FREE, dtype=np.int8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = OCCUPIED
    return grid


def test_the_step_moves_towards_the_only_unknown_cell():
    """One step of the plan is all the controller needs each cycle."""
    grid = _room()
    grid[4, 7] = UNKNOWN
    assert nearest_frontier_step(grid, (4, 4)) == (4, 5)


def test_the_nearest_frontier_wins_over_a_distant_one():
    grid = _room(11)
    grid[5, 3] = UNKNOWN
    grid[5, 9] = UNKNOWN
    assert nearest_frontier_step(grid, (5, 5)) == (5, 4)


def test_the_plan_goes_around_an_obstacle():
    """A straight line would drive into the wall; the search must detour."""
    grid = _room(9)
    grid[1:8, 5] = OCCUPIED
    grid[4, 5] = FREE
    grid[4, 7] = UNKNOWN
    step = nearest_frontier_step(grid, (2, 3))
    assert step is not None
    assert grid[step] != OCCUPIED


def test_a_fully_explored_map_reports_no_frontier():
    """That is the signal the mission is finished, not an error."""
    assert nearest_frontier_step(_room(), (4, 4)) is None


def test_unknown_space_sealed_behind_walls_is_not_a_frontier():
    """Chasing an unreachable cell would strand the robot against a wall."""
    grid = _room(11)
    grid[3:8, 3:8] = OCCUPIED
    grid[5, 5] = UNKNOWN
    assert nearest_frontier_step(grid, (1, 1)) is None


def test_a_plan_keeps_clear_of_walls_when_clearance_is_requested():
    """A cell-level path hugs walls, and the safety controller then brakes.

    Measured on the real robot: 110 safety interventions in 30 seconds and
    under a metre travelled, because every planned step grazed an obstacle.
    """
    grid = _room(11)
    grid[1, 1:10] = OCCUPIED
    grid[9, 5] = UNKNOWN
    step = nearest_frontier_step(grid, (5, 5), clearance_cells=2)
    assert step is not None
    # Moving away from the wall, not along it.
    assert step[0] > 5


def test_clearance_is_dropped_when_it_would_seal_the_only_route():
    """A corridor narrower than the robot must not end the mission early."""
    grid = np.full((9, 9), OCCUPIED, dtype=np.int8)
    grid[4, 1:8] = FREE
    grid[4, 7] = UNKNOWN
    assert nearest_frontier_step(grid, (4, 1), clearance_cells=2) == (4, 2)


def test_a_lookahead_aims_further_along_the_path():
    """Aiming at the adjacent cell makes the heading swing on every update.

    Measured on the real robot: 87 of 141 actions were turns in place, and it
    covered under a metre in thirty seconds.  Aiming further ahead smooths the
    heading so the robot drives instead of pivoting.
    """
    grid = _room(15)
    grid[7, 12] = UNKNOWN
    near = nearest_frontier_step(grid, (7, 3), lookahead=1)
    far = nearest_frontier_step(grid, (7, 3), lookahead=4)
    assert near == (7, 4)
    assert far == (7, 7)


def test_the_lookahead_stops_at_the_goal_on_a_short_path():
    """Overshooting past the target would point the robot at nothing."""
    grid = _room(9)
    grid[4, 6] = UNKNOWN
    assert nearest_frontier_step(grid, (4, 5), lookahead=9) == (4, 6)


def test_a_robot_outside_the_grid_is_rejected():
    with pytest.raises(ValueError):
        nearest_frontier_step(_room(), (99, 99))


def test_an_aligned_heading_drives_forward():
    assert action_for_heading(0.0, 0.02) == FORWARD


def test_a_small_error_to_the_left_turns_softly_while_advancing():
    """Soft turns keep the robot moving, so small corrections cost nothing."""
    assert action_for_heading(0.0, 0.4) == SOFT_LEFT


def test_a_small_error_to_the_right_turns_softly():
    assert action_for_heading(0.0, -0.4) == SOFT_RIGHT


def test_a_large_error_turns_in_place():
    """Driving forward while badly misaligned would overshoot the target."""
    assert action_for_heading(0.0, 2.0) == LEFT
    assert action_for_heading(0.0, -2.0) == RIGHT


def test_the_shortest_rotation_is_chosen_across_the_pi_boundary():
    """Unwrapped, these differ by 6 rad and would turn the long way round.

    Wrapped, the error is 0.28 rad, so the answer is a soft correction in the
    opposite direction: asserting the soft turn pins both the direction and
    the magnitude.
    """
    assert action_for_heading(3.0, -3.0) == SOFT_LEFT
    assert action_for_heading(-3.0, 3.0) == SOFT_RIGHT


def test_a_wrapped_error_can_still_require_turning_in_place():
    """The wrap must not shrink every error into a soft correction."""
    assert action_for_heading(3.0, -2.0) == LEFT
    assert action_for_heading(-3.0, 2.0) == RIGHT


@pytest.mark.parametrize('yaw', [math.nan, math.inf])
def test_a_non_finite_heading_is_rejected(yaw):
    with pytest.raises(ValueError):
        action_for_heading(yaw, 0.0)
