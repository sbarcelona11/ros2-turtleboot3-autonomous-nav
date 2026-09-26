"""Frontier planning: head for the nearest reachable unknown cell.

This is the deterministic counterpart to the learned policy.  A DQN trained
on discovery reward harvests the easy area and then circles ground it has
already monitored, because crossing known space earns nothing on the way.  A
breadth-first search over the map has no such blind spot: it always knows
which unexplored cell is closest and how to get there, so it keeps making
progress until nothing reachable is left.

The module has no ROS dependency so the planning contract stays testable.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np

from turtleboot3_autonomous_nav.control import (
    FORWARD,
    LEFT,
    RIGHT,
    SOFT_LEFT,
    SOFT_RIGHT,
)


UNKNOWN = -1
OCCUPIED_THRESHOLD = 50

SOFT_TURN_THRESHOLD = 0.15
"""Heading error, in radians, above which the robot corrects while advancing."""

HARD_TURN_THRESHOLD = 0.60
"""Heading error above which the robot turns in place instead of advancing."""


def clearance_cells_for(stop_distance: float, resolution: float) -> int:
    """Cells of inflation so a planned path stays out of braking range.

    A hand-tuned constant drifts from the controller it has to respect: three
    cells of inflation against a 0.20 m stop distance had the robot grazing
    walls and the safety controller intervening on most steps.  One cell past
    the threshold keeps a path that merely touches it from triggering.
    """
    if not float(resolution) > 0.0:
        raise ValueError('resolution must be positive')
    if float(stop_distance) <= 0.0:
        return 0
    return int(math.ceil(float(stop_distance) / float(resolution))) + 1


def nearest_frontier_step(
    grid: np.ndarray,
    robot_cell: tuple[int, int],
    clearance_cells: int = 0,
    lookahead: int = 1,
) -> tuple[int, int] | None:
    """Return the next cell towards the closest reachable unknown cell.

    ``clearance_cells`` inflates the obstacles by the robot's radius.  Without
    it a cell-level path grazes walls and the safety controller brakes on
    every step; with it the plan keeps its distance.  If the inflation would
    seal the only route, it is dropped rather than ending the mission early.

    ``lookahead`` returns a cell further along the path.  Aiming at the very
    next cell makes the desired heading swing wildly as the robot advances,
    which turns the mission into a sequence of pivots instead of driving.

    ``None`` means no unknown cell can be reached, which is how a finished
    mission reports itself: everything the robot can drive to has been seen.
    """
    occupancy = np.asarray(grid)
    if occupancy.ndim != 2 or occupancy.size == 0:
        raise ValueError('grid must be a non-empty two-dimensional array')
    height, width = occupancy.shape
    start = (int(robot_cell[0]), int(robot_cell[1]))
    if not (0 <= start[0] < height and 0 <= start[1] < width):
        raise ValueError('robot_cell must lie inside the grid')

    steps = max(1, int(lookahead))
    if int(clearance_cells) > 0:
        step = _search(
            occupancy, start, _inflated(occupancy, int(clearance_cells)), steps
        )
        if step is not None:
            return step
    return _search(occupancy, start, occupancy >= OCCUPIED_THRESHOLD, steps)


def distance_to_nearest_frontier(
    grid: np.ndarray,
    robot_cell: tuple[int, int],
    clearance_cells: int = 0,
) -> int | None:
    """Return the number of cells to the closest reachable unknown cell.

    This is the quantity the reward was missing.  Coverage pays for cells
    discovered, so crossing ground that is already mapped earns nothing on the
    way, and a policy trained on it learns to stay where the easy discoveries
    are: 63 % of the trained agent's decisions were turns in place.  A distance
    gives that journey a gradient to descend.

    It also tells the episode budget the difference between a robot that is
    stuck and one that is travelling.  Crossing this 4.85 m arena takes about
    48 seconds at 0.10 m/s, and the stall budget is 25, so an agent that
    correctly set off for the far side had its episode ended halfway there.

    ``None`` means nothing reachable is left to discover, which is how a
    finished exploration reports itself.  Distances count four-connected steps,
    the same moves the planner uses, so an obstacle in the way is reflected.
    """
    occupancy = np.asarray(grid)
    if occupancy.ndim != 2 or occupancy.size == 0:
        raise ValueError('grid must be a non-empty two-dimensional array')
    height, width = occupancy.shape
    start = (int(robot_cell[0]), int(robot_cell[1]))
    if not (0 <= start[0] < height and 0 <= start[1] < width):
        raise ValueError('robot_cell must lie inside the grid')

    if int(occupancy[start]) == UNKNOWN:
        return 0
    if int(clearance_cells) > 0:
        found = _search_distance(
            occupancy, start, _inflated(occupancy, int(clearance_cells))
        )
        if found is not None:
            return found
    return _search_distance(occupancy, start, occupancy >= OCCUPIED_THRESHOLD)


def _search_distance(
    occupancy: np.ndarray,
    start: tuple[int, int],
    blocked: np.ndarray,
) -> int | None:
    """Breadth-first search returning the step count to the nearest unknown."""
    height, width = occupancy.shape
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        cell, depth = queue.popleft()
        row, column = cell
        for neighbour in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if neighbour in seen:
                continue
            if not (0 <= neighbour[0] < height and 0 <= neighbour[1] < width):
                continue
            seen.add(neighbour)
            if int(occupancy[neighbour]) == UNKNOWN:
                return depth + 1
            if blocked[neighbour]:
                # Reachable as a place to see, never as a place to route through.
                continue
            queue.append((neighbour, depth + 1))
    return None


def _inflated(occupancy: np.ndarray, cells: int) -> np.ndarray:
    """Grow the occupied mask so a planned path clears the robot's body."""
    blocked = occupancy >= OCCUPIED_THRESHOLD
    grown = blocked.copy()
    for _ in range(cells):
        padded = grown.copy()
        padded[1:, :] |= grown[:-1, :]
        padded[:-1, :] |= grown[1:, :]
        padded[:, 1:] |= grown[:, :-1]
        padded[:, :-1] |= grown[:, 1:]
        grown = padded
    return grown


def _search(
    occupancy: np.ndarray,
    start: tuple[int, int],
    blocked: np.ndarray,
    lookahead: int,
) -> tuple[int, int] | None:
    """Breadth-first search returning a waypoint along the path to a frontier."""
    height, width = occupancy.shape
    parent: dict[tuple[int, int], tuple[int, int]] = {start: start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        row, column = cell
        for neighbour in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if neighbour in parent:
                continue
            if not (0 <= neighbour[0] < height and 0 <= neighbour[1] < width):
                continue
            parent[neighbour] = cell
            if int(occupancy[neighbour]) == UNKNOWN:
                return _waypoint(parent, start, neighbour, lookahead)
            if blocked[neighbour]:
                # An inflated cell is a valid goal - the robot only has to see
                # it - but never a cell to route through.
                del parent[neighbour]
                continue
            queue.append(neighbour)
    return None


def _waypoint(
    parent: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    lookahead: int,
) -> tuple[int, int]:
    """Return the cell ``lookahead`` steps along the path, or the goal itself."""
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path[min(lookahead, len(path) - 1)]


def action_for_heading(current_yaw: float, desired_yaw: float) -> int:
    """Return the discrete action that reduces the heading error.

    Small errors are corrected while driving, because stopping to turn for
    every minor correction wastes the mission's time budget; large ones turn
    in place, because advancing while badly aimed overshoots the target.
    """
    for value in (current_yaw, desired_yaw):
        if not math.isfinite(float(value)):
            raise ValueError('headings must be finite')
    error = math.atan2(
        math.sin(float(desired_yaw) - float(current_yaw)),
        math.cos(float(desired_yaw) - float(current_yaw)),
    )
    magnitude = abs(error)
    if magnitude < SOFT_TURN_THRESHOLD:
        return FORWARD
    if magnitude < HARD_TURN_THRESHOLD:
        return SOFT_LEFT if error > 0.0 else SOFT_RIGHT
    return LEFT if error > 0.0 else RIGHT
