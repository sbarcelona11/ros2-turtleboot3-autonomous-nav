"""The planner must chase what the mission actually measures.

Coverage counts cells observed from close range, but the occupancy grid marks
a cell known as soon as any ray touches it.  A frontier search over occupancy
therefore runs out of targets while a fifth of the arena has never been
approached: measured on the robot, the mission declared itself complete at
81.4% coverage.  Publishing the monitoring state as its own grid makes the
planner's frontier and the mission's metric the same thing.
"""

import numpy as np

from turtleboot3_autonomous_nav.grid_mapping import OccupancyGridModel


UNKNOWN, FREE, OCCUPIED = -1, 0, 100


def _model(inspection_radius=1.0):
    return OccupancyGridModel(
        40, 40, 0.05, (-1.0, -1.0), inspection_radius=inspection_radius
    )


def test_a_cell_seen_from_afar_is_a_frontier_for_the_planner():
    """It is known, so occupancy hides it; it is unmonitored, so it is work."""
    model = _model(inspection_radius=0.2)
    model.update_scan((0.0, 0.0, 0.0), np.asarray([0.9]), 0.0, 0.01, 3.5)

    occupancy = model.occupancy_grid()
    monitoring = model.monitoring_grid()

    far = model._world_to_cell(0.6, 0.0)[::-1]
    assert occupancy[far] != UNKNOWN
    assert monitoring[far] == UNKNOWN


def test_a_monitored_cell_is_free_in_both_grids():
    model = _model(inspection_radius=1.0)
    model.update_scan((0.0, 0.0, 0.0), np.asarray([0.5]), 0.0, 0.01, 3.5)

    near = model._world_to_cell(0.2, 0.0)[::-1]
    assert model.occupancy_grid()[near] == FREE
    assert model.monitoring_grid()[near] == FREE


def test_obstacles_stay_occupied_so_the_planner_routes_around_them():
    """A wall the robot has approached must not become a driveable frontier."""
    model = _model(inspection_radius=2.0)
    model.update_scan((0.0, 0.0, 0.0), np.asarray([0.5]), 0.0, 0.01, 3.5)

    monitoring = model.monitoring_grid()
    assert np.any(monitoring == OCCUPIED)
    assert set(np.unique(monitoring)).issubset({UNKNOWN, FREE, OCCUPIED})


def test_an_unobserved_map_is_all_frontier():
    assert np.all(_model().monitoring_grid() == UNKNOWN)
