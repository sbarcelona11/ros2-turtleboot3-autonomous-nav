"""A planned path must never enter the distance at which the robot brakes.

With three cells of inflation (0.15 m) and a stop distance of 0.20 m, the
planner routed the robot closer to walls than the safety controller tolerates:
133 interventions in thirty seconds and visible wall grazing.  Deriving the
inflation from the stop distance keeps the two from drifting apart, the way a
hand-tuned constant did.
"""

import pytest

from turtleboot3_autonomous_nav.control import ControlConfig
from turtleboot3_autonomous_nav.frontier import clearance_cells_for


def test_the_inflation_exceeds_the_distance_the_controller_brakes_at():
    """Equal is not enough: a path grazing the threshold still triggers it."""
    cells = clearance_cells_for(ControlConfig().stop_distance, 0.05)
    assert cells * 0.05 > ControlConfig().stop_distance


def test_a_finer_map_needs_more_cells_for_the_same_distance():
    assert clearance_cells_for(0.20, 0.025) > clearance_cells_for(0.20, 0.05)


def test_a_coarser_map_needs_fewer_cells():
    assert clearance_cells_for(0.20, 0.10) < clearance_cells_for(0.20, 0.05)


@pytest.mark.parametrize('resolution', [0.0, -0.05])
def test_a_non_positive_resolution_is_rejected(resolution):
    with pytest.raises(ValueError):
        clearance_cells_for(0.20, resolution)


def test_a_non_positive_stop_distance_plans_without_inflation():
    """Disabling the brake must not make every route impassable."""
    assert clearance_cells_for(0.0, 0.05) == 0
