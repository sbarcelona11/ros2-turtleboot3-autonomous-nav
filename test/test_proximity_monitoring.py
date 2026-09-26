"""Monitoring a cell requires having been near it, not merely having seen it.

A 3.5 m LiDAR resolves most of a 4.85 m room from a single sweep, so counting
any cell a ray touched makes coverage saturate without the robot moving: a
random policy scores almost as well as a good one and the learning signal
disappears.  Requiring close observation turns the mission into a coverage
path problem, where covering the scenario means driving through it.
"""

import numpy as np
import pytest

from turtleboot3_autonomous_nav.grid_mapping import OccupancyGridModel


def _model(inspection_radius=1.5):
    return OccupancyGridModel(
        100, 100, 0.05, (-2.5, -2.5), inspection_radius=inspection_radius
    )


def _ray_ahead(model, distance, range_max=3.5):
    """Trace a single ray along +x from the origin."""
    model.update_scan((0.0, 0.0, 0.0), np.asarray([distance]), 0.0, 0.01, range_max)


def test_a_cell_seen_from_beyond_the_radius_is_known_but_not_monitored():
    """Line of sight alone must not count as having monitored the area."""
    model = _model(inspection_radius=1.0)
    _ray_ahead(model, 2.5)
    assert model.value_at(2.0, 0.0) != OccupancyGridModel.UNKNOWN
    assert not model.is_monitored(2.0, 0.0)


def test_a_cell_seen_from_within_the_radius_is_monitored():
    """Close observation is what the mission counts as monitoring."""
    model = _model(inspection_radius=1.0)
    _ray_ahead(model, 2.5)
    assert model.is_monitored(0.5, 0.0)


def test_coverage_counts_only_monitored_cells():
    """A long ray reveals many cells but monitors only the nearby ones."""
    far = _model(inspection_radius=1.0)
    _ray_ahead(far, 2.5)
    near = _model(inspection_radius=2.5)
    _ray_ahead(near, 2.5)
    assert far.coverage_fraction() < near.coverage_fraction()


def test_reachable_coverage_also_requires_close_observation():
    """The mission target must reflect monitoring, not visibility."""
    model = _model(inspection_radius=1.0)
    _ray_ahead(model, 2.5)
    assert model.reachable_coverage_fraction() < 1.0


def test_without_a_radius_seeing_a_cell_monitors_it():
    """The default keeps the plain visibility behaviour for other callers."""
    model = _model(inspection_radius=None)
    _ray_ahead(model, 2.5)
    assert model.is_monitored(2.0, 0.0)
    assert model.coverage_fraction() > 0.0


def test_reset_clears_monitoring():
    """A new episode must start with nothing monitored."""
    model = _model()
    _ray_ahead(model, 1.0)
    model.reset()
    assert model.coverage_fraction() == 0.0
    assert not model.is_monitored(0.5, 0.0)


def test_monitoring_accumulates_as_the_robot_moves():
    """Driving on must add the newly approached area to the covered set."""
    model = _model(inspection_radius=0.5)
    _ray_ahead(model, 2.5)
    before = model.coverage_fraction()
    model.update_scan((1.5, 0.0, 0.0), np.asarray([1.0]), 0.0, 0.01, 3.5)
    assert model.coverage_fraction() > before


@pytest.mark.parametrize('radius', [0.0, -1.0])
def test_a_non_positive_radius_is_rejected(radius):
    with pytest.raises(ValueError):
        OccupancyGridModel(10, 10, 0.05, (0.0, 0.0), inspection_radius=radius)
