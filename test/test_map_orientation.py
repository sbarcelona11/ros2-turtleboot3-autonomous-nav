"""A ray must land where the robot is pointing.

A heading applied with the wrong sign or convention rotates the whole map,
and the arena is square, so its extent looks identical either way: checking
the bounding box cannot catch it.  These place a single ray from a known pose
and assert the cell it marks, which a rotation would move.
"""

import math

import numpy as np
import pytest

from turtleboot3_autonomous_nav.grid_mapping import OccupancyGridModel


def _model():
    return OccupancyGridModel(200, 200, 0.05, (-5.0, -5.0), occupied_threshold=1)


def _single_ray(model, yaw, distance=2.0):
    """Trace one ray straight ahead of a robot at the origin facing `yaw`."""
    model.update_scan((0.0, 0.0, yaw), np.asarray([distance]), 0.0, 0.01, 8.0)


@pytest.mark.parametrize('yaw,expected', [
    (0.0, (2.0, 0.0)),
    (math.pi / 2.0, (0.0, 2.0)),
    (math.pi, (-2.0, 0.0)),
    (-math.pi / 2.0, (0.0, -2.0)),
])
def test_a_ray_marks_the_cell_the_robot_faces(yaw, expected):
    """Each quarter turn must move the hit a quarter of the way round."""
    model = _model()
    _single_ray(model, yaw)
    assert model.value_at(*expected) == OccupancyGridModel.OCCUPIED


def test_a_diagonal_heading_lands_on_the_diagonal():
    """A sign error shows up as a mirrored diagonal, not a missing hit."""
    model = _model()
    _single_ray(model, math.pi / 4.0, distance=math.sqrt(2.0))
    assert model.value_at(1.0, 1.0) == OccupancyGridModel.OCCUPIED


def test_the_opposite_heading_does_not_mark_the_same_cell():
    """Without this a 180 degree error would still satisfy the checks above."""
    model = _model()
    _single_ray(model, math.pi)
    assert model.value_at(2.0, 0.0) != OccupancyGridModel.OCCUPIED


def test_free_space_is_traced_between_the_robot_and_the_hit():
    model = _model()
    _single_ray(model, 0.0)
    assert model.value_at(1.0, 0.0) == OccupancyGridModel.FREE
