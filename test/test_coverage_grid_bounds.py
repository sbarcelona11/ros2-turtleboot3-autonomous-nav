"""The coverage grid must stop at the wall, not past it.

Coverage is measured against the region the robot can still reach, and that
region is grown by flooding through every cell that is not occupied.  A grid
that extends beyond the arena therefore only excludes the outside while the
observed wall face is a solid line of occupied cells.  It never is: a 360-beam
laser samples a wall more sparsely the further away it is, and a cell needs
two hits to count as occupied, so the face comes out perforated and the flood
escapes through the holes.

Measured on a finished run: of the 680 cells still counted as pending, 456
were outside the arena entirely - two thirds of the apparent shortfall was
wall body and open ground beyond it, which no laser can ever reach.  Coverage
read 91.2 % with nothing left inside worth that much.

Sizing the grid to the arena removes the leak at its source, and needs no
sealing logic that could hide a real unexplored pocket.
"""

from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

ARENA_HALF_SIZE = 2.425
"""Half the inner span of the stage4 room, in metres: walls at +-2.425."""


def _grid_parameters():
    config = yaml.safe_load((PACKAGE_ROOT / 'config' / 'exploration.yaml').read_text())
    return config['coverage_mapper']['ros__parameters']


def test_the_grid_starts_exactly_at_the_wall():
    """Any margin outside the arena leaks into the reachable region."""
    parameters = _grid_parameters()
    assert parameters['map_origin_x'] == -ARENA_HALF_SIZE
    assert parameters['map_origin_y'] == -ARENA_HALF_SIZE


def test_the_grid_ends_exactly_at_the_opposite_wall():
    """Too small a grid would drop real floor from the measurement instead."""
    parameters = _grid_parameters()
    resolution = parameters['map_resolution']
    for size, origin in ((parameters['map_width'], parameters['map_origin_x']),
                         (parameters['map_height'], parameters['map_origin_y'])):
        assert abs(origin + size * resolution - ARENA_HALF_SIZE) < resolution / 2


def test_no_cell_centre_lies_outside_the_arena():
    """A cell whose centre is past the wall is unobservable by construction."""
    parameters = _grid_parameters()
    resolution = parameters['map_resolution']
    for size, origin in ((parameters['map_width'], parameters['map_origin_x']),
                         (parameters['map_height'], parameters['map_origin_y'])):
        assert origin + 0.5 * resolution > -ARENA_HALF_SIZE
        assert origin + (size - 0.5) * resolution < ARENA_HALF_SIZE


def test_training_counts_the_same_cells_as_the_mapper():
    """The reward scales with cells discovered; a stale count rescales it."""
    parameters = _grid_parameters()
    training = yaml.safe_load((PACKAGE_ROOT / 'config' / 'training.yaml').read_text())
    counts = [section['ros__parameters']['map_cell_count']
              for section in training.values()
              if 'map_cell_count' in section.get('ros__parameters', {})]
    assert counts, 'training no longer declares how many cells the arena has'
    for count in counts:
        assert count == parameters['map_width'] * parameters['map_height']
