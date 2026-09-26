import numpy as np

from turtleboot3_autonomous_nav.grid_mapping import OccupancyGridModel
from turtleboot3_autonomous_nav.reset_provenance import is_post_reset_timestamp


def test_hit_marks_free_cells_then_occupied_endpoint():
    grid = OccupancyGridModel(20, 20, 1.0, (-10.0, -10.0))

    grid.update_scan((0.0, 0.0, 0.0), np.array([3.0]), 0.0, 1.0, 8.0)

    assert grid.value_at(1.0, 0.0) == 0
    assert grid.value_at(3.0, 0.0) == 100


def test_max_range_measurement_marks_free_endpoint_not_occupied():
    grid = OccupancyGridModel(20, 20, 1.0, (-10.0, -10.0))

    grid.update_scan((0.0, 0.0, 0.0), np.array([8.0]), 0.0, 1.0, 8.0)

    assert grid.value_at(8.0, 0.0) == 0


def test_invalid_ranges_do_not_change_the_map():
    grid = OccupancyGridModel(20, 20, 1.0, (-10.0, -10.0))

    changed = grid.update_scan(
        (0.0, 0.0, 0.0), np.array([np.nan, np.inf, -1.0, 0.0]), 0.0, 1.0, 8.0
    )

    assert changed == 0
    assert grid.coverage_fraction() == 0.0
    assert grid.value_at(1.0, 0.0) == -1


def test_invalid_pose_or_scan_angles_are_ignored():
    grid = OccupancyGridModel(20, 20, 1.0, (-10.0, -10.0))

    changed = grid.update_scan((np.nan, 0.0, 0.0), np.array([3.0]), 0.0, 1.0, 8.0)

    assert changed == 0
    assert grid.coverage_fraction() == 0.0


def test_ray_that_leaves_map_marks_only_in_bounds_cells_free():
    grid = OccupancyGridModel(4, 4, 1.0, (-2.0, -2.0))

    grid.update_scan((0.0, 0.0, 0.0), np.array([5.0]), 0.0, 1.0, 8.0)

    assert grid.value_at(1.0, 0.0) == 0
    assert grid.value_at(5.0, 0.0) == -1
    assert grid.coverage_fraction() == 1.0 / 16.0


def test_long_ray_is_clipped_before_bresenham_traversal(monkeypatch):
    grid = OccupancyGridModel(20, 20, 1.0, (-10.0, -10.0))
    traversals = []
    original_bresenham = grid._bresenham_cells

    def record_bresenham(start_x, start_y, end_x, end_y):
        traversals.append((start_x, start_y, end_x, end_y))
        return original_bresenham(start_x, start_y, end_x, end_y)

    monkeypatch.setattr(grid, "_bresenham_cells", record_bresenham)

    grid.update_scan((0.0, 0.0, 0.0), np.array([1_000.0]), 0.0, 1.0, 1_000.0)

    assert traversals == [(10, 10, 19, 10)]
    assert grid.coverage_fraction() == 9.0 / 400.0


def test_coverage_counts_new_known_cells_once_and_never_decreases():
    grid = OccupancyGridModel(20, 20, 1.0, (-10.0, -10.0))

    first_change = grid.update_scan((0.0, 0.0, 0.0), np.array([3.0]), 0.0, 1.0, 8.0)
    first_coverage = grid.coverage_fraction()
    repeated_change = grid.update_scan((0.0, 0.0, 0.0), np.array([3.0]), 0.0, 1.0, 8.0)
    second_coverage = grid.coverage_fraction()
    third_change = grid.update_scan(
        (0.0, 0.0, 0.0), np.array([3.0]), np.pi / 2.0, 1.0, 8.0
    )

    assert first_change == 3
    assert repeated_change == 0
    assert third_change == 3
    assert first_coverage == second_coverage
    assert grid.coverage_fraction() > second_coverage


def test_occupied_endpoint_requires_configured_evidence_threshold():
    grid = OccupancyGridModel(20, 20, 1.0, (-10.0, -10.0), occupied_threshold=2)

    grid.update_scan((0.0, 0.0, 0.0), np.array([3.0]), 0.0, 1.0, 8.0)
    assert grid.value_at(3.0, 0.0) == -1

    grid.update_scan((0.0, 0.0, 0.0), np.array([3.0]), 0.0, 1.0, 8.0)
    assert grid.value_at(3.0, 0.0) == 100


def test_reset_clears_all_episode_coverage_and_cell_evidence():
    """A new Gazebo episode must not inherit mapper coverage from the prior one."""
    grid = OccupancyGridModel(20, 20, 1.0, (-10.0, -10.0), occupied_threshold=2)
    grid.update_scan((0.0, 0.0, 0.0), np.array([3.0]), 0.0, 1.0, 8.0)
    grid.update_scan((0.0, 0.0, 0.0), np.array([3.0]), 0.0, 1.0, 8.0)

    grid.reset()

    assert grid.coverage_fraction() == 0.0
    assert grid.value_at(1.0, 0.0) == -1
    grid.update_scan((0.0, 0.0, 0.0), np.array([3.0]), 0.0, 1.0, 8.0)
    assert grid.value_at(3.0, 0.0) == -1


def test_reset_cutoff_rejects_a_delayed_pre_reset_sensor_message():
    """A queued scan from before reset must not repopulate the cleared grid."""
    assert is_post_reset_timestamp(stamp_ns=999, cutoff_ns=1_000) is False
    assert is_post_reset_timestamp(stamp_ns=1_000, cutoff_ns=1_000) is False
    assert is_post_reset_timestamp(stamp_ns=1_001, cutoff_ns=1_000) is True
    assert is_post_reset_timestamp(stamp_ns=0, cutoff_ns=1_000) is False
