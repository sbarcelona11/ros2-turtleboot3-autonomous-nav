"""DQN observation provenance and map-source contract tests."""

from types import SimpleNamespace

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from nav_msgs.msg import OccupancyGrid

from turtleboot3_autonomous_nav.observation import (
    OBSERVATION_CONTRACT,
    build_observation,
    has_observation_contract,
    observation_label,
    require_observation_contract,
)
from turtleboot3_autonomous_nav.observation_builder import ObservationBuilder


def test_observation_label_carries_the_monitoring_map_contract():
    assert OBSERVATION_CONTRACT == 'monitoring-map-v1'
    assert observation_label(7) == 'monitoring-map-v1:episode:7'
    assert has_observation_contract(observation_label(7))
    assert not has_observation_contract('episode:7')


def test_checkpoint_config_must_declare_the_current_observation_contract():
    require_observation_contract({'observation_contract': OBSERVATION_CONTRACT})

    with pytest.raises(ValueError, match='checkpoint observation contract is incompatible'):
        require_observation_contract({})


def _grid(data: np.ndarray) -> OccupancyGrid:
    message = OccupancyGrid()
    message.info.width = data.shape[1]
    message.info.height = data.shape[0]
    message.info.resolution = 0.05
    message.info.origin.position.x = -2.5
    message.info.origin.position.y = -2.5
    message.data = data.ravel().tolist()
    return message


def test_builder_publishes_monitoring_map_gains_with_the_contract_label():
    """Raw LiDAR visibility must not erase monitoring-grid coverage targets."""
    rclpy.init()
    node = ObservationBuilder()
    captured = []
    robot_cell = (50, 50)
    scan = np.full(360, 1.0)
    raw_coverage_grid = np.zeros((100, 100), dtype=np.int8)
    monitoring_grid = np.full((100, 100), -1, dtype=np.int8)
    try:
        node._publisher.publish = captured.append
        node._reset_epoch = 3
        node._latest_scan = scan
        node._position = (0.0, 0.0)
        node._on_map(_grid(monitoring_grid), SimpleNamespace(source_timestamp=1))
        node._publish_if_ready()

        assert len(captured) == 1
        assert captured[0].layout.dim[0].label == observation_label(3)
        assert not np.array_equal(
            captured[0].data[12:20],
            build_observation(scan, raw_coverage_grid, robot_cell, 0.0, 0.0, 0.0)[
                12:20
            ],
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()
