"""Exercise reset callbacks with real ROS messages and controlled delivery order."""

import json
import tempfile
import time
from types import SimpleNamespace

import pytest

rclpy = pytest.importorskip("rclpy")
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, Float32MultiArray, MultiArrayDimension
from std_srvs.srv import SetBool, Trigger

from turtleboot3_autonomous_nav.coverage_mapper import CoverageMapper
from turtleboot3_autonomous_nav import dqn_trainer
from turtleboot3_autonomous_nav.observation import OBSERVATION_SIZE, observation_label


def delivery(timestamp):
    return {"source_timestamp": timestamp}


def sensor_messages(simulation_seconds):
    odom = Odometry()
    odom.header.stamp.sec = simulation_seconds
    odom.pose.pose.orientation.w = 1.0
    scan = LaserScan()
    scan.header.stamp.sec = simulation_seconds
    scan.ranges = [3.0]
    scan.range_max = 8.0
    scan.angle_increment = 1.0
    return odom, scan


def test_mapper_delayed_old_messages_cannot_restore_coverage_after_reset():
    rclpy.init()
    mapper = CoverageMapper()
    try:
        old_odom, old_scan = sensor_messages(90)
        old_publication = time.time_ns()
        mapper._on_odometry(old_odom, delivery(old_publication))
        mapper._on_scan(old_scan, delivery(old_publication))
        assert mapper._grid.coverage_fraction() > 0

        ack = mapper._on_reset(Trigger.Request(), Trigger.Response())
        assert ack.success
        cutoff = json.loads(ack.message)["cutoff_ns"]
        assert cutoff >= old_publication
        mapper._on_odometry(old_odom, delivery(old_publication))
        mapper._on_scan(old_scan, delivery(old_publication))
        mapper._on_odometry(old_odom, delivery(cutoff))
        assert mapper._latest_pose is None
        assert mapper._grid.coverage_fraction() == 0

        # reset.all rewinds header time. Fresh DDS publication still qualifies.
        fresh_odom, fresh_scan = sensor_messages(0)
        mapper._on_scan(fresh_scan, delivery(cutoff + 1))
        assert mapper._grid.coverage_fraction() == 0
        mapper._on_odometry(fresh_odom, delivery(cutoff + 2))
        mapper._on_scan(old_scan, delivery(old_publication))
        assert mapper._grid.coverage_fraction() == 0
        mapper._on_scan(fresh_scan, delivery(cutoff + 3))
        assert mapper._grid.coverage_fraction() > 0
        current_pose = mapper._latest_pose
        old_odom.pose.pose.position.x = 9.0
        mapper._on_odometry(old_odom, delivery(old_publication))
        assert mapper._latest_pose == current_pose
    finally:
        mapper.destroy_node()
        rclpy.shutdown()


def test_trainer_delayed_messages_cannot_start_or_contaminate_episode(monkeypatch):
    # Replace only external service transport; exercise the actual ROS node and
    # reset response callback, including its success/provenance handling.
    pending = Future()
    client = SimpleNamespace(
        wait_for_service=lambda **_: True, call_async=lambda _: pending
    )
    monkeypatch.setattr(Node, "create_client", lambda *_, **__: client)

    def exercise(trainer):
        trainer._reset_gate.world_reset_succeeded()
        ack = Trigger.Response(
            success=True, message=json.dumps({"cutoff_ns": 1_000, "epoch": 1})
        )
        trainer._on_mapper_reset(ack)
        trainer._on_builder_reset(ack)
        trainer._on_controller_enabled(SetBool.Response(success=True, message=ack.message))
        old_odom, old_scan = sensor_messages(90)
        for timestamp in (999, 1_000, 0):
            trainer._on_odometry(old_odom, delivery(timestamp))
            trainer._on_scan(old_scan, delivery(timestamp))
            assert not trainer._running_episode
        fresh_odom, fresh_scan = sensor_messages(0)
        trainer._on_odometry(fresh_odom, delivery(1_001))
        assert not trainer._running_episode
        trainer._on_scan(fresh_scan, delivery(1_002))
        assert trainer._running_episode
        observation = Float32MultiArray(data=[0.0] * OBSERVATION_SIZE)
        observation.layout.dim = [MultiArrayDimension(
            label=observation_label(1), size=OBSERVATION_SIZE, stride=OBSERVATION_SIZE)]
        trainer._on_observation(observation, delivery(999))
        trainer._on_coverage(Float32(data=0.9), delivery(999))
        trainer._on_intervention(Bool(data=True), delivery(999))
        trainer._on_recovery(Bool(data=True), delivery(999))
        assert trainer._previous_state is None
        assert trainer._coverage == 0
        assert not trainer._intervention_seen
        assert not trainer._recovery_seen
        trainer._on_observation(observation, delivery(1_003))
        assert trainer._previous_state is not None
        trainer._on_coverage(Float32(data=0.01), delivery(1_004))
        assert trainer._coverage == pytest.approx(0.01)
        trainer._on_observation(observation, delivery(999))
        assert len(trainer._replay) == 0

    monkeypatch.setattr(rclpy, "spin", exercise)
    dqn_trainer.main(['--ros-args', '-p', f'model_directory:={tempfile.mkdtemp()}',
                     '-p', 'resume:=false'])


def test_dds_publication_time_survives_queued_delivery_and_sim_clock_rewind():
    """The actual RMW metadata must retain publication time, not receipt time."""
    rclpy.init()
    mapper = CoverageMapper()
    publisher = Node("reset_barrier_sensor_fixture")
    odom_publisher = publisher.create_publisher(Odometry, "/odom", 10)
    scan_publisher = publisher.create_publisher(LaserScan, "/scan", 10)
    seen = []
    for subscription in mapper.subscriptions:
        callback = subscription.callback

        def capture(message, info, callback=callback):
            seen.append(info["source_timestamp"])
            callback(message, info)

        subscription.callback = capture

    def spin_until(condition):
        deadline = time.monotonic() + 5.0
        while not condition() and time.monotonic() < deadline:
            rclpy.spin_once(mapper, timeout_sec=0.01)
        assert condition(), "DDS delivery did not complete before the deadline"

    try:
        spin_until(
            lambda: odom_publisher.get_subscription_count() > 0
            and scan_publisher.get_subscription_count() > 0
        )
        old_odom, old_scan = sensor_messages(90)
        before_publish = time.time_ns()
        odom_publisher.publish(old_odom)
        scan_publisher.publish(old_scan)
        # Publish before reset, but do not execute either subscription callback.
        ack = mapper._on_reset(Trigger.Request(), Trigger.Response())
        cutoff = json.loads(ack.message)["cutoff_ns"]
        spin_until(lambda: len(seen) >= 2)
        assert all(before_publish <= timestamp <= cutoff for timestamp in seen)
        assert mapper._latest_pose is None
        assert mapper._grid.coverage_fraction() == 0

        fresh_odom, fresh_scan = sensor_messages(0)
        odom_publisher.publish(fresh_odom)
        spin_until(lambda: mapper._latest_pose is not None)
        scan_publisher.publish(fresh_scan)
        spin_until(lambda: mapper._grid.coverage_fraction() > 0)
        assert all(timestamp > cutoff for timestamp in seen[2:])
    finally:
        publisher.destroy_node()
        mapper.destroy_node()
        rclpy.shutdown()
