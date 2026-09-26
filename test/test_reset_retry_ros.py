"""A slow reset service must not end a training campaign.

Sensor freshness failures are safety failures: moving a robot on stale scans
is unsafe, so those still abort.  A world-reset RPC is different - it is a
request to the simulator between episodes, with the robot already stopped.
Treating one slow reply as fatal cost this project 18 episodes of training,
so reset requests are retried a bounded number of times first.
"""

import json
import tempfile

import pytest

rclpy = pytest.importorskip('rclpy')
from rclpy.node import Node
from rclpy.task import Future
from std_srvs.srv import SetBool

from turtleboot3_autonomous_nav import dqn_trainer


class ServiceTransport:
    def __init__(self):
        self.ready = True
        self.calls = []

    def service_is_ready(self):
        return self.ready

    def wait_for_service(self, **kwargs):
        return self.ready

    def call_async(self, request):
        future = Future()
        self.calls.append((request, future))
        return future

    def reply(self, response):
        self.calls[-1][1].set_result(response)


@pytest.fixture
def clients(monkeypatch):
    created = {}

    def create_client(node, service_type, name, *args, **kwargs):
        return created.setdefault(name, ServiceTransport())

    monkeypatch.setattr(Node, 'create_client', create_client)
    return created


def _run(monkeypatch, exercise, model_directory=None):
    """Run one trainer that never touches the package's own models directory.

    The trainer resumes from `model_directory` when it starts, so a default
    path would make these tests read each other's snapshots and pass or fail
    depending on the order they ran in.
    """
    monkeypatch.setattr(rclpy, 'spin', exercise)
    dqn_trainer.main([
        '--ros-args',
        '-p', f'model_directory:={model_directory or tempfile.mkdtemp()}',
        '-p', 'resume:=false',
    ])


def _expire(trainer):
    """Force the pending deadline to have passed and poll once."""
    trainer._deadline_ns = 0
    trainer._poll_reset()


def _enable_response():
    return SetBool.Response(
        success=True, message=json.dumps({'cutoff_ns': 1, 'epoch': 1})
    )


def test_a_timed_out_reset_request_is_retried_before_aborting(monkeypatch, clients):
    """One slow reply must not discard the whole campaign."""
    def exercise(trainer):
        controller = clients['/safe_motion_controller/enable']
        trainer._poll_reset()
        assert len(controller.calls) == 1

        _expire(trainer)
        trainer._poll_reset()

        assert len(controller.calls) == 2
        assert not trainer._failed

    _run(monkeypatch, exercise)


def test_training_aborts_once_the_retries_are_exhausted(monkeypatch, clients):
    """A service that never answers is still a failure, just not immediately."""
    def exercise(trainer):
        trainer._poll_reset()
        for _ in range(trainer._config.reset_retry_limit + 1):
            _expire(trainer)
            trainer._poll_reset()
        assert trainer._failed

    with pytest.raises(RuntimeError):
        _run(monkeypatch, exercise)


def test_a_successful_reply_clears_the_retry_budget(monkeypatch, clients):
    """Retries must not accumulate across the steps of a reset sequence."""
    def exercise(trainer):
        controller = clients['/safe_motion_controller/enable']
        trainer._poll_reset()
        _expire(trainer)
        trainer._poll_reset()
        assert trainer._service_attempts == 1

        controller.reply(_enable_response())
        rclpy.spin_once(trainer, timeout_sec=0)

        assert trainer._service_attempts == 0

    _run(monkeypatch, exercise)


def test_a_stale_sensor_failure_still_aborts_without_retrying(monkeypatch, clients):
    """Waiting for fresh sensor data is a safety condition, not an RPC."""
    def exercise(trainer):
        trainer._pending_service = None
        trainer._active_service = None
        trainer._arm_deadline('post-reset odometry and scan')

        _expire(trainer)

        assert trainer._failed

    with pytest.raises(RuntimeError):
        _run(monkeypatch, exercise)
