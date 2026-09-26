"""The trainer must actually write and read its progress.

The helpers are covered by `test_training_resume`; what matters here is the
wiring, because a snapshot that is never written protects nothing.
"""

import math
import tempfile

import pytest

rclpy = pytest.importorskip('rclpy')
from rclpy.node import Node
from rclpy.task import Future

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


def test_finishing_an_episode_writes_the_campaign_snapshot(
    monkeypatch, clients, tmp_path
):
    """Without this the next Gazebo crash costs the whole campaign again."""
    path = tmp_path / 'training_state.pt'

    def exercise(trainer):
        trainer._state_path = path
        trainer._training_episodes = 7
        trainer._global_steps = 4_200
        trainer._finish_episode('time_limit')

        assert path.exists()
        _, progress = dqn_trainer.load_training_state(
            path,
            trainer._config.observation_size,
            trainer._config.action_count,
        )
        assert progress['training_episodes'] == 8
        assert progress['global_steps'] == 4_200

    _run(monkeypatch, exercise)


def test_a_stored_campaign_is_picked_up_on_resume(monkeypatch, clients, tmp_path):
    """The counters drive epsilon, so restoring them continues the schedule."""
    path = tmp_path / 'training_state.pt'

    def exercise(trainer):
        trainer._state_path = path
        trainer._training_episodes = 30
        trainer._global_steps = 18_000
        trainer._best_mean_coverage = 0.61
        trainer._save_campaign()

        trainer._training_episodes = 0
        trainer._global_steps = 0
        trainer._best_mean_coverage = -math.inf
        trainer._resume_campaign()

        assert trainer._training_episodes == 30
        assert trainer._global_steps == 18_000
        assert trainer._best_mean_coverage == pytest.approx(0.61)

    _run(monkeypatch, exercise)


def test_a_campaign_without_a_snapshot_starts_clean(monkeypatch, clients, tmp_path):
    """A first run must not fail because there is nothing to resume."""
    def exercise(trainer):
        trainer._state_path = tmp_path / 'absent.pt'
        trainer._training_episodes = 0
        trainer._resume_campaign()
        assert trainer._training_episodes == 0

    _run(monkeypatch, exercise)
