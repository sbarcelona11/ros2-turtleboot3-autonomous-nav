"""Exercise policy adapters with real ROS messages and controlled service transport."""

import json
import tempfile
from dataclasses import replace
import time
from types import SimpleNamespace

import pytest

rclpy = pytest.importorskip('rclpy')
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.task import Future
from ros_gz_interfaces.srv import ControlWorld
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, Float32MultiArray, MultiArrayDimension
from std_srvs.srv import SetBool, Trigger

from turtleboot3_autonomous_nav import dqn_explorer, dqn_trainer, frontier_explorer
from turtleboot3_autonomous_nav.dqn import DQNPolicy, save_checkpoint
from turtleboot3_autonomous_nav.control import POLICY_ACTION_COUNT
from turtleboot3_autonomous_nav.observation import (
    OBSERVATION_CONTRACT,
    OBSERVATION_SIZE,
    observation_label,
)


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
def transport(monkeypatch):
    clients = {}

    def create_client(node, service_type, name, *args, **kwargs):
        return clients.setdefault(name, ServiceTransport())

    monkeypatch.setattr(Node, 'create_client', create_client)
    return clients


def provenance(response_type, cutoff=1_000, epoch=1):
    return response_type(success=True, message=json.dumps({'cutoff_ns': cutoff, 'epoch': epoch}))


def deliver(stamp=2_000):
    return {'source_timestamp': stamp}


def observation(epoch=1):
    message = Float32MultiArray(data=[0.0] * OBSERVATION_SIZE)
    message.layout.dim = [MultiArrayDimension(
        label=observation_label(epoch), size=OBSERVATION_SIZE, stride=OBSERVATION_SIZE)]
    return message


def complete_trainer_reset(trainer, clients):
    trainer._poll_reset()
    control = clients['/safe_motion_controller/enable']
    assert control.calls[-1][0].data is False
    control.reply(provenance(SetBool.Response))
    trainer._poll_reset()
    clients['/world/dqn/control'].reply(ControlWorld.Response(success=True))
    trainer._poll_reset()
    clients['/coverage_mapper/reset'].reply(provenance(Trigger.Response))
    trainer._poll_reset()
    clients['/observation_builder/reset'].reply(provenance(Trigger.Response, 1_100, 7))
    trainer._poll_reset()
    assert control.calls[-1][0].data is True
    control.reply(provenance(SetBool.Response, 1_200, 2))


def test_trainer_stops_before_reset_and_requires_current_observation_epoch(monkeypatch, transport):
    def exercise(trainer):
        complete_trainer_reset(trainer, transport)
        trainer._on_odometry(Odometry(), deliver())
        trainer._on_scan(LaserScan(), deliver())
        assert trainer._running_episode
        trainer._on_observation(observation(epoch=6), deliver())
        assert trainer._previous_state is None
        trainer._on_observation(observation(epoch=7), deliver())
        assert trainer._previous_state is not None

    monkeypatch.setattr(rclpy, 'spin', exercise)
    dqn_trainer.main(['--ros-args', '-p', f'model_directory:={tempfile.mkdtemp()}',
                     '-p', 'resume:=false'])


@pytest.mark.parametrize('phase', ['controller_service', 'disable_reply', 'world_reply', 'mapper_reply',
                                    'builder_reply', 'enable_reply', 'sensors', 'observation'])
def test_trainer_reset_phases_have_wall_clock_deadlines(monkeypatch, transport, phase):
    wall_time = [time.monotonic_ns()]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: wall_time[0])

    def exercise(trainer):
        control = transport['/safe_motion_controller/enable']
        if phase == 'controller_service':
            control.ready = False
        trainer._poll_reset()
        if phase not in ('controller_service', 'disable_reply'):
            control.reply(provenance(SetBool.Response))
            trainer._poll_reset()
        if phase not in ('controller_service', 'disable_reply', 'world_reply'):
            transport['/world/dqn/control'].reply(ControlWorld.Response(success=True))
            trainer._poll_reset()
        if phase in ('builder_reply', 'enable_reply', 'sensors', 'observation'):
            transport['/coverage_mapper/reset'].reply(provenance(Trigger.Response))
            trainer._poll_reset()
        if phase in ('enable_reply', 'sensors', 'observation'):
            transport['/observation_builder/reset'].reply(provenance(Trigger.Response, 1_100, 7))
            trainer._poll_reset()
        if phase in ('sensors', 'observation'):
            control.reply(provenance(SetBool.Response, 1_200, 2))
        if phase == 'observation':
            trainer._on_odometry(Odometry(), deliver())
            trainer._on_scan(LaserScan(), deliver())
        # A reset RPC is retried before it is fatal, so exhaust the budget;
        # the sensor phases have no retries and fail on the first deadline.
        for _ in range(trainer._config.reset_retry_limit + 1):
            wall_time[0] += 11_000_000_000
            trainer._poll_reset()
        assert trainer._failed
        assert not trainer._running_episode
        assert trainer._previous_state is None
        # Failure must request a stop and continue suppressing old callbacks.
        control.ready = True
        trainer._poll_reset()
        assert control.calls[-1][0].data is False
        trainer._on_observation(observation(epoch=7), deliver())
        assert trainer._previous_state is None

    monkeypatch.setattr(rclpy, 'spin', exercise)
    with pytest.raises(RuntimeError, match='Training aborted'):
        dqn_trainer.main(['--ros-args', '-p', f'model_directory:={tempfile.mkdtemp()}',
                     '-p', 'resume:=false'])


def test_trainer_episode_limit_requests_acknowledged_stop(monkeypatch, transport):
    def exercise(trainer):
        complete_trainer_reset(trainer, transport)
        trainer._config = replace(trainer._config, max_steps=1, max_episodes=1)
        trainer._on_odometry(Odometry(), deliver())
        trainer._on_scan(LaserScan(), deliver())
        trainer._on_observation(observation(epoch=7), deliver())
        trainer._on_observation(observation(epoch=7), deliver())
        assert not trainer._running_episode
        trainer._poll_reset()
        control = transport['/safe_motion_controller/enable']
        assert control.calls[-1][0].data is False
        assert len(transport['/world/dqn/control'].calls) == 1

    monkeypatch.setattr(rclpy, 'spin', exercise)
    dqn_trainer.main(['--ros-args', '-p', f'model_directory:={tempfile.mkdtemp()}',
                     '-p', 'resume:=false'])


def test_evaluation_diagnostics_publish_persist_and_resume(monkeypatch, transport, tmp_path):
    published = []

    def exercise(trainer):
        assert trainer._diagnostics_publisher.topic_name == '/training_episode_diagnostics'
        trainer._diagnostics_publisher = SimpleNamespace(publish=published.append)
        metrics = []
        trainer._metrics_publisher = SimpleNamespace(publish=metrics.append)
        trainer._config = replace(trainer._config, evaluation_episodes=2)
        trainer._is_evaluation = True
        complete_trainer_reset(trainer, transport)
        trainer._on_odometry(Odometry(), deliver())
        trainer._on_scan(LaserScan(), deliver())
        actions = iter([0, 3])  # Forward, then turn in place.
        monkeypatch.setattr(trainer._policy, 'select_training_action', lambda *_: next(actions))
        trainer._on_observation(observation(epoch=7), deliver())
        trainer._on_intervention(Bool(data=True), deliver(1))  # Stale event.
        trainer._on_intervention(Bool(data=True), deliver())
        trainer._on_intervention(Bool(data=True), deliver())  # Repeated active state.
        trainer._on_observation(observation(epoch=7), deliver())
        trainer._on_coverage(Float32(data=0.6), deliver())
        trainer._on_reachable_coverage(Float32(data=0.8), deliver())
        trainer._on_observation(observation(epoch=7), deliver())
        record = json.loads(published[-1].data)
        assert record == pytest.approx({
            'episode': 1, 'is_evaluation': True,
            'reason': 'target_coverage', 'reachable_coverage': 0.8,
            'grid_coverage': 0.6, 'interventions': 1,
            'policy_actions': 2, 'turn_actions': 1, 'turn_ratio': 0.5,
        })
        assert list(metrics[-1].data[6:]) == pytest.approx([0.8, 1.0, 0.5])
        assert not (tmp_path / 'best.metrics.json').exists()
        _, saved = dqn_trainer.load_training_state(tmp_path / 'training_state.pt', 90, 5)
        assert saved['evaluation_diagnostics'] == [record]

        trainer._evaluation_diagnostics.clear()
        trainer._evaluation_coverages.clear()
        trainer._resume_campaign()
        assert trainer._is_evaluation
        assert trainer._evaluation_diagnostics[0].as_dict() == record
        complete_trainer_reset(trainer, transport)
        trainer._on_odometry(Odometry(), deliver())
        trainer._on_scan(LaserScan(), deliver())
        trainer._publish_metrics()
        assert list(metrics[-1].data[7:]) == [0.0, 0.0]
        trainer._finish_episode('time_limit')

        stored = json.loads((tmp_path / 'best.metrics.json').read_text())['metrics']
        assert stored['episodes'] == [json.loads(message.data) for message in published]
        assert stored['episodes'][1] == {
            'episode': 2, 'is_evaluation': True,
            'reason': 'time_limit', 'reachable_coverage': 0.0,
            'grid_coverage': 0.0, 'interventions': 0,
            'policy_actions': 0, 'turn_actions': 0, 'turn_ratio': 0.0,
        }
        assert stored['mean_coverage'] == pytest.approx(0.4)
        assert stored['mean_grid_coverage'] == pytest.approx(0.3)
        assert stored['mean_interventions'] == 0.5
        assert stored['mean_turn_ratio'] == 0.25
        assert trainer._evaluation_diagnostics == []
        assert trainer._evaluation_coverages == []
        _, saved = dqn_trainer.load_training_state(tmp_path / 'training_state.pt', 90, 5)
        assert saved['evaluation_diagnostics'] == []

    monkeypatch.setattr(rclpy, 'spin', exercise)
    dqn_trainer.main(['--ros-args', '-p', f'model_directory:={tmp_path}',
                     '-p', 'resume:=false'])


def test_training_diagnostics_identify_episode_and_count_intervention_edges(monkeypatch, transport, tmp_path):
    def exercise(trainer):
        published = []
        trainer._diagnostics_publisher = SimpleNamespace(publish=published.append)
        complete_trainer_reset(trainer, transport)
        trainer._on_odometry(Odometry(), deliver())
        trainer._on_scan(LaserScan(), deliver())
        for active in (True, True, False, True):
            trainer._on_intervention(Bool(data=active), deliver())
        trainer._finish_episode('time_limit')
        record = json.loads(published[-1].data)
        assert record['episode'] == 1
        assert record['is_evaluation'] is False
        assert record['interventions'] == 2
        assert trainer._evaluation_diagnostics == []

    monkeypatch.setattr(rclpy, 'spin', exercise)
    dqn_trainer.main(['--ros-args', '-p', f'model_directory:={tmp_path}',
                     '-p', 'resume:=false'])


def test_resume_restarts_partial_evaluation_without_diagnostics(monkeypatch, transport, tmp_path):
    def exercise(trainer):
        dqn_trainer.save_training_state(
            trainer._state_path, trainer._policy,
            {'observation_contract': OBSERVATION_CONTRACT},
            {'best_mean_coverage': 0.3, 'evaluation_coverages': [0.6],
             'is_evaluation': True, 'training_episodes': 10},
        )
        trainer._resume_campaign()
        assert trainer._is_evaluation
        assert trainer._training_episodes == 10
        assert trainer._best_mean_coverage == 0.3
        assert trainer._evaluation_coverages == []
        assert trainer._evaluation_diagnostics == []

    monkeypatch.setattr(rclpy, 'spin', exercise)
    dqn_trainer.main(['--ros-args', '-p', f'model_directory:={tmp_path}',
                     '-p', 'resume:=false'])


def test_trainer_reset_failure_exits_unsuccessfully_after_stop_ack(monkeypatch, transport):
    def exercise(trainer):
        trainer._poll_reset()
        control = transport['/safe_motion_controller/enable']
        control.reply(provenance(SetBool.Response))
        trainer._poll_reset()
        transport['/world/dqn/control'].reply(ControlWorld.Response(success=False))
        trainer._poll_reset()
        assert control.calls[-1][0].data is False
        control.reply(provenance(SetBool.Response))
        assert not rclpy.ok()

    monkeypatch.setattr(rclpy, 'spin', exercise)
    with pytest.raises(RuntimeError, match='Training aborted'):
        dqn_trainer.main(['--ros-args', '-p', f'model_directory:={tempfile.mkdtemp()}',
                     '-p', 'resume:=false'])


@pytest.mark.parametrize('model_path', ['', '   ', '/missing/policy.pt'])
def test_explorer_rejects_missing_model_before_activation(monkeypatch, model_path):
    monkeypatch.setattr(rclpy, 'spin', lambda _: None)
    try:
        with pytest.raises((ValueError, FileNotFoundError)):
            dqn_explorer.main(['--ros-args', '-p', f'model_path:={json.dumps(model_path)}'])
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_inference_rejects_a_checkpoint_without_the_monitoring_contract(tmp_path):
    """Legacy DQN weights cannot be activated against monitoring-map inputs."""
    checkpoint = tmp_path / 'legacy.pt'
    save_checkpoint(checkpoint, DQNPolicy(90, 5), {}, {})

    with pytest.raises(ValueError, match='observation contract'):
        dqn_explorer.main(['--ros-args', '-p', f'model_path:={checkpoint}'])


def test_inference_starts_with_the_monitoring_observation_contract(monkeypatch, tmp_path):
    """A versioned checkpoint reaches ROS activation instead of being rejected."""
    checkpoint = tmp_path / 'current.pt'
    save_checkpoint(
        checkpoint,
        DQNPolicy(90, 5),
        {'observation_contract': OBSERVATION_CONTRACT},
        {},
    )
    spun = []
    monkeypatch.setattr(rclpy, 'spin', spun.append)

    dqn_explorer.main(['--ros-args', '-p', f'model_path:={checkpoint}'])

    assert len(spun) == 1


def test_inference_rejects_a_live_observation_without_the_monitoring_contract(
    monkeypatch, tmp_path
):
    """An old layout label must not cause an action after deployment starts."""
    checkpoint = tmp_path / 'current.pt'
    save_checkpoint(
        checkpoint,
        DQNPolicy(90, 5),
        {'observation_contract': OBSERVATION_CONTRACT},
        {},
    )

    def exercise(explorer):
        explorer._enabled = True
        actions = []
        explorer._publisher = SimpleNamespace(publish=actions.append)
        legacy_observation = observation()
        legacy_observation.layout.dim[0].label = 'episode:1'
        explorer._on_observation(legacy_observation)
        assert actions == []

    monkeypatch.setattr(rclpy, 'spin', exercise)
    dqn_explorer.main(['--ros-args', '-p', f'model_path:={checkpoint}'])


@pytest.mark.parametrize('reason', ['max_steps', 'target_coverage'])
def test_mission_termination_disables_controller_and_suppresses_later_actions(
    monkeypatch, transport, tmp_path, reason
):
    model = tmp_path / 'policy.pt'
    save_checkpoint(
        model,
        DQNPolicy(OBSERVATION_SIZE, POLICY_ACTION_COUNT, seed=1),
        {'observation_contract': OBSERVATION_CONTRACT},
        {},
    )

    def exercise(explorer):
        from turtleboot3_autonomous_nav.safe_motion_controller import SafeMotionController
        controller = SafeMotionController()
        decisions = []
        controller._publish_decision = decisions.append
        explorer._poll_control()
        control = transport['/safe_motion_controller/enable']
        control.reply(controller._on_enable(control.calls[-1][0], SetBool.Response()))
        actions = []
        def publish_action(message):
            actions.append(message)
            controller._on_action(message, deliver(time.time_ns()))
        explorer._publisher = SimpleNamespace(publish=publish_action)
        controller._on_scan(LaserScan(ranges=[2.0], angle_increment=1.0, range_max=3.5), deliver(time.time_ns()))
        controller._on_odometry(Odometry(), deliver(time.time_ns()))
        explorer._on_observation(observation())
        assert len(actions) == 1
        controller._on_control_timer()
        assert abs(decisions[-1].linear_x) + abs(decisions[-1].angular_z) > 0
        if reason == 'target_coverage':
            explorer._on_coverage(Float32(data=0.8))
        else:
            explorer._on_observation(observation())
        assert explorer._finished
        explorer._poll_control()
        assert control.calls[-1][0].data is False
        control.reply(controller._on_enable(control.calls[-1][0], SetBool.Response()))
        assert decisions[-1].linear_x == decisions[-1].angular_z == 0
        explorer._on_observation(observation())
        assert len(actions) == (1 if reason == 'target_coverage' else 2)
        controller.destroy_node()

    monkeypatch.setattr(rclpy, 'spin', exercise)
    dqn_explorer.main(['--ros-args', '-p', f'model_path:={model}', '-p', 'max_steps:=2'])


def test_dqn_explorer_finishes_only_when_reachable_coverage_reaches_target(
    monkeypatch, transport, tmp_path
):
    model = tmp_path / 'policy.pt'
    save_checkpoint(
        model,
        DQNPolicy(OBSERVATION_SIZE, POLICY_ACTION_COUNT, seed=1),
        {'observation_contract': OBSERVATION_CONTRACT},
        {},
    )

    def exercise(explorer):
        records = []
        explorer.get_logger().info = records.append
        explorer._poll_control()
        control = transport['/safe_motion_controller/enable']
        control.reply(provenance(SetBool.Response))

        explorer._on_map_coverage(Float32(data=0.99))
        assert not explorer._finished
        assert 'Mapa descubierto: 99%' in records
        explorer._poll_control()
        assert len(control.calls) == 1

        explorer._on_reachable_coverage(Float32(data=0.98))
        assert explorer._finished
        explorer._poll_control()
        assert control.calls[-1][0].data is False
        control.reply(provenance(SetBool.Response))

    monkeypatch.setattr(rclpy, 'spin', exercise)
    dqn_explorer.main([
        '--ros-args', '-p', f'model_path:={model}', '-p', 'target_coverage:=0.98'
    ])


def test_dqn_explorer_logs_each_new_whole_map_coverage_percent(
    monkeypatch, transport, tmp_path
):
    model = tmp_path / 'policy.pt'
    save_checkpoint(
        model,
        DQNPolicy(OBSERVATION_SIZE, POLICY_ACTION_COUNT, seed=1),
        {'observation_contract': OBSERVATION_CONTRACT},
        {},
    )
    records = []

    def exercise(explorer):
        explorer.get_logger().info = records.append
        for coverage in (0.031, 0.039, 0.040, 0.049, 0.050, float('nan')):
            explorer._on_map_coverage(Float32(data=coverage))

    monkeypatch.setattr(rclpy, 'spin', exercise)
    dqn_explorer.main(['--ros-args', '-p', f'model_path:={model}'])

    assert [record for record in records if record.startswith('Mapa descubierto:')][:3] == [
        'Mapa descubierto: 3%', 'Mapa descubierto: 4%', 'Mapa descubierto: 5%'
    ]


def test_frontier_explorer_logs_each_new_whole_map_coverage_percent(
    monkeypatch, transport
):
    records = []

    def exercise(explorer):
        explorer.get_logger().info = records.append
        for coverage in (0.031, 0.039, 0.040, 0.049, 0.050, float('nan')):
            explorer._on_map_coverage(Float32(data=coverage))

    monkeypatch.setattr(rclpy, 'spin', exercise)
    frontier_explorer.main([])

    assert [record for record in records if record.startswith('Mapa descubierto:')] == [
        'Mapa descubierto: 3%', 'Mapa descubierto: 4%', 'Mapa descubierto: 5%'
    ]


@pytest.mark.parametrize('explorer_name', ['dqn', 'frontier'])
def test_explorers_log_simulated_elapsed_time_at_mission_end(
    monkeypatch, transport, tmp_path, explorer_name
):
    """The video comparison needs a duration measured in simulation time."""
    records = []
    clock_ns = [10_000_000_000]

    def exercise(explorer):
        explorer.get_logger().info = records.append
        explorer.get_clock = lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=clock_ns[0])
        )
        explorer._poll_control()
        control = transport['/safe_motion_controller/enable']
        control.reply(SetBool.Response(success=True))
        clock_ns[0] += 12_500_000_000
        explorer._on_coverage(Float32(data=0.98))

    monkeypatch.setattr(rclpy, 'spin', exercise)
    if explorer_name == 'dqn':
        model = tmp_path / 'policy.pt'
        save_checkpoint(
            model,
            DQNPolicy(OBSERVATION_SIZE, POLICY_ACTION_COUNT, seed=1),
            {'observation_contract': OBSERVATION_CONTRACT},
            {},
        )
        dqn_explorer.main([
            '--ros-args', '-p', f'model_path:={model}', '-p', 'target_coverage:=0.98'
        ])
    else:
        frontier_explorer.main(['--ros-args', '-p', 'target_coverage:=0.98'])

    assert records[-1] == (
        'Mission ended: target_coverage; reachable_coverage=98.00%; '
        'simulated_time=12.50s; steps=0.'
    )
