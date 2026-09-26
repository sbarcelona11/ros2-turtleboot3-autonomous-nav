"""DQN training over resettable Gazebo exploration episodes.

The reward, termination, checkpoint, and replay-update helpers deliberately
have no ROS dependency.  This keeps the learning contract testable on a host
that does not have a ROS installation; :func:`main` contains the ROS adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
import time
from pathlib import Path
from typing import Any

from turtleboot3_autonomous_nav.control import action_commands_translation
from turtleboot3_autonomous_nav.dqn import (
    DQNPolicy,
    ReplayBuffer,
    load_checkpoint,
    save_checkpoint,
)
from turtleboot3_autonomous_nav.frontier import distance_to_nearest_frontier
from turtleboot3_autonomous_nav.observation import (
    OBSERVATION_CONTRACT,
    observation_label,
    require_observation_contract,
)
from turtleboot3_autonomous_nav.reset_provenance import (
    is_post_reset_timestamp,
    source_timestamp_ns,
)


@dataclass(frozen=True)
class TrainerConfig:
    """All trainer limits, reward weights, and DQN schedule parameters."""

    observation_size: int = 90
    action_count: int = 5
    observation_contract: str = OBSERVATION_CONTRACT
    # The mission budget is simulated time; the step cap only guards against a
    # clock that stops advancing, so it is deliberately generous.
    #
    # It already holds a complete exploration: episodes reach a median of 594
    # decisions against the 518 the deterministic explorer needs, at 4.95 per
    # simulated second.  Wall-clock step rate is roughly half that, because the
    # simulation runs at about 0.44 of real time during training; the two are
    # easy to confuse and only the simulated one bounds what an episode can do.
    max_episode_seconds: float = 120.0
    max_steps: int = 5_000
    max_episodes: int = 100
    target_coverage: float = 0.75
    stall_seconds: float = 20.0
    map_cell_count: int = 160_000
    frontier_threshold: float = 0.50
    new_cell_reward: float = 0.10
    frontier_bonus: float = 0.50
    step_penalty: float = 0.01
    no_progress_penalty: float = 0.10
    # Per cell of ground closed towards the nearest frontier.  Without it the
    # reward pays only on arrival, so crossing mapped ground earns nothing on
    # the way and standing still is the safest thing a policy can do: 63 % of
    # the trained agent's decisions were turns in place.
    frontier_approach_reward: float = 0.05
    # Metres the robot must actually cover in a step for it to count as
    # travelling.  Accepting a frontier distance that merely fell was not
    # enough: that distance drifts by a cell on its own as the laser resolves
    # the map, so a robot spinning on the spot kept resetting the stall clock
    # and turning in place stopped costing it the rest of the episode.  The
    # policy trained that way spun through 79.8 % of its decisions.  A step of
    # driving covers about 0.04 m; a turn in place covers none.
    progress_displacement: float = 0.01
    # An intervention used to cost 1.00 against a step cost of 0.044, so a
    # single brush with a wall was worth twenty-three steps of doing nothing.
    # The policy learned the obvious answer and stopped going near anything.
    # Keeping it a few steps' worth still discourages contact without making
    # avoidance dominate the return.
    intervention_penalty: float = 0.10
    recovery_penalty: float = 0.50
    replay_capacity: int = 100_000
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 0.001
    target_sync_steps: int = 1_000
    epsilon_start: float = 1.00
    epsilon_min: float = 0.05
    evaluation_interval: int = 10
    evaluation_episodes: int = 3
    reset_timeout_seconds: float = 10.0
    # A reset RPC is a request to the simulator between episodes, with the
    # robot already stopped, so a slow reply is retried instead of ending the
    # campaign.  Sensor freshness failures are safety failures and still abort.
    reset_retry_limit: int = 3
    # Stability bounds. A squared loss makes the gradient grow without limit
    # with the temporal-difference error, which is how this policy diverged:
    # coverage fell monotonically across five evaluations as the learned
    # policy took over from exploration.
    huber_delta: float = 1.0
    gradient_clip: float = 10.0
    reward_clip: float = 1.0

    def __post_init__(self) -> None:
        require_observation_contract(
            {"observation_contract": self.observation_contract}
        )
        for name in (
            "observation_size",
            "action_count",
            "max_steps",
            "max_episodes",
            "map_cell_count",
            "replay_capacity",
            "batch_size",
            "target_sync_steps",
            "evaluation_interval",
            "evaluation_episodes",
            "reset_retry_limit",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.target_coverage <= 1.0:
            raise ValueError("target_coverage must be in (0.0, 1.0]")
        if not 0.0 <= self.frontier_threshold <= 1.0:
            raise ValueError("frontier_threshold must be in [0.0, 1.0]")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon bounds must be in [0.0, 1.0]")
        if not 0.0 < self.gamma <= 1.0 or self.learning_rate <= 0.0:
            raise ValueError("gamma and learning_rate must be positive and bounded")
        if not math.isfinite(self.reset_timeout_seconds) or self.reset_timeout_seconds <= 0:
            raise ValueError("reset_timeout_seconds must be finite and positive")
        for name in ("max_episode_seconds", "stall_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def epsilon_decay(self) -> float:
        """Per-episode multiplicative decay that reaches epsilon_min by max_episodes.

        A fixed decay constant desynchronizes from whatever ``max_episodes`` a
        run is given: e.g. 0.995 needs ~600 training episodes to reach a 0.05
        floor, so a 100-episode run never leaves ~60%+ random exploration.
        Deriving decay from the actual schedule keeps annealing complete by
        the configured episode count regardless of how it is set.
        """
        if self.epsilon_start <= 0.0:
            return 1.0
        ratio = self.epsilon_min / self.epsilon_start
        return ratio ** (1.0 / self.max_episodes)


_NOMINAL_STEP_SECONDS = 0.1
"""Fallback step duration used when the simulated clock rewinds or stalls."""


class EpisodeResetGate:
    """Accept only data received after both reset operations acknowledge.

    The mapper supplies a reset epoch and system-time cutoff in its reset ACK.
    A new episode needs later DDS publication timestamps for odometry and scan,
    so callback delivery order and simulation-clock rewinds cannot admit old data.
    """

    def __init__(self) -> None:
        self._phase = "idle"
        self._cutoff_ns: int | None = None
        self._epoch: int | None = None
        self._fresh = {"odom": False, "scan": False}

    def begin_reset(self) -> None:
        """Start a reset transaction without accepting any sensor data."""
        self._phase = "waiting_for_world"

    def world_reset_succeeded(self) -> None:
        """Advance only after the asynchronous Gazebo acknowledgement succeeds."""
        if self._phase != "waiting_for_world":
            raise RuntimeError("world reset acknowledgement is out of order")
        self._phase = "waiting_for_mapper"

    def mapper_reset_succeeded(self, cutoff_ns: int, epoch: int) -> None:
        """Arm the gate with provenance supplied by the mapper reset ACK."""
        if self._phase != "waiting_for_mapper":
            raise RuntimeError("mapper reset acknowledgement is out of order")
        if int(cutoff_ns) < 0 or int(epoch) < 1:
            raise ValueError("mapper reset provenance must be non-negative and current")
        self._cutoff_ns = int(cutoff_ns)
        self._epoch = int(epoch)
        self._fresh = {"odom": False, "scan": False}
        self._phase = "waiting_for_fresh_data"

    def record_sensor(self, name: str, stamp_ns: int) -> bool:
        """Record a post-ACK required sensor timestamp and return acceptance."""
        if name not in self._fresh:
            raise ValueError(f"unknown required sensor: {name}")
        if self._phase != "waiting_for_fresh_data" or not self.accepts_timestamp(
            stamp_ns
        ):
            return False
        self._fresh[name] = True
        return True

    def accepts_timestamp(self, stamp_ns: int) -> bool:
        """Keep filtering old publications even after the episode has started."""
        return (
            self._phase in ("waiting_for_fresh_data", "running")
            and self._cutoff_ns is not None
            and is_post_reset_timestamp(stamp_ns, self._cutoff_ns)
        )

    @property
    def ready(self) -> bool:
        """Whether an episode can begin with isolated map and sensor state."""
        return self._phase == "waiting_for_fresh_data" and all(self._fresh.values())

    def start_episode(self) -> None:
        """Consume a ready gate so later callbacks cannot reinitialize the episode."""
        if not self.ready:
            raise RuntimeError("cannot start an episode before reset data is fresh")
        self._phase = "running"


def parse_reset_provenance(message: str) -> tuple[int, int]:
    """Parse the mapper-issued reset cutoff and epoch from a Trigger response."""
    try:
        payload = json.loads(message)
        cutoff_ns = payload["cutoff_ns"]
        epoch = payload["epoch"]
    except (TypeError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "mapper reset acknowledgement lacks valid provenance"
        ) from error
    if (
        isinstance(cutoff_ns, bool)
        or isinstance(epoch, bool)
        or not isinstance(cutoff_ns, int)
        or not isinstance(epoch, int)
        or cutoff_ns < 0
        or epoch < 1
    ):
        raise ValueError("mapper reset acknowledgement has invalid provenance")
    return cutoff_ns, epoch


def is_making_progress(
    new_cells: int,
    previous_distance: int | None,
    distance: int | None,
    displacement: float = 0.0,
    config: TrainerConfig | None = None,
) -> bool:
    """Whether this step advanced the mission, by discovery or by travel.

    The stall budget used to accept only discovery, and crossing this arena
    takes twice that budget, so an agent that correctly set off for the far
    side was cut short every time: 117 of 127 episodes in a campaign ended on
    `stalled` against 10 that reached the time limit.

    Accepting a frontier distance that merely fell was the wrong repair.  That
    distance drifts by a cell on its own as the laser resolves the map, so a
    robot spinning on the spot saw it fall often enough to keep resetting the
    clock.  Turning in place had been punished by ending the episode and
    forfeiting the rest of its return, and that pressure vanished: stall
    terminations fell to 1 in 39 and the policy trained afterwards spun through
    79.8 % of its decisions, worse than the 63.4 % that prompted the change.

    So travelling requires that the robot actually covered ground.  A turn in
    place covers none, whatever the map does around it.
    """
    if int(new_cells) > 0:
        return True
    if previous_distance is None or distance is None:
        return False
    weights = config or TrainerConfig()
    if float(displacement) < weights.progress_displacement:
        return False
    return int(distance) < int(previous_distance)


def frontier_shaping(
    previous_distance: int | None,
    distance: int | None,
    config: TrainerConfig | None = None,
) -> float:
    """Return the reward for ground closed towards the nearest frontier.

    This is the plain difference of the distance, not the gamma-discounted
    potential of Ng, Harada and Russell.  That formulation is policy-invariant,
    but with gamma below one it pays ``scale * (1 - gamma) * distance`` for
    standing perfectly still, which rewards loitering far from any frontier -
    the exact behaviour being corrected here.  The difference is zero when the
    robot does not move, and an approach followed by a retreat nets zero, so it
    cannot be farmed by oscillating.

    A missing distance shapes nothing: before the first map there is no signal,
    and once exploration finishes there is no frontier left to approach.
    """
    if previous_distance is None or distance is None:
        return 0.0
    weights = config or TrainerConfig()
    return (float(previous_distance) - float(distance)) * weights.frontier_approach_reward


def episode_reward(
    new_cells: int,
    reached_frontier: bool,
    safety_intervention: bool,
    recovery_active: bool,
    no_progress: bool = False,
    config: TrainerConfig | None = None,
    delta_seconds: float = 1.0,
    frontier_shaping_reward: float = 0.0,
) -> float:
    """Return the reward for one control step from observable episode events.

    Costs that represent spending time - driving and making no progress - are
    charged per simulated second, so a faster observation rate does not make
    the same behaviour look more expensive.  Discovering a cell, reaching a
    frontier and triggering the safety controller are events, and are charged
    once regardless of how long the step took.
    """
    if int(new_cells) < 0:
        raise ValueError("new_cells cannot be negative")
    if not math.isfinite(float(delta_seconds)) or float(delta_seconds) <= 0.0:
        raise ValueError("delta_seconds must be finite and positive")
    weights = config or TrainerConfig()
    elapsed = float(delta_seconds)
    reward = float(new_cells) * weights.new_cell_reward
    reward += float(frontier_shaping_reward)
    reward -= weights.step_penalty * elapsed
    if reached_frontier:
        reward += weights.frontier_bonus
    if no_progress:
        reward -= weights.no_progress_penalty * elapsed
    if safety_intervention:
        reward -= weights.intervention_penalty
    if recovery_active:
        reward -= weights.recovery_penalty
    if weights.reward_clip > 0.0:
        # One burst of newly seen cells must not dwarf every other transition.
        reward = min(max(reward, -weights.reward_clip), weights.reward_clip)
    return reward


def episode_end_reason(
    steps: int,
    coverage: float,
    stalled_seconds: float,
    config: TrainerConfig,
    elapsed_seconds: float = 0.0,
) -> str | None:
    """Return the terminal condition reached by an episode, if any.

    The mission budget is simulated time.  The step cap remains as a guard for
    a clock that stops advancing, which would otherwise never end an episode.
    """
    if coverage >= config.target_coverage:
        return "target_coverage"
    if float(elapsed_seconds) >= config.max_episode_seconds:
        return "time_limit"
    if steps >= config.max_steps:
        return "max_steps"
    if float(stalled_seconds) >= config.stall_seconds:
        return "stalled"
    return None


def configure_reset_all(request: Any) -> None:
    """Configure a ``ControlWorld`` request for Gazebo's complete reset.

    ``ros_gz_interfaces/srv/ControlWorld`` wraps a ``gz.msgs.WorldControl``;
    the reset-all flag is therefore ``request.world_control.reset.all``.
    """
    try:
        request.world_control.reset.all = True
    except AttributeError as error:
        raise TypeError("ControlWorld request lacks world_control.reset.all") from error


@dataclass(frozen=True)
class EpisodeDiagnostics:
    """Mission outcomes and policy decisions for one completed episode."""

    reason: str
    reachable_coverage: float
    grid_coverage: float
    interventions: int
    policy_actions: int
    turn_actions: int
    episode: int = 0
    is_evaluation: bool = False

    def as_dict(self) -> dict[str, float | int | str | bool]:
        """Return JSON-ready fields, including the fraction of turning actions."""
        return {
            **asdict(self),
            "turn_ratio": self.turn_actions / self.policy_actions
            if self.policy_actions else 0.0,
        }


def summarize_evaluations(
    diagnostics: Sequence[EpisodeDiagnostics],
) -> dict[str, Any]:
    """Average episode outcomes without weighting by episode length."""
    if not diagnostics:
        raise ValueError("evaluation summary requires at least one episode")
    episodes = [diagnostic.as_dict() for diagnostic in diagnostics]
    return {
        "mean_coverage": sum(item.reachable_coverage for item in diagnostics)
        / len(diagnostics),
        "mean_grid_coverage": sum(item.grid_coverage for item in diagnostics)
        / len(diagnostics),
        "mean_interventions": sum(item.interventions for item in diagnostics)
        / len(diagnostics),
        "mean_turn_ratio": sum(item["turn_ratio"] for item in episodes) / len(episodes),
        "episodes": episodes,
    }


def save_if_improved(
    checkpoint_path: str | Path,
    metrics_path: str | Path,
    policy: DQNPolicy,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    best_mean_coverage: float,
) -> bool:
    """Persist an evaluation checkpoint only if mean coverage strictly improves."""
    mean_coverage = float(metrics.get("mean_coverage", float("nan")))
    if not math.isfinite(mean_coverage):
        raise ValueError("metrics.mean_coverage must be finite")
    if mean_coverage <= float(best_mean_coverage):
        return False

    checkpoint = Path(checkpoint_path)
    metric_file = Path(metrics_path)
    save_checkpoint(checkpoint, policy, config, metrics)
    metric_file.parent.mkdir(parents=True, exist_ok=True)
    metric_file.write_text(
        json.dumps(
            {"config": dict(config), "metrics": dict(metrics)}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return True


_NO_BEST_COVERAGE = "-inf"
"""Sentinel for a campaign that has not completed an evaluation block yet."""


def save_training_state(
    path: str | Path,
    policy: DQNPolicy,
    config: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> None:
    """Persist enough to continue a campaign after the simulator dies.

    Replay memory is deliberately excluded: it is large, and the checkpoint
    format already stores no experience.  A resume continues from the learned
    weights and refills the buffer.
    """
    best = float(progress["best_mean_coverage"])
    stored = dict(progress)
    stored["best_mean_coverage"] = (
        _NO_BEST_COVERAGE if not math.isfinite(best) else best
    )
    stored["evaluation_coverages"] = [
        float(value) for value in progress["evaluation_coverages"]
    ]
    save_checkpoint(path, policy, config, stored)


def load_training_state(
    path: str | Path, observation_size: int, action_count: int
) -> tuple[DQNPolicy, dict[str, Any]] | None:
    """Restore a campaign, or return ``None`` when there is nothing to resume."""
    state_path = Path(path)
    if not state_path.exists():
        return None
    policy, config, progress = load_checkpoint(state_path)
    require_observation_contract(config)
    if (
        policy.observation_size != int(observation_size)
        or policy.action_count != int(action_count)
    ):
        raise ValueError(
            "training state does not match the configured network dimensions"
        )
    restored = dict(progress)
    best = restored.get("best_mean_coverage")
    restored["best_mean_coverage"] = (
        float("-inf") if best == _NO_BEST_COVERAGE else float(best)
    )
    restored["evaluation_coverages"] = [
        float(value) for value in restored.get("evaluation_coverages", [])
    ]
    restored["is_evaluation"] = bool(restored.get("is_evaluation", False))
    for name in ("training_episodes", "total_episodes", "global_steps"):
        restored[name] = int(restored.get(name, 0))
    return policy, restored


def optimize_replay(
    policy: DQNPolicy,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    learning_rate: float,
    config: TrainerConfig | None = None,
) -> float | None:
    """Apply one batched TD update to the policy and return its mean loss.

    The whole batch moves through the network as matrices.  Doing it one
    transition at a time in Python costs over a hundred milliseconds per
    step, which caps the whole training loop at a handful of steps per
    second and starves a DQN of the samples it needs to converge.  Target
    values still come from the copied target network, which this update
    leaves untouched.
    """
    if len(replay) < batch_size:
        return None
    if not 0.0 < float(gamma) <= 1.0 or float(learning_rate) <= 0.0:
        raise ValueError("gamma and learning_rate must be positive and bounded")
    import numpy as np

    state = policy.state_dict()
    weights = [np.asarray(layer["weights"], dtype=float) for layer in state["layers"]]
    biases = [np.asarray(layer["bias"], dtype=float) for layer in state["layers"]]
    transitions = replay.sample(batch_size)

    states = np.asarray([transition.state for transition in transitions], dtype=float)
    next_states = np.asarray(
        [transition.next_state for transition in transitions], dtype=float
    )
    actions = np.asarray([transition.action for transition in transitions], dtype=int)
    rewards = np.asarray([transition.reward for transition in transitions], dtype=float)
    pending = ~np.asarray([transition.done for transition in transitions], dtype=bool)

    activations, pre_activations = _forward_batch(weights, biases, states)
    next_values = _forward_batch(
        [np.asarray(layer["weights"], dtype=float) for layer in state["target_layers"]],
        [np.asarray(layer["bias"], dtype=float) for layer in state["target_layers"]],
        next_states,
    )[0][-1]
    targets = rewards + float(gamma) * next_values.max(axis=1) * pending
    rows = np.arange(batch_size)
    errors = activations[-1][rows, actions] - targets

    # Huber: the gradient is the error only while it is small, and constant
    # beyond that, so one surprising transition cannot dominate the update.
    limits = TrainerConfig() if config is None else config
    huber_delta = float(limits.huber_delta)
    bounded_errors = np.clip(errors, -huber_delta, huber_delta)
    deltas = np.zeros_like(activations[-1])
    deltas[rows, actions] = bounded_errors

    gradients = []
    for index in range(len(weights) - 1, -1, -1):
        gradients.append((index, deltas.T @ activations[index], deltas.sum(axis=0)))
        if index > 0:
            deltas = (deltas @ weights[index]) * (pre_activations[index - 1] > 0.0)

    scale = float(learning_rate) / float(batch_size)
    norm = math.sqrt(
        sum(
            float(np.sum(gradient_weights**2) + np.sum(gradient_bias**2))
            for _, gradient_weights, gradient_bias in gradients
        )
    )
    if limits.gradient_clip > 0.0 and norm > limits.gradient_clip:
        scale *= limits.gradient_clip / norm
    for index, gradient_weights, gradient_bias in gradients:
        weights[index] -= scale * gradient_weights
        biases[index] -= scale * gradient_bias

    for layer, layer_weights, layer_bias in zip(state["layers"], weights, biases):
        layer["weights"] = layer_weights.tolist()
        layer["bias"] = layer_bias.tolist()
    policy.load_state_dict(state)
    magnitudes = np.abs(errors)
    losses = np.where(
        magnitudes <= huber_delta,
        0.5 * errors**2,
        huber_delta * (magnitudes - 0.5 * huber_delta),
    )
    return float(losses.mean())


def _forward_batch(
    weights: Sequence[Any], biases: Sequence[Any], batch: Any
) -> tuple[list[Any], list[Any]]:
    """Run a batch through the network, keeping activations for the backward pass."""
    activations = [batch]
    pre_activations = []
    for index, (layer_weights, layer_bias) in enumerate(zip(weights, biases)):
        pre_activation = activations[-1] @ layer_weights.T + layer_bias
        pre_activations.append(pre_activation)
        activations.append(
            pre_activation
            if index == len(weights) - 1
            else pre_activation * (pre_activation > 0.0)
        )
    return activations, pre_activations


def main(args: list[str] | None = None) -> None:
    """Run the ROS 2 adapter that collects and trains on Gazebo episodes."""
    import numpy as np
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.node import Node
    from rclpy.clock import Clock, ClockType
    from ros_gz_interfaces.srv import ControlWorld
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32, String
    from std_srvs.srv import SetBool, Trigger

    class DQNTrainer(Node):
        """Collect asynchronous ROS observations into bounded DQN episodes."""

        def __init__(self) -> None:
            super().__init__("dqn_trainer")
            self._declare_parameters()
            self._config = TrainerConfig(
                **{
                    field: self.get_parameter(field).value
                    for field in TrainerConfig.__dataclass_fields__
                }
            )
            self._policy = DQNPolicy(
                self._config.observation_size, self._config.action_count
            )
            self._replay = ReplayBuffer(self._config.replay_capacity)
            self._action_publisher = self.create_publisher(
                Int32, "/exploration_action", 10
            )
            self._metrics_publisher = self.create_publisher(
                Float32MultiArray, "/training_metrics", 10
            )
            self._diagnostics_publisher = self.create_publisher(
                String, "/training_episode_diagnostics", 10
            )
            self.create_subscription(
                Float32MultiArray, "/dqn_observation", self._on_observation, 10
            )
            self.create_subscription(
                Float32, "/coverage_metrics", self._on_coverage, 10
            )
            self.create_subscription(
                Float32, "/coverage_reachable", self._on_reachable_coverage, 10
            )
            self.create_subscription(
                Bool, "/safety_intervention", self._on_intervention, 10
            )
            self.create_subscription(Bool, "/recovery_active", self._on_recovery, 10)
            self.create_subscription(Odometry, "/odom", self._on_odometry, 10)
            self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
            self.create_subscription(
                OccupancyGrid, "/monitoring_map", self._on_monitoring_map, 10
            )
            self._world_control = self.create_client(ControlWorld, "/world/dqn/control")
            self._mapper_reset = self.create_client(Trigger, "/coverage_mapper/reset")
            self._builder_reset = self.create_client(Trigger, "/observation_builder/reset")
            self._controller_enable = self.create_client(SetBool, "/safe_motion_controller/enable")

            self._coverage = 0.0
            self._reachable_coverage = 0.0
            self._monitoring_map: OccupancyGrid | None = None
            self._pose: tuple[float, float] | None = None
            self._frontier_distance: int | None = None
            self._progress_pose: tuple[float, float] | None = None
            self._coverage_baseline = 0.0
            self._episode_start_seconds = 0.0
            self._last_step_seconds = 0.0
            self._previous_state: tuple[float, ...] | None = None
            self._previous_action: int | None = None
            self._steps = 0
            self._stalled_seconds = 0.0
            self._frontier_distance = None
            self._progress_pose = None
            self._episode_reward = 0.0
            self._episode_interventions = 0
            self._episode_actions = 0
            self._episode_turns = 0
            self._intervention_seen = False
            self._recovery_seen = False
            self._intervention_active = False
            self._recovery_active = False
            self._reset_gate = EpisodeResetGate()
            self._running_episode = False
            self._is_evaluation = False
            self._training_episodes = 0
            self._total_episodes = 0
            self._evaluation_coverages: list[float] = []
            self._evaluation_diagnostics: list[EpisodeDiagnostics] = []
            self._best_mean_coverage = float("-inf")
            self._global_steps = 0
            self._failed = False
            self._deadline_ns: int | None = None
            self._wait_reason = ""
            self._pending_service = None
            self._active_service = None
            self._service_attempts = 0
            self._pending_future = None
            self._observation_epoch: int | None = None
            self.create_timer(0.1, self._poll_reset, clock=Clock(clock_type=ClockType.STEADY_TIME))
            self._model_directory = Path(
                str(self.declare_parameter("model_directory", "models").value)
            )
            self._state_path = self._model_directory / "training_state.pt"
            if bool(self.declare_parameter("resume", True).value):
                self._resume_campaign()
            self._request_world_reset()

        def _resume_campaign(self) -> None:
            """Continue a campaign the simulator interrupted, if one is stored."""
            restored = load_training_state(
                self._state_path,
                self._config.observation_size,
                self._config.action_count,
            )
            if restored is None:
                return
            self._policy, progress = restored
            self._training_episodes = progress["training_episodes"]
            self._total_episodes = progress["total_episodes"]
            self._global_steps = progress["global_steps"]
            self._best_mean_coverage = progress["best_mean_coverage"]
            self._is_evaluation = progress["is_evaluation"]
            self._evaluation_coverages = list(progress["evaluation_coverages"])
            self._evaluation_diagnostics = [
                EpisodeDiagnostics(**{
                    field: record[field]
                    for field in EpisodeDiagnostics.__dataclass_fields__
                    if field in record
                })
                for record in progress.get("evaluation_diagnostics", [])
            ]
            if len(self._evaluation_coverages) != len(self._evaluation_diagnostics):
                self.get_logger().warning(
                    "Restarting partial evaluation batch: saved episode diagnostics "
                    "are incomplete. Policy and training progress are preserved."
                )
                self._evaluation_coverages.clear()
                self._evaluation_diagnostics.clear()
            self.get_logger().info(
                f"Resumed campaign at training episode {self._training_episodes} "
                f"with {self._global_steps} steps. Replay memory starts empty."
            )

        def _save_campaign(self) -> None:
            """Record progress so a simulator crash costs one episode, not a run."""
            try:
                save_training_state(
                    self._state_path,
                    self._policy,
                    asdict(self._config),
                    {
                        "training_episodes": self._training_episodes,
                        "total_episodes": self._total_episodes,
                        "global_steps": self._global_steps,
                        "best_mean_coverage": self._best_mean_coverage,
                        "is_evaluation": self._is_evaluation,
                        "evaluation_coverages": self._evaluation_coverages,
                        "evaluation_diagnostics": [
                            diagnostic.as_dict()
                            for diagnostic in self._evaluation_diagnostics
                        ],
                    },
                )
            except OSError as error:
                # Losing one snapshot must not end a run that is otherwise fine.
                self.get_logger().warning(f"Could not save training state: {error}")

        def _declare_parameters(self) -> None:
            defaults = TrainerConfig()
            for field, value in asdict(defaults).items():
                self.declare_parameter(field, value)

        def _on_coverage(self, message: Float32, info: Any) -> None:
            if not self._reset_gate.accepts_timestamp(source_timestamp_ns(info)):
                return
            coverage = float(message.data)
            if not math.isfinite(coverage):
                return
            coverage = min(max(coverage, 0.0), 1.0)
            if self._running_episode:
                self._coverage = coverage

        def _on_reachable_coverage(self, message: Float32, info: Any) -> None:
            """Track mission progress against the region the robot can reach."""
            if not self._reset_gate.accepts_timestamp(source_timestamp_ns(info)):
                return
            coverage = float(message.data)
            if not math.isfinite(coverage):
                return
            if self._running_episode:
                self._reachable_coverage = min(max(coverage, 0.0), 1.0)

        def _on_intervention(self, message: Bool, info: Any) -> None:
            if not self._reset_gate.accepts_timestamp(source_timestamp_ns(info)):
                return
            active = bool(message.data)
            if active and not self._intervention_active:
                self._intervention_seen = True
                self._episode_interventions += 1
            self._intervention_active = active

        def _on_recovery(self, message: Bool, info: Any) -> None:
            if not self._reset_gate.accepts_timestamp(source_timestamp_ns(info)):
                return
            active = bool(message.data)
            self._recovery_seen |= active and not self._recovery_active
            self._recovery_active = active

        def _on_odometry(self, message: Odometry, info: Any) -> None:
            position = message.pose.pose.position
            self._pose = (float(position.x), float(position.y))
            stamp_ns = source_timestamp_ns(info)
            if self._reset_gate.record_sensor("odom", stamp_ns):
                self._begin_episode_when_reset_data_is_fresh()

        def _on_monitoring_map(self, message: "OccupancyGrid") -> None:
            self._monitoring_map = message

        def _measure_frontier_distance(self) -> int | None:
            """Cells to the nearest unmonitored place the robot can still reach.

            Measured on the monitoring map rather than the occupancy map for
            the same reason the deterministic explorer plans on it: a cell
            resolved by a distant ray stops being unknown while still needing
            a visit, and searching on occupancy declared a mission complete at
            81.4 %.
            """
            grid, pose = self._monitoring_map, self._pose
            if grid is None or pose is None:
                return None
            resolution = float(grid.info.resolution)
            if not resolution > 0.0:
                return None
            column = int((pose[0] - grid.info.origin.position.x) / resolution)
            row = int((pose[1] - grid.info.origin.position.y) / resolution)
            if not (0 <= row < grid.info.height and 0 <= column < grid.info.width):
                return None
            cells = np.asarray(grid.data, dtype=int).reshape(
                grid.info.height, grid.info.width
            )
            return distance_to_nearest_frontier(cells, (row, column))

        def _on_scan(self, message: LaserScan, info: Any) -> None:
            stamp_ns = source_timestamp_ns(info)
            if self._reset_gate.record_sensor("scan", stamp_ns):
                self._begin_episode_when_reset_data_is_fresh()

        def _begin_episode_when_reset_data_is_fresh(self) -> None:
            if not self._reset_gate.ready:
                return
            self._reset_gate.start_episode()
            self._running_episode = True
            self._arm_deadline("post-reset observation")
            self._episode_start_seconds = self._simulated_seconds()
            self._last_step_seconds = self._episode_start_seconds
            self._coverage = 0.0
            self._reachable_coverage = 0.0
            self._coverage_baseline = 0.0
            self._previous_state = None
            self._previous_action = None
            self._steps = 0
            self._stalled_seconds = 0.0
            self._frontier_distance = None
            self._progress_pose = None
            self._episode_reward = 0.0
            self._episode_interventions = 0
            self._episode_actions = 0
            self._episode_turns = 0
            self._intervention_seen = False
            self._recovery_seen = False
            self.get_logger().info(
                f"Started {'evaluation' if self._is_evaluation else 'training'} "
                f"episode {self._total_episodes + 1} after reset ACKs and fresh "
                "odometry and scan."
            )

        def _on_observation(self, message: Float32MultiArray, info: Any) -> None:
            if not self._running_episode or not self._reset_gate.accepts_timestamp(
                source_timestamp_ns(info)
            ):
                return
            if not message.layout.dim or message.layout.dim[0].label != observation_label(
                self._observation_epoch
            ):
                return
            try:
                observation = tuple(float(value) for value in message.data)
                if len(observation) != self._config.observation_size or not all(
                    math.isfinite(value) for value in observation
                ):
                    raise ValueError("malformed observation")
            except (TypeError, ValueError) as error:
                self.get_logger().warning(f"Rejected DQN observation: {error}")
                return
            self._arm_deadline("episode observation")

            if self._previous_state is not None and self._previous_action is not None:
                self._record_transition(observation)
                if not self._running_episode:
                    return
            epsilon = 0.0 if self._is_evaluation else self._epsilon()
            action = self._policy.select_training_action(observation, epsilon)
            self._episode_actions += 1
            if not action_commands_translation(action):
                self._episode_turns += 1
            self._action_publisher.publish(Int32(data=action))
            self._previous_state = observation
            self._previous_action = action

        def _simulated_seconds(self) -> float:
            """Return the node's simulated clock in seconds."""
            return self.get_clock().now().nanoseconds / 1e9

        def _record_transition(self, next_state: tuple[float, ...]) -> None:
            assert (
                self._previous_state is not None and self._previous_action is not None
            )
            now_seconds = self._simulated_seconds()
            # A rewound or stalled clock must not produce a zero or negative
            # step; falling back to the nominal period keeps the cost sane.
            delta_seconds = now_seconds - self._last_step_seconds
            if not 0.0 < delta_seconds < self._config.max_episode_seconds:
                delta_seconds = _NOMINAL_STEP_SECONDS
            self._last_step_seconds = now_seconds
            elapsed_seconds = max(0.0, now_seconds - self._episode_start_seconds)
            new_cells = max(
                0,
                int(
                    round(
                        (self._coverage - self._coverage_baseline)
                        * self._config.map_cell_count
                    )
                ),
            )
            self._coverage_baseline = self._coverage
            previous_distance, distance = (
                self._frontier_distance,
                self._measure_frontier_distance(),
            )
            self._frontier_distance = distance
            previous_pose, self._progress_pose = self._progress_pose, self._pose
            displacement = 0.0
            if previous_pose is not None and self._pose is not None:
                displacement = math.dist(previous_pose, self._pose)
            progressing = is_making_progress(
                new_cells, previous_distance, distance, displacement, self._config
            )
            no_progress = not progressing
            self._stalled_seconds = (
                self._stalled_seconds + delta_seconds if no_progress else 0.0
            )
            reached_frontier = (
                new_cells > 0
                and max(next_state[12:20], default=0.0)
                >= self._config.frontier_threshold
            )
            reward = episode_reward(
                new_cells,
                reached_frontier,
                self._intervention_seen,
                self._recovery_seen,
                no_progress,
                self._config,
                delta_seconds,
                frontier_shaping(previous_distance, distance, self._config),
            )
            self._intervention_seen = False
            self._recovery_seen = False
            self._steps += 1
            self._global_steps += 1
            self._episode_reward += reward
            reason = episode_end_reason(
                self._steps,
                self._reachable_coverage,
                self._stalled_seconds,
                self._config,
                elapsed_seconds,
            )
            done = reason is not None
            self._replay.add(
                self._previous_state, self._previous_action, reward, next_state, done
            )
            if not self._is_evaluation:
                optimize_replay(
                    self._policy,
                    self._replay,
                    self._config.batch_size,
                    self._config.gamma,
                    self._config.learning_rate,
                    self._config,
                )
                if self._global_steps % self._config.target_sync_steps == 0:
                    self._policy.copy_target_network()
            self._publish_metrics()
            if done:
                self._finish_episode(reason)

        def _finish_episode(self, reason: str) -> None:
            self._running_episode = False
            self._total_episodes += 1
            diagnostic = EpisodeDiagnostics(
                reason, self._reachable_coverage, self._coverage,
                self._episode_interventions, self._episode_actions,
                self._episode_turns, self._total_episodes,
                self._is_evaluation,
            )
            self._diagnostics_publisher.publish(
                String(data=json.dumps(diagnostic.as_dict(), sort_keys=True))
            )
            if self._is_evaluation:
                self._evaluation_coverages.append(self._reachable_coverage)
                self._evaluation_diagnostics.append(diagnostic)
                if len(self._evaluation_coverages) >= self._config.evaluation_episodes:
                    metrics = {
                        **summarize_evaluations(self._evaluation_diagnostics),
                        "episode": self._total_episodes,
                        "evaluation_episodes": len(self._evaluation_coverages),
                        "global_steps": self._global_steps,
                    }
                    if save_if_improved(
                        self._model_directory / "best.pt",
                        self._model_directory / "best.metrics.json",
                        self._policy,
                        asdict(self._config),
                        metrics,
                        self._best_mean_coverage,
                    ):
                        self._best_mean_coverage = metrics["mean_coverage"]
                    self._evaluation_coverages.clear()
                    self._evaluation_diagnostics.clear()
                    self._is_evaluation = False
            else:
                self._training_episodes += 1
                if self._training_episodes % self._config.evaluation_interval == 0:
                    self._is_evaluation = True
            self.get_logger().info(
                f"Episode ended: {reason}; reachable coverage="
                f"{self._reachable_coverage:.3f}; grid coverage={self._coverage:.3f}; "
                f"reward={self._episode_reward:.3f}"
            )
            self._save_campaign()
            if (
                self._training_episodes >= self._config.max_episodes
                and not self._is_evaluation
            ):
                self.get_logger().info("Configured training episode limit reached.")
                self._request_service(
                    self._controller_enable, SetBool.Request(data=False),
                    "controller stop", lambda _: self._shutdown(),
                )
                return
            self._request_world_reset()

        def _request_world_reset(self) -> None:
            self._running_episode = False
            self._previous_state = self._previous_action = None
            self._observation_epoch = None
            self._reset_gate.begin_reset()
            self._request_service(
                self._controller_enable, SetBool.Request(data=False),
                "controller disable before reset", self._request_world_after_stop,
            )

        def _request_world_after_stop(self, response: Any) -> None:
            request = ControlWorld.Request()
            configure_reset_all(request)
            self._request_service(
                self._world_control, request, "Gazebo world reset", self._on_world_reset,
            )

        def _on_world_reset(self, response: Any) -> None:
            self._reset_gate.world_reset_succeeded()
            self._request_mapper_reset()

        def _request_mapper_reset(self) -> None:
            self._request_service(
                self._mapper_reset, Trigger.Request(), "coverage mapper reset", self._on_mapper_reset,
            )

        def _on_mapper_reset(self, response: Any) -> None:
            self._mapper_provenance = parse_reset_provenance(response.message)
            self._request_service(
                self._builder_reset, Trigger.Request(), "observation builder reset", self._on_builder_reset,
            )

        def _on_builder_reset(self, response: Any) -> None:
            self._builder_cutoff, self._observation_epoch = parse_reset_provenance(response.message)
            self._request_service(
                self._controller_enable, SetBool.Request(data=True),
                "controller enable after reset", self._on_controller_enabled,
            )

        def _on_controller_enabled(self, response: Any) -> None:
            controller_cutoff, _ = parse_reset_provenance(response.message)
            mapper_cutoff, epoch = self._mapper_provenance
            self._reset_gate.mapper_reset_succeeded(
                max(mapper_cutoff, self._builder_cutoff, controller_cutoff), epoch,
            )
            self._coverage = 0.0
            self._reachable_coverage = 0.0
            self._coverage_baseline = 0.0
            self._intervention_active = self._recovery_active = False
            self._arm_deadline("post-reset odometry and scan")

        def _arm_deadline(self, reason: str) -> None:
            self._wait_reason = reason
            self._deadline_ns = time.monotonic_ns() + int(
                self._config.reset_timeout_seconds * 1_000_000_000
            )

        def _request_service(self, client, request, label, callback) -> None:
            """Bound both discovery and response without blocking the executor."""
            self._pending_service = (client, request, label, callback)
            self._active_service = (client, request, label, callback)
            self._service_attempts = 0
            self._arm_deadline(f"{label} service discovery")

        def _retry_active_service(self) -> bool:
            """Reissue a timed-out reset request while retries remain."""
            if self._active_service is None or self._failed:
                return False
            if self._service_attempts >= self._config.reset_retry_limit:
                return False
            self._service_attempts += 1
            client, request, label, callback = self._active_service
            self.get_logger().warning(
                f"{label} timed out; retry "
                f"{self._service_attempts}/{self._config.reset_retry_limit}."
            )
            future, self._pending_future = self._pending_future, None
            if future is not None:
                future.cancel()
            self._pending_service = self._active_service
            self._arm_deadline(f"{label} service discovery")
            return True

        def _poll_reset(self) -> None:
            if self._deadline_ns is not None and time.monotonic_ns() >= self._deadline_ns:
                if self._failed:
                    self.get_logger().error("Controller stop acknowledgement timed out.")
                    self._shutdown()
                elif self._retry_active_service():
                    return
                else:
                    self._fail(f"Timed out waiting for {self._wait_reason}.")
                return
            if self._pending_service is None:
                return
            client, request, label, callback = self._pending_service
            if not client.service_is_ready():
                return
            self._pending_service = None
            self._arm_deadline(f"{label} response")
            try:
                future = client.call_async(request)
                self._pending_future = future
                future.add_done_callback(lambda result: self._service_done(result, label, callback))
            except Exception as error:
                self._fail(f"{label} failed: {error}")

        def _service_done(self, future, label, callback) -> None:
            if future is not self._pending_future:
                return
            self._pending_future = None
            self._deadline_ns = None
            self._active_service = None
            self._service_attempts = 0
            try:
                response = future.result()
                if response is None or not response.success:
                    raise RuntimeError(getattr(response, 'message', 'request rejected'))
                callback(response)
            except Exception as error:
                if self._failed:
                    self.get_logger().error(f"Controller stop failed: {error}")
                    self._shutdown()
                else:
                    self._fail(f"{label} failed: {error}")

        def _fail(self, message: str) -> None:
            if self._failed:
                self._shutdown()
                return
            self.get_logger().error(message + " Training cannot continue safely.")
            self._failed = True
            self._running_episode = False
            self._previous_state = self._previous_action = None
            self._reset_gate.begin_reset()
            future, self._pending_future = self._pending_future, None
            if future is not None:
                future.cancel()
            self._request_service(
                self._controller_enable, SetBool.Request(data=False),
                "controller stop after failure", lambda _: self._shutdown(),
            )

        def _shutdown(self) -> None:
            self._deadline_ns = None
            if rclpy.ok():
                rclpy.shutdown()

        def _epsilon(self) -> float:
            decayed = self._config.epsilon_start * (
                self._config.epsilon_decay**self._training_episodes
            )
            return max(self._config.epsilon_min, decayed)

        def _publish_metrics(self) -> None:
            self._metrics_publisher.publish(
                Float32MultiArray(
                    data=[
                        float(self._total_episodes + 1),
                        float(self._steps),
                        self._coverage,
                        self._episode_reward,
                        self._epsilon(),
                        (
                            self._best_mean_coverage
                            if math.isfinite(self._best_mean_coverage)
                            else 0.0
                        ),
                        self._reachable_coverage,
                        float(self._episode_interventions),
                        self._episode_turns / self._episode_actions
                        if self._episode_actions else 0.0,
                    ]
                )
            )

    rclpy.init(args=args)
    node = None
    try:
        node = DQNTrainer()
        rclpy.spin(node)
        if node._failed:
            raise RuntimeError('Training aborted after a reset or sensor safety failure')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
