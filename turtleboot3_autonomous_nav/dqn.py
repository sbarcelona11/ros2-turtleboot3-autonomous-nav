"""Pure-Python DQN inference, replay, and checkpoint primitives."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import pickle
import random
from typing import Any, Protocol

from turtleboot3_autonomous_nav.metrics import copy_metric_snapshot


class RandomSource(Protocol):
    """The small random interface used for epsilon-greedy action selection."""

    def random(self) -> float: ...

    def randrange(self, stop: int) -> int: ...


@dataclass(frozen=True)
class Transition:
    """One immutable state transition available to DQN training."""

    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool


class ReplayBuffer:
    """A fixed-capacity FIFO buffer of immutable DQN transitions."""

    def __init__(self, capacity: int = 100_000) -> None:
        if not _is_positive_int(capacity):
            raise ValueError('capacity must be a positive integer')
        self.capacity = capacity
        self._transitions: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._transitions)

    def add(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> None:
        """Store a transition, evicting the oldest entry at capacity."""
        self._transitions.append(
            Transition(
                _finite_vector(state, 'state'),
                int(action),
                _finite_scalar(reward, 'reward'),
                _finite_vector(next_state, 'next_state'),
                bool(done),
            )
        )

    def sample(
        self, batch_size: int, rng: random.Random | None = None
    ) -> list[Transition]:
        """Draw distinct transitions uniformly from the stored experience."""
        if not _is_positive_int(batch_size):
            raise ValueError('batch_size must be a positive integer')
        if batch_size > len(self._transitions):
            raise ValueError('batch_size cannot exceed the replay size')
        return (rng or random).sample(tuple(self._transitions), batch_size)


@dataclass
class _DenseLayer:
    """One fully connected affine layer represented with Python lists."""

    weights: list[list[float]]
    bias: list[float]

    def forward(self, values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            bias + sum(weight * value for weight, value in zip(row, values))
            for row, bias in zip(self.weights, self.bias)
        )

    def as_dict(self) -> dict[str, list[list[float]] | list[float]]:
        return {'weights': deepcopy(self.weights), 'bias': list(self.bias)}


class DQNPolicy:
    """A two-hidden-layer fully connected Q-network with a copied target net."""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        hidden_sizes: tuple[int, int] = (128, 128),
        seed: int | None = None,
    ) -> None:
        if not _is_positive_int(observation_size):
            raise ValueError('observation_size must be a positive integer')
        if not _is_positive_int(action_count):
            raise ValueError('action_count must be a positive integer')
        if (
            len(hidden_sizes) != 2
            or not all(_is_positive_int(size) for size in hidden_sizes)
        ):
            raise ValueError('hidden_sizes must contain two positive integers')

        self.observation_size = observation_size
        self.action_count = action_count
        self.hidden_sizes = hidden_sizes
        initializer = random.Random(seed)
        layer_sizes = (observation_size, *hidden_sizes, action_count)
        self._layers = [
            _new_layer(layer_sizes[index], layer_sizes[index + 1], initializer)
            for index in range(len(layer_sizes) - 1)
        ]
        self._target_layers = deepcopy(self._layers)
        self._rng = random.Random(seed)

    def q_values(self, observation: Sequence[float]) -> tuple[float, ...]:
        """Evaluate online Q-values after validating the observation dimension."""
        return self._evaluate(self._layers, self._validated_observation(observation))

    def target_q_values(self, observation: Sequence[float]) -> tuple[float, ...]:
        """Evaluate the copied target network for a validated observation."""
        return self._evaluate(
            self._target_layers, self._validated_observation(observation)
        )

    def select_action(self, observation: Sequence[float]) -> int:
        """Return the greedy inference action without exploration."""
        values = self.q_values(observation)
        return max(range(self.action_count), key=values.__getitem__)

    def select_training_action(
        self,
        observation: Sequence[float],
        epsilon: float,
        rng: RandomSource | None = None,
    ) -> int:
        """Select epsilon-greedily while still validating every observation."""
        if not math.isfinite(float(epsilon)) or not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError('epsilon must be finite and between 0.0 and 1.0')
        values = self.q_values(observation)
        source = rng or self._rng
        if source.random() < epsilon:
            return source.randrange(self.action_count)
        return max(range(self.action_count), key=values.__getitem__)

    def copy_target_network(self) -> None:
        """Replace target-network weights with the current online weights."""
        self._target_layers = deepcopy(self._layers)

    def state_dict(self) -> dict[str, Any]:
        """Return dimensions and both network weight sets for checkpointing."""
        return {
            'observation_size': self.observation_size,
            'action_count': self.action_count,
            'hidden_sizes': list(self.hidden_sizes),
            'layers': [layer.as_dict() for layer in self._layers],
            'target_layers': [layer.as_dict() for layer in self._target_layers],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load compatible online and target weights into this policy."""
        expected = {
            'observation_size': self.observation_size,
            'action_count': self.action_count,
            'hidden_sizes': list(self.hidden_sizes),
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f'checkpoint {name} does not match this policy')
        layer_sizes = (self.observation_size, *self.hidden_sizes, self.action_count)
        self._layers = _layers_from_state(state.get('layers'), layer_sizes)
        self._target_layers = _layers_from_state(
            state.get('target_layers'), layer_sizes
        )

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> 'DQNPolicy':
        """Construct a policy using the dimensions and weights in a checkpoint."""
        if not isinstance(state, Mapping):
            raise ValueError('checkpoint policy state must be a mapping')
        observation_size = state.get('observation_size')
        action_count = state.get('action_count')
        raw_hidden_sizes = state.get('hidden_sizes')
        if (
            not _is_positive_int(observation_size)
            or not _is_positive_int(action_count)
            or not isinstance(raw_hidden_sizes, list)
            or len(raw_hidden_sizes) != 2
            or not all(_is_positive_int(size) for size in raw_hidden_sizes)
        ):
            raise ValueError('checkpoint contains invalid policy dimensions')
        policy = cls(observation_size, action_count, tuple(raw_hidden_sizes))
        policy.load_state_dict(state)
        return policy

    def _validated_observation(
        self, observation: Sequence[float]
    ) -> tuple[float, ...]:
        values = _finite_vector(observation, 'observation')
        if len(values) != self.observation_size:
            raise ValueError(
                f'expected observation size {self.observation_size}, got {len(values)}'
            )
        return values

    @staticmethod
    def _evaluate(
        layers: Sequence[_DenseLayer], observation: tuple[float, ...]
    ) -> tuple[float, ...]:
        values = observation
        for index, layer in enumerate(layers):
            values = layer.forward(values)
            if index < len(layers) - 1:
                values = tuple(max(0.0, value) for value in values)
        return values


def save_checkpoint(
    path: str | Path,
    policy: DQNPolicy,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    """Persist trusted policy weights, dimensions, configuration, and metrics."""
    if not isinstance(policy, DQNPolicy):
        raise TypeError('policy must be a DQNPolicy')
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format_version': 1,
        'policy': policy.state_dict(),
        'config': copy_metric_snapshot(config),
        'metrics': copy_metric_snapshot(metrics),
    }
    with checkpoint_path.open('wb') as checkpoint_file:
        pickle.dump(payload, checkpoint_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str | Path) -> tuple[DQNPolicy, dict[str, Any], dict[str, Any]]:
    """Restore a trusted checkpoint and its policy dimensions before inference."""
    with Path(path).open('rb') as checkpoint_file:
        payload = pickle.load(checkpoint_file)
    if not isinstance(payload, Mapping) or payload.get('format_version') != 1:
        raise ValueError('unsupported DQN checkpoint format')
    policy = DQNPolicy.from_state_dict(payload.get('policy'))
    config = copy_metric_snapshot(payload.get('config'))
    metrics = copy_metric_snapshot(payload.get('metrics'))
    return policy, config, metrics


def _new_layer(
    input_size: int, output_size: int, initializer: random.Random
) -> _DenseLayer:
    limit = math.sqrt(6.0 / float(input_size + output_size))
    return _DenseLayer(
        weights=[
            [initializer.uniform(-limit, limit) for _ in range(input_size)]
            for _ in range(output_size)
        ],
        bias=[0.0] * output_size,
    )


def _layers_from_state(
    raw_layers: Any, layer_sizes: tuple[int, ...]
) -> list[_DenseLayer]:
    if not isinstance(raw_layers, list) or len(raw_layers) != len(layer_sizes) - 1:
        raise ValueError('checkpoint contains an invalid number of layers')
    layers: list[_DenseLayer] = []
    for index, raw_layer in enumerate(raw_layers):
        if not isinstance(raw_layer, Mapping):
            raise ValueError('checkpoint layer must be a mapping')
        input_size, output_size = layer_sizes[index : index + 2]
        raw_weights = raw_layer.get('weights')
        raw_bias = raw_layer.get('bias')
        if (
            not isinstance(raw_weights, list)
            or len(raw_weights) != output_size
            or not isinstance(raw_bias, list)
            or len(raw_bias) != output_size
        ):
            raise ValueError('checkpoint layer dimensions are invalid')
        weights = [_finite_vector(row, 'checkpoint weight row') for row in raw_weights]
        if any(len(row) != input_size for row in weights):
            raise ValueError('checkpoint weight dimensions are invalid')
        bias = _finite_vector(raw_bias, 'checkpoint bias')
        layers.append(_DenseLayer([list(row) for row in weights], list(bias)))
    return layers


def _finite_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f'{name} must be a numeric sequence')
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be a numeric sequence') from error
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f'{name} values must be finite')
    return result


def _finite_scalar(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be numeric') from error
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
