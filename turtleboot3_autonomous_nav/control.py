"""Pure action-to-velocity safety decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


FORWARD = 0
SOFT_LEFT = 1
SOFT_RIGHT = 2
LEFT = 3
RIGHT = 4
RECOVER = 5

POLICY_ACTION_COUNT = 5
"""Actions a policy may select.

``RECOVER`` is deliberately outside this range: it holds the robot turning in
place for ``recovery_duration`` seconds, while every other action lasts only
until the next one arrives.  Exposing it to the policy lets a single choice
dominate the time budget, so unsticking stays a controller behaviour that the
controller triggers when a commanded move produces no motion.
"""


@dataclass(frozen=True)
class ControlConfig:
    """Limits and emergency-stop threshold for differential-drive commands."""

    stop_distance: float = 0.20
    # Full speed up to the braking threshold puts the robot into walls hard
    # enough to slip its wheels, which corrupts the odometric heading, and
    # hard enough for this physics engine to tip it over.
    slow_distance: float = 0.60
    tip_limit: float = math.pi / 4.0
    # Backing out is the only motion that frees a robot pinned against a wall.
    reverse_speed: float = 0.08
    rear_clearance: float = 0.25
    # Touching, as opposed to close enough to brake.  The laser clips at its
    # 0.12 m minimum, so a reading at that floor means the obstacle is against
    # the robot; 0.15 m leaves room for that floor without catching an
    # obstacle the robot could still turn away from.
    contact_distance: float = 0.15
    # The moving cylinders reach the robot from the side, where the forward
    # taper does nothing.  A side obstacle slows it without stopping it: a
    # corridor has walls on both sides and halting there ends the mission.
    side_slow_distance: float = 0.35
    side_speed_floor: float = 0.40
    # Momentum is what tips the robot when a moving obstacle reaches it.
    linear_speed: float = 0.10
    soft_turn_speed: float = 0.5
    turn_speed: float = 1.0
    emergency_turn_speed: float = 1.0
    recovery_turn_speed: float = 1.2
    max_linear_speed: float = 0.22
    max_angular_speed: float = 1.5


@dataclass(frozen=True)
class TwistDecision:
    """A bounded command plus the safety and recovery events it caused."""

    linear_x: float
    angular_z: float
    intervention: bool
    recovery: bool = False


def approach_speed_scale(front_range: float, config: ControlConfig) -> float:
    """Return the fraction of nominal speed the clearance ahead allows.

    Braking from full speed at the last moment is what makes the contacts
    violent; tapering from ``slow_distance`` down to ``stop_distance`` lets
    the robot arrive slowly instead.  An unreadable range counts as blocked.
    """
    distance = float(front_range)
    if not math.isfinite(distance) or distance <= config.stop_distance:
        return 0.0
    if distance >= config.slow_distance:
        return 1.0
    span = config.slow_distance - config.stop_distance
    if span <= 0.0:
        return 1.0
    return (distance - config.stop_distance) / span


def side_speed_scale(sector_ranges: np.ndarray, config: ControlConfig) -> float:
    """Return the speed fraction the clearance beside the robot allows.

    Bounded below by ``side_speed_floor`` because walls on both sides are the
    normal case in a corridor, and braking to a halt there would end the
    mission rather than protect it.  An unreadable side counts as close.
    """
    ranges = np.asarray(sector_ranges, dtype=float).reshape(-1)
    sides = ranges[1:3] if ranges.size > 2 else np.asarray([])
    if sides.size == 0:
        return 1.0
    nearest = np.min(np.where(np.isfinite(sides), sides, 0.0))
    if nearest >= config.side_slow_distance:
        return 1.0
    span = config.side_slow_distance
    if span <= 0.0:
        return 1.0
    fraction = max(0.0, float(nearest)) / span
    return config.side_speed_floor + (1.0 - config.side_speed_floor) * fraction


def is_tipped(roll: float, pitch: float, limit: float | None = None) -> bool:
    """Whether the robot has left an upright attitude.

    A tipped robot cannot execute any action, so a mission that keeps issuing
    them wastes the run: measured once at roll 94.7 and pitch 90.0 degrees,
    lying on its side while the planner steered it for minutes.  An
    unreadable attitude counts as tipped, because it must stop the robot.
    """
    threshold = ControlConfig().tip_limit if limit is None else float(limit)
    for angle in (roll, pitch):
        value = float(angle)
        if not math.isfinite(value) or abs(value) > threshold:
            return True
    return False


def action_commands_translation(action: int) -> bool:
    """Whether an action asks the robot to move, rather than turn in place.

    Progress can only be expected from an action that commands a non-zero
    linear velocity.  Turning in place, and the recovery turn itself, never
    translate the robot, so treating them as a lack of progress would make
    the recovery renew the very condition that triggered it.
    """
    commanded = _action_twist(int(action), ControlConfig())
    return commanded is not None and commanded[0] > 0.0


def stale_twist() -> TwistDecision:
    """Return the fail-safe decision used when required sensor input is stale."""
    return TwistDecision(0.0, 0.0, intervention=True, recovery=False)


def control_sector_ranges(
    scan_ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    range_max: float,
) -> np.ndarray:
    """Reduce a scan to ``[front, left, right, rear]`` safety-sector minima.

    The rear sector exists so recovery can back the robot out of an obstacle:
    turning alone cannot free a robot whose nose is against a wall.

    A sector with no positive finite reading is represented by ``0.0`` rather
    than assumed clear. A positive infinite return represents a valid
    out-of-range reading only when the scan provides a valid ``range_max``.
    """
    ranges = np.asarray(scan_ranges, dtype=float).reshape(-1)
    if ranges.size == 0 or not np.isfinite(angle_increment) or angle_increment == 0.0:
        return np.zeros(4, dtype=float)

    angles = angle_min + np.arange(ranges.size) * angle_increment
    wrapped_angles = _wrap_angles(angles)
    masks = (
        np.abs(wrapped_angles) <= math.pi / 6.0,
        (wrapped_angles > math.pi / 6.0)
        & (wrapped_angles <= 5.0 * math.pi / 6.0),
        (wrapped_angles < -math.pi / 6.0)
        & (wrapped_angles >= -5.0 * math.pi / 6.0),
        np.abs(wrapped_angles) > 5.0 * math.pi / 6.0,
    )
    maximum = float(range_max)
    has_valid_maximum = np.isfinite(maximum) and maximum > 0.0
    sectors = np.zeros(4, dtype=float)
    for index, mask in enumerate(masks):
        sector = ranges[mask]
        finite_positive = sector[np.isfinite(sector) & (sector > 0.0)]
        if finite_positive.size:
            sectors[index] = float(np.min(finite_positive))
        elif has_valid_maximum and np.any(np.isposinf(sector)):
            sectors[index] = maximum
    return sectors


def safe_twist(
    action: int,
    sector_ranges: np.ndarray,
    stalled: bool,
    config: ControlConfig,
) -> TwistDecision:
    """Translate an action into a safe twist using ``[front, left, right]`` ranges."""
    ranges = np.asarray(sector_ranges, dtype=float).reshape(-1)
    front_range = ranges[0] if ranges.size else 0.0
    front_is_blocked = not np.isfinite(front_range) or front_range <= config.stop_distance

    if stalled or action == RECOVER:
        # Turning alone cannot free a robot whose nose is against a wall, so
        # back out when the rear sector says there is room to do it.
        rear_range = ranges[3] if ranges.size > 3 else 0.0
        reverse = (
            -_clamp_positive(config.reverse_speed, config.max_linear_speed)
            if np.isfinite(rear_range) and rear_range > config.rear_clearance
            else 0.0
        )
        return TwistDecision(
            reverse,
            _clearer_side_turn(ranges, config.recovery_turn_speed, config),
            intervention=False,
            recovery=True,
        )

    if np.isfinite(front_range) and front_range <= config.contact_distance:
        # The nose is resting on something.  Turning in place here levers the
        # robot up onto it: measured climbing from 0.023 to 0.029 m of height
        # in 70 ms at a full-rate turn, and over at 174 degrees of roll.  Back
        # out when there is room, and turn at half rate when there is not,
        # rather than waiting the twenty seconds the stall timer needs.
        rear_range = ranges[3] if ranges.size > 3 else 0.0
        reverse = (
            -_clamp_positive(config.reverse_speed, config.max_linear_speed)
            if np.isfinite(rear_range) and rear_range > config.rear_clearance
            else 0.0
        )
        return TwistDecision(
            reverse,
            _clearer_side_turn(ranges, config.soft_turn_speed, config),
            intervention=True,
            recovery=reverse < 0.0,
        )

    commanded = _action_twist(action, config)
    if commanded is None:
        return TwistDecision(0.0, 0.0, intervention=True)

    linear_x, angular_z = commanded
    if linear_x > 0.0 and front_is_blocked:
        return TwistDecision(
            0.0,
            _clearer_side_turn(ranges, config.emergency_turn_speed, config),
            intervention=True,
        )

    # Arrive slowly rather than braking at the last moment: a hard contact
    # slips the wheels, which corrupts the heading, and can tip the robot.
    return TwistDecision(
        linear_x
        * approach_speed_scale(front_range, config)
        * side_speed_scale(ranges, config),
        angular_z,
        intervention=False,
    )


def _action_twist(action: int, config: ControlConfig) -> tuple[float, float] | None:
    """Return the bounded nominal twist for a valid discrete action."""
    linear_speed = _clamp_positive(config.linear_speed, config.max_linear_speed)
    soft_turn = _clamp_signed(config.soft_turn_speed, config.max_angular_speed)
    turn = _clamp_signed(config.turn_speed, config.max_angular_speed)
    actions = {
        FORWARD: (linear_speed, 0.0),
        SOFT_LEFT: (linear_speed, soft_turn),
        SOFT_RIGHT: (linear_speed, -soft_turn),
        LEFT: (0.0, turn),
        RIGHT: (0.0, -turn),
    }
    return actions.get(int(action))


def _clamp_positive(value: float, maximum: float) -> float:
    return min(max(float(value), 0.0), max(float(maximum), 0.0))


def _clamp_signed(value: float, maximum: float) -> float:
    limit = max(float(maximum), 0.0)
    return min(max(float(value), -limit), limit)


def _clearer_side_turn(
    ranges: np.ndarray, turn_speed: float, config: ControlConfig
) -> float:
    """Return a bounded turn towards the largest finite side clearance."""
    left_range = ranges[1] if ranges.size > 1 else 0.0
    right_range = ranges[2] if ranges.size > 2 else 0.0
    left_clearance = float(left_range) if np.isfinite(left_range) else 0.0
    right_clearance = float(right_range) if np.isfinite(right_range) else 0.0
    direction = 1.0 if left_clearance >= right_clearance else -1.0
    magnitude = _clamp_positive(turn_speed, config.max_angular_speed)
    return direction * magnitude


def _wrap_angles(angles: np.ndarray) -> np.ndarray:
    return (angles + math.pi) % (2.0 * math.pi) - math.pi
