"""Reward the journey towards unexplored ground, not only the arrival.

Coverage pays per cell discovered, so crossing ground that is already mapped
earns nothing on the way.  The trained agent learned the consequence exactly:
63 % of its decisions were turns in place, and a mission plateaued at 51.0 %
with the robot motionless 18 cm from a wall, repeating one action.  From where
it stood, every direction that did not discover something immediately scored
the same.

The shaping term is the plain difference of the frontier distance rather than
the gamma-discounted potential of Ng, Harada and Russell.  With gamma below one
that formulation pays `scale * (1 - gamma) * distance` for standing perfectly
still, which is a reward for loitering far from any frontier - the very
behaviour being corrected.  The plain difference is zero when the robot does
not move, and an approach followed by a retreat nets exactly zero, so it cannot
be farmed by oscillating either.
"""

import pytest

from turtleboot3_autonomous_nav.dqn_trainer import (
    TrainerConfig,
    episode_reward,
    frontier_shaping,
)


def test_approaching_a_frontier_is_rewarded():
    config = TrainerConfig()
    assert frontier_shaping(10, 8, config) > 0.0


def test_retreating_from_a_frontier_is_penalised():
    config = TrainerConfig()
    assert frontier_shaping(8, 10, config) < 0.0


def test_standing_still_earns_nothing():
    """A reward for holding position is what produced the spinning policy."""
    config = TrainerConfig()
    assert frontier_shaping(10, 10, config) == 0.0


def test_an_approach_then_a_retreat_cancels_out():
    """Otherwise oscillating in place becomes a way to farm reward."""
    config = TrainerConfig()
    out = frontier_shaping(10, 7, config)
    back = frontier_shaping(7, 10, config)
    assert out + back == pytest.approx(0.0)


def test_the_reward_scales_with_how_far_the_robot_travelled():
    config = TrainerConfig()
    assert frontier_shaping(10, 5, config) == pytest.approx(
        5.0 * config.frontier_approach_reward)


def test_an_unknown_distance_shapes_nothing():
    """Before the first map, and after exploration finishes, there is no signal."""
    config = TrainerConfig()
    assert frontier_shaping(None, 8, config) == 0.0
    assert frontier_shaping(8, None, config) == 0.0
    assert frontier_shaping(None, None, config) == 0.0


def test_shaping_reaches_the_step_reward():
    """The term has to arrive in the reward, not merely be computable."""
    config = TrainerConfig()
    plain = episode_reward(0, False, False, False, config=config)
    shaped = episode_reward(0, False, False, False, config=config,
                            frontier_shaping_reward=0.2)
    assert shaped > plain


def test_shaping_cannot_escape_the_reward_clip():
    """One huge jump in frontier distance must not dwarf every transition."""
    config = TrainerConfig()
    shaped = episode_reward(0, False, False, False, config=config,
                            frontier_shaping_reward=500.0)
    assert shaped <= config.reward_clip


def test_approaching_outweighs_the_cost_of_the_time_it_takes():
    """Travel has to pay, or the policy is right to stay where it is.

    A step lasts about 0.4 simulated seconds at the measured decision rate, and
    the robot covers roughly one cell of ground in that time.
    """
    config = TrainerConfig()
    step_cost = config.step_penalty * 0.4
    assert frontier_shaping(10, 9, config) > step_cost
