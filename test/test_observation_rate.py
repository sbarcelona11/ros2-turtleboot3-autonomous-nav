"""Decisions must come at the rate the robot can act on, not the sensor rate.

The simulated LiDAR publishes above 40 Hz, but a differential-drive robot
exploring at 0.15 m/s moves under four millimetres between those scans.
Deciding that often fills the replay buffer with nearly identical
transitions, multiplies the training cost per episode, and makes the episode
horizon depend on how fast the simulation happens to run.
"""

import pytest

from turtleboot3_autonomous_nav.observation_builder import should_publish_observation


ONE_MS = 1_000_000


def test_the_first_observation_is_always_published():
    assert should_publish_observation(0, None, 100 * ONE_MS)


def test_an_observation_within_the_period_is_skipped():
    assert not should_publish_observation(40 * ONE_MS, 0, 100 * ONE_MS)


def test_an_observation_after_the_period_is_published():
    assert should_publish_observation(100 * ONE_MS, 0, 100 * ONE_MS)


def test_a_clock_that_jumps_backwards_publishes():
    """An episode reset rewinds the clock; control must not stall."""
    assert should_publish_observation(0, 900 * ONE_MS, 100 * ONE_MS)


def test_a_non_positive_period_publishes_every_observation():
    assert should_publish_observation(1, 0, 0)


@pytest.mark.parametrize('period', [None, 'fast'])
def test_an_invalid_period_is_rejected(period):
    with pytest.raises((TypeError, ValueError)):
        should_publish_observation(1, 0, period)
