"""An episode must mean the same mission however fast the simulation runs.

A step is one observation, so counting steps ties the episode budget to the
observation rate, which swings with machine load and Gazebo's real-time
factor.  Measured on this project the robot advanced 12 mm per step in one run
and 2.4 mm in the next: the same `max_steps` bought five times less driving,
coverage numbers stopped being comparable between runs, and the value function
was trained against a horizon that changed underneath it.

Ending on simulated time, and charging the per-step penalties per second
rather than per step, makes both the budget and the return rate-invariant.
"""

import pytest

from turtleboot3_autonomous_nav.dqn_trainer import (
    TrainerConfig,
    episode_end_reason,
    episode_reward,
)


def _config(**overrides):
    return TrainerConfig(**overrides)


def test_an_episode_ends_when_the_simulated_time_budget_is_spent():
    config = _config(max_episode_seconds=60.0)
    assert episode_end_reason(10, 0.1, 0, config, elapsed_seconds=60.0) == 'time_limit'


def test_an_episode_continues_while_time_remains():
    config = _config(max_episode_seconds=60.0)
    assert episode_end_reason(10, 0.1, 0, config, elapsed_seconds=59.0) is None


def test_reaching_the_target_still_ends_the_episode_before_the_time_limit():
    """Finishing the mission early must not be masked by the new budget."""
    config = _config(max_episode_seconds=60.0, target_coverage=0.9)
    assert (
        episode_end_reason(10, 0.95, 0, config, elapsed_seconds=1.0)
        == 'target_coverage'
    )


def test_the_step_cap_still_guards_against_a_stalled_clock():
    """If simulated time stops advancing the episode must not run forever."""
    config = _config(max_steps=100, max_episode_seconds=1e9)
    assert episode_end_reason(100, 0.1, 0, config, elapsed_seconds=0.0) == 'max_steps'


def test_the_step_penalty_is_charged_per_second_not_per_step():
    """Doubling the observation rate must not double the cost of driving."""
    config = _config(step_penalty=0.5, no_progress_penalty=0.0, new_cell_reward=0.0)
    slow = episode_reward(0, False, False, False, False, config, delta_seconds=1.0)
    fast = episode_reward(0, False, False, False, False, config, delta_seconds=0.5)
    assert slow == pytest.approx(-0.5)
    assert fast == pytest.approx(-0.25)


def test_the_no_progress_penalty_is_also_charged_per_second():
    config = _config(step_penalty=0.0, no_progress_penalty=1.0, new_cell_reward=0.0)
    reward = episode_reward(0, False, False, False, True, config, delta_seconds=0.25)
    assert reward == pytest.approx(-0.25)


def test_discovery_and_safety_terms_do_not_scale_with_time():
    """Finding a cell or hitting something is an event, not a duration."""
    config = _config(
        new_cell_reward=1.0,
        step_penalty=0.0,
        no_progress_penalty=0.0,
        intervention_penalty=2.0,
    )
    for delta in (0.1, 1.0):
        assert episode_reward(
            3, False, True, False, False, config, delta_seconds=delta
        ) == pytest.approx(1.0)


@pytest.mark.parametrize('delta', [0.0, -1.0])
def test_a_non_positive_time_step_is_rejected(delta):
    with pytest.raises(ValueError):
        episode_reward(0, False, False, False, False, None, delta_seconds=delta)


def test_a_non_positive_time_budget_is_rejected():
    with pytest.raises(ValueError):
        TrainerConfig(max_episode_seconds=0.0)


def test_a_stall_is_measured_in_simulated_seconds():
    """Otherwise the stall budget shrinks as the observation rate rises."""
    config = _config(stall_seconds=10.0)
    assert episode_end_reason(5, 0.1, 10.0, config, elapsed_seconds=11.0) == 'stalled'
    assert episode_end_reason(5, 0.1, 9.0, config, elapsed_seconds=11.0) is None


def test_a_non_positive_stall_budget_is_rejected():
    with pytest.raises(ValueError):
        TrainerConfig(stall_seconds=0.0)
