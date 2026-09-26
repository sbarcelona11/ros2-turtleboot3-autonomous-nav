"""Crossing mapped ground towards a frontier is work; spinning is not.

`stall_seconds` ends an episode after 25 simulated seconds without a new cell.
Crossing this 4.85 m arena at 0.10 m/s takes about 48 seconds, so an agent that
made the right decision - stop harvesting what is in front and travel to the
unexplored far side - had its episode ended halfway there.  117 of 127 episodes
in the first campaign ended on `stalled` against 10 that reached the time limit.

Accepting a frontier distance that merely decreased was the wrong repair, and
it made the behaviour worse rather than better.  That distance moves by a cell
on its own as the laser resolves the map, so a robot spinning on the spot saw
it fall often enough to keep resetting the stall clock.  Turning in place had
been punished by ending the episode and losing the rest of its return; that
pressure disappeared.  The policy trained afterwards took 79.8 % of its
decisions as turns in place and 3.2 % as driving forward, worse than the 63.4 %
that prompted the change.  Stall terminations went from 117 of 127 to 1 of 39.

Travelling has to mean the robot moved.
"""

from turtleboot3_autonomous_nav.dqn_trainer import TrainerConfig, is_making_progress

CONFIG = TrainerConfig()
MOVED = CONFIG.progress_displacement * 2.0
SPINNING = 0.0


def test_the_displacement_threshold_is_below_what_a_step_of_driving_covers():
    """At 0.10 m/s a step of about 0.4 s covers 0.04 m; spinning covers none."""
    assert 0.0 < CONFIG.progress_displacement < 0.04


def test_discovering_cells_is_progress_however_the_robot_moved():
    assert is_making_progress(12, None, None, SPINNING, CONFIG)


def test_driving_closer_to_a_frontier_is_progress():
    """This is the journey the stall budget used to cut short."""
    assert is_making_progress(0, 40, 39, MOVED, CONFIG)


def test_spinning_while_the_distance_drifts_down_is_not_progress():
    """The map resolves as the laser sweeps, so the distance falls on its own."""
    assert not is_making_progress(0, 40, 39, SPINNING, CONFIG)


def test_holding_the_same_distance_is_not_progress():
    assert not is_making_progress(0, 40, 40, MOVED, CONFIG)


def test_retreating_from_a_frontier_is_not_progress():
    assert not is_making_progress(0, 39, 40, MOVED, CONFIG)


def test_discovering_counts_even_while_retreating():
    """Backing out of a corner can still reveal cells; that is real work."""
    assert is_making_progress(5, 39, 40, MOVED, CONFIG)


def test_an_unknown_distance_falls_back_to_discovery_alone():
    """Before the first map arrives there is no distance to compare."""
    assert not is_making_progress(0, None, 12, MOVED, CONFIG)
    assert not is_making_progress(0, 12, None, MOVED, CONFIG)
    assert is_making_progress(3, None, None, MOVED, CONFIG)


def test_a_finished_exploration_is_not_reported_as_progress():
    """With no frontier left, only the coverage target should end the episode."""
    assert not is_making_progress(0, None, None, MOVED, CONFIG)


def test_creeping_below_the_threshold_is_not_travelling():
    """A robot nudged by a contact has not set off anywhere."""
    assert not is_making_progress(
        0, 40, 39, CONFIG.progress_displacement / 2.0, CONFIG)
