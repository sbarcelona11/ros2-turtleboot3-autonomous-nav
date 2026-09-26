"""The monitor prints differently in a terminal than under a launcher.

A terminal can render a single overwritten status line, but `ros2 launch`
captures stdout through a pipe that shows every write as its own prefixed
line.  The monitor therefore throttles and terminates lines differently
depending on where its output goes.
"""

import pytest

from turtleboot3_autonomous_nav.training_monitor import format_line, should_emit


def test_a_terminal_shows_every_update():
    """Interactive use keeps the live line responsive."""
    assert should_emit(True, elapsed=0.0, interval=2.0)


def test_a_launcher_pipe_waits_for_the_interval():
    """Unthrottled writes would flood the launcher log with one line each."""
    assert not should_emit(False, elapsed=0.5, interval=2.0)


def test_a_launcher_pipe_emits_once_the_interval_passes():
    assert should_emit(False, elapsed=2.0, interval=2.0)


def test_a_terminal_line_is_overwritten_in_place():
    """The carriage return keeps the status on a single terminal row."""
    assert format_line('estado', True) == '\restado'


def test_a_launcher_line_is_a_plain_log_record():
    """Under a launcher each update must be a complete, prefixable line."""
    assert format_line('estado', False) == 'estado\n'


@pytest.mark.parametrize('interval', [0.0, -1.0])
def test_a_non_positive_interval_is_rejected(interval):
    with pytest.raises(ValueError):
        should_emit(False, elapsed=1.0, interval=interval)
