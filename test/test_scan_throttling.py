"""The mapper must stay responsive to its reset service.

Tracing 360 rays through the grid costs tens of milliseconds, and scans
arrive far faster than the map needs to be updated.  On a single-threaded
executor an unthrottled scan callback leaves no room for the reset service,
so an episode reset times out and training aborts.  Integrating scans at a
bounded rate keeps the node available without losing map quality: the robot
moves under two centimetres between accepted scans.
"""

import pytest

from turtleboot3_autonomous_nav.coverage_mapper import should_process_scan


ONE_MS = 1_000_000


def test_the_first_scan_is_always_processed():
    assert should_process_scan(0, None, 100 * ONE_MS)


def test_a_scan_arriving_within_the_period_is_skipped():
    """Skipping is what frees the executor for the reset service."""
    assert not should_process_scan(50 * ONE_MS, 0, 100 * ONE_MS)


def test_a_scan_arriving_after_the_period_is_processed():
    assert should_process_scan(100 * ONE_MS, 0, 100 * ONE_MS)


def test_a_clock_that_jumps_backwards_processes_the_scan():
    """A simulation reset rewinds the clock; the map must not stall."""
    assert should_process_scan(0, 500 * ONE_MS, 100 * ONE_MS)


def test_a_non_positive_period_processes_every_scan():
    """Disabling the throttle must keep the original behaviour available."""
    assert should_process_scan(1, 0, 0)


@pytest.mark.parametrize('period', [None, 'fast'])
def test_an_invalid_period_is_rejected(period):
    with pytest.raises((TypeError, ValueError)):
        should_process_scan(1, 0, period)
