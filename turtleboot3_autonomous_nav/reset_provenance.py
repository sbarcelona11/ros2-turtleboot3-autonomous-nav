"""Publication-time barriers independent of Gazebo's resettable ROS clock.

The Gazebo bridge and these nodes run on the same host clock. RMW publication
timestamps are Unix nanoseconds, unlike sensor headers and /clock, which rewind
on reset.all. Unsupported (zero) source timestamps fail closed after reset.
"""

from collections.abc import Mapping
from typing import Any


def source_timestamp_ns(info: Any) -> int:
    """Get publisher provenance; never substitute callback receipt time."""
    # Jazzy rclpy supplies a dict; accepting the object form also supports RMW
    # adapters that expose message info fields as attributes.
    stamp = (
        info.get("source_timestamp", 0)
        if isinstance(info, Mapping)
        else getattr(info, "source_timestamp", 0)
    )
    return stamp if isinstance(stamp, int) and not isinstance(stamp, bool) else 0


def is_post_reset_timestamp(stamp_ns: int, cutoff_ns: int | None) -> bool:
    """Require a publication strictly after the acknowledged reset boundary."""
    return cutoff_ns is None or stamp_ns > max(0, cutoff_ns)
