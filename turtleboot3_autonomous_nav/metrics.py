"""Small helpers for preserving training metrics in model checkpoints."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any


def copy_metric_snapshot(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return an independent copy of a named metrics snapshot.

    Checkpoints must not retain a mutable reference to the trainer's live
    metrics dictionary. Values remain deliberately generic so callers can
    record scalar metrics as well as episode counters and schedules.
    """
    if not isinstance(metrics, Mapping):
        raise TypeError('metrics must be a mapping')
    return deepcopy(dict(metrics))
