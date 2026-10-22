"""Process ownership shared by the standalone scheduler and embedded ticks."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from hermes_constants import get_hermes_home

# Only the daemon registers lifetime leases. A fork must never inherit ownership.
_owned: dict[str, tuple[int, object]] = {}


def _key(home: Path) -> str:
    return str(home.resolve())


def acquire(home: Path):
    from cron.scheduler import _acquire_tick_lock

    directory = home / "cron"
    directory.mkdir(parents=True, exist_ok=True)
    return _acquire_tick_lock(directory / ".scheduler-owner.lock")


@contextmanager
def tick_ownership():
    """Keep embedded scans out for the lifetime of a daemon, including async jobs."""
    from cron.scheduler import _release_tick_lock

    home = get_hermes_home()
    owner = _owned.get(_key(home))
    if owner is not None and owner[0] == os.getpid():
        yield True
        return
    handle = acquire(home)
    try:
        yield handle is not None
    finally:
        if handle is not None:
            _release_tick_lock(handle)


class SchedulerOwnership:
    """Hold each served home's lease until its jobs drain or the process dies."""

    def __init__(self):
        self._handles: dict[str, object] = {}

    def claim(self, home: Path) -> bool:
        key = _key(home)
        if key in self._handles:
            return True
        handle = acquire(home)
        if handle is None:
            return False
        self._handles[key] = handle
        _owned[key] = (os.getpid(), handle)
        return True

    def close(self):
        from cron.scheduler import _release_tick_lock

        for key, handle in self._handles.items():
            _owned.pop(key, None)
            _release_tick_lock(handle)
        self._handles.clear()
