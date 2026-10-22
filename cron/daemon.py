"""Standalone cron process. Invoked through ``hermes cron daemon`` for PM boot."""
from __future__ import annotations

import logging
import signal
import threading


def run_daemon(*, interval: float = 60, multiplex: bool = True) -> int:
    from cron.scheduler import _shutdown_parallel_pool
    from cron.scheduler_provider import InProcessCronScheduler
    from cron.scheduler_service import SchedulerOwnership
    from hermes_cli.profiles import profiles_to_serve
    from hermes_constants import get_hermes_home

    if interval <= 0:
        raise ValueError("Cron interval must be positive")
    stop = threading.Event()
    ownership = SchedulerOwnership()
    # Refuse a second daemon before recovery or heartbeat writes touch the store.
    if not ownership.claim(get_hermes_home()):
        logging.getLogger(__name__).error("Cron scheduler already owned by another process")
        return 1

    def homes():
        return [(name, home) for name, home in profiles_to_serve(multiplex=multiplex)
                if ownership.claim(home)]

    previous = {sig: signal.signal(sig, lambda *_: stop.set())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        InProcessCronScheduler().start(stop, interval=interval, profile_homes=homes)
    finally:
        # Keep leases through the drain: a successor must not recover live runs.
        _shutdown_parallel_pool()
        ownership.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


def cron_daemon_command(args):
    return run_daemon(interval=args.interval, multiplex=not args.single_profile)
