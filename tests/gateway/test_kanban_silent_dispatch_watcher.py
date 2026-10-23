"""Exercise actual embedded watcher wiring without workers or Telegram."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock


def test_embedded_loop_alerts_once_after_elapsed_threshold(tmp_path, monkeypatch):
    from gateway import kanban_watchers as watchers
    from gateway import kanban_dispatch_health as health
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as kbd
    sent = []
    original = health.observe
    def observe(path, **kwargs):
        return original(path, **kwargs, send=lambda text: sent.append(text) or True)
    monkeypatch.setattr(health, 'observe', observe)
    clock = iter([0, 901, 902])
    monkeypatch.setattr(watchers.time, 'time', lambda: current[0])
    monkeypatch.setattr(watchers.asyncio, 'sleep', _noop)
    monkeypatch.setattr(watchers, '_kanban_dispatch_allowed', lambda: True)
    monkeypatch.setattr(watchers, '_resolve_auto_decompose_settings', lambda load: (False, 0))
    monkeypatch.setattr(watchers, '_resolve_dispatcher_settings', lambda *a: SimpleNamespace(interval=60))
    monkeypatch.setattr(kb, 'kanban_home', lambda: tmp_path)
    monkeypatch.setattr(kbd, 'reap_worker_zombies', lambda: [])
    dispatcher = Mock()
    dispatcher.tick_once.return_value = []
    dispatcher.health_causes.return_value = ['probe/t_test: credenciales/cuota']
    monkeypatch.setattr(watchers, '_KanbanDispatcher', lambda *a: dispatcher)
    current = [next(clock)]
    class Runner(watchers.GatewayKanbanWatchersMixin):
        _running = True
        def _kanban_dispatcher_boot(self):
            return lambda: {}, kb, {}
        async def _sleep_between_ticks(self, interval):
            try:
                current[0] = next(clock)
            except StopIteration:
                self._running = False
    asyncio.run(Runner()._kanban_dispatcher_watcher())
    assert len(sent) == 1 and 't_test' in sent[0]


async def _noop(*args):
    pass
