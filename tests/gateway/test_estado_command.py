"""Owner-only fleet inspection works without a model, including during a turn."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.slash_commands_estado import GatewayEstadoCommandsMixin, compact_fleet_status
from hermes_cli.commands import should_bypass_active_session


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"platform": Platform.DISCORD}, {"chat_type": "group"},
    {"chat_id": "999"}, {"user_id": "999"}, {"home": None},
    {"allowed": "999"}, {"event_user": "999"}, {"allowed": "*"},
])
async def test_estado_rejects_other_identity_without_running_script(monkeypatch, change):
    import gateway.slash_commands_estado as estado

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="123")
    for key in ("platform", "chat_type", "chat_id", "user_id"):
        if key in change:
            setattr(source, key, change[key])
    runner = GatewayEstadoCommandsMixin()
    home = change.get("home", SimpleNamespace(chat_id="123"))
    runner._intake_adapter_for = lambda source: SimpleNamespace(config=SimpleNamespace(home_channel=home))
    monkeypatch.setattr(estado, "platform_gate_env", lambda name: change.get("allowed", "123"))
    execute = Mock(side_effect=AssertionError("unauthorized execution"))
    monkeypatch.setattr(estado, "_read_fleet_status", execute)
    reply = await runner._handle_estado_command(MessageEvent(
        text="/estado", source=source, user_id=change.get("event_user", "123")))
    assert "restringido" in reply
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_estado_executes_fixed_script_with_timeout_and_no_arguments(monkeypatch, tmp_path):
    import gateway.slash_commands_estado as estado
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    runner = GatewayEstadoCommandsMixin()
    runner._intake_adapter_for = lambda source: SimpleNamespace(
        config=SimpleNamespace(home_channel=SimpleNamespace(chat_id="123")))
    monkeypatch.setattr(estado, "platform_gate_env", lambda name: "123")
    execute = Mock(return_value=subprocess.CompletedProcess([], 0, "Gateway: VIVO\nAlertas:\n  sin alertas", ""))
    monkeypatch.setattr(estado.subprocess, "run", execute)
    event = MessageEvent(text="/estado ; rm -rf /", source=SessionSource(
        platform=Platform.TELEGRAM, chat_id="123", user_id="123"))
    reply = await runner._handle_estado_command(event)
    assert "Gateway: VIVO" in reply
    assert execute.call_args.args[0] == ["python3", str(tmp_path / "alfred/scripts/fleet_status.py")]
    assert execute.call_args.kwargs["timeout"] == 25
    assert should_bypass_active_session(event.get_command())


def test_compact_status_reserves_every_priority_and_preserves_unknown():
    text = ("Gateway: VIVO\nKanban alfred-fleet:\n"
            + "  healthy: blocked=0 stalled=0\n" * 100
            + "  stuck: blocked=1 stalled=0\nCron:\n"
            + "  [success] job · fallos 24 h: 0\n" * 100
            + "  [error] broken · fallos 24 h: 1\nComercial:\n  enviados hoy: sin datos\n"
            + "Alertas:\n  revisar flota\n" + "  aviso\n" * 500)
    result = compact_fleet_status(text)
    assert len(result.encode("utf-16-le")) // 2 < 4096
    assert "stuck" in result and "broken" in result
    assert "enviados hoy: sin datos" in result and "revisar flota" in result
    assert "healthy" not in result


@pytest.mark.parametrize("failure", [
    subprocess.TimeoutExpired("python3", 25), OSError("private local error"),
])
def test_estado_failure_has_no_raw_error_output(monkeypatch, failure):
    import gateway.slash_commands_estado as estado

    monkeypatch.setattr(estado.subprocess, "run", Mock(side_effect=failure))
    reply = estado._read_fleet_status()
    assert "Estado de flota:" in reply
    assert "private local error" not in reply


def test_compact_status_bounds_telegram_utf16_and_does_not_invent_health():
    result = compact_fleet_status("Gateway: " + "🚨" * 2000 +
                                "\nKanban alfred-fleet:\n  sin datos\nCron:\n  jobs: sin datos\n"
                                "Comercial:\n  sin datos\nAlertas:\n  " + "🚨" * 2000)
    assert len(result.encode("utf-16-le")) // 2 < 4096
    assert "jobs: sin datos" in result
    assert "sin fallos" not in result


@pytest.mark.asyncio
async def test_registered_estado_resolves_real_handler(monkeypatch):
    import gateway.slash_commands_estado as estado
    from gateway.run_busy import GatewayBusySessionMixin
    from gateway.slash_commands import GatewaySlashCommandsMixin

    class Runner(GatewayBusySessionMixin, GatewaySlashCommandsMixin):
        pass

    runner = Runner()
    runner._intake_adapter_for = lambda source: SimpleNamespace(
        config=SimpleNamespace(home_channel=SimpleNamespace(chat_id="123")))
    monkeypatch.setattr(estado, "platform_gate_env", lambda name: "123")
    monkeypatch.setattr(estado, "_read_fleet_status", lambda: "Gateway: VIVO")
    handler = runner._command_handler_table(["estado"])["estado"]
    result = await handler(MessageEvent(text="/estado", source=SessionSource(
        platform=Platform.TELEGRAM, chat_id="123", user_id="123")))
    assert result == "Gateway: VIVO"
