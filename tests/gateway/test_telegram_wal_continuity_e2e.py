"""Synthetic two-generation Telegram WAL continuity canary.

No gateway service, provider, subprocess, DNS, HTTP, or Telegram Bot API is
used.  The test shares only a temporary SQLite ledger between generations.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "get_hermes_home", lambda: home)
    return home


def _event() -> MessageEvent:
    return MessageEvent(
        text="synthetic continuity probe",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="synthetic-message-47",
        platform_update_id=470047,
    )


def _bind_continuity_methods(runner, adapter) -> None:
    runner._prepare_inbound_turn = GatewayRunner._prepare_inbound_turn.__get__(
        runner, GatewayRunner
    )
    runner._recover_inbound_turns = GatewayRunner._recover_inbound_turns.__get__(
        runner, GatewayRunner
    )
    runner._redeliver_pending_delivery_components = (
        GatewayRunner._redeliver_pending_delivery_components.__get__(runner, GatewayRunner)
    )
    runner._complete_inbound_turn = GatewayRunner._complete_inbound_turn.__get__(
        runner, GatewayRunner
    )
    runner._adapter_for_source = lambda _source: adapter
    runner._complete_resume_delivery_handoff = AsyncMock(return_value=True)
    adapter.gateway_runner = runner


async def _drain_adapter(adapter) -> None:
    tasks = list(adapter._background_tasks)
    if tasks:
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_two_generation_telegram_wal_recovers_once_without_network():
    generation_one, adapter_one = make_restart_runner()
    _bind_continuity_methods(generation_one, adapter_one)
    event = _event()
    session_key = "agent:main:telegram:dm:123456"

    turn_id = await generation_one._prepare_inbound_turn(event, session_key)

    with dl._connect() as conn:
        inbound = conn.execute(
            "SELECT turn_id, state, platform, chat_id FROM inbound_turns"
        ).fetchall()
        components_before = conn.execute(
            "SELECT count(*) FROM delivery_components"
        ).fetchone()[0]
        conn.execute(
            "UPDATE inbound_turns SET owner_pid=99999999, owner_started_at=NULL"
        )
    assert inbound == [(turn_id, "received", "telegram", "123456")]
    assert components_before == 0
    assert adapter_one.sent == []

    generation_two, adapter_two = make_restart_runner()
    _bind_continuity_methods(generation_two, adapter_two)
    adapter_two.set_message_handler(AsyncMock(return_value="synthetic recovered reply"))

    assert await generation_two._recover_inbound_turns() == 1
    await _drain_adapter(adapter_two)
    assert await generation_two._recover_inbound_turns() == 0

    with dl._connect() as conn:
        inbound_after = conn.execute(
            "SELECT turn_id, state, platform, chat_id FROM inbound_turns"
        ).fetchall()
        components_after = conn.execute(
            "SELECT kind, state, platform, chat_id, payload FROM delivery_components"
        ).fetchall()
        nonterminal = conn.execute(
            """SELECT count(*) FROM inbound_turns WHERE state IN ('received','claimed')"""
        ).fetchone()[0] + conn.execute(
            """SELECT count(*) FROM delivery_components
               WHERE state IN ('pending','attempting','failed')"""
        ).fetchone()[0]
    assert inbound_after == [(turn_id, "completed", "telegram", "123456")]
    assert len(components_after) == 1
    assert components_after[0][:4] == ("text", "delivered", "telegram", "123456")
    assert "synthetic recovered reply" in components_after[0][4]
    assert nonterminal == 0
    assert adapter_two.sent == ["synthetic recovered reply"]


@pytest.mark.asyncio
async def test_delivery_ledger_rollback_disables_inbound_write_and_recovery(monkeypatch):
    runner, adapter = make_restart_runner()
    _bind_continuity_methods(runner, adapter)
    session_key = "agent:main:telegram:dm:123456"
    dl.record_inbound_turn(
        turn_id="existing-turn",
        session_key=session_key,
        platform="telegram",
        chat_id="123456",
        thread_id=None,
        payload={"text": "existing durable work"},
    )
    dl.record_delivery_plan(
        turn_id="existing-turn",
        session_key=session_key,
        platform="telegram",
        chat_id="123456",
        thread_id=None,
        components=[{"kind": "text", "payload": {"content": "existing reply"}}],
    )
    with dl._connect() as conn:
        conn.execute(
            "UPDATE inbound_turns SET owner_pid=99999999, owner_started_at=NULL"
        )
        before = {
            "inbound": conn.execute(
                """SELECT turn_id, state, attempts, owner_pid, owner_started_at,
                          updated_at FROM inbound_turns ORDER BY turn_id"""
            ).fetchall(),
            "components": conn.execute(
                """SELECT component_id, state, attempts, owner_pid,
                          owner_started_at, updated_at
                   FROM delivery_components ORDER BY component_id"""
            ).fetchall(),
        }

    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: False)

    assert await runner._prepare_inbound_turn(_event(), session_key) is None
    assert await runner._redeliver_pending_delivery_components() == 0
    assert await runner._recover_inbound_turns() == 0
    assert adapter.sent == []
    with dl._connect() as conn:
        after = {
            "inbound": conn.execute(
                """SELECT turn_id, state, attempts, owner_pid, owner_started_at,
                          updated_at FROM inbound_turns ORDER BY turn_id"""
            ).fetchall(),
            "components": conn.execute(
                """SELECT component_id, state, attempts, owner_pid,
                          owner_started_at, updated_at
                   FROM delivery_components ORDER BY component_id"""
            ).fetchall(),
        }
    assert after == before
