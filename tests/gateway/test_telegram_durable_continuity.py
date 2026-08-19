"""Controlled Telegram durable-continuity canary.

This deliberately uses the real Telegram adapter handoff and the real gateway
startup recovery method, with only the network-facing ``handle_message``
endpoint replaced.  No Bot API call is made.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource
from gateway.run import GatewayRunner
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(enabled=True, token="test-token")
    adapter._held_inbound_events = []
    adapter._held_inbound_redispatch_task = None
    adapter._drop_delayed_deliveries = False
    adapter._running = True
    return adapter


def _event():
    return MessageEvent(
        text="durable canary input",
        message_type=MessageType.TEXT,
        message_id="7788",
        platform_update_id=9911,
        user_id="42",
        user_name="canary",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123", user_id="42", user_name="canary", chat_type="dm",
        ),
    )


@pytest.mark.asyncio
async def test_accepted_telegram_input_recovers_once_through_adapter_and_startup(tmp_path, monkeypatch):
    """Accepted input is journalled before dispatch, recovered once, never replayed."""
    import gateway.telegram_inbound_ledger as ledger

    monkeypatch.setattr(ledger, "get_hermes_home", lambda: tmp_path)
    adapter = _adapter()

    async def crash_between_journal_and_dispatch(_event):
        raise SystemExit("controlled process death")

    adapter.handle_message = crash_between_journal_and_dispatch
    event = _event()
    with pytest.raises(SystemExit, match="controlled process death"):
        await adapter._dispatch_durable_text_event(event, "agent:main:telegram:dm:123")

    # The input survived the simulated process death before any handler began.
    # Use an impossible former owner to model the restarted process.
    with ledger._LOCK, ledger._transaction() as conn:
        conn.execute("UPDATE telegram_inbound_obligations SET owner_pid=999999")
    with patch.object(ledger, "_owner_alive", return_value=False):
        rows = ledger.claim_recoverable()
    assert len(rows) == 1
    assert rows[0]["payload"]["text"] == "durable canary input"

    # Put the row back into its pre-startup state and run the actual startup
    # recovery path. (A real restart has a dead former owner, hence the patch.)
    with ledger._LOCK, ledger._transaction() as conn:
        conn.execute(
            "UPDATE telegram_inbound_obligations SET state='accepted', owner_pid=999999"
        )
    recovered_adapter = _adapter()
    recovered_adapter.handle_message = AsyncMock()
    recovered_adapter._dispatch_durable_text_event = AsyncMock(
        wraps=TelegramAdapter._dispatch_durable_text_event.__get__(
            recovered_adapter, TelegramAdapter
        )
    )
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: recovered_adapter}
    runner._is_user_authorized = lambda source, *, allow_adapter_delegation=True: True
    with patch.object(ledger, "_owner_alive", return_value=False):
        assert await runner._recover_telegram_inbound_obligations() == 1
        assert await runner._recover_telegram_inbound_obligations() == 0

    recovered_adapter._dispatch_durable_text_event.assert_awaited_once()
    recovered_adapter.handle_message.assert_awaited_once()
    recovered = recovered_adapter.handle_message.await_args.args[0]
    assert recovered.text == "durable canary input"
    assert getattr(recovered, "_hermes_telegram_inbound_replay") is True


@pytest.mark.asyncio
async def test_polling_queue_journals_before_cursor_advance_and_recovers_once(
    tmp_path, monkeypatch
):
    """PTB may advance its cursor only after the adapter's durable ingress write."""
    import gateway.telegram_inbound_ledger as ledger

    monkeypatch.setattr(ledger, "get_hermes_home", lambda: tmp_path)
    event = _event()
    adapter = _adapter()
    adapter.config.extra = {"allow_from": ["42"]}
    adapter._telegram_ingress_obligations = {}
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 60
    adapter._text_batch_split_delay_seconds = 60
    adapter._should_process_message = lambda message, *, is_command=False: True
    adapter._build_message_event = lambda message, msg_type, update_id=None: event
    adapter._clean_bot_trigger_text = lambda text: text
    adapter._apply_telegram_group_observe_attribution = lambda built: built

    message = SimpleNamespace(
        text=event.text,
        entities=[],
        from_user=SimpleNamespace(id=42, username="canary", full_name="Canary"),
        chat=SimpleNamespace(id=123, type="private", is_forum=False),
        sender_chat=None,
    )
    update = SimpleNamespace(
        update_id=event.platform_update_id,
        message=message,
        effective_message=message,
    )

    class FakeBuilder:
        def update_queue(self, queue):
            self.queue = queue
            return self

        def build(self):
            return SimpleNamespace(update_queue=self.queue)

    # This models PTB's polling loop: enqueue the update, then advance the
    # in-memory cursor.  The application consumer is deliberately not started,
    # so _handle_text_message can never run before the simulated process death.
    app = adapter._build_application_with_durable_ingress(FakeBuilder())
    fake_updater = SimpleNamespace(_last_update_id=None)
    await app.update_queue.put(update)
    fake_updater._last_update_id = update.update_id + 1

    assert fake_updater._last_update_id == 9912
    assert app.update_queue.qsize() == 1
    with ledger._LOCK, ledger._transaction() as conn:
        row = conn.execute(
            "SELECT state, owner_pid FROM telegram_inbound_obligations"
        ).fetchone()
        assert row[0] == "accepted"
        conn.execute(
            "UPDATE telegram_inbound_obligations SET owner_pid=999999"
        )

    # Boot two has no queued PTB object, only the pre-cursor durable row. It
    # reapplies current authorization, dispatches once, and boot three finds
    # no replayable input.
    recovered_adapter = _adapter()
    recovered_adapter.config.extra = {"allow_from": ["42"]}
    recovered_adapter._pending_text_batches = {}
    recovered_adapter._pending_text_batch_tasks = {}
    recovered_adapter._text_batch_delay_seconds = 0.001
    recovered_adapter._text_batch_split_delay_seconds = 0.001
    recovered_adapter.handle_message = AsyncMock()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: recovered_adapter}
    runner._adapter_profile_for_source = lambda source: None
    runner._adapter_authorization_is_upstream = lambda platform, profile=None: False
    runner.pairing_store = None
    runner.pairing_stores = {}

    with patch.object(ledger, "_owner_alive", return_value=False):
        assert await runner._recover_telegram_inbound_obligations() == 1
        await asyncio.sleep(0.02)
        assert await runner._recover_telegram_inbound_obligations() == 0

    recovered_adapter.handle_message.assert_awaited_once()
    recovered = recovered_adapter.handle_message.await_args.args[0]
    assert recovered.text == "durable canary input"
    assert getattr(recovered, "_hermes_telegram_inbound_replay") is True


@pytest.mark.asyncio
async def test_real_adapter_ingress_and_startup_sequence_dispatch_once_with_current_auth(
    tmp_path, monkeypatch
):
    """Two boots recover one accepted update through auth, then replay nothing."""
    import gateway.telegram_inbound_ledger as ledger

    monkeypatch.setattr(ledger, "get_hermes_home", lambda: tmp_path)
    event = _event()
    adapter = _adapter()
    adapter.config.extra = {"allow_from": ["42"]}
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 60
    adapter._text_batch_split_delay_seconds = 60
    adapter._should_process_message = lambda message, *, is_command=False: True
    adapter._ensure_forum_commands = AsyncMock()
    adapter._build_message_event = (
        lambda message, msg_type, update_id=None: event
    )
    adapter._clean_bot_trigger_text = lambda text: text
    adapter._cache_replied_media = AsyncMock()
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    message = SimpleNamespace(
        text=event.text,
        from_user=SimpleNamespace(id=42, username="canary", full_name="Canary"),
        chat=SimpleNamespace(id=123, type="private", is_forum=False),
        sender_chat=None,
    )
    update = SimpleNamespace(
        update_id=event.platform_update_id,
        message=message,
        effective_message=message,
    )

    # Enter through Telegram's actual text-update handler.  The long debounce
    # models a process death after PTB accepted the update but before dispatch.
    await adapter._handle_text_message(update, SimpleNamespace())
    assert adapter._is_user_authorized_from_message(message) is True
    assert len(adapter._pending_text_batches) == 1
    for task in adapter._pending_text_batch_tasks.values():
        task.cancel()
    await asyncio.gather(*adapter._pending_text_batch_tasks.values(), return_exceptions=True)
    with ledger._LOCK, ledger._transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM telegram_inbound_obligations WHERE state='accepted'"
        ).fetchone()[0] == 1
        conn.execute("UPDATE telegram_inbound_obligations SET owner_pid=999999")

    recovered_adapter = _adapter()
    recovered_adapter.config.extra = {"allow_from": ["42"]}
    recovered_adapter._pending_text_batches = {}
    recovered_adapter._pending_text_batch_tasks = {}
    recovered_adapter._text_batch_delay_seconds = 0.001
    recovered_adapter._text_batch_split_delay_seconds = 0.001
    authenticated_dispatches = []

    async def dispatch_after_current_auth(event):
        assert str(event.source.user_id) in {
            str(value) for value in recovered_adapter.config.extra["allow_from"]
        }
        obligation_ids = getattr(event, "_telegram_inbound_obligation_ids", None) or [
            event._telegram_inbound_obligation_id
        ]
        assert ledger.mark_execution_started(obligation_ids)
        authenticated_dispatches.append(event)

    recovered_adapter.handle_message = dispatch_after_current_auth
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: recovered_adapter}
    runner._adapter_profile_for_source = lambda source: None
    runner._adapter_authorization_is_upstream = lambda platform, profile=None: False
    runner.pairing_store = None
    runner.pairing_stores = {}
    runner._redeliver_pending_obligations = AsyncMock(return_value=0)
    runner._schedule_resume_pending_sessions = lambda: 0
    runner._finish_startup_restore = AsyncMock()

    with patch.object(ledger, "_owner_alive", return_value=False):
        assert await runner._reconcile_startup_obligations() == 1
        await asyncio.sleep(0.02)
        assert len(authenticated_dispatches) == 1
        assert await runner._reconcile_startup_obligations() == 0
        await asyncio.sleep(0.02)

    assert len(authenticated_dispatches) == 1
    with ledger._LOCK, ledger._transaction() as conn:
        assert conn.execute(
            "SELECT state FROM telegram_inbound_obligations"
        ).fetchone()[0] == "executing"


@pytest.mark.asyncio
async def test_startup_obligation_reconciliation_fails_closed_before_resume():
    """An inbound recovery failure cannot fall through to generic auto-resume."""
    runner = object.__new__(GatewayRunner)
    runner._redeliver_pending_obligations = AsyncMock(return_value=0)
    runner._recover_telegram_inbound_obligations = AsyncMock(
        side_effect=RuntimeError("controlled recovery failure")
    )
    scheduled = []
    runner._schedule_resume_pending_sessions = lambda: scheduled.append(True)
    runner._finish_startup_restore = AsyncMock()

    with pytest.raises(RuntimeError, match="controlled recovery failure"):
        await runner._reconcile_startup_obligations()

    assert scheduled == []
    runner._finish_startup_restore.assert_not_awaited()


@pytest.mark.asyncio
async def test_held_telegram_text_retries_through_durable_journal_before_handle_message():
    """A retry after a journal failure must not bypass the acceptance journal."""
    adapter = _adapter()
    event = _event()
    adapter._held_inbound_events = [event]
    adapter.handle_message = AsyncMock()
    dispatch_attempts = 0

    async def journal_failed(_event, _session_key=None, *, hold_on_failure=True):
        nonlocal dispatch_attempts
        dispatch_attempts += 1
        assert hold_on_failure is False
        return False

    adapter._dispatch_durable_text_event = journal_failed

    await adapter._redispatch_held_inbound()

    adapter.handle_message.assert_not_awaited()
    assert dispatch_attempts == 1
    assert adapter._held_inbound_events == [event]


@pytest.mark.asyncio
async def test_plain_text_journalled_before_debounce_recovers_once_with_aggregation(tmp_path, monkeypatch):
    """A crash during the debounce interval recovers accepted text exactly once."""
    import gateway.telegram_inbound_ledger as ledger

    monkeypatch.setattr(ledger, "get_hermes_home", lambda: tmp_path)
    adapter = _adapter()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 10
    adapter._text_batch_split_delay_seconds = 10
    adapter.handle_message = AsyncMock()
    event = _event()
    event.text = "queued before crash"
    continuation = _event()
    continuation.message_id = "7789"
    continuation.platform_update_id = 9912
    continuation.text = "continuation survives too"
    session_key = "agent:main:telegram:dm:123"

    # This is the polling-offset-to-flush interval: accepted text must already
    # be on disk while no runner work has started.
    await adapter._accept_text_for_batch(event, session_key)
    await adapter._accept_text_for_batch(continuation, session_key)
    assert adapter.handle_message.await_count == 0
    assert len(adapter._pending_text_batches) == 1
    with ledger._LOCK, ledger._transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM telegram_inbound_obligations WHERE state='accepted'"
        ).fetchone()[0] == 2

    for task in adapter._pending_text_batch_tasks.values():
        task.cancel()
    await asyncio.gather(*adapter._pending_text_batch_tasks.values(), return_exceptions=True)
    with ledger._LOCK, ledger._transaction() as conn:
        conn.execute("UPDATE telegram_inbound_obligations SET owner_pid=999999")

    # A new process has no in-memory batch. Startup recovery rebuilds it and
    # dispatches the persisted input once after the normal debounce path.
    recovered_adapter = _adapter()
    recovered_adapter.config.extra = {"allow_from": ["42"]}
    recovered_adapter._pending_text_batches = {}
    recovered_adapter._pending_text_batch_tasks = {}
    recovered_adapter._text_batch_delay_seconds = 0.001
    recovered_adapter._text_batch_split_delay_seconds = 0.001
    recovered_adapter.handle_message = AsyncMock()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: recovered_adapter}
    runner._is_user_authorized = lambda source, *, allow_adapter_delegation=True: True
    with patch.object(ledger, "_owner_alive", return_value=False):
        assert await runner._recover_telegram_inbound_obligations() == 2
    await asyncio.sleep(0.02)

    recovered_adapter.handle_message.assert_awaited_once()
    recovered = recovered_adapter.handle_message.await_args.args[0]
    assert recovered.text == "queued before crash\ncontinuation survives too"
    assert len(recovered._telegram_inbound_obligation_ids) == 2
    with patch.object(ledger, "_owner_alive", return_value=False):
        assert await runner._recover_telegram_inbound_obligations() == 0


@pytest.mark.asyncio
async def test_live_batch_flush_preserves_every_obligation_until_atomic_execution_seal(
    tmp_path, monkeypatch
):
    """A live flush must seal every accepted fragment as one turn.

    Losing the later ID here would leave it ``accepted`` after its sibling has
    started execution.  A restart could then replay that suffix as a separate
    agent turn.
    """
    import gateway.telegram_inbound_ledger as ledger

    monkeypatch.setattr(ledger, "get_hermes_home", lambda: tmp_path)
    adapter = _adapter()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 10
    adapter._text_batch_split_delay_seconds = 10
    adapter.handle_message = AsyncMock()
    session_key = "agent:main:telegram:dm:123"
    first = _event()
    first.text = "first fragment"
    second = _event()
    second.message_id = "7789"
    second.platform_update_id = 9912
    second.text = "second fragment"

    assert await adapter._accept_text_for_batch(first, session_key)
    assert await adapter._accept_text_for_batch(second, session_key)
    for task in adapter._pending_text_batch_tasks.values():
        task.cancel()
    await asyncio.gather(*adapter._pending_text_batch_tasks.values(), return_exceptions=True)

    adapter._text_batch_delay_seconds = 0
    adapter._text_batch_split_delay_seconds = 0
    adapter._TEXT_BATCH_FAST_DELAY_S = 0
    adapter._TEXT_BATCH_SHORT_DELAY_S = 0
    await adapter._flush_text_batch(session_key)

    flushed = adapter.handle_message.await_args.args[0]
    assert len(flushed._telegram_inbound_obligation_ids) == 2
    assert ledger.mark_execution_started(flushed._telegram_inbound_obligation_ids)
    with patch.object(ledger, "_owner_alive", return_value=False):
        assert ledger.claim_recoverable() == []


@pytest.mark.asyncio
async def test_busy_session_pending_merge_preserves_every_obligation_until_restart_boundary(
    tmp_path, monkeypatch
):
    """Busy-session text merging must not leave a consumed suffix replayable.

    This uses the adapter's real active-session pending-message path, rather
    than the Telegram debounce batch.  The eventual runner turn receives the
    merged event and must seal every accepted fragment before a restart sweep.
    """
    import gateway.telegram_inbound_ledger as ledger

    monkeypatch.setattr(ledger, "get_hermes_home", lambda: tmp_path)
    adapter = _adapter()
    adapter.platform = Platform.TELEGRAM
    adapter._message_handler = AsyncMock()
    adapter._pending_messages = {}
    adapter._active_sessions = {}
    adapter._session_tasks = {}
    adapter._busy_session_handler = None
    adapter._busy_text_mode = "interrupt"

    session_key = "agent:main:telegram:dm:123"
    first = _event()
    first.text = "first busy fragment"
    second = _event()
    second.message_id = "7790"
    second.platform_update_id = 9913
    second.text = "second busy fragment"
    assert await adapter._journal_durable_text_event(first, session_key)
    assert await adapter._journal_durable_text_event(second, session_key)
    expected_obligation_ids = [
        *first._telegram_inbound_obligation_ids,
        *second._telegram_inbound_obligation_ids,
    ]

    # Model the first turn still running while the second accepted update
    # enters BasePlatformAdapter.handle_message's pending-message branch.
    adapter._pending_messages[session_key] = first
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter.handle_message(second)

    pending = adapter._pending_messages[session_key]
    assert pending.text == "first busy fragment\nsecond busy fragment"
    assert pending._telegram_inbound_obligation_ids == expected_obligation_ids

    # This is the runner's execution boundary: a later restart must find no
    # separately recoverable suffix after the merged turn starts.
    assert ledger.mark_execution_started(pending._telegram_inbound_obligation_ids)
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    with patch.object(ledger, "_owner_alive", return_value=False):
        assert await runner._recover_telegram_inbound_obligations() == 0


def test_restored_telegram_event_keeps_runner_context_inside_durable_payload(tmp_path, monkeypatch):
    """Recovery restores routing/context without persisting raw Telegram data."""
    import json
    import gateway.telegram_inbound_ledger as ledger

    monkeypatch.setattr(ledger, "get_hermes_home", lambda: tmp_path)
    event = _event()
    event.source.message_id = "source-message-7788"
    event.source.chat_name = "private chat"
    event.source.chat_topic = "topic context"
    # Trust and authorization are process-local and must fail closed after
    # recovery, even if a malformed durable row attempted to carry them.
    event.source.is_bot = True
    event.source.role_authorized = True
    event.reply_to_message_id = "7001"
    event.reply_to_text = "quoted reply context"
    event.reply_to_author_id = "99"
    event.reply_to_author_name = "reply author"
    event.reply_to_is_own_message = True
    event.auto_skill = ["skill-a", "skill-b"]
    event.channel_prompt = "channel-only system context"
    event.channel_context = "observed channel context"
    event.raw_message = SimpleNamespace(text="must never be stored")
    event.metadata = {"untrusted": "must never be stored"}

    obligation_id = ledger.record_accepted(event, "agent:main:telegram:dm:123")
    assert obligation_id
    with ledger._LOCK, ledger._transaction() as conn:
        encoded = conn.execute(
            "SELECT payload FROM telegram_inbound_obligations WHERE obligation_id=?",
            (obligation_id,),
        ).fetchone()[0]
    payload = json.loads(encoded)
    assert payload["reply_to_text"] == "quoted reply context"
    assert payload["channel_prompt"] == "channel-only system context"
    assert "raw_message" not in payload
    assert "metadata" not in payload

    restored = ledger.restore_event(
        {"obligation_id": obligation_id, "payload": payload}
    )
    assert restored.source.message_id == "source-message-7788"
    assert restored.source.is_bot is False
    assert restored.source.role_authorized is False
    assert restored.reply_to_message_id == "7001"
    assert restored.reply_to_text == "quoted reply context"
    assert restored.reply_to_author_id == "99"
    assert restored.reply_to_author_name == "reply author"
    assert restored.reply_to_is_own_message is True
    assert restored.auto_skill == ["skill-a", "skill-b"]
    assert restored.channel_prompt == "channel-only system context"
    assert restored.channel_context == "observed channel context"
    assert restored.raw_message is None
    assert restored.metadata == {}


def test_execution_boundary_seals_recovered_input_against_replay(tmp_path, monkeypatch):
    """One aggregated turn seals every fragment against a duplicate replay."""
    import gateway.telegram_inbound_ledger as ledger

    monkeypatch.setattr(ledger, "get_hermes_home", lambda: tmp_path)
    event = _event()
    continuation = _event()
    continuation.message_id = "7789"
    continuation.platform_update_id = 9912
    oid = ledger.record_accepted(event, "agent:main:telegram:dm:123")
    continuation_oid = ledger.record_accepted(continuation, "agent:main:telegram:dm:123")
    assert oid and continuation_oid
    assert ledger.mark_execution_started([oid, continuation_oid])
    assert not ledger.mark_execution_started([oid, continuation_oid])
    with patch.object(ledger, "_owner_alive", return_value=False):
        assert ledger.claim_recoverable() == []


def test_discarded_accepted_input_is_terminal_and_never_recovered(tmp_path, monkeypatch):
    """An intentional stop/reset/stale cleanup may not resurrect old input."""
    import gateway.telegram_inbound_ledger as ledger

    monkeypatch.setattr(ledger, "get_hermes_home", lambda: tmp_path)
    obligation_id = ledger.record_accepted(_event(), "agent:main:telegram:dm:123")
    assert obligation_id

    assert ledger.mark_discarded([obligation_id])
    with ledger._LOCK, ledger._transaction() as conn:
        assert conn.execute(
            "SELECT state FROM telegram_inbound_obligations WHERE obligation_id=?",
            (obligation_id,),
        ).fetchone()[0] == "discarded"
    with patch.object(ledger, "_owner_alive", return_value=False):
        assert ledger.claim_recoverable() == []


def test_discard_refuses_an_execution_started_input(tmp_path, monkeypatch):
    """A terminal discard cannot overwrite the at-most-once execution seal."""
    import gateway.telegram_inbound_ledger as ledger

    monkeypatch.setattr(ledger, "get_hermes_home", lambda: tmp_path)
    obligation_id = ledger.record_accepted(_event(), "agent:main:telegram:dm:123")
    assert obligation_id
    assert ledger.mark_execution_started(obligation_id)
    assert not ledger.mark_discarded([obligation_id])


@pytest.mark.asyncio
async def test_stop_and_reset_use_terminal_adapter_discard_for_pending_input():
    """The shared stop/new/reset funnel must not merely pop Telegram work."""
    runner = object.__new__(GatewayRunner)
    discarded = []
    adapter = SimpleNamespace(
        _discard_pending_message=lambda session_key: discarded.append(session_key) or True
    )
    runner._peek_session_state = lambda _session_key: None
    runner._invalidate_session_run_generation = lambda _session_key, *, reason: 1
    runner._adapter_for_source = lambda _source: adapter
    session_key = "agent:main:telegram:dm:123"

    await runner._interrupt_and_clear_session(
        session_key,
        _event().source,
        interrupt_reason="stop",
        invalidation_reason="stop_command",
        release_running_state=False,
    )

    assert discarded == [session_key]


def test_restored_telegram_media_reply_keeps_cached_media_and_effective_type(tmp_path, monkeypatch):
    """Recovery must preserve the agent-visible attachment representation."""
    import json
    import gateway.telegram_inbound_ledger as ledger

    monkeypatch.setattr(ledger, "get_hermes_home", lambda: tmp_path)
    event = _event()
    event.message_type = MessageType.PHOTO
    event.text = "please inspect this image"
    event.media_urls = ["/tmp/hermes/cache/inbound-photo.jpg"]
    event.media_types = ["image/jpeg"]
    obligation_id = ledger.record_accepted(event, "agent:main:telegram:dm:123")
    assert obligation_id

    with ledger._LOCK, ledger._transaction() as conn:
        payload = json.loads(conn.execute(
            "SELECT payload FROM telegram_inbound_obligations WHERE obligation_id=?",
            (obligation_id,),
        ).fetchone()[0])
    restored = ledger.restore_event({"obligation_id": obligation_id, "payload": payload})

    assert restored.message_type is MessageType.PHOTO
    assert restored.media_urls == ["/tmp/hermes/cache/inbound-photo.jpg"]
    assert restored.media_types == ["image/jpeg"]
