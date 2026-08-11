"""Tests for the resume_pending session continuity path.

Covers the behaviour introduced to fix the ``Gateway shutting down ...
task will be interrupted`` follow-up bug (spec: PR #11852, builds on
PRs #9850, #9934, #7536):

1. When a gateway restart drain times out and agents are force-interrupted,
   the affected sessions are flagged ``resume_pending=True`` — not
   ``suspended`` — so the next user message on the same session_key
   auto-resumes from the existing transcript instead of getting routed
   through ``suspend_recently_active()`` and converted into a fresh
   session.

2. ``suspended=True`` (from ``/stop`` or stuck-loop escalation) still
   wins over ``resume_pending`` — the forced-wipe path is preserved.

3. The restart-resume system note injected into the next user message is
   a superset of the existing tool-tail auto-continue note (from
   PR #9934), using session-entry metadata rather than just transcript
   shape so it fires even when the interrupted transcript does NOT end
   with a ``tool`` role.

4. The existing ``.restart_failure_counts`` stuck-loop counter from
   PR #7536 remains the single source of escalation — no parallel
   counter is added on ``SessionEntry``.
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import (
    _AGENT_PENDING_SENTINEL,
    _auto_continue_freshness_window,
    _coerce_gateway_timestamp,
    _is_fresh_gateway_interruption,
    _last_transcript_timestamp,
    _should_inject_resume_pending,
    _should_clear_resume_pending_after_turn,
    build_resume_recovery_note,
)
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from tests.gateway.restart_test_helpers import (
    make_restart_runner,
    make_restart_source,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_resume_pending_is_cleared_only_after_successful_turn():
    """Interrupted/failed drain results must keep the restart recovery marker.

    Regression for dogfood failure: during gateway restart the interrupted run
    returned an empty final response and was normalized into a user-facing
    fallback, but the gateway cleared ``resume_pending`` before startup could
    auto-resume it.
    """
    assert _should_clear_resume_pending_after_turn({"final_response": "done"}) is True
    assert _should_clear_resume_pending_after_turn({"completed": True}) is True
    assert _should_clear_resume_pending_after_turn({"interrupted": True}) is False
    assert _should_clear_resume_pending_after_turn({"completed": False}) is False
    assert _should_clear_resume_pending_after_turn({"failed": True}) is False
    assert _should_clear_resume_pending_after_turn({"partial": True}) is False
    assert _should_clear_resume_pending_after_turn({"error": "boom"}) is False


def test_current_turn_write_ahead_marker_does_not_fake_a_restart():
    """The marker written by this turn is for the *next process* only.

    A normal inbound message must not receive restart-recovery guidance merely
    because the write-ahead marker became visible before the agent thread read
    the session entry.  A marker inherited from the previous process still
    activates recovery.
    """
    entry = MagicMock(resume_pending=True)
    assert _should_inject_resume_pending(
        entry, inherited_at_turn_start=False, interruption_is_fresh=True,
        resume_mark_is_fresh=True,
    ) is False
    assert _should_inject_resume_pending(
        entry, inherited_at_turn_start=True, interruption_is_fresh=True,
        resume_mark_is_fresh=True,
    ) is True


@pytest.mark.asyncio
async def test_delivery_handoff_requires_and_consumes_success_authorization():
    runner, _adapter = make_restart_runner()
    clear_pending = AsyncMock(return_value=True)
    runner._async_session_store = MagicMock()
    runner._async_session_store._store = runner.session_store
    runner._async_session_store.clear_resume_pending = clear_pending
    session_key = "agent:main:telegram:dm:123"
    turn_a = "turn-a"
    turn_b = "turn-b"

    assert (
        await runner._complete_resume_delivery_handoff(session_key, turn_a) is False
    )
    clear_pending.assert_not_awaited()

    runner._resume_delivery_handoff_ready[turn_a] = session_key
    assert (
        await runner._complete_resume_delivery_handoff(session_key, turn_b) is False
    )
    clear_pending.assert_not_awaited()

    assert await runner._complete_resume_delivery_handoff(session_key, turn_a) is True
    clear_pending.assert_awaited_once_with(session_key)
    assert turn_a not in runner._resume_delivery_handoff_ready


@pytest.mark.asyncio
async def test_command_is_written_to_inbound_wal_before_dispatch():
    from gateway import delivery_ledger as dl

    runner, _adapter = make_restart_runner()
    source = make_restart_source(chat_id="command-chat")
    event = MessageEvent(
        text="/queue inspect",
        message_type=MessageType.TEXT,
        source=source,
        message_id="command-message-1",
    )
    session_key = "agent:main:telegram:dm:command-chat"

    turn_id = await runner._prepare_inbound_turn(event, session_key)

    assert turn_id
    with dl._connect() as conn:
        row = conn.execute(
            "SELECT session_key, state, payload FROM inbound_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
    assert row is not None
    assert row[0:2] == (session_key, "received")
    assert '\"text\":\"/queue inspect\"' in row[2]


@pytest.mark.asyncio
async def test_disabled_delivery_ledger_skips_inbound_wal_before_dispatch(monkeypatch):
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: False)
    runner, _adapter = make_restart_runner()
    source = make_restart_source(chat_id="legacy-chat")
    event = MessageEvent(
        text="legacy delivery",
        message_type=MessageType.TEXT,
        source=source,
        message_id="legacy-message-1",
    )

    turn_id = await runner._prepare_inbound_turn(
        event, "agent:main:telegram:dm:legacy-chat"
    )

    assert turn_id is None
    assert "_hermes_turn_id" not in event.metadata
    assert not dl._db_path().exists()


@pytest.mark.asyncio
async def test_completed_platform_event_never_reaches_hooks_or_agent(
    tmp_path, monkeypatch
):
    from gateway import delivery_ledger as dl
    from gateway.run import GatewayRunner

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    runner, adapter = make_restart_runner()
    adapter.gateway_runner = runner
    runner._prepare_inbound_turn = GatewayRunner._prepare_inbound_turn.__get__(
        runner, GatewayRunner
    )
    source = make_restart_source(chat_id="dedup-chat")
    first = MessageEvent(
        text="once",
        message_type=MessageType.TEXT,
        source=source,
        message_id="platform-event-1",
    )
    session_key = build_session_key(source)
    turn_id = await runner._prepare_inbound_turn(first, session_key)
    assert turn_id is not None
    dl.mark_inbound_turn_completed(turn_id, session_key=session_key)
    hooks = AsyncMock()

    async def run_agent(event):
        await hooks(event)

    agent = AsyncMock(side_effect=run_agent)
    adapter.set_message_handler(agent)
    duplicate = MessageEvent(
        text="once",
        message_type=MessageType.TEXT,
        source=source,
        message_id="platform-event-1",
    )

    await adapter.handle_message(duplicate)
    await asyncio.sleep(0)

    hooks.assert_not_awaited()
    agent.assert_not_awaited()
    assert adapter._session_tasks == {}


@pytest.mark.asyncio
async def test_simultaneous_platform_redelivery_runs_only_one_agent(
    tmp_path, monkeypatch
):
    from gateway import delivery_ledger as dl
    from gateway.run import GatewayRunner

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    runner, adapter = make_restart_runner()
    adapter.gateway_runner = runner
    runner._prepare_inbound_turn = GatewayRunner._prepare_inbound_turn.__get__(
        runner, GatewayRunner
    )
    source = make_restart_source(chat_id="race-chat")
    events = [
        MessageEvent(
            text="once",
            message_type=MessageType.TEXT,
            source=source,
            message_id="platform-race-1",
        )
        for _ in range(2)
    ]
    session_key = build_session_key(source)

    turn_ids = await asyncio.gather(
        *(runner._prepare_inbound_turn(event, session_key) for event in events)
    )

    assert len([turn_id for turn_id in turn_ids if turn_id]) == 1
    assert sum(bool(event.metadata.get("_hermes_duplicate_inbound")) for event in events) == 1


@pytest.mark.asyncio
async def test_completed_event_stays_deduplicated_after_session_alias_rebind(
    tmp_path, monkeypatch
):
    from gateway import delivery_ledger as dl
    from gateway.run import GatewayRunner

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    runner, _adapter = make_restart_runner()
    runner._prepare_inbound_turn = GatewayRunner._prepare_inbound_turn.__get__(
        runner, GatewayRunner
    )
    source = make_restart_source(chat_id="alias-chat")
    quick_key = build_session_key(source)
    canonical_key = f"{quick_key}:topic:canonical"
    first = MessageEvent(
        text="once",
        message_type=MessageType.TEXT,
        source=source,
        message_id="platform-alias-1",
    )
    turn_id = await runner._prepare_inbound_turn(first, quick_key)
    assert turn_id is not None
    dl.rebind_inbound_turn(turn_id, canonical_key)
    dl.mark_inbound_turn_completed(turn_id, session_key=canonical_key)
    runner._continuity_session_aliases = {quick_key: canonical_key}
    duplicate = MessageEvent(
        text="once",
        message_type=MessageType.TEXT,
        source=source,
        message_id="platform-alias-1",
    )

    duplicate_turn_id = await runner._prepare_inbound_turn(duplicate, quick_key)

    assert duplicate_turn_id is None
    assert duplicate.metadata["_hermes_duplicate_inbound"] is True
    with dl._connect() as conn:
        rows = conn.execute("SELECT turn_id, state FROM inbound_turns").fetchall()
    assert rows == [(turn_id, "completed")]


@pytest.mark.parametrize("kind", ["image", "tts", "voice", "video", "document"])
@pytest.mark.asyncio
async def test_ambiguous_multimedia_redelivery_emits_visible_marker_before_media(
    tmp_path, monkeypatch, kind
):
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    monkeypatch.setattr(dl, "_owner_alive", lambda *_args: False)
    runner, adapter = make_restart_runner()
    turn_id = f"turn-{kind}"
    path = tmp_path / f"artifact-{kind}"
    path.write_bytes(b"artifact")
    payload = (
        {"url": "https://example.test/image.png", "alt": "diagram"}
        if kind == "image"
        else {"path": str(path), "caption": "answer"}
    )
    session_key = f"agent:main:telegram:dm:{kind}"
    dl.record_inbound_turn(
        turn_id=turn_id,
        session_key=session_key,
        platform="telegram",
        chat_id=kind,
        thread_id=None,
        payload={"text": "work"},
    )
    component_id = dl.record_delivery_plan(
        turn_id=turn_id,
        session_key=session_key,
        platform="telegram",
        chat_id=kind,
        thread_id=None,
        components=[{"kind": kind, "payload": payload}],
    )[0]
    dl.mark_delivery_component_failed(component_id, "gateway died mid-send")
    calls = []

    async def marker(**kwargs):
        calls.append(("marker", kwargs["content"]))
        return SendResult(success=True, message_id="marker-ack")

    adapter._send_with_retry = marker
    adapter._send_image_with_ack = AsyncMock(
        side_effect=lambda **_kwargs: calls.append(("image", "sent"))
        or SendResult(success=True, message_id="media-ack")
    )
    adapter.play_tts = AsyncMock(
        side_effect=lambda **_kwargs: calls.append(("tts", "sent"))
        or SendResult(success=True, message_id="media-ack")
    )
    adapter.send_voice = AsyncMock(
        side_effect=lambda **_kwargs: calls.append(("voice", "sent"))
        or SendResult(success=True, message_id="media-ack")
    )
    adapter.send_video = AsyncMock(
        side_effect=lambda **_kwargs: calls.append(("video", "sent"))
        or SendResult(success=True, message_id="media-ack")
    )
    adapter.send_document = AsyncMock(
        side_effect=lambda **_kwargs: calls.append(("document", "sent"))
        or SendResult(success=True, message_id="media-ack")
    )

    recovered = await runner._redeliver_pending_delivery_components()

    assert recovered == 1
    assert calls[0][0] == "marker"
    assert "Recovered reply" in calls[0][1]
    assert calls[1][0] == kind


@pytest.mark.asyncio
async def test_ambiguous_media_marker_failure_blocks_media_and_keeps_obligation(
    tmp_path, monkeypatch
):
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    monkeypatch.setattr(dl, "_owner_alive", lambda *_args: False)
    runner, adapter = make_restart_runner()
    session_key = "agent:main:telegram:dm:image-blocked"
    dl.record_inbound_turn(
        turn_id="turn-image-blocked",
        session_key=session_key,
        platform="telegram",
        chat_id="image-blocked",
        thread_id=None,
        payload={"text": "work"},
    )
    component_id = dl.record_delivery_plan(
        turn_id="turn-image-blocked",
        session_key=session_key,
        platform="telegram",
        chat_id="image-blocked",
        thread_id=None,
        components=[
            {
                "kind": "image",
                "payload": {"url": "https://example.test/image.png", "alt": ""},
            }
        ],
    )[0]
    dl.mark_delivery_component_failed(component_id, "gateway died mid-send")
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=False, error="marker rejected")
    )
    adapter._send_image_with_ack = AsyncMock()

    recovered = await runner._redeliver_pending_delivery_components()

    assert recovered == 0
    adapter._send_image_with_ack.assert_not_awaited()
    with dl._connect() as conn:
        state = conn.execute(
            "SELECT state FROM delivery_components WHERE component_id=?",
            (component_id,),
        ).fetchone()[0]
    assert state == "failed"


def _make_source(platform=Platform.TELEGRAM, chat_id="123", user_id="u1"):
    return SessionSource(platform=platform, chat_id=chat_id, user_id=user_id)


def _make_store(tmp_path):
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _build_agent_history(history: list) -> list:
    """Mirror gateway/run.py's ``history → agent_history`` conversion.

    This is the transformation that strips ``timestamp`` off tool/tool_call
    rows before the agent sees them.  Tests that check the freshness gate
    must go through this conversion so they exercise the *real* data the
    note-injection code sees.
    """
    agent_history: list = []
    for msg in history:
        role = msg.get("role")
        if not role or role in {"session_meta", "system"}:
            continue
        has_tool_calls = "tool_calls" in msg
        has_tool_call_id = "tool_call_id" in msg
        is_tool_message = role == "tool"
        if has_tool_calls or has_tool_call_id or is_tool_message:
            agent_history.append({k: v for k, v in msg.items() if k != "timestamp"})
        else:
            content = msg.get("content")
            if content:
                agent_history.append({"role": role, "content": content})
    return agent_history


def _simulate_note_injection(
    history: list,
    user_message: str,
    resume_entry: SessionEntry | None,
    *,
    agent_history: list | None = None,
    window_secs: float | None = None,
) -> str:
    """Mirror the note-injection logic in gateway/run.py _run_agent().

    The freshness signal reads ``history[-1].timestamp`` (the raw transcript
    row), NOT ``agent_history[-1].timestamp`` (which has been stripped).
    Tests pass the raw ``history`` — ``agent_history`` is derived from it
    via the real conversion if not supplied explicitly.
    """
    if agent_history is None:
        agent_history = _build_agent_history(history)

    window = (
        float(window_secs)
        if window_secs is not None
        else _auto_continue_freshness_window()
    )
    interruption_is_fresh = _is_fresh_gateway_interruption(
        _last_transcript_timestamp(history),
        window_secs=window,
    )

    message = user_message
    resume_mark_is_fresh = False
    if resume_entry is not None and getattr(resume_entry, "resume_pending", False):
        resume_mark_is_fresh = _is_fresh_gateway_interruption(
            getattr(resume_entry, "last_resume_marked_at", None),
            window_secs=window,
        )
    is_resume_pending = bool(
        resume_entry is not None
        and getattr(resume_entry, "resume_pending", False)
        and (interruption_is_fresh or resume_mark_is_fresh)
    )
    has_fresh_tool_tail = bool(
        agent_history
        and agent_history[-1].get("role") == "tool"
        and interruption_is_fresh
    )

    if is_resume_pending:
        reason = getattr(resume_entry, "resume_reason", None) or "restart_timeout"
        # Real production note builder — extracted to module scope in
        # gateway/run.py so tests exercise the actual strings.
        message = build_resume_recovery_note(reason, message)
    elif has_fresh_tool_tail:
        message = (
            "[System note: A new message has arrived. The conversation "
            "history contains pending tool outputs from an interrupted turn. "
            "IGNORE those pending results. Address the user's NEW message "
            "below FIRST. Do NOT re-execute old tool calls from the history.]\n\n"
            + message
        )

    # Empty-turn safety net: mirrors gateway/run.py — a blank
    # auto-resume turn on a resume_pending session must never reach the model.
    if (
        isinstance(message, str)
        and not message.strip()
        and resume_entry is not None
        and getattr(resume_entry, "resume_pending", False)
    ):
        sn_reason = getattr(resume_entry, "resume_reason", None) or "restart_timeout"
        message = build_resume_recovery_note(sn_reason, "")
    return message


# ---------------------------------------------------------------------------
# SessionEntry field + serialization
# ---------------------------------------------------------------------------


class TestSessionEntryResumeFields:
    def test_defaults(self):
        now = datetime.now()
        entry = SessionEntry(
            session_key="agent:main:telegram:dm:1",
            session_id="sid",
            created_at=now,
            updated_at=now,
        )
        assert entry.resume_pending is False
        assert entry.resume_reason is None
        assert entry.last_resume_marked_at is None


# ---------------------------------------------------------------------------
# SessionStore.mark_resume_pending / clear_resume_pending
# ---------------------------------------------------------------------------


class TestMarkResumePending:
    def test_marks_existing_session(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)

        assert store.mark_resume_pending(entry.session_key) is True
        refreshed = store._entries[entry.session_key]
        assert refreshed.resume_pending is True
        assert refreshed.resume_reason == "restart_timeout"
        assert refreshed.last_resume_marked_at is not None

    def test_custom_reason_persists(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)

        store.mark_resume_pending(entry.session_key, reason="shutdown_timeout")
        assert store._entries[entry.session_key].resume_reason == "shutdown_timeout"


class TestClearResumePending:

    def test_returns_false_when_not_pending(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        # Not marked
        assert store.clear_resume_pending(entry.session_key) is False


# ---------------------------------------------------------------------------
# SessionStore.get_or_create_session resume_pending behaviour
# ---------------------------------------------------------------------------


class TestGetOrCreateResumePending:

    def test_resume_pending_follows_compression_tip(self, tmp_path):
        """Interrupted platform mappings must not stay pinned to compressed roots."""
        store = _make_store(tmp_path)
        source = _make_source(
            platform=Platform.WEIXIN,
            chat_id="wx-chat",
            user_id="wx-user",
        )
        first = store.get_or_create_session(source)
        original_sid = first.session_id
        store.mark_resume_pending(first.session_key)

        with patch.object(
            store, "_compression_tip_for_session_id", return_value="child-session"
        ) as mock_tip:
            second = store.get_or_create_session(source)

        assert second.session_id == "child-session"
        assert second.resume_pending is True
        mock_tip.assert_called_with(original_sid)


# ---------------------------------------------------------------------------
# SessionStore.suspend_recently_active skip behaviour
# ---------------------------------------------------------------------------


class TestSuspendRecentlyActiveSkipsResumePending:
    def test_resume_pending_entries_not_suspended(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.mark_resume_pending(entry.session_key)

        count = store.suspend_recently_active()
        assert count == 0
        e = store._entries[entry.session_key]
        assert e.suspended is False
        assert e.resume_pending is True


# ---------------------------------------------------------------------------
# Restart-resume system-note injection
# ---------------------------------------------------------------------------


class TestResumePendingSystemNote:
    def _pending_entry(self, reason="restart_timeout") -> SessionEntry:
        now = datetime.now()
        return SessionEntry(
            session_key="agent:main:telegram:dm:1",
            session_id="sid",
            created_at=now,
            updated_at=now,
            resume_pending=True,
            resume_reason=reason,
            last_resume_marked_at=now,
        )


    def test_empty_message_noninteractive_note_continues_task(self):
        """Non-interactive platforms (webhook, API server): nobody can answer
        'what next?', so the resumed turn must complete the interrupted work
        instead of acknowledging (#57056)."""
        note = build_resume_recovery_note("restart_timeout", "", interactive=False)
        assert "CONTINUE the interrupted task" in note
        assert "session was restored" not in note
        assert "ask what they would like to do next" not in note
        # Must not tell the model to skip the unfinished work it should finish.
        assert "skip any unfinished work" not in note
        # But still guards against re-running already-recorded tool calls.
        assert "already appear in the history" in note

    def test_empty_message_interactive_note_continues_without_human_reconstruction(self):
        note = build_resume_recovery_note(
            "restart_timeout", "", interactive=True
        )
        assert "CONTINUE the interrupted task" in note
        assert "ask what they would like to do next" not in note
        assert "inspect current state with read-only tools" in note
        assert "do not ask the user to reconstruct" in note


    def test_resume_pending_fires_without_tool_tail(self):
        """Key improvement over PR #9934: the restart-resume note fires
        even when the transcript's last role is NOT ``tool``."""
        entry = self._pending_entry()
        history = [
            {"role": "user", "content": "run a long thing", "timestamp": time.time() - 10},
            {"role": "assistant", "content": "ok, starting...", "timestamp": time.time()},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=entry)
        assert "[System note:" in result
        assert "gateway restart" in result
        assert "NEW message" in result


    def test_no_resume_pending_preserves_tool_tail_note(self):
        """Regression: the old PR #9934 tool-tail behaviour is unchanged."""
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ], "timestamp": time.time() - 1},
            {"role": "tool", "tool_call_id": "c1", "content": "result",
             "timestamp": time.time()},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=None)
        assert "[System note:" in result
        assert "pending tool outputs" in result
        assert "Do NOT re-execute" in result

    def test_stale_resume_pending_does_not_inject_restart_note(self):
        """Old restart markers must not revive an unrelated stale task.

        The transcript's last row is from an hour ago — well outside the
        default 1h freshness window (fixture uses window=1800 to exercise
        the stale path without tying the test to the production default).
        """
        entry = self._pending_entry()
        entry.last_resume_marked_at = datetime.now() - timedelta(hours=1)

        history = [
            {"role": "assistant", "content": "old in progress",
             "timestamp": time.time() - 3600},
        ]
        result = _simulate_note_injection(
            history=history,
            user_message="start a new task",
            resume_entry=entry,
            window_secs=1800,
        )
        assert result == "start a new task"


    def test_stale_tool_tail_does_not_inject_auto_continue_note(self):
        """The core bug fix: stale tool-tail must not revive a dead task.

        Uses window_secs=1800 (30 min) to verify the gate fires at 1h —
        keeps the test stable regardless of the production default.
        """
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ], "timestamp": time.time() - 3601},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "stale result",
                "timestamp": time.time() - 3600,
            },
        ]
        result = _simulate_note_injection(
            history,
            "start a new task",
            resume_entry=None,
            window_secs=1800,
        )
        assert result == "start a new task"

    def test_stale_tool_tail_with_production_data_shape(self):
        """Regression guard for #16802: exercise the REAL production path
        where ``agent_history`` has been stripped of timestamps.

        The original PR #16802 fix read ``agent_history[-1].get("timestamp")``
        — which is always ``None`` at runtime because the gateway strips
        ``timestamp`` off tool/tool_call rows in ``history → agent_history``.
        This test builds a stale history, runs it through the real
        ``_build_agent_history`` conversion, then asserts:

          1. The stripped ``agent_history`` carries NO timestamp (protects
             against someone "fixing" the original PR by re-adding the
             stripped field — which would break the API contract).
          2. The freshness gate still correctly classifies the transcript
             as stale because the signal is read from ``history`` BEFORE
             the strip.
          3. No auto-continue note is injected.
        """
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ], "timestamp": time.time() - 7201},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "stale result",
                "timestamp": time.time() - 7200,  # 2 hours old
            },
        ]
        agent_history = _build_agent_history(history)

        # Invariant 1: strip contract preserved
        assert agent_history[-1]["role"] == "tool"
        assert "timestamp" not in agent_history[-1], (
            "agent_history tool rows must NOT carry a timestamp — the "
            "freshness gate must read from raw history, not agent_history"
        )

        # Invariant 2+3: stale classification, no note injection
        result = _simulate_note_injection(
            history,
            "start a new task",
            resume_entry=None,
            agent_history=agent_history,
        )
        assert result == "start a new task"

    def test_freshness_gate_disabled_via_zero_window(self):
        """window_secs=0 restores pre-fix behaviour (always inject)."""
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ], "timestamp": time.time() - 86400},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "day-old result",
                "timestamp": time.time() - 86400,  # 24 hours old
            },
        ]
        result = _simulate_note_injection(
            history, "ping", resume_entry=None, window_secs=0,
        )
        assert "[System note:" in result
        assert "pending tool outputs" in result
        assert "Do NOT re-execute" in result

    def test_legacy_history_without_timestamps_still_injects(self):
        """Transcripts predating timestamp persistence must keep the old
        behaviour — freshness unknown → treat as fresh."""
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=None)
        assert "[System note:" in result
        assert "pending tool outputs" in result
        assert "Do NOT re-execute" in result


# ---------------------------------------------------------------------------
# Freshness helpers
# ---------------------------------------------------------------------------


class TestFreshnessHelpers:


    def test_coerce_iso_string(self):
        iso = "2026-04-18T12:00:00+00:00"
        expected = datetime.fromisoformat(iso).timestamp()
        assert _coerce_gateway_timestamp(iso) == pytest.approx(expected, abs=1e-3)


    def test_coerce_rejects_garbage(self):
        assert _coerce_gateway_timestamp(None) is None
        assert _coerce_gateway_timestamp("") is None
        assert _coerce_gateway_timestamp("not-a-timestamp") is None
        assert _coerce_gateway_timestamp(True) is None  # bool rejected
        assert _coerce_gateway_timestamp(False) is None
        assert _coerce_gateway_timestamp([1, 2, 3]) is None


    def test_is_fresh_window_bounds(self):
        now = 1_700_000_000.0
        # 1h window, 30min old → fresh
        assert _is_fresh_gateway_interruption(
            now - 1800, now=now, window_secs=3600,
        ) is True
        # 1h window, 2h old → stale
        assert _is_fresh_gateway_interruption(
            now - 7200, now=now, window_secs=3600,
        ) is False
        # 1h window, exactly at boundary → fresh (<=)
        assert _is_fresh_gateway_interruption(
            now - 3600, now=now, window_secs=3600,
        ) is True


    def test_last_transcript_timestamp_skips_meta(self):
        history = [
            {"role": "user", "content": "hi", "timestamp": 100.0},
            {"role": "assistant", "content": "hey", "timestamp": 200.0},
            {"role": "session_meta", "content": "tools:{}", "timestamp": 999.0},
            {"role": "system", "content": "ignore", "timestamp": 999.0},
        ]
        assert _last_transcript_timestamp(history) == 200.0


    def test_auto_continue_freshness_window_reads_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "7200")
        assert _auto_continue_freshness_window() == 7200.0

    def test_auto_continue_freshness_window_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_AUTO_CONTINUE_FRESHNESS", raising=False)
        # Default is 1 hour
        assert _auto_continue_freshness_window() == 3600.0


# ---------------------------------------------------------------------------
# Drain-timeout path marks sessions resume_pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_timeout_marks_resume_pending():
    """End-to-end: a drain timeout during gateway stop should flag every
    active session as resume_pending BEFORE the interrupt fires, so the
    next startup's suspend_recently_active() does not destroy them."""
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()
    runner._restart_drain_timeout = 0.05

    running_agent = MagicMock()
    session_key_one = "agent:main:telegram:dm:A"
    session_key_two = "agent:main:telegram:dm:B"
    runner._running_agents = {
        session_key_one: running_agent,
        session_key_two: MagicMock(),
    }

    # Plug a mock session_store that records marks.
    session_store = MagicMock()
    session_store.mark_resume_pending = MagicMock(return_value=True)
    runner.session_store = session_store

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.write_runtime_status"
    ):
        await runner.stop()

    # Both active sessions were marked with the shutdown_timeout reason.
    calls = session_store.mark_resume_pending.call_args_list
    marked = {args[0][0] for args in calls}
    assert marked == {session_key_one, session_key_two}
    for args in calls:
        assert args[0][1] == "shutdown_timeout"


# ---------------------------------------------------------------------------
# Abrupt-crash active-turn durability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_turn_is_marked_durable_before_session_start_hook(tmp_path):
    """A SIGKILL can arrive before graceful shutdown gets a chance to mark work.

    The active session therefore must be persisted as resume-pending immediately
    after its canonical session entry is resolved and before any hook or agent
    work can create an external effect.  A fresh SessionStore proves the marker
    reached disk rather than merely changing an in-memory object.
    """
    from gateway.run import GatewayRunner

    runner, _adapter = make_restart_runner()
    source = make_restart_source(chat_id="abrupt-crash-chat")
    session_key = runner._session_key_for_source(source)
    runner.session_store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    entry = runner.session_store.get_or_create_session(source)
    assert entry.session_key == session_key
    runner._resume_delivery_handoff_ready["stale-turn"] = session_key
    runner._recover_telegram_topic_thread_id = lambda source: None
    runner._is_telegram_topic_lane = lambda source: False
    runner.hooks.emit = AsyncMock(side_effect=RuntimeError("stop-after-durable-mark"))

    await GatewayRunner._handle_message_with_agent(
        runner, MessageEvent(text="continue", message_type=MessageType.TEXT, source=source),
        source, session_key, 1,
    )

    reloaded = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    recovered = reloaded.get_or_create_session(source)
    assert recovered.session_key == session_key
    assert recovered.resume_pending is True
    assert recovered.resume_reason == "turn_in_progress"
    assert "stale-turn" not in runner._resume_delivery_handoff_ready


@pytest.mark.asyncio
async def test_canonical_boundary_rebinds_every_represented_inbound_turn():
    from gateway.run import GatewayRunner

    runner, _adapter = make_restart_runner()
    source = make_restart_source(chat_id="canonical-rebind")
    quick = "agent:main:telegram:dm:canonical-rebind"
    canonical = "agent:surgical:telegram:dm:canonical-rebind"
    entry = SessionEntry(
        session_key=canonical,
        session_id="sid-canonical",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.mark_resume_pending.return_value = True
    runner._async_session_store = None
    runner._recover_telegram_topic_thread_id = lambda source: None
    runner._is_telegram_topic_lane = lambda source: False
    runner.hooks.emit = AsyncMock(side_effect=RuntimeError("stop-after-rebind"))
    event = MessageEvent(
        text="combined",
        message_type=MessageType.TEXT,
        source=source,
        metadata={
            "_hermes_turn_id": "turn-primary",
            "_hermes_inbound_turn_ids": ["turn-primary", "turn-followup"],
        },
    )

    with patch("gateway.delivery_ledger.rebind_inbound_turns") as rebind:
        await GatewayRunner._handle_message_with_agent(
            runner, event, source, quick, 1
        )

    rebind.assert_called_once_with(
        ["turn-primary", "turn-followup"],
        expected_session_key=quick,
        canonical_session_key=canonical,
    )
    assert event.metadata["_hermes_canonical_session_key"] == canonical


@pytest.mark.asyncio
async def test_startup_auto_resumes_turn_in_progress_after_abrupt_crash():
    """A fresh process must synthesize exactly one continuation for an active
    turn marker left behind by an ungraceful process death."""
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="abrupt-crash-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:abrupt-crash-chat",
        session_id="sid-abrupt",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="turn_in_progress",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio.sleep(0)

    assert scheduled == 1
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_auto_resume_skips_session_owned_by_delivery_plan():
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="ledger-owned-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:ledger-owned-chat",
        session_id="sid-ledger-owned",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="turn_in_progress",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    with patch(
        "gateway.delivery_ledger.session_has_delivery_plan",
        return_value=True,
    ):
        scheduled = runner._schedule_resume_pending_sessions()
        await asyncio.sleep(0)

    assert scheduled == 0
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_delivery_ledger_ignores_historical_plan_for_auto_resume():
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="legacy-resume-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:legacy-resume-chat",
        session_id="sid-legacy-resume",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="turn_in_progress",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    with (
        patch("gateway.delivery_ledger.ledger_enabled", return_value=False),
        patch(
            "gateway.delivery_ledger.session_has_delivery_plan",
            return_value=True,
        ) as has_plan,
    ):
        scheduled = runner._schedule_resume_pending_sessions()
        await asyncio.sleep(0)

    assert scheduled == 1
    has_plan.assert_not_called()
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_terminal_inline_command_plan_preserves_parent_resume_marker(
    tmp_path, monkeypatch
):
    """A bypass command delivery must not finalize another active turn's marker."""
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    monkeypatch.setattr(dl, "_owner_alive", lambda *_args: False)
    runner, _adapter = make_restart_runner()
    session_key = "agent:main:telegram:dm:inline-command-chat"
    turn_id = "inline-command-turn"
    dl.record_inbound_turn(
        turn_id=turn_id,
        session_key=session_key,
        platform="telegram",
        chat_id="inline-command-chat",
        thread_id=None,
        payload={"text": "/status"},
    )
    component_id = dl.record_delivery_plan(
        turn_id=turn_id,
        session_key=session_key,
        platform="telegram",
        chat_id="inline-command-chat",
        thread_id=None,
        components=[
            {
                "kind": "text",
                "payload": {
                    "content": "status ok",
                    "preserve_resume_pending": True,
                },
            }
        ],
    )[0]
    dl.mark_delivery_component_delivered(component_id)
    clear_pending = AsyncMock(return_value=True)
    runner._async_session_store = MagicMock()
    runner._async_session_store.clear_resume_pending = clear_pending

    recovered = await runner._recover_inbound_turns()

    assert recovered == 1
    clear_pending.assert_not_awaited()
    with dl._connect() as conn:
        state = conn.execute(
            "SELECT state FROM inbound_turns WHERE turn_id=?", (turn_id,)
        ).fetchone()[0]
    assert state == "completed"


@pytest.mark.asyncio
async def test_runner_completion_binds_all_represented_turns_to_canonical_session(
    tmp_path, monkeypatch
):
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    runner, _adapter = make_restart_runner()
    canonical = "agent:surgical:telegram:dm:completion-owner"
    other = "agent:other:telegram:dm:completion-owner"
    for turn_id, session_key in (
        ("turn-primary", canonical),
        ("turn-followup", other),
    ):
        dl.record_inbound_turn(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="completion-owner",
            thread_id=None,
            payload={"text": turn_id},
        )
    event = MessageEvent(
        text="combined",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="completion-owner"),
        metadata={
            "_hermes_turn_id": "turn-primary",
            "_hermes_inbound_turn_ids": ["turn-primary", "turn-followup"],
            "_hermes_canonical_session_key": canonical,
        },
    )

    with pytest.raises(ValueError, match="session ownership mismatch"):
        await runner._complete_inbound_turn(event)

    with dl._connect() as conn:
        states = dict(conn.execute(
            "SELECT turn_id, state FROM inbound_turns ORDER BY turn_id"
        ).fetchall())
    assert states == {
        "turn-followup": "received",
        "turn-primary": "received",
    }


@pytest.mark.asyncio
async def test_redelivered_inline_command_plan_preserves_parent_resume_marker(
    tmp_path, monkeypatch
):
    """A recovered bypass reply must not consume the interrupted parent marker."""
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    monkeypatch.setattr(dl, "_owner_alive", lambda *_args: False)
    runner, adapter = make_restart_runner()
    turn_id = "inline-command-redelivery"
    session_key = "agent:main:telegram:dm:inline-redelivery-chat"
    dl.record_inbound_turn(
        turn_id=turn_id,
        session_key=session_key,
        platform="telegram",
        chat_id="inline-redelivery-chat",
        thread_id=None,
        payload={"text": "/status"},
    )
    component_id = dl.record_delivery_plan(
        turn_id=turn_id,
        session_key=session_key,
        platform="telegram",
        chat_id="inline-redelivery-chat",
        thread_id=None,
        components=[
            {
                "kind": "text",
                "payload": {
                    "content": "status ok",
                    "preserve_resume_pending": True,
                },
            }
        ],
    )[0]
    dl.mark_delivery_component_failed(component_id, "gateway died")
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="recovered-ack")
    )
    clear_pending = AsyncMock(return_value=True)
    runner._async_session_store = MagicMock()
    runner._async_session_store.clear_resume_pending = clear_pending

    recovered = await runner._redeliver_pending_delivery_components()

    assert recovered == 1
    clear_pending.assert_not_awaited()
    with dl._connect() as conn:
        states = conn.execute(
            "SELECT "
            "(SELECT state FROM inbound_turns WHERE turn_id=?), "
            "(SELECT state FROM delivery_components WHERE component_id=?)",
            (turn_id, component_id),
        ).fetchone()
    assert states == ("completed", "delivered")


@pytest.mark.asyncio
async def test_restart_delivers_ready_post_delivery_effect_exactly_once(
    tmp_path, monkeypatch
):
    """A persisted post-delivery notice survives restart without a closure."""
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    monkeypatch.setattr(dl, "_owner_alive", lambda *_args: False)
    runner, adapter = make_restart_runner()
    turn_id = "turn-durable-effect"
    session_key = "agent:main:telegram:dm:effect-chat"
    dl.record_inbound_turn(
        turn_id=turn_id,
        session_key=session_key,
        platform="telegram",
        chat_id="effect-chat",
        thread_id="topic-7",
        payload={"text": "work"},
    )
    component_id = dl.record_delivery_plan(
        turn_id=turn_id,
        session_key=session_key,
        platform="telegram",
        chat_id="effect-chat",
        thread_id="topic-7",
        components=[{"kind": "text", "payload": {"content": "answer"}}],
    )[0]
    dl.mark_delivery_component_delivered(component_id)
    effect_id = dl.record_post_delivery_effect(
        turn_id=turn_id,
        session_key=session_key,
        platform="telegram",
        chat_id="effect-chat",
        thread_id="topic-7",
        effect_key="goal-status",
        kind="goal_status_notice",
        payload={"content": "Goal achieved"},
    )
    adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="effect-ack")
    )

    assert await runner._deliver_ready_post_delivery_effects() == 1
    assert await runner._deliver_ready_post_delivery_effects() == 0

    adapter.send.assert_awaited_once_with(
        "effect-chat",
        "Goal achieved",
        metadata={"thread_id": "topic-7"},
    )
    with dl._connect() as conn:
        state = conn.execute(
            "SELECT state FROM post_delivery_effects WHERE effect_id=?", (effect_id,)
        ).fetchone()[0]
    assert state == "delivered"


@pytest.mark.asyncio
async def test_post_delivery_effects_drain_successive_ordinals_in_one_pass(
    tmp_path, monkeypatch
):
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    monkeypatch.setattr(dl, "_owner_alive", lambda *_args: False)
    runner, adapter = make_restart_runner()
    turn_id = "turn-ordered-durable-effects"
    session_key = "agent:main:telegram:dm:ordered-effect-chat"
    component_id = dl.record_delivery_plan(
        turn_id=turn_id,
        session_key=session_key,
        platform="telegram",
        chat_id="ordered-effect-chat",
        thread_id=None,
        components=[{"kind": "text", "payload": {"content": "answer"}}],
    )[0]
    dl.mark_delivery_component_delivered(component_id)
    effect_ids = [
        dl.record_post_delivery_effect(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="ordered-effect-chat",
            thread_id=None,
            effect_key=f"ordered-{ordinal}",
            kind="goal_status_notice",
            payload={"content": content},
        )
        for ordinal, content in enumerate(("First notice", "Second notice"))
    ]
    adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="effect-ack")
    )

    assert await runner._deliver_ready_post_delivery_effects() == 2
    assert [call.args[1] for call in adapter.send.await_args_list] == [
        "First notice",
        "Second notice",
    ]
    with dl._connect() as conn:
        states = [
            conn.execute(
                "SELECT state FROM post_delivery_effects WHERE effect_id=?",
                (effect_id,),
            ).fetchone()[0]
            for effect_id in effect_ids
        ]
    assert states == ["delivered", "delivered"]


@pytest.mark.asyncio
async def test_post_delivery_effect_drain_stops_at_first_failure(
    tmp_path, monkeypatch
):
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    monkeypatch.setattr(dl, "_owner_alive", lambda *_args: False)
    runner, adapter = make_restart_runner()
    effect_ids = []
    for turn_id, content in (("turn-a", "Fail first"), ("turn-b", "Do not send")):
        session_key = f"agent:main:telegram:dm:{turn_id}"
        component_id = dl.record_delivery_plan(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id=turn_id,
            thread_id=None,
            components=[{"kind": "text", "payload": {"content": "answer"}}],
        )[0]
        dl.mark_delivery_component_delivered(component_id)
        effect_ids.append(
            dl.record_post_delivery_effect(
                turn_id=turn_id,
                session_key=session_key,
                platform="telegram",
                chat_id=turn_id,
                thread_id=None,
                effect_key="stop-on-failure",
                kind="goal_status_notice",
                payload={"content": content},
            )
        )
    adapter.send = AsyncMock(side_effect=RuntimeError("transport unavailable"))

    assert await runner._deliver_ready_post_delivery_effects() == 0
    assert adapter.send.await_count == 1
    with dl._connect() as conn:
        states = [
            conn.execute(
                "SELECT state FROM post_delivery_effects WHERE effect_id=?",
                (effect_id,),
            ).fetchone()[0]
            for effect_id in effect_ids
        ]
    assert states == ["failed", "pending"]


@pytest.mark.asyncio
async def test_post_delivery_effect_checkpoint_failure_labels_retry(
    tmp_path, monkeypatch
):
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    monkeypatch.setattr(dl, "_owner_alive", lambda *_args: False)
    runner, adapter = make_restart_runner()
    turn_id = "turn-durable-effect-checkpoint-failure"
    session_key = "agent:main:telegram:dm:effect-chat"
    component_id = dl.record_delivery_plan(
        turn_id=turn_id,
        session_key=session_key,
        platform="telegram",
        chat_id="effect-chat",
        thread_id="topic-7",
        components=[{"kind": "text", "payload": {"content": "answer"}}],
    )[0]
    dl.mark_delivery_component_delivered(component_id)
    effect_id = dl.record_post_delivery_effect(
        turn_id=turn_id,
        session_key=session_key,
        platform="telegram",
        chat_id="effect-chat",
        thread_id="topic-7",
        effect_key="goal-status",
        kind="goal_status_notice",
        payload={"content": "Goal achieved"},
    )
    adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="effect-ack")
    )
    original_mark_delivered = dl.mark_post_delivery_effect_delivered
    calls = 0

    def fail_first_checkpoint(candidate_effect_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("checkpoint unavailable")
        original_mark_delivered(candidate_effect_id)

    monkeypatch.setattr(
        dl, "mark_post_delivery_effect_delivered", fail_first_checkpoint
    )

    assert await runner._deliver_ready_post_delivery_effects() == 0
    assert await runner._deliver_ready_post_delivery_effects() == 1
    assert adapter.send.await_args_list[0].args[1] == "Goal achieved"
    assert adapter.send.await_args_list[1].args[1] == (
        dl.RECOVERED_MARKER + "Goal achieved"
    )
    assert adapter.send.await_count == 2
    with dl._connect() as conn:
        state = conn.execute(
            "SELECT state FROM post_delivery_effects WHERE effect_id=?", (effect_id,)
        ).fetchone()[0]
    assert state == "delivered"


@pytest.mark.asyncio
async def test_completed_historical_plan_does_not_block_new_auto_resume(
    tmp_path, monkeypatch
):
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "delivery.db")
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="reused-session-chat")
    session_key = "agent:main:telegram:dm:reused-session-chat"
    old_turn = "old-completed-turn"
    dl.record_inbound_turn(
        turn_id=old_turn,
        session_key=session_key,
        platform="telegram",
        chat_id="reused-session-chat",
        thread_id=None,
        payload={"text": "old"},
    )
    component_id = dl.record_delivery_plan(
        turn_id=old_turn,
        session_key=session_key,
        platform="telegram",
        chat_id="reused-session-chat",
        thread_id=None,
        components=[{"kind": "text", "payload": {"content": "done"}}],
    )[0]
    dl.mark_delivery_component_delivered(component_id)
    dl.mark_inbound_turn_completed(old_turn, session_key=session_key)
    pending_entry = SessionEntry(
        session_key=session_key,
        session_id="sid-reused",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="turn_in_progress",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio.sleep(0)

    assert scheduled == 1
    adapter.handle_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# Gateway startup auto-resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_auto_resume_skips_unauthorized_owner():
    """A resume-pending session whose owner is no longer authorized under the
    current allowlist must not receive a synthesized agent turn on restart.

    Auto-resume dispatches a full agent turn without going through the normal
    inbound-message auth gate, so it re-checks _is_user_authorized here
    (issue #23778).  An unauthorized owner is skipped WITHOUT claiming a
    _running_agents slot or persisting one — the slot claim happens only
    after this gate passes.
    """
    runner, adapter = make_restart_runner()
    runner._is_user_authorized = lambda _source: False
    runner._persist_active_agents = MagicMock()
    source = make_restart_source(chat_id="revoked-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:revoked-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_timeout",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio.sleep(0)

    assert scheduled == 0
    adapter.handle_message.assert_not_called()
    # No slot was claimed and nothing was persisted for the skipped session.
    assert pending_entry.session_key not in runner._running_agents
    runner._persist_active_agents.assert_not_called()


@pytest.mark.asyncio
async def test_reconnect_reschedule_is_platform_scoped():
    """The platform filter limits the pass to that platform's sessions, so
    reconnecting one platform never resumes another's pending session."""
    runner, adapter = make_restart_runner()
    tg_source = make_restart_source(chat_id="tg-chat")
    discord_source = SessionSource(
        platform=Platform.DISCORD, chat_id="dc-chat", chat_type="dm", user_id="u1"
    )
    tg_entry = SessionEntry(
        session_key="agent:main:telegram:dm:tg-chat",
        session_id="sid-tg",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=tg_source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    discord_entry = SessionEntry(
        session_key="agent:main:discord:dm:dc-chat",
        session_id="sid-dc",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=discord_source,
        platform=Platform.DISCORD,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {
        tg_entry.session_key: tg_entry,
        discord_entry.session_key: discord_entry,
    }
    adapter.handle_message = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}

    scheduled = runner._schedule_resume_pending_sessions(platform=Platform.TELEGRAM)
    await asyncio.sleep(0)

    # Only the telegram session is resumed; the discord session waits for its
    # own reconnect.
    assert scheduled == 1
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source == tg_source


@pytest.mark.asyncio
async def test_startup_restore_waits_for_resume_before_draining_inbound():
    """Queued inbound turns replay only after startup resume tasks finish."""
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_queue = []
    runner._startup_restore_tasks = []

    source = make_restart_source(chat_id="restore-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:restore-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}

    resume_done = asyncio.Event()
    seen: list[str] = []

    async def fake_handle_message(event: MessageEvent) -> None:
        if event.internal:
            seen.append("resume-start")
            task = asyncio.create_task(resume_done.wait())
            adapter._session_tasks[pending_entry.session_key] = task
            return
        seen.append(f"inbound:{event.text}")

    adapter.handle_message = fake_handle_message

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio.sleep(0)

    inbound = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
    )
    assert await runner._handle_message(inbound) is None
    assert scheduled == 1
    assert seen == ["resume-start"]
    assert runner._startup_restore_queue == [inbound]
    assert inbound.metadata["_hermes_inbound_completion_deferred"] is True

    finish_task = asyncio.create_task(runner._finish_startup_restore())
    await asyncio.sleep(0)
    assert seen == ["resume-start"]

    resume_done.set()
    await finish_task

    assert seen == ["resume-start", "inbound:hello"]
    assert runner._startup_restore_queue == []
    assert "_hermes_inbound_completion_deferred" not in inbound.metadata
    assert runner._startup_restore_in_progress is False


def test_stale_resume_candidate_releases_staged_inbound_for_one_replay():
    from gateway import delivery_ledger as dl

    runner, _adapter = make_restart_runner()
    source = make_restart_source(chat_id="stale-accepted")
    session_key = "agent:main:telegram:dm:stale-accepted"
    entry = SessionEntry(
        session_key=session_key,
        session_id="sid-stale",
        created_at=datetime.now() - timedelta(hours=4),
        updated_at=datetime.now() - timedelta(hours=4),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now() - timedelta(hours=4),
    )
    runner.session_store._entries = {session_key: entry}
    dl.record_inbound_turn(
        turn_id="accepted-turn",
        session_key=session_key,
        platform="telegram",
        chat_id="stale-accepted",
        thread_id=None,
        payload={"text": "accepted"},
    )
    dl.release_inbound_turn_claim("accepted-turn")
    claimed = dl.sweep_recoverable_inbound_turns()
    runner._startup_inbound_by_session = {session_key: claimed}

    with patch("gateway.delivery_ledger.session_has_delivery_plan", return_value=False):
        assert runner._schedule_resume_pending_sessions() == 0

    assert runner._startup_inbound_by_session == {}
    replay = dl.sweep_recoverable_inbound_turns()
    duplicate = dl.sweep_recoverable_inbound_turns()
    assert [row["turn_id"] for row in replay] == ["accepted-turn"]
    assert duplicate == []


@pytest.mark.asyncio
async def test_startup_restore_missing_adapter_releases_inbound_claim():
    runner, _adapter = make_restart_runner()
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="missing-adapter"),
        metadata={"_hermes_turn_id": "turn-missing-adapter"},
    )
    runner._startup_restore_queue = [event]
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._release_inbound_event_claims = AsyncMock(return_value=1)

    assert await runner._drain_startup_restore_queue() == 0
    runner._release_inbound_event_claims.assert_awaited_once_with(event)
    assert runner._startup_restore_queue == []


@pytest.mark.asyncio
async def test_startup_resume_adapter_exception_releases_claim_before_acceptance():
    runner, adapter = make_restart_runner()
    session_key = "agent:main:telegram:dm:resume-failure"
    event = MessageEvent(
        text="recover",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="resume-failure"),
        internal=True,
        metadata={"_hermes_turn_id": "turn-resume-failure"},
    )
    adapter.handle_message = AsyncMock(side_effect=RuntimeError("adapter rejected"))
    runner._release_inbound_event_claims = AsyncMock(return_value=1)

    with pytest.raises(RuntimeError, match="adapter rejected"):
        await runner._run_startup_resume_event(adapter, event, session_key)

    runner._release_inbound_event_claims.assert_awaited_once_with(event)


# ---------------------------------------------------------------------------
# Shutdown banner wording
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_notifies_home_channel_even_without_active_sessions():
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )

    await runner._notify_active_sessions_of_shutdown()

    assert adapter.sent == [
        "⚠️ Gateway restarting — Your current task will be interrupted. "
        "Send any message after restart and I'll try to resume where you left off."
    ]


@pytest.mark.asyncio
async def test_restart_home_channel_notification_not_deduped_across_threads():
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    session_key = "agent:main:telegram:group:999"
    runner.session_store._entries[session_key] = MagicMock(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="999",
            chat_type="group",
            user_id="u1",
            thread_id="topic-7",
        )
    )
    runner._running_agents[session_key] = MagicMock()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="999",
        name="Ops Home",
    )

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent) == 2
    assert adapter.sent_calls[0][2] == {"thread_id": "topic-7"}
    assert adapter.sent_calls[1][2] is None


# ---------------------------------------------------------------------------
# Stuck-loop escalation integration
# ---------------------------------------------------------------------------


class TestStuckLoopEscalation:
    """The existing .restart_failure_counts counter (PR #7536) remains the
    single source of terminal escalation — no parallel counter on
    SessionEntry was added.  After the configured threshold, the startup
    path flips suspended=True which overrides resume_pending."""

    def test_escalation_via_stuck_loop_counter_overrides_resume_pending(
        self, tmp_path, monkeypatch
    ):
        """Simulate a session that keeps getting restart-interrupted and
        hits the stuck-loop threshold: next startup should force it to
        fresh-session despite resume_pending being set."""
        import json

        from gateway.run import GatewayRunner

        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.mark_resume_pending(entry.session_key, reason="restart_timeout")

        # Simulate counter already at threshold (3 consecutive interrupted
        # restarts).  _suspend_stuck_loop_sessions will flip suspended=True.
        counts_file = tmp_path / ".restart_failure_counts"
        counts_file.write_text(json.dumps({entry.session_key: 3}))

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        runner = object.__new__(GatewayRunner)
        runner.session_store = store

        suspended_count = GatewayRunner._suspend_stuck_loop_sessions(runner)
        assert suspended_count == 1
        assert store._entries[entry.session_key].suspended is True
        # resume_pending is still set on the entry, but suspended wins in
        # get_or_create_session so the next message still gets a new sid.
        second = store.get_or_create_session(source)
        assert second.session_id != entry.session_id
        assert second.auto_reset_reason == "suspended"


@pytest.mark.asyncio
async def test_auto_resume_sets_sentinel_before_task_execution():
    """Auto-resume must claim the session slot before the task starts.

    Regression for #45456: between ``asyncio.create_task()`` and the task's
    first await (where ``_process_message_background`` sets the real
    sentinel), an inbound message could arrive and spin up a duplicate
    AIAgent.  The fix pre-claims the slot so the inbound path sees it as
    occupied.
    """
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="race-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:race-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}

    # Slow mock: hold the task open so we can inspect _running_agents
    # while it's in-flight.
    gate = asyncio.Event()

    async def _slow_handle(event):
        await gate.wait()

    adapter.handle_message = _slow_handle

    scheduled = runner._schedule_resume_pending_sessions()

    assert scheduled == 1
    # The sentinel must be set immediately — before the task starts executing.
    assert pending_entry.session_key in runner._running_agents
    assert runner._running_agents[pending_entry.session_key] is _AGENT_PENDING_SENTINEL
    assert pending_entry.session_key in runner._running_agents_ts

    # Release the task and let it complete.
    gate.set()
    await asyncio.sleep(0.05)

    # After the task completes, the sentinel should be cleaned up.
    assert pending_entry.session_key not in runner._running_agents


@pytest.mark.asyncio
async def test_auto_resume_runs_agent_exactly_once_through_full_path():
    """Full-path regression: the pre-claim must NOT make auto-resume a no-op.

    The two tests above mock ``adapter.handle_message`` outright, so they
    only prove the sentinel is set/cleaned around a stub — they never
    exercise the real dispatch chain.  This drives the production path
    end to end:

        _schedule_resume_pending_sessions
          -> _guarded_handle_message
            -> adapter.handle_message            (real)
              -> _process_message_background      (real)
                -> _handle_message                (real)

    The risk the pre-claim introduces is a *self-bounce*: the resume
    turn's own ``_handle_message`` sees the sentinel it pre-claimed at
    the early running-agent guard, queues the event into
    ``_pending_messages`` and returns ``None`` without running the
    agent.  The adapter's late-arrival drain (in
    ``_process_message_background``'s ``finally``) re-dispatches the
    queued event, and because the guard wrapper's ``finally`` releases
    the pre-claim before the spawned drain task starts, the agent runs
    exactly once.  This test locks that invariant in: the resume agent
    must run once — never zero (regression) and never twice (the bug
    the fix targets).
    """
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="full-path-chat")
    session_key = runner._session_key_for_source(source)
    pending_entry = SessionEntry(
        session_key=session_key,
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {session_key: pending_entry}

    # Wire the REAL runner pipeline that _handle_message depends on.
    from gateway.run import GatewayRunner

    runner._handle_message = GatewayRunner._handle_message.__get__(
        runner, GatewayRunner
    )
    runner._release_running_agent_state = (
        GatewayRunner._release_running_agent_state.__get__(runner, GatewayRunner)
    )
    runner._check_slash_access = lambda *a, **k: None
    runner._begin_session_run_generation = lambda session_key: 1
    runner._is_session_run_current = lambda session_key, generation: True
    runner._invalidate_session_run_generation = lambda *a, **k: 0
    runner._claim_active_session_slot = lambda session_key, source: (object(), None)
    runner._active_session_leases = {}
    runner._busy_ack_ts = {}
    runner._post_turn_goal_continuation = AsyncMock()
    runner.session_store.get_or_create_session.return_value = None

    # Count how many times an actual agent run is started for this session.
    agent_runs: list[str] = []

    async def _fake_run(event, source, _quick_key, run_generation):
        agent_runs.append(_quick_key)
        return "RESUMED OK"

    runner._handle_message_with_agent = _fake_run

    # Route the adapter's real background pipeline at the real handler,
    # and stub the leaf send/typing calls so delivery is a no-op.
    adapter.set_message_handler(runner._handle_message)
    adapter.send = AsyncMock()
    adapter._keep_typing = AsyncMock()
    adapter._stop_typing_refresh = AsyncMock()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="1")
    )
    adapter._run_processing_hook = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()
    assert scheduled == 1
    # Pre-claim must be visible immediately.
    assert runner._running_agents.get(session_key) is _AGENT_PENDING_SENTINEL

    # Let the guarded task, the background task, and the late-arrival
    # drain task all settle.
    for _ in range(20):
        await asyncio.sleep(0.02)

    # Exactly one agent run for the resumed session — not zero (the
    # pre-claim did not swallow the resume) and not two (no duplicate).
    assert agent_runs == [session_key]
    # No leaked sentinel and no orphaned queued event.
    assert session_key not in runner._running_agents
    assert session_key not in getattr(adapter, "_pending_messages", {})


# ---------------------------------------------------------------------------
# Startup-restore inbound gate must be BOUNDED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_restore_gate_releases_when_resume_turn_outlives_timeout(
    monkeypatch,
):
    """A single slow boot-resume turn must not hold the inbound gate shut.

    While ``_startup_restore_in_progress`` is set, every inbound message is
    QUEUED instead of answered.  The gate is opened by
    ``_finish_startup_restore``, which waits on the synthetic boot
    auto-resume turns.  Without a bound, one pathologically long resumed
    turn holds the gate — and therefore every channel's inbound queue —
    for the entire duration of that turn.
    """
    monkeypatch.setenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", "0.05")

    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_queue = []
    runner._background_tasks = set()

    seen: list[str] = []
    never_finishes = asyncio.Event()

    async def slow_resume_turn() -> None:
        await never_finishes.wait()

    async def fake_handle_message(event: MessageEvent) -> None:
        seen.append(f"inbound:{event.text}")

    adapter.handle_message = fake_handle_message

    slow_task = asyncio.create_task(slow_resume_turn())
    runner._startup_restore_tasks = [slow_task]

    inbound = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
    )
    assert await runner._handle_message(inbound) is None
    assert runner._startup_restore_queue == [inbound]

    # The gate must release on the bound even though the resume turn is
    # still running.
    await asyncio.wait_for(runner._finish_startup_restore(), timeout=5)

    assert seen == ["inbound:hello"], (
        "startup-restore gate never released: queued inbound was not drained "
        "while a slow boot-resume turn was still running"
    )
    assert runner._startup_restore_queue == []
    assert runner._startup_restore_in_progress is False
    # The slow turn is NOT cancelled — it finishes in the background.
    assert not slow_task.done()

    never_finishes.set()
    await slow_task


