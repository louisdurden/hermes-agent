"""Producer-hook tests: _process_message_background records delivery
obligations around the final send (gateway/platforms/base.py).

Contract: obligation recorded pending before the first send, each component
transitions to attempting immediately before its own transport, and ACK marks
it delivered/failed. Durable-plan failures block every unregistered send.
"""

import asyncio
import json
import os
import stat
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl
from gateway.platforms import base as base_platform
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    merge_pending_message_event,
    stage_delivery_artifact,
)
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(base_platform, "_HERMES_HOME", home)
    monkeypatch.setattr(dl, "get_hermes_home", lambda: home)

    image_counter = 0

    async def _cache_test_image(url, ext=".jpg", retries=2):
        nonlocal image_counter
        image_counter += 1
        path = tmp_path / f"remote-{image_counter}{ext}"
        path.write_bytes(b"test-image")
        return str(path)

    monkeypatch.setattr(base_platform, "cache_image_from_url", _cache_test_image)
    yield


class _Adapter(BasePlatformAdapter):  # type: ignore[misc]
    """Minimal concrete adapter driving the real base-class pipeline."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False):  # pragma: no cover
        return True

    async def disconnect(self):  # pragma: no cover - unused
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="m1")


def _event(text="hello agent"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK, chat_id="C1", chat_type="channel"
        ),
        message_id="msg-42",
    )


def _rows():
    with dl._connect() as conn:
        return conn.execute(
            "SELECT obligation_id, state, content FROM delivery_obligations"
        ).fetchall()


def _component_rows():
    with dl._connect() as conn:
        return conn.execute(
            """SELECT ordinal, kind, state, payload, last_error FROM delivery_components
               ORDER BY ordinal"""
        ).fetchall()


def _effect_rows():
    with dl._connect() as conn:
        return conn.execute(
            """SELECT turn_id, session_key, platform, chat_id, thread_id,
                      effect_key, kind, payload, state
               FROM post_delivery_effects ORDER BY created_at, effect_id"""
        ).fetchall()


@pytest.mark.parametrize(
    ("path", "is_voice", "force_document", "source_kind", "expected"),
    [
        ("clip.mp3", False, False, "media", "voice"),
        ("clip.mp4", False, False, "media", "video"),
        ("report.pdf", False, False, "media", "document"),
        ("chart.png", False, False, "local", "image"),
        ("chart.png", False, True, "local", "document"),
    ],
)
def test_non_text_effect_kind_is_canonical_before_plan_recording(
    path, is_voice, force_document, source_kind, expected
):
    assert base_platform._delivery_effect_kind_for_path(
        Platform.SLACK,
        path,
        is_voice=is_voice,
        force_document=force_document,
        source_kind=source_kind,
    ) == expected


async def _run(adapter, event, response="final answer"):
    if adapter.gateway_runner is None:
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(
            return_value=True
        )
    if not isinstance(
        getattr(adapter.gateway_runner, "_prepare_inbound_turn", None), AsyncMock
    ):
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
    if not isinstance(
        getattr(adapter.gateway_runner, "_complete_inbound_turn", None), AsyncMock
    ):
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)
    adapter._message_handler = AsyncMock(return_value=response)
    session_key = "agent:main:slack:channel:C1"
    turn_id = adapter.gateway_runner._prepare_inbound_turn.return_value
    if not isinstance(turn_id, str) or not turn_id:
        turn_id = "turn-msg-42"
    with dl._connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM inbound_turns WHERE turn_id=?", (turn_id,)
        ).fetchone()
    if not exists:
        dl.record_inbound_turn(
            turn_id=turn_id,
            session_key=session_key,
            platform="slack",
            chat_id=event.source.chat_id,
            thread_id=getattr(event.source, "thread_id", None),
            payload={"text": event.text},
        )
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(event, session_key)


class TestProducerHook:
    def test_staged_artifact_is_synced_before_plan_commit(
        self, tmp_path, monkeypatch
    ):
        source = tmp_path / "voice.ogg"
        source.write_bytes(b"durable-audio")
        order = []
        real_fsync = os.fsync

        def tracked_fsync(fd):
            mode = os.fstat(fd).st_mode
            order.append("directory" if stat.S_ISDIR(mode) else "file")
            real_fsync(fd)

        monkeypatch.setattr(base_platform.os, "fsync", tracked_fsync)
        staged = stage_delivery_artifact(str(source), "turn-fsync")
        order.append("sqlite")
        dl.record_delivery_plan(
            turn_id="turn-fsync",
            session_key="session",
            platform="telegram",
            chat_id="1",
            thread_id=None,
            components=[{"kind": "tts", "payload": {"path": staged}}],
        )

        assert order == ["file", "directory", "directory", "sqlite"]

    def test_staged_artifact_fsync_failure_blocks_plan_and_cleans_copy(
        self, tmp_path, monkeypatch
    ):
        source = tmp_path / "voice.ogg"
        source.write_bytes(b"durable-audio")
        monkeypatch.setattr(
            base_platform.os, "fsync", MagicMock(side_effect=OSError("sync failed"))
        )
        plan = MagicMock()

        with pytest.raises(OSError, match="sync failed"):
            staged = stage_delivery_artifact(str(source), "turn-fsync-fail")
            plan(staged)

        plan.assert_not_called()
        outbox = base_platform.get_delivery_outbox_dir()
        assert [path for path in outbox.rglob("*") if path.is_file()] == []

    @pytest.mark.asyncio
    async def test_late_background_review_waits_for_atomic_plan_commit(
        self, monkeypatch
    ):
        from gateway.run import _DurableBackgroundReviewCollector

        adapter = _Adapter()
        session_key = "agent:main:slack:channel:C1"
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(
            return_value=True
        )
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)
        adapter.gateway_runner._deliver_ready_post_delivery_effects = AsyncMock(
            return_value=1
        )
        collector = _DurableBackgroundReviewCollector()
        commit_entered = threading.Event()
        allow_commit = threading.Event()
        late_started = threading.Event()
        late_finished = threading.Event()
        controller_errors = []
        real_atomic = dl.record_delivery_plan_with_post_delivery_effects

        def gated_atomic(**kwargs):
            commit_entered.set()
            if not allow_commit.wait(2):
                raise TimeoutError("test did not release atomic commit")
            return real_atomic(**kwargs)

        monkeypatch.setattr(
            dl, "record_delivery_plan_with_post_delivery_effects", gated_atomic
        )

        async def handler(event):
            event.metadata["_hermes_background_review_effect_collector"] = collector
            return "final answer"

        adapter._message_handler = handler
        dl.record_inbound_turn(
            turn_id="turn-msg-42",
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "hello agent"},
        )
        adapter._active_sessions[session_key] = asyncio.Event()

        def add_late():
            late_started.set()
            collector.add("Late memory update")
            late_finished.set()

        def control_race():
            if not commit_entered.wait(2):
                controller_errors.append("atomic commit was never entered")
                allow_commit.set()
                return
            late_thread = threading.Thread(target=add_late)
            late_thread.start()
            if not late_started.wait(2):
                controller_errors.append("late callback never started")
            if late_finished.wait(0.1):
                controller_errors.append("late callback returned before commit")
            allow_commit.set()
            late_thread.join(2)
            if late_thread.is_alive():
                controller_errors.append("late callback deadlocked")

        controller = threading.Thread(target=control_race)
        controller.start()
        await adapter._process_message_background(_event(), session_key)
        controller.join(2)

        assert controller_errors == []
        assert late_finished.is_set()
        assert [row[5] for row in _effect_rows()] == ["background-review:0"]

    @pytest.mark.asyncio
    async def test_failed_atomic_plan_releases_late_background_review_waiter(
        self, monkeypatch
    ):
        from gateway.run import _DurableBackgroundReviewCollector

        adapter = _Adapter()
        session_key = "agent:main:slack:channel:C1"
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(
            return_value=True
        )
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)
        collector = _DurableBackgroundReviewCollector()
        commit_entered = threading.Event()
        allow_failure = threading.Event()
        late_finished = threading.Event()
        late_errors = []

        def failing_atomic(**_kwargs):
            commit_entered.set()
            if not allow_failure.wait(2):
                raise TimeoutError("test did not release failed commit")
            raise RuntimeError("forced atomic failure")

        monkeypatch.setattr(
            dl, "record_delivery_plan_with_post_delivery_effects", failing_atomic
        )

        async def handler(event):
            event.metadata["_hermes_background_review_effect_collector"] = collector
            return "final answer"

        adapter._message_handler = handler
        dl.record_inbound_turn(
            turn_id="turn-msg-42",
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "hello agent"},
        )
        adapter._active_sessions[session_key] = asyncio.Event()

        def add_late():
            try:
                collector.add("Late memory update")
            except Exception as exc:
                late_errors.append(str(exc))
            finally:
                late_finished.set()

        def control_failure():
            if commit_entered.wait(2):
                late_thread = threading.Thread(target=add_late)
                late_thread.start()
                allow_failure.set()
                late_thread.join(2)
            else:
                allow_failure.set()

        controller = threading.Thread(target=control_failure)
        controller.start()
        await adapter._process_message_background(_event(), session_key)
        controller.join(2)

        assert late_finished.is_set()
        assert late_errors == ["post-delivery effect delivery plan mismatch"]
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_background_review_collector_is_sealed_before_primary_send(self):
        from gateway.run import _DurableBackgroundReviewCollector

        adapter = _Adapter()
        session_key = "agent:main:slack:channel:C1"
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(
            return_value=True
        )
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)
        adapter.gateway_runner._deliver_ready_post_delivery_effects = AsyncMock(
            return_value=1
        )
        collector = _DurableBackgroundReviewCollector()
        collector.add("Memory updated")

        async def handler(event):
            event.metadata["_hermes_background_review_effect_collector"] = collector
            return "final answer"

        send_rows = []

        async def send(chat_id, content, reply_to=None, metadata=None):
            send_rows.append(_effect_rows())
            return SendResult(success=True, message_id="m1")

        adapter._message_handler = handler
        adapter.send = send
        dl.record_inbound_turn(
            turn_id="turn-msg-42",
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "hello agent"},
        )
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter._process_message_background(_event(), session_key)

        assert send_rows[0][0][5:] == (
            "background-review:0",
            "background_review_notice",
            '{"content":"Memory updated"}',
            "pending",
        )
        collector.add("Late memory update")
        assert [row[5] for row in _effect_rows()] == [
            "background-review:0",
            "background-review:1",
        ]
        assert json.loads(_effect_rows()[1][7]) == {
            "content": "Late memory update"
        }

    @pytest.mark.asyncio
    async def test_goal_status_intent_is_durable_before_primary_send(self):
        adapter = _Adapter()
        session_key = "agent:main:slack:channel:C1"
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(
            return_value=True
        )
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)
        adapter.gateway_runner._deliver_ready_post_delivery_effects = AsyncMock(
            return_value=1
        )

        async def handler(event):
            event.metadata["_hermes_post_delivery_effects"] = [
                {
                    "effect_key": "goal-status",
                    "kind": "goal_status_notice",
                    "payload": {"content": "Goal achieved"},
                }
            ]
            return "final answer"

        send_rows = []

        async def send(chat_id, content, reply_to=None, metadata=None):
            send_rows.append(_effect_rows())
            adapter.sent.append(content)
            return SendResult(success=True, message_id="m1")

        adapter._message_handler = handler
        adapter.send = send
        dl.record_inbound_turn(
            turn_id="turn-msg-42",
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "hello agent"},
        )
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter._process_message_background(_event(), session_key)

        assert adapter.sent == ["final answer"]
        adapter.gateway_runner._deliver_ready_post_delivery_effects.assert_awaited_once()
        assert send_rows == [[
            (
                "turn-msg-42",
                session_key,
                "slack",
                "C1",
                None,
                "goal-status",
                "goal_status_notice",
                '{"content":"Goal achieved"}',
                "pending",
            )
        ]]

    @pytest.mark.asyncio
    async def test_post_delivery_effect_failure_blocks_primary_send(self, monkeypatch):
        adapter = _Adapter()
        session_key = "agent:main:slack:channel:C1"
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(
            return_value=True
        )
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)

        async def handler(event):
            event.metadata["_hermes_post_delivery_effects"] = [
                {
                    "effect_key": "goal-status",
                    "kind": "goal_status_notice",
                    "payload": {"content": "Goal achieved"},
                }
            ]
            return "final answer"

        adapter._message_handler = handler
        dl.record_inbound_turn(
            turn_id="turn-msg-42",
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "hello agent"},
        )
        monkeypatch.setattr(
            dl,
            "record_delivery_plan_with_post_delivery_effects",
            MagicMock(side_effect=RuntimeError("effect persistence failed")),
        )
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter._process_message_background(_event(), session_key)

        assert adapter.sent == []
        adapter.gateway_runner._complete_inbound_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_plan_persistence_failure_does_not_release_post_delivery_callback(
        self, monkeypatch
    ):
        adapter = _Adapter()
        fired = []
        session_key = "agent:main:slack:channel:C1"
        adapter.register_post_delivery_callback(
            session_key, lambda: fired.append("released")
        )
        monkeypatch.setattr(
            dl,
            "record_delivery_plan_with_post_delivery_effects",
            MagicMock(side_effect=RuntimeError("plan persistence failed")),
        )

        await _run(adapter, _event(), response="final answer")

        assert fired == []
        assert session_key in adapter._post_delivery_callbacks
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_durable_delivery_releases_callback_by_canonical_session_key(self):
        adapter = _Adapter()
        fired = []
        fast_key = "agent:main:slack:channel:C1"
        canonical_key = "agent:main:slack:channel:C1:thread:canonical"
        adapter.register_post_delivery_callback(
            canonical_key, lambda: fired.append("released"), generation=7
        )
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(
            return_value=True
        )
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)

        async def handler(event):
            event.metadata["_hermes_canonical_session_key"] = canonical_key
            return "final answer"

        adapter._message_handler = handler
        dl.record_inbound_turn(
            turn_id="turn-msg-42",
            session_key=canonical_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "hello agent"},
        )
        adapter._active_sessions[fast_key] = asyncio.Event()
        setattr(adapter._active_sessions[fast_key], "_hermes_run_generation", 7)

        await adapter._process_message_background(_event(), fast_key)

        assert adapter.sent == ["final answer"]
        assert fired == ["released"]
        assert canonical_key not in adapter._post_delivery_callbacks

    @pytest.mark.asyncio
    async def test_durable_delivery_never_releases_generation_callback_without_owner(self):
        adapter = _Adapter()
        fired = []
        session_key = "agent:main:slack:channel:C1"
        adapter.register_post_delivery_callback(
            session_key, lambda: fired.append("released"), generation=8
        )

        await _run(adapter, _event(), response="final answer")

        assert adapter.sent == ["final answer"]
        assert fired == []
        assert session_key in adapter._post_delivery_callbacks

    @pytest.mark.asyncio
    async def test_partial_durable_delivery_retains_post_delivery_callback(
        self, tmp_path
    ):
        adapter = _Adapter()
        fired = []
        session_key = "agent:main:slack:channel:C1"
        adapter.register_post_delivery_callback(
            session_key, lambda: fired.append("released")
        )
        adapter.send_document = AsyncMock(
            return_value=SendResult(success=False, error="transport failed")
        )
        source = tmp_path / "report.pdf"
        source.write_bytes(b"pdf")

        await _run(adapter, _event(), response=f"final text\n{source}")

        assert adapter.sent == ["final text"]
        adapter.send_document.assert_awaited_once()
        assert fired == []
        assert session_key in adapter._post_delivery_callbacks

    @pytest.mark.asyncio
    async def test_component_checkpoint_failure_retains_post_delivery_callback(
        self, tmp_path, monkeypatch
    ):
        adapter = _Adapter()
        fired = []
        session_key = "agent:main:slack:channel:C1"
        adapter.register_post_delivery_callback(
            session_key, lambda: fired.append("released")
        )
        adapter.send_document = AsyncMock(
            return_value=SendResult(success=True, message_id="doc")
        )
        real_mark_attempting = base_platform._mark_delivery_component_attempting_fail_closed
        calls = 0

        def fail_second_component(component_id):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise base_platform.DurableDeliveryCheckpointError(
                    "document checkpoint failed"
                )
            return real_mark_attempting(component_id)

        monkeypatch.setattr(
            base_platform,
            "_mark_delivery_component_attempting_fail_closed",
            fail_second_component,
        )
        source = tmp_path / "report.pdf"
        source.write_bytes(b"pdf")

        await _run(adapter, _event(), response=f"final text\n{source}")

        assert adapter.sent == ["final text"]
        adapter.send_document.assert_not_awaited()
        assert fired == []
        assert session_key in adapter._post_delivery_callbacks

    @pytest.mark.parametrize("extension", ["mp3", "mp4"])
    @pytest.mark.asyncio
    async def test_forced_document_plan_matches_live_transport(self, tmp_path, extension):
        adapter = _Adapter()
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(return_value="turn-msg-42")
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(return_value=True)
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)
        adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
        adapter.send_video = AsyncMock(return_value=SendResult(success=True, message_id="video"))
        adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))
        source = tmp_path / f"clip.{extension}"
        source.write_bytes(b"media")

        await _run(adapter, _event(), response=f"MEDIA:{source}\n[[as_document]]")

        adapter.send_voice.assert_not_awaited()
        adapter.send_video.assert_not_awaited()
        adapter.send_document.assert_awaited_once()
        assert [(row[1], row[2]) for row in _component_rows()] == [
            ("document", "delivered")
        ]

    @pytest.mark.asyncio
    async def test_startup_restore_deferred_turn_stays_recoverable_before_drain(self):
        adapter = _Adapter()
        runner = MagicMock()
        runner._adapter_for_source = lambda source: adapter
        runner._complete_inbound_turn = AsyncMock(return_value=True)
        adapter.gateway_runner = runner
        session_key = "agent:main:slack:channel:C1"
        event = _event()
        event.metadata["_hermes_turn_id"] = "turn-msg-42"
        dl.record_inbound_turn(
            turn_id="turn-msg-42",
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "queued during restore"},
        )

        async def queue_during_restore(message_event):
            message_event.metadata["_hermes_inbound_completion_deferred"] = True
            return None

        adapter._message_handler = queue_during_restore
        adapter._active_sessions[session_key] = asyncio.Event()
        await adapter._process_message_background(event, session_key)

        runner._complete_inbound_turn.assert_not_awaited()
        with dl._connect() as conn:
            state = conn.execute(
                "SELECT state FROM inbound_turns WHERE turn_id='turn-msg-42'"
            ).fetchone()[0]
            conn.execute(
                "UPDATE inbound_turns SET owner_pid=99999999, owner_started_at=NULL "
                "WHERE turn_id='turn-msg-42'"
            )
        assert state == "received"
        recoverable_ids = [
            row["turn_id"] for row in dl.sweep_recoverable_inbound_turns()
        ]
        assert recoverable_ids.count("turn-msg-42") == 1

    @pytest.mark.asyncio
    async def test_producer_persists_effective_slack_workspace_routing(self):
        adapter = _Adapter()
        event = MessageEvent(
            text="hello agent",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=Platform.SLACK,
                chat_id="C1",
                chat_type="channel",
                thread_id="171.001",
                scope_id="T-WORKSPACE-2",
            ),
            message_id="msg-42",
        )

        await _run(adapter, event, response="workspace reply")

        rows = _component_rows()
        payload = json.loads(rows[0][3])
        assert payload["routing_metadata"] == {
            "thread_id": "171.001",
            "slack_team_id": "T-WORKSPACE-2",
            "notify": True,
        }

    @pytest.mark.asyncio
    async def test_clarify_turn_is_non_replay_at_plan_commit_crash(self, monkeypatch):
        adapter = _Adapter()
        session_key = "agent:main:slack:channel:C1"
        adapter._clarify_inbound_turn_ids = {session_key: ["turn-clarify"]}
        dl.record_inbound_turn(
            turn_id="turn-clarify",
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "clarification"},
        )
        dl.record_inbound_turn(
            turn_id="turn-msg-42",
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "primary"},
        )
        real_record_plan = dl.record_delivery_plan

        def commit_then_crash(**kwargs):
            real_record_plan(**kwargs)
            raise RuntimeError("simulated crash after plan commit")

        monkeypatch.setattr(dl, "record_delivery_plan", commit_then_crash)

        await _run(adapter, _event(), response="final answer")

        with dl._connect() as conn:
            state = conn.execute(
                "SELECT state FROM inbound_turns WHERE turn_id='turn-clarify'"
            ).fetchone()[0]
        assert state == "completed"
        recoverable_ids = {
            row["turn_id"] for row in dl.sweep_recoverable_inbound_turns()
        }
        assert "turn-clarify" not in recoverable_ids
        assert dl.delivery_plan_exists("turn-msg-42") is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("include_text", [False, True])
    async def test_ledger_off_attachment_ack_hands_off_then_completes(
        self, include_text, tmp_path, monkeypatch
    ):
        adapter = _Adapter()
        clear_pending = AsyncMock(return_value=True)
        complete_inbound = AsyncMock(return_value=True)
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_resume_delivery_handoff = clear_pending
        adapter.gateway_runner._complete_inbound_turn = complete_inbound
        adapter.send_document = AsyncMock(
            return_value=SendResult(success=True, message_id="doc-1")
        )
        monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: False)

        source = tmp_path / "report.pdf"
        source.write_bytes(b"pdf")
        prefix = "delivered text\n" if include_text else ""
        await _run(adapter, _event(), response=f"{prefix}{source}")

        assert _component_rows() == []
        adapter.send_document.assert_awaited_once()
        clear_pending.assert_awaited_once()
        complete_inbound.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ledger_off_partial_attachment_failure_keeps_inbound(
        self, tmp_path, monkeypatch
    ):
        adapter = _Adapter()
        clear_pending = AsyncMock(return_value=True)
        complete_inbound = AsyncMock(return_value=True)
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_resume_delivery_handoff = clear_pending
        adapter.gateway_runner._complete_inbound_turn = complete_inbound
        adapter.send_document = AsyncMock(
            return_value=SendResult(success=False, error="transport failed")
        )
        adapter._notify_media_delivery_failure = AsyncMock()
        monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: False)

        source = tmp_path / "report.pdf"
        source.write_bytes(b"pdf")
        await _run(adapter, _event(), response=f"delivered text\n{source}")

        assert adapter.sent == ["delivered text"]
        adapter.send_document.assert_awaited_once()
        clear_pending.assert_not_awaited()
        complete_inbound.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_durable_file_failure_does_not_send_unplanned_fallback(
        self, tmp_path
    ):
        adapter = _Adapter()
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(
            return_value=True
        )
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)
        adapter.send_document = AsyncMock(
            return_value=SendResult(success=False, error="transport failed")
        )
        adapter._notify_media_delivery_failure = AsyncMock()
        source = tmp_path / "report.pdf"
        source.write_bytes(b"pdf")

        await _run(adapter, _event(), response=str(source))

        adapter.send_document.assert_awaited_once()
        adapter._notify_media_delivery_failure.assert_not_awaited()
        rows = _component_rows()
        assert [(row[1], row[2]) for row in rows] == [("document", "failed")]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["image", "media", "file"])
    async def test_attachment_staging_failure_is_fail_closed(
        self, kind, tmp_path, monkeypatch
    ):
        adapter = _Adapter()
        clear_pending = AsyncMock(return_value=True)
        complete_inbound = AsyncMock(return_value=True)
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_resume_delivery_handoff = clear_pending
        adapter.gateway_runner._complete_inbound_turn = complete_inbound
        adapter.send = AsyncMock(
            return_value=SendResult(success=True, message_id="must-not-send")
        )

        if kind == "image":
            response = "![chart](https://example.test/chart.png)"
        elif kind == "media":
            source = tmp_path / "clip.mp4"
            source.write_bytes(b"video")
            response = f"MEDIA:{source}"
        else:
            source = tmp_path / "report.pdf"
            source.write_bytes(b"pdf")
            response = str(source)

        monkeypatch.setattr(
            base_platform,
            "stage_delivery_artifact",
            MagicMock(side_effect=OSError("outbox unavailable")),
        )

        await _run(adapter, _event(), response=response)

        assert _component_rows() == []
        clear_pending.assert_not_awaited()
        complete_inbound.assert_not_awaited()
        adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tts_staging_failure_is_fail_closed(self, tmp_path, monkeypatch):
        from tools import tts_tool

        adapter = _Adapter()
        adapter._should_auto_tts_for_chat = lambda _chat_id: True
        clear_pending = AsyncMock(return_value=True)
        complete_inbound = AsyncMock(return_value=True)
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_resume_delivery_handoff = clear_pending
        adapter.gateway_runner._complete_inbound_turn = complete_inbound

        generated = tmp_path / "voice.ogg"

        def _fake_tts(**_kwargs):
            generated.write_bytes(b"voice")
            return json.dumps({"success": True, "file_path": str(generated)})

        monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)
        monkeypatch.setattr(tts_tool, "text_to_speech_tool", _fake_tts)
        monkeypatch.setattr(
            base_platform,
            "stage_delivery_artifact",
            MagicMock(side_effect=OSError("outbox unavailable")),
        )
        event = _event()
        event.message_type = MessageType.VOICE

        await _run(adapter, event, response="spoken answer")

        assert _component_rows() == []
        clear_pending.assert_not_awaited()
        complete_inbound.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_staging_failure_rolls_back_prior_sibling(
        self, tmp_path, monkeypatch
    ):
        adapter = _Adapter()
        clear_pending = AsyncMock(return_value=True)
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_resume_delivery_handoff = clear_pending
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)
        original_stage = base_platform.stage_delivery_artifact
        calls = 0

        def _stage_then_fail(path, turn_id):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("outbox unavailable")
            return original_stage(path, turn_id)

        monkeypatch.setattr(
            base_platform, "stage_delivery_artifact", _stage_then_fail
        )
        await _run(
            adapter,
            _event(),
            response=(
                "![one](https://example.test/one.png)\n"
                "![two](https://example.test/two.png)"
            ),
        )

        outbox = base_platform.get_delivery_outbox_dir()
        assert _component_rows() == []
        assert [path for path in outbox.rglob("*") if path.is_file()] == []
        clear_pending.assert_not_awaited()

    def test_outbox_artifact_survives_source_cleanup_until_durable_ack(
        self, tmp_path, monkeypatch
    ):
        hermes_home = tmp_path / "hermes-home"
        outbox = hermes_home / "cache" / "delivery_outbox"
        monkeypatch.setattr(base_platform, "_HERMES_HOME", hermes_home)
        monkeypatch.setattr(dl, "get_hermes_home", lambda: hermes_home)
        source = tmp_path / "voice.ogg"
        source.write_bytes(b"durable-audio")

        staged = stage_delivery_artifact(str(source), "turn-artifact")
        source.unlink()
        assert Path(staged).parent.parent == outbox
        assert Path(staged).read_bytes() == b"durable-audio"

        component_id = dl.record_delivery_plan(
            turn_id="turn-artifact",
            session_key="session",
            platform="telegram",
            chat_id="1",
            thread_id=None,
            components=[{"kind": "tts", "payload": {"path": staged}}],
        )[0]
        assert Path(staged).exists()

        dl.mark_delivery_component_delivered(component_id)

        assert not Path(staged).exists()

    def test_merged_followups_preserve_all_inbound_turn_ids(self):
        first = _event()
        first.metadata["_hermes_turn_id"] = "turn-1"
        second = _event()
        second.text = "second"
        second.metadata["_hermes_turn_id"] = "turn-2"
        pending = {"session": first}

        merge_pending_message_event(pending, "session", second, merge_text=True)

        assert pending["session"].metadata["_hermes_inbound_turn_ids"] == [
            "turn-1",
            "turn-2",
        ]

    @pytest.mark.asyncio
    async def test_partial_image_ack_leaves_only_failed_image_recoverable(self):
        adapter = _Adapter()
        runner = MagicMock()
        runner._adapter_for_source = lambda source: adapter
        runner._prepare_inbound_turn = AsyncMock(return_value="turn-images")
        runner._complete_resume_delivery_handoff = AsyncMock(return_value=True)
        runner._complete_inbound_turn = AsyncMock(return_value=True)
        adapter.gateway_runner = runner
        results = iter(
            [
                SendResult(success=True, message_id="image-1"),
                SendResult(success=False, error="rejected"),
            ]
        )

        async def send_image(**kwargs):
            return next(results)

        adapter._send_image_with_ack = send_image
        await _run(
            adapter,
            _event(),
            response=(
                "caption\n\n![one](https://example.test/one.png)\n"
                "![two](https://example.test/two.png)"
            ),
        )

        rows = _component_rows()
        image_states = [row[2] for row in rows if row[1] == "image"]
        assert image_states == ["delivered", "failed"]
        runner._complete_inbound_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_successful_turn_completes_deferred_clarify_wal_rows(self):
        adapter = _Adapter()
        session_key = "agent:main:slack:channel:C1"
        adapter._clarify_inbound_turn_ids = {session_key: ["turn-clarify"]}
        dl.record_inbound_turn(
            turn_id="turn-clarify",
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "clarification"},
        )
        dl.record_inbound_turn(
            turn_id="turn-msg-42",
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "primary"},
        )

        runner = MagicMock()
        runner._adapter_for_source = lambda source: adapter
        runner._prepare_inbound_turn = AsyncMock(return_value="turn-msg-42")
        runner._complete_resume_delivery_handoff = AsyncMock(return_value=True)
        runner._complete_inbound_turn = AsyncMock(return_value=True)
        adapter.gateway_runner = runner

        await _run(adapter, _event())

        with dl._connect() as conn:
            state = conn.execute(
                "SELECT state FROM inbound_turns WHERE turn_id='turn-clarify'"
            ).fetchone()[0]
        assert state == "completed"
        runner._complete_inbound_turn.assert_awaited_once()
        assert session_key not in adapter._clarify_inbound_turn_ids

    @pytest.mark.asyncio
    async def test_active_session_followup_is_preflighted_before_queue(self):
        adapter = _Adapter()
        runner = MagicMock()
        runner._prepare_inbound_turn = AsyncMock(return_value="turn-followup")
        runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner = runner
        handler = AsyncMock()
        adapter.set_message_handler(handler)
        event = _event()
        event.message_type = MessageType.PHOTO
        event.media_urls = ["https://example.test/followup.jpg"]
        session_key = "agent:main:slack:channel:C1"
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter.handle_message(event)

        runner._prepare_inbound_turn.assert_awaited_once_with(event, session_key)
        assert event.metadata["_hermes_turn_id"] == "turn-followup"
        assert adapter._pending_messages[session_key] is event
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multipart_plan_is_durable_before_first_send(self):
        adapter = _Adapter()
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(
            return_value=True
        )

        observed = []

        async def _observe_text(chat_id, content, reply_to=None, metadata=None):
            observed.append(("text", _component_rows()))
            return SendResult(success=True, message_id="text-1")

        async def _observe_image(
            *, chat_id, image_url, alt_text="", metadata=None
        ):
            observed.append(("image", _component_rows()))
            return SendResult(success=True, message_id="image-1")

        adapter.send = _observe_text
        adapter._send_image_with_ack = _observe_image
        await _run(
            adapter,
            _event(),
            response="caption\n\n![chart](https://example.test/chart.png)",
        )

        assert observed[0][0] == "text"
        first_rows = observed[0][1]
        assert first_rows[0][0:3] == (0, "text", "attempting")
        text_payload = json.loads(first_rows[0][3])
        assert text_payload == {
            "content": "caption",
            "routing_metadata": {"notify": True},
        }
        assert first_rows[0][4] is None
        assert first_rows[1][0:3] == (1, "image", "pending")
        image_payload = json.loads(first_rows[1][3])
        assert image_payload["routing_metadata"] == {"notify": True}
        assert image_payload["alt"] == "chart"
        assert image_payload["url"].startswith("file://")
        assert "delivery_outbox/turn-msg-42/" in image_payload["url"]
        assert first_rows[1][4] is None
        image_send_rows = observed[1][1]
        assert [row[1] for row in image_send_rows] == ["text", "image"]
        assert [row[2] for row in image_send_rows] == ["delivered", "attempting"]

    @pytest.mark.asyncio
    async def test_inbound_preflight_precedes_processing_hook(self):
        adapter = _Adapter()
        runner = MagicMock()
        runner._adapter_for_source = lambda source: adapter
        runner._prepare_inbound_turn = AsyncMock(return_value="turn-msg-42")
        runner._complete_resume_delivery_handoff = AsyncMock(return_value=True)
        adapter.gateway_runner = runner

        observed = []

        async def _observe_start(_event):
            observed.append(
                (
                    runner._prepare_inbound_turn.await_count,
                    _event.metadata.get("_hermes_turn_id"),
                )
            )

        adapter.on_processing_start = _observe_start
        await _run(adapter, _event())

        assert observed == [(1, "turn-msg-42")]
        runner._prepare_inbound_turn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resume_marker_clears_after_durable_obligation_before_send(self):
        adapter = _Adapter()
        clear_pending = AsyncMock(return_value=True)
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._complete_resume_delivery_handoff = clear_pending

        clear_count_at_send = []

        async def _observe_handoff_before_send(
            chat_id, content, reply_to=None, metadata=None
        ):
            rows = _component_rows()
            clear_count_at_send.append(
                (len(rows), rows[0][2] if rows else None, clear_pending.await_count)
            )
            return SendResult(success=True, message_id="m1")

        adapter.send = _observe_handoff_before_send
        await _run(adapter, _event())
        assert clear_count_at_send == [(1, "attempting", 1)]
        clear_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1", "turn-msg-42"
        )

    @pytest.mark.asyncio
    async def test_without_ledger_resume_marker_clears_only_after_successful_ack(
        self, monkeypatch
    ):
        adapter = _Adapter()
        clear_pending = AsyncMock(return_value=True)
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._complete_resume_delivery_handoff = clear_pending
        monkeypatch.setattr(dl, "ledger_enabled", lambda: False)

        async def _assert_not_cleared_before_ack(
            chat_id, content, reply_to=None, metadata=None
        ):
            clear_pending.assert_not_awaited()
            return SendResult(success=True, message_id="m1")

        adapter.send = _assert_not_cleared_before_ack
        await _run(adapter, _event())
        clear_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1", "turn-msg-42"
        )

    @pytest.mark.asyncio
    async def test_clear_failure_keeps_plan_pending_and_sends_nothing(self):
        adapter = _Adapter()
        clear_pending = AsyncMock(side_effect=RuntimeError("disk unavailable"))
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._complete_resume_delivery_handoff = clear_pending
        adapter.send = AsyncMock(
            return_value=SendResult(success=True, message_id="must-not-send")
        )

        await _run(adapter, _event())

        rows = _component_rows()
        assert len(rows) == 1
        assert rows[0][2] == "pending"
        assert clear_pending.await_count == 1
        adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_attempting_checkpoint_failure_sends_nothing(self, monkeypatch):
        adapter = _Adapter()
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(
            return_value=True
        )
        adapter.send = AsyncMock(
            return_value=SendResult(success=True, message_id="must-not-send")
        )
        monkeypatch.setattr(
            dl,
            "mark_delivery_component_attempting",
            MagicMock(side_effect=OSError("state db unavailable")),
        )

        await _run(adapter, _event())

        adapter.send.assert_not_awaited()
        rows = _component_rows()
        assert len(rows) == 1
        assert rows[0][2] == "pending"

    @pytest.mark.asyncio
    async def test_delivery_plan_and_handoff_use_canonical_session_key(self):
        adapter = _Adapter()
        runner = MagicMock()
        runner._prepare_inbound_turn = AsyncMock(return_value="turn-msg-42")
        runner._complete_resume_delivery_handoff = AsyncMock(return_value=True)
        runner._complete_inbound_turn = AsyncMock(return_value=True)
        runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner = runner

        async def handler(event):
            canonical = "agent:surgical:slack:channel:C1"
            dl.rebind_inbound_turns(
                ["turn-msg-42"],
                expected_session_key="agent:main:slack:channel:C1",
                canonical_session_key=canonical,
            )
            event.metadata["_hermes_canonical_session_key"] = canonical
            return "final answer"

        adapter.set_message_handler(handler)
        event = _event()
        dl.record_inbound_turn(
            turn_id="turn-msg-42",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": event.text},
        )

        await adapter._process_message_background(
            event, "agent:main:slack:channel:C1"
        )

        rows = _component_rows()
        assert len(rows) == 1
        with dl._connect() as conn:
            session_key = conn.execute(
                "SELECT session_key FROM delivery_components LIMIT 1"
            ).fetchone()[0]
        assert session_key == "agent:surgical:slack:channel:C1"
        runner._complete_resume_delivery_handoff.assert_awaited_once_with(
            "agent:surgical:slack:channel:C1", "turn-msg-42"
        )

    @pytest.mark.asyncio
    async def test_normal_turn_records_and_delivers(self):
        adapter = _Adapter()
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(
            return_value=True
        )
        await _run(adapter, _event())

        assert adapter.sent == ["final answer"]
        rows = _component_rows()
        assert len(rows) == 1
        assert rows[0][2] == "delivered"
        assert rows[0][1] == "text"
        assert json.loads(rows[0][3])["content"] == "final answer"
        adapter.gateway_runner._complete_inbound_turn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_failure_leaves_failed_row(self):
        adapter = _Adapter()
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._prepare_inbound_turn = AsyncMock(
            return_value="turn-msg-42"
        )
        adapter.gateway_runner._complete_inbound_turn = AsyncMock(return_value=True)
        adapter.gateway_runner._complete_resume_delivery_handoff = AsyncMock(
            return_value=True
        )
        adapter.send = AsyncMock(
            return_value=SendResult(success=False, error="chat_not_found")
        )
        await _run(adapter, _event())

        rows = _component_rows()
        assert len(rows) == 1
        assert rows[0][2] == "failed"
        adapter.gateway_runner._complete_inbound_turn.assert_not_awaited()


    @pytest.mark.asyncio
    async def test_crash_between_attempting_and_ack_is_recoverable(self):
        """The core scenario (#58818): process dies mid-send. The row must
        be claimable by a later process and carry the ambiguity marker."""
        adapter = _Adapter()

        async def _dies_mid_send(chat_id, content, reply_to=None, metadata=None):
            raise ConnectionError("gateway killed mid-await")

        adapter.send = _dies_mid_send
        # _send_with_retry raising propagates; the background task catches
        # broadly — drive only through the send block by tolerating the error.
        try:
            await _run(adapter, _event())
        except Exception:
            pass

        rows = _component_rows()
        assert len(rows) == 1
        # Row is stuck in 'attempting' (or failed if retry wrapper caught it):
        # either way it is non-delivered and recoverable.
        assert rows[0][2] in ("attempting", "failed")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_components SET owner_pid=999999999, owner_started_at=1"
            )
        claimed = dl.sweep_recoverable_delivery_components()
        assert len(claimed) == 1
        assert claimed[0]["prior_state"] == "attempting"
