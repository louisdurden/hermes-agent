"""Producer-hook tests: _process_message_background records delivery
obligations around the final send (gateway/platforms/base.py).

Contract: obligation recorded (pending→attempting) BEFORE the send await,
delivered/failed by SendResult afterward; slash commands, ephemeral
replies, and empty responses are never recorded; ledger failures never
block the send.
"""

import asyncio
import json
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
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(event, session_key)


class TestProducerHook:
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
        assert first_rows[1][0:3] == (1, "image", "attempting")
        image_payload = json.loads(first_rows[1][3])
        assert image_payload["routing_metadata"] == {"notify": True}
        assert image_payload["alt"] == "chart"
        assert image_payload["url"].startswith("file://")
        assert "delivery_outbox/turn-msg-42/" in image_payload["url"]
        assert first_rows[1][4] is None
        assert [row[1] for row in observed[1][1]] == ["text", "image"]

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
    async def test_clear_failure_keeps_obligation_attempting_for_boot_recovery(self):
        adapter = _Adapter()
        clear_pending = AsyncMock(side_effect=RuntimeError("disk unavailable"))
        adapter.gateway_runner = MagicMock()
        adapter.gateway_runner._adapter_for_source = lambda source: adapter
        adapter.gateway_runner._complete_resume_delivery_handoff = clear_pending

        await _run(adapter, _event())

        rows = _component_rows()
        assert len(rows) == 1
        assert rows[0][2] == "attempting"
        assert clear_pending.await_count == 1

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
            event.metadata["_hermes_canonical_session_key"] = (
                "agent:surgical:slack:channel:C1"
            )
            return "final answer"

        adapter.set_message_handler(handler)
        event = _event()

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
