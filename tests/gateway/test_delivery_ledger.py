"""Tests for the gateway delivery-obligation ledger (gateway/delivery_ledger.py).

State machine, dead-owner claiming, attempts cap, stale cutoff, retention,
id stability, and the startup redelivery sweep's contract:
- pending rows redeliver plainly (send never started, no dup risk)
- attempting/failed rows carry the recovered-reply marker (honest
  at-least-once; ambiguity is labeled, never silently resent)
- rows owned by a LIVE process are never claimed
- poison rows abandon at the attempts cap / stale cutoff
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Isolated state.db per test (autouse HERMES_HOME isolation already
    redirects get_hermes_home; make the redirect explicit and per-test)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


def _record(oid="ob-1", session_key="agent:main:slack:channel:C1", **kw):
    dl.record_obligation(
        obligation_id=oid,
        session_key=session_key,
        platform=kw.get("platform", "slack"),
        chat_id=kw.get("chat_id", "C1"),
        thread_id=kw.get("thread_id", "171.001"),
        content=kw.get("content", "the final answer"),
    )


def _row(oid):
    with dl._connect() as conn:
        r = conn.execute(
            """SELECT state, attempts, owner_pid, content
               FROM delivery_obligations WHERE obligation_id=?""",
            (oid,),
        ).fetchone()
    return None if r is None else {
        "state": r[0], "attempts": r[1], "owner_pid": r[2], "content": r[3],
    }


def _orphan(oid):
    """Make the row look like it belongs to a dead process."""
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?",
            (oid,),
        )


class TestStateMachine:
    def test_record_starts_pending(self):
        _record()
        assert _row("ob-1")["state"] == "pending"


class TestObligationId:
    def test_stable_and_distinct(self):
        a = dl.compute_obligation_id("sk1", "msg1", "hello")
        assert a == dl.compute_obligation_id("sk1", "msg1", "hello")
        # Different thread (baked into session_key) → different id. This is
        # the cron-topic collision class from the earlier outbox attempt.
        assert a != dl.compute_obligation_id("sk1:threadB", "msg1", "hello")
        assert a != dl.compute_obligation_id("sk1", "msg2", "hello")
        assert a != dl.compute_obligation_id("sk1", "msg1", "other")
        assert len(a) == 24


class TestInboundTurnWal:
    def test_preflight_row_rebinds_to_canonical_session(self):
        dl.record_inbound_turn(
            turn_id="turn-rebind",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "hello"},
        )

        dl.rebind_inbound_turn(
            "turn-rebind", "agent:surgical:slack:channel:C1"
        )

        with dl._connect() as conn:
            session_key = conn.execute(
                "SELECT session_key FROM inbound_turns WHERE turn_id='turn-rebind'"
            ).fetchone()[0]
        assert session_key == "agent:surgical:slack:channel:C1"

    def test_orphaned_event_is_claimed_once_with_payload_intact(self):
        payload = {
            "text": "continue the interrupted work",
            "message_type": "text",
            "message_id": "msg-42",
            "media_urls": ["/tmp/input.png"],
        }
        turn_id = dl.compute_inbound_turn_id(
            "agent:main:slack:channel:C1", "msg-42", payload
        )
        dl.record_inbound_turn(
            turn_id=turn_id,
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id="171.001",
            payload=payload,
        )
        with dl._connect() as conn:
            conn.execute(
                "UPDATE inbound_turns SET owner_pid=999999999, "
                "owner_started_at=1 WHERE turn_id=?",
                (turn_id,),
            )

        claimed = dl.sweep_recoverable_inbound_turns()

        assert len(claimed) == 1
        assert claimed[0]["turn_id"] == turn_id
        assert claimed[0]["payload"] == payload
        assert dl.sweep_recoverable_inbound_turns() == []


class TestMultipartDeliveryPlan:
    def test_release_live_claim_allows_one_later_replay(self):
        dl.record_inbound_turn(
            turn_id="turn-release",
            session_key="agent:main:telegram:dm:release",
            platform="telegram",
            chat_id="release",
            thread_id=None,
            payload={"text": "retry me"},
        )

        assert dl.release_inbound_turn_claim("turn-release") is True
        first = dl.sweep_recoverable_inbound_turns()
        second = dl.sweep_recoverable_inbound_turns()

        assert [row["turn_id"] for row in first] == ["turn-release"]
        assert second == []

    def test_fused_secondary_turns_transfer_to_plan_atomically(self):
        session_key = "agent:main:telegram:dm:fused"
        for turn_id in ("turn-primary", "turn-secondary"):
            dl.record_inbound_turn(
                turn_id=turn_id,
                session_key=session_key,
                platform="telegram",
                chat_id="fused",
                thread_id=None,
                payload={"text": turn_id},
            )

        dl.record_delivery_plan(
            turn_id="turn-primary",
            session_key=session_key,
            platform="telegram",
            chat_id="fused",
            thread_id=None,
            components=[{"kind": "text", "payload": {"content": "combined"}}],
            represented_turn_ids=["turn-primary", "turn-secondary"],
        )

        with dl._connect() as conn:
            states = dict(
                conn.execute(
                    "SELECT turn_id, state FROM inbound_turns ORDER BY turn_id"
                ).fetchall()
            )
        assert states == {
            "turn-primary": "received",
            "turn-secondary": "completed",
        }
        recoverable_ids = {
            row["turn_id"] for row in dl.sweep_recoverable_inbound_turns()
        }
        assert "turn-secondary" not in recoverable_ids

    def test_failed_fused_plan_rolls_back_secondary_transfer(self):
        session_key = "agent:main:telegram:dm:fused-failure"
        for turn_id in ("turn-primary", "turn-secondary"):
            dl.record_inbound_turn(
                turn_id=turn_id,
                session_key=session_key,
                platform="telegram",
                chat_id="fused-failure",
                thread_id=None,
                payload={"text": turn_id},
            )

        with pytest.raises(KeyError):
            dl.record_delivery_plan(
                turn_id="turn-primary",
                session_key=session_key,
                platform="telegram",
                chat_id="fused-failure",
                thread_id=None,
                components=[
                    {"kind": "text", "payload": {"content": "combined"}},
                    {},
                ],
                represented_turn_ids=["turn-primary", "turn-secondary"],
            )

        with dl._connect() as conn:
            states = dict(
                conn.execute(
                    "SELECT turn_id, state FROM inbound_turns ORDER BY turn_id"
                ).fetchall()
            )
            component_count = conn.execute(
                "SELECT COUNT(*) FROM delivery_components"
            ).fetchone()[0]
        assert states == {
            "turn-primary": "received",
            "turn-secondary": "received",
        }
        assert component_count == 0

    def test_prune_keeps_terminal_plan_while_inbound_is_active(self, monkeypatch):
        monkeypatch.setattr(dl, "_MAX_ROWS", 1)
        for turn_id in ("turn-protected", "turn-disposable"):
            dl.record_inbound_turn(
                turn_id=turn_id,
                session_key=f"agent:main:telegram:dm:{turn_id}",
                platform="telegram",
                chat_id=turn_id,
                thread_id=None,
                payload={"text": turn_id},
            )
            ids = dl.record_delivery_plan(
                turn_id=turn_id,
                session_key=f"agent:main:telegram:dm:{turn_id}",
                platform="telegram",
                chat_id=turn_id,
                thread_id=None,
                components=[{"kind": "text", "payload": {"content": turn_id}}],
            )
            dl.mark_delivery_component_delivered(ids[0])
        dl.mark_inbound_turn_completed("turn-disposable")

        dl._prune()

        assert dl.delivery_plan_exists("turn-protected") is True
        assert dl.delivery_plan_exists("turn-disposable") is False

    def test_session_ownership_ignores_completed_historical_plan(self):
        session_key = "agent:main:slack:channel:C1"
        turn_id = "turn-historical-complete"
        dl.record_inbound_turn(
            turn_id=turn_id,
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "old"},
        )
        component_id = dl.record_delivery_plan(
            turn_id=turn_id,
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "text", "payload": {"content": "done"}}],
        )[0]

        assert dl.session_has_delivery_plan(session_key) is True
        dl.mark_delivery_component_delivered(component_id)
        assert dl.session_has_delivery_plan(session_key) is True
        dl.mark_inbound_turn_completed(turn_id)
        assert dl.session_has_delivery_plan(session_key) is False

    def test_poison_component_is_abandoned_at_retry_cap(self):
        component_id = dl.record_delivery_plan(
            turn_id="turn-poison",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "file", "payload": {"path": "/missing"}}],
        )[0]
        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_components
                   SET state='failed', attempts=?, owner_pid=999999999,
                       owner_started_at=1 WHERE component_id=?""",
                (dl.MAX_ATTEMPTS, component_id),
            )

        assert dl.sweep_recoverable_delivery_components() == []
        with dl._connect() as conn:
            state = conn.execute(
                "SELECT state FROM delivery_components WHERE component_id=?",
                (component_id,),
            ).fetchone()[0]
        assert state == "abandoned"
        assert dl.delivery_plan_terminal("turn-poison") is True
        assert dl.delivery_plan_complete("turn-poison") is False

    def test_plan_records_every_component_atomically_and_tracks_each_state(self):
        components = [
            {"kind": "text", "payload": {"content": "caption"}},
            {
                "kind": "images",
                "payload": {"images": ["https://example.test/image.png"]},
            },
        ]
        component_ids = dl.record_delivery_plan(
            turn_id="turn-1",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id="171.001",
            components=components,
        )

        assert len(component_ids) == 2
        with dl._connect() as conn:
            rows = conn.execute(
                """SELECT component_id, ordinal, kind, state
                   FROM delivery_components WHERE turn_id='turn-1'
                   ORDER BY ordinal"""
            ).fetchall()
        assert [(row[1], row[2], row[3]) for row in rows] == [
            (0, "text", "pending"),
            (1, "images", "pending"),
        ]

        dl.mark_delivery_component_attempting(component_ids[0])
        dl.mark_delivery_component_delivered(component_ids[0])
        with dl._connect() as conn:
            states = conn.execute(
                """SELECT state FROM delivery_components
                   WHERE turn_id='turn-1' ORDER BY ordinal"""
            ).fetchall()
        assert [row[0] for row in states] == ["delivered", "pending"]


class TestSweep:
    def test_live_owner_rows_never_claimed(self):
        _record()  # owner = this (live) process
        assert dl.sweep_recoverable() == []

    def test_dead_owner_pending_claimed_without_marker(self):
        _record()
        _orphan("ob-1")
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        assert claimed[0]["needs_marker"] is False
        assert claimed[0]["attempts"] == 1
        # Claim re-stamps ownership: a second sweep in the same (live)
        # process must not double-claim.
        assert dl.sweep_recoverable() == []


class TestPrune:
    def test_old_terminal_inbound_and_components_are_pruned(self):
        dl.record_inbound_turn(
            turn_id="turn-old",
            session_key="session",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "x"},
        )
        component_id = dl.record_delivery_plan(
            turn_id="turn-old",
            session_key="session",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "text", "payload": {"content": "x"}}],
        )[0]
        dl.mark_inbound_turn_completed("turn-old")
        dl.mark_delivery_component_delivered(component_id)
        old = time.time() - dl._RETENTION_SECONDS - 60
        with dl._connect() as conn:
            conn.execute(
                "UPDATE inbound_turns SET updated_at=? WHERE turn_id='turn-old'",
                (old,),
            )
            conn.execute(
                "UPDATE delivery_components SET updated_at=? WHERE turn_id='turn-old'",
                (old,),
            )

        dl._prune()

        with dl._connect() as conn:
            assert conn.execute(
                "SELECT 1 FROM inbound_turns WHERE turn_id='turn-old'"
            ).fetchone() is None
            assert conn.execute(
                "SELECT 1 FROM delivery_components WHERE turn_id='turn-old'"
            ).fetchone() is None

    def test_old_delivered_rows_pruned(self):
        _record()
        dl.mark_delivered("ob-1")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET updated_at=? WHERE obligation_id=?",
                (time.time() - dl._RETENTION_SECONDS - 60, "ob-1"),
            )
        dl._prune()
        assert _row("ob-1") is None


class TestLedgerEnabled:
    def test_default_on(self):
        assert dl.ledger_enabled({}) is True
        assert dl.ledger_enabled({"gateway": {}}) is True


class TestGatewayRedeliverySweep:
    """Drive the real GatewayRunner._redeliver_pending_obligations."""

    @staticmethod
    def _runner(adapter=None, platform=None):
        from gateway.config import Platform
        from gateway.run import GatewayRunner

        platform = platform or Platform.SLACK
        runner = object.__new__(GatewayRunner)
        runner.adapters = {platform: adapter} if adapter else {}
        _store = MagicMock()
        _store.clear_resume_pending = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @staticmethod
    def _adapter(success=True):
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=MagicMock(success=success, error="" if success else "nope")
        )
        return adapter

    @pytest.mark.asyncio
    async def test_startup_redelivers_only_unconfirmed_multipart_components(self):
        payload = {
            "text": "original request",
            "message_type": "text",
            "source": {
                "platform": "slack",
                "chat_id": "C1",
                "chat_type": "channel",
                "thread_id": "171.001",
            },
        }
        dl.record_inbound_turn(
            turn_id="turn-1",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id="171.001",
            payload=payload,
        )
        ids = dl.record_delivery_plan(
            turn_id="turn-1",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id="171.001",
            components=[
                {"kind": "text", "payload": {"content": "caption"}},
                {
                    "kind": "image",
                    "payload": {
                        "url": "https://example.test/chart.png",
                        "alt": "chart",
                        "routing_metadata": {
                            "thread_id": "171.001",
                            "slack_team_id": "T-WORKSPACE-2",
                            "notify": True,
                        },
                    },
                },
            ],
        )
        dl.mark_delivery_component_delivered(ids[0])
        with dl._DB_LOCK, dl._transaction() as conn:
            conn.execute(
                "UPDATE delivery_components SET owner_pid=99999999, owner_started_at=1"
            )

        adapter = MagicMock()
        adapter._send_image_with_ack = AsyncMock(
            return_value=MagicMock(success=True, error="")
        )
        adapter._send_with_retry = AsyncMock()
        runner = self._runner(adapter)
        runner.session_store = MagicMock()
        runner.session_store.clear_resume_pending = AsyncMock(return_value=True)
        runner._async_session_store = runner.session_store

        recovered = await runner._redeliver_pending_delivery_components()

        assert recovered == 1
        adapter._send_with_retry.assert_not_awaited()
        runner.session_store.clear_resume_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1"
        )
        adapter._send_image_with_ack.assert_awaited_once_with(
            chat_id="C1",
            image_url="https://example.test/chart.png",
            alt_text="chart",
            metadata={
                "thread_id": "171.001",
                "slack_team_id": "T-WORKSPACE-2",
                "notify": True,
            },
        )
        with dl._connect() as conn:
            inbound_state = conn.execute(
                "SELECT state FROM inbound_turns WHERE turn_id='turn-1'"
            ).fetchone()[0]
        assert inbound_state == "completed"

    @pytest.mark.asyncio
    async def test_startup_redelivery_preserves_telegram_dm_topic_routing(self):
        from gateway.config import Platform

        metadata = {
            "thread_id": "42",
            "telegram_dm_topic_reply_fallback": True,
            "direct_messages_topic_id": "42",
            "telegram_reply_to_message_id": "9001",
            "notify": True,
        }
        dl.record_delivery_plan(
            turn_id="turn-tg-topic",
            session_key="agent:main:telegram:dm:U1:topic:42",
            platform="telegram",
            chat_id="U1",
            thread_id="42",
            components=[
                {
                    "kind": "text",
                    "payload": {
                        "content": "topic reply",
                        "routing_metadata": metadata,
                    },
                }
            ],
        )
        with dl._DB_LOCK, dl._transaction() as conn:
            conn.execute(
                "UPDATE delivery_components SET owner_pid=99999999, owner_started_at=1"
            )

        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock(
            return_value=MagicMock(success=True, error="")
        )
        runner = self._runner(adapter, Platform.TELEGRAM)
        runner.session_store = MagicMock()
        runner.session_store.clear_resume_pending = AsyncMock(return_value=True)
        runner._async_session_store = runner.session_store

        assert await runner._redeliver_pending_delivery_components() == 1
        adapter._send_with_retry.assert_awaited_once_with(
            chat_id="U1",
            content="topic reply",
            metadata=metadata,
        )
        assert dl.delivery_plan_complete("turn-tg-topic") is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tts_state", ["pending", "attempting"])
    async def test_recovered_tts_caption_covers_text_without_duplicate(self, tts_state):
        from gateway.config import Platform

        turn_id = f"turn-tts-{tts_state}"
        ids = dl.record_delivery_plan(
            turn_id=turn_id,
            session_key=f"agent:main:telegram:dm:{tts_state}",
            platform="telegram",
            chat_id=tts_state,
            thread_id=None,
            components=[
                {
                    "kind": "tts",
                    "payload": {
                        "path": "/tmp/recovered.ogg",
                        "caption": "spoken and shown",
                        "covers_text": True,
                        "routing_metadata": {"notify": True},
                    },
                },
                {
                    "kind": "text",
                    "payload": {
                        "content": "spoken and shown",
                        "routing_metadata": {"notify": True},
                    },
                },
            ],
        )
        if tts_state == "attempting":
            dl.mark_delivery_component_attempting(ids[0])
        with dl._DB_LOCK, dl._transaction() as conn:
            conn.execute(
                "UPDATE delivery_components SET owner_pid=99999999, owner_started_at=1"
            )

        adapter = MagicMock()
        adapter.play_tts = AsyncMock(return_value=MagicMock(success=True, error=""))
        adapter._send_with_retry = AsyncMock()
        runner = self._runner(adapter, Platform.TELEGRAM)
        runner.session_store = MagicMock()
        runner.session_store.clear_resume_pending = AsyncMock(return_value=True)
        runner._async_session_store = runner.session_store

        assert await runner._redeliver_pending_delivery_components() == 1
        adapter.play_tts.assert_awaited_once()
        adapter._send_with_retry.assert_not_awaited()
        assert dl.delivery_plan_complete(turn_id) is True

    @pytest.mark.asyncio
    async def test_failed_recovered_tts_leaves_text_as_fallback(self):
        from gateway.config import Platform

        ids = dl.record_delivery_plan(
            turn_id="turn-tts-fallback",
            session_key="agent:main:telegram:dm:fallback",
            platform="telegram",
            chat_id="fallback",
            thread_id=None,
            components=[
                {
                    "kind": "tts",
                    "payload": {
                        "path": "/tmp/recovered.ogg",
                        "caption": "fallback text",
                        "covers_text": True,
                    },
                },
                {"kind": "text", "payload": {"content": "fallback text"}},
            ],
        )
        with dl._DB_LOCK, dl._transaction() as conn:
            conn.execute(
                "UPDATE delivery_components SET owner_pid=99999999, owner_started_at=1"
            )

        adapter = MagicMock()
        adapter.play_tts = AsyncMock(
            return_value=MagicMock(success=False, error="audio failed")
        )
        adapter._send_with_retry = AsyncMock(
            return_value=MagicMock(success=True, error="")
        )
        runner = self._runner(adapter, Platform.TELEGRAM)
        runner.session_store = MagicMock()
        runner.session_store.clear_resume_pending = AsyncMock(return_value=True)
        runner._async_session_store = runner.session_store

        assert await runner._redeliver_pending_delivery_components() == 1
        adapter.play_tts.assert_awaited_once()
        adapter._send_with_retry.assert_awaited_once()
        assert dl.delivery_plan_complete("turn-tts-fallback") is False
        with dl._connect() as conn:
            text_state = conn.execute(
                "SELECT state FROM delivery_components WHERE component_id=?",
                (ids[1],),
            ).fetchone()[0]
        assert text_state == "delivered"

        with dl._DB_LOCK, dl._transaction() as conn:
            conn.execute(
                """UPDATE delivery_components
                   SET owner_pid=99999999, owner_started_at=1
                   WHERE component_id=?""",
                (ids[0],),
            )
        adapter.play_tts.reset_mock()
        adapter.play_tts.return_value = MagicMock(success=True, error="")
        adapter._send_with_retry.reset_mock()

        assert await runner._redeliver_pending_delivery_components() == 1
        adapter.play_tts.assert_awaited_once_with(
            chat_id="fallback",
            audio_path="/tmp/recovered.ogg",
            caption=None,
            metadata={},
        )
        adapter._send_with_retry.assert_not_awaited()
        assert dl.delivery_plan_complete("turn-tts-fallback") is True

    @pytest.mark.asyncio
    async def test_complete_plan_retries_marker_clear_without_rerunning_agent(self):
        dl.record_inbound_turn(
            turn_id="turn-finalize",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "do not rerun", "message_type": "text"},
        )
        ids = dl.record_delivery_plan(
            turn_id="turn-finalize",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "text", "payload": {"content": "done"}}],
        )
        dl.mark_delivery_component_delivered(ids[0])
        with dl._DB_LOCK, dl._transaction() as conn:
            conn.execute(
                "UPDATE inbound_turns SET owner_pid=99999999, owner_started_at=1"
            )

        adapter = MagicMock()
        adapter.handle_message = AsyncMock()
        runner = self._runner(adapter)
        runner._startup_inbound_by_session = {}
        runner.session_store = MagicMock()
        runner.session_store._entries = {}
        runner.session_store.clear_resume_pending = AsyncMock(return_value=True)
        runner._async_session_store = runner.session_store

        recovered = await runner._recover_inbound_turns()

        assert recovered == 1
        adapter.handle_message.assert_not_awaited()
        runner.session_store.clear_resume_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1"
        )
        with dl._connect() as conn:
            state = conn.execute(
                "SELECT state FROM inbound_turns WHERE turn_id='turn-finalize'"
            ).fetchone()[0]
        assert state == "completed"

    @pytest.mark.asyncio
    async def test_pending_redelivers_plain_and_clears_resume(self):
        _record()  # pending
        _orphan("ob-1")
        adapter = self._adapter()
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 1
        sent = adapter.send.call_args.kwargs
        assert sent["content"] == "the final answer"  # no marker
        assert sent["metadata"] == {"thread_id": "171.001"}
        assert _row("ob-1")["state"] == "delivered"
        runner._async_session_store.clear_resume_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1"
        )

    @pytest.mark.asyncio
    async def test_attempting_redelivers_with_marker(self):
        _record()
        dl.mark_attempting("ob-1")
        _orphan("ob-1")
        adapter = self._adapter()
        runner = self._runner(adapter)

        await runner._redeliver_pending_obligations()

        sent = adapter.send.call_args.kwargs
        assert sent["content"].startswith(dl.RECOVERED_MARKER)
        assert sent["content"].endswith("the final answer")


class TestGatewayInboundReplay:
    @pytest.mark.asyncio
    async def test_incomplete_delivery_plan_never_replays_agent(self):
        payload = {
            "text": "do not run twice",
            "message_type": "text",
            "source": {
                "platform": "slack",
                "chat_id": "C1",
                "chat_type": "channel",
            },
        }
        dl.record_inbound_turn(
            turn_id="turn-planned",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload=payload,
        )
        dl.record_delivery_plan(
            turn_id="turn-planned",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "text", "payload": {"content": "owed"}}],
        )
        with dl._connect() as conn:
            conn.execute(
                "UPDATE inbound_turns SET owner_pid=999999999, owner_started_at=1"
            )

        from gateway.config import Platform
        from gateway.run import GatewayRunner

        adapter = MagicMock()
        adapter.handle_message = AsyncMock()
        runner = object.__new__(GatewayRunner)
        runner.adapters = {Platform.SLACK: adapter}
        runner.session_store = MagicMock()
        runner.session_store._entries = {}
        runner._startup_inbound_by_session = {}

        assert await runner._recover_inbound_turns() == 0
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_orphaned_preflight_replays_original_event_once(self):
        payload = {
            "text": "continue the interrupted work",
            "message_type": "text",
            "message_id": "msg-42",
            "platform_update_id": 77,
            "media_urls": ["/tmp/input.png"],
            "media_types": ["image/png"],
            "reply_to_message_id": None,
            "reply_to_text": None,
            "channel_context": None,
            "internal": False,
            "timestamp": "2026-08-01T09:00:00",
            "source": {
                "platform": "slack",
                "chat_id": "C1",
                "chat_type": "channel",
                "thread_id": "171.001",
            },
        }
        turn_id = dl.compute_inbound_turn_id(
            "agent:main:slack:channel:C1", "77", payload
        )
        dl.record_inbound_turn(
            turn_id=turn_id,
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id="171.001",
            payload=payload,
        )
        with dl._connect() as conn:
            conn.execute(
                "UPDATE inbound_turns SET owner_pid=999999999, "
                "owner_started_at=1 WHERE turn_id=?",
                (turn_id,),
            )

        from gateway.config import Platform
        from gateway.run import GatewayRunner

        adapter = MagicMock()
        adapter.handle_message = AsyncMock()
        runner = object.__new__(GatewayRunner)
        runner.adapters = {Platform.SLACK: adapter}
        runner.session_store = MagicMock()
        runner.session_store._entries = {}
        runner._startup_inbound_by_session = {}

        count = await runner._recover_inbound_turns()

        assert count == 1
        adapter.handle_message.assert_awaited_once()
        replay = adapter.handle_message.await_args.args[0]
        assert replay.text == payload["text"]
        assert replay.media_urls == payload["media_urls"]
        assert replay.metadata["_hermes_turn_id"] == turn_id


class TestAttemptsOnlySpentOnRealSends:
    """``attempts`` is the redelivery budget — it must buy a send.

    ``self.adapters`` only holds a platform after its ``connect()`` succeeded,
    and the sweep claimed every dead-owner row regardless. A platform that
    failed to connect this boot therefore burned one attempt per boot while
    the caller's ``adapter is None`` branch skipped it without sending — so
    after MAX_ATTEMPTS boots the row abandoned having never been sent once,
    losing exactly the response the ledger exists to guarantee. That failure
    correlates with the crash that created the obligation: the network
    trouble that killed the send tends to still be there on the next boot.
    """

    def test_absent_platform_does_not_burn_attempts(self):
        _record(platform="telegram")
        dl.mark_attempting("ob-1")

        for _ in range(dl.MAX_ATTEMPTS + 2):
            _orphan("ob-1")
            assert dl.sweep_recoverable(deliverable_platforms={"discord"}) == []

        row = dl.debug_rows()
        assert "abandoned" not in row
        with dl._connect() as conn:
            state, attempts = conn.execute(
                "SELECT state, attempts FROM delivery_obligations "
                "WHERE obligation_id=?", ("ob-1",),
            ).fetchone()
        assert attempts == 0, "an unsendable boot must not spend the budget"
        assert state == "attempting"

    def test_row_still_delivers_once_its_platform_returns(self):
        _record(platform="telegram")
        for _ in range(dl.MAX_ATTEMPTS + 2):
            _orphan("ob-1")
            dl.sweep_recoverable(deliverable_platforms={"discord"})

        _orphan("ob-1")
        claimed = dl.sweep_recoverable(deliverable_platforms={"telegram"})
        assert len(claimed) == 1
        assert claimed[0]["attempts"] == 1


class TestUnconnectedPlatformKeepsItsBudget:
    """End-to-end through the real runner: boots where the platform failed to
    connect must not consume the row's redelivery budget."""

    @staticmethod
    def _runner_without_slack():
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.adapters = {}  # slack failed to connect this boot
        _store = MagicMock()
        _store.clear_resume_pending = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @pytest.mark.asyncio
    async def test_row_survives_boots_where_its_platform_is_down(self):
        _record(platform="slack")
        dl.mark_attempting("ob-1")

        for _ in range(dl.MAX_ATTEMPTS + 1):
            _orphan("ob-1")
            runner = self._runner_without_slack()
            assert await runner._redeliver_pending_obligations() == 0

        assert _row("ob-1")["state"] != "abandoned", (
            "the obligation was abandoned without a single send being attempted"
        )
        assert _row("ob-1")["attempts"] == 0

