"""Tests for the gateway delivery-obligation ledger (gateway/delivery_ledger.py).

State machine, dead-owner claiming, attempts cap, stale cutoff, retention,
id stability, and the startup redelivery sweep's contract:
- pending rows redeliver plainly (send never started, no dup risk)
- attempting/failed rows carry the recovered-reply marker (honest
  at-least-once; ambiguity is labeled, never silently resent)
- rows owned by a LIVE process are never claimed
- poison rows abandon at the attempts cap / stale cutoff
"""

import sqlite3
import time
from pathlib import Path
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
    def test_platform_redelivery_keeps_one_turn_id_when_timestamp_changes(self):
        first = {
            "text": "continue",
            "message_type": "text",
            "platform_update_id": 4242,
            "timestamp": "2026-08-09T12:00:00-03:00",
        }
        redelivered = {
            **first,
            "timestamp": "2026-08-09T12:00:05-03:00",
        }

        first_id = dl.compute_inbound_turn_id(
            "agent:main:telegram:dm:C1", "4242", first
        )
        replay_id = dl.compute_inbound_turn_id(
            "agent:main:telegram:dm:C1", "4242", redelivered
        )

        assert first_id == replay_id

    def test_reference_boundaries_do_not_collide(self):
        payload = {"text": "continue"}

        left = dl.compute_inbound_turn_id("a|b", "c", payload)
        right = dl.compute_inbound_turn_id("a", "b|c", payload)

        assert left != right

    def test_unversioned_historical_tables_are_quarantined_before_use(self):
        import sqlite3

        path = dl._db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE inbound_turns (legacy TEXT)")
            conn.execute("INSERT INTO inbound_turns VALUES ('do-not-replay')")
            conn.execute("CREATE TABLE delivery_components (legacy TEXT)")

        dl.record_inbound_turn(
            turn_id="turn-fresh-schema",
            session_key="session",
            platform="telegram",
            chat_id="C1",
            thread_id=None,
            payload={"text": "fresh"},
        )

        with dl._connect() as conn:
            row = conn.execute(
                "SELECT state FROM inbound_turns WHERE turn_id='turn-fresh-schema'"
            ).fetchone()
            archives = {
                name
                for (name,) in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name LIKE '%_pre_durable_v1%'"
                ).fetchall()
            }
        assert row == ("received",)
        assert any(name.startswith("inbound_turns_pre_durable_v1") for name in archives)
        assert any(
            name.startswith("delivery_components_pre_durable_v1")
            for name in archives
        )

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

    def test_all_represented_rows_rebind_atomically_to_canonical_session(self):
        quick = "agent:main:telegram:dm:represented"
        canonical = "agent:surgical:telegram:dm:represented"
        for turn_id in ("turn-primary", "turn-followup"):
            dl.record_inbound_turn(
                turn_id=turn_id,
                session_key=quick,
                platform="telegram",
                chat_id="represented",
                thread_id=None,
                payload={"text": turn_id},
            )

        dl.rebind_inbound_turns(
            ["turn-primary", "turn-followup"],
            expected_session_key=quick,
            canonical_session_key=canonical,
        )

        with dl._connect() as conn:
            rows = dict(conn.execute(
                "SELECT turn_id, session_key FROM inbound_turns ORDER BY turn_id"
            ).fetchall())
        assert rows == {
            "turn-followup": canonical,
            "turn-primary": canonical,
        }

    def test_rebind_mismatch_rolls_back_every_represented_row(self):
        quick = "agent:main:telegram:dm:represented"
        other = "agent:other:telegram:dm:represented"
        canonical = "agent:surgical:telegram:dm:represented"
        for turn_id, session_key in (
            ("turn-primary", quick),
            ("turn-wrong-owner", other),
        ):
            dl.record_inbound_turn(
                turn_id=turn_id,
                session_key=session_key,
                platform="telegram",
                chat_id="represented",
                thread_id=None,
                payload={"text": turn_id},
            )

        with pytest.raises(ValueError, match="session ownership mismatch"):
            dl.rebind_inbound_turns(
                ["turn-primary", "turn-wrong-owner"],
                expected_session_key=quick,
                canonical_session_key=canonical,
            )

        with dl._connect() as conn:
            rows = dict(conn.execute(
                "SELECT turn_id, session_key FROM inbound_turns ORDER BY turn_id"
            ).fetchall())
        assert rows == {
            "turn-primary": quick,
            "turn-wrong-owner": other,
        }

    def test_completion_is_atomic_and_bound_to_canonical_session(self):
        canonical = "agent:surgical:telegram:dm:represented"
        other = "agent:other:telegram:dm:represented"
        for turn_id, session_key in (
            ("turn-primary", canonical),
            ("turn-wrong-owner", other),
        ):
            dl.record_inbound_turn(
                turn_id=turn_id,
                session_key=session_key,
                platform="telegram",
                chat_id="represented",
                thread_id=None,
                payload={"text": turn_id},
            )

        with pytest.raises(ValueError, match="session ownership mismatch"):
            dl.mark_inbound_turns_completed(
                ["turn-primary", "turn-wrong-owner"],
                session_key=canonical,
            )

        with dl._connect() as conn:
            states = dict(conn.execute(
                "SELECT turn_id, state FROM inbound_turns ORDER BY turn_id"
            ).fetchall())
        assert states == {
            "turn-primary": "received",
            "turn-wrong-owner": "received",
        }

        dl.mark_inbound_turns_completed(["turn-primary"], session_key=canonical)
        with dl._connect() as conn:
            state = conn.execute(
                "SELECT state FROM inbound_turns WHERE turn_id='turn-primary'"
            ).fetchone()[0]
        assert state == "completed"

    def test_single_completion_rejects_wrong_canonical_session(self):
        canonical = "agent:surgical:telegram:dm:single"
        dl.record_inbound_turn(
            turn_id="turn-single-owner",
            session_key=canonical,
            platform="telegram",
            chat_id="single",
            thread_id=None,
            payload={"text": "single"},
        )

        with pytest.raises(ValueError, match="session ownership mismatch"):
            dl.mark_inbound_turn_completed(
                "turn-single-owner",
                session_key="agent:other:telegram:dm:single",
            )

        with dl._connect() as conn:
            state = conn.execute(
                "SELECT state FROM inbound_turns WHERE turn_id='turn-single-owner'"
            ).fetchone()[0]
        assert state == "received"


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

        with pytest.raises(ValueError, match="unsupported delivery component kind"):
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

    def test_fused_plan_rejects_foreign_or_missing_represented_rows_atomically(self):
        canonical = "agent:surgical:telegram:dm:fused-owner"
        dl.record_inbound_turn(
            turn_id="turn-primary",
            session_key=canonical,
            platform="telegram",
            chat_id="fused-owner",
            thread_id=None,
            payload={"text": "primary"},
        )
        dl.record_inbound_turn(
            turn_id="turn-foreign",
            session_key="agent:other:telegram:dm:fused-owner",
            platform="telegram",
            chat_id="fused-owner",
            thread_id=None,
            payload={"text": "foreign"},
        )

        with pytest.raises(ValueError, match="session ownership mismatch"):
            dl.record_delivery_plan(
                turn_id="turn-primary",
                session_key=canonical,
                platform="telegram",
                chat_id="fused-owner",
                thread_id=None,
                components=[{"kind": "text", "payload": {"content": "combined"}}],
                represented_turn_ids=[
                    "turn-primary", "turn-foreign", "turn-missing"
                ],
            )

        with dl._connect() as conn:
            states = dict(conn.execute(
                "SELECT turn_id, state FROM inbound_turns ORDER BY turn_id"
            ).fetchall())
            component_count = conn.execute(
                "SELECT COUNT(*) FROM delivery_components"
            ).fetchone()[0]
        assert states == {
            "turn-foreign": "received",
            "turn-primary": "received",
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
        dl.mark_inbound_turn_completed(
            "turn-disposable",
            session_key="agent:main:telegram:dm:turn-disposable",
        )

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
        dl.mark_inbound_turn_completed(turn_id, session_key=session_key)
        assert dl.session_has_delivery_plan(session_key) is False

    def test_poison_component_is_abandoned_at_retry_cap(self):
        component_id = dl.record_delivery_plan(
            turn_id="turn-poison",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "document", "payload": {"path": "/missing"}}],
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

    def test_unacknowledged_tts_artifact_survives_retry_abandonment(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / ".hermes"
        outbox = home / "cache" / "delivery_outbox" / "turn-tts"
        outbox.mkdir(parents=True)
        artifact = outbox / "voice.ogg"
        artifact.write_bytes(b"recoverable voice")
        monkeypatch.setattr(dl, "get_hermes_home", lambda: home)
        component_id = dl.record_delivery_plan(
            turn_id="turn-tts",
            session_key="agent:main:telegram:dm:C1",
            platform="telegram",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "tts", "payload": {"path": str(artifact)}}],
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
        assert Path(artifact).read_bytes() == b"recoverable voice"

    def test_abandoned_tts_row_and_artifact_are_reclaimed_after_retention(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / ".hermes"
        outbox = home / "cache" / "delivery_outbox" / "turn-retained-tts"
        outbox.mkdir(parents=True)
        artifact = outbox / "voice.ogg"
        artifact.write_bytes(b"recoverable voice")
        monkeypatch.setattr(dl, "get_hermes_home", lambda: home)
        component_id = dl.record_delivery_plan(
            turn_id="turn-retained-tts",
            session_key="agent:main:telegram:dm:C1",
            platform="telegram",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "tts", "payload": {"path": str(artifact)}}],
        )[0]
        old = time.time() - dl._RETENTION_SECONDS - 1
        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_components
                   SET state='abandoned', updated_at=? WHERE component_id=?""",
                (old, component_id),
            )

        dl._prune()

        with dl._connect() as conn:
            row = conn.execute(
                "SELECT state, payload FROM delivery_components WHERE component_id=?",
                (component_id,),
            ).fetchone()
        assert row is None
        assert artifact.exists() is False

    def test_post_commit_cleanup_rechecks_inbound_ownership_before_unlink(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / ".hermes"
        outbox = home / "cache" / "delivery_outbox" / "turn-raced-tts"
        outbox.mkdir(parents=True)
        artifact = outbox / "voice.ogg"
        artifact.write_bytes(b"reclaimed ownership")
        monkeypatch.setattr(dl, "get_hermes_home", lambda: home)
        turn_id = "turn-raced-tts"
        session_key = "agent:main:telegram:dm:C1"
        component_id = dl.record_delivery_plan(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "tts", "payload": {"path": str(artifact)}}],
        )[0]
        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_components
                   SET state='abandoned', updated_at=? WHERE component_id=?""",
                (time.time() - dl._RETENTION_SECONDS - 1, component_id),
            )
        original_cleanup = dl._cleanup_pruned_component
        inserted = False

        def reclaim_before_unlink(cleanup_turn_id, kind, payload):
            nonlocal inserted
            if not inserted:
                inserted = True
                dl.record_inbound_turn(
                    turn_id=turn_id,
                    session_key=session_key,
                    platform="telegram",
                    chat_id="C1",
                    thread_id=None,
                    payload={"text": "retry"},
                )
            original_cleanup(cleanup_turn_id, kind, payload)

        monkeypatch.setattr(dl, "_cleanup_pruned_component", reclaim_before_unlink)

        dl._prune()

        with dl._connect() as conn:
            inbound = conn.execute(
                "SELECT state FROM inbound_turns WHERE turn_id=?", (turn_id,)
            ).fetchone()
        assert inbound == ("received",)
        assert artifact.read_bytes() == b"reclaimed ownership"

    def test_post_commit_cleanup_failure_does_not_escape_prune(self, monkeypatch):
        component_id = dl.record_delivery_plan(
            turn_id="turn-cleanup-failure",
            session_key="agent:main:telegram:dm:C1",
            platform="telegram",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "tts", "payload": {"path": "/missing"}}],
        )[0]
        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_components
                   SET state='abandoned', updated_at=? WHERE component_id=?""",
                (time.time() - dl._RETENTION_SECONDS - 1, component_id),
            )

        def fail_cleanup(*_args):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(dl, "_cleanup_pruned_component", fail_cleanup)

        dl._prune()

        with dl._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM delivery_components WHERE component_id=?",
                (component_id,),
            ).fetchone()
        assert row is None

    def test_abandoned_tts_retention_does_not_prune_active_inbound_ownership(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / ".hermes"
        outbox = home / "cache" / "delivery_outbox" / "turn-active-tts"
        outbox.mkdir(parents=True)
        artifact = outbox / "voice.ogg"
        artifact.write_bytes(b"still owned")
        monkeypatch.setattr(dl, "get_hermes_home", lambda: home)
        turn_id = "turn-active-tts"
        session_key = "agent:main:telegram:dm:C1"
        dl.record_inbound_turn(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="C1",
            thread_id=None,
            payload={"text": "speak"},
        )
        component_id = dl.record_delivery_plan(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "tts", "payload": {"path": str(artifact)}}],
            represented_turn_ids=[turn_id],
        )[0]
        old = time.time() - dl._RETENTION_SECONDS - 1
        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_components
                   SET state='abandoned', updated_at=? WHERE component_id=?""",
                (old, component_id),
            )

        dl._prune()

        with dl._connect() as conn:
            row = conn.execute(
                "SELECT state FROM delivery_components WHERE component_id=?",
                (component_id,),
            ).fetchone()
        assert row == ("abandoned",)
        assert artifact.read_bytes() == b"still owned"

    def test_abandoned_tts_row_and_artifact_are_reclaimed_by_row_cap(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / ".hermes"
        outbox = home / "cache" / "delivery_outbox" / "turn-capped-tts"
        outbox.mkdir(parents=True)
        artifact = outbox / "voice.ogg"
        artifact.write_bytes(b"bounded evidence")
        monkeypatch.setattr(dl, "get_hermes_home", lambda: home)
        monkeypatch.setattr(dl, "_MAX_ROWS", 1)
        component_id = dl.record_delivery_plan(
            turn_id="turn-capped-tts",
            session_key="agent:main:telegram:dm:C1",
            platform="telegram",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "tts", "payload": {"path": str(artifact)}}],
        )[0]
        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_components
                   SET state='abandoned', updated_at=? WHERE component_id=?""",
                (time.time() - 10, component_id),
            )
            conn.execute(
                """INSERT INTO delivery_components
                   (component_id, turn_id, session_key, platform, chat_id,
                    thread_id, ordinal, kind, payload, state, attempts,
                    created_at, updated_at, owner_pid, owner_started_at)
                   VALUES ('newer-terminal', 'newer-turn', 'newer-session',
                           'telegram', 'C1', NULL, 0, 'text', '{}', 'delivered',
                           0, ?, ?, NULL, NULL)""",
                (time.time(), time.time()),
            )

        dl._prune()

        with dl._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM delivery_components WHERE component_id=?",
                (component_id,),
            ).fetchone()
        assert row is None
        assert artifact.exists() is False

    def test_plan_records_every_component_atomically_and_tracks_each_state(self):
        components = [
            {"kind": "text", "payload": {"content": "caption"}},
            {
                "kind": "image",
                "payload": {"url": "https://example.test/image.png"},
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
            (1, "image", "pending"),
        ]

        dl.mark_delivery_component_attempting(component_ids[0])
        dl.mark_delivery_component_delivered(component_ids[0])
        with dl._connect() as conn:
            states = conn.execute(
                """SELECT state FROM delivery_components
                   WHERE turn_id='turn-1' ORDER BY ordinal"""
            ).fetchall()
        assert [row[0] for row in states] == ["delivered", "pending"]

    def test_attempting_transition_rejects_missing_or_terminal_component(self):
        component_id = dl.record_delivery_plan(
            turn_id="turn-terminal",
            session_key="agent:main:telegram:dm:terminal",
            platform="telegram",
            chat_id="terminal",
            thread_id=None,
            components=[{"kind": "text", "payload": {"content": "done"}}],
        )[0]
        dl.mark_delivery_component_attempting(component_id)
        dl.mark_delivery_component_delivered(component_id)

        with pytest.raises(ValueError, match="cannot transition to attempting"):
            dl.mark_delivery_component_attempting(component_id)
        with pytest.raises(ValueError, match="cannot transition to attempting"):
            dl.mark_delivery_component_attempting("missing-component")

    def test_idempotent_plan_reuse_rejects_component_content_collision(self):
        session_key = "agent:main:telegram:dm:collision"
        turn_id = "turn-component-collision"
        dl.record_inbound_turn(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="collision",
            thread_id=None,
            payload={"text": "request"},
        )
        original_ids = dl.record_delivery_plan(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="collision",
            thread_id=None,
            components=[{"kind": "text", "payload": {"content": "original"}}],
        )

        with pytest.raises(ValueError, match="delivery component collision"):
            dl.record_delivery_plan(
                turn_id=turn_id,
                session_key=session_key,
                platform="telegram",
                chat_id="collision",
                thread_id=None,
                components=[{"kind": "text", "payload": {"content": "changed"}}],
            )

        with dl._connect() as conn:
            rows = conn.execute(
                "SELECT component_id, payload FROM delivery_components WHERE turn_id=?",
                (turn_id,),
            ).fetchall()
        assert rows == [(original_ids[0], '{"content":"original"}')]

    def test_single_represented_turn_rejects_wrong_canonical_session(self):
        canonical = "agent:surgical:telegram:dm:single-plan"
        turn_id = "turn-single-plan-owner"
        dl.record_inbound_turn(
            turn_id=turn_id,
            session_key=canonical,
            platform="telegram",
            chat_id="single-plan",
            thread_id=None,
            payload={"text": "request"},
        )

        with pytest.raises(ValueError, match="session ownership mismatch"):
            dl.record_delivery_plan(
                turn_id=turn_id,
                session_key="agent:other:telegram:dm:single-plan",
                platform="telegram",
                chat_id="single-plan",
                thread_id=None,
                components=[{"kind": "text", "payload": {"content": "unsafe"}}],
                represented_turn_ids=[turn_id],
            )

        with dl._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM delivery_components WHERE turn_id=?",
                (turn_id,),
            ).fetchone()[0]
        assert count == 0

    def test_idempotent_plan_reuse_rejects_component_cardinality_change(self):
        session_key = "agent:main:telegram:dm:cardinality"
        turn_id = "turn-component-cardinality"
        dl.record_inbound_turn(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="cardinality",
            thread_id=None,
            payload={"text": "request"},
        )
        dl.record_delivery_plan(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="cardinality",
            thread_id=None,
            components=[
                {"kind": "text", "payload": {"content": "one"}},
                {"kind": "text", "payload": {"content": "two"}},
            ],
        )

        with pytest.raises(ValueError, match="delivery plan cardinality collision"):
            dl.record_delivery_plan(
                turn_id=turn_id,
                session_key=session_key,
                platform="telegram",
                chat_id="cardinality",
                thread_id=None,
                components=[{"kind": "text", "payload": {"content": "one"}}],
            )


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
    def test_row_cap_never_prunes_active_legacy_obligations(self, monkeypatch):
        monkeypatch.setattr(dl, "_MAX_ROWS", 1)
        _record("active-1", content="one")
        _record("active-2", content="two")

        dl._prune()

        first = _row("active-1")
        second = _row("active-2")
        assert first is not None and first["state"] == "pending"
        assert second is not None and second["state"] == "pending"

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
        dl.mark_inbound_turn_completed("turn-old", session_key="session")
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


class TestCanonicalDeliveryComponentKinds:
    @pytest.mark.parametrize("kind", ["media", "file", "unknown", "", None])
    def test_record_plan_rejects_obsolete_or_unknown_kind_atomically(self, kind):
        session_key = "agent:main:slack:channel:C1"
        for turn_id in ("turn-primary", "turn-secondary"):
            dl.record_inbound_turn(
                turn_id=turn_id,
                session_key=session_key,
                platform="slack",
                chat_id="C1",
                thread_id=None,
                payload={"text": turn_id},
            )

        with pytest.raises(ValueError, match="unsupported delivery component kind"):
            dl.record_delivery_plan(
                turn_id="turn-primary",
                session_key=session_key,
                platform="slack",
                chat_id="C1",
                thread_id=None,
                components=[
                    {"kind": "text", "payload": {"content": "partial"}},
                    {"kind": kind, "payload": {"path": "/tmp/unsafe"}},
                ],
                represented_turn_ids=["turn-primary", "turn-secondary"],
            )

        with dl._connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM delivery_components").fetchone()[0] == 0
            states = conn.execute(
                "SELECT turn_id, state FROM inbound_turns ORDER BY turn_id"
            ).fetchall()
        assert states == [
            ("turn-primary", "received"),
            ("turn-secondary", "received"),
        ]

    @pytest.mark.parametrize(
        "component",
        [
            {"kind": "text", "payload": {}},
            {"kind": "image", "payload": {"url": ""}},
            {"kind": "tts", "payload": {"path": ""}},
            {"kind": "voice", "payload": {}},
            {"kind": "video", "payload": []},
            {"kind": "document", "payload": None},
            {"kind": "text", "payload": {"content": "   "}},
        ],
    )
    def test_record_plan_rejects_invalid_payload_before_writing(self, component):
        with pytest.raises(ValueError, match="invalid delivery component payload"):
            dl.record_delivery_plan(
                turn_id="turn-invalid-payload",
                session_key="agent:main:slack:channel:C1",
                platform="slack",
                chat_id="C1",
                thread_id=None,
                components=[component],
            )
        with dl._connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM delivery_components").fetchone()[0] == 0


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
    async def test_startup_recovery_fails_closed_for_corrupt_unknown_kind(self):
        session_key = "agent:main:slack:channel:C1"
        dl.record_inbound_turn(
            turn_id="turn-corrupt-kind",
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "request"},
        )
        component_id = dl.record_delivery_plan(
            turn_id="turn-corrupt-kind",
            session_key=session_key,
            platform="slack",
            chat_id="C1",
            thread_id=None,
            components=[
                {"kind": "document", "payload": {"path": "/tmp/report.pdf"}}
            ],
        )[0]
        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_components
                   SET kind='unknown', owner_pid=99999999, owner_started_at=1
                   WHERE component_id=?""",
                (component_id,),
            )

        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock()
        adapter._send_image_with_ack = AsyncMock()
        adapter.play_tts = AsyncMock()
        adapter.send_voice = AsyncMock()
        adapter.send_video = AsyncMock()
        adapter.send_document = AsyncMock()
        runner = self._runner(adapter)

        assert await runner._redeliver_pending_delivery_components() == 0
        for sender in (
            adapter._send_with_retry,
            adapter._send_image_with_ack,
            adapter.play_tts,
            adapter.send_voice,
            adapter.send_video,
            adapter.send_document,
        ):
            sender.assert_not_awaited()
        with dl._connect() as conn:
            state, error = conn.execute(
                "SELECT state, last_error FROM delivery_components WHERE component_id=?",
                (component_id,),
            ).fetchone()
            inbound_state = conn.execute(
                "SELECT state FROM inbound_turns WHERE turn_id='turn-corrupt-kind'"
            ).fetchone()[0]
        assert state == "failed"
        assert "unsupported delivery component kind" in error
        assert inbound_state == "received"

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
        dl.record_inbound_turn(
            turn_id="turn-tg-topic",
            session_key="agent:main:telegram:dm:U1:topic:42",
            platform="telegram",
            chat_id="U1",
            thread_id="42",
            payload={"text": "topic request"},
        )
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
        session_key = f"agent:main:telegram:dm:{tts_state}"
        dl.record_inbound_turn(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id=tts_state,
            thread_id=None,
            payload={"text": "speak"},
        )
        ids = dl.record_delivery_plan(
            turn_id=turn_id,
            session_key=session_key,
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
        adapter._send_with_retry = AsyncMock(
            return_value=MagicMock(success=True, error="")
        )
        runner = self._runner(adapter, Platform.TELEGRAM)
        runner.session_store = MagicMock()
        runner.session_store.clear_resume_pending = AsyncMock(return_value=True)
        runner._async_session_store = runner.session_store

        assert await runner._redeliver_pending_delivery_components() == 1
        adapter.play_tts.assert_awaited_once()
        if tts_state == "attempting":
            adapter._send_with_retry.assert_awaited_once()
            assert (
                "Recovered reply"
                in adapter._send_with_retry.await_args.kwargs["content"]
            )
        else:
            adapter._send_with_retry.assert_not_awaited()
        assert dl.delivery_plan_complete(turn_id) is True

    @pytest.mark.asyncio
    async def test_failed_recovered_tts_leaves_text_as_fallback(self):
        from gateway.config import Platform

        dl.record_inbound_turn(
            turn_id="turn-tts-fallback",
            session_key="agent:main:telegram:dm:fallback",
            platform="telegram",
            chat_id="fallback",
            thread_id=None,
            payload={"text": "fallback"},
        )
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
        adapter._send_with_retry.assert_awaited_once()
        assert (
            "Recovered reply"
            in adapter._send_with_retry.await_args.kwargs["content"]
        )
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
    async def test_terminal_plan_finalizes_when_resume_marker_is_already_clear(self):
        dl.record_inbound_turn(
            turn_id="turn-already-clear",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={"text": "do not rerun", "message_type": "text"},
        )
        component_id = dl.record_delivery_plan(
            turn_id="turn-already-clear",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "text", "payload": {"content": "done"}}],
        )[0]
        dl.mark_delivery_component_delivered(component_id)
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
        runner.session_store.clear_resume_pending = AsyncMock(return_value=False)
        runner._async_session_store = runner.session_store

        assert await runner._recover_inbound_turns() == 1
        adapter.handle_message.assert_not_awaited()
        with dl._connect() as conn:
            state = conn.execute(
                "SELECT state FROM inbound_turns WHERE turn_id='turn-already-clear'"
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
    async def test_disabled_delivery_ledger_skips_component_redelivery(
        self, monkeypatch
    ):
        component_id = dl.record_delivery_plan(
            turn_id="turn-disabled-component",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            components=[{"kind": "text", "payload": {"content": "do not send"}}],
        )[0]
        monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: False)
        sweep = MagicMock(side_effect=AssertionError("disabled ledger was swept"))
        monkeypatch.setattr(dl, "sweep_recoverable_delivery_components", sweep)

        from gateway.config import Platform
        from gateway.run import GatewayRunner

        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock()
        runner = object.__new__(GatewayRunner)
        runner.adapters = {Platform.SLACK: adapter}

        assert await runner._redeliver_pending_delivery_components() == 0
        sweep.assert_not_called()
        adapter._send_with_retry.assert_not_awaited()
        with dl._connect() as conn:
            state = conn.execute(
                "SELECT state FROM delivery_components WHERE component_id=?",
                (component_id,),
            ).fetchone()[0]
        assert state == "pending"

    @pytest.mark.asyncio
    async def test_disabled_delivery_ledger_skips_startup_inbound_recovery(
        self, monkeypatch
    ):
        dl.record_inbound_turn(
            turn_id="turn-disabled-recovery",
            session_key="agent:main:slack:channel:C1",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            payload={
                "text": "do not replay",
                "message_type": "text",
                "source": {
                    "platform": "slack",
                    "chat_id": "C1",
                    "chat_type": "channel",
                },
            },
        )
        with dl._connect() as conn:
            conn.execute(
                "UPDATE inbound_turns SET owner_pid=999999999, owner_started_at=1"
            )
        monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: False)

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
        with dl._connect() as conn:
            state = conn.execute(
                "SELECT state FROM inbound_turns "
                "WHERE turn_id='turn-disabled-recovery'"
            ).fetchone()[0]
        assert state == "received"

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


class TestPostDeliveryEffects:
    def test_effect_ordinals_are_global_durable_and_late_effect_appends(self):
        kwargs = {
            "turn_id": "turn-ordered-effects",
            "session_key": "agent:main:telegram:dm:effects",
            "platform": "telegram",
            "chat_id": "effects",
            "thread_id": None,
        }
        effects = [
            {
                "effect_key": "goal-status",
                "kind": "goal_status_notice",
                "payload": {"content": "Goal achieved"},
            },
            {
                "effect_key": "background-review:0",
                "kind": "background_review_notice",
                "payload": {"content": "Memory updated"},
            },
        ]
        dl.record_delivery_plan_with_post_delivery_effects(
            **kwargs,
            components=[{"kind": "text", "payload": {"content": "answer"}}],
            post_delivery_effects=effects,
        )
        late_id = dl.record_post_delivery_effect(
            **kwargs,
            effect_key="background-review:1",
            kind="background_review_notice",
            payload={"content": "Second update"},
        )
        assert dl.record_post_delivery_effect(
            **kwargs,
            effect_key="background-review:1",
            kind="background_review_notice",
            payload={"content": "Second update"},
        ) == late_id

        with dl._connect() as conn:
            rows = conn.execute(
                "SELECT effect_key, ordinal FROM post_delivery_effects "
                "WHERE turn_id=? ORDER BY ordinal",
                (kwargs["turn_id"],),
            ).fetchall()
        assert rows == [
            ("goal-status", 0),
            ("background-review:0", 1),
            ("background-review:1", 2),
        ]

        with pytest.raises(ValueError, match="post-delivery effect collision"):
            dl.record_delivery_plan_with_post_delivery_effects(
                **kwargs,
                components=[{"kind": "text", "payload": {"content": "answer"}}],
                post_delivery_effects=list(reversed(effects)),
            )

    def test_competing_owners_cannot_skip_or_ack_an_effect_head(self, monkeypatch):
        kwargs = {
            "turn_id": "turn-competing-effect-owners",
            "session_key": "agent:main:telegram:dm:effects",
            "platform": "telegram",
            "chat_id": "effects",
            "thread_id": None,
        }
        component_id = dl.record_delivery_plan_with_post_delivery_effects(
            **kwargs,
            components=[{"kind": "text", "payload": {"content": "answer"}}],
            post_delivery_effects=[
                {
                    "effect_key": "goal-status",
                    "kind": "goal_status_notice",
                    "payload": {"content": "Goal achieved"},
                },
                {
                    "effect_key": "background-review:0",
                    "kind": "background_review_notice",
                    "payload": {"content": "Memory updated"},
                },
            ],
        )[0]
        dl.mark_delivery_component_delivered(component_id)
        owner_a = (101, 1001)
        owner_b = (202, 2002)
        monkeypatch.setattr(dl, "_owner_stamp", lambda: owner_a)
        monkeypatch.setattr(
            dl, "_owner_alive", lambda pid, started: (pid, started) == owner_a
        )
        first = dl.sweep_ready_post_delivery_effects()
        assert [(row["effect_key"], row["ordinal"]) for row in first] == [
            ("goal-status", 0)
        ]

        monkeypatch.setattr(dl, "_owner_stamp", lambda: owner_b)
        assert dl.sweep_ready_post_delivery_effects() == []
        with pytest.raises(ValueError, match="delivery claim mismatch"):
            dl.mark_post_delivery_effect_delivered(first[0]["effect_id"])
        with pytest.raises(ValueError, match="failure claim mismatch"):
            dl.mark_post_delivery_effect_failed(first[0]["effect_id"], "stale")

        monkeypatch.setattr(dl, "_owner_alive", lambda _pid, _started: False)
        takeover = dl.sweep_ready_post_delivery_effects()
        assert len(takeover) == 1
        assert takeover[0]["effect_key"] == "goal-status"
        assert takeover[0]["prior_state"] == "attempting"
        assert takeover[0]["attempts"] == 2

        monkeypatch.setattr(dl, "_owner_stamp", lambda: owner_a)
        with pytest.raises(ValueError, match="delivery claim mismatch"):
            dl.mark_post_delivery_effect_delivered(takeover[0]["effect_id"])
        monkeypatch.setattr(dl, "_owner_stamp", lambda: owner_b)
        dl.mark_post_delivery_effect_delivered(takeover[0]["effect_id"])
        second = dl.sweep_ready_post_delivery_effects()
        assert [(row["effect_key"], row["ordinal"]) for row in second] == [
            ("background-review:0", 1)
        ]

    def test_abandoned_effect_head_cascades_without_sending_successors(self):
        kwargs = {
            "turn_id": "turn-abandoned-effect-head",
            "session_key": "agent:main:telegram:dm:effects",
            "platform": "telegram",
            "chat_id": "effects",
            "thread_id": None,
        }
        component_id = dl.record_delivery_plan_with_post_delivery_effects(
            **kwargs,
            components=[{"kind": "text", "payload": {"content": "answer"}}],
            post_delivery_effects=[
                {
                    "effect_key": "head",
                    "kind": "goal_status_notice",
                    "payload": {"content": "Head"},
                },
                {
                    "effect_key": "successor",
                    "kind": "background_review_notice",
                    "payload": {"content": "Successor"},
                },
            ],
        )[0]
        dl.mark_delivery_component_delivered(component_id)
        with dl._connect() as conn:
            conn.execute(
                """UPDATE post_delivery_effects
                   SET state='failed', attempts=?, owner_pid=NULL,
                       owner_started_at=NULL
                   WHERE turn_id=? AND ordinal=0""",
                (dl.MAX_ATTEMPTS, kwargs["turn_id"]),
            )

        assert dl.sweep_ready_post_delivery_effects() == []
        with dl._connect() as conn:
            states = conn.execute(
                """SELECT state FROM post_delivery_effects
                   WHERE turn_id=? ORDER BY ordinal""",
                (kwargs["turn_id"],),
            ).fetchall()
        assert states == [("abandoned",), ("abandoned",)]

    def test_atomic_plan_and_effects_commit_together(self):
        kwargs = {
            "turn_id": "turn-atomic-effects",
            "session_key": "agent:main:telegram:dm:effects",
            "platform": "telegram",
            "chat_id": "effects",
            "thread_id": None,
        }

        component_ids = dl.record_delivery_plan_with_post_delivery_effects(
            **kwargs,
            components=[{"kind": "text", "payload": {"content": "answer"}}],
            post_delivery_effects=[{
                "effect_key": "background-review:0",
                "kind": "background_review_notice",
                "payload": {"content": "Memory updated"},
            }],
        )

        with dl._connect() as conn:
            assert conn.execute(
                "SELECT component_id FROM delivery_components WHERE turn_id=?",
                (kwargs["turn_id"],),
            ).fetchall() == [(component_ids[0],)]
            assert conn.execute(
                "SELECT effect_key FROM post_delivery_effects WHERE turn_id=?",
                (kwargs["turn_id"],),
            ).fetchall() == [("background-review:0",)]

    def test_atomic_effect_insert_failure_rolls_back_plan_and_effects(self):
        turn_id = "turn-atomic-effects-rollback"
        # Force a database failure after the component insert.  The public API
        # must leave neither half recoverable.
        with dl._connect() as conn:
            dl._initialize_schema(conn)
            conn.execute(
                """CREATE TRIGGER reject_atomic_effect BEFORE INSERT
                   ON post_delivery_effects BEGIN
                   SELECT RAISE(ABORT, 'forced effect failure');
                   END"""
            )

        with pytest.raises(sqlite3.IntegrityError, match="forced effect failure"):
            dl.record_delivery_plan_with_post_delivery_effects(
                turn_id=turn_id,
                session_key="agent:main:telegram:dm:effects",
                platform="telegram",
                chat_id="effects",
                thread_id=None,
                components=[{"kind": "text", "payload": {"content": "answer"}}],
                post_delivery_effects=[{
                    "effect_key": "background-review:0",
                    "kind": "background_review_notice",
                    "payload": {"content": "Memory updated"},
                }],
            )

        with dl._connect() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM delivery_components WHERE turn_id=?", (turn_id,)
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM post_delivery_effects WHERE turn_id=?", (turn_id,)
            ).fetchone()[0] == 0

    def test_effect_registration_rejects_missing_delivery_plan(self):
        with pytest.raises(
            ValueError, match="post-delivery effect delivery plan mismatch"
        ):
            dl.record_post_delivery_effect(
                turn_id="missing-turn",
                session_key="agent:main:telegram:dm:effects",
                platform="telegram",
                chat_id="effects",
                thread_id=None,
                effect_key="goal-status",
                kind="goal_status_notice",
                payload={"content": "Goal achieved"},
            )

    @pytest.mark.parametrize(
        ("field", "wrong_value"),
        [
            ("session_key", "agent:other:telegram:dm:effects"),
            ("platform", "slack"),
            ("chat_id", "other-chat"),
            ("thread_id", "other-thread"),
        ],
    )
    def test_effect_registration_rejects_destination_mismatch(
        self, field, wrong_value
    ):
        kwargs = {
            "turn_id": "turn-effect-mismatch",
            "session_key": "agent:main:telegram:dm:effects",
            "platform": "telegram",
            "chat_id": "effects",
            "thread_id": "topic-7",
        }
        dl.record_delivery_plan(
            **kwargs,
            components=[{"kind": "text", "payload": {"content": "answer"}}],
        )
        effect_kwargs = {
            **kwargs,
            "effect_key": "goal-status",
            "kind": "goal_status_notice",
            "payload": {"content": "Goal achieved"},
        }
        effect_kwargs[field] = wrong_value
        with pytest.raises(
            ValueError, match="post-delivery effect delivery plan mismatch"
        ):
            dl.record_post_delivery_effect(**effect_kwargs)

    def test_sweep_abandons_effect_with_destination_mismatched_to_plan(self):
        session_key = "agent:main:telegram:dm:effects"
        turn_id = "turn-corrupt-effect-destination"
        component_id = dl.record_delivery_plan(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="effects",
            thread_id="topic-7",
            components=[{"kind": "text", "payload": {"content": "answer"}}],
        )[0]
        effect_id = dl.record_post_delivery_effect(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="effects",
            thread_id="topic-7",
            effect_key="goal-status",
            kind="goal_status_notice",
            payload={"content": "Goal achieved"},
        )
        dl.mark_delivery_component_delivered(component_id)
        with dl._DB_LOCK, dl._transaction() as conn:
            conn.execute(
                "UPDATE post_delivery_effects SET platform='slack' WHERE effect_id=?",
                (effect_id,),
            )

        assert dl.sweep_ready_post_delivery_effects({"telegram"}) == []
        with dl._connect() as conn:
            state, attempts, last_error = conn.execute(
                "SELECT state, attempts, last_error FROM post_delivery_effects "
                "WHERE effect_id=?",
                (effect_id,),
            ).fetchone()
        assert state == "abandoned"
        assert attempts == 0
        assert last_error == "post-delivery effect delivery plan mismatch"

    def test_effect_waits_for_complete_plan_then_is_claimed_once(self):
        session_key = "agent:main:telegram:dm:effects"
        turn_id = "turn-effects"
        component_id = dl.record_delivery_plan(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="effects",
            thread_id=None,
            components=[{"kind": "text", "payload": {"content": "answer"}}],
        )[0]

        effect_id = dl.record_post_delivery_effect(
            turn_id=turn_id,
            session_key=session_key,
            platform="telegram",
            chat_id="effects",
            thread_id=None,
            effect_key="goal-status",
            kind="goal_status_notice",
            payload={"content": "Goal achieved"},
        )

        assert dl.sweep_ready_post_delivery_effects() == []
        dl.mark_delivery_component_delivered(component_id)
        assert dl.sweep_ready_post_delivery_effects(deliverable_platforms=set()) == []
        claimed = dl.sweep_ready_post_delivery_effects()
        assert [row["effect_id"] for row in claimed] == [effect_id]
        assert claimed[0]["payload"] == {"content": "Goal achieved"}
        assert claimed[0]["attempts"] == 1
        assert dl.sweep_ready_post_delivery_effects() == []

        dl.mark_post_delivery_effect_delivered(effect_id)
        with pytest.raises(ValueError, match="failure claim mismatch"):
            dl.mark_post_delivery_effect_failed(effect_id, "stale failure")
        assert dl.post_delivery_effects_complete(turn_id) is True
        assert dl.sweep_ready_post_delivery_effects() == []

    def test_effect_registration_is_idempotent_and_collision_safe(self):
        kwargs = {
            "turn_id": "turn-idempotent-effect",
            "session_key": "agent:main:telegram:dm:effects",
            "platform": "telegram",
            "chat_id": "effects",
            "thread_id": None,
            "effect_key": "background-review:0",
            "kind": "background_review_notice",
            "payload": {"content": "Memory updated"},
        }
        dl.record_delivery_plan(
            turn_id=kwargs["turn_id"],
            session_key=kwargs["session_key"],
            platform=kwargs["platform"],
            chat_id=kwargs["chat_id"],
            thread_id=kwargs["thread_id"],
            components=[{"kind": "text", "payload": {"content": "answer"}}],
        )
        first = dl.record_post_delivery_effect(**kwargs)
        assert dl.record_post_delivery_effect(**kwargs) == first

        with pytest.raises(ValueError, match="post-delivery effect collision"):
            dl.record_post_delivery_effect(**{**kwargs, "payload": {"content": "different"}})

