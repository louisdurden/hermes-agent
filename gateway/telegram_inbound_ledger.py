"""Fail-closed durable acceptance journal for Telegram text dispatch.

Telegram advances its polling offset before the gateway has started the agent
turn.  A process death in that small interval used to leave neither a
transcript row nor restart work.  This journal is deliberately an *inbound*
counterpart to ``delivery_ledger``: it records the accepted event before the
adapter hands it to the runner, then startup claims only records whose former
owner is gone.

The execution boundary is at-most-once.  Once the runner has begun handling a
journalled event it is never replayed automatically: replaying an input whose
agent may already have run would be a silent duplicate side effect.  An
unstarted accepted row is therefore recovered once; an executing row is kept
for inspection and is not replayed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home

_LOCK = threading.Lock()
_RETENTION_SECONDS = 7 * 24 * 60 * 60


def _connect() -> sqlite3.Connection:
    path = get_hermes_home() / "state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="state.db (telegram_inbound_ledger)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS telegram_inbound_obligations (
                obligation_id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                payload TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                owner_pid INTEGER,
                owner_started_at INTEGER
            )"""
        )
    except Exception:
        conn.close()
        raise
    return conn


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _owner_stamp() -> tuple[int, Optional[int]]:
    from gateway.status import get_process_start_time

    pid = os.getpid()
    try:
        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: Any, started_at: Any) -> bool:
    if not pid:
        return False
    try:
        from gateway.delivery_ledger import _owner_alive as _delivery_owner_alive

        return bool(_delivery_owner_alive(pid, started_at))
    except Exception:
        return False


def _event_payload(event) -> Dict[str, Any]:
    source = event.source
    # Keep all recovery-relevant source fields in the one durable payload.
    # ``SessionSource.to_dict`` deliberately excludes transport-local and
    # authorization signals; those must never become restorable authority
    # after a restart.
    source_payload = source.to_dict()
    return {
        "text": event.text or "",
        # These are already normalized, locally cached media references that
        # the runner would have received before a restart.  Keep only the
        # bounded, agent-visible representation; never retain Telegram SDK
        # objects or raw transport metadata in this ledger.
        "media_urls": list(event.media_urls or []),
        "media_types": list(event.media_types or []),
        "message_type": event.message_type.value,
        "message_id": str(event.message_id) if event.message_id is not None else None,
        "platform_update_id": event.platform_update_id,
        "user_id": event.user_id,
        "user_name": event.user_name,
        "source": source_payload,
        "timestamp": event.timestamp.timestamp(),
        # This context is consumed by the runner after the restart boundary.
        # It belongs only in the already-required state-DB payload: never
        # persist Telegram's raw SDK object or free-form
        # metadata, which can carry unrelated and unbounded data.
        "reply_to_message_id": event.reply_to_message_id,
        "reply_to_text": event.reply_to_text,
        "reply_to_author_id": event.reply_to_author_id,
        "reply_to_author_name": event.reply_to_author_name,
        "reply_to_is_own_message": bool(event.reply_to_is_own_message),
        "auto_skill": event.auto_skill,
        "channel_prompt": event.channel_prompt,
        "channel_context": event.channel_context,
        "allow_gateway_control": bool(event.allow_gateway_control),
        # Batches are persisted per Telegram update, then reassembled through
        # the ordinary debounce path after a restart.
        "batching": bool(getattr(event, "_hermes_telegram_batching", False)),
    }


def record_accepted(event, session_key: str) -> Optional[str]:
    """Durably record an accepted Telegram text event before dispatch.

    Returns ``None`` on any persistence failure.  The adapter treats that as
    fail-closed and does not start the agent turn.
    """
    try:
        payload = _event_payload(event)
        ref = payload["platform_update_id"] or payload["message_id"]
        if ref is None:
            return None
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        oid = hashlib.sha256(f"{session_key}|{ref}".encode("utf-8", "replace")).hexdigest()[:24]
        now = time.time()
        pid, started = _owner_stamp()
        with _LOCK, _transaction() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO telegram_inbound_obligations
                   (obligation_id, session_key, payload, state, created_at, updated_at,
                    owner_pid, owner_started_at)
                   VALUES (?, ?, ?, 'accepted', ?, ?, ?, ?)""",
                (oid, session_key, encoded, now, now, pid, started),
            )
        # A Telegram retry carrying the same update must not start a second
        # agent turn.  The existing row is the sole durable authority.
        if not cursor.rowcount:
            return None
        setattr(event, "_telegram_inbound_obligation_id", oid)
        return oid
    except Exception:
        return None


def mark_execution_started(obligation_id: str | List[str]) -> bool:
    """Atomically seal one aggregated batch's accepted records for execution."""
    try:
        obligation_ids = [obligation_id] if isinstance(obligation_id, str) else list(obligation_id)
        if not obligation_ids:
            return False
        placeholders = ", ".join("?" for _ in obligation_ids)
        with _LOCK, _transaction() as conn:
            rows = conn.execute(
                f"SELECT obligation_id, state FROM telegram_inbound_obligations "
                f"WHERE obligation_id IN ({placeholders})", obligation_ids
            ).fetchall()
            if len(rows) != len(obligation_ids) or any(row[1] not in {"accepted", "claimed"} for row in rows):
                return False
            conn.execute(
                f"UPDATE telegram_inbound_obligations SET state='executing', updated_at=? "
                f"WHERE obligation_id IN ({placeholders})",
                [time.time(), *obligation_ids],
            )
        return True
    except Exception:
        return False


def mark_discarded(obligation_id: str | List[str]) -> bool:
    """Terminally close accepted work intentionally discarded by the user.

    Only rows that have not crossed the execution boundary may transition to
    ``discarded``.  This keeps stop/reset/stale-lock cleanup from turning an
    already-running turn into a replayable or ambiguous record.
    """
    try:
        obligation_ids = [obligation_id] if isinstance(obligation_id, str) else list(obligation_id)
        obligation_ids = list(dict.fromkeys(oid for oid in obligation_ids if oid))
        if not obligation_ids:
            return False
        placeholders = ", ".join("?" for _ in obligation_ids)
        with _LOCK, _transaction() as conn:
            rows = conn.execute(
                f"SELECT obligation_id, state FROM telegram_inbound_obligations "
                f"WHERE obligation_id IN ({placeholders})", obligation_ids
            ).fetchall()
            if len(rows) != len(obligation_ids) or any(
                row[1] not in {"accepted", "claimed"} for row in rows
            ):
                return False
            conn.execute(
                f"UPDATE telegram_inbound_obligations SET state='discarded', updated_at=? "
                f"WHERE obligation_id IN ({placeholders})",
                [time.time(), *obligation_ids],
            )
        return True
    except Exception:
        return False


def claim_recoverable(
    profile_name: Optional[str] = None, *, match_profile: bool = False
) -> List[Dict[str, Any]]:
    """Atomically claim unstarted events left by a dead process.

    ``match_profile`` confines multiplexed adapters to their own durable rows;
    ``profile_name=None`` then denotes the primary/default profile.
    """
    pid, started = _owner_stamp()
    now = time.time()
    claimed: List[Dict[str, Any]] = []
    with _LOCK, _transaction() as conn:
        rows = conn.execute(
            "SELECT obligation_id, session_key, payload, owner_pid, owner_started_at "
            "FROM telegram_inbound_obligations WHERE state IN ('accepted', 'claimed') "
            "ORDER BY created_at, obligation_id"
        ).fetchall()
        for oid, session_key, encoded, old_pid, old_started in rows:
            payload = json.loads(encoded)
            if match_profile:
                row_profile = payload.get("source", {}).get("profile")
                if row_profile != profile_name:
                    continue
            # A second recovery sweep in the same live process must not steal
            # its own pre-execution claim. A later process can reclaim it only
            # after this owner has actually died.
            if old_pid == pid and old_started == started:
                continue
            if _owner_alive(old_pid, old_started):
                continue
            cursor = conn.execute(
                   """UPDATE telegram_inbound_obligations
                       SET state='claimed', owner_pid=?, owner_started_at=?, updated_at=?
                       WHERE obligation_id=? AND state IN ('accepted', 'claimed')
                       AND (owner_pid IS ? OR owner_pid=?)""",
                (pid, started, now, oid, old_pid, old_pid),
            )
            if cursor.rowcount:
                claimed.append({"obligation_id": oid, "session_key": session_key, "payload": payload})
        conn.execute(
            "DELETE FROM telegram_inbound_obligations WHERE updated_at < ? AND state='executing'",
            (now - _RETENTION_SECONDS,),
        )
    return claimed


def restore_event(row):
    """Rebuild the normalized event without importing Telegram SDK types."""
    from datetime import datetime
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType, SessionSource

    payload = row["payload"]
    source_data = payload["source"]
    # All ledger rows are Telegram-only. A malformed or mismatched source is
    # unsafe to replay, so let the caller retain its claim rather than falling
    # back to a less-specific routing identity.
    if source_data.get("platform", Platform.TELEGRAM.value) != Platform.TELEGRAM.value:
        raise ValueError("Telegram inbound ledger row has a non-Telegram source")
    source = SessionSource.from_dict({
        "platform": Platform.TELEGRAM.value,
        **source_data,
    })
    message_type = MessageType(payload.get("message_type", MessageType.TEXT.value))
    media_urls = payload.get("media_urls", [])
    media_types = payload.get("media_types", [])
    if not isinstance(media_urls, list) or not all(isinstance(url, str) for url in media_urls):
        raise ValueError("Telegram inbound ledger row has invalid media URLs")
    if not isinstance(media_types, list) or not all(isinstance(media_type, str) for media_type in media_types):
        raise ValueError("Telegram inbound ledger row has invalid media MIME types")
    event = MessageEvent(
        text=payload["text"],
        message_type=message_type,
        user_id=payload.get("user_id"),
        user_name=payload.get("user_name"),
        message_id=payload.get("message_id"),
        platform_update_id=payload.get("platform_update_id"),
        source=source,
        timestamp=datetime.fromtimestamp(payload.get("timestamp") or time.time()),
        reply_to_message_id=payload.get("reply_to_message_id"),
        reply_to_text=payload.get("reply_to_text"),
        reply_to_author_id=payload.get("reply_to_author_id"),
        reply_to_author_name=payload.get("reply_to_author_name"),
        reply_to_is_own_message=bool(payload.get("reply_to_is_own_message", False)),
        auto_skill=payload.get("auto_skill"),
        channel_prompt=payload.get("channel_prompt"),
        channel_context=payload.get("channel_context"),
        allow_gateway_control=bool(payload.get("allow_gateway_control", True)),
        media_urls=media_urls,
        media_types=media_types,
    )
    setattr(event, "_telegram_inbound_obligation_id", row["obligation_id"])
    setattr(event, "_hermes_telegram_inbound_replay", True)
    setattr(event, "_hermes_telegram_batching", bool(payload.get("batching")))
    return event
