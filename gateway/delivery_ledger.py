"""Durable delivery-obligation ledger for gateway final responses.

A final agent response that was generated but not yet confirmed-delivered
to the messaging platform is the one artifact the gateway can lose without
a trace: the turn already burned its tokens, the text exists only in a
Python local, and a crash / planned restart between finalize and platform
ACK drops it silently (#58818, #41696, #63695).

This module records a small durable row per outbound final response in the
shared ``state.db`` (same file and conventions as
``tools.async_delegation`` — WAL, owner pid + process-start-time liveness,
bounded retention). The gateway writes three checkpoints around the send:

    record_obligation()   state='pending'     before any send attempt
    mark_attempting()     state='attempting'  immediately before the await
    mark_delivered() /    state='delivered'   only on SendResult.success
    mark_failed()         state='failed'      on a definitive rejection

On startup, ``sweep_recoverable()`` claims rows whose owning process is
dead and hands them to the gateway for redelivery. Crash semantics are
explicit about ambiguity (the contract review of the earlier
delivery-outbox attempt, #61790, closed it for silently resending
ambiguous sends):

- ``pending``     — the send never started: redeliver plainly, no dup risk.
- ``attempting``  — crashed mid-await: the platform MAY already have the
  message. Redelivered WITH a visible recovered-reply marker so the
  contract is honest at-least-once, never a silent duplicate.
- ``failed``      — definitively rejected once; the restart is a natural
  retry boundary. Also carries the marker.
- ``delivered``   — nothing to do; retention prunes.

Poison rows cannot spin: attempts are capped, stale rows expire, and both
transition to ``abandoned`` (kept briefly for inspection, then pruned).

Legacy text obligations remain best-effort for backward compatibility.
Inbound WAL writes and multipart delivery plans are fail-closed: callers must
not queue, execute, or send the represented turn when either durable write
fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import unquote

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()

# Redelivery policy knobs (module constants; deliberately not config — the
# ledger itself is gated by ``gateway.delivery_ledger`` and these bounds
# only matter in the rare recovery path).
MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 500

# Visible prefix for redeliveries that might duplicate an already-received
# message (crash mid-send / post-rejection retry). Honest at-least-once.
RECOVERED_MARKER = (
    "♻️ Recovered reply — the gateway restarted during delivery, "
    "so this may be a duplicate:\n\n"
)


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        # A PRAGMA/DDL failure after a successful connect() must not leak the
        # just-opened connection back to the caller.
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (delivery_ledger)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS gateway_delivery_ledger_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )"""
    )
    schema_marker = conn.execute(
        "SELECT value FROM gateway_delivery_ledger_meta "
        "WHERE key='durable_inbound_schema'"
    ).fetchone()
    if not schema_marker:
        # Earlier experimental runtimes created tables with these names but
        # no source/schema contract.  Never interpret those rows as accepted
        # turns or pending output: quarantine the whole unversioned table and
        # start the owned schema below from empty state.
        for table in ("inbound_turns", "delivery_components"):
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not exists:
                continue
            suffix = 0
            while True:
                archive = f"{table}_pre_durable_v1"
                if suffix:
                    archive = f"{archive}_{suffix}"
                collision = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (archive,),
                ).fetchone()
                if not collision:
                    break
                suffix += 1
            conn.execute(f'ALTER TABLE "{table}" RENAME TO "{archive}"')
        conn.execute(
            "INSERT INTO gateway_delivery_ledger_meta (key, value) "
            "VALUES ('durable_inbound_schema', '1')"
        )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_obligations (
            obligation_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            content TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS inbound_turns (
            turn_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            payload TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_components (
            component_id TEXT PRIMARY KEY,
            turn_id TEXT NOT NULL,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            ordinal INTEGER NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT,
            UNIQUE(turn_id, ordinal)
        )"""
    )
    effect_schema_marker = conn.execute(
        "SELECT value FROM gateway_delivery_ledger_meta "
        "WHERE key='post_delivery_effects_schema'"
    ).fetchone()
    if not effect_schema_marker:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='post_delivery_effects'"
        ).fetchone()
        if exists:
            suffix = 0
            while True:
                archive = "post_delivery_effects_pre_durable_v1"
                if suffix:
                    archive = f"{archive}_{suffix}"
                collision = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (archive,),
                ).fetchone()
                if not collision:
                    break
                suffix += 1
            conn.execute(
                f'ALTER TABLE post_delivery_effects RENAME TO "{archive}"'
            )
        conn.execute(
            "INSERT INTO gateway_delivery_ledger_meta (key, value) "
            "VALUES ('post_delivery_effects_schema', '1')"
        )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS post_delivery_effects (
            effect_id TEXT PRIMARY KEY,
            turn_id TEXT NOT NULL,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            effect_key TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT,
            UNIQUE(turn_id, effect_key),
            UNIQUE(turn_id, ordinal)
        )"""
    )
    effect_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(post_delivery_effects)")
    }
    if "ordinal" not in effect_columns:
        conn.execute("ALTER TABLE post_delivery_effects ADD COLUMN ordinal INTEGER")
        conn.execute(
            """WITH ranked AS (
                   SELECT effect_id,
                          ROW_NUMBER() OVER (
                              PARTITION BY turn_id ORDER BY created_at, effect_id
                          ) - 1 AS ordinal
                   FROM post_delivery_effects
               )
               UPDATE post_delivery_effects
               SET ordinal=(
                   SELECT ranked.ordinal FROM ranked
                   WHERE ranked.effect_id=post_delivery_effects.effect_id
               )"""
        )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS
               post_delivery_effects_turn_ordinal_idx
               ON post_delivery_effects(turn_id, ordinal)"""
    )
    conn.execute(
        """INSERT INTO gateway_delivery_ledger_meta (key, value)
           VALUES ('post_delivery_effects_schema', '2')
           ON CONFLICT(key) DO UPDATE SET value=excluded.value"""
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back the
    transaction; they do not close the connection. Using ``with _connect()``
    alone therefore leaks a connection — and its WAL/SHM file descriptors — on
    every call, deferring the close to the garbage collector. On a long-running
    gateway that exhausts ``RLIMIT_NOFILE`` (the cron-ledger sibling of this
    bug was #69567 / PR #69594). ``record_obligation`` runs on every outbound
    final response, so this ledger is the highest-frequency leaker.
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


@contextmanager
def _immediate_transaction() -> Iterator[sqlite3.Connection]:
    """Open a fresh ``BEGIN IMMEDIATE`` transaction without nesting it.

    ``_connect()`` may leave schema-initialization writes pending on a new
    connection.  Commit those first, then acquire the write reservation used
    to keep the ownership recheck and artifact unlink indivisible.
    """
    conn = _connect()
    try:
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
    finally:
        conn.close()


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """True when the recorded owning process still exists (pid + start time)."""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import get_process_start_time

        current_start = get_process_start_time(pid)
    except Exception:
        current_start = None
    if current_start is None:
        # No such process (or unreadable) — treat unreadable-but-extant
        # processes as alive only if the pid exists.
        try:
            os.kill(pid, 0)  # windows-footgun: ok — EPERM counts as alive below
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    if started_at is None:
        return True
    try:
        return int(current_start) == int(started_at)
    except (TypeError, ValueError):
        return True


def compute_obligation_id(session_key: str, message_ref: str, content: str) -> str:
    """Stable id: same turn + same content re-records idempotently, while
    distinct threads/topics on the same chat can never collide (the
    session_key carries platform, chat and thread; ``message_ref`` is the
    triggering inbound message id, distinguishing turns in one session)."""
    payload = f"{session_key}|{message_ref}|{content}"
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:24]


def compute_inbound_turn_id(
    session_key: str, message_ref: str, payload: Dict[str, Any]
) -> str:
    """Stable identity for one normalized inbound platform event."""
    if message_ref:
        raw = json.dumps(
            [session_key, message_ref], separators=(",", ":"), ensure_ascii=False
        )
    else:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        raw = f"{session_key}|{canonical}"
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:24]


def record_inbound_turn(
    *,
    turn_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    payload: Dict[str, Any],
) -> bool:
    """Write-ahead one inbound event and report exclusive first receipt."""
    now = time.time()
    pid, started = _owner_stamp()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    with _DB_LOCK, _transaction() as conn:
        inserted = conn.execute(
            """INSERT OR IGNORE INTO inbound_turns
               (turn_id, session_key, platform, chat_id, thread_id, payload,
                state, attempts, created_at, updated_at, owner_pid,
                owner_started_at)
               VALUES (?, ?, ?, ?, ?, ?, 'received', 0, ?, ?, ?, ?)""",
            (
                turn_id,
                session_key,
                platform,
                str(chat_id),
                str(thread_id) if thread_id else None,
                encoded,
                now,
                now,
                pid,
                started,
            ),
        )
    _prune()
    return bool(inserted.rowcount)


def mark_inbound_turns_executing(turn_ids: List[str]) -> bool:
    """Cross the non-replayable execution boundary for accepted inputs.

    Inbound rows are deliberately recoverable only until the gateway is about
    to consume them.  A crash after this transition is ambiguous: replaying
    could repeat tool or active-turn side effects, so the row stays terminal.
    """
    normalized = _normalized_turn_ids(turn_ids)
    placeholders = ",".join("?" for _ in normalized)
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            f"SELECT turn_id, state FROM inbound_turns WHERE turn_id IN ({placeholders})",
            normalized,
        ).fetchall()
        if len(rows) != len(normalized) or any(row[1] not in {"received", "claimed"} for row in rows):
            return False
        cursor = conn.execute(
            f"UPDATE inbound_turns SET state='executing', updated_at=? WHERE turn_id IN ({placeholders})",
            (time.time(), *normalized),
        )
    return cursor.rowcount == len(normalized)


def mark_inbound_turns_discarded(turn_ids: List[str]) -> bool:
    """Terminally close work intentionally dropped before execution."""
    normalized = _normalized_turn_ids(turn_ids)
    placeholders = ",".join("?" for _ in normalized)
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            f"SELECT turn_id, state FROM inbound_turns WHERE turn_id IN ({placeholders})",
            normalized,
        ).fetchall()
        if len(rows) != len(normalized) or any(row[1] not in {"received", "claimed"} for row in rows):
            return False
        cursor = conn.execute(
            f"UPDATE inbound_turns SET state='discarded', owner_pid=NULL, owner_started_at=NULL, updated_at=? WHERE turn_id IN ({placeholders})",
            (time.time(), *normalized),
        )
    return cursor.rowcount == len(normalized)


def _normalized_turn_ids(turn_ids: List[str]) -> List[str]:
    normalized = list(dict.fromkeys(
        turn_id for turn_id in turn_ids
        if isinstance(turn_id, str) and turn_id
    ))
    if not normalized or len(normalized) != len(turn_ids):
        raise ValueError("turn IDs must be non-empty and unique")
    return normalized


def mark_inbound_turns_completed(
    turn_ids: List[str], *, session_key: str
) -> None:
    """Complete represented WAL rows only under their canonical owner."""
    normalized = _normalized_turn_ids(turn_ids)
    placeholders = ",".join("?" for _ in normalized)
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            f"SELECT turn_id, session_key FROM inbound_turns "
            f"WHERE turn_id IN ({placeholders})",
            normalized,
        ).fetchall()
        if len(rows) != len(normalized) or any(row[1] != session_key for row in rows):
            raise ValueError("inbound turn session ownership mismatch")
        conn.execute(
            f"""UPDATE inbound_turns SET state='completed', updated_at=?,
                       owner_pid=NULL, owner_started_at=NULL, last_error=NULL
                   WHERE turn_id IN ({placeholders})""",
            (time.time(), *normalized),
        )


def mark_inbound_turn_completed(turn_id: str, *, session_key: str) -> None:
    """Complete one WAL row only under its canonical session owner."""
    mark_inbound_turns_completed([turn_id], session_key=session_key)


def rebind_inbound_turn(turn_id: str, session_key: str) -> None:
    """Attach a preflight row to the canonical session resolved by the runner."""
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE inbound_turns SET session_key=?, updated_at=?
               WHERE turn_id=? AND state!='completed'""",
            (session_key, time.time(), turn_id),
        )


def rebind_inbound_turns(
    turn_ids: List[str], *, expected_session_key: str, canonical_session_key: str
) -> None:
    """Atomically move every represented WAL row from quick to canonical key."""
    normalized = _normalized_turn_ids(turn_ids)
    placeholders = ",".join("?" for _ in normalized)
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            f"SELECT turn_id, session_key, state FROM inbound_turns "
            f"WHERE turn_id IN ({placeholders})",
            normalized,
        ).fetchall()
        allowed_keys = {expected_session_key, canonical_session_key}
        if (
            len(rows) != len(normalized)
            or any(row[1] not in allowed_keys or row[2] == "completed" for row in rows)
        ):
            raise ValueError("inbound turn session ownership mismatch")
        conn.execute(
            f"""UPDATE inbound_turns SET session_key=?, updated_at=?
                   WHERE turn_id IN ({placeholders}) AND state!='completed'""",
            (canonical_session_key, time.time(), *normalized),
        )


def release_inbound_turn_claim(turn_id: str) -> bool:
    """Return this process's undelivered inbound ownership to the replay pool."""
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE inbound_turns
               SET state='received', owner_pid=NULL, owner_started_at=NULL,
                   updated_at=?
               WHERE turn_id=? AND state IN ('received', 'claimed')
                 AND owner_pid=? AND owner_started_at IS ?""",
            (time.time(), turn_id, pid, started),
        )
    return bool(cursor.rowcount)


def sweep_recoverable_inbound_turns(
    *, deliverable_platforms: Optional[set] = None
) -> List[Dict[str, Any]]:
    """Atomically claim incomplete inbound events owned by dead processes."""
    now = time.time()
    pid, started = _owner_stamp()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT turn_id, session_key, platform, chat_id, thread_id,
                      payload, state, attempts, created_at, owner_pid,
                      owner_started_at
               FROM inbound_turns
               WHERE state IN ('received', 'claimed')
               ORDER BY created_at, turn_id"""
        ).fetchall()
        for row in rows:
            (
                turn_id,
                session_key,
                platform,
                chat_id,
                thread_id,
                payload,
                _state,
                attempts,
                created_at,
                owner_pid,
                owner_started_at,
            ) = row
            if _owner_alive(owner_pid, owner_started_at):
                continue
            if attempts >= MAX_ATTEMPTS or now - created_at > STALE_AFTER_SECONDS:
                conn.execute(
                    "UPDATE inbound_turns SET state='abandoned', updated_at=? "
                    "WHERE turn_id=?",
                    (now, turn_id),
                )
                continue
            if deliverable_platforms is not None and platform not in deliverable_platforms:
                continue
            cursor = conn.execute(
                """UPDATE inbound_turns
                   SET state='claimed', owner_pid=?, owner_started_at=?,
                       attempts=attempts+1, updated_at=?
                   WHERE turn_id=? AND owner_pid IS ?""",
                (pid, started, now, turn_id, owner_pid),
            )
            if not cursor.rowcount:
                continue
            claimed.append(
                {
                    "turn_id": turn_id,
                    "session_key": session_key,
                    "platform": platform,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "payload": json.loads(payload),
                    "attempts": attempts + 1,
                }
            )
    return claimed


def _delivery_component_id(
    turn_id: str, ordinal: int, kind: str, payload: Dict[str, Any]
) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    raw = f"{turn_id}|{ordinal}|{kind}|{canonical}"
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:24]


DELIVERY_COMPONENT_KINDS = frozenset(
    {"text", "image", "tts", "voice", "video", "document"}
)

POST_DELIVERY_EFFECT_KINDS = frozenset(
    {"background_review_notice", "goal_status_notice"}
)


def _post_delivery_effect_id(turn_id: str, effect_key: str) -> str:
    raw = f"{turn_id}|{effect_key}"
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:24]


def record_post_delivery_effect(
    *,
    turn_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    effect_key: str,
    kind: str,
    payload: Dict[str, Any],
) -> str:
    """Persist one reconstructable post-delivery action idempotently."""
    if kind not in POST_DELIVERY_EFFECT_KINDS:
        raise ValueError("unsupported post-delivery effect kind")
    if not isinstance(effect_key, str) or not effect_key.strip():
        raise ValueError("invalid post-delivery effect key")
    if not isinstance(payload, dict):
        raise ValueError("invalid post-delivery effect payload")
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("invalid post-delivery effect payload")

    effect_key = effect_key.strip()
    effect_id = _post_delivery_effect_id(turn_id, effect_key)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    stored_thread_id = str(thread_id) if thread_id else None
    expected_destination = (
        session_key,
        platform,
        str(chat_id),
        stored_thread_id,
    )
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        plan_destinations = conn.execute(
            """SELECT session_key, platform, chat_id, thread_id
               FROM delivery_components WHERE turn_id=?""",
            (turn_id,),
        ).fetchall()
        if not plan_destinations or any(
            tuple(destination) != expected_destination
            for destination in plan_destinations
        ):
            raise ValueError("post-delivery effect delivery plan mismatch")
        stored = conn.execute(
            """SELECT turn_id, session_key, platform, chat_id, thread_id,
                      effect_key, kind, payload, ordinal
               FROM post_delivery_effects WHERE effect_id=?""",
            (effect_id,),
        ).fetchone()
        if stored is None:
            ordinal = conn.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 "
                "FROM post_delivery_effects WHERE turn_id=?",
                (turn_id,),
            ).fetchone()[0]
            conn.execute(
                """INSERT INTO post_delivery_effects
               (effect_id, turn_id, session_key, platform, chat_id, thread_id,
                effect_key, ordinal, kind, payload, state, attempts,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)""",
                (
                    effect_id, turn_id, session_key, platform, str(chat_id),
                    stored_thread_id, effect_key, ordinal, kind, encoded, now, now,
                ),
            )
        stored = conn.execute(
            """SELECT turn_id, session_key, platform, chat_id, thread_id,
                      effect_key, kind, payload
               FROM post_delivery_effects WHERE effect_id=?""",
            (effect_id,),
        ).fetchone()
        expected = (
            turn_id, session_key, platform, str(chat_id), stored_thread_id,
            effect_key, kind, encoded,
        )
        if stored != expected:
            raise ValueError("post-delivery effect collision")
    _prune()
    return effect_id


def sweep_ready_post_delivery_effects(
    deliverable_platforms: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """Claim effects only once every primary delivery component is delivered."""
    now = time.time()
    pid, started = _owner_stamp()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT pe.* FROM post_delivery_effects AS pe
               WHERE pe.state IN ('pending','attempting','failed')
               ORDER BY pe.turn_id, pe.ordinal"""
        ).fetchall()
        for row in rows:
            abandoned_predecessor = conn.execute(
                """SELECT 1 FROM post_delivery_effects
                   WHERE turn_id=? AND ordinal<? AND state='abandoned' LIMIT 1""",
                (row["turn_id"], row["ordinal"]),
            ).fetchone()
            if abandoned_predecessor:
                conn.execute(
                    """UPDATE post_delivery_effects
                       SET state='abandoned', updated_at=?, owner_pid=NULL,
                           owner_started_at=NULL,
                           last_error='post-delivery effect predecessor abandoned'
                       WHERE effect_id=?""",
                    (now, row["effect_id"]),
                )
                continue
            predecessor = conn.execute(
                """SELECT 1 FROM post_delivery_effects
                   WHERE turn_id=? AND ordinal<?
                     AND state NOT IN ('delivered','abandoned') LIMIT 1""",
                (row["turn_id"], row["ordinal"]),
            ).fetchone()
            if predecessor:
                continue
            plan_rows = conn.execute(
                """SELECT session_key, platform, chat_id, thread_id, state
                   FROM delivery_components WHERE turn_id=?""",
                (row["turn_id"],),
            ).fetchall()
            effect_destination = (
                row["session_key"],
                row["platform"],
                row["chat_id"],
                row["thread_id"],
            )
            if not plan_rows or any(
                tuple(plan_row[:4]) != effect_destination for plan_row in plan_rows
            ):
                conn.execute(
                    """UPDATE post_delivery_effects
                       SET state='abandoned', updated_at=?, owner_pid=NULL,
                           owner_started_at=NULL,
                           last_error='post-delivery effect delivery plan mismatch'
                       WHERE effect_id=?""",
                    (now, row["effect_id"]),
                )
                continue
            if (
                deliverable_platforms is not None
                and row["platform"] not in deliverable_platforms
            ):
                continue
            plan_total = len(plan_rows)
            plan_delivered = sum(plan_row[4] == "delivered" for plan_row in plan_rows)
            plan_terminal = sum(
                plan_row[4] in {"delivered", "abandoned"} for plan_row in plan_rows
            )
            plan_complete = plan_total > 0 and plan_delivered == plan_total
            plan_failed_terminally = (
                plan_total > 0
                and plan_terminal == plan_total
                and not plan_complete
            )
            stale = now - row["created_at"] > STALE_AFTER_SECONDS
            if plan_failed_terminally or stale:
                conn.execute(
                    """UPDATE post_delivery_effects
                       SET state='abandoned', updated_at=?, owner_pid=NULL,
                           owner_started_at=NULL
                       WHERE effect_id=?""",
                    (now, row["effect_id"]),
                )
                continue
            if not plan_complete:
                continue
            if _owner_alive(row["owner_pid"], row["owner_started_at"]):
                continue
            if row["attempts"] >= MAX_ATTEMPTS:
                conn.execute(
                    """UPDATE post_delivery_effects
                       SET state='abandoned', updated_at=? WHERE effect_id=?""",
                    (now, row["effect_id"]),
                )
                continue
            updated = conn.execute(
                """UPDATE post_delivery_effects
                   SET state='attempting', attempts=attempts+1, updated_at=?,
                       owner_pid=?, owner_started_at=?
                   WHERE effect_id=? AND state IN ('pending','attempting','failed')
                     AND owner_pid IS ? AND owner_started_at IS ?""",
                (
                    now, pid, started, row["effect_id"],
                    row["owner_pid"], row["owner_started_at"],
                ),
            )
            if updated.rowcount != 1:
                continue
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            item["prior_state"] = row["state"]
            item["state"] = "attempting"
            item["attempts"] = row["attempts"] + 1
            claimed.append(item)
            # Claim one effect at a time. The runner resweeps only after a
            # durable ACK, so a later row is never left attempting when an
            # earlier transport or checkpoint fails.
            break
    return claimed


def mark_post_delivery_effect_delivered(effect_id: str) -> None:
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        updated = conn.execute(
            """UPDATE post_delivery_effects
               SET state='delivered', updated_at=?, owner_pid=NULL,
                   owner_started_at=NULL, last_error=NULL
               WHERE effect_id=? AND state='attempting'
                 AND owner_pid IS ? AND owner_started_at IS ?""",
            (time.time(), effect_id, pid, started),
        )
        if updated.rowcount != 1:
            raise ValueError("post-delivery effect delivery claim mismatch")


def mark_post_delivery_effect_failed(effect_id: str, error: str) -> None:
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        updated = conn.execute(
            """UPDATE post_delivery_effects
               SET state='failed', updated_at=?, owner_pid=NULL,
                   owner_started_at=NULL, last_error=?
               WHERE effect_id=? AND state='attempting'
                 AND owner_pid IS ? AND owner_started_at IS ?""",
            (time.time(), error[:1000], effect_id, pid, started),
        )
        if updated.rowcount != 1:
            raise ValueError("post-delivery effect failure claim mismatch")


def post_delivery_effects_complete(turn_id: str) -> bool:
    """True when no registered effect remains non-terminal for this turn."""
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            """SELECT COUNT(*) FROM post_delivery_effects
               WHERE turn_id=? AND state NOT IN ('delivered','abandoned')""",
            (turn_id,),
        ).fetchone()
    return bool(row and row[0] == 0)


def record_delivery_plan(
    *,
    turn_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    components: List[Dict[str, Any]],
    represented_turn_ids: Optional[List[str]] = None,
    post_delivery_effects: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Persist every outbound component and optional effects in one transaction."""
    normalized_components: List[tuple[str, Dict[str, Any]]] = []
    required_payload_key = {
        "text": "content",
        "image": "url",
        "tts": "path",
        "voice": "path",
        "video": "path",
        "document": "path",
    }
    for component in components:
        kind = component.get("kind")
        if not isinstance(kind, str) or kind not in DELIVERY_COMPONENT_KINDS:
            raise ValueError("unsupported delivery component kind")
        payload = component.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("invalid delivery component payload")
        required_key = required_payload_key[kind]
        if (
            not isinstance(payload.get(required_key), str)
            or not payload[required_key].strip()
        ):
            raise ValueError("invalid delivery component payload")
        normalized_components.append((kind, payload))

    normalized_effects: List[tuple[str, str, str]] = []
    for effect in post_delivery_effects or []:
        if not isinstance(effect, dict):
            raise ValueError("invalid durable post-delivery effect")
        effect_key = effect.get("effect_key")
        kind = effect.get("kind")
        payload = effect.get("payload")
        if kind not in POST_DELIVERY_EFFECT_KINDS:
            raise ValueError("unsupported post-delivery effect kind")
        if not isinstance(effect_key, str) or not effect_key.strip():
            raise ValueError("invalid post-delivery effect key")
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("content"), str)
            or not payload["content"].strip()
        ):
            raise ValueError("invalid post-delivery effect payload")
        normalized_effects.append((
            effect_key.strip(),
            kind,
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
        ))

    now = time.time()
    pid, started = _owner_stamp()
    component_ids: List[str] = []
    with _DB_LOCK, _transaction() as conn:
        represented: List[str] = []
        if represented_turn_ids is not None:
            represented = _normalized_turn_ids(represented_turn_ids)
        if represented:
            placeholders = ",".join("?" for _ in represented)
            rows = conn.execute(
                f"SELECT turn_id, session_key, state FROM inbound_turns "
                f"WHERE turn_id IN ({placeholders})",
                represented,
            ).fetchall()
            if (
                len(rows) != len(represented)
                or turn_id not in represented
                or any(
                    row[1] != session_key
                    # ``executing`` is sealed against inbound replay, but it
                    # is still the live turn that is entitled to durably
                    # record the response it just produced.  Completed and
                    # discarded rows remain ineligible to create a plan.
                    or row[2] not in {"received", "claimed", "executing"}
                    for row in rows
                )
            ):
                raise ValueError("inbound turn session ownership mismatch")
        existing_component_count = conn.execute(
            "SELECT COUNT(*) FROM delivery_components WHERE turn_id=?",
            (turn_id,),
        ).fetchone()[0]
        if existing_component_count and existing_component_count != len(components):
            raise ValueError("delivery plan cardinality collision")
        for ordinal, (kind, payload) in enumerate(normalized_components):
            component_id = _delivery_component_id(turn_id, ordinal, kind, payload)
            component_ids.append(component_id)
            encoded_payload = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            stored_thread_id = str(thread_id) if thread_id else None
            conn.execute(
                """INSERT OR IGNORE INTO delivery_components
                   (component_id, turn_id, session_key, platform, chat_id,
                    thread_id, ordinal, kind, payload, state, attempts,
                    created_at, updated_at, owner_pid, owner_started_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0,
                           ?, ?, ?, ?)""",
                (
                    component_id,
                    turn_id,
                    session_key,
                    platform,
                    str(chat_id),
                    stored_thread_id,
                    ordinal,
                    kind,
                    encoded_payload,
                    now,
                    now,
                    pid,
                    started,
                ),
            )
            stored = conn.execute(
                """SELECT component_id, session_key, platform, chat_id,
                          thread_id, kind, payload
                   FROM delivery_components
                   WHERE turn_id=? AND ordinal=?""",
                (turn_id, ordinal),
            ).fetchone()
            expected = (
                component_id,
                session_key,
                platform,
                str(chat_id),
                stored_thread_id,
                kind,
                encoded_payload,
            )
            if stored != expected:
                raise ValueError("delivery component collision")
        secondary_turn_ids = [candidate for candidate in represented if candidate != turn_id]
        if secondary_turn_ids:
            placeholders = ",".join("?" for _ in secondary_turn_ids)
            conn.execute(
                f"""UPDATE inbound_turns
                       SET state='completed', updated_at=?,
                           owner_pid=NULL, owner_started_at=NULL
                       WHERE turn_id IN ({placeholders})
                         AND state IN ('received', 'claimed', 'executing')""",
                (now, *secondary_turn_ids),
            )
        stored_thread_id = str(thread_id) if thread_id else None
        for ordinal, (effect_key, kind, encoded_payload) in enumerate(normalized_effects):
            effect_id = _post_delivery_effect_id(turn_id, effect_key)
            conn.execute(
                """INSERT OR IGNORE INTO post_delivery_effects
                   (effect_id, turn_id, session_key, platform, chat_id, thread_id,
                    effect_key, ordinal, kind, payload, state, attempts,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)""",
                (
                    effect_id, turn_id, session_key, platform, str(chat_id),
                    stored_thread_id, effect_key, ordinal, kind, encoded_payload,
                    now, now,
                ),
            )
            stored = conn.execute(
                """SELECT turn_id, session_key, platform, chat_id, thread_id,
                          effect_key, ordinal, kind, payload
                   FROM post_delivery_effects WHERE effect_id=?""",
                (effect_id,),
            ).fetchone()
            expected = (
                turn_id, session_key, platform, str(chat_id), stored_thread_id,
                effect_key, ordinal, kind, encoded_payload,
            )
            if stored != expected:
                raise ValueError("post-delivery effect collision")
    _prune()
    return component_ids


def record_delivery_plan_with_post_delivery_effects(
    *,
    turn_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    components: List[Dict[str, Any]],
    post_delivery_effects: List[Dict[str, Any]],
    represented_turn_ids: Optional[List[str]] = None,
) -> List[str]:
    """Atomically commit a primary plan and every effect captured before seal."""
    return record_delivery_plan(
        turn_id=turn_id,
        session_key=session_key,
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        components=components,
        represented_turn_ids=represented_turn_ids,
        post_delivery_effects=post_delivery_effects,
    )


def mark_delivery_component_attempting(component_id: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        updated = conn.execute(
            """UPDATE delivery_components
               SET state='attempting', attempts=attempts+1, updated_at=?
               WHERE component_id=? AND state='pending'""",
            (time.time(), component_id),
        )
        if updated.rowcount != 1:
            raise ValueError("delivery component cannot transition to attempting")


def _cleanup_staged_component(kind: str, payload_json: Optional[str]) -> None:
    if not payload_json:
        return
    try:
        payload = json.loads(payload_json)
        candidate = ""
        if kind in {"tts", "voice", "video", "document", "media", "file"}:
            candidate = str(payload.get("path") or "")
        elif kind == "image":
            url = str(payload.get("url") or "")
            if url.startswith("file://"):
                candidate = unquote(url[7:])
        if not candidate:
            return
        root = get_hermes_home() / "cache" / "delivery_outbox"
        resolved_root = root.resolve()
        resolved_candidate = Path(candidate).expanduser().resolve()
        resolved_candidate.relative_to(resolved_root)
        resolved_candidate.unlink(missing_ok=True)
        try:
            resolved_candidate.parent.rmdir()
        except OSError:
            pass
    except (OSError, ValueError, json.JSONDecodeError):
        logger.debug("staged delivery artifact cleanup failed", exc_info=True)


def _cleanup_pruned_component(turn_id: str, kind: str, payload_json: str) -> None:
    """Unlink a pruned artifact only while no inbound row owns its turn.

    The component-row deletion has already committed.  A fresh immediate
    transaction closes the post-commit race: no process can establish new
    inbound ownership between this check and the filesystem cleanup.
    """
    with _DB_LOCK, _immediate_transaction() as conn:
        active = conn.execute(
            """SELECT 1 FROM inbound_turns
               WHERE turn_id=? AND state IN ('received','claimed') LIMIT 1""",
            (turn_id,),
        ).fetchone()
        if active:
            return
        _cleanup_staged_component(kind, payload_json)


def mark_delivery_component_delivered(component_id: str) -> None:
    staged: Optional[tuple[str, str]] = None
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT kind, payload FROM delivery_components WHERE component_id=?",
            (component_id,),
        ).fetchone()
        if row:
            staged = (str(row[0]), str(row[1]))
        conn.execute(
            """UPDATE delivery_components
               SET state='delivered', updated_at=?, last_error=NULL
               WHERE component_id=?""",
            (time.time(), component_id),
        )
    if staged:
        _cleanup_staged_component(*staged)


def mark_delivery_components_delivered(component_ids: List[str]) -> None:
    """ACK a related component set atomically after one transport success."""
    ids = list(dict.fromkeys(str(value) for value in component_ids if value))
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    staged: List[tuple[str, str]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            f"""SELECT kind, payload FROM delivery_components
                   WHERE component_id IN ({placeholders})""",
            ids,
        ).fetchall()
        staged = [(str(row[0]), str(row[1])) for row in rows]
        conn.execute(
            f"""UPDATE delivery_components
                   SET state='delivered', updated_at=?, last_error=NULL
                   WHERE component_id IN ({placeholders})""",
            (time.time(), *ids),
        )
    for artifact in staged:
        _cleanup_staged_component(*artifact)


def mark_delivery_component_failed(component_id: str, error: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE delivery_components
               SET state='failed', updated_at=?, last_error=?
               WHERE component_id=?""",
            (time.time(), error[:1000], component_id),
        )


def delivery_plan_exists(turn_id: str) -> bool:
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT 1 FROM delivery_components WHERE turn_id=? LIMIT 1",
            (turn_id,),
        ).fetchone()
    return row is not None


def delivery_components_by_kind(turn_id: str, kind: str) -> List[Dict[str, str]]:
    """Return durable sibling IDs/states for recovery dependency decisions."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT component_id, state FROM delivery_components
               WHERE turn_id=? AND kind=? ORDER BY ordinal""",
            (turn_id, kind),
        ).fetchall()
    return [
        {"component_id": str(row[0]), "state": str(row[1])} for row in rows
    ]


def session_has_delivery_plan(session_key: str) -> bool:
    """Return whether the ledger still owns delivery for a session."""
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            """SELECT 1
               FROM delivery_components AS dc
               LEFT JOIN inbound_turns AS it ON it.turn_id = dc.turn_id
               WHERE dc.session_key=?
                 AND (
                     dc.state NOT IN ('delivered','abandoned')
                     OR it.state IN ('received','claimed')
                 )
               LIMIT 1""",
            (session_key,),
        ).fetchone()
    return row is not None


def delivery_plan_terminal(turn_id: str) -> bool:
    """Return true when every component is delivered or abandoned."""
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            """SELECT COUNT(*),
                      SUM(CASE WHEN state IN ('delivered','abandoned') THEN 0 ELSE 1 END)
               FROM delivery_components WHERE turn_id=?""",
            (turn_id,),
        ).fetchone()
    return bool(row and row[0] > 0 and row[1] == 0)


def delivery_plan_complete(turn_id: str) -> bool:
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN state='delivered' THEN 1 ELSE 0 END) AS done
               FROM delivery_components WHERE turn_id=?""",
            (turn_id,),
        ).fetchone()
    return bool(row and row[0] > 0 and row[0] == row[1])


def delivery_plan_preserves_resume_pending(turn_id: str) -> bool:
    """Return whether this plan must not consume a different turn's marker."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            "SELECT payload FROM delivery_components WHERE turn_id=?",
            (turn_id,),
        ).fetchall()
    for row in rows:
        try:
            if json.loads(row[0]).get("preserve_resume_pending") is True:
                return True
        except (TypeError, json.JSONDecodeError):
            continue
    return False


def sweep_recoverable_delivery_components(
    deliverable_platforms: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    now = time.time()
    pid, started = _owner_stamp()
    claimed: List[Dict[str, Any]] = []
    abandoned_artifacts: List[tuple[str, str]] = []
    with _DB_LOCK, _transaction() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM delivery_components
               WHERE state IN ('pending', 'attempting', 'failed')
               ORDER BY turn_id, ordinal"""
        ).fetchall()
        for row in rows:
            if deliverable_platforms and row["platform"] not in deliverable_platforms:
                continue
            if _owner_alive(row["owner_pid"], row["owner_started_at"]):
                continue
            if (
                row["attempts"] >= MAX_ATTEMPTS
                or now - row["created_at"] > STALE_AFTER_SECONDS
            ):
                abandoned_artifacts.append(
                    (str(row["kind"]), str(row["payload"]))
                )
                conn.execute(
                    """UPDATE delivery_components
                       SET state='abandoned', updated_at=?
                       WHERE component_id=?""",
                    (now, row["component_id"]),
                )
                continue
            updated = conn.execute(
                """UPDATE delivery_components
                   SET state='attempting', attempts=attempts+1, updated_at=?,
                       owner_pid=?, owner_started_at=?
                   WHERE component_id=? AND state IN ('pending', 'attempting', 'failed')
                     AND owner_pid IS ? AND owner_started_at IS ?""",
                (
                    now,
                    pid,
                    started,
                    row["component_id"],
                    row["owner_pid"],
                    row["owner_started_at"],
                ),
            )
            if updated.rowcount != 1:
                continue
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            item["prior_state"] = row["state"]
            item["state"] = "attempting"
            claimed.append(item)
    for kind, payload_json in abandoned_artifacts:
        # A generated TTS file is the only recoverable copy of an owed voice
        # response. Retry exhaustion is not a transport ACK, so retain it for
        # operator recovery instead of deleting evidence of the obligation.
        if kind != "tts":
            _cleanup_staged_component(kind, payload_json)
    return claimed


def record_obligation(
    *,
    obligation_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    content: str,
) -> None:
    """Record a final response as owed to the platform (state='pending')."""
    now = time.time()
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)""",
            (obligation_id, session_key, platform, str(chat_id),
             str(thread_id) if thread_id else None, content, now, now,
             pid, started),
        )
    _prune()


def mark_attempting(obligation_id: str) -> None:
    _update_state(obligation_id, "attempting")


def mark_delivered(obligation_id: str) -> None:
    _update_state(obligation_id, "delivered")


def mark_failed(obligation_id: str, error: str = "") -> None:
    _update_state(obligation_id, "failed", error=error)


def _update_state(obligation_id: str, state: str, error: str = "") -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE delivery_obligations
               SET state=?, updated_at=?, last_error=?
               WHERE obligation_id=?""",
            (state, time.time(), error[:500] if error else None, obligation_id),
        )


def sweep_recoverable(
    now: Optional[float] = None,
    *,
    deliverable_platforms: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Claim undelivered rows owned by dead processes; return them for
    redelivery.

    Claiming atomically re-stamps the owner to THIS process and increments
    ``attempts``, so a second gateway racing the same sweep cannot
    double-claim (the UPDATE is guarded on the previous owner stamp).
    Rows over the attempts cap or older than the stale cutoff transition to
    'abandoned' instead of being returned.

    ``deliverable_platforms`` (platform value strings) restricts claiming to
    platforms the caller can actually send on this boot.  ``attempts`` is the
    redelivery budget, so it must only be spent on a real send: a platform
    that failed to connect would otherwise burn one attempt per boot and hit
    the cap having never been sent once.  Rows for absent platforms are left
    untouched for a later boot; the stale cutoff still bounds them.
    """
    now = now if now is not None else time.time()
    pid, started = _owner_stamp()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, created_at,
                      owner_pid, owner_started_at
               FROM delivery_obligations
               WHERE state IN ('pending', 'attempting', 'failed')"""
        ).fetchall()
        for (oid, session_key, platform, chat_id, thread_id, content, state,
             attempts, created_at, owner_pid, owner_started_at) in rows:
            if _owner_alive(owner_pid, owner_started_at):
                continue  # a live gateway still owns this row
            if attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=? WHERE obligation_id=?""",
                    (now, oid),
                )
                continue
            if (
                deliverable_platforms is not None
                and platform not in deliverable_platforms
            ):
                # No adapter for this platform this boot — the caller cannot
                # send, so claiming would spend an attempt on a no-op.
                continue
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET owner_pid=?, owner_started_at=?, attempts=attempts+1,
                       updated_at=?
                   WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
                (pid, started, now, oid, owner_pid, owner_pid),
            )
            if cursor.rowcount:
                claimed.append({
                    "obligation_id": oid,
                    "session_key": session_key,
                    "platform": platform,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "content": content,
                    # pending = send never started, redeliver plainly;
                    # attempting/failed = ambiguous or rejected, carry marker.
                    "needs_marker": state != "pending",
                    "attempts": attempts + 1,
                })
    return claimed


def _prune(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    cutoff = now - _RETENTION_SECONDS
    staged_cleanup: List[tuple[str, str, str]] = []
    try:
        with _transaction() as conn:
            conn.execute(
                """DELETE FROM delivery_obligations
                   WHERE state IN ('delivered', 'abandoned') AND updated_at < ?""",
                (cutoff,),
            )
            total = conn.execute(
                "SELECT COUNT(*) FROM delivery_obligations"
            ).fetchone()[0]
            excess = max(0, total - _MAX_ROWS)
            if excess:
                conn.execute(
                    """DELETE FROM delivery_obligations WHERE obligation_id IN (
                         SELECT obligation_id FROM delivery_obligations
                         WHERE state IN ('delivered', 'abandoned')
                         ORDER BY updated_at ASC
                         LIMIT ?)""",
                    (excess,),
                )
            for table, id_column, terminal_states in (
                ("inbound_turns", "turn_id", ("completed", "abandoned", "discarded")),
                ("delivery_components", "component_id", ("delivered", "abandoned")),
                ("post_delivery_effects", "effect_id", ("delivered", "abandoned")),
            ):
                placeholders = ",".join("?" for _ in terminal_states)
                active_inbound_guard = ""
                if table in {"delivery_components", "post_delivery_effects"}:
                    active_inbound_guard = (
                        " AND NOT EXISTS (SELECT 1 FROM inbound_turns AS active "
                        f"WHERE active.turn_id={table}.turn_id "
                        "AND active.state IN ('received','claimed'))"
                    )
                if table == "delivery_components":
                    staged_cleanup.extend(
                        (str(row[0]), str(row[1]), str(row[2]))
                        for row in conn.execute(
                            f"SELECT turn_id, kind, payload FROM {table} "
                            f"WHERE state IN ({placeholders}) AND updated_at < ?"
                            + active_inbound_guard,
                            (*terminal_states, cutoff),
                        ).fetchall()
                    )
                conn.execute(
                    f"DELETE FROM {table} WHERE state IN ({placeholders}) "
                    "AND updated_at < ?"
                    + active_inbound_guard,
                    (*terminal_states, cutoff),
                )
                table_total = conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                table_excess = max(0, table_total - _MAX_ROWS)
                if table_excess:
                    if table == "delivery_components":
                        staged_cleanup.extend(
                            (str(row[0]), str(row[1]), str(row[2]))
                            for row in conn.execute(
                                f"""SELECT turn_id, kind, payload FROM {table}
                                     WHERE state IN ({placeholders})
                                     {active_inbound_guard}
                                     ORDER BY updated_at ASC LIMIT ?""",
                                (*terminal_states, table_excess),
                            ).fetchall()
                        )
                    conn.execute(
                        f"""DELETE FROM {table} WHERE {id_column} IN (
                             SELECT {id_column} FROM {table}
                             WHERE state IN ({placeholders})
                             {active_inbound_guard}
                             ORDER BY updated_at ASC LIMIT ?)
                        """,
                        (*terminal_states, table_excess),
                    )
    except Exception:
        logger.debug("delivery ledger prune failed", exc_info=True)
        return
    for artifact in staged_cleanup:
        try:
            _cleanup_pruned_component(*artifact)
        except Exception:
            logger.debug("pruned delivery artifact cleanup failed", exc_info=True)


def ledger_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Read the ``gateway.delivery_ledger`` config gate (default on)."""
    try:
        if config is None:
            from hermes_cli.config import load_config

            config = load_config()
        gw = config.get("gateway") or {}
        value = gw.get("delivery_ledger", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        return bool(value)
    except Exception:
        return True


def debug_rows(limit: int = 20) -> str:
    """Human-readable dump for ad-hoc inspection (sqlite3-free path)."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, state, attempts,
                      created_at, updated_at, last_error
               FROM delivery_obligations
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return json.dumps(
        [
            {
                "id": r[0], "session": r[1], "state": r[2], "attempts": r[3],
                "created_at": r[4], "updated_at": r[5], "last_error": r[6],
            }
            for r in rows
        ],
        indent=2,
    )
