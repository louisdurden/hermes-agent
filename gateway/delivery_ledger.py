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

Everything here is best-effort by design: ledger failures must never block
or delay an actual send. Callers wrap every call in try/except.
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
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    raw = f"{session_key}|{message_ref}|{canonical}"
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:24]


def record_inbound_turn(
    *,
    turn_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    payload: Dict[str, Any],
) -> None:
    """Write-ahead one inbound event before hooks or agent work."""
    now = time.time()
    pid, started = _owner_stamp()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
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


def mark_inbound_turn_completed(turn_id: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE inbound_turns SET state='completed', updated_at=?,
                      last_error=NULL WHERE turn_id=?""",
            (time.time(), turn_id),
        )


def rebind_inbound_turn(turn_id: str, session_key: str) -> None:
    """Attach a preflight row to the canonical session resolved by the runner."""
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE inbound_turns SET session_key=?, updated_at=?
               WHERE turn_id=? AND state!='completed'""",
            (session_key, time.time(), turn_id),
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


def record_delivery_plan(
    *,
    turn_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    components: List[Dict[str, Any]],
    represented_turn_ids: Optional[List[str]] = None,
) -> List[str]:
    """Persist every outbound component in one transaction before first send."""
    now = time.time()
    pid, started = _owner_stamp()
    component_ids: List[str] = []
    with _DB_LOCK, _transaction() as conn:
        for ordinal, component in enumerate(components):
            kind = str(component["kind"])
            payload = component.get("payload") or {}
            component_id = _delivery_component_id(turn_id, ordinal, kind, payload)
            component_ids.append(component_id)
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
                    str(thread_id) if thread_id else None,
                    ordinal,
                    kind,
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ),
                    now,
                    now,
                    pid,
                    started,
                ),
            )
        represented = list(
            dict.fromkeys(
                candidate
                for candidate in (represented_turn_ids or [])
                if isinstance(candidate, str) and candidate
            )
        )
        secondary_turn_ids = [candidate for candidate in represented if candidate != turn_id]
        if secondary_turn_ids:
            placeholders = ",".join("?" for _ in secondary_turn_ids)
            conn.execute(
                f"""UPDATE inbound_turns
                       SET state='completed', updated_at=?,
                           owner_pid=NULL, owner_started_at=NULL
                       WHERE turn_id IN ({placeholders})
                         AND state IN ('received', 'claimed')""",
                (now, *secondary_turn_ids),
            )
    _prune()
    return component_ids


def mark_delivery_component_attempting(component_id: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE delivery_components
               SET state='attempting',
                   attempts=attempts + CASE WHEN state='attempting' THEN 0 ELSE 1 END,
                   updated_at=?
               WHERE component_id=? AND state NOT IN ('delivered', 'abandoned')""",
            (time.time(), component_id),
        )


def _cleanup_staged_component(kind: str, payload_json: Optional[str]) -> None:
    if not payload_json:
        return
    try:
        payload = json.loads(payload_json)
        candidate = ""
        if kind in {"tts", "media", "file"}:
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
                         ORDER BY CASE state
                                    WHEN 'delivered' THEN 0
                                    WHEN 'abandoned' THEN 1
                                    ELSE 2
                                  END, updated_at ASC
                         LIMIT ?)""",
                    (excess,),
                )
            for table, id_column, terminal_states in (
                ("inbound_turns", "turn_id", ("completed", "abandoned")),
                ("delivery_components", "component_id", ("delivered", "abandoned")),
            ):
                placeholders = ",".join("?" for _ in terminal_states)
                active_inbound_guard = ""
                if table == "delivery_components":
                    active_inbound_guard = (
                        " AND NOT EXISTS (SELECT 1 FROM inbound_turns AS active "
                        "WHERE active.turn_id=delivery_components.turn_id "
                        "AND active.state IN ('received','claimed'))"
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
