"""Fleet supervision on the existing dispatcher tick, without user notifications.

A progress heartbeat is one observation at least 60 seconds after the previous
observation, independent of worker keepalives or a faster dispatch interval.
Two unchanged heartbeats park a task; the first observation only sets a baseline.
Alfred consumes durable records in ``kanban/supervisor-inbox``; nothing is sent
through requester notification subscriptions.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FLEET_BOARD = "alfred-fleet"
PROGRESS_HEARTBEAT_SECONDS = 60
_SUPERVISOR = "alfred-progress-watchdog"
_IGNORED_EVENTS = ("heartbeat", "progress_observed", "progress_stalled")
_PATH_RE = re.compile(r"(?<!\w)(?:~/|/)[^\s<>\"`]+")


def _artifact_signature(conn: sqlite3.Connection, task: sqlite3.Row) -> list:
    paths = {r[0] for r in conn.execute(
        "SELECT stored_path FROM task_attachments WHERE task_id = ?", (task["id"],),
    )}
    for field in ("body", "result"):
        paths.update(match.rstrip(".,;:)'\"]}") for match in _PATH_RE.findall(task[field] or ""))
    stats = []
    for value in sorted(paths):
        path = Path(value).expanduser()
        try:
            stat = path.stat()
            stats.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            stats.append((str(path), None, None))
    return stats


def _signature(conn: sqlite3.Connection, task: sqlite3.Row) -> str:
    event = conn.execute(
        "SELECT COALESCE(MAX(id), 0) FROM task_events WHERE task_id = ? "
        "AND kind NOT IN (?, ?, ?) AND NOT (kind = 'commented' AND "
        "COALESCE(json_extract(CASE WHEN json_valid(payload) THEN payload ELSE '{}' END, '$.author'), '') = ?)",
        (task["id"], *_IGNORED_EVENTS, _SUPERVISOR),
    ).fetchone()[0]
    comment = conn.execute(
        "SELECT COALESCE(MAX(id), 0) FROM task_comments WHERE task_id = ? AND author != ?",
        (task["id"], _SUPERVISOR),
    ).fetchone()[0]
    data = (task["current_run_id"], task["started_at"], event, comment, _artifact_signature(conn, task))
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    from hermes_cli import kanban_db as kb
    with kb.write_txn(conn):
        conn.execute("CREATE TABLE IF NOT EXISTS fleet_progress_snapshots ("
                     "task_id TEXT PRIMARY KEY, signature TEXT NOT NULL, observed_at INTEGER NOT NULL, "
                     "idle_beats INTEGER NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS fleet_supervisor_inbox ("
                     "key TEXT PRIMARY KEY, body TEXT NOT NULL)")


def _park_stalled(
    conn: sqlite3.Connection, task: sqlite3.Row, now: int, expected_signature: str | None = None,
) -> bool:
    from hermes_cli import kanban_db as kb
    reason = "stalled: two 60-second progress heartbeats without a new event, comment or artifact change"
    with kb.write_txn(conn):
        current = conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
        if current is None or current["status"] != "running":
            return False
        if expected_signature is not None and _signature(conn, current) != expected_signature:
            return False
        changed = conn.execute(
            "UPDATE tasks SET status = 'blocked', block_kind = 'stalled', "
            "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND status = 'running' AND current_run_id IS ?",
            (task["id"], task["current_run_id"]),
        ).rowcount
        if not changed:
            return False
        run_id = kb._end_or_synthesize_run(
            conn, task["id"], outcome="blocked", status="blocked", summary=reason, synthesize=True,
        )
        conn.execute("INSERT INTO task_comments(task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
                     (task["id"], _SUPERVISOR, reason, now))
        # A dedicated event avoids the user-facing blocked notification channel.
        kb._append_event(conn, task["id"], "progress_stalled", {"reason": "stalled", "recipient": "alfred"}, run_id=run_id)
        key = f"stalled-{task['id']}-{run_id or now}"
        body = json.dumps({"recipient": "alfred", "task_id": task["id"], "assignee": task["assignee"],
                           "reason": "stalled", "observed_at": now})
        conn.execute("INSERT OR IGNORE INTO fleet_supervisor_inbox VALUES (?, ?)", (key, body))
    # A standalone Kanban worker has no requester delivery route. Alert Alfred
    # through his queue, never through a cron whose output goes to Telegram.
    if task["created_by"] != _SUPERVISOR:
        kb.create_task(
            conn, title="Alfred: diagnose stalled task " + task["id"], assignee="default",
            created_by=_SUPERVISOR, board=FLEET_BOARD, idempotency_key=key,
            body="task_type: research\nresource_key: " + key + "\n"
                 "Internal supervisor notice for Alfred only: task " + task["id"] + " is stalled. "
                 "Inspect its events, comments and artifacts; record the confirmed diagnosis "
                 "as a comment and a durable local artifact. Do not send messages, publish, "
                 "modify profiles or notify Rodrigo. Do not retry external effects. "
                 "Complete with result JSON artifact_path. Preserve uncertainty.",
        )
    return True


def _daily_summary(conn: sqlite3.Connection, now: int) -> None:
    from hermes_cli import kanban_db as kb
    day = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
    key = f"daily-{day}"
    if conn.execute("SELECT 1 FROM fleet_supervisor_inbox WHERE key = ?", (key,)).fetchone():
        return
    rows = conn.execute(
        "SELECT t.assignee, t.block_kind FROM tasks t WHERE t.status = 'blocked' "
        "AND COALESCE((SELECT MAX(e.created_at) FROM task_events e WHERE e.task_id = t.id "
        "AND e.kind IN ('blocked', 'progress_stalled', 'spawn_auto_blocked')), t.created_at) < ?",
        (now - 72 * 3600,),
    ).fetchall()
    if not rows:
        return
    groups = Counter(f"{r['assignee'] or 'unassigned'}/{r['block_kind'] or 'untyped'}" for r in rows)
    line = f"{day} Alfred: bloqueadas >72 h ({len(rows)}): " + ", ".join(f"{key}={n}" for key, n in sorted(groups.items()))
    with kb.write_txn(conn):
        conn.execute("INSERT OR IGNORE INTO fleet_supervisor_inbox VALUES (?, ?)", (key, line))


def _flush_inbox(conn: sqlite3.Connection) -> None:
    from hermes_cli import kanban_db as kb
    directory = kb.kanban_home() / "kanban" / "supervisor-inbox" / FLEET_BOARD
    records = conn.execute("SELECT key, body FROM fleet_supervisor_inbox").fetchall()
    if not records:
        return
    directory.mkdir(parents=True, exist_ok=True)
    for key, body in records:
        suffix = ".txt" if key.startswith("daily-") else ".json"
        target = directory / (key + suffix)
        if not target.exists():
            temp = target.with_suffix(".tmp")
            temp.write_text(body + "\n", encoding="utf-8")
            temp.replace(target)


def observe_progress(conn: sqlite3.Connection, *, now: int | None = None) -> list[str]:
    """Observe fleet progress under the dispatch lock; return newly parked ids."""
    from hermes_cli import kanban_db as kb
    now = int(time.time()) if now is None else now
    _ensure_schema(conn)
    stalled = []
    for task in conn.execute("SELECT * FROM tasks WHERE status = 'running'").fetchall():
        previous = conn.execute("SELECT * FROM fleet_progress_snapshots WHERE task_id = ?", (task["id"],)).fetchone()
        if previous and now - previous["observed_at"] < PROGRESS_HEARTBEAT_SECONDS:
            continue
        signature = _signature(conn, task)
        idle = previous["idle_beats"] + 1 if previous and previous["signature"] == signature else 0
        with kb.write_txn(conn):
            conn.execute("INSERT OR REPLACE INTO fleet_progress_snapshots VALUES (?, ?, ?, ?)",
                         (task["id"], signature, now, idle))
        if idle >= 2 and _park_stalled(conn, task, now, signature):
            stalled.append(task["id"])
    with kb.write_txn(conn):
        conn.execute("DELETE FROM fleet_progress_snapshots WHERE task_id NOT IN (SELECT id FROM tasks WHERE status = 'running')")
    _daily_summary(conn, now)
    _flush_inbox(conn)
    return stalled


# Stable local job identities, not mutable display names or prompt substrings.
ALFRED_SUPERVISOR_JOB = "9d6eb13533db"
ALFRED_BRIEF_JOB = "5c25e0604dd6"


def render_supervisor_context(job_id: str) -> str:
    """Read bounded fleet facts for Alfred's existing default-profile cron jobs.

    Rendering does not acknowledge delivery: failed cron runs retain the facts,
    and active stalled tasks disappear only after Alfred resolves their state.
    """
    from hermes_constants import get_default_hermes_root, get_hermes_home
    from hermes_cli import kanban_db as kb

    if job_id != ALFRED_BRIEF_JOB:
        return ""
    if get_hermes_home().resolve() != get_default_hermes_root().resolve():
        return ""
    path = kb.kanban_db_path(board=FLEET_BOARD)
    if not path.exists():
        return ""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'fleet_supervisor_inbox'").fetchone():
            return ""
        day = datetime.fromtimestamp(time.time(), timezone.utc).date().isoformat()
        row = conn.execute("SELECT body FROM fleet_supervisor_inbox WHERE key = ?", (f"daily-{day}",)).fetchone()
        return row[0] if row else ""
