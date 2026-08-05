"""Regression tests for #72680 (retargeted).

The earlier attempt (#73171) snapshotted GatewayRunner._pending_messages, which
on current main has no writers — the live container is the per-agent
``agent._session_messages`` flushed via ``_flush_messages_to_session_db``.
When that flush raises (FTS/SQLite corruption) the in-memory transcript must
be dumped to a recovery snapshot instead of lost.

These tests exercise the real preservation path:
``_finalize_shutdown_agents`` -> flush raises -> ``_preserve_agent_history_on_shutdown``
-> ``flush_agent_history_to_file``.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.shutdown_flush import (
    flush_agent_history_to_file,
)


def _make_flush_dir(tmp_path: Path) -> Path:
    """Create a temp flush dir and monkeypatch _get_flush_dir to use it."""
    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir(parents=True, exist_ok=True)
    return flush_dir


def test_preserves_agent_history_when_flush_raises(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    history = [{"role": "user", "content": "lost msg"}]
    flush_agent_history_to_file("sess:abc123", history)

    files = list(flush_dir.glob("*.json"))
    assert files, "expected recovery snapshot"
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["issue"] == "#72680"
    assert data["session_id"] == "sess:abc123"
    assert data["count"] == 1
    assert data["messages"][0]["content"] == "lost msg"


def test_no_recovery_file_on_empty_history(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    flush_agent_history_to_file("sess:abc123", [])
    assert not list(flush_dir.glob("*.json"))


def test_unfinalized_required_transform_recovery_excludes_assistant_bytes(
    tmp_path, monkeypatch
):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    history = [
        {"role": "user", "content": "prior user input"},
        {"role": "assistant", "content": "safe prior assistant output"},
        {"role": "user", "content": "safe user input"},
        {
            "role": "assistant",
            "content": "PROVISIONAL_OUTPUT",
            "reasoning": "PROVISIONAL_REASONING",
        },
    ]

    flush_agent_history_to_file(
        "sess:abc123", history, include_assistant=False
    )

    files = list(flush_dir.glob("*.json"))
    assert files
    data = json.loads(files[0].read_text(encoding="utf-8"))
    serialized = json.dumps(data)
    assert "safe prior assistant output" in serialized
    assert "safe user input" in serialized
    assert "PROVISIONAL_OUTPUT" not in serialized
    assert "PROVISIONAL_REASONING" not in serialized
    assert data["count"] == 3


@pytest.mark.asyncio
async def test_gateway_marks_unfinalized_required_transform_history_unsafe():
    from gateway.run import GatewayRunner

    agent = SimpleNamespace(
        session_id="sess:abc123",
        _buffer_model_output=True,
        _output_transform_finalized=False,
        _session_messages=[
            {"role": "user", "content": "safe"},
            {"role": "assistant", "content": "PROVISIONAL_OUTPUT"},
        ],
        _flush_messages_to_session_db=lambda _messages: (_ for _ in ()).throw(
            RuntimeError("synthetic flush failure")
        ),
    )
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._cleanup_agent_resources_off_loop = AsyncMock()

    with (
        patch("gateway.shutdown_flush.flush_agent_history_to_file") as preserve,
        patch("hermes_cli.lifecycle.finalize_session"),
    ):
        await runner._finalize_shutdown_agents({"synthetic": agent})

    preserve.assert_called_once_with(
        "sess:abc123", agent._session_messages, include_assistant=False
    )


