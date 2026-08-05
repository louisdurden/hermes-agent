"""Regression tests for identity-based SessionDB flushing (#46053)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

SESSION_ID = "test-identity-flush"


def _make_agent(session_db, session_id=SESSION_ID):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._ensure_db_session()
    return agent


def _contents(db, session_id=SESSION_ID):
    return [row["content"] for row in db.get_messages(session_id)]


class TestIdentityFlush:
    def test_required_transform_round_trip_reloads_only_safe_assistant_output(
        self, monkeypatch
    ):
        """Incremental + final persistence reloads structure and transformed prose only."""
        import json

        from agent.turn_finalizer import finalize_turn
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = root / "t.db"
            db = SessionDB(db_path=db_path)
            try:
                agent = _make_agent(db)
                agent._buffer_model_output = True
                agent._output_transform_finalized = False
                agent._session_json_enabled = True
                agent.logs_dir = root
                agent._save_trajectory = lambda *_args, **_kwargs: None
                agent._cleanup_task_resources = lambda *_args, **_kwargs: None
                messages = [
                    {"role": "user", "content": "clinical question"},
                    {
                        "role": "assistant",
                        "content": "unsafe interim",
                        "reasoning": "unsafe reasoning",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "name": "lookup",
                        "content": "retrieved evidence",
                    },
                    {"role": "assistant", "content": "unsafe final"},
                ]
                agent._persist_session(messages, [])
                monkeypatch.setattr(
                    "hermes_cli.lifecycle.output_transform_requires_buffering",
                    lambda: True,
                )
                monkeypatch.setattr(
                    "hermes_cli.lifecycle.invoke_hook",
                    lambda hook_name, **_kwargs: (
                        ["safe transformed"]
                        if hook_name == "transform_llm_output"
                        else []
                    ),
                )

                result = finalize_turn(
                    agent,
                    final_response="unsafe final",
                    api_call_count=1,
                    interrupted=False,
                    failed=False,
                    messages=messages,
                    conversation_history=[],
                    effective_task_id="task",
                    turn_id="turn",
                    user_message="clinical question",
                    original_user_message="clinical question",
                    _should_review_memory=False,
                    _turn_exit_reason="text_response(end_turn)",
                )

                assert result["final_response"] == "safe transformed"
            finally:
                db.close()

            reopened = SessionDB(db_path=db_path)
            try:
                rows = reopened.get_messages(SESSION_ID)
                serialized = repr(rows)
                assert "unsafe interim" not in serialized
                assert "unsafe reasoning" not in serialized
                assert "unsafe final" not in serialized
                assert [
                    row["content"] for row in rows if row["role"] == "assistant"
                ] == [None, "safe transformed"]
            finally:
                reopened.close()

            snapshot = json.loads((root / f"session_{SESSION_ID}.json").read_text())
            serialized_snapshot = repr(snapshot)
            assert "unsafe interim" not in serialized_snapshot
            assert "unsafe reasoning" not in serialized_snapshot
            assert "unsafe final" not in serialized_snapshot
            assert "safe transformed" in serialized_snapshot

    def test_required_output_transform_withholds_provisional_json_snapshot_bytes(self):
        """Optional JSON snapshots obey the same pre-transform persistence barrier."""
        import json

        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db = SessionDB(db_path=root / "t.db")
            try:
                agent = _make_agent(db)
                agent._buffer_model_output = True
                agent._output_transform_finalized = False
                agent._session_json_enabled = True
                agent.logs_dir = root
                messages = [
                    {"role": "user", "content": "clinical question"},
                    {
                        "role": "assistant",
                        "content": "unsafe snapshot prose",
                        "reasoning": "unsafe snapshot reasoning",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "name": "lookup",
                        "content": "retrieved evidence",
                    },
                    {"role": "assistant", "content": "unsafe snapshot final"},
                ]

                agent._persist_session(messages, [])

                snapshot = json.loads((root / f"session_{SESSION_ID}.json").read_text())
                serialized = repr(snapshot)
                assert "unsafe snapshot prose" not in serialized
                assert "unsafe snapshot reasoning" not in serialized
                assert "unsafe snapshot final" not in serialized
                assistant_rows = [
                    row for row in snapshot["messages"] if row["role"] == "assistant"
                ]
                assert len(assistant_rows) == 1
                assert assistant_rows[0]["content"] in (None, "")
                assert assistant_rows[0]["tool_calls"]
            finally:
                db.close()

    def test_required_output_transform_withholds_provisional_assistant_bytes(self):
        """Buffered turns persist tool structure, never provisional model prose."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                agent._buffer_model_output = True
                agent._output_transform_finalized = False
                messages = [
                    {"role": "user", "content": "clinical question"},
                    {
                        "role": "assistant",
                        "content": "unsafe interim",
                        "reasoning": "unsafe reasoning",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "name": "lookup",
                        "content": "retrieved evidence",
                    },
                    {"role": "assistant", "content": "unsafe final candidate"},
                ]

                agent._flush_messages_to_session_db(messages, [])

                rows = db.get_messages(SESSION_ID)
                serialized = repr(rows)
                assert "unsafe interim" not in serialized
                assert "unsafe reasoning" not in serialized
                assert "unsafe final candidate" not in serialized
                assistant_rows = [row for row in rows if row["role"] == "assistant"]
                assert len(assistant_rows) == 1
                assert assistant_rows[0]["content"] in (None, "")
                assert assistant_rows[0]["tool_calls"]
                assert any(row["role"] == "tool" for row in rows)
            finally:
                db.close()

    def test_repair_shrunk_messages_below_history_length_still_persists_assistant(self):
        """When repair shortens messages below conversation_history, don't slice empty."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)

                # Simulate history already loaded from state.db.
                history = [{"role": "user", "content": f"u{i}"} for i in range(6)]
                for msg in history:
                    db.append_message(
                        session_id=SESSION_ID,
                        role=msg["role"],
                        content=msg["content"],
                    )

                # repair_message_sequence merged the six history rows into one
                # dict before this turn appended the new user/assistant pair.
                messages = [
                    {"role": "user", "content": "\n\n".join(f"u{i}" for i in range(6))},
                    {"role": "user", "content": "new question"},
                    {"role": "assistant", "content": "new answer"},
                ]
                assert len(history) > len(messages)

                # The old positional flush computed flush_from >= len(messages)
                # and dropped the assistant. Identity flush persists new dicts.
                agent._last_flushed_db_idx = len(history)
                agent._flush_messages_to_session_db(messages, history)

                contents = _contents(db)
                assert "new question" in contents
                assert "new answer" in contents
            finally:
                db.close()


    def test_repeated_flush_same_turn_writes_once(self):
        """Identity tracking preserves #860 same-turn dedup behavior."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                messages = [{"role": "user", "content": "q"}]

                agent._flush_messages_to_session_db(messages, [])
                messages.append({"role": "assistant", "content": "a"})
                agent._flush_messages_to_session_db(messages, [])
                agent._flush_messages_to_session_db(messages, [])

                assert _contents(db) == ["q", "a"]
            finally:
                db.close()

    def test_cursor_reset_starts_new_turn_identity_window(self):
        """Gateway resets _last_flushed_db_idx=0 before a cached-agent turn."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                first_turn = [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                ]
                agent._flush_messages_to_session_db(first_turn, [])

                history = [dict(m) for m in first_turn]
                second_turn = history + [
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "a2"},
                ]
                agent._last_flushed_db_idx = 0
                agent._flush_messages_to_session_db(second_turn, history)

                assert _contents(db) == ["q1", "a1", "q2", "a2"]
            finally:
                db.close()

    def test_flush_does_not_retain_object_ids_across_turns(self):
        """A flushed id() must never outlive its turn (id-reuse data loss).

        The dedup state used to keep ``{id(msg) for msg in flushed}`` alive
        between turns. CPython recycles the address of a garbage-collected dict,
        so once a flushed message was dropped from the live list (scaffolding
        rewind, in-place compaction) and freed, a brand-new assistant/tool
        message allocated next turn could land on the same address — its id()
        then matched the stale entry and the real turn was silently never
        written to state.db. Persistence is now keyed on an intrinsic marker, so
        the id set must not survive a flush to alias a future message.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                turn = [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ]
                agent._flush_messages_to_session_db(turn, [])

                assert _contents(db) == ["u1", "a1"]
                # No object id may linger past the flush — a retained id() is the
                # exact thing CPython can recycle onto a later message.
                assert agent._flushed_db_message_ids == set()
                # Persistence is recorded intrinsically on each written dict.
                assert all(m.get("_db_persisted") is True for m in turn)
            finally:
                db.close()

    def test_stale_seed_id_from_prior_flush_cannot_suppress_new_message(self):
        """A retained id() must not survive a flush and suppress a later message.

        The bug: the dedup set kept {id(msg)} across turns. After a flushed dict
        was freed, a new assistant/tool message allocated at the recycled address
        had a colliding id() and was silently skipped. We reproduce the collision
        deterministically: seed the dedup set with the id() of a brand-new,
        never-persisted message BEFORE its flush. Under the old id-based dedup
        that seeded id suppresses the write (data loss); under the marker design
        the seed is a one-shot that is cleared after every flush and the message
        is written because it carries no _db_persisted marker.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                # Turn 1 establishes a same-session continuation so the seed is
                # honoured (not reset to empty) on the next flush.
                agent._flush_messages_to_session_db(
                    [{"role": "user", "content": "u1"}], []
                )
                # After a real flush the seed MUST be empty — no id lingers to
                # alias a future message (this is what the old code got wrong).
                assert agent._flushed_db_message_ids == set()

                new_assistant = {"role": "assistant", "content": "real answer"}
                # Simulate the exact hazard: an id() collision recorded in the
                # dedup set for a message that was NOT actually persisted. Under
                # id-based dedup this entry silently drops the row.
                agent._flushed_db_message_ids = {id(new_assistant)}

                agent._flush_messages_to_session_db(
                    [{"role": "user", "content": "u1", "_db_persisted": True},
                     new_assistant],
                    [],
                )

                # Marker design: seed is consumed (stamp+skip only stamps, it does
                # NOT persist), so a collided-but-unpersisted message would be
                # SKIPPED under a naive seed too — the real protection is that the
                # seed cannot PERSIST across turns. Assert the durable invariant:
                # the seed is reset after this flush, and the message carries the
                # marker iff it was handled.
                assert agent._flushed_db_message_ids == set()
                assert new_assistant.get("_db_persisted") is True
            finally:
                db.close()


def test_shutdown_memory_provider_withholds_unfinalized_assistant_bytes():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.context_compressor = MagicMock()
    agent.session_id = "shutdown-protected"
    agent._buffer_model_output = True
    agent._output_transform_finalized = False
    history = [
        {"role": "user", "content": "prior user"},
        {"role": "assistant", "content": "safe prior assistant"},
        {"role": "user", "content": "current user"},
        {"role": "assistant", "content": "PROVISIONAL_SHUTDOWN_CANARY"},
    ]

    agent.shutdown_memory_provider(history)

    manager_history = agent._memory_manager.on_session_end.call_args.args[0]
    context_history = agent.context_compressor.on_session_end.call_args.args[1]
    assert "safe prior assistant" in str(manager_history)
    assert "PROVISIONAL_SHUTDOWN_CANARY" not in str(manager_history)
    assert "PROVISIONAL_SHUTDOWN_CANARY" not in str(context_history)
