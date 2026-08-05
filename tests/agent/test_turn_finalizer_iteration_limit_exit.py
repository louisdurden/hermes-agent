"""Regression tests for iteration-limit exit normalization (#61631)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_finalizer import finalize_turn


class _LimitAgent:
    def __init__(
        self,
        *,
        max_iterations=60,
        budget_remaining=0,
        completion_explainer=False,
    ):
        self.max_iterations = max_iterations
        self.iteration_budget = SimpleNamespace(
            remaining=budget_remaining, used=max_iterations, max_total=max_iterations
        )
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages = None
        self._handle_max_iterations_called = False
        self._completion_explainer = completion_explainer

    def _handle_max_iterations(self, messages, api_call_count):
        self._handle_max_iterations_called = True
        return "summary from extra call"

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return self._completion_explainer

    def _format_turn_completion_explanation(self, _reason):
        return "iteration-limit explanation"

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _finalize(
    agent,
    *,
    final_response,
    exit_reason,
    api_call_count=60,
    pending_verification_response=None,
):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
        _pending_verification_response=pending_verification_response,
    )
















def test_transform_dispatch_failure_is_fail_closed_and_persisted_safe(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("dispatcher unavailable")

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _raise)
    agent = _LimitAgent()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "unsafe original response"},
    ]

    result = finalize_turn(
        agent,
        final_response="unsafe original response",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="text_response(end_turn)",
    )

    assert result["response_transformed"] is True
    assert result["final_response"] == (
        "A required output transform failed; response withheld."
    )
    assert agent.persisted_messages[-1]["content"] == result["final_response"]


def test_unconfirmed_required_transform_with_empty_results_is_fail_closed(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.lifecycle.output_transform_requires_buffering", lambda: True
    )
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda *_a, **_kw: False)
    agent = _LimitAgent()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "unsafe original response"},
    ]

    result = finalize_turn(
        agent,
        final_response="unsafe original response",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="text_response(end_turn)",
    )

    assert result["final_response"] == (
        "A required output transform failed; response withheld."
    )
    assert result["response_transformed"] is True
    assert agent.persisted_messages[-1]["content"] == result["final_response"]


@pytest.mark.parametrize("transform_result", [None, ""])
def test_confirmed_required_transform_without_nonempty_value_is_fail_closed(
    monkeypatch, transform_result
):
    monkeypatch.setattr(
        "hermes_cli.lifecycle.output_transform_requires_buffering", lambda: True
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", lambda *_a, **_kw: [transform_result]
    )
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda *_a, **_kw: True)
    agent = _LimitAgent()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "PROVISIONAL_OUTPUT_CANARY"},
    ]

    result = finalize_turn(
        agent,
        final_response="PROVISIONAL_OUTPUT_CANARY",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="text_response(end_turn)",
    )

    assert result["final_response"] == (
        "A required output transform failed; response withheld."
    )
    assert result["response_transformed"] is True
    assert agent.persisted_messages[-1]["content"] == result["final_response"]


def test_interrupted_partial_response_is_transformed_before_persistence(monkeypatch):
    safe = "withheld by required transform"

    def _invoke(hook_name, **_kwargs):
        return [safe] if hook_name == "transform_llm_output" else []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke)
    agent = _LimitAgent()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "unsafe partial response"},
    ]

    result = finalize_turn(
        agent,
        final_response="unsafe partial response",
        api_call_count=1,
        interrupted=True,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="interrupted_by_user",
    )

    assert result["final_response"] == safe
    assert result["response_transformed"] is True
    assert agent.persisted_messages[-1]["content"] == safe


def test_required_transform_sanitizes_interims_reasoning_and_post_transform_explainer(
    monkeypatch,
):
    safe = "withheld"
    monkeypatch.setattr(
        "hermes_cli.lifecycle.output_transform_requires_buffering", lambda: True
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda hook_name, **_kwargs: [safe] if hook_name == "transform_llm_output" else [],
    )
    agent = _LimitAgent(completion_explainer=True)
    messages = [
        {"role": "user", "content": "clinical task"},
        {
            "role": "assistant",
            "content": "unsafe interim",
            "reasoning": "unsafe clinical reasoning",
        },
        {"role": "assistant", "content": "unsafe final"},
    ]

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
        user_message="clinical task",
        original_user_message="clinical task",
        _should_review_memory=False,
        _turn_exit_reason="partial_stream_recovery",
    )

    assistants = [
        message for message in agent.persisted_messages
        if message.get("role") == "assistant"
    ]
    assert result["final_response"] == safe
    assert result["last_reasoning"] is None
    assert "iteration-limit explanation" not in result["final_response"]
    assert all(not message.get("reasoning") for message in assistants)
    assert all("_db_persisted" not in message for message in assistants)
    assert [message.get("content") for message in assistants] == [safe]
    assert agent._output_transform_finalized is True


def test_captured_required_transform_is_monotonic_when_registry_changes(monkeypatch):
    safe = "withheld"
    monkeypatch.setattr(
        "hermes_cli.lifecycle.output_transform_requires_buffering", lambda: False
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda hook_name, **_kwargs: [safe] if hook_name == "transform_llm_output" else [],
    )
    agent = _LimitAgent()
    agent._buffer_model_output = True
    messages = [
        {"role": "user", "content": "clinical task"},
        {"role": "assistant", "content": "unsafe final"},
    ]

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
        user_message="clinical task",
        original_user_message="clinical task",
        _should_review_memory=False,
        _turn_exit_reason="text_response(end_turn)",
    )

    assert result["final_response"] == safe
    assert result["response_transformed"] is True
    assert agent._buffer_model_output is False
    assert agent._output_transform_finalized is True


@pytest.mark.parametrize(
    ("exit_reason", "interrupted", "failed"),
    [
        ("interrupted_by_user", True, False),
        ("all_retries_exhausted_no_response", False, False),
        ("provider_failure", False, True),
    ],
)
def test_pending_response_does_not_mask_later_terminal_exit(
    monkeypatch, exit_reason, interrupted, failed
):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()

    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=interrupted,
        failed=failed,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
        _pending_verification_response="stale premature report",
    )

    assert result["final_response"] is None
    assert result["turn_exit_reason"] == exit_reason
    assert result["completed"] is False
    assert agent._handle_max_iterations_called is False


def test_pending_response_records_kanban_timeout(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db._record_task_failure", record)
    agent = _LimitAgent()

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="unknown",
        pending_verification_response="composed report",
    )

    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    record.assert_called_once_with(
        conn,
        "task-123",
        error=(
            "Iteration budget exhausted (60/60) — task could not complete "
            "within the allowed iterations"
        ),
        outcome="timed_out",
        release_claim=True,
        end_run=True,
        event_payload_extra={"budget_used": 60, "budget_max": 60},
    )


def test_published_pending_candidate_is_not_duplicated_by_finalizer(monkeypatch):
    """When budget exhaustion preserves a verification candidate that is
    already the tail assistant message, the finalizer must NOT append a
    duplicate. The content-comparison guard prevents this. (#65919 §7)
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    report = "the composed report"

    result = finalize_turn(
        agent,
        final_response=report,
        api_call_count=60,
        interrupted=False,
        failed=False,
        # The candidate is already in messages as the tail assistant.
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": report},
        ],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
        _pending_verification_response=report,
    )

    # The tail assistant already matches final_response — no duplicate appended.
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["user", "assistant"]
    # Persisted messages should also have no duplicate.
    assert agent.persisted_messages is not None
    persisted_roles = [m["role"] for m in agent.persisted_messages]
    assert persisted_roles == ["user", "assistant"]


