from unittest.mock import patch

from agent.turn_request_assembly import append_goa_context


def test_append_goa_context_does_nothing_without_opt_in():
    messages = [{"role": "user", "content": "hello"}]
    with patch("agent.goa_loop.aggregate_guidance") as aggregate:
        guidance = append_goa_context(messages, {"enabled": False}, "hello")
    assert guidance == ""
    assert messages == [{"role": "user", "content": "hello"}]
    aggregate.assert_not_called()


def test_append_goa_context_generates_once_and_reuses_guidance():
    messages = [{"role": "user", "content": "hello"}]
    with patch("agent.goa_loop.aggregate_guidance", return_value="consider safety") as aggregate:
        guidance = append_goa_context(messages, {"enabled": True}, "hello")
        rebuilt = [{"role": "user", "content": "hello"}]
        reused = append_goa_context(rebuilt, {"enabled": True}, "hello", guidance)
    assert reused == guidance == "consider safety"
    assert messages[-1]["content"].endswith("consider safety")
    assert rebuilt[-1]["content"].endswith("consider safety")
    aggregate.assert_called_once()
