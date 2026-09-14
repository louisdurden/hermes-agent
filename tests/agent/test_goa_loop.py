from unittest.mock import patch

from agent.goa_loop import aggregate_guidance


def test_aggregate_guidance_is_noop_when_disabled():
    with patch("agent.goa_loop.call_llm") as call:
        assert aggregate_guidance(user_prompt="prompt", config={"enabled": False}) == ""
    call.assert_not_called()


def test_aggregate_guidance_requires_literal_true():
    with patch("agent.goa_loop.call_llm") as call:
        assert aggregate_guidance(user_prompt="prompt", config={"enabled": "true"}) == ""
    call.assert_not_called()
