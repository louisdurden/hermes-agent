import json
from types import SimpleNamespace

from agent.goa_loop import aggregate_goa_context
from hermes_cli.goa_config import build_goa_turn_prompt, decode_goa_turn


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _config():
    return {
        "enabled": False,
        "top_k": 3,
        "profiles": ["fleetliterature", "fleetcontent", "fleetpremium"],
        "meta": {"provider": "openai-codex", "model": "gpt-5.6-sol", "reasoning_effort": "medium"},
    }


def test_goa_one_turn_selects_both_directions_pools_and_stays_in_budget(monkeypatch):
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        stage = kwargs["task"].removeprefix("goa_")
        if stage == "selection":
            return _response(json.dumps({"selected": ["fleetliterature", "fleetcontent", "fleetpremium"]}))
        if stage == "ranking":
            return _response(json.dumps({"scores": {"fleetliterature": 0.9, "fleetcontent": 0.6, "fleetpremium": 0.3}}))
        if stage == "pooling":
            return _response("pooled guidance")
        return _response(prompt)

    monkeypatch.setattr("agent.goa_loop.call_llm", fake_call_llm)
    result = aggregate_goa_context(user_prompt="clinical evidence", config=_config())

    assert "pooled guidance" in result
    assert len(calls) == 12
    assert sum(call["max_tokens"] for call in calls) == 7000
    refinements = [call["messages"][-1]["content"] for call in calls if call["task"] == "goa_refinement"]
    assert len(refinements) == 4
    assert any("fleetliterature:" in text for text in refinements)
    assert any("fleetpremium:" in text for text in refinements)
    assert all(call["tools"] is None for call in calls)


def test_goa_degrades_explicitly_when_a_profile_node_fails(monkeypatch):
    def fake_call_llm(**kwargs):
        stage = kwargs["task"].removeprefix("goa_")
        prompt = kwargs["messages"][-1]["content"]
        if stage == "selection":
            return _response(json.dumps({"selected": ["fleetliterature", "fleetcontent", "fleetpremium"]}))
        if stage == "answer" and "fleetcontent" in prompt:
            raise RuntimeError("profile offline")
        return _response(json.dumps({"scores": {"fleetliterature": 1, "fleetpremium": 0}}) if stage == "ranking" else "pooled")

    monkeypatch.setattr("agent.goa_loop.call_llm", fake_call_llm)
    result = aggregate_goa_context(user_prompt="clinical evidence", config=_config())

    assert "Failed nodes: fleetcontent" in result


def test_goa_marker_is_explicit_and_round_trips():
    encoded = build_goa_turn_prompt("only this turn", _config())
    prompt, config = decode_goa_turn(encoded)

    assert prompt == "only this turn"
    assert config["enabled"] is False
    assert config["top_k"] == 3
