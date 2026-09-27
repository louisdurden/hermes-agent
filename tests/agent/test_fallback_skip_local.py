from types import SimpleNamespace

import agent.chat_completion_helpers as h


def _agent(base_url):
    return SimpleNamespace(base_url=base_url, _fallback_chain=[{"provider": "anthropic", "model": "x"}],
                           _fallback_index=0)


def test_skips_fallback_for_local_primary_when_enabled(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"fallback_skip_local": True})
    agent = _agent("http://127.0.0.1:8000/v1")
    assert h._skip_fallback_for_local_primary(agent) is True
    assert h.try_activate_fallback(agent) is False
    assert agent._fallback_index == 0


def test_keeps_fallback_for_cloud_primary(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"fallback_skip_local": True})
    assert h._skip_fallback_for_local_primary(_agent("https://api.openai.com/v1")) is False


def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    assert h._skip_fallback_for_local_primary(_agent("http://127.0.0.1:8000/v1")) is False
