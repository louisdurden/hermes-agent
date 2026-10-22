"""Drive the real cron run and agent loop across provider failures, without vendor calls."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import run_job
from run_agent import AIAgent


class ProviderError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status_code = status
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": message}}


@pytest.mark.parametrize("status,message", [
    (429, "rate limit exceeded"),
    (429, "usage limit reached"),
    (401, "invalid API key"),
    (503, "overloaded"),
])
def test_cron_uses_next_configured_model_in_same_run(tmp_path, monkeypatch, status, message):
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: primary-model\n  provider: openrouter\n"
        "cron:\n  preflight: false\n"
        "fallback_providers:\n  - provider: openrouter\n    model: backup-model\n    api_mode: chat_completions\n",
        encoding="utf-8",
    )
    runtime = {"api_key": "test-key", "base_url": "https://example.invalid/v1",
               "provider": "openrouter", "api_mode": "chat_completions"}
    calls = []
    agents = []

    def construct(**kwargs):
        kwargs.update(skip_memory=True, skip_context_files=True, load_soul_identity=False,
                      session_db=None, enabled_toolsets=[])
        agent = AIAgent(**kwargs)
        agent._api_max_retries = 1
        agents.append(agent)
        return agent

    def api_call(agent, api_kwargs):
        calls.append(agent.model)
        if agent.model == "primary-model":
            raise ProviderError(status, message)
        assert agent.model == "backup-model"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="backup answered", tool_calls=None),
                                     finish_reason="stop")],
            model="backup-model", usage=None,
        )

    client = MagicMock()
    client.api_key = "test-key"
    client.base_url = runtime["base_url"]
    client._custom_headers = client.default_headers = None
    with (
        patch("cron.scheduler._get_hermes_home", return_value=tmp_path),
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._reload_dotenv_and_publish_delivery_target"),
        patch("cron.scheduler._open_cron_session_db", return_value=None),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime),
        patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=client),
        patch("run_agent.AIAgent", side_effect=construct),
        patch.object(AIAgent, "_interruptible_api_call", autospec=True, side_effect=api_call),
        patch.object(AIAgent, "_persist_session"),
        patch.object(AIAgent, "_save_trajectory"),
        patch.object(AIAgent, "_cleanup_task_resources"),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "backup-model")) as resolve,
        patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
        patch("cron.scheduler._write_usage_audit") as audit,
    ):
        success, _output, final, error = run_job(
            {"id": "failover", "name": "failover", "prompt": "hello"}, defer_agent_teardown=[])
    assert success, error
    assert final == "backup answered"
    assert calls[-1] == "backup-model"
    assert calls.count("backup-model") == 1
    resolve.assert_called_once()
    assert agents[0].model == "backup-model"
    assert audit.call_args.args[0]["model"] == "backup-model"
