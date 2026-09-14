from unittest.mock import patch

from cli import HermesCLI


def _make_cli(enabled):
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {"goa": {"enabled": enabled}}
    cli._pending_agent_seed = None
    cli._pending_goa_config = None
    cli._agent_running = False
    return cli


def test_goa_command_is_fail_closed_when_not_enabled():
    cli = _make_cli(False)
    with patch("cli._cprint") as output:
        assert cli.process_command("/goa inspect this") is True
    assert cli._pending_agent_seed is None
    assert cli._pending_goa_config is None
    assert "disabled" in output.call_args.args[0].lower()


def test_goa_command_queues_prompt_after_opt_in():
    cli = _make_cli(True)
    with patch("cli._cprint"):
        assert cli.process_command("/goa inspect this") is True
    assert cli._pending_agent_seed == "inspect this"
    assert cli._pending_goa_config["enabled"] is True
    assert cli._pending_goa_config["token_budget"] == 7000
    assert cli._pending_goa_config["invocation_budget"] == 12


def test_goa_command_without_prompt_only_shows_usage():
    cli = _make_cli(True)
    with patch("cli._cprint") as output:
        assert cli.process_command("/goa") is True
    assert cli._pending_agent_seed is None
    assert cli._pending_goa_config is None
    assert "usage" in output.call_args.args[0].lower()
