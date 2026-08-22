import queue
from unittest.mock import patch

import pytest

from cli import HermesCLI


def _make_cli(goa_config):
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {"goa": goa_config} if goa_config is not None else {}
    cli._pending_input = queue.Queue()
    cli._pending_agent_seed = "existing seed"
    cli._pending_goa_config = {"existing": "config"}
    return cli


@pytest.mark.parametrize("goa_config", [None, {"enabled": False}])
def test_goa_rejects_when_disabled_without_mutating_pending_turn(goa_config):
    cli = _make_cli(goa_config)

    with patch("cli._cprint") as cprint:
        assert cli.process_command("/goa inspect the flaky test") is True

    assert cli._pending_agent_seed == "existing seed"
    assert cli._pending_goa_config == {"existing": "config"}
    assert "GoA está deshabilitado" in cprint.call_args.args[0]


def test_goa_queues_one_shot_when_enabled():
    cli = _make_cli({"enabled": True})

    with patch("cli._cprint"):
        assert cli.process_command("/goa inspect the flaky test") is True

    assert cli._pending_agent_seed == "inspect the flaky test"
    assert cli._pending_goa_config["enabled"] is True
