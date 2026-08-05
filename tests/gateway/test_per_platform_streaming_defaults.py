"""Per-platform streaming defaults + dashboard exposure.

Streaming is smooth on Telegram (native sendMessageDraft) but flickers on
edit-only platforms like Discord and Slack (repeated editMessage). The shipped
defaults encode that: display.platforms.telegram.streaming=true,
.discord.streaming=false, .slack.streaming=false. These are gap-fillers (user
values win via deep-merge) and, because the dashboard schema is generated from
DEFAULT_CONFIG, they automatically appear as editable toggles in the web UI.
"""

from __future__ import annotations


def test_buffered_transform_disables_every_partial_delivery_path():
    from gateway.run import _model_output_delivery_flags

    assert _model_output_delivery_flags(
        streaming_enabled=True,
        interim_enabled=True,
        transform_required=True,
    ) == (False, False, False)


def test_output_transform_hook_requires_buffering(monkeypatch):
    from gateway.run import _output_transform_requires_buffering

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "transform_llm_output")

    assert _output_transform_requires_buffering() is True


def test_default_per_platform_streaming_flags():
    from hermes_cli.config import DEFAULT_CONFIG
    plats = DEFAULT_CONFIG["display"]["platforms"]
    assert plats["telegram"]["streaming"] is True
    assert plats["discord"]["streaming"] is False
    assert plats["slack"]["streaming"] is False


