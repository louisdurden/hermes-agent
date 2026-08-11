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


def test_durable_inbound_buffers_final_text_and_streaming_tts():
    from gateway.run import _model_output_delivery_flags

    assert _model_output_delivery_flags(
        streaming_enabled=True,
        interim_enabled=True,
        transform_required=False,
        durable_final_required=True,
    ) == (False, False, False)


def test_disabled_delivery_ledger_restores_legacy_streaming(monkeypatch):
    from gateway import delivery_ledger as dl
    from gateway.run import _durable_delivery_enabled, _model_output_delivery_flags

    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: False)

    assert _durable_delivery_enabled() is False
    assert _model_output_delivery_flags(
        streaming_enabled=True,
        interim_enabled=True,
        transform_required=False,
        durable_final_required=_durable_delivery_enabled(),
    ) == (True, True, True)


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


