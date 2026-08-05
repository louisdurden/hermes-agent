"""Tests for the ``transform_llm_output`` plugin hook.

The hook fires inside ``AIAgent.run_conversation`` once the tool-calling
loop has produced a final response. Driving the full agent loop from a
unit test would be prohibitively heavy, so these tests exercise the
invoke_hook dispatch semantics that the wiring in ``run_agent.py``
depends on:

    for _hook_result in _transform_results:
        if isinstance(_hook_result, str) and _hook_result:
            final_response = _hook_result
            break  # First non-empty string wins

Mirrors ``test_transform_tool_result_hook.py`` which tests the equivalent
contract for the generic tool-result seam.
"""

from pathlib import Path

import pytest
import yaml

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import PluginManager, VALID_HOOKS


def _make_enabled_plugin(hermes_home: Path, name: str, register_body: str) -> Path:
    """Create a plugin under <hermes_home>/plugins/<name> and opt it in."""
    plugin_dir = hermes_home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "0.1.0"}), encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        f"    {register_body}\n",
        encoding="utf-8",
    )
    cfg_path = hermes_home / "config.yaml"
    cfg = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg.setdefault("plugins", {}).setdefault("enabled", []).append(name)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return plugin_dir


def test_transform_llm_output_in_valid_hooks():
    assert "transform_llm_output" in VALID_HOOKS


def test_hook_receives_expected_kwargs(tmp_path, monkeypatch):
    """Hook callback should see response_text + session_id + model + platform."""
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_enabled_plugin(
        hermes_home, "capture_hook",
        register_body=(
            'ctx.register_hook("transform_llm_output", '
            'lambda **kw: f"{kw[\'response_text\']}|{kw[\'session_id\']}|'
            '{kw[\'model\']}|{kw[\'platform\']}")'
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    mgr = PluginManager()
    mgr.discover_and_load()

    results = mgr.invoke_hook(
        "transform_llm_output",
        response_text="hello world",
        session_id="s1",
        model="anthropic/claude-sonnet-4.6",
        platform="cli",
    )
    assert results == ["hello world|s1|anthropic/claude-sonnet-4.6|cli"]






def test_hook_exception_uses_registered_error_fallback(tmp_path, monkeypatch):
    """A safety-critical transform can replace output even when it raises."""
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_enabled_plugin(
        hermes_home, "fail_closed_hook",
        register_body=(
            'def _boom(**kw):\n'
            '        raise RuntimeError("boom")\n'
            '    ctx.register_hook('
            '"transform_llm_output", _boom, '
            'on_error="withheld by required transform")'
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    mgr = PluginManager()
    mgr.discover_and_load()

    results = mgr.invoke_hook(
        "transform_llm_output",
        response_text="unsafe original",
        session_id="s1",
        model="m",
        platform="telegram",
    )

    assert results == ["withheld by required transform"]


def test_transform_hook_exception_without_fallback_propagates(tmp_path, monkeypatch):
    """Complete-output transforms are safety boundaries and fail closed."""
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_enabled_plugin(
        hermes_home, "raising_hook",
        register_body=(
            'def _boom(**kw):\n'
            '        raise RuntimeError("boom")\n'
            '    ctx.register_hook("transform_llm_output", _boom)'
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    mgr = PluginManager()
    mgr.discover_and_load()

    with pytest.raises(RuntimeError, match="boom"):
        mgr.invoke_hook(
            "transform_llm_output",
            response_text="keep me",
            session_id="s1",
            model="m",
            platform="cli",
        )


def test_transform_hook_exception_with_none_fallback_propagates(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_enabled_plugin(
        hermes_home,
        "none_fallback_hook",
        register_body=(
            'def _boom(**kw):\n'
            '        raise RuntimeError("boom-none")\n'
            '    ctx.register_hook("transform_llm_output", _boom, on_error=None)'
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    mgr = PluginManager()
    mgr.discover_and_load()

    with pytest.raises(RuntimeError, match="boom-none"):
        mgr.invoke_hook("transform_llm_output", response_text="unsafe")


def test_no_plugins_returns_empty_results(tmp_path, monkeypatch):
    """With no plugins loaded, invoke_hook returns [] and the response is unchanged."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_empty"))
    plugins_mod._plugin_manager = PluginManager()

    mgr = plugins_mod._plugin_manager
    results = mgr.invoke_hook(
        "transform_llm_output",
        response_text="unchanged",
        session_id="",
        model="m",
        platform="",
    )
    assert results == []


def test_required_transform_buffers_display_tts_and_interim_callbacks():
    from run_agent import AIAgent

    delivered = []
    agent = AIAgent.__new__(AIAgent)
    agent._buffer_model_output = True
    agent.stream_delta_callback = lambda text: delivered.append(("display", text))
    agent._stream_callback = lambda text: delivered.append(("tts", text))
    agent.interim_assistant_callback = (
        lambda text, **_kwargs: delivered.append(("interim", text))
    )
    agent.reasoning_callback = lambda text: delivered.append(("reasoning", text))
    agent._stream_needs_break = False
    agent._stream_think_scrubber = None
    agent._stream_context_scrubber = None
    agent._current_streamed_assistant_text = ""

    agent._fire_stream_delta("unsafe clinical delta")
    agent._fire_reasoning_delta("unsafe clinical reasoning")
    agent._fire_streamed_codex_commentary("unsafe clinical commentary")
    agent._emit_interim_assistant_message(
        {"role": "assistant", "content": "unsafe clinical interim"}
    )

    assert delivered == []


def test_required_transform_drops_scrubber_flush_tails_before_display_or_tts():
    from run_agent import AIAgent

    class Scrubber:
        def __init__(self, tail: str):
            self.tail = tail

        def feed(self, text: str) -> str:
            return text

        def flush(self) -> str:
            return self.tail

    delivered = []
    agent = AIAgent.__new__(AIAgent)
    agent._buffer_model_output = True
    agent.stream_delta_callback = lambda text: delivered.append(("display", text))
    agent._stream_callback = lambda text: delivered.append(("tts", text))
    agent._stream_think_scrubber = Scrubber("UNSAFE_THINK_TAIL")
    agent._stream_context_scrubber = Scrubber("UNSAFE_CONTEXT_TAIL")
    agent._current_streamed_assistant_text = ""

    agent._reset_stream_delivery_tracking()

    assert delivered == []
    assert agent._current_streamed_assistant_text == ""


def test_output_transform_buffering_introspection_fails_closed(monkeypatch):
    from hermes_cli.lifecycle import output_transform_requires_buffering

    def _raise(_hook_name):
        raise RuntimeError("plugin registry unavailable")

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", _raise)

    assert output_transform_requires_buffering() is True
