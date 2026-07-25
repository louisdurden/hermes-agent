"""Tests for GatewayRunner._is_duplicate_telegram_update (FX-150/151, 2026-07-25).

Real-world motivation: under Mac-primary/Railway-failover, a brief overlap
window can exist where two separate processes (the Mac's gateway and a
freshly-promoted Railway gateway) both attempt to serve the same Telegram
update_id before Telegram's single-active-poller rule kicks one out. This
guard makes a redelivered update_id a no-op reply regardless of which
process (or restart of the same process) sees it, as long as the prior
delivery's dedup entry is still on disk within the window.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import gateway.run as run_mod


def _make_event(update_id, platform="telegram"):
    return SimpleNamespace(
        platform_update_id=update_id,
        source=SimpleNamespace(platform=SimpleNamespace(value=platform)),
        internal=False,
    )


def _guard(self_stub, event):
    return run_mod.GatewayRunner._is_duplicate_telegram_update(self_stub, event)


class _SelfStub:
    _TELEGRAM_DEDUP_WINDOW_SECONDS = run_mod.GatewayRunner._TELEGRAM_DEDUP_WINDOW_SECONDS
    _TELEGRAM_DEDUP_MAX_ENTRIES = run_mod.GatewayRunner._TELEGRAM_DEDUP_MAX_ENTRIES


def test_first_delivery_is_not_a_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)
    assert _guard(_SelfStub(), _make_event(111)) is False


def test_same_update_id_redelivered_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)
    stub = _SelfStub()
    assert _guard(stub, _make_event(222)) is False
    assert _guard(stub, _make_event(222)) is True


def test_different_update_ids_are_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)
    stub = _SelfStub()
    assert _guard(stub, _make_event(1)) is False
    assert _guard(stub, _make_event(2)) is False


def test_persists_across_a_fresh_process_reading_the_same_store(tmp_path, monkeypatch):
    """Simulates a restarted/handed-off process (new object, same disk state)."""
    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)
    stub_a = _SelfStub()
    assert _guard(stub_a, _make_event(333)) is False

    stub_b = _SelfStub()  # stands in for a fresh process instance
    assert _guard(stub_b, _make_event(333)) is True


def test_non_telegram_platform_is_never_gated(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)
    stub = _SelfStub()
    event = _make_event(444, platform="discord")
    assert _guard(stub, event) is False
    assert _guard(stub, event) is False  # still False -- not tracked at all


def test_missing_update_id_is_never_gated(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)
    stub = _SelfStub()
    event = _make_event(None)
    assert _guard(stub, event) is False


def test_entries_older_than_window_are_pruned_and_reusable(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)
    store = tmp_path / ".telegram_processed_updates.json"
    stale_ts = time.time() - run_mod.GatewayRunner._TELEGRAM_DEDUP_WINDOW_SECONDS - 5
    store.write_text(json.dumps({"entries": {"555": stale_ts}}))

    stub = _SelfStub()
    # The stale entry should have been pruned, so this update_id reads as new.
    assert _guard(stub, _make_event(555)) is False


def test_store_is_bounded_to_max_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)
    stub = _SelfStub()
    cap = run_mod.GatewayRunner._TELEGRAM_DEDUP_MAX_ENTRIES
    for update_id in range(cap + 20):
        _guard(stub, _make_event(update_id))
    store = tmp_path / ".telegram_processed_updates.json"
    data = json.loads(store.read_text())
    assert len(data["entries"]) <= cap


def test_corrupted_store_fails_open_to_not_gated_rather_than_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)
    store = tmp_path / ".telegram_processed_updates.json"
    store.write_text("not valid json{{{")
    stub = _SelfStub()
    assert _guard(stub, _make_event(666)) is False
