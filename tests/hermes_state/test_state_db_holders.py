"""Behavioral tests for the state-holder and repair-admission authority."""

import errno
import os
import sys
from types import SimpleNamespace

import pytest

import hermes_state_holders


@pytest.mark.linux_only
def test_foreign_holder_accepts_same_inode_reached_through_an_alias(
    tmp_path, monkeypatch
):
    """Descriptor identity is authoritative even when /proc spells another path."""
    db_path = tmp_path / "state.db"
    db_path.touch()
    alias_path = tmp_path / "namespace-alias" / "state.db"

    proc_root = tmp_path / "proc"
    for pid in (111, 222):
        (proc_root / str(pid) / "fd").mkdir(parents=True)
    os.symlink(db_path, proc_root / "222" / "fd" / "3")

    monkeypatch.setattr(hermes_state_holders.os, "getpid", lambda: 111)
    real_listdir = os.listdir

    def _listdir(path):
        if isinstance(path, str):
            path = path.replace("/proc", str(proc_root))
        return real_listdir(path)

    monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)

    def _readlink(path):
        if path == "/proc/222/fd/3":
            return str(alias_path)
        return os.readlink(path.replace("/proc", str(proc_root)))

    monkeypatch.setattr(hermes_state_holders.os, "readlink", _readlink)
    real_stat = os.stat

    def _stat(path, *args, **kwargs):
        path = str(path).replace("/proc", str(proc_root))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(hermes_state_holders.os, "stat", _stat)

    assert hermes_state_holders.foreign_state_db_holders(db_path) == [
        (222, str(alias_path))
    ]


def test_foreign_holder_scan_timeout_remains_fail_closed(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    db_path.touch()

    def _timed_out(_attrs):
        raise TimeoutError(errno.ETIMEDOUT, "process scan timed out")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        hermes_state_holders,
        "psutil",
        SimpleNamespace(process_iter=_timed_out),
    )

    holders = hermes_state_holders.foreign_state_db_holders(db_path)

    assert len(holders) == 1
    assert holders[0][0] == -1
    assert holders[0][1].startswith("open-file scan failed:")
