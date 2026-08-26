"""Regression tests for uv-managed Python rotation breaking a live venv."""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path

from hermes_cli.runtime_venv_preflight import run_preflight


def _site_packages(venv: Path) -> Path:
    return venv / "lib" / "python3.11" / "site-packages"


def _make_fake_uv(tmp_path: Path, interpreter: Path) -> tuple[Path, Path]:
    log = tmp_path / "uv-calls.jsonl"
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir(parents=True, exist_ok=True)
    uv.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json
            import os
            import sys
            from pathlib import Path

            log = Path({str(log)!r})
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sys.argv[1:]) + "\\n")

            if sys.argv[1:4] == ["python", "find", "--managed-python"]:
                print({str(interpreter)!r})
                raise SystemExit(0)

            if sys.argv[1] == "venv":
                target = Path(sys.argv[-1])
                bin_dir = target / "bin"
                bin_dir.mkdir(parents=True, exist_ok=True)
                python = bin_dir / "python"
                if python.exists() or python.is_symlink():
                    python.unlink()
                python.symlink_to({str(interpreter)!r})
                (target / "pyvenv.cfg").write_text(
                    "home = " + str(Path({str(interpreter)!r}).parent) + "\\n"
                    "implementation = CPython\\n"
                    "version_info = 3.11\\n"
                    "include-system-site-packages = false\\n",
                    encoding="utf-8",
                )
                raise SystemExit(0)

            raise SystemExit(64)
            """
        ),
        encoding="utf-8",
    )
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    return uv, log


def _make_versioned_interpreter(tmp_path: Path) -> Path:
    interpreter = (
        tmp_path
        / "uv"
        / "python"
        / "cpython-3.11.16-macos-aarch64-none"
        / "bin"
        / "python3.11"
    )
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    return interpreter


def test_repairs_missing_versioned_uv_home_without_removing_site_packages(tmp_path):
    venv = tmp_path / "hermes-agent" / "venv"
    old_home = (
        tmp_path
        / "uv"
        / "python"
        / "cpython-3.11.15-macos-aarch64-none"
        / "bin"
    )
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        f"home = {old_home}\nimplementation = CPython\nversion_info = 3.11\n",
        encoding="utf-8",
    )
    sentinel = _site_packages(venv) / "keep-me.dist-info" / "METADATA"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("Name: keep-me\n", encoding="utf-8")

    current = _make_versioned_interpreter(tmp_path)
    uv, log = _make_fake_uv(tmp_path, current)

    result = run_preflight(venv=venv, uv=uv)

    assert result["status"] == "repaired"
    assert result["python_version"].startswith("3.11.")
    assert sentinel.read_text(encoding="utf-8") == "Name: keep-me\n"
    assert (venv / "pyvenv.cfg.runtime-preflight.bak").read_text(
        encoding="utf-8"
    ).startswith(f"home = {old_home}\n")
    assert f"home = {current.parent}" in (venv / "pyvenv.cfg").read_text(
        encoding="utf-8"
    )
    assert [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] == [
        ["python", "find", "--managed-python", "3.11"],
        ["venv", "--python", str(current), "--allow-existing", str(venv)],
    ]


def test_healthy_second_run_is_idempotent_and_does_not_call_uv(tmp_path):
    venv = tmp_path / "hermes-agent" / "venv"
    old_home = tmp_path / "uv" / "python" / "cpython-3.11.15" / "bin"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {old_home}\n", encoding="utf-8")
    _site_packages(venv).mkdir(parents=True)
    current = _make_versioned_interpreter(tmp_path)
    uv, log = _make_fake_uv(tmp_path, current)

    assert run_preflight(venv=venv, uv=uv)["status"] == "repaired"
    calls_after_repair = log.read_text(encoding="utf-8")

    result = run_preflight(venv=venv, uv=uv)

    assert result["status"] == "healthy"
    assert log.read_text(encoding="utf-8") == calls_after_repair


def test_fails_closed_when_uv_cannot_resolve_current_python(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    missing_home = tmp_path / "missing" / "bin"
    cfg = venv / "pyvenv.cfg"
    original = f"home = {missing_home}\nversion_info = 3.11\n"
    cfg.write_text(original, encoding="utf-8")
    sentinel = _site_packages(venv) / "sentinel"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep", encoding="utf-8")
    uv = tmp_path / "uv"
    uv.write_text(f"#!{sys.executable}\nraise SystemExit(23)\n", encoding="utf-8")
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)

    result = run_preflight(venv=venv, uv=uv)

    assert result == {
        "status": "blocked",
        "reason": "uv-python-find-failed",
        "returncode": 23,
    }
    assert cfg.read_text(encoding="utf-8") == original
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_fails_closed_on_malformed_pyvenv_cfg(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("version_info = 3.11\n", encoding="utf-8")
    uv = tmp_path / "uv"
    uv.write_text("must not run", encoding="utf-8")

    assert run_preflight(venv=venv, uv=uv) == {
        "status": "blocked",
        "reason": "pyvenv-home-missing",
    }
