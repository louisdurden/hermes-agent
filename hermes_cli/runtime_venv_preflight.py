"""Fail-closed preflight for a venv backed by uv-managed Python.

This module deliberately uses only the Python standard library so an external
watchdog can execute it with the system Python before starting Hermes.  It
repairs only the narrow failure where ``pyvenv.cfg`` points at a base ``home``
that no longer exists after uv rotates a patch release.  Other failures are
reported without mutating the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


def _read_pyvenv_cfg(venv: Path) -> dict[str, str]:
    cfg_path = venv / "pyvenv.cfg"
    try:
        lines = cfg_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip().lower()] = value.strip()
    return values


def venv_home_missing(venv: Path) -> bool:
    """Return whether an existing venv declares a base ``home`` that is gone.

    A malformed config raises ``ValueError`` so callers can fail closed rather
    than treating an unreadable runtime as healthy.
    """
    cfg = _read_pyvenv_cfg(Path(venv))
    home_text = cfg.get("home", "").strip()
    if not home_text:
        raise ValueError("pyvenv-home-missing")
    return not Path(home_text).is_dir()


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "PYTHONHOME",
        "PYTHONPATH",
        "UV_PROJECT_ENVIRONMENT",
        "VIRTUAL_ENV",
    ):
        env.pop(key, None)
    env["UV_MANAGED_PYTHON"] = "1"
    env["UV_NO_CONFIG"] = "1"
    return env


def _probe_python(python: Path, *, timeout: float) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import encodings, json, sys; "
                "print(json.dumps({'version': sys.version.split()[0]}))",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_clean_env(),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    if result.returncode != 0:
        return False, ""
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError):
        return False, ""
    return True, str(payload.get("version") or "")


@contextmanager
def _repair_lock(venv: Path) -> Iterator[bool]:
    """Serialize repairs without leaving a stale PID lock behind."""
    lock_path = venv.parent / f".{venv.name}.runtime-preflight.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
    except OSError:
        yield False
        return
    try:
        if os.name == "nt":
            # The incident and this repair path are POSIX-specific. Refuse to
            # mutate on Windows until an equivalent non-blocking lock exists.
            yield False
            return
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True
    finally:
        handle.close()


def run_preflight(
    *,
    venv: Path,
    uv: Path,
    python_spec: str = "3.11",
    timeout: float = 30.0,
) -> dict[str, object]:
    """Verify or narrowly repair *venv* after a uv Python patch rotation.

    A healthy existing ``home`` is never touched, even if the interpreter probe
    fails for some unrelated reason.  Repair uses ``uv venv --allow-existing``;
    this updates the interpreter metadata and launchers in place while retaining
    the existing site-packages directory.
    """
    venv = Path(venv)
    uv = Path(uv)
    cfg = _read_pyvenv_cfg(venv)
    home_text = cfg.get("home", "")
    if not home_text:
        return {"status": "blocked", "reason": "pyvenv-home-missing"}

    home = Path(home_text).expanduser()
    python = _venv_python(venv)
    if home.is_dir():
        healthy, version = _probe_python(python, timeout=timeout)
        if healthy:
            return {"status": "healthy", "python_version": version}
        return {"status": "blocked", "reason": "venv-probe-failed"}

    if not uv.is_file() or not os.access(uv, os.X_OK):
        return {"status": "blocked", "reason": "uv-unavailable"}

    with _repair_lock(venv) as acquired:
        if not acquired:
            return {"status": "blocked", "reason": "repair-lock-unavailable"}

        # Another watchdog may have repaired the venv before this process won
        # the lock. Re-read and exit without invoking uv when that happened.
        refreshed_home = Path(_read_pyvenv_cfg(venv).get("home", "")).expanduser()
        if refreshed_home.is_dir():
            healthy, version = _probe_python(python, timeout=timeout)
            if healthy:
                return {"status": "healthy", "python_version": version}
            return {"status": "blocked", "reason": "venv-probe-failed"}

        env = _clean_env()
        try:
            found = subprocess.run(
                [str(uv), "python", "find", "--managed-python", python_spec],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {"status": "blocked", "reason": "uv-python-find-failed"}
        if found.returncode != 0:
            return {
                "status": "blocked",
                "reason": "uv-python-find-failed",
                "returncode": found.returncode,
            }

        current_python = Path(found.stdout.strip()).expanduser()
        if not current_python.is_file():
            return {"status": "blocked", "reason": "uv-python-not-found"}
        current_ok, current_version = _probe_python(current_python, timeout=timeout)
        if not current_ok or not current_version.startswith(f"{python_spec}."):
            return {"status": "blocked", "reason": "uv-python-version-mismatch"}

        cfg_path = venv / "pyvenv.cfg"
        backup_path = venv / "pyvenv.cfg.runtime-preflight.bak"
        try:
            shutil.copy2(cfg_path, backup_path)
        except OSError:
            return {"status": "blocked", "reason": "pyvenv-backup-failed"}

        try:
            repair = subprocess.run(
                [
                    str(uv),
                    "venv",
                    "--python",
                    str(current_python),
                    "--allow-existing",
                    str(venv),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {"status": "blocked", "reason": "uv-venv-repair-failed"}
        if repair.returncode != 0:
            return {
                "status": "blocked",
                "reason": "uv-venv-repair-failed",
                "returncode": repair.returncode,
            }

        repaired_home_text = _read_pyvenv_cfg(venv).get("home", "")
        repaired_home = Path(repaired_home_text).expanduser()
        healthy, version = _probe_python(python, timeout=timeout)
        if not repaired_home.is_dir() or not healthy:
            return {"status": "blocked", "reason": "post-repair-probe-failed"}
        return {"status": "repaired", "python_version": version}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Repair a Hermes venv whose uv-managed Python home rotated away."
    )
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--python", default="3.11", dest="python_spec")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    result = run_preflight(
        venv=args.venv,
        uv=args.uv,
        python_spec=args.python_spec,
        timeout=args.timeout,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] in {"healthy", "repaired"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
