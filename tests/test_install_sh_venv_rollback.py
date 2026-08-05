"""Contract + behaviour test: install.sh must not destroy a working venv before
python-deps has proven it can rebuild one.

Background (2026-08-04 incident)
--------------------------------
The desktop bootstrap runs install stages as SEPARATE processes: `venv` first,
then `python-deps`. ``setup_venv()`` unconditionally did ``rm -rf venv`` on an
existing environment, so the old venv was already gone by the time
``install_deps()`` ran.

On 2026-08-04 a ~2 minute PyPI network blip made every python-deps tier fail::

    error: Request failed after 3 retries in 140.2s
      Caused by: Failed to fetch: `https://pypi.org/simple/fire/`
      Caused by: tcp connect error
      Caused by: Bad file descriptor (os error 9)

The user was left with an empty venv (3 entries in site-packages) and a Hermes
Desktop that refused to start ("Hermes bootstrap failed at stage 'python-deps'"),
with a misleading remediation hint about build-essential. The install itself was
fine minutes later — the only lasting damage was the destroyed venv.

The fix: ``setup_venv()`` moves the old venv to ``venv.prev`` instead of deleting
it, ``install_deps()`` arms an EXIT trap that rolls it back on any non-zero exit,
and discards the snapshot only once dependencies are proven installed. A
transient outage now degrades to "no upgrade" instead of "app is bricked".
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


# --------------------------------------------------------------------------
# Contract: the destructive pattern must not come back.
# --------------------------------------------------------------------------

def test_setup_venv_never_hard_deletes_an_existing_venv() -> None:
    text = INSTALL_SH.read_text()

    # The "already exists, recreating" branches must route through the snapshot
    # helper, never straight to rm -rf.
    recreate_blocks = re.findall(
        r'log_info "Virtual environment already exists, recreating\.\.\."\s*\n\s*(\S+[^\n]*)',
        text,
    )
    assert recreate_blocks, "expected the 'already exists, recreating' branches to still exist"
    for follow_up in recreate_blocks:
        assert follow_up.strip().startswith("venv_snapshot_take"), (
            "setup_venv must move the existing venv aside via venv_snapshot_take, "
            f"not destroy it — found: {follow_up!r}. A transient python-deps "
            "failure would otherwise leave the user with no venv at all."
        )


def test_install_deps_arms_and_disarms_the_rollback_trap() -> None:
    text = INSTALL_SH.read_text()

    install_deps = text.split("install_deps() {", 1)[1].split("\nsetup_path() {", 1)[0]

    assert "trap venv_snapshot_restore_on_failure EXIT" in install_deps, (
        "install_deps must arm the rollback trap; it exits with an explicit "
        "`exit 1` on total install failure, so a plain `if install_deps` at the "
        "stage dispatch site cannot catch it"
    )
    assert "venv_snapshot_discard" in install_deps, (
        "install_deps must discard the snapshot once deps are proven installed"
    )
    assert "trap - EXIT" in install_deps, (
        "install_deps must disarm the trap after success so an unrelated later "
        "exit cannot resurrect a stale venv.prev"
    )


# --------------------------------------------------------------------------
# Behaviour: the rollback actually restores the old venv.
# --------------------------------------------------------------------------

def _extract_snapshot_helpers() -> str:
    """Pull the venv snapshot helper block out of install.sh.

    Sourcing install.sh directly is not an option: it dispatches at the bottom.
    """
    text = INSTALL_SH.read_text()
    start = text.index("VENV_SNAPSHOT_DIR=")
    end = text.index("setup_venv() {", start)
    return text[start:end]


_HARNESS = """
set -u
log_info()    {{ echo "INFO: $*"; }}
log_warn()    {{ echo "WARN: $*"; }}
log_error()   {{ echo "ERROR: $*"; }}
log_success() {{ echo "OK: $*"; }}

INSTALL_DIR="{install_dir}"

{helpers}

# --- simulate the `venv` stage: an existing, working venv is moved aside ---
venv_snapshot_take

# uv then creates a fresh (still empty) venv in its place
mkdir -p "$INSTALL_DIR/venv"

# --- simulate the `python-deps` stage failing, as it did on 2026-08-04 ---
(
    trap venv_snapshot_restore_on_failure EXIT
    echo "INFO: pretending uv failed to reach PyPI"
    exit 1
)
"""


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_failed_python_deps_restores_the_previous_venv(tmp_path: Path) -> None:
    install_dir = tmp_path / "hermes-agent"
    site_packages = install_dir / "venv" / "lib" / "python3.11" / "site-packages"
    site_packages.mkdir(parents=True)
    # A marker proving THIS venv (with its deps) is the one that comes back.
    (site_packages / "openai").mkdir()
    (install_dir / "venv" / "bin").mkdir(parents=True)
    (install_dir / "venv" / "bin" / "hermes").write_text("#!/bin/sh\n")

    script = _HARNESS.format(
        install_dir=install_dir,
        helpers=_extract_snapshot_helpers(),
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60
    )

    assert (site_packages / "openai").is_dir(), (
        "the previously installed dependencies must be restored after a failed "
        f"python-deps stage.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (install_dir / "venv" / "bin" / "hermes").is_file(), (
        "the restored venv must still carry its entry points"
    )
    assert not (install_dir / "venv.prev").exists(), (
        "venv.prev must be consumed by the restore, not left behind"
    )
    assert "restoring the previous virtual environment" in result.stdout, (
        f"the rollback must tell the user what happened.\nstdout:\n{result.stdout}"
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_successful_python_deps_discards_the_snapshot(tmp_path: Path) -> None:
    install_dir = tmp_path / "hermes-agent"
    (install_dir / "venv").mkdir(parents=True)
    (install_dir / "venv" / "old-marker").write_text("stale\n")

    script = _HARNESS.format(
        install_dir=install_dir,
        helpers=_extract_snapshot_helpers(),
    ).replace(
        '    echo "INFO: pretending uv failed to reach PyPI"\n    exit 1',
        "    venv_snapshot_discard\n    trap - EXIT",
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60
    )

    assert not (install_dir / "venv.prev").exists(), (
        "a successful install must discard venv.prev, otherwise the stale tree "
        f"accumulates on every upgrade.\nstdout:\n{result.stdout}"
    )
    assert not (install_dir / "venv" / "old-marker").exists(), (
        "the freshly built venv must survive — the old one must NOT be restored "
        "over it after a successful install"
    )
