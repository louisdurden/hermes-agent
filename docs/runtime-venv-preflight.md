# Runtime venv preflight for uv Python rotations

## Failure mode

A uv-managed CPython patch rotation can remove the directory recorded as
`home` in `venv/pyvenv.cfg` before the long-lived Hermes venv is refreshed.
The venv launcher then fails before Hermes can import its CLI, commonly with
`ModuleNotFoundError: encodings`. That means `hermes update`, Kanban dispatch,
and cron monitors cannot repair themselves from inside the broken venv.

## Preflight contract

`hermes_cli/runtime_venv_preflight.py` is a standard-library-only executable
intended for an external monitor that already has an independent Python. Run it
before invoking any command from the Hermes venv:

```sh
/usr/bin/python3 hermes_cli/runtime_venv_preflight.py \
  --venv /path/to/hermes-agent/venv \
  --uv /absolute/path/to/uv \
  --python 3.11
```

The command exits `0` only when the venv is healthy or a repair completed and
passed an `encodings` import probe. It emits one compact JSON object:

- `healthy`: `pyvenv.cfg` has an existing `home` and the venv probe succeeds;
  uv is not called.
- `repaired`: the previous `home` was absent, uv resolved the current managed
  Python 3.11, and `uv venv --allow-existing` refreshed the existing venv.
- `blocked`: the preflight could not prove the narrow repair was safe or could
  not verify the result. The caller must not start Hermes.

The repair never removes or recreates the venv directory. `--allow-existing`
retains its site-packages. Before mutation, the preflight copies `pyvenv.cfg`
to `pyvenv.cfg.runtime-preflight.bak`. A non-blocking file lock serializes
concurrent monitors, and a second healthy run is a no-op.

## Monitor integration

The cron scheduler performs the preflight in-process before spawning any
Python monitor or collection script. It first inspects `sys.prefix/pyvenv.cfg`;
a healthy `home` is a zero-subprocess fast path, while a missing `home` invokes
the managed uv repair. A blocked repair prevents the monitor child from being
spawned, so a collection failure cannot masquerade as a content change.

Standalone watchdogs should call the preflight with absolute paths and inspect
its exit code before every Kanban dispatch or monitor read. Do not import
Hermes from that external watchdog: the point is to remain executable while
the Hermes venv itself cannot start. Do not add provider credentials or
provider update logic to this path.

## Rollback

To roll back the monitor integration, stop invoking the preflight and revert
the commit that added this module. If the in-place metadata refresh itself must
be rolled back, first stop processes using the venv, then restore
`venv/pyvenv.cfg.runtime-preflight.bak` as `venv/pyvenv.cfg`. That backup may
refer to the removed uv generation and therefore is diagnostic by default; do
not restart Hermes from it unless its `home` exists and the venv probe passes.
The site-packages directory is not part of this rollback because the preflight
does not delete or replace it.
