"""A retained run scope cannot hold a CWD lock: CWD belongs to the task, not the process."""

import concurrent.futures
import os
import threading

import pytest

from cron.scheduler import _CronRunScope
from tools.terminal_tool import get_session_cwd


@pytest.mark.parametrize("same_directory", [False, True])
def test_retained_scope_does_not_block_other_jobs(tmp_path, monkeypatch, same_directory):
    first = tmp_path / "first"
    second = first if same_directory else tmp_path / "second"
    first.mkdir()
    second.mkdir(exist_ok=True)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    holding = threading.Event()
    release = threading.Event()
    task_ids = []

    def run(name, directory, hold=False):
        scope = _CronRunScope({"workdir": str(directory)}, name, name)
        scope.enter()
        try:
            task_ids.append(scope.task_id)
            if hold:
                holding.set()
                assert release.wait(10)
            assert get_session_cwd(scope.task_id) == str(directory)
            assert os.environ["TERMINAL_CWD"] == str(tmp_path)
            return scope.task_id
        finally:
            scope.exit()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        holder = pool.submit(run, "retained", first, True)
        try:
            assert holding.wait(5)
            # This completes while the original scope remains held, well below 60 seconds.
            pool.submit(run, "next", second).result(timeout=5)
            assert not holder.done()
        finally:
            release.set()
        holder.result(timeout=5)
    assert len(set(task_ids)) == 2
    assert all(get_session_cwd(task_id) is None for task_id in task_ids)
