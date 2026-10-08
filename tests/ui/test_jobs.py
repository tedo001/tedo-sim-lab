"""The Qt job queue: worker processes and thread-pool tasks, with progress and cancel."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from app.services.jobs import JobQueue, worker_command
from core.common.cancel import Cancelled
from core.common.paths import CODE_ROOT
from core.experiment_engine.spec import dump_spec, load_spec_text

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SPEC = """\
name: queue-test
task: image_classification
data: {dataset: mnist}
model: {model: simple_cnn}
training: {kind: torch, epochs: 2}
runtime: {device: cpu}
"""


def run_dir(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / name
    dump_spec(load_spec_text(SPEC), folder / "experiment.yaml")
    return folder


def queue_for(runner: str, store=None, **kwargs) -> JobQueue:
    import sys

    def factory(folder: Path):
        command = worker_command(folder, python=sys.executable, code_root=CODE_ROOT,
                                 extra=["--runner", f"tedo_test_runners:{runner}"])
        command.env["PYTHONPATH"] = os.pathsep.join([str(FIXTURES), command.env["PYTHONPATH"]])
        return command

    return JobQueue(factory, store=store, **kwargs)


def test_run_streams_events_and_completes(qtbot, tmp_path: Path) -> None:
    queue = queue_for("CountingRunner")
    events = []
    queue.job_event.connect(lambda job_id, event: events.append(event))
    with qtbot.waitSignal(queue.job_finished, timeout=60_000) as finished:
        job_id = queue.submit_run(run_dir(tmp_path, "r1"))
    assert finished.args == [job_id, "completed"]
    # stdout and stderr are read separately, so only stdout's order is guaranteed
    kinds = [event.type for event in events if event.payload.get("stream") != "stderr"]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end" and "epoch_end" in kinds
    assert any(e.type == "log" and e.payload.get("stream") == "stderr" for e in events)
    assert queue.job(job_id).status == "completed"


def test_failed_run_keeps_its_error(qtbot, tmp_path: Path) -> None:
    queue = queue_for("FailingRunner")
    with qtbot.waitSignal(queue.job_finished, timeout=60_000) as finished:
        job_id = queue.submit_run(run_dir(tmp_path, "r1"))
    assert finished.args[1] == "failed"
    assert "RuntimeError" in queue.job(job_id).error and "hf_abc" not in queue.job(job_id).error


def test_cancel_a_running_worker(qtbot, tmp_path: Path) -> None:
    queue = queue_for("WaitingRunner")
    job_id = queue.submit_run(run_dir(tmp_path, "r1"))
    qtbot.waitSignal(queue.job_event, timeout=30_000).wait()  # it has started
    started = time.monotonic()
    with qtbot.waitSignal(queue.job_finished, timeout=30_000) as finished:
        queue.cancel(job_id)
    assert finished.args == [job_id, "cancelled"]
    assert time.monotonic() - started < 10


def test_unresponsive_worker_is_killed_after_the_grace_period(qtbot, tmp_path: Path) -> None:
    queue = queue_for("WaitingRunner", cancel_grace_ms=200)
    job_id = queue.submit_run(run_dir(tmp_path, "r1"))
    qtbot.waitSignal(queue.job_event, timeout=30_000).wait()
    process = queue._processes[job_id]
    with qtbot.waitSignal(queue.job_finished, timeout=30_000) as finished:
        queue.cancel(job_id)
        process.closeWriteChannel()  # pretend the worker never reads "cancel"
        process.write(b"")
    assert finished.args[1] == "cancelled"


def test_runs_queue_one_at_a_time_and_queued_ones_cancel(qtbot, tmp_path: Path) -> None:
    queue = queue_for("WaitingRunner", max_concurrent_runs=1)
    first = queue.submit_run(run_dir(tmp_path, "a"))
    second = queue.submit_run(run_dir(tmp_path, "b"))
    assert queue.job(first).status == "running" and queue.job(second).status == "queued"
    with qtbot.waitSignal(queue.job_finished, timeout=5_000) as finished:
        queue.cancel(second)
    assert finished.args == [second, "cancelled"]
    with qtbot.waitSignal(queue.job_finished, timeout=30_000):
        queue.cancel(first)
    assert not queue.active()


def test_a_missing_interpreter_fails_the_job(qtbot, tmp_path: Path) -> None:
    queue = JobQueue(lambda folder: worker_command(folder, python=str(tmp_path / "no-python"),
                                                   code_root=CODE_ROOT))
    with qtbot.waitSignal(queue.job_finished, timeout=30_000) as finished:
        job_id = queue.submit_run(run_dir(tmp_path, "r1"))
    assert finished.args[1] == "failed" and "could not start" in queue.job(job_id).error


def test_tasks_report_progress_and_results(qtbot) -> None:
    queue = queue_for("CountingRunner")
    progress = []
    queue.job_progress.connect(lambda job_id, fraction, message: progress.append((fraction, message)))

    def work(token, report):
        for index in range(3):
            report((index + 1) / 3, f"step {index + 1}")
        return 42

    with qtbot.waitSignal(queue.job_finished, timeout=10_000) as finished:
        job_id = queue.submit_task(work, title="answer")
    assert finished.args == [job_id, "completed"]
    assert queue.job(job_id).result == 42
    qtbot.waitUntil(lambda: len(progress) == 3, timeout=5_000)
    assert progress[-1] == (1.0, "step 3")


def test_task_failure_and_cancel(qtbot) -> None:
    queue = queue_for("CountingRunner")

    def boom(token, report):
        raise ValueError("bad input with password=hunter22")

    with qtbot.waitSignal(queue.job_finished, timeout=10_000) as finished:
        job_id = queue.submit_task(boom, title="boom")
    assert finished.args[1] == "failed" and "hunter22" not in queue.job(job_id).error

    def slow(token, report):
        while not token.wait(0.01):
            pass
        raise Cancelled()

    job_id = queue.submit_task(slow, title="slow")
    with qtbot.waitSignal(queue.job_finished, timeout=10_000) as finished:
        queue.cancel(job_id)
    assert finished.args == [job_id, "cancelled"]


def test_jobs_are_recorded_in_the_database(qtbot, tmp_path: Path, workspace) -> None:
    from core.tracking import LabStore
    store = LabStore.open(workspace)
    queue = queue_for("CountingRunner", store=store)
    with qtbot.waitSignal(queue.job_finished, timeout=60_000):
        job_id = queue.submit_run(run_dir(tmp_path, "r1"), title="recorded")
    row = store.jobs()[0]
    assert (row["id"], row["title"], row["status"]) == (job_id, "recorded", "completed")
    assert row["started_at"] and row["ended_at"]
    store.close()


@pytest.mark.parametrize("python", ["/opt/exp-env/bin/python", r"C:\envs\lab\python.exe"])
def test_worker_command_targets_any_interpreter(python: str) -> None:
    command = worker_command(Path("/w/run"), python=python, code_root=CODE_ROOT)
    assert command.program == python
    assert command.arguments[:3] == ["-m", "core.experiment_engine.worker", str(Path("/w/run"))]
    assert "--cancel-on-eof" in command.arguments
    assert command.env["PYTHONPATH"].split(os.pathsep)[0] == str(CODE_ROOT)
