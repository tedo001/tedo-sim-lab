"""The runner registry and the worker process protocol (events, exit codes, cancel)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.common.paths import CODE_ROOT
from core.experiment_engine.events import ProgressEvent, parse_line
from core.experiment_engine.runner import NoRunnerError, RunnerRegistry
from core.experiment_engine.spec import dump_spec, load_spec_text
from core.experiment_engine.worker import EXIT_CANCELLED, EXIT_FAILED, EXIT_OK

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SPEC = """\
name: worker-test
task: image_classification
data: {dataset: mnist}
model: {model: simple_cnn}
training: {kind: torch, epochs: 2}
runtime: {device: cpu}
"""


def make_run_dir(tmp_path: Path, spec_text: str = SPEC) -> Path:
    run_dir = tmp_path / "experiments" / "exp1" / "runs" / "run42"
    dump_spec(load_spec_text(spec_text), run_dir / "experiment.yaml")
    return run_dir


def worker(run_dir: Path, *args: str, stdin: str | None = "") -> subprocess.Popen[str]:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(FIXTURES), str(CODE_ROOT)]),
               TEDO_LAB_WORKSPACE=str(run_dir.parents[3]))  # tmp_path: no datasets there
    return subprocess.Popen([sys.executable, "-m", "core.experiment_engine.worker", str(run_dir),
                             *args], cwd=CODE_ROOT, env=env, text=True, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def events_of(stdout: str) -> list[ProgressEvent]:
    return [ProgressEvent.from_json(line) for line in stdout.splitlines() if line.strip()]


def test_a_run_streams_clean_events(tmp_path: Path) -> None:
    process = worker(make_run_dir(tmp_path), "--runner", "tedo_test_runners:CountingRunner")
    stdout, stderr = process.communicate(timeout=60)
    assert process.returncode == EXIT_OK, stderr
    events = events_of(stdout)  # every stdout line parses: prints went to stderr
    kinds = [event.type for event in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.count("epoch_end") == 2 and kinds.count("step") == 6
    assert {event.run_id for event in events} == {"run42"}
    assert events[-1].payload["status"] == "completed"
    assert events[0].payload["device"] == "cpu"
    assert "noise a library printed" in stderr


def test_failures_exit_1_with_a_masked_error(tmp_path: Path) -> None:
    process = worker(make_run_dir(tmp_path), "--runner", "tedo_test_runners:FailingRunner")
    stdout, _ = process.communicate(timeout=60)
    assert process.returncode == EXIT_FAILED
    end = events_of(stdout)[-1]
    assert end.payload["status"] == "failed"
    assert "RuntimeError" in end.payload["error"] and "hf_abc" not in end.payload["error"]


def test_cancel_on_stdin_exits_2(tmp_path: Path) -> None:
    process = worker(make_run_dir(tmp_path), "--runner", "tedo_test_runners:WaitingRunner")
    assert process.stdout.readline()  # run_start: it is running
    started = time.monotonic()
    process.stdin.write("cancel\n")
    process.stdin.flush()
    stdout, _ = process.communicate(timeout=30)
    assert process.returncode == EXIT_CANCELLED
    assert time.monotonic() - started < 10
    assert events_of(stdout)[-1].payload["status"] == "cancelled"


def test_closed_stdin_cancels_only_when_asked(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    process = worker(run_dir, "--runner", "tedo_test_runners:WaitingRunner", "--cancel-on-eof")
    process.stdin.close()
    assert process.wait(timeout=30) == EXIT_CANCELLED
    process = worker(run_dir, "--runner", "tedo_test_runners:CountingRunner")
    process.stdin.close()
    assert process.wait(timeout=60) == EXIT_OK


def test_task_with_no_runner_fails_clearly(tmp_path: Path) -> None:
    process = worker(make_run_dir(tmp_path, SPEC.replace("image_classification", "text_classification")))
    stdout, _ = process.communicate(timeout=60)
    assert process.returncode == EXIT_FAILED
    assert "no runner handles 'text_classification'" in events_of(stdout)[-1].payload["error"]


def test_missing_dataset_is_explained_before_training(tmp_path: Path) -> None:
    process = worker(make_run_dir(tmp_path))
    stdout, _ = process.communicate(timeout=120)
    assert process.returncode == EXIT_FAILED
    assert "MNIST is not downloaded yet" in events_of(stdout)[-1].payload["error"]


def test_experimental_tasks_explain_when_they_arrive(tmp_path: Path) -> None:
    spec = SPEC.replace("image_classification", "object_detection")
    process = worker(make_run_dir(tmp_path, spec))
    stdout, _ = process.communicate(timeout=60)
    assert process.returncode == EXIT_FAILED
    assert "v0.5" in events_of(stdout)[-1].payload["error"]


def test_bad_spec_and_unavailable_device_fail_cleanly(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    (run_dir / "experiment.yaml").write_text("name: [broken\n")
    process = worker(run_dir, "--runner", "tedo_test_runners:CountingRunner")
    stdout, _ = process.communicate(timeout=60)
    assert process.returncode == EXIT_FAILED and "SpecError" in events_of(stdout)[-1].payload["error"]


def test_parse_line_tolerates_noise() -> None:
    assert parse_line('{"type": "step", "run_id": "r", "payload": {"step": 1}}').type == "step"
    noise = parse_line("Downloading weights 45%", "r")
    assert noise.type == "log" and noise.payload["line"] == "Downloading weights 45%"
    assert parse_line('{"not": "an event"}').type == "log"
    assert "hf_xyz" not in parse_line("token hf_xyzabcdefghijklmnopqrstuvw1234").payload["line"]


def test_registry_loads_entries_and_prefers_stable_runners() -> None:
    registry = RunnerRegistry()
    assert registry.load_config(CODE_ROOT / "configs" / "runners.yaml") == []
    assert {runner.id for runner in registry.all()} >= {"object_detection", "segmentation", "ocr"}
    assert registry.task_maturity()["object_detection"] == "experimental"
    assert registry.for_spec(load_spec_text(SPEC)).id == "torch_classification"
    with pytest.raises(NoRunnerError):
        registry.for_spec(load_spec_text(SPEC.replace("image_classification", "text_classification")))
    problems = registry.load_entries(["labs.nowhere:Runner", "core.common.paths:AppPaths"])
    assert len(problems) == 2 and "not an ExperimentRunner" in problems[1]
