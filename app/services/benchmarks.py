"""Benchmarks: inference latency and throughput by batch size, run in the worker process
(``worker <folder> --benchmark``) so the app never touches the GPU itself. Each benchmark is a
folder under ``<workspace>/results/benchmarks/`` holding its request (``benchmark.json``) and,
once done, its results (``results.json``) or what went wrong (``error.txt``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from core.common.paths import AppPaths

from .jobs import JobQueue
from .runs import slug

__all__ = ["Benchmark", "BenchmarkService"]

REQUEST, RESULTS, ERROR = "benchmark.json", "results.json", "error.txt"


@dataclass(frozen=True)
class Benchmark:
    id: str
    folder: Path
    request: dict[str, Any]
    results: dict[str, Any] | None
    #: "queued", "running", "done", "failed" or "cancelled".
    status: str
    error: str = ""

    @property
    def title(self) -> str:
        return str(self.request.get("title") or self.id)

    @property
    def best_throughput(self) -> float | None:
        rows = (self.results or {}).get("rows") or []
        return max((row["throughput"] for row in rows), default=None)

    def row(self, batch: int) -> dict[str, float] | None:
        return next((row for row in (self.results or {}).get("rows", []) if row["batch"] == batch), None)


class BenchmarkService(QObject):
    changed = Signal()
    finished = Signal(str, str)  # benchmark id, status

    def __init__(self, paths: AppPaths, jobs: JobQueue, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.paths, self.jobs = paths, jobs
        self.root = paths.results / "benchmarks"
        self._jobs: dict[str, str] = {}  # job id → benchmark id
        jobs.job_started.connect(self._started)
        jobs.job_finished.connect(self._finished)

    def start(self, request: dict[str, Any], title: str) -> str:
        """Queue a benchmark; ``request`` is what :func:`labs.benchmark.latency.run_benchmark` takes
        (paths relative to the workspace). Returns the benchmark id."""
        bench_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug(title, 30)}"
        folder = self.root / bench_id
        number = 2
        while folder.exists():
            folder, number = self.root / f"{bench_id}-{number}", number + 1
        folder.mkdir(parents=True)
        (folder / REQUEST).write_text(json.dumps({**request, "title": title}, indent=2), encoding="utf-8")
        job_id = self.jobs.submit_run(folder, title=f"Benchmark {title}", args=("--benchmark",))
        self._jobs[job_id] = folder.name
        self.changed.emit()
        return folder.name

    def cancel(self, bench_id: str) -> None:
        for job_id, known in self._jobs.items():
            if known == bench_id:
                self.jobs.cancel(job_id)

    def _started(self, job_id: str) -> None:
        if job_id in self._jobs:
            self.changed.emit()

    def _finished(self, job_id: str, status: str) -> None:
        bench_id = self._jobs.pop(job_id, None)
        if bench_id is None:
            return
        if status != "completed":
            error = self.jobs.job(job_id).error or status
            (self.root / bench_id / ERROR).write_text(error, encoding="utf-8")
        self.changed.emit()
        self.finished.emit(bench_id, status)

    # Reading ------------------------------------------------------------------------
    def _status(self, folder: Path) -> tuple[str, str]:
        if (folder / RESULTS).is_file():
            return "done", ""
        if (folder / ERROR).is_file():
            error = (folder / ERROR).read_text(encoding="utf-8")
            return ("cancelled" if error == "cancelled" else "failed"), error
        for job_id, bench_id in self._jobs.items():
            if bench_id == folder.name:
                return ("running" if self.jobs.job(job_id).status == "running" else "queued"), ""
        return "failed", "stopped before it finished (the app was closed)"

    def get(self, bench_id: str) -> Benchmark:
        folder = self.root / bench_id
        request = json.loads((folder / REQUEST).read_text(encoding="utf-8"))
        done = folder / RESULTS
        results = json.loads(done.read_text(encoding="utf-8")) if done.is_file() else None
        status, error = self._status(folder)
        return Benchmark(bench_id, folder, request, results, status, error)

    def all(self) -> list[Benchmark]:
        if not self.root.is_dir():
            return []
        folders = sorted((f for f in self.root.iterdir() if (f / REQUEST).is_file()), reverse=True)
        return [self.get(folder.name) for folder in folders]

    def for_run(self, run_id: str) -> list[Benchmark]:
        """Finished benchmarks of a run's checkpoint, newest first."""
        return [b for b in self.all() if b.results and b.results.get("run_id") == run_id]
