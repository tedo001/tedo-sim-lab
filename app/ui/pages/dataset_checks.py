"""The Dataset Hub's *Validate*: the data checks a first experiment on the chosen dataset would
make before training (duplicates, the same sample in two splits, conflicting or missing labels,
class balance, licence metadata), run in the experiment Python because reading image sets needs
its libraries. The spec checked is written to ``results/data_checks/<dataset>.yaml``."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from PySide6.QtWidgets import QWidget

from core.common.config import experiment_python
from core.experiment_engine.spec import ExperimentSpec, dump_spec

from ...services.context import AppContext
from ...services.jobs import NoExperimentPython, worker_command
from ..widgets import Card, label
from .data_checks_view import ChecksTable

__all__ = ["DatasetChecks"]


class DatasetChecks(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Data checks", "the splits a first experiment would train and test on",
                         parent=parent)
        self.ctx = ctx
        self.report: dict[str, Any] | None = None
        self._job: str | None = None
        self.state = label("", "CardCaption", wrap=True)
        self.add(self.state)
        self.checks = ChecksTable()
        self.add(self.checks)
        self.setVisible(False)
        ctx.jobs.job_finished.connect(self._done)

    def check(self, spec: ExperimentSpec) -> str | None:
        """Check ``spec``'s data in the background; returns the job id."""
        self.setVisible(True)
        self.checks.clear_rows()
        folder = self.ctx.paths.results / "data_checks"
        target = dump_spec(spec, folder / f"{spec.data.dataset}.yaml")
        try:
            command = worker_command(target, python=experiment_python(self.ctx.config),
                                     code_root=self.ctx.paths.code_root, workspace=self.ctx.paths.workspace)
        except NoExperimentPython as exc:
            self.state.setText(str(exc))
            return None

        def work(cancel: Any, progress: Any) -> dict[str, Any]:
            import os

            done = subprocess.run([command.program, "-m", "core.experiment_engine.data_checks", str(target)],
                                  cwd=command.cwd, env={**os.environ, **command.env}, capture_output=True,
                                  text=True, timeout=900, stdin=subprocess.DEVNULL)
            lines = done.stdout.strip().splitlines()
            if not lines:
                raise RuntimeError((done.stderr.strip().splitlines() or ["no answer"])[-1])
            answer = json.loads(lines[-1])
            if "error" in answer:
                raise RuntimeError(answer["error"])
            return answer

        self.state.setText(f"Checking {spec.data.dataset} as {spec.task.value.replace('_', ' ')}…")
        self._job = self.ctx.jobs.submit_task(work, title=f"Check {spec.data.dataset}")
        return self._job

    def _done(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        job = self.ctx.jobs.job(job_id)
        if status != "completed":
            self.state.setText(f"The checks could not run: {job.error}")
            return
        self.report = report = job.result
        sizes = ", ".join(f"{name} {count:,}" for name, count in report.get("splits", {}).items())
        self.state.setText(f"{report['summary']}" + (f" · {sizes}" if sizes else "")
                           + (f" · {report['note']}" if report.get("note") else ""))
        self.checks.show_checks(report["checks"])
