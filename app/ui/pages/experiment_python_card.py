"""Settings → Experiment Python: which Python runs experiments, what it has (checked in a
subprocess, so CUDA never starts in the app), choose another one, or let the lab make one."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QFileDialog, QHBoxLayout, QPushButton, QWidget

from core.common import experiment_python
from core.common.experiment_env import frozen

from ...services.context import AppContext
from ...services.experiment_setup import SetupPlan
from ..icons import icon
from ..widgets import Card, KeyValues, Pill, label

__all__ = ["ExperimentPythonCard"]


class ExperimentPythonCard(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Experiment Python", "runs training, evaluation, benchmarks and plugin installs",
                         parent=parent)
        self.ctx = ctx
        self._check_job: str | None = None
        self._make_job: str | None = None
        self.state = Pill("Checking…", "planned")
        self.add_head_widget(self.state)
        self.facts = KeyValues([("Python", "—"), ("Version", "—"), ("PyTorch", "—"), ("CUDA", "—"),
                                ("Missing", "—")])
        self.add(self.facts)
        row = QHBoxLayout()
        self.check_button = QPushButton(icon("refresh-cw"), "Check")
        self.check_button.clicked.connect(self.check)
        self.choose_button = QPushButton(icon("folder-open"), "Choose a Python…")
        self.choose_button.clicked.connect(self._choose)
        self.own_button = QPushButton("Use this app's Python")
        self.own_button.clicked.connect(lambda: self.use(""))
        self.own_button.setVisible(not frozen())
        for button in (self.check_button, self.choose_button, self.own_button):
            row.addWidget(button)
        row.addStretch(1)
        self.add(row)

        self.plan = SetupPlan()
        self.add(label("Make a new environment for the lab: PyTorch (the CUDA build when an NVIDIA GPU "
                       "is present) and the lab's libraries, a few GB.", "Body", wrap=True))
        self.plan_text = label("", "Mono", wrap=True)
        self.add(self.plan_text)
        row = QHBoxLayout()
        self.cpu_only = QCheckBox("CPU-only PyTorch")
        self.cpu_only.toggled.connect(self._replan)
        self.make_button = QPushButton(icon("package-plus"), "Make the environment")
        self.make_button.setObjectName("Primary")
        self.make_button.clicked.connect(self.make)
        self.cancel_button = QPushButton(icon("x"), "Cancel")
        self.cancel_button.clicked.connect(lambda: self._make_job and ctx.jobs.cancel(self._make_job))
        for widget in (self.cpu_only, self.make_button, self.cancel_button):
            row.addWidget(widget)
        row.addStretch(1)
        self.add(row)
        self.progress = label("", "Mono", wrap=True)
        self.add(self.progress)

        ctx.setup.checked.connect(self._checked)
        ctx.setup.made.connect(self._made)
        ctx.jobs.job_progress.connect(self._progress)
        self._replan()
        self.check()

    # Checking -----------------------------------------------------------------------------------
    def check(self) -> None:
        python = experiment_python(self.ctx.config)
        self.facts.set_value("Python", python or "none yet")
        if not python:
            self._show_state("Not set up", "warn")
            self._update()
            return
        self._show_state("Checking…", "planned")
        self._check_job = self.ctx.setup.check(python)
        self._update()

    def _checked(self, job_id: str, facts: object) -> None:
        if job_id != self._check_job:
            return
        self._check_job = None
        self.facts.set_value("Version", getattr(facts, "version", "") or "—")
        packages = getattr(facts, "packages", {})
        self.facts.set_value("PyTorch", packages.get("torch") or "not installed")
        self.facts.set_value("CUDA", "available" if getattr(facts, "cuda", False) else "not available")
        missing = facts.missing() if hasattr(facts, "missing") and not facts.error else []
        self.facts.set_value("Missing", ", ".join(missing) or "nothing")
        if facts.error:
            self._show_state("Not usable", "fail")
            self.progress.setText(facts.error)
        else:
            self._show_state("Ready" if not missing else "Incomplete", "ok" if not missing else "warn")
        self._update()

    def _show_state(self, words: str, tone: str) -> None:
        self.state.setText(words)
        self.state.set_tone(tone)

    # Choosing ------------------------------------------------------------------------------------
    def _choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Python for experiments", "",
                                              "Python (python.exe python*);;All files (*)")
        if path:
            self.use(path)

    def use(self, python: str) -> None:
        self.ctx.set_experiment_python(python)
        self.check()

    # Making ---------------------------------------------------------------------------------------
    def _replan(self) -> None:
        self.plan = SetupPlan(cpu_only=self.cpu_only.isChecked())
        if self.plan.base is None:
            self.plan_text.setText("No Python 3.11 or newer was found on this computer. Install it from "
                                   "https://www.python.org/downloads/ (tick “Add python.exe to PATH”), "
                                   "then reopen this page.")
        else:
            build = {"cuda": "CUDA", "cpu": "CPU"}.get(self.plan.build, "standard")
            self.plan_text.setText(f"From {' '.join(self.plan.base)} into {self.plan.env_dir} · "
                                   f"PyTorch {build} build")
        self._update()

    def make(self) -> str | None:
        if self._make_job or not self.plan.steps:
            return None
        self._make_job = self.ctx.setup.make(self.plan)
        self.progress.setText("Starting…")
        self._update()
        return self._make_job

    def _progress(self, job_id: str, fraction: float, message: str) -> None:
        if job_id == self._make_job:
            self.progress.setText((f"{fraction:.0%} · " if fraction >= 0 else "") + message)

    def _made(self, job_id: str, python: str, error: str) -> None:
        if job_id != self._make_job:
            return
        self._make_job = None
        if python:
            self.progress.setText("The environment is ready and is now the experiment Python.")
            self.use(python)
        else:
            self.progress.setText(f"Not finished: {error}")
            self._update()

    def _update(self) -> None:
        busy = self._make_job is not None
        self.make_button.setEnabled(bool(self.plan.steps) and not busy)
        self.cancel_button.setEnabled(busy)
        self.cpu_only.setEnabled(not busy and self.plan.build != "default")
        self.check_button.setEnabled(self._check_job is None and not busy)
        self.choose_button.setEnabled(not busy)
