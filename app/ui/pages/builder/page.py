"""Experiment Builder: describe an experiment step by step, see its ``experiment.yaml``,
check it, and queue it."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from core.experiment_engine.spec import ExperimentSpec, SpecError, dump_spec, dump_spec_text, load_spec

from ....services.context import AppContext
from ....services.runs import slug
from ...icons import icon
from ...widgets import Card, Page, Pill, ResponsiveRow, label
from .form import SpecForm

__all__ = ["ExperimentBuilderPage"]


def _problems(error: ValidationError) -> list[str]:
    return [f"{'.'.join(map(str, e['loc'])) or 'spec'}: {e['msg']}" for e in error.errors()]


class ExperimentBuilderPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Experiment Builder", "task → data → model → training → device", parent)
        self.ctx = ctx
        self.spec: ExperimentSpec | None = None
        self.problems: list[str] = []
        open_button = QPushButton(icon("folder-open"), "Open experiment.yaml…")
        open_button.clicked.connect(self._ask_open)
        self.head.add_action(open_button)

        self.form = SpecForm(ctx)
        summary = Card("Check and run")
        self.state = Pill("", "planned")
        summary.add_head_widget(self.state)
        self.problem_text = label("", "Body", wrap=True)
        summary.add(self.problem_text)
        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self.run)
        self.save_button = QPushButton("Save experiment.yaml…")
        self.save_button.clicked.connect(self._ask_save)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        summary.add(buttons)
        self.result = label("", "CardCaption", wrap=True)
        summary.add(self.result)
        yaml_card = Card("experiment.yaml", "what the worker will read")
        self.yaml = QPlainTextEdit()
        self.yaml.setReadOnly(True)
        self.yaml.setObjectName("Code")
        self.yaml.setMinimumHeight(420)
        self.yaml.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        yaml_card.add(self.yaml)
        holder = QWidget()
        stack = QVBoxLayout(holder)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(12)
        stack.addWidget(summary)
        stack.addWidget(yaml_card)
        stack.addStretch(1)
        self.body.addWidget(ResponsiveRow([(self.form, 3), (holder, 2)], breakpoint=980))
        self.body.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.refresh)
        self.form.changed.connect(self._timer.start)
        ctx.experiments.draft_changed.connect(self.load)
        ctx.downloads.finished.connect(lambda *_: self._timer.start())
        if ctx.experiments.draft is not None:
            self.load(ctx.experiments.draft)
        self.refresh()

    # State ----------------------------------------------------------------------
    def load(self, spec: ExperimentSpec) -> None:
        self.form.load(spec)
        self.result.setText("")
        self.refresh()

    def refresh(self) -> None:
        try:
            self.spec = self.form.spec()
        except ValidationError as exc:
            self.spec, self.problems = None, _problems(exc)
            self.yaml.setPlainText("")
        else:
            self.yaml.setPlainText(dump_spec_text(self.spec))
            self.problems = self.ctx.experiments.problems(self.spec)
            if not self.form.model.acknowledged:
                self.problems.append("Accept the terms of the pretrained weights first")
        ready = not self.problems
        self.state.setText("Ready to run" if ready else f"{len(self.problems)} to fix")
        self.state.set_tone("ok" if ready else "warn")
        self.problem_text.setText("Everything checks out: the run goes to the queue and the Training page "
                                  "follows it." if ready else "\n".join(f"– {p}" for p in self.problems))
        self.run_button.setEnabled(ready)
        self.save_button.setEnabled(self.spec is not None)

    def run(self) -> str | None:
        self.refresh()
        if self.spec is None or self.problems:
            return None
        run_id = self.ctx.experiments.launch(self.spec)
        self.result.setText(f"Queued as run {run_id}.")
        self.ctx.navigate("training")
        return run_id

    # Files ----------------------------------------------------------------------
    def save_to(self, path: Path) -> Path:
        if self.spec is None:
            raise SpecError("the form does not describe a valid experiment yet")
        dump_spec(self.spec, path)
        self.result.setText(f"Saved {path.name}.")
        return path

    def open_file(self, path: Path) -> bool:
        try:
            spec = load_spec(path)
        except (OSError, SpecError) as exc:
            self.result.setText(str(exc))
            return False
        self.load(spec)
        self.result.setText(f"Opened {path.name}.")
        return True

    def _ask_save(self) -> None:
        if self.spec is None:
            return
        start = self.ctx.paths.experiments / f"{slug(self.spec.name)}.yaml"
        path, _ = QFileDialog.getSaveFileName(self, "Save experiment.yaml", str(start), "YAML (*.yaml *.yml)")
        if path:
            self.save_to(Path(path))

    def _ask_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open experiment.yaml", str(self.ctx.paths.experiments),
                                              "YAML (*.yaml *.yml)")
        if path:
            self.open_file(Path(path))
