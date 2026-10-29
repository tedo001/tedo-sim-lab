"""The Experiment Builder's Research card: why the experiment is run (question, hypothesis,
baseline run, limitations, for its reproducibility report; never part of the result) and what the
data checks do before training (``data.validation``)."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QLineEdit, QPlainTextEdit, QWidget

from core.experiment_engine.spec import ExperimentSpec, ResearchSpec

from ....services.context import AppContext
from ...widgets import Card, label
from .fields import combo, form

__all__ = ["ResearchSection", "VALIDATION"]

VALIDATION = (("Warn about problems", "warn"), ("Stop the run on problems", "fail"), ("Off", "off"))


def _text_box(placeholder: str) -> QPlainTextEdit:
    box = QPlainTextEdit()
    box.setPlaceholderText(placeholder)
    box.setFixedHeight(56)
    box.setTabChangesFocus(True)
    return box


class ResearchSection(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Research", "for the run's reproducibility report", parent=parent)
        self.ctx = ctx
        rows = form(self)
        self.question = QLineEdit()
        self.question.setPlaceholderText("What do you want to find out?")
        self.hypothesis = _text_box("What you expect, and how the result would show it")
        self.baseline = QComboBox()
        self.baseline.setMinimumWidth(220)
        self.baseline.setMaximumWidth(420)
        self.limitations = _text_box("What this experiment cannot tell (the lab adds what it sees itself)")
        self.validation = combo(list(VALIDATION))
        self.validation.setToolTip("Before training: duplicate samples, the same sample in two splits, "
                                   "conflicting labels, missing labels, class imbalance, licence metadata")
        rows.addRow("Question", self.question)
        rows.addRow("Hypothesis", self.hypothesis)
        rows.addRow("Baseline run", self.baseline)
        rows.addRow("Limitations", self.limitations)
        rows.addRow("Data checks", self.validation)
        self.add(label("The report compares every score with the baseline run's.", "CardCaption", wrap=True))
        self.refresh_runs()

    def refresh_runs(self) -> None:
        """The finished runs a new experiment can be measured against, newest first."""
        chosen = self.baseline.currentData()
        self.baseline.blockSignals(True)
        self.baseline.clear()
        self.baseline.addItem("None", None)
        for row in self.ctx.store.runs_with_experiments(limit=200):
            if row["status"] in ("completed", "early_stopped"):
                self.baseline.addItem(f"{row['experiment_name']} · {row['id'][:8]}", row["id"])
        self.baseline.setCurrentIndex(max(self.baseline.findData(chosen), 0))
        self.baseline.blockSignals(False)

    def research(self) -> ResearchSpec:
        return ResearchSpec(question=self.question.text().strip(),
                            hypothesis=self.hypothesis.toPlainText().strip(),
                            baseline=self.baseline.currentData(),
                            limitations=self.limitations.toPlainText().strip())

    def data_validation(self) -> str:
        return self.validation.currentData() or "warn"

    def load(self, spec: ExperimentSpec) -> None:
        research = spec.research
        self.question.setText(research.question)
        self.hypothesis.setPlainText(research.hypothesis)
        self.limitations.setPlainText(research.limitations)
        self.refresh_runs()
        if research.baseline and self.baseline.findData(research.baseline) < 0:
            self.baseline.addItem(f"{research.baseline[:8]} (not in this workspace)", research.baseline)
        self.baseline.setCurrentIndex(max(self.baseline.findData(research.baseline), 0))
        self.validation.setCurrentIndex(max(self.validation.findData(spec.data.validation), 0))
