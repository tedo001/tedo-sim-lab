"""The Experiment Builder's scikit-learn settings: target and feature columns, test share,
scaling and feature selection, cross-validation, and a grid or random search."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from core.common.vocab import Task
from core.dataset_registry.adapter import TabularAdapter
from core.experiment_engine.spec import ExperimentSpec, SearchSpec, SklearnTrainingSpec, TransformSpec
from labs.classical_ml.datasets import columns_of

from ....services.context import AppContext
from ...widgets import Card, label
from .fields import combo, double, form, spin, yaml_box, yaml_mapping, yaml_text
from .torch_section import form_widget

__all__ = ["SCORES", "SklearnSection"]

#: Scores a search or cross-validation can optimise, per task: (title, scikit-learn name).
SCORES = {
    Task.TABULAR_CLASSIFICATION: (("Accuracy", "accuracy"), ("Balanced accuracy", "balanced_accuracy"),
                                  ("Macro F1", "f1_macro"), ("ROC AUC (one vs rest)", "roc_auc_ovr")),
    Task.TABULAR_REGRESSION: (("R²", "r2"), ("Mean absolute error", "neg_mean_absolute_error"),
                              ("Root mean squared error", "neg_root_mean_squared_error")),
}


class SklearnSection:
    """Builds ``data`` (rows inside the data step) and ``cards`` (the fitting step)."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.data = QWidget()
        rows = form_widget(self.data)
        self.target = combo([])
        self.features = QListWidget()
        self.features.setMaximumHeight(132)
        self.features.setMaximumWidth(320)
        self.test_fraction = double(0.05, 0.5, 0.2, 2, 0.05)
        self.subset = spin(0, 100_000_000, 0, special="all")
        self.scaling = combo([("None", None), ("Standardise (mean 0, sd 1)", "standardize"),
                              ("Min–max to 0 … 1", "minmax")])
        self.select = QCheckBox("Keep only the")
        self.k = spin(1, 10_000, 10)
        select_row = QHBoxLayout()
        select_row.addWidget(self.select)
        select_row.addWidget(self.k)
        select_row.addWidget(label("features most related to the target", "Body"))
        select_row.addStretch(1)
        for title, widget in (("Target column", self.target), ("Feature columns", self.features),
                              ("Test share", self.test_fraction), ("Rows", self.subset),
                              ("Scaling", self.scaling)):
            rows.addRow(title, widget)
        rows.addRow("", select_row)
        rows.addRow("", label("Missing values are filled in (median, or most frequent for text) and text "
                              "columns are one-hot encoded, so every model accepts every table.",
                              "CardCaption", wrap=True))

        fitting = Card("4 · Fitting")
        self._fitting_rows = rows = form(fitting)
        self.cv_folds = spin(0, 20, 0, special="off")
        self.search = combo([("No search", None), ("Grid search (every combination)", "grid"),
                             ("Random search", "random")])
        self.space = yaml_box("C: [0.1, 1, 10]\ngamma: [scale, 0.01]")
        self.n_iter = spin(1, 10_000, 10)
        self.score = combo([])
        self.seed = spin(0, 2_147_483_647, 42)
        for title, widget in (("Cross-validation folds", self.cv_folds), ("Search", self.search),
                              ("Search space", self.space), ("Random tries", self.n_iter),
                              ("Optimise", self.score), ("Seed", self.seed)):
            rows.addRow(title, widget)
        fitting.add(label("Fits run on the CPU in seconds for these tables. A search keeps the best "
                          "settings by cross-validation (5 folds unless set) and then scores the test rows "
                          "once. XGBoost can use a GPU with the model option device: cuda.", "CardCaption",
                          wrap=True))
        self.search.currentIndexChanged.connect(self._search_changed)
        self.cards = QWidget()
        stack = QVBoxLayout(self.cards)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.addWidget(fitting)
        self._search_changed()

    # Choices that depend on the task and the dataset ---------------------------------
    def set_task(self, task: Task) -> None:
        current = self.score.currentData()
        self.score.blockSignals(True)
        self.score.clear()
        self.score.addItem("The task's default (accuracy or R²)", None)
        for title, name in SCORES.get(task, ()):
            self.score.addItem(title, name)
        self.score.setCurrentIndex(max(self.score.findData(current), 0))
        self.score.blockSignals(False)
        supervised = task in SCORES
        for widget in (self.target, self.test_fraction, self.select, self.k, self.cv_folds, self.search):
            widget.setEnabled(supervised)
        self._search_changed()

    def set_dataset(self, card_id: str | None) -> None:
        """List ``card_id``'s columns: its own target chosen, every feature checked."""
        columns, target = [], None
        if card_id is not None and card_id in self.ctx.catalog.datasets:
            adapter = self.ctx.catalog.datasets.adapter(card_id)
            if isinstance(adapter, TabularAdapter):
                columns, target = columns_of(adapter, self.ctx.paths.datasets)
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItem(f"The dataset's target ({target})" if target else "None", None)
        for column in columns:
            self.target.addItem(column, column)
        self.target.setCurrentIndex(0)
        self.target.blockSignals(False)
        self.features.blockSignals(True)
        self.features.clear()
        for column in columns:
            item = QListWidgetItem(column)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.features.addItem(item)
        self.features.blockSignals(False)

    def _checked(self) -> list[str] | None:
        """Checked feature columns; ``None`` when all are (the spec then lists none)."""
        items = [self.features.item(i) for i in range(self.features.count())]
        chosen = [item.text() for item in items if item.checkState() == Qt.CheckState.Checked]
        return None if len(chosen) == len(items) else chosen

    def _search_changed(self) -> None:
        method = self.search.currentData() if self.search.isEnabled() else None
        for widget, shown in ((self.space, method is not None), (self.n_iter, method == "random"),
                              (self.score, method is not None)):
            self._fitting_rows.setRowVisible(widget, shown)

    # Spec ⇄ controls -------------------------------------------------------------------
    def data_fields(self) -> dict[str, Any]:
        steps = []
        if self.scaling.currentData():
            steps.append(TransformSpec(name=self.scaling.currentData()))
        if self.select.isChecked() and self.select.isEnabled():
            steps.append(TransformSpec(name="select_k_best", params={"k": self.k.value()}))
        return {"preprocessing": tuple(steps), "subset": self.subset.value() or None}

    def training(self, task: Task) -> SklearnTrainingSpec:
        supervised = task in SCORES
        method = self.search.currentData() if supervised else None
        search = None
        if method is not None:
            space = yaml_mapping(self.space.toPlainText(), "Search space")
            if not space:
                raise ValueError("Search space: list the values to try, e.g. C: [0.1, 1, 10]")
            search = SearchSpec(method=method, n_iter=self.n_iter.value() if method == "random" else None,
                                scoring=self.score.currentData(),
                                space={key: tuple(value) if isinstance(value, list) else (value,)
                                       for key, value in space.items()})
        target = self.target.currentData() if supervised else None
        features = self._checked()
        if features is not None:
            features = [name for name in features if name != target]  # the target is never a feature
            if not features:
                raise ValueError("Feature columns: check at least one column to learn from")
        return SklearnTrainingSpec(target=target, features=tuple(features) if features is not None else None,
                                   test_fraction=self.test_fraction.value(),
                                   cv_folds=(self.cv_folds.value() or None) if supervised else None,
                                   search=search)

    def load(self, spec: ExperimentSpec) -> None:
        steps = {step.name: step for step in spec.data.preprocessing}
        scaling = "standardize" if "standardize" in steps else "minmax" if "minmax" in steps else None
        self.scaling.setCurrentIndex(max(self.scaling.findData(scaling), 0))
        self.select.setChecked("select_k_best" in steps)
        if "select_k_best" in steps:
            self.k.setValue(int(steps["select_k_best"].params.get("k", 10)))
        self.subset.setValue(spec.data.subset or 0)
        self.seed.setValue(spec.seed)
        training = spec.training
        if not isinstance(training, SklearnTrainingSpec):
            return
        if training.target and self.target.findData(training.target) < 0:
            self.target.addItem(training.target, training.target)
        self.target.setCurrentIndex(max(self.target.findData(training.target), 0))
        for i in range(self.features.count()):
            item = self.features.item(i)
            item.setCheckState(Qt.CheckState.Checked if training.features is None
                               or item.text() in training.features else Qt.CheckState.Unchecked)
        self.test_fraction.setValue(training.test_fraction)
        self.cv_folds.setValue(training.cv_folds or 0)
        search = training.search
        self.search.setCurrentIndex(max(self.search.findData(search.method if search else None), 0))
        self.space.setPlainText(yaml_text(dict(search.space)) if search else "")
        if search and search.n_iter:
            self.n_iter.setValue(search.n_iter)
        self.score.setCurrentIndex(max(self.score.findData(search.scoring if search else None), 0))
