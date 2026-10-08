"""What a scikit-learn run found, read from its ``results.json``: scores, ROC and
precision-recall curves, predicted against actual, clusters or projections, permutation
importance, SHAP values and the search. Shared by the Training and Evaluation pages; a card
with nothing to show stays hidden."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QComboBox, QVBoxLayout, QWidget

from ..widgets import Card, DataTable, ResponsiveRow, label
from ..widgets.plots import GROUP_COLOURS, BarList, CurveChart, ScatterChart

__all__ = ["METRIC_TITLES", "SklearnResults", "read_results"]

#: Metric key → words, in the order they are listed.
METRIC_TITLES = {
    "test_acc": "Accuracy", "test_balanced_acc": "Balanced accuracy", "test_precision": "Precision (macro)",
    "test_recall": "Recall (macro)", "test_f1": "F1 (macro)", "test_roc_auc": "ROC AUC (mean of classes)",
    "test_log_loss": "Log loss", "test_r2": "R²", "test_mae": "Mean absolute error",
    "test_rmse": "Root mean squared error", "test_residual_std": "Residual standard deviation",
    "cv_mean": "Cross-validation score (mean)", "cv_std": "Cross-validation score (sd)",
    "train_acc": "Training accuracy", "train_f1": "Training F1", "train_r2": "Training R²",
    "train_rmse": "Training RMSE", "clusters": "Clusters", "silhouette": "Silhouette",
    "inertia": "Inertia (within-cluster squares)", "adjusted_rand": "Adjusted Rand vs. the labels",
    "components": "Components", "explained_variance": "Variance explained",
}


def read_results(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "results.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        return None


def _format(key: str, value: float) -> str:
    if key in ("clusters", "components"):
        return f"{int(value)}"
    if key in ("test_acc", "test_balanced_acc", "train_acc", "explained_variance"):
        return f"{value:.2%}"
    return f"{value:.4f}" if abs(value) < 1e4 else f"{value:,.1f}"


def _combo() -> QComboBox:
    box = QComboBox()
    box.setMaximumWidth(260)
    return box


class SklearnResults(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.results: dict[str, Any] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.scores_card = Card("Scores", "test rows held out from fitting", padded=False)
        self.scores = DataTable(("Measure", "Value"), mono_columns=(1,), stretch_column=0)
        self.scores_card.add(self.scores)

        self.curves_card = Card("ROC and precision-recall", "one class against the rest")
        self.curve_class = _combo()
        self.curve_class.currentIndexChanged.connect(lambda _: self._show_curve())
        self.curves_card.add_head_widget(self.curve_class)
        self.roc = CurveChart("ROC curve", "false positive rate", "true positive rate", chance=True)
        self.pr = CurveChart("Precision-recall", "recall", "precision")
        self.curves_card.add(ResponsiveRow([(self.roc, 1), (self.pr, 1)], breakpoint=640))

        self.scatter_card = Card("Points")
        self.highlight = _combo()
        self.highlight.currentIndexChanged.connect(self._highlight)
        self.scatter_card.add_head_widget(self.highlight)
        self.scatter = ScatterChart("", "", "")
        self.scatter.setMinimumHeight(340)
        self.scatter_card.add(self.scatter)
        self.legend = label("", "CardCaption", wrap=True)
        self.scatter_card.add(self.legend)

        self.importance_card = Card("Permutation importance", "how much the test score drops when a column "
                                                              "is shuffled (5 repeats)")
        self.importance = BarList()
        self.importance_card.add(self.importance)
        self.importance_note = label("", "CardCaption", wrap=True)
        self.importance_card.add(self.importance_note)
        self.shap_card = Card("SHAP values", "mean |SHAP| per encoded feature over up to 200 test rows")
        self.shap = BarList()
        self.shap_card.add(self.shap)

        self.search_card = Card("Search", "best settings first; scores are cross-validated", padded=False)
        self.search = DataTable(("Settings", "Mean", "sd"), mono_columns=(0, 1, 2), stretch_column=0)
        self.search_card.add(self.search)
        for card in (self.scores_card, self.curves_card, self.scatter_card, self.importance_card,
                     self.shap_card, self.search_card):
            layout.addWidget(card)

    # Filling ----------------------------------------------------------------------
    def show_results(self, results: dict[str, Any] | None) -> None:
        self.results = results or {}
        found = self.results
        metrics = found.get("metrics", {})
        self.scores.clear_rows()
        for key, title in METRIC_TITLES.items():
            if key in metrics:
                self.scores.add_row((title, _format(key, float(metrics[key]))))
        self.scores_card.setVisible(bool(metrics))
        unsupervised = found.get("task") in ("clustering", "dimensionality_reduction")
        self.scores_card.caption.setText("fitted on every row; labels, when the table has them, only compared"
                                         if unsupervised else "test rows held out from fitting")

        curves = found.get("curves", [])
        self.curve_class.blockSignals(True)
        self.curve_class.clear()
        for curve in curves:
            self.curve_class.addItem(f"{curve['class']} against the rest", curve["class"])
        self.curve_class.blockSignals(False)
        self.curve_class.setVisible(len(curves) > 1)
        self.curves_card.setVisible(bool(curves))
        self._show_curve()
        self._show_scatter(found)

        rows = found.get("importance", [])
        helpful = [row for row in rows if row[1] > 0]
        self.importance.set_rows(helpful)
        self.importance_note.setText(f"{len(rows) - len(helpful)} more columns made no measurable difference."
                                     if len(rows) > len(helpful) else "")
        self.importance_note.setVisible(len(rows) > len(helpful))
        self.importance_card.setVisible(bool(rows))
        self.shap.set_rows(found.get("shap", []))
        self.shap_card.setVisible(bool(found.get("shap")))
        search = found.get("search")
        self.search.clear_rows()
        if search:
            for row in search.get("top", []):
                settings = ", ".join(f"{key}={value}" for key, value in row["params"].items())
                self.search.add_row((settings, f"{row['mean']:.4f}", f"{row['std']:.4f}"))
            self.search_card.caption.setText(f"{search['scoring']}, cross-validated; best settings first")
        self.search_card.setVisible(bool(search))

    def _show_curve(self) -> None:
        curves = (self.results or {}).get("curves", [])
        index = self.curve_class.currentIndex()
        if not curves or index < 0:
            return
        curve = curves[index]
        self.roc.set_curve(curve["roc"], f"AUC {curve['roc_auc']:.3f}")
        self.pr.set_curve(curve["pr"], f"AP {curve['average_precision']:.3f}")

    def _show_scatter(self, found: dict[str, Any]) -> None:
        task = found.get("task")
        groups: list[str] = []
        if "predicted_vs_actual" in found:
            title, x_label, y_label, points = ("Predicted against actual", "actual", "predicted",
                                               found["predicted_vs_actual"])
            caption = "Each dot is a test row; on the dashed line the prediction is exact."
        elif task in ("clustering", "dimensionality_reduction") and found.get("points"):
            groups = list(found.get("groups") or [])
            if task == "clustering":
                title = "Clusters"
                caption = "Rows on the first two principal components of the prepared data."
            else:
                title, caption = "First two components", "Rows on the first two components."
            x_label, y_label, points = "component 1", "component 2", found["points"]
            if not groups:
                points = [p[:2] for p in points]
        else:
            self.scatter_card.hide()
            return
        self.scatter_card.show()
        self.scatter_card.title.setText(title)
        self.scatter.title, self.scatter.x_label, self.scatter.y_label = "", x_label, y_label
        self.scatter.identity = "predicted_vs_actual" in found
        self.highlight.blockSignals(True)
        self.highlight.clear()
        for name in groups:
            self.highlight.addItem(f"Highlight: {name}", name)
        self.highlight.blockSignals(False)
        self.highlight.setVisible(len(groups) > len(GROUP_COLOURS))
        self.scatter.set_points(points, groups)
        if 0 < len(groups) <= len(GROUP_COLOURS):
            names = ("blue", "orange", "green")
            caption += " " + ", ".join(f"{names[i]}: {name}" for i, name in enumerate(groups)) + "."
        elif len(groups) > len(GROUP_COLOURS):
            caption += " One group at a time is coloured (choose above); the rest are grey."
        self.legend.setText(caption)

    def _highlight(self, index: int) -> None:
        self.scatter.highlight = max(index, 0)
        self.scatter.update()
