"""Compare Experiments' *Model performance* card: the ticked runs as shapes on a radar, one spoke
per score they all have (accuracy, precision, recall, F1, ROC AUC, best validation accuracy; mAP,
AP50, AP75 and mean IoU for detection runs; silhouette and variance explained are not shares of a
correct answer and are left out). *Absolute* reads every spoke from 0 to 1; *Relative to the best*
divides each score by the best ticked run's, so small differences show. At most four runs."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QComboBox, QWidget

from ..widgets import Card, label
from ..widgets.radar import MAX_SHAPES, RadarChart

__all__ = ["MEASURES", "PerformanceCard", "radar_axes"]

#: Scores where 1 is perfect and higher is better: (metric key, spoke name).
MEASURES = (("test_map", "mAP"), ("test_map50", "AP50"), ("test_map75", "AP75"), ("test_mean_iou", "IoU"),
            ("test_acc", "Accuracy"), ("test_precision", "Precision"), ("test_recall", "Recall"),
            ("test_f1", "F1 score"), ("test_roc_auc", "ROC AUC"), ("best_val_acc", "Best val. accuracy"))


def radar_axes(runs: list[dict[str, float]]) -> list[tuple[str, str]]:
    """The measures every run has, in the standard order."""
    return [(key, title) for key, title in MEASURES if runs and all(key in run for run in runs)]


class PerformanceCard(Card):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Model performance", "test scores on one radar; 1 is perfect", parent=parent)
        self.scale = QComboBox()
        self.scale.addItem("Absolute (0–1)", False)
        self.scale.addItem("Relative to the best run", True)
        self.scale.setMinimumWidth(180)
        self.scale.currentIndexChanged.connect(lambda _: self._draw())
        self.add_head_widget(self.scale)
        self.chart = RadarChart()
        self.add(self.chart)
        self.note = label("", "CardCaption", wrap=True)
        self.add(self.note)
        self._runs: list[tuple[str, dict[str, float]]] = []

    def show_runs(self, runs: list[tuple[str, dict[str, Any]]]) -> None:
        """(run name, its metrics) for every ticked run, in tick order."""
        self._runs = runs
        self._draw()

    def _draw(self) -> None:
        shown = self._runs[:MAX_SHAPES]
        axes = radar_axes([metrics for _name, metrics in shown])
        relative = bool(self.scale.currentData())
        self.chart.set_data([title for _key, title in axes],
                            [(name, [float(metrics[key]) for key, _title in axes]) for name, metrics in shown]
                            if len(axes) >= 3 else [], relative=relative)
        notes = []
        if len(self._runs) > MAX_SHAPES:
            notes.append(f"The first {MAX_SHAPES} ticked runs are drawn; more overlapping shapes cannot be "
                         "read. The table above has every run.")
        if shown and len(axes) < 3:
            notes.append("These runs share fewer than three scores (for example a regression and a "
                         "classification run), so there is no radar to draw.")
        elif relative:
            notes.append("Each spoke is divided by the best ticked run's score: the rim is the best, not "
                         "perfection.")
        self.note.setText(" ".join(notes))
        self.note.setVisible(bool(notes))
        self.setVisible(bool(self._runs))
