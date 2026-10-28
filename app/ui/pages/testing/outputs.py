"""What Model Testing shows for an answer: the classes ranked by probability, a regression value,
boxes drawn on the image, a caption or a transcript."""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QVBoxLayout, QWidget

from labs.computer_vision.coco import Annotation

from ...widgets import DataTable, label
from ...widgets.annotated import AnnotatedImage
from ...widgets.plots import BarList

__all__ = ["OutputPanel", "detections"]


def detections(boxes: list[dict[str, Any]]) -> list[Annotation]:
    """Detector boxes (x0, y0, x1, y1) as annotations the image widget draws."""
    found = []
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = box["box"]
        found.append(Annotation(index, f"{box['label']} {box['score']:.0%}", (x0, y0, x1 - x0, y1 - y0),
                                (x1 - x0) * (y1 - y0), False))
    return found


class OutputPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.headline = label("", "BodyStrong", wrap=True)
        self.detail = label("", "CardCaption", wrap=True)
        self.bars = BarList()
        self.image = AnnotatedImage()
        self.table = DataTable(("Object", "Score", "Box (x0, y0, x1, y1)"), mono_columns=(1, 2),
                               stretch_column=0)
        self.text = label("", "Body", wrap=True, selectable=True)
        for widget in (self.headline, self.detail, self.bars, self.image, self.table, self.text):
            layout.addWidget(widget)
        self.clear()

    def clear(self, message: str = "") -> None:
        self.headline.setText(message)
        self.detail.setText("")
        for widget in (self.bars, self.image, self.table, self.text, self.detail):
            widget.hide()

    def show_ranking(self, classes: list[str], probabilities: list[float], what: str = "Predicted") -> None:
        self.clear(f"{what}: {classes[0]} ({probabilities[0]:.1%})" if classes else "No answer")
        self.bars.set_rows([[name, p] for name, p in zip(classes, probabilities, strict=False)], limit=10)
        self.bars.show()

    def show_prediction(self, prediction: dict[str, Any]) -> None:
        if "value" in prediction:
            self.clear(f"Predicted value: {prediction['value']:.4g}")
            return
        self.show_ranking(prediction["classes"], prediction["probabilities"])

    def show_detections(self, path: str, result: dict[str, Any]) -> None:
        boxes = result.get("boxes", [])
        self.clear(f"{len(boxes)} object{'s' if len(boxes) != 1 else ''} found")
        self.image.set_content(QImage(path), int(result.get("width", 0)), int(result.get("height", 0)),
                               detections(boxes))
        self.image.show()
        self.table.clear_rows()
        for box in boxes:
            self.table.add_row((box["label"], f"{box['score']:.3f}",
                                ", ".join(f"{v:.0f}" for v in box["box"])))
        self.table.setVisible(bool(boxes))

    def show_text(self, headline: str, text: str, detail: str = "") -> None:
        self.clear(headline)
        self.text.setText(text)
        self.text.show()
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))
