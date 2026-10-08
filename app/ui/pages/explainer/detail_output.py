"""The output layer: flatten → linear → softmax, for one class (CNN Explainer's softmax
and flatten views). The class score is a weighted sum over every value of the last
maps; softmax turns the scores into probabilities that add up to one."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QGridLayout, QSizePolicy, QVBoxLayout, QWidget

from labs.computer_vision.explainer import Trace, linear_contributions, softmax_terms

from ...theme.tokens import COLORS, FONT_FAMILY
from ...widgets import label
from .colours import heat_colour, mono_font
from .detail_base import MapColumn, fmt

__all__ = ["OutputDetail", "ScoreTable"]

_ROW = 22


class ScoreTable(QWidget):
    """Every class with its score (diverging colour, number) and probability (bar, number).
    Click a row to explain that class."""

    class_clicked = Signal(int)

    def __init__(self, names: tuple[str, ...], logits: np.ndarray, probabilities: np.ndarray,
                 selected: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.names, self.logits, self.probabilities, self.selected = names, logits, probabilities, selected
        self.limit = float(np.abs(logits).max()) or 1.0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(_ROW * (len(names) + 1))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:
        return QSize(360, self.height())

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        width = self.width()
        name_w, score_w = min(width * 0.36, 150), 64
        bar_x = name_w + score_w + 12
        bar_w = max(width - bar_x - 52, 30)
        painter.setFont(mono_font(10))
        painter.setPen(QColor(COLORS["text_faint"]))
        painter.drawText(QRectF(0, 0, name_w, _ROW), int(Qt.AlignmentFlag.AlignVCenter), "class")
        painter.drawText(QRectF(name_w, 0, score_w, _ROW), int(Qt.AlignmentFlag.AlignVCenter), "score z")
        painter.drawText(QRectF(bar_x, 0, bar_w, _ROW), int(Qt.AlignmentFlag.AlignVCenter), "probability")
        text_font = QFont(FONT_FAMILY)
        text_font.setPixelSize(12)
        for i, name in enumerate(self.names):
            top = _ROW * (i + 1)
            if i == self.selected:
                painter.fillRect(QRectF(0, top, width, _ROW), QColor(COLORS["selected"]))
            painter.setFont(text_font)
            painter.setPen(QColor(COLORS["text"] if i == self.selected else COLORS["text_dim"]))
            painter.drawText(QRectF(6, top, name_w - 6, _ROW), int(Qt.AlignmentFlag.AlignVCenter), name)
            swatch = QRectF(name_w, top + 6, 10, _ROW - 12)
            painter.fillRect(swatch, heat_colour(float(self.logits[i]), self.limit))
            painter.setFont(mono_font(11))
            painter.drawText(QRectF(name_w + 14, top, score_w, _ROW), int(Qt.AlignmentFlag.AlignVCenter),
                             fmt(float(self.logits[i])))
            bar = QRectF(bar_x, top + 7, bar_w, _ROW - 14)
            painter.fillRect(bar, QColor(COLORS["border_soft"]))
            share = bar_w * float(self.probabilities[i])
            painter.fillRect(QRectF(bar.left(), bar.top(), share, bar.height()),
                             QColor(COLORS["series_1"]))
            painter.drawText(QRectF(bar.right() + 6, top, 46, _ROW), int(Qt.AlignmentFlag.AlignVCenter),
                             f"{self.probabilities[i]:.3f}")
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        row = int(event.position().y() // _ROW) - 1
        if 0 <= row < len(self.names):
            self.class_clicked.emit(row)


class OutputDetail(QWidget):
    class_clicked = Signal(int)

    def __init__(self, trace: Trace, unit: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        architecture = trace.net.architecture
        index = len(architecture.layers) - 1
        names = architecture.class_names
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        flat = architecture.input_of(index)[0]
        maps = architecture.input_of(index - 1)
        layout.addWidget(label(
            f"Flatten lays the last {maps[0]} maps of {maps[1]}×{maps[2]} out as one vector of {flat} values "
            "(channel by channel, row by row). Each class score z is a weighted sum of all of them plus a "
            "bias. Softmax then turns the scores into probabilities: exp(z) for the class divided by the "
            "sum of exp(z) over all classes, so they are positive and add up to 1, and the largest score "
            "gets a disproportionately large share.", "Body", wrap=True))

        terms = softmax_terms(trace.logits)
        self.table = ScoreTable(names, terms.logits, trace.probabilities, unit)
        self.table.class_clicked.connect(self.class_clicked)
        layout.addWidget(self.table)
        shifted = " − max z" if terms.logits[0] != trace.logits[0] else ""
        self.formula = label(
            f"p({names[unit]}) = exp({fmt(float(terms.logits[unit]))}{shifted}) / Σ exp(z{shifted}) "
            f"= {fmt(float(terms.exps[unit]))} / {fmt(terms.total)} = {terms.probability(unit):.4f}",
            "Mono", wrap=True, selectable=True)
        layout.addWidget(self.formula)

        contributions, bias = linear_contributions(trace, index, unit)
        layout.addWidget(label(
            f"Where the score for “{names[unit]}” comes from: weight × value for each of the {flat} "
            f"flattened values, shown in place on the last maps (blue pushes the score up, red down). "
            f"Sum {fmt(float(contributions.sum()))} + bias {fmt(bias)} = {fmt(float(trace.logits[unit]))}.",
            "CardCaption", wrap=True))
        limit = float(np.abs(contributions).max()) or 1.0
        grid = QGridLayout()
        grid.setSpacing(8)
        self.contribution_maps: list[MapColumn] = []
        for c in range(contributions.shape[0]):
            column = MapColumn(f"ch {c + 1} · {fmt(float(contributions[c].sum()))}", side=64)
            column.view.set_map(contributions[c], limit)
            grid.addWidget(column, c // 5, c % 5)
            self.contribution_maps.append(column)
        grid.setColumnStretch(5, 1)
        layout.addLayout(grid)
