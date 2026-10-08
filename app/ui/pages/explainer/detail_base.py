"""Shared pieces of the detail views (CNN Explainer's "interactive formula" views): an
animation that walks the output cell by cell, number grids for the arithmetic, and a
titled map."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ...icons import icon
from ...theme.tokens import COLORS
from ...widgets import label
from .colours import heat_colour, mono_font
from .maps import MapView

__all__ = ["MapColumn", "NumberGrid", "StepView", "fmt"]

STEP_MS = 250


def fmt(value: float) -> str:
    """Numbers as the views print them: 3 significant digits, a real minus sign."""
    return f"{value:.3g}".replace("-", "−")


class MapColumn(QWidget):
    """A caption over a :class:`MapView`."""

    def __init__(self, caption: str, side: int = 160, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.caption = label(caption, "CardCaption")
        self.view = MapView(side)
        layout.addWidget(self.caption)
        layout.addWidget(self.view)
        layout.addStretch(1)


class NumberGrid(QWidget):
    """A small matrix with every value written in its cell, coloured on the diverging scale
    (``limit``) or in greys (``limit=None``). One cell can be marked."""

    def __init__(self, cell: int = 42, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cell = cell
        self.values = np.zeros((1, 1))
        self.limit: float | None = 1.0
        self.marked: tuple[int, int] | None = None
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_values(self, values: np.ndarray, limit: float | None,
                   marked: tuple[int, int] | None = None) -> None:
        self.values = np.atleast_2d(values)
        self.limit, self.marked = limit, marked
        self.setFixedSize(self.sizeHint())
        self.update()

    def sizeHint(self) -> QSize:
        rows, columns = self.values.shape
        return QSize(columns * self.cell + 1, rows * self.cell + 1)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setFont(mono_font(10))
        rows, columns = self.values.shape
        for row in range(rows):
            for column in range(columns):
                value = float(self.values[row, column])
                rect = QRectF(column * self.cell, row * self.cell, self.cell, self.cell)
                if self.limit is None:
                    level = int(round(min(max(value, 0.0), 1.0) * 255))
                    fill = QColor(level, level, level)
                else:
                    fill = heat_colour(value, self.limit)
                painter.fillRect(rect, fill)
                painter.setPen(QPen(QColor(COLORS["surface"]), 1))
                painter.drawRect(rect)
                ink = "#0F1114" if fill.lightnessF() > 0.6 else COLORS["text"]
                painter.setPen(QColor(ink))
                painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), f"{value:.2f}".replace("-", "−"))
        if self.marked is not None:
            painter.setPen(QPen(QColor(COLORS["text"]), 2))
            row, column = self.marked
            inset = QRectF(column * self.cell, row * self.cell, self.cell, self.cell).adjusted(1, 1, -1, -1)
            painter.drawRect(inset)
        painter.end()


class StepView(QWidget):
    """Base for views that walk an output map one cell at a time.

    Subclasses set :attr:`rows`/:attr:`columns`, connect their maps' ``hovered`` to
    :meth:`hover`, and implement :meth:`show_step`.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.rows = self.columns = 1
        self.position = (0, 0)
        self.timer = QTimer(self)
        self.timer.setInterval(STEP_MS)
        self.timer.timeout.connect(self.advance)
        self.play_button = QPushButton()
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggled)
        self._toggled(False)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(10)

    def controls(self, hint: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.play_button)
        row.addWidget(label(hint, "CardCaption", wrap=True), 1)
        return row

    @property
    def playing(self) -> bool:
        return self.timer.isActive()

    def play(self) -> None:
        self.play_button.setChecked(True)

    def pause(self) -> None:
        self.play_button.setChecked(False)

    def _toggled(self, on: bool) -> None:
        self.play_button.setIcon(icon("pause" if on else "play"))
        self.play_button.setText("Pause" if on else "Play")
        if on:
            self.timer.start()
        else:
            self.timer.stop()

    def advance(self) -> None:
        flat = (self.position[0] * self.columns + self.position[1] + 1) % (self.rows * self.columns)
        self.go_to(*divmod(flat, self.columns))

    def go_to(self, y: int, x: int) -> None:
        self.position = (min(max(y, 0), self.rows - 1), min(max(x, 0), self.columns - 1))
        self.show_step(*self.position)

    def hover(self, y: int, x: int) -> None:
        """Pointer over an output cell: stop the animation and show that cell."""
        self.pause()
        self.go_to(y, x)

    def show_step(self, y: int, x: int) -> None:
        raise NotImplementedError

    def start(self) -> None:
        self._toggled(False)
        self.go_to(0, 0)
        self.play()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.pause()
        super().hideEvent(event)

