"""The circuit drawn as it is written on paper: one wire per qubit, time to the right, gates as
boxes, controls as dots joined to their target (⊕ for NOT), swaps as two crosses. A click on a
cell asks to place the chosen gate there; a right-click removes the gate under the mouse."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from labs.simulation.quantum import Circuit, Op

from ...theme.tokens import COLORS, MONO_FAMILY
from ...widgets.plots import _font

__all__ = ["CircuitCanvas"]

CELL, ROW, LEFT, TOP = 58, 48, 54, 18


class CircuitCanvas(QWidget):
    clicked = Signal(int, int)        # column, qubit
    remove_requested = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.circuit = Circuit(2)
        self.pending: list[int] = []   # qubits chosen so far for a multi-qubit gate
        self.pending_column: int | None = None
        self._hover: tuple[int, int] | None = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_circuit(self, circuit: Circuit) -> None:
        self.circuit = circuit
        self.updateGeometry()
        self.setFixedHeight(self.sizeHint().height())
        self.update()

    def set_pending(self, column: int | None, qubits: list[int]) -> None:
        self.pending_column, self.pending = column, list(qubits)
        self.update()

    def sizeHint(self) -> QSize:
        columns = max(self.circuit.columns + 2, 10)
        return QSize(LEFT + columns * CELL + 16, TOP + self.circuit.qubits * ROW + 10)

    def minimumSizeHint(self) -> QSize:
        return QSize(LEFT + 6 * CELL, TOP + self.circuit.qubits * ROW + 10)

    def cell_at(self, point: QPointF) -> tuple[int, int] | None:
        column = int((point.x() - LEFT) // CELL)
        qubit = int((point.y() - TOP) // ROW)
        if column < 0 or not 0 <= qubit < self.circuit.qubits:
            return None
        return column, qubit

    def _centre(self, column: int, qubit: int) -> QPointF:
        return QPointF(LEFT + column * CELL + CELL / 2, TOP + qubit * ROW + ROW / 2)

    # Painting -------------------------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(_font(MONO_FAMILY, 12))
        for qubit in range(self.circuit.qubits):
            y = TOP + qubit * ROW + ROW / 2
            painter.setPen(QColor(COLORS["text_dim"]))
            painter.drawText(QRectF(0, y - 10, LEFT - 10, 20),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), f"q{qubit}")
            painter.setPen(QPen(QColor(COLORS["border"]), 1.5))
            painter.drawLine(QPointF(LEFT, y), QPointF(self.width() - 8, y))
        if self._hover is not None:
            column, qubit = self._hover
            painter.fillRect(QRectF(LEFT + column * CELL + 2, TOP + qubit * ROW + 2, CELL - 4, ROW - 4),
                             QColor(COLORS["raised"]))
        for op in self.circuit.ops:
            self._paint_op(painter, op)
        if self.pending_column is not None:
            for qubit in self.pending:
                painter.setPen(QPen(QColor(COLORS["accent_hover"]), 2, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(self._centre(self.pending_column, qubit), 9, 9)

    def _paint_op(self, painter: QPainter, op: Op) -> None:
        accent = QColor(COLORS["accent"])
        qubits = op.qubits
        if len(qubits) > 1:
            top, bottom = self._centre(op.column, min(qubits)), self._centre(op.column, max(qubits))
            painter.setPen(QPen(accent, 2))
            painter.drawLine(top, bottom)
        for control in op.controls:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(accent)
            painter.drawEllipse(self._centre(op.column, control), 5, 5)
        base = op.gate[1:] if op.controls and op.gate.startswith("C") else op.gate
        if base in ("X",) and op.controls:  # ⊕
            centre = self._centre(op.column, op.targets[0])
            painter.setPen(QPen(accent, 2))
            painter.setBrush(QColor(COLORS["bg"]))
            painter.drawEllipse(centre, 11, 11)
            painter.drawLine(QPointF(centre.x() - 11, centre.y()), QPointF(centre.x() + 11, centre.y()))
            painter.drawLine(QPointF(centre.x(), centre.y() - 11), QPointF(centre.x(), centre.y() + 11))
            return
        if base == "SWAP":
            painter.setPen(QPen(accent, 2))
            for target in op.targets:
                c = self._centre(op.column, target)
                painter.drawLine(QPointF(c.x() - 7, c.y() - 7), QPointF(c.x() + 7, c.y() + 7))
                painter.drawLine(QPointF(c.x() - 7, c.y() + 7), QPointF(c.x() + 7, c.y() - 7))
            return
        label = Op(base, op.targets, (), op.params).label() if op.controls else op.label()
        for target in op.targets:
            centre = self._centre(op.column, target)
            box = QRectF(centre.x() - CELL / 2 + 5, centre.y() - 15, CELL - 10, 30)
            painter.setPen(QPen(accent, 1.5))
            painter.setBrush(QColor(COLORS["selected"]))
            painter.drawRoundedRect(box, 4, 4)
            painter.setPen(QColor(COLORS["text"]))
            painter.setFont(_font(MONO_FAMILY, 11 if len(label) <= 6 else 9))
            text = painter.fontMetrics().elidedText(label, Qt.TextElideMode.ElideRight, int(box.width()) - 4)
            painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)
            painter.setFont(_font(MONO_FAMILY, 12))

    # Mouse -------------------------------------------------------------------------------------------
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        cell = self.cell_at(event.position())
        if cell != self._hover:
            self._hover = cell
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._hover = None
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        cell = self.cell_at(event.position())
        if cell is None:
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.remove_requested.emit(*cell)
        elif event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(*cell)
