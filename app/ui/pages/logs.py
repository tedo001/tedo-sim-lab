"""Logs: what the lab records, as it happens. Read app.log, warnings and errors (errors.log), an
older rotation or a run's run.log; filter by level, source and text; follow new lines; see a
record whole (double-click a traceback line to open the file at that line); change what is
recorded (the log level) without restarting."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QGuiApplication, QHideEvent, QMouseEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableView,
    QWidget,
)

from core.common.log_reader import LogFollower, LogRecord, matches, parse_lines

from ...services.context import AppContext
from ...services.project import TRACEBACK_LINE
from ..icons import icon
from ..theme import mono_font
from ..theme.tokens import COLORS
from ..widgets import Page, label

__all__ = ["LogsPage"]

MAX_RECORDS = 20_000
LEVEL_FILTERS = (("All levels", 0), ("Info and above", logging.INFO),
                 ("Warnings and errors", logging.WARNING), ("Errors", logging.ERROR))
_COLOURS = {"DEBUG": "text_faint", "INFO": "text_dim", "WARNING": "warn", "ERROR": "fail", "CRITICAL": "fail"}


class LogModel(QAbstractTableModel):
    HEADERS = ("Time", "Level", "Source", "Message")

    def __init__(self) -> None:
        super().__init__()
        self.records: list[LogRecord] = []
        self.shown: list[int] = []
        self.level, self.logger, self.text = 0, "", ""
        self._font = mono_font()

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if parent.isValid() else len(self.shown)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = 0):  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = 0):
        record = self.record(index.row())
        column = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            first = record.message + (" …" if record.details else "")
            return (record.time[11:] or record.time, record.level, record.logger, first)[column]
        if role == Qt.ItemDataRole.FontRole:
            return self._font
        if role == Qt.ItemDataRole.ForegroundRole and column == 1:
            return QColor(COLORS[_COLOURS.get(record.level, "text_dim")])
        if role == Qt.ItemDataRole.ToolTipRole and column == 0:
            return record.time
        return None

    def record(self, row: int) -> LogRecord:
        return self.records[self.shown[row]]

    def _keeps(self, record: LogRecord) -> bool:
        return matches(record, level=self.level, logger=self.logger, text=self.text)

    def set_filter(self, level: int, logger: str, text: str) -> None:
        self.beginResetModel()
        self.level, self.logger, self.text = level, logger, text.strip()
        self.shown = [i for i, record in enumerate(self.records) if self._keeps(record)]
        self.endResetModel()

    def replace(self, records: list[LogRecord]) -> None:
        self.beginResetModel()
        self.records = records[-MAX_RECORDS:]
        self.shown = [i for i, record in enumerate(self.records) if self._keeps(record)]
        self.endResetModel()

    def extend(self, records: list[LogRecord]) -> None:
        if len(self.records) + len(records) > MAX_RECORDS:
            self.replace(self.records + records)  # drop the oldest
            return
        start = len(self.records)
        self.records.extend(records)
        added = [start + i for i, record in enumerate(records) if self._keeps(record)]
        if added:
            self.beginInsertRows(QModelIndex(), len(self.shown), len(self.shown) + len(added) - 1)
            self.shown.extend(added)
            self.endInsertRows()

    def loggers(self) -> list[str]:
        return sorted({r.logger for r in self.records if r.logger})


class _Details(QPlainTextEdit):
    """The selected record whole; double-click a traceback line to open that file."""

    def __init__(self, open_file) -> None:  # noqa: ANN001 (callable)
        super().__init__()
        self.setReadOnly(True)
        self.setFont(mono_font())
        self.setObjectName("Code")
        self.open_file = open_file

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        line = self.cursorForPosition(event.position().toPoint()).block().text()
        match = TRACEBACK_LINE.search(line)
        if match and Path(match["path"]).is_file():
            self.open_file(Path(match["path"]), int(match["line"]))
            return
        super().mouseDoubleClickEvent(event)


class LogsPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Logs", "what the lab records; secrets are masked before anything is written",
                         parent)
        self.ctx = ctx
        self.follower: LogFollower | None = None
        self.timer = QTimer(self, interval=700)
        self.timer.timeout.connect(self.poll)
        folder = QPushButton(icon("folder-open"), "Open folder")
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ctx.paths.logs))))
        copy = QPushButton(icon("copy"), "Copy shown")
        copy.clicked.connect(self.copy_shown)
        self.head.add_action(folder)
        self.head.add_action(copy)

        sources = QHBoxLayout()
        self.source = QComboBox()
        self.source.setMinimumWidth(160)  # long run names shorten in a narrow pane
        self.source.activated.connect(lambda _i: self.open_source(self.source.currentData()))
        self.level_box = QComboBox()
        self.level_box.setToolTip("What the lab records from now on (saved in settings.yaml)")
        for name in ("DEBUG", "INFO", "WARNING", "ERROR"):
            self.level_box.addItem(f"Record {name.lower()} and above", name)
        self.level_box.setCurrentIndex(self.level_box.findData(ctx.config.log_level))
        self.level_box.activated.connect(lambda _i: self.set_log_level(self.level_box.currentData()))
        ctx.settings.changed.connect(self._settings_changed)
        sources.addWidget(label("Log", "KvKey"))
        sources.addWidget(self.source, 1)
        sources.addWidget(self.level_box)
        self.body.addLayout(sources)

        filters = QHBoxLayout()
        self.level_filter = QComboBox()
        self.level_filter.setMinimumWidth(130)
        for title, value in LEVEL_FILTERS:
            self.level_filter.addItem(title, value)
        self.logger_filter = QComboBox()
        self.logger_filter.setMinimumWidth(120)
        self.logger_filter.addItem("All sources", "")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search the records")
        self.search.setMinimumWidth(110)
        self.search.setClearButtonEnabled(True)
        self.follow = QCheckBox("Follow new lines")
        self.follow.setChecked(True)
        for widget in (self.level_filter, self.logger_filter):
            widget.currentIndexChanged.connect(lambda _i: self.apply_filter())
        self.search.textChanged.connect(lambda _t: self.apply_filter())
        filters.addWidget(self.level_filter)
        filters.addWidget(self.logger_filter)
        filters.addWidget(self.search, 1)
        filters.addWidget(self.follow)
        self.body.addLayout(filters)

        self.model = LogModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setObjectName("LogTable")
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for column in range(3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.selectionModel().currentRowChanged.connect(
            lambda current, _previous: self.show_record(current.row()))
        self.details = _Details(ctx.open_file)
        self.details.setPlaceholderText("Choose a record to see it whole.")
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(self.details)
        splitter.setSizes([440, 160])
        splitter.setMinimumHeight(560)
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.body.addWidget(splitter, 1)
        self.summary = label("", "CardCaption", wrap=True)
        self.body.addWidget(self.summary)
        self.fill_sources()
        self.open_source(ctx.logs.app_log)

    # Sources -----------------------------------------------------------------------------------
    def fill_sources(self) -> None:
        current = self.source.currentData()
        self.source.clear()
        logs = self.ctx.logs
        self.source.addItem("App log · app.log", logs.app_log)
        self.source.addItem("Warnings and errors · errors.log", logs.errors_log)
        for path in logs.files():
            if path not in (logs.app_log, logs.errors_log):
                self.source.addItem(f"Older · {path.name}", path)
        for run in self.ctx.store.runs_with_experiments(limit=30):
            run_log = self.ctx.store.resolve_path(run["run_dir"]) / "run.log"
            if run_log.is_file():
                self.source.addItem(f"Run · {run['experiment_name']} · {run['id'][:8]}", run_log)
        index = self.source.findData(current) if current is not None else 0
        self.source.setCurrentIndex(max(index, 0))

    def open_source(self, path: Path) -> None:
        index = self.source.findData(path)
        if index >= 0:
            self.source.setCurrentIndex(index)
        self.follower = LogFollower(path)
        self.model.replace(parse_lines(self.follower.read()))
        self._refresh_loggers()
        self.details.clear()
        self._summarise()
        if self.follow.isChecked():
            self.table.scrollToBottom()

    def show_problems(self) -> None:
        """Warnings and errors, newest at the bottom (the top bar's problem count leads here)."""
        self.open_source(self.ctx.logs.app_log)
        self.level_filter.setCurrentIndex(self.level_filter.findData(logging.WARNING))
        self.table.scrollToBottom()

    # Reading -----------------------------------------------------------------------------------
    def poll(self) -> None:
        if self.follower is None:
            return
        lines = self.follower.read()
        if not lines:
            return
        previous = self.model.records[-1] if self.model.records else None
        records = parse_lines(lines, previous)
        self.model.extend(records)
        if previous is not None and self.table.currentIndex().isValid():
            self.show_record(self.table.currentIndex().row())  # its details may have grown
        if any(r.logger and self.logger_filter.findData(r.logger) < 0 for r in records):
            self._refresh_loggers()
        self._summarise()
        if self.follow.isChecked():
            self.table.scrollToBottom()

    def apply_filter(self) -> None:
        self.model.set_filter(self.level_filter.currentData() or 0, self.logger_filter.currentData() or "",
                              self.search.text())
        self.details.clear()
        self._summarise()
        if self.follow.isChecked():
            self.table.scrollToBottom()

    def _refresh_loggers(self) -> None:
        current = self.logger_filter.currentData()
        self.logger_filter.blockSignals(True)
        self.logger_filter.clear()
        self.logger_filter.addItem("All sources", "")
        for name in self.model.loggers():
            self.logger_filter.addItem(name, name)
        self.logger_filter.setCurrentIndex(max(self.logger_filter.findData(current), 0))
        self.logger_filter.blockSignals(False)

    def show_record(self, row: int) -> None:
        if 0 <= row < len(self.model.shown):
            self.details.setPlainText(self.model.record(row).text())

    def _summarise(self) -> None:
        records = self.model.records
        errors = sum(1 for r in records if r.levelno >= logging.ERROR)
        warnings = sum(1 for r in records if r.levelno == logging.WARNING)
        path = self.follower.path if self.follower else None
        where = f" · {path}" if path is not None and path.exists() else " · nothing written yet"
        self.summary.setText(f"{len(self.model.shown):,} of {len(records):,} records shown · {errors} "
                             f"error{'s' if errors != 1 else ''} · {warnings} warning"
                             f"{'s' if warnings != 1 else ''}{where}")

    def copy_shown(self) -> str:
        text = "\n".join(self.model.record(row).text() for row in range(len(self.model.shown)))
        QGuiApplication.clipboard().setText(text)
        return text

    def set_log_level(self, level: str) -> None:
        self.ctx.settings.update(log_level=level)
        self.level_box.setCurrentIndex(self.level_box.findData(level))

    def _settings_changed(self, names: set[str]) -> None:
        if "log_level" in names:
            self.level_box.setCurrentIndex(self.level_box.findData(self.ctx.config.log_level))

    # Visibility --------------------------------------------------------------------------------
    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self.fill_sources()  # new runs, rotated files
        self.poll()
        self.timer.start()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        super().hideEvent(event)
        self.timer.stop()
