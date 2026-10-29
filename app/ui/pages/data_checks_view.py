"""The data checks as a table (check · result · what was found), shared by the Training page
(a run's ``data_validation.json``), the Dataset Hub's *Validate* and the COCO inspector."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtWidgets import QWidget

from ..widgets import DataTable, Pill

__all__ = ["ChecksTable", "TONES"]

TONES = {"ok": ("OK", "ok"), "warn": ("Warning", "warn"), "error": ("Error", "fail")}


class ChecksTable(DataTable):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(("Check", "Result", "Found"), stretch_column=2, parent=parent)
        self.setWordWrap(True)

    def show_checks(self, checks: Sequence[dict[str, Any]]) -> None:
        self.clear_rows()
        order = {"error": 0, "warn": 1, "ok": 2}
        for check in sorted(checks, key=lambda c: order.get(c["status"], 3)):
            words, tone = TONES.get(check["status"], (check["status"], "planned"))
            row = self.add_row((check["title"], Pill(words, tone), check["detail"]))
            self.item(row, 2).setToolTip(check["detail"])
