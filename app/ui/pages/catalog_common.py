"""Pills and filters shared by the Dataset Hub and the Model Zoo: a licence category or an
install status always reads the same way, as a tinted pill with words."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QWidget

from core.common.licensing import CATEGORY_LABELS, LicenseCategory, LicenseInfo

from ..widgets import Pill

__all__ = ["LICENCE_TONES", "STATUS_WORDS", "FilterRow", "licence_pill", "status_pill"]

LICENCE_TONES = {
    LicenseCategory.OPEN_SOURCE: "ok", LicenseCategory.RESEARCH_ONLY: "warn",
    LicenseCategory.NON_COMMERCIAL: "warn", LicenseCategory.COMMERCIAL: "info",
    LicenseCategory.GATED: "fail", LicenseCategory.PROPRIETARY: "fail",
    LicenseCategory.UNSPECIFIED: "planned",
}
#: Install status → (words, tone).
STATUS_WORDS = {
    "ready": ("Ready", "ok"), "available": ("Available", "ok"), "not_downloaded": ("Not downloaded", "warn"),
    "not_installed": ("Not installed", "warn"), "not_connected": ("Not connected", "planned"),
    "experimental": ("Experimental", "experimental"), "planned": ("Planned", "planned"),
    "catalog_only": ("Catalogue only", "planned"),
}


def licence_pill(licence: LicenseInfo) -> Pill:
    pill = Pill(licence.label, LICENCE_TONES.get(licence.category, "planned"))
    pill.setToolTip(licence.name)
    return pill


def status_pill(status: str, planned_for: str | None = None) -> Pill:
    words, tone = STATUS_WORDS.get(status, (status.replace("_", " ").capitalize(), "planned"))
    if status == "planned" and planned_for:
        words = f"Planned · {planned_for}"
    return Pill(words, tone)


class FilterRow(QWidget):
    """Search text, a first choice (modality, framework…) and a licence category."""

    def __init__(self, placeholder: str, first: Iterable[tuple[str, object]],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText(placeholder)
        self.search.setClearButtonEnabled(True)
        self.first = QComboBox()
        for title, data in first:
            self.first.addItem(title, data)
        self.licence = QComboBox()
        self.licence.addItem("Any licence", None)
        for category, words in CATEGORY_LABELS.items():
            self.licence.addItem(words, category)
        row.addWidget(self.search, 1)
        row.addWidget(self.first)
        row.addWidget(self.licence)

    def matches(self, text: str, first: object, category: LicenseCategory) -> bool:
        wanted = self.search.text().strip().lower()
        return ((not wanted or wanted in text.lower())
                and self.first.currentData() in (None, first)
                and self.licence.currentData() in (None, category))

    def connect(self, slot) -> None:  # noqa: ANN001 (any callable)
        self.search.textChanged.connect(lambda _: slot())
        self.first.currentIndexChanged.connect(lambda _: slot())
        self.licence.currentIndexChanged.connect(lambda _: slot())
