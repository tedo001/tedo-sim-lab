"""The explanation records of the run on Model Testing: every Grad-CAM, saliency and layer view made
from it, with its input, target class and methods, each checked again against the saved input and the
run's checkpoint (a record whose checkpoint has changed since is marked)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QPushButton, QWidget

from labs.inference.explanations import list_records, verify

from ...icons import icon
from ...widgets import Card, DataTable, Pill, label

__all__ = ["ExplanationRecords"]


class ExplanationRecords(Card):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Explanation records", "input, model version, target and method of each",
                         padded=False, parent=parent)
        self.run_dir: Path | None = None
        self.records: list[dict] = []
        self.open_button = QPushButton(icon("folder-open"), "Open folder")
        self.open_button.clicked.connect(self._open)
        self.add_head_widget(self.open_button)
        self.table = DataTable(("When", "Target", "Input", "Checkpoint", "Traceable"), mono_columns=(0, 3),
                               stretch_column=2)
        self.add(self.table)
        self.note = label("", "CardCaption", wrap=True)
        self.add(self.note)

    def refresh(self, run_dir: Path | None) -> None:
        self.run_dir = run_dir
        self.table.clear_rows()
        self.records = list_records(run_dir) if run_dir else []
        for record in self.records[:50]:
            problems = verify(Path(record["folder"]), run_dir)
            pill = Pill("Yes" if not problems else "Changed", "ok" if not problems else "warn")
            pill.setToolTip("; ".join(problems) or "the input, outputs and checkpoint match the record")
            target = record["target"]
            how = "chosen" if target["how"].startswith("chosen") else "predicted"
            self.table.add_row((record["created_at"][:19].replace("T", " "), f"{target['name']} ({how})",
                                Path(record["input"]["original_path"]).name,
                                record["model"]["checkpoint"]["sha256"][:12], pill))
        self.note.setText(f"{len(self.records)} saved in {run_dir.name}/explanations/"
                          if self.records and run_dir
                          else "Each Look inside is saved here with its input and the checkpoint's hash.")

    def _open(self) -> None:
        if self.run_dir is not None:
            folder = self.run_dir / "explanations"
            folder.mkdir(exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
