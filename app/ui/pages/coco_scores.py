"""Under the COCO inspector: the checks of the open annotation file, and scores of a model's
detections against it (COCO results JSON: ``image_id``, ``category_id``, ``bbox`` or
``segmentation``, ``score``): mAP over IoU 0.50:0.95, AP50, AP75, mean IoU of the matches and AP
per class, box or mask. Scoring runs in the background; the scores can be saved as JSON."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from labs.computer_vision.coco import CocoFile
from labs.computer_vision.detection_metrics import check_annotations, coco_map

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, KeyValues, label
from ..widgets.plots import BarList
from .data_checks_view import ChecksTable

__all__ = ["CocoScores", "as_json"]


def as_json(coco: CocoFile) -> dict[str, Any]:
    """The annotation file again as COCO JSON (what the checks and the scorer read)."""
    return {"images": list(coco.images.values()), "categories": list(coco.categories.values()),
            "licenses": list(coco.licences.values()),
            "annotations": [a for items in coco.by_image.values() for a in items]}


class CocoScores(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.truth: dict[str, Any] | None = None
        self.scores: dict[str, Any] | None = None
        self._job: str | None = None
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(12)
        checks = Card("Annotation checks", "what would make scores against this file wrong")
        self.checks = ChecksTable()
        checks.add(self.checks)
        column.addWidget(checks)

        score = Card("Score detections", "COCO mean average precision, box or mask")
        row = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItem("Boxes (bbox)", "bbox")
        self.kind.addItem("Masks (segm)", "segm")
        self.kind.setMinimumWidth(140)
        self.open_button = QPushButton(icon("file-json"), "Open detections…")
        self.open_button.setToolTip("A COCO results file: a list of {image_id, category_id, bbox or "
                                    "segmentation, score}")
        self.open_button.clicked.connect(self._ask)
        self.save_button = QPushButton(icon("save"), "Save scores")
        self.save_button.clicked.connect(lambda: self.save())
        self.save_button.setEnabled(False)
        for widget in (self.kind, self.open_button, self.save_button):
            row.addWidget(widget)
        row.addStretch(1)
        score.add(row)
        self.values = KeyValues((("mAP 0.50:0.95", "—"), ("AP50", "—"), ("AP75", "—"),
                                 ("Mean IoU of matches", "—"), ("Detections", "—")))
        score.add(self.values)
        self.per_class = BarList()
        score.add(self.per_class)
        self.note = label("Scored the way the COCO evaluation does: detections matched greedily by score, "
                          "crowd regions ignored, at most 100 per image and class, precision at 101 recall "
                          "points.", "CardCaption", wrap=True)
        score.add(self.note)
        column.addWidget(score)
        ctx.jobs.job_finished.connect(self._done)

    def set_file(self, coco: CocoFile) -> None:
        self.truth = as_json(coco)
        self.checks.show_checks(check_annotations(self.truth))
        self.scores = None
        self.save_button.setEnabled(False)

    def _ask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open detections", str(self.ctx.paths.results),
                                              "COCO results (*.json)")
        if path:
            self.score_file(Path(path))

    def score_file(self, path: Path) -> str | None:
        """Score the detections in ``path`` in the background; returns the job id."""
        if self.truth is None:
            return None
        truth, kind = self.truth, self.kind.currentData()

        def work(cancel: Any, progress: Any) -> dict[str, Any]:
            predictions = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(predictions, dict):  # some exporters wrap the list
                predictions = predictions.get("annotations", [])
            return {**coco_map(truth, predictions, kind=kind), "detections_file": str(path)}

        self.note.setText(f"Scoring {path.name}…")
        self._job = self.ctx.jobs.submit_task(work, title=f"Score {path.name}")
        return self._job

    def _done(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        job = self.ctx.jobs.job(job_id)
        if status != "completed":
            self.note.setText(f"Could not score them: {job.error}")
            return
        self.scores = scores = job.result
        for key, title in (("map", "mAP 0.50:0.95"), ("map50", "AP50"), ("map75", "AP75"),
                           ("mean_iou_matched", "Mean IoU of matches")):
            self.values.set_value(title, f"{scores[key]:.4f}")
        self.values.set_value("Detections", f"{scores['predictions']:,} on {scores['images']:,} images")
        rows = sorted(((name, v["ap"]) for name, v in scores["per_class"].items()), key=lambda r: -r[1])
        self.per_class.set_rows(rows[:15])
        self.note.setText(scores["method"])
        self.save_button.setEnabled(True)

    def save(self) -> Path | None:
        if self.scores is None:
            return None
        folder = self.ctx.paths.reports / f"coco-scores-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "scores.json"
        target.write_text(json.dumps(self.scores, indent=2), encoding="utf-8")
        self.note.setText(f"Saved to {target}")
        return target
