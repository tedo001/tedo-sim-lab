"""Synthetic scenes: shapes on textured backgrounds with exact boxes and outlines — a detection or
segmentation dataset you can make as large as you like, the same again from the same seed. A
dataset is written to ``datasets/synthetic/`` as PNG images and a COCO ``annotations.json``,
ready for the Dataset Hub's COCO inspector."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from labs.simulation.vision import CATEGORIES, SceneSpec, generate_dataset, render_scene

from ....services.context import AppContext
from ...icons import icon
from ...widgets import Card, PathLabel, label
from ...widgets.sim_views import ImageView

__all__ = ["SceneLab"]

PREVIEWS = 6


def _int_spin(low: int, high: int, value: int, prefix: str = "") -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(low, high)
    spin.setValue(value)
    if prefix:
        spin.setPrefix(prefix)
    return spin


class SceneLab(QWidget):
    kind = "scenes"
    message = Signal(str)

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.dataset: Path | None = None
        self._job: str | None = None
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()

        scene = Card("Scene", "every image follows from the seed")
        form = QFormLayout()
        self.width_spin = _int_spin(16, 2048, 160)
        self.height_spin = _int_spin(16, 2048, 120)
        self.min_objects = _int_spin(0, 50, 1)
        self.max_objects = _int_spin(0, 50, 4)
        self.min_size = _int_spin(4, 2048, 14)
        self.max_size = _int_spin(4, 2048, 44)
        self.shape_boxes = {name: QCheckBox(name) for name in CATEGORIES}
        shapes = QHBoxLayout()
        for box_widget in self.shape_boxes.values():
            box_widget.setChecked(True)
            shapes.addWidget(box_widget)
        self.noise = QDoubleSpinBox()
        self.noise.setRange(0, 0.5)
        self.noise.setSingleStep(0.01)
        self.noise.setValue(0.04)
        self.background = QComboBox()
        self.background.addItems(["gradient", "plain", "texture"])
        self.overlap = QCheckBox("Shapes may cover each other")
        self.overlap.setChecked(True)
        self.overlap.setToolTip("Annotations then describe the visible part of each shape; "
                                "unticked, shapes are placed apart")
        self.seed = _int_spin(0, 999_999, 0)
        seed_row = QHBoxLayout()
        seed_row.addWidget(self.seed, 1)
        dice = QPushButton(icon("dices"), "New seed")
        dice.clicked.connect(lambda: self.seed.setValue(int(np.random.default_rng().integers(0, 999_999))))
        seed_row.addWidget(dice)
        for title, widget in (("Width", self.width_spin), ("Height", self.height_spin),
                              ("Objects (min)", self.min_objects), ("Objects (max)", self.max_objects),
                              ("Size (min px)", self.min_size), ("Size (max px)", self.max_size)):
            form.addRow(title, widget)
        form.addRow("Shapes", shapes)
        form.addRow("Noise σ", self.noise)
        form.addRow("Background", self.background)
        form.addRow("Overlap", self.overlap)
        form.addRow("Seed", seed_row)
        scene.add(form)
        top.addWidget(scene, 0, Qt.AlignmentFlag.AlignTop)

        preview = Card("Preview", "the first images of the dataset, with their boxes")
        grid = QGridLayout()
        self.previews = [ImageView() for _ in range(PREVIEWS)]
        for index, view in enumerate(self.previews):
            view.setMinimumSize(150, 120)
            grid.addWidget(view, index // 3, index % 3)
        preview.add(grid)
        self.stats = label("", "CardCaption", wrap=True)
        preview.add(self.stats)
        top.addWidget(preview, 1, Qt.AlignmentFlag.AlignTop)
        box.addLayout(top)

        dataset = Card("Make a dataset", "PNG images and COCO annotations in the workspace")
        row = QHBoxLayout()
        self.count = _int_spin(1, 100_000, 200, "images ")
        self.name = QLineEdit("synthetic shapes")
        self.name.textChanged.connect(lambda _t: self._show_target())
        self.generate_button = QPushButton(icon("database"), "Make the dataset")
        self.generate_button.setObjectName("Primary")
        self.generate_button.clicked.connect(self.generate)
        self.inspect_button = QPushButton(icon("scan-eye"), "Inspect in the Dataset Hub")
        self.inspect_button.setEnabled(False)
        self.inspect_button.clicked.connect(self.inspect)
        row.addWidget(self.count)
        row.addWidget(self.name, 1)
        row.addWidget(self.generate_button)
        row.addWidget(self.inspect_button)
        dataset.add(row)
        self.target = PathLabel("")
        dataset.add(self.target)
        self.progress = label("", "CardCaption", wrap=True)
        dataset.add(self.progress)
        box.addWidget(dataset)
        box.addStretch(1)
        for widget in (self.width_spin, self.height_spin, self.min_objects, self.max_objects, self.min_size,
                       self.max_size, self.seed):
            widget.valueChanged.connect(lambda _v: self.refresh())
        self.noise.valueChanged.connect(lambda _v: self.refresh())
        self.background.currentIndexChanged.connect(lambda _i: self.refresh())
        for box_widget in (*self.shape_boxes.values(), self.overlap):
            box_widget.toggled.connect(lambda _on: self.refresh())
        ctx.jobs.job_finished.connect(self._generated)
        ctx.jobs.job_progress.connect(self._progress)
        self._show_target()
        self.refresh()

    def spec(self) -> SceneSpec:
        shapes = tuple(name for name, widget in self.shape_boxes.items() if widget.isChecked())
        return SceneSpec(self.width_spin.value(), self.height_spin.value(), self.min_objects.value(),
                         self.max_objects.value(), self.min_size.value(), self.max_size.value(), shapes,
                         self.noise.value(), self.background.currentText(), self.seed.value(),
                         overlap=self.overlap.isChecked())

    def refresh(self) -> None:
        spec = self.spec()
        try:
            spec.check()
        except ValueError as exc:
            self.stats.setText(str(exc))
            self.generate_button.setEnabled(self._job is not None)
            return
        self.generate_button.setEnabled(True)
        rng = np.random.default_rng(spec.seed)
        total = 0
        for view in self.previews:
            image, objects = render_scene(spec, rng)
            total += len(objects)
            view.set_image(image, [(*o["bbox"], o["category"][0]) for o in objects])
        self.stats.setText(f"{total} objects in these {PREVIEWS} images · boxes from the exact shapes")

    def _folder(self) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", self.name.text().strip()).strip("-").lower() or "synthetic"
        return self.ctx.paths.datasets / "synthetic" / slug

    def _show_target(self) -> None:
        self.target.set_full_text(str(self._folder()))

    def generate(self) -> None:
        """Make the dataset in the background; while it runs, the same button stops it."""
        if self._job is not None:
            self.ctx.jobs.cancel(self._job)
            return
        folder = self._folder()
        if folder.exists() and any(folder.iterdir()):
            self.message.emit(f"{folder} already holds files: choose another name.")
            return
        spec, count, name = self.spec(), self.count.value(), self.name.text().strip() or "synthetic shapes"
        self.progress.setText(f"Making {count} images…")
        self._job = self.ctx.jobs.submit_task(
            lambda cancel, progress: generate_dataset(spec, count, folder, name, progress=progress,
                                                      cancel=cancel),
            title=f"Make the dataset {name}")
        self.generate_button.setText("Stop")
        self.generate_button.setIcon(icon("square"))

    def _progress(self, job_id: str, fraction: float, text: str) -> None:
        if job_id == self._job:
            self.progress.setText(f"{text} ({fraction:.0%})")

    def _generated(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        self.generate_button.setText("Make the dataset")
        self.generate_button.setIcon(icon("database"))
        self.refresh()
        job = self.ctx.jobs.job(job_id)
        if status != "completed":
            self.progress.setText("Stopped; nothing was kept." if status == "cancelled"
                                  else f"Failed: {job.error or status}")
            return
        self.dataset = Path(job.result)
        self.progress.setText(f"Done: {self.dataset}")
        self.inspect_button.setEnabled(True)
        self.message.emit(f"Made the dataset in {self.dataset.parent}")

    def inspect(self) -> None:
        if self.dataset is None:
            return
        hub = self.ctx.show_page("dataset_hub")
        coco = hub.show_view("coco")
        coco.set_images_dir(self.dataset.parent)
        coco.open_file(self.dataset)

    def settings(self) -> dict:
        return {"spec": self.spec().to_dict(), "count": self.count.value(), "name": self.name.text()}

    def load(self, settings: dict) -> None:
        spec = SceneSpec.from_dict(settings.get("spec", {}))
        values = ((self.width_spin, spec.width), (self.height_spin, spec.height),
                  (self.min_objects, spec.min_objects), (self.max_objects, spec.max_objects),
                  (self.min_size, spec.min_size), (self.max_size, spec.max_size),
                  (self.seed, spec.seed), (self.count, int(settings.get("count", 200))))
        for widget, value in values:
            widget.setValue(value)
        self.noise.setValue(spec.noise)
        self.background.setCurrentText(spec.background)
        self.overlap.setChecked(spec.overlap)
        for name, widget in self.shape_boxes.items():
            widget.setChecked(name in spec.shapes)
        self.name.setText(settings.get("name", self.name.text()))
        self.refresh()

    def outputs(self) -> dict[str, bytes | str]:
        from labs.simulation.vision.png import encode_png

        image, _ = render_scene(self.spec(), np.random.default_rng(self.spec().seed))
        return {"preview.png": encode_png(image)}

    def default_name(self) -> str:
        return self.name.text().strip() or "Synthetic scenes"
