"""The Image lab: an image (a generated test pattern or a file of yours) through a pipeline of
operations you build — blur, edges, thresholds, morphology, noise, geometry, the Fourier domain —
each with live settings, before and after side by side, with the result's histogram. Pipelines run
in the background, so large images never freeze the window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from labs.simulation.vision import OPERATIONS, SAMPLES, Step, run_pipeline, sample
from labs.simulation.vision.png import encode_png

from ....services.jobs import JobQueue
from ...icons import icon
from ...widgets import Card, label
from ...widgets.sim_views import ImageView, LineChart
from .param_form import ParamForm

__all__ = ["ImageLab", "from_qimage"]

MAX_SIDE = 1024


def from_qimage(image: QImage) -> np.ndarray:
    """A QImage as float RGB in [0, 1] (shrunk to at most MAX_SIDE pixels on its long side)."""
    if max(image.width(), image.height()) > MAX_SIDE:
        image = image.scaled(MAX_SIDE, MAX_SIDE, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height, stride = image.width(), image.height(), image.bytesPerLine()
    buffer = np.frombuffer(image.constBits(), dtype=np.uint8, count=stride * height).reshape(height, stride)
    return (buffer[:, :width * 3].reshape(height, width, 3).astype(np.float32) / 255.0).copy()


class ImageLab(QWidget):
    kind = "image"
    message = Signal(str)

    def __init__(self, jobs: JobQueue, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.jobs = jobs
        self.source: dict = {"sample": "Shapes", "size": 256}
        self.image = sample("Shapes")
        self.result: np.ndarray | None = None
        self.steps: list[Step] = [Step("gaussian", {"sigma": 1.5}), Step("canny")]
        self._job: str | None = None
        self._generation = 0
        self._timer = QTimer(self, singleShot=True, interval=120)
        self._timer.timeout.connect(self.process)
        jobs.job_finished.connect(self._processed)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()

        source = Card("Image", "a generated test pattern or a file of yours")
        row = QHBoxLayout()
        self.source_box = QComboBox()
        self.source_box.addItems(list(SAMPLES))
        self.source_box.insertSeparator(len(SAMPLES))
        self.source_box.addItem("Open an image file…")
        self.source_box.activated.connect(self._source_chosen)
        self.size_box = QComboBox()
        for size in (128, 256, 512):
            self.size_box.addItem(f"{size} px", size)
        self.size_box.setCurrentIndex(1)
        self.size_box.activated.connect(lambda _i: self._source_chosen(self.source_box.currentIndex()))
        self.chain_button = QPushButton(icon("arrow-up"), "Result → input")
        self.chain_button.setToolTip("Start again from the current result")
        self.chain_button.clicked.connect(self.use_result)
        row.addWidget(self.source_box, 1)
        row.addWidget(self.size_box)
        row.addWidget(self.chain_button)
        source.add(row)
        self.source_label = label("", "CardCaption", wrap=True)
        source.add(self.source_label)

        pipeline = Card("Pipeline", "top to bottom; untick a step to skip it")
        self.list = QListWidget()
        self.list.setMinimumHeight(150)
        self.list.currentRowChanged.connect(self._select)
        self.list.itemChanged.connect(self._ticked)
        pipeline.add(self.list)
        row = QHBoxLayout()
        self.op_box = QComboBox()
        for op in sorted(OPERATIONS.values(), key=lambda o: (o.group, o.title)):
            self.op_box.addItem(f"{op.group}: {op.title}", op.id)
        add = QPushButton(icon("plus"), "Add")
        add.clicked.connect(lambda: self.add_step(self.op_box.currentData()))
        row.addWidget(self.op_box, 1)
        row.addWidget(add)
        pipeline.add(row)
        row = QHBoxLayout()
        for text, name, slot in (("Up", "arrow-up", lambda: self.move(-1)),
                                 ("Down", "arrow-down", lambda: self.move(1)),
                                 ("Remove", "trash-2", self.remove_step),
                                 ("Clear", "eraser", self.clear_steps)):
            button = QPushButton(icon(name), text)
            button.clicked.connect(slot)
            row.addWidget(button)
        pipeline.add(row)
        self.form = ParamForm()
        self.form.changed.connect(self._params_changed)
        pipeline.add(self.form)
        top.addWidget(source, 1, Qt.AlignmentFlag.AlignTop)
        top.addWidget(pipeline, 1, Qt.AlignmentFlag.AlignTop)
        box.addLayout(top)

        views = Card("Before and after", "")
        row = QHBoxLayout()
        self.before = ImageView("input")
        self.after = ImageView("result")
        row.addWidget(self.before)
        row.addWidget(self.after)
        views.add(row)
        self.timing = label("", "CardCaption", wrap=True)
        views.add(self.timing)
        self.histogram = LineChart("Histogram of the result", "intensity", "pixels")
        views.add(self.histogram)
        box.addWidget(views)
        box.addStretch(1)
        self._show_steps()
        self._set_image(self.image, "Generated test pattern: Shapes (256 px)")

    # The source ------------------------------------------------------------------------------------
    def _source_chosen(self, index: int) -> None:
        text = self.source_box.itemText(index)
        if text in SAMPLES:
            size = int(self.size_box.currentData())
            self.source = {"sample": text, "size": size}
            self._set_image(sample(text, size), f"Generated test pattern: {text} ({size} px)")
        elif text.startswith("Open"):
            kinds = "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp)"
            chosen, _ = QFileDialog.getOpenFileName(self, "Open an image", str(Path.home()), kinds)
            if chosen:
                self.open_file(Path(chosen))

    def open_file(self, path: Path) -> bool:
        image = QImage(str(path))
        if image.isNull():
            self.message.emit(f"{path.name} could not be read as an image")
            return False
        self.source = {"file": str(path)}
        self._set_image(from_qimage(image), f"{path} ({image.width()} × {image.height()} px)")
        return True

    def use_result(self) -> None:
        if self.result is not None:
            self.source = {"result_of": [s.to_dict() for s in self.steps]}
            self._set_image(self.result.copy(), "The previous result")

    def _set_image(self, image: np.ndarray, caption: str) -> None:
        self.image = image
        self.before.set_image(image)
        self.source_label.setText(caption)
        self.schedule()

    # The pipeline -------------------------------------------------------------------------------------
    def _show_steps(self, select: int | None = None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for number, step in enumerate(self.steps, 1):
            op = OPERATIONS[step.op]
            settings = ", ".join(f"{k} {v}" for k, v in step.params.items() if not isinstance(v, tuple))
            item = QListWidgetItem(f"{number}. {op.title}" + (f"  ({settings})" if settings else ""))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if step.enabled else Qt.CheckState.Unchecked)
            item.setToolTip(op.help)
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self.steps:
            self.list.setCurrentRow(min(select if select is not None else 0, len(self.steps) - 1))
        else:
            self.form.set_params((), {})

    def _select(self, row: int) -> None:
        if 0 <= row < len(self.steps):
            step = self.steps[row]
            self.form.set_params(OPERATIONS[step.op].params, step.params)

    def _ticked(self, item: QListWidgetItem) -> None:
        row = self.list.row(item)
        if 0 <= row < len(self.steps):
            self.steps[row].enabled = item.checkState() == Qt.CheckState.Checked
            self.schedule()

    def _params_changed(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self.steps):
            self.steps[row].params = self.form.values()
            self._show_steps(row)
            self.schedule()

    def add_step(self, op: str) -> None:
        self.steps.append(Step(op, OPERATIONS[op].defaults()))
        self._show_steps(len(self.steps) - 1)
        self.schedule()

    def remove_step(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self.steps):
            del self.steps[row]
            self._show_steps(row)
            self.schedule()

    def clear_steps(self) -> None:
        self.steps = []
        self._show_steps()
        self.schedule()

    def move(self, offset: int) -> None:
        row = self.list.currentRow()
        other = row + offset
        if 0 <= row < len(self.steps) and 0 <= other < len(self.steps):
            self.steps[row], self.steps[other] = self.steps[other], self.steps[row]
            self._show_steps(other)
            self.schedule()

    # Processing -------------------------------------------------------------------------------------------
    def schedule(self) -> None:
        self._timer.start()

    def process(self) -> None:
        self._generation += 1
        generation, image = self._generation, self.image.copy()
        steps = [Step(s.op, dict(s.params), s.enabled) for s in self.steps]
        self._job = self.jobs.submit_task(lambda cancel, progress: (generation, *run_pipeline(image, steps)),
                                          title="Run an image pipeline", quiet=True)

    def _processed(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        job = self.jobs.job(job_id)
        if status != "completed":
            self.message.emit(f"The pipeline failed: {job.error}")
            return
        generation, result, timings = job.result
        if generation != self._generation:
            return  # a newer run is on its way
        self.result = result
        self.after.set_image(result)
        total = sum(timings)
        self.timing.setText(f"{result.shape[1]} × {result.shape[0]} px, "
                            f"{'colour' if result.ndim == 3 else 'grey'} · {total:.1f} ms in total")
        counts, edges = np.histogram(result, bins=64, range=(0.0, 1.0))
        self.histogram.set_line((edges[:-1] + edges[1:]) / 2, counts.astype(float), fill=True,
                                x_range=(0.0, 1.0))

    # Saving -----------------------------------------------------------------------------------------------
    def settings(self) -> dict:
        return {"source": self.source, "pipeline": [step.to_dict() for step in self.steps]}

    def load(self, settings: dict) -> None:
        self.steps = [Step.from_dict(item) for item in settings.get("pipeline", [])]
        source = settings.get("source", {})
        if "file" in source and Path(source["file"]).is_file():
            self.open_file(Path(source["file"]))
        else:
            name = source.get("sample", "Shapes") if source.get("sample") in SAMPLES else "Shapes"
            size = int(source.get("size", 256))
            self.source = {"sample": name, "size": size}
            self._set_image(sample(name, size), f"Generated test pattern: {name} ({size} px)")
        self._show_steps()

    def outputs(self) -> dict[str, bytes | str]:
        return {} if self.result is None else {"result.png": encode_png(self.result)}

    def default_name(self) -> str:
        return " → ".join(OPERATIONS[s.op].title for s in self.steps[:3]) or "Image"
