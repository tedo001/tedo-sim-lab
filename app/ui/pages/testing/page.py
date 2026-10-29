"""Model Testing: use a model. Pick a finished run or a downloaded pretrained model, then
- Try it: one input (image or drawing, WAV file, text, a row of values) and the answer;
- What-if: vary one feature of a row and watch every class's probability (a model simulation);
- Batch: a folder of files or a CSV file, scored when it carries labels, exported as CSV;
- Visualise: inside an image or spectrogram network, layer by layer, with Grad-CAM and saliency.
Detection, zero-shot classification, captioning and speech recognition use pretrained models.
Everything runs in the inference worker, never in the window's process."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QPushButton, QSpinBox, QWidget

from ....services.context import AppContext
from ...icons import icon
from ...widgets import Card, FlowGrid, Page, Segmented, label
from ...widgets.plots import CurveChart
from .batch import BatchTest
from .inputs import InputPanel
from .outputs import OutputPanel
from .records import ExplanationRecords
from .subjects import Subject, subjects
from .visualise import LayerMaps

__all__ = ["TestingPage"]

TABS = (("try", "Try it"), ("whatif", "What-if"), ("batch", "Batch"), ("visualise", "Visualise"))


class TestingPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Model Testing", "try trained and pretrained models; see inside them", parent)
        self.ctx = ctx
        self.items: list[Subject] = []
        self.subject: Subject | None = None
        self.described: dict[str, Any] = {}
        self.pending: str | None = None
        refresh = QPushButton(icon("refresh-cw"), "Refresh")
        refresh.clicked.connect(self.refresh)
        self.head.add_action(refresh)

        pick = Card("What to test", "a finished run, or a pretrained model downloaded in the Model Zoo")
        row = QHBoxLayout()
        self.chooser = QComboBox()
        self.chooser.setMinimumWidth(240)
        self.chooser.setMaximumWidth(520)
        self.chooser.currentIndexChanged.connect(lambda _: self.choose())
        row.addWidget(self.chooser, 1)
        self.stop = QPushButton(icon("circle-stop"), "Stop")
        self.stop.clicked.connect(self.cancel)
        row.addWidget(self.stop)
        row.addStretch(1)
        pick.add(row)
        self.about = label("", "CardCaption", wrap=True)
        self.worker = label("", "Mono")
        pick.add(self.about)
        pick.add(self.worker)
        self.body.addWidget(pick)
        self.tabs = Segmented(TABS)
        self.tabs.changed.connect(self._tab)
        self.body.addWidget(self.tabs)

        self.try_card = Card("Input")
        self.inputs = InputPanel(ctx)
        self.try_card.add(self.inputs)
        self.go = QPushButton(icon("play"), "Run the model")
        self.go.setObjectName("Primary")
        self.go.clicked.connect(self.try_it)
        self.try_card.add(self.go)
        self.answer_card = Card("Answer")
        self.outputs = OutputPanel()
        self.answer_card.add(self.outputs)

        self.whatif_card = Card("What-if", "the row from Try it, with one feature swept over a range")
        rows = QHBoxLayout()
        self.feature = QComboBox()
        self.feature.setMinimumWidth(160)
        self.low, self.high = QDoubleSpinBox(), QDoubleSpinBox()
        for box, value in ((self.low, 0.0), (self.high, 1.0)):
            box.setRange(-1e9, 1e9)
            box.setDecimals(3)
            box.setValue(value)
            box.setMaximumWidth(130)
        self.steps = QSpinBox()
        self.steps.setRange(5, 201)
        self.steps.setValue(41)
        sweep = QPushButton(icon("play"), "Sweep")
        sweep.clicked.connect(self.sweep)
        for title, widget in (("Feature", self.feature), ("from", self.low), ("to", self.high), ("steps",
                                                                                                 self.steps)):
            rows.addWidget(label(title, "CardCaption"))
            rows.addWidget(widget)
        rows.addWidget(sweep)
        rows.addStretch(1)
        self.whatif_card.add(rows)
        self.whatif_note = label("", "CardCaption", wrap=True)
        self.whatif_card.add(self.whatif_note)
        self.charts_holder = QWidget()
        self.charts_layout = QHBoxLayout(self.charts_holder)
        self.charts_layout.setContentsMargins(0, 0, 0, 0)
        self.whatif_card.add(self.charts_holder)

        self.batch_card = Card("Batch")
        self.batch = BatchTest(ctx, self._send)
        self.batch_card.add(self.batch)

        self.vis_card = Card("Visualise", "what every layer makes of the input from Try it")
        vis_row = QHBoxLayout()
        self.target = QComboBox()
        self.target.setMinimumWidth(160)
        vis_row.addWidget(label("Explain the class", "CardCaption"))
        vis_row.addWidget(self.target)
        self.vis_go = QPushButton(icon("eye"), "Look inside")
        self.vis_go.setObjectName("Primary")
        self.vis_go.clicked.connect(self.visualise)
        vis_row.addWidget(self.vis_go)
        vis_row.addStretch(1)
        self.vis_card.add(vis_row)
        self.vis_note = label("", "CardCaption", wrap=True)
        self.vis_card.add(self.vis_note)
        self.maps = LayerMaps()
        self.vis_card.add(self.maps)
        self.records = ExplanationRecords()
        for widget in (self.try_card, self.answer_card, self.whatif_card, self.batch_card, self.vis_card,
                       self.records):
            self.body.addWidget(widget)
        self.body.addStretch(1)
        ctx.inference.state_changed.connect(lambda _s: self._worker_state())
        ctx.experiments.run_changed.connect(lambda _id: self.refresh())
        self.refresh()
        self._tab("try")

    # Choosing ------------------------------------------------------------------------------------------
    def refresh(self) -> None:
        current = self.subject.key if self.subject else None
        self.items = subjects(self.ctx)
        self.chooser.blockSignals(True)
        self.chooser.clear()
        for item in self.items:
            self.chooser.addItem(item.title, item.key)
        if not self.items:
            self.chooser.addItem("Nothing to test yet: train a model or download a pretrained one", None)
        self.chooser.setCurrentIndex(max(self.chooser.findData(current), 0))
        self.chooser.blockSignals(False)
        self.choose()

    def open_run(self, run_id: str) -> None:
        """Show ``run_id`` (from Training, the labs or the Model Registry)."""
        self.refresh()
        index = self.chooser.findData(f"run:{run_id}")
        if index >= 0:
            self.chooser.setCurrentIndex(index)

    def open_model(self, card_id: str) -> None:
        self.refresh()
        index = self.chooser.findData(f"model:{card_id}")
        if index >= 0:
            self.chooser.setCurrentIndex(index)

    def choose(self) -> None:
        key = self.chooser.currentData()
        self.subject = next((item for item in self.items if item.key == key), None)
        self.described = {}
        self.outputs.clear()
        self.maps.clear()
        self.records.refresh(self.subject.run_dir if self.subject and self.subject.is_run
                             and self.subject.kind in ("image", "audio") else None)
        subject = self.subject
        kind = subject.kind if subject else "image"
        self.inputs.configure(kind, [])
        self.about.setText(f"A {subject.describe}." if subject else "")
        available = {"try": subject is not None, "whatif": kind in ("table", "quantum"),
                     "batch": bool(subject and subject.is_run), "visualise": kind in ("image", "audio")}
        for name, button in self.tabs.buttons.items():
            button.setEnabled(available[name])
        if not available.get(self.tabs.current, False):
            self.tabs.buttons["try"].setChecked(True)
            self._tab("try")
        if subject and subject.is_run:
            self.batch.configure(subject.run_dir, kind, [])
            if kind in ("table", "quantum", "image", "audio", "text"):
                self._describe()
        self._worker_state()
        self._tab(self.tabs.current)

    def _describe(self) -> None:
        """Ask the worker what the run takes (features, classes) to fill the forms."""
        subject = self.subject

        def done(result: dict[str, Any]) -> None:
            if self.subject is not subject:
                return
            self.described = result.get("model", {})
            features = list(self.described.get("features", []))
            if subject.kind in ("table", "quantum"):
                self.inputs.configure(subject.kind, features)
                self.feature.clear()
                self.feature.addItems(features)
                self.batch.configure(subject.run_dir, subject.kind, features)
            self.target.clear()
            self.target.addItem("the predicted class", None)
            for index, name in enumerate(self.described.get("classes", [])):
                self.target.addItem(name, index)
            self.about.setText(f"A {subject.describe}: {len(self.described.get('classes', []))} classes"
                               + (f", features {', '.join(features)}" if features else "") + ".")

        self._send("predict", {"run": str(subject.run_dir), "inputs": []}, None, done,
                   lambda message: self.about.setText(f"Could not load the model: {message}"))

    # Requests ------------------------------------------------------------------------------------------
    def _send(self, task: str, args: dict, partial: Any, done: Any, failed: Any) -> str:
        def finish(result: dict[str, Any]) -> None:
            self.pending = None
            self._worker_state()
            done(result)

        def fail(message: str) -> None:
            self.pending = None
            self._worker_state()
            failed(message)

        self.pending = self.ctx.inference.request(task, args, on_partial=partial, on_done=finish,
                                                  on_error=fail)
        self._worker_state()
        return self.pending

    def cancel(self) -> None:
        if self.pending:
            self.ctx.inference.cancel(self.pending)

    def _worker_state(self) -> None:
        info = self.ctx.inference.info
        state = self.ctx.inference.state
        where = f" on {info.get('device')} · torch {info.get('torch')}" if state == "ready" and info else ""
        self.worker.setText(f"inference worker: {state}{where}" + (" · working…" if self.pending else ""))
        self.stop.setEnabled(bool(self.pending))
        self.go.setEnabled(self.subject is not None and not self.pending)

    def _tab(self, name: str) -> None:
        shown = {"try": (self.try_card, self.answer_card), "whatif": (self.whatif_card,),
                 "batch": (self.batch_card,), "visualise": (self.vis_card, self.records)}
        for key, cards in shown.items():
            for card in cards:
                wanted = card is not self.records or self.records.run_dir is not None
                card.setVisible(key == name and wanted)

    # Try it ------------------------------------------------------------------------------------------
    def try_it(self) -> None:
        subject = self.subject
        if subject is None:
            return
        try:
            value = self.inputs.value()
        except ValueError as exc:
            self.outputs.clear(str(exc).capitalize())
            return
        self.outputs.clear("Working…")
        failed = lambda message: self.outputs.clear(f"Failed: {message}")  # noqa: E731
        if subject.is_run:
            self._send("predict", {"run": str(subject.run_dir), "inputs": [value]}, None,
                       lambda r: self.outputs.show_prediction(r["predictions"][0]), failed)
        elif subject.kind == "detect":
            self._send("detect", {"model": subject.repo, "image": value,
                                  "threshold": self.inputs.threshold.value()},
                       None, lambda r: self.outputs.show_detections(value, r), failed)
        elif subject.kind == "zero_shot":
            labels = [part.strip() for part in self.inputs.labels.text().split(",") if part.strip()]
            self._send("zero_shot", {"model": subject.repo, "image": value, "labels": labels}, None,
                       lambda r: self.outputs.show_ranking(r["labels"], r["probabilities"],
                                                           "Best match"), failed)
        elif subject.kind == "caption":
            self._send("caption", {"model": subject.repo, "image": value,
                                   "prompt": self.inputs.prompt.text()},
                       None, lambda r: self.outputs.show_text("Caption", r.get("caption", "")), failed)
        elif subject.kind == "speech":
            self._send("transcribe", {"model": subject.repo, "audio": value},
                       lambda part: self.outputs.show_text("Transcribing…", part.get("text", "")),
                       lambda r: self.outputs.show_text("Transcript", r["text"],
                                                        f"{r['seconds_of_audio']:.1f} s of audio"), failed)

    # What-if -----------------------------------------------------------------------------------------
    def sweep(self) -> None:
        subject = self.subject
        if subject is None or not self.feature.currentText():
            return
        self.whatif_note.setText("Working…")
        args = {"run": str(subject.run_dir), "base": self.inputs.row(), "feature": self.feature.currentText(),
                "low": self.low.value(), "high": self.high.value(), "steps": self.steps.value()}
        self._send("sweep", args, None, self._swept, lambda m: self.whatif_note.setText(f"Failed: {m}"))

    def _swept(self, result: dict[str, Any]) -> None:
        while self.charts_layout.count():
            widget = self.charts_layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        values, classes = result["values"], result["classes"]
        regression = len(result["probabilities"]) == 1 and len(classes) <= 1
        charts = []
        for name, series in list(zip(classes, result["probabilities"], strict=False))[:8]:
            chart = CurveChart(f"P({name})" if not regression else "predicted value",
                               self.feature.currentText(),
                               "probability" if not regression else "value")
            chart.x_range = (min(values), max(values))
            if regression:
                chart.y_range = (min(series), max(series) or 1.0)
            chart.set_curve(list(zip(values, series, strict=True)), f"{series[0]:.2f} → {series[-1]:.2f}")
            chart.setMinimumWidth(200)
            charts.append(chart)
        self.charts_layout.addWidget(FlowGrid(charts, 260, fewest=1))
        self.whatif_note.setText(f"{len(values)} points; the other features stay as in Try it."
                                 + (" The first eight classes are shown." if len(classes) > 8 else ""))

    # Visualise ---------------------------------------------------------------------------------------
    def visualise(self) -> None:
        subject = self.subject
        if subject is None:
            return
        try:
            value = self.inputs.value()
        except ValueError as exc:
            self.vis_note.setText(f"{str(exc).capitalize()} on the Try it tab.")
            return
        self.vis_note.setText("Working…")
        args = {"run": str(subject.run_dir), "input": value if isinstance(value, dict) else {"path": value}}
        if self.target.currentData() is not None:
            args["target"] = int(self.target.currentData())

        def shown(result: dict[str, Any]) -> None:
            record = result.get("record", {})
            self.vis_note.setText("Layers in the order the input passes through them."
                                  + (f" Saved as explanation record {record['id']}." if record else ""))
            self.maps.show_result(result)
            self.records.refresh(subject.run_dir)

        self._send("visualize", args, None, shown, lambda m: self.vis_note.setText(f"Failed: {m}"))
