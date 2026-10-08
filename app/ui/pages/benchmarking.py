"""Benchmarking: how fast a model infers, by batch size and device. Time a finished run's
checkpoint (PyTorch or scikit-learn) or an image model built untrained; the work runs in the
worker process, never in the app. Results stay in the workspace's results/benchmarks/."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QWidget,
)

from core.common.vocab import Task

from ...services.benchmarks import Benchmark
from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, KeyValues, Page, Pill, ResponsiveRow, label
from ..widgets.plots import BarList
from .home_cards import local_time

__all__ = ["BenchmarkingPage", "parse_batches"]

STATUS_TONES = {"queued": "planned", "running": "info", "done": "ok", "failed": "fail", "cancelled": "warn"}


def parse_batches(text: str) -> list[int]:
    """"1, 8 32" → [1, 8, 32]; raises ``ValueError`` with a reason."""
    parts = [part for part in text.replace(",", " ").split() if part]
    if not parts or not all(part.isdigit() and int(part) > 0 for part in parts):
        raise ValueError("batch sizes are whole numbers above 0, e.g. 1, 8, 32")
    return sorted({int(part) for part in parts})


def _spin(low: int, high: int, value: int) -> QSpinBox:
    box = QSpinBox()
    box.setRange(low, high)
    box.setValue(value)
    box.setMaximumWidth(120)
    return box


class BenchmarkingPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Benchmarking", "latency and throughput by batch size and device", parent)
        self.ctx = ctx
        self.ids: list[str] = []
        self.selected: str | None = None

        setup = Card("Benchmark a model", "runs in the worker process; the app never uses the GPU itself")
        rows = QFormLayout()
        rows.setHorizontalSpacing(14)
        self.source = QComboBox()
        self.source.setMaximumWidth(460)
        self.source.currentIndexChanged.connect(lambda _: self._source_changed())
        self.channels = _spin(1, 4, 3)
        self.size = _spin(8, 1024, 32)
        self.shape_row = QWidget()
        shape = QHBoxLayout(self.shape_row)
        shape.setContentsMargins(0, 0, 0, 0)
        for widget in (self.channels, label("channels ×", "Body"), self.size, label("pixels square", "Body")):
            shape.addWidget(widget)
        shape.addStretch(1)
        self.device = QComboBox()
        self.device.addItem("CPU", "cpu")
        info = ctx.hardware.info
        if info is not None and info.cuda_available:
            for gpu in info.gpus:
                self.device.addItem(f"cuda:{gpu.index} · {gpu.name}", f"cuda:{gpu.index}")
        self.device.currentIndexChanged.connect(lambda _: self._source_changed())
        self.device.setMaximumWidth(320)
        self.precision = QComboBox()
        self.precision.setMaximumWidth(320)
        self.precision.addItem("fp32", "fp32")
        self.precision.addItem("fp16 (CUDA only)", "fp16")
        self.batches = QLineEdit("1, 8, 32, 64")
        self.batches.setMaximumWidth(240)
        self.warmup = _spin(0, 1000, 5)
        self.iterations = _spin(1, 10_000, 30)
        for title, widget in (("Model", self.source), ("Input", self.shape_row), ("Device", self.device),
                              ("Precision", self.precision), ("Batch sizes", self.batches),
                              ("Warm-up passes", self.warmup), ("Timed passes", self.iterations)):
            rows.addRow(title, widget)
        setup.add(rows)
        actions = QHBoxLayout()
        self.run_button = QPushButton(icon("timer"), "Run benchmark")
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self.run)
        self.message = label("", "CardCaption", wrap=True)
        actions.addWidget(self.run_button)
        actions.addWidget(self.message, 1)
        setup.add(actions)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(6)
        self.progress.setTextVisible(False)
        self.progress.hide()
        setup.add(self.progress)
        self.body.addWidget(setup)

        history = Card("Benchmarks", padded=False)
        self.table = DataTable(("Model", "Device", "Best throughput", "Status", "When"),
                               mono_columns=(1, 2, 4), stretch_column=0)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        history.add(self.table)
        self.empty = label("No benchmarks yet.", "CardCaption")
        self.empty.setContentsMargins(14, 10, 14, 12)
        history.add(self.empty)
        self.body.addWidget(history)

        self.detail = Card("Results")
        self.state = Pill("", "planned")
        self.detail.add_head_widget(self.state)
        self.values = KeyValues((("Model", ""), ("Device", ""), ("Parameters", ""), ("Size", ""),
                                 ("Passes", ""), ("Machine", "")))
        self.detail.add(self.values)
        self.rows = DataTable(("Batch", "Mean ms", "Median ms", "95th pct ms", "Per second", "Peak GPU MB"),
                              mono_columns=(0, 1, 2, 3, 4, 5), stretch_column=0)
        self.detail.add(self.rows)
        throughput, latency = Card("Throughput", "samples per second"), Card("Latency", "mean ms per batch")
        self.throughput_bars, self.latency_bars = BarList(), BarList()
        throughput.add(self.throughput_bars)
        latency.add(self.latency_bars)
        self.detail.add(ResponsiveRow([(throughput, 1), (latency, 1)], breakpoint=760))
        self.note = label("", "Body", wrap=True)
        self.detail.add(self.note)
        self.body.addWidget(self.detail)
        self.body.addStretch(1)

        ctx.benchmarks.changed.connect(self.refresh)
        ctx.benchmarks.finished.connect(lambda *_: self.message.setText(""))
        ctx.experiments.run_changed.connect(lambda _: self._fill_sources())
        self._fill_sources()
        self.refresh()

    # Choices ----------------------------------------------------------------------------
    def _fill_sources(self) -> None:
        current = self.source.currentData()
        self.source.blockSignals(True)
        self.source.clear()
        for view in self.ctx.experiments.views(limit=200):
            if view.finished_ok and self.ctx.models.checkpoint_of(view.id) is not None:
                self.source.addItem(f"Run · {view.name} · {view.model} · {view.id[:8]}", f"run:{view.id}")
        models = self.ctx.catalog.models
        for card in models.all():
            if Task.IMAGE_CLASSIFICATION in card.tasks and models.status(card.id) == "ready":
                self.source.addItem(f"Untrained · {card.name}", f"card:{card.id}")
        self.source.setCurrentIndex(max(self.source.findData(current), 0))
        self.source.blockSignals(False)
        self._source_changed()

    def choice(self) -> tuple[str, str] | None:
        """("run", run id) or ("card", model card id) for the chosen model."""
        data = self.source.currentData()
        return tuple(data.split(":", 1)) if data else None  # type: ignore[return-value]

    def _source_changed(self) -> None:
        choice = self.choice()
        self.shape_row.setEnabled(choice is not None and choice[0] == "card")
        tabular = choice is not None and choice[0] == "run" and self.ctx.experiments.view(choice[1]).tabular
        cuda = str(self.device.currentData()).startswith("cuda")
        self.device.setEnabled(not tabular)
        self.precision.setEnabled(cuda and not tabular)
        if not cuda or tabular:
            self.precision.setCurrentIndex(0)
        self.run_button.setEnabled(choice is not None)

    def request(self) -> tuple[dict, str]:
        """The benchmark request and its title; ``ValueError`` when a field is wrong."""
        choice = self.choice()
        if choice is None:
            raise ValueError("choose a model")
        request = {"device": self.device.currentData() if self.device.isEnabled() else "cpu",
                   "precision": self.precision.currentData(),
                   "batch_sizes": parse_batches(self.batches.text()),
                   "warmup": self.warmup.value(), "iterations": self.iterations.value()}
        kind, key = choice
        if kind == "run":
            view = self.ctx.experiments.view(key)
            request["run_dir"] = view.run_dir.relative_to(self.ctx.paths.workspace).as_posix() \
                if view.run_dir.is_relative_to(self.ctx.paths.workspace) else str(view.run_dir)
            return request, f"{view.name} ({view.model})"
        request.update(model=key, input_shape=[self.channels.value(), self.size.value(), self.size.value()])
        return request, f"{self.ctx.catalog.models.get(key).name} untrained"

    def run(self) -> str | None:
        try:
            request, title = self.request()
        except ValueError as exc:
            self.message.setText(str(exc))
            return None
        self.selected = self.ctx.benchmarks.start(request, title)
        self.message.setText("Queued; results appear below when it finishes.")
        self.refresh()
        return self.selected

    # Results ----------------------------------------------------------------------------
    def refresh(self) -> None:
        benchmarks = self.ctx.benchmarks.all()
        self.ids = [b.id for b in benchmarks]
        self.table.blockSignals(True)
        self.table.clear_rows()
        for bench in benchmarks:
            best = bench.best_throughput
            device = (bench.results or {}).get("device") or bench.request.get("device", "")
            self.table.add_row((bench.title, device, "—" if best is None else f"{best:,.0f} /s",
                                Pill(bench.status.capitalize(), STATUS_TONES.get(bench.status, "planned")),
                                local_time((bench.results or {}).get("created_at"))))
        self.table.blockSignals(False)
        self.table.setVisible(bool(benchmarks))
        self.empty.setVisible(not benchmarks)
        self.progress.setVisible(any(b.status in ("queued", "running") for b in benchmarks))
        if self.selected not in self.ids:
            self.selected = self.ids[0] if self.ids else None
        if self.selected is not None:
            self.table.blockSignals(True)
            self.table.selectRow(self.ids.index(self.selected))
            self.table.blockSignals(False)
        self._show()

    def _selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.selected = self.ids[rows[0].row()]
            self._show()

    def _show(self) -> None:
        bench: Benchmark | None = self.ctx.benchmarks.get(self.selected) if self.selected else None
        self.detail.setVisible(bench is not None)
        if bench is None:
            return
        self.detail.set_title(bench.title)
        self.state.setText(bench.status.capitalize())
        self.state.set_tone(STATUS_TONES.get(bench.status, "planned"))
        results = bench.results or {}
        self.values.set_value("Model", results.get("model", bench.title))
        self.values.set_value("Device", f"{results.get('device', bench.request.get('device'))} · "
                                        f"{results.get('precision', bench.request.get('precision'))}")
        parameters = results.get("parameters")
        self.values.set_value("Parameters", f"{parameters:,}" if parameters else "—")
        self.values.set_value("Size", f"{results['size_mb']:.1f} MB" if results.get("size_mb") else "—")
        self.values.set_value("Passes", f"{bench.request.get('warmup')} warm-up, "
                                        f"{bench.request.get('iterations')} timed per batch size")
        self.values.set_value("Machine", f"{results.get('cpu', '—')} · Python {results.get('python', '—')}")
        self.rows.clear_rows()
        for row in results.get("rows", []):
            peak = row.get("peak_mem_mb")
            self.rows.add_row((str(row["batch"]), f"{row['mean_ms']:.2f}", f"{row['p50_ms']:.2f}",
                               f"{row['p95_ms']:.2f}", f"{row['throughput']:,.0f}",
                               "—" if peak is None else f"{peak:,.0f}"))
        measured = results.get("rows", [])
        self.throughput_bars.set_rows([(f"batch {r['batch']}", r["throughput"]) for r in measured])
        self.latency_bars.set_rows([(f"batch {r['batch']}", r["mean_ms"]) for r in measured])
        notes = [results.get("note", ""), bench.error]
        self.note.setText(" ".join(n for n in notes if n))
        self.note.setVisible(bool(self.note.text()))
