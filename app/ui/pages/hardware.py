"""Hardware Monitor: what the machine has, and how busy it is right now."""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QWidget

from core.hardware.info import HardwareInfo

from ...services.context import AppContext
from ..widgets import Card, DataTable, KeyValues, Page, Pill, ResponsiveRow, label
from .resources import ResourceCharts, ResourceStats, gb

__all__ = ["HardwarePage"]

_PENDING = "Detecting…"


def _yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"


class HardwarePage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Hardware Monitor", "sampled every second · last two minutes", parent)
        self.ctx = ctx
        service = ctx.hardware
        self.detect_button = QPushButton("Detect again")
        self.detect_button.setToolTip("Re-run the hardware probe, e.g. after installing a "
                                      "CUDA build of PyTorch or a new driver")
        self.detect_button.clicked.connect(self._reprobe)
        self.head.add_action(self.detect_button)

        self.stats = ResourceStats(service)
        self.body.addWidget(self.stats)

        charts = Card("Utilisation", "percent of capacity")
        self.charts = ResourceCharts(service, columns=2)
        charts.add(self.charts)
        self.body.addWidget(charts)

        self.system = KeyValues((("Operating system", _PENDING), ("CPU", _PENDING),
                                 ("Cores", _PENDING), ("Memory", _PENDING),
                                 ("Python", _PENDING), ("Interpreter", _PENDING)))
        system = Card("System")
        system.add(self.system)

        self.accelerators = KeyValues((("PyTorch", _PENDING), ("CUDA available", _PENDING),
                                       ("CUDA build", _PENDING), ("cuDNN", _PENDING),
                                       ("Apple MPS", _PENDING), ("NVIDIA driver", _PENDING)))
        accelerators = Card("Accelerators", padded=True)
        self.accelerator_pill = Pill(_PENDING, "planned")
        accelerators.add_head_widget(self.accelerator_pill)
        accelerators.add(self.accelerators)
        self.gpu_table = DataTable(("#", "GPU", "Memory", "Source"), mono_columns=(0, 2),
                                   stretch_column=1)
        accelerators.add(self.gpu_table)
        self.body.addWidget(ResponsiveRow([(system, 1), (accelerators, 1)], breakpoint=760))

        self.notes = Card("Notes", "why something is missing")
        self.notes_column = label("", "Body", wrap=True)
        self.notes.add(self.notes_column)
        self.notes.hide()
        self.body.addWidget(self.notes)
        self.body.addStretch(1)

        service.info_changed.connect(self._show_info)
        self._show_info(service.info)

    def _reprobe(self) -> None:
        self.accelerator_pill.setText(_PENDING)
        self.accelerator_pill.set_tone("planned")
        self.ctx.hardware.reprobe()

    def _show_info(self, info: HardwareInfo | None) -> None:
        service = self.ctx.hardware
        if info is None:
            failed = service.info_error
            if failed:
                self.accelerator_pill.setText("Probe failed")
                self.accelerator_pill.set_tone("fail")
                self._set_notes([failed])
            elif not service.probing:
                self.accelerator_pill.setText("Not detected")
            return
        cores = (f"{info.cpu_cores_physical} physical · {info.cpu_cores_logical} logical"
                 if info.cpu_cores_physical else str(info.cpu_cores_logical or "unknown"))
        for key, value in (("Operating system", f"{info.os} ({info.machine})"),
                           ("CPU", info.cpu_model), ("Cores", cores),
                           ("Memory", gb(info.ram_total_gb)), ("Python", info.python),
                           ("Interpreter", info.python_executable)):
            self.system.set_value(key, value)
        for key, value in (("PyTorch", info.torch_version or "not installed"),
                           ("CUDA available", _yes_no(info.cuda_available)),
                           ("CUDA build", info.torch_cuda_build or "CPU-only build"),
                           ("cuDNN", info.cudnn_version or "—"),
                           ("Apple MPS", _yes_no(info.mps_available)),
                           ("NVIDIA driver", info.nvidia_driver or "not found")):
            self.accelerators.set_value(key, value)
        if info.cuda_available:
            self.accelerator_pill.setText("CUDA ready")
            self.accelerator_pill.set_tone("ok")
        elif info.mps_available:
            self.accelerator_pill.setText("MPS ready")
            self.accelerator_pill.set_tone("ok")
        else:
            self.accelerator_pill.setText("CPU only")
            self.accelerator_pill.set_tone("warn")
        self.gpu_table.clear_rows()
        for gpu in info.gpus:
            self.gpu_table.add_row((str(gpu.index), gpu.name, gb(gpu.vram_total_gb), gpu.source))
        self.gpu_table.setVisible(bool(info.gpus))
        self._set_notes(list(info.notes))

    def _set_notes(self, notes: list[str]) -> None:
        self.notes_column.setText("\n".join(f"– {note}" for note in notes))
        self.notes.setVisible(bool(notes))
