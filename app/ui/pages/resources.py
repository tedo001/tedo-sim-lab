"""Live resource tiles and charts, shared by Home and the Hardware Monitor."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QWidget

from core.hardware.info import HardwareInfo
from core.hardware.sampler import ResourceSample, SampleHistory

from ...services.hardware import HardwareService
from ..widgets import StatStrip, StatTile, TimeSeriesChart, label

__all__ = ["ResourceCharts", "ResourceStats", "gb", "no_gpu_reason", "used_of"]


def gb(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} GB"


def used_of(used: float | None, total: float | None) -> str:
    """``3.2 / 4.0 GB``, or ``—`` when either side is unknown."""
    return "—" if used is None or total is None else f"{used:.1f} / {total:.1f} GB"


def no_gpu_reason(service: HardwareService) -> str:
    """Why there is no GPU line, in a few words (the details are on the Hardware Monitor)."""
    info = service.info
    if info is not None and info.mps_available:
        return "Apple GPU (MPS): live use is not reported"
    return "No NVIDIA GPU detected"


class ResourceStats(StatStrip):
    """CPU, memory, GPU and VRAM headline numbers."""

    def __init__(self, service: HardwareService, extra: dict[str, StatTile] | None = None,
                 parent: QWidget | None = None) -> None:
        tiles = {"cpu": StatTile("CPU"), "ram": StatTile("Memory"), "gpu": StatTile("GPU"),
                 "vram": StatTile("GPU memory")}
        tiles.update(extra or {})
        super().__init__(tiles, parent)
        self.service = service
        service.sampled.connect(self.show_sample)
        service.info_changed.connect(lambda _info: self.show_sample(service.history.latest))
        self.show_sample(service.history.latest)

    def show_sample(self, sample: ResourceSample | None) -> None:
        info: HardwareInfo | None = self.service.info
        if sample is None:
            return
        cores = info.cpu_cores_logical if info else None
        self["cpu"].set(f"{sample.cpu_pct:.0f}%", f"{cores} logical cores" if cores else "all cores")
        self["ram"].set(used_of(sample.ram_used_gb, sample.ram_total_gb),
                        f"{sample.ram_pct:.0f}% in use")
        gpu = sample.gpu
        if gpu is None:
            self["gpu"].set("—", no_gpu_reason(self.service))
            self["vram"].set("—", "training runs on the CPU")
            return
        self["gpu"].set("—" if gpu.util_pct is None else f"{gpu.util_pct:.0f}%", gpu.name)
        pct = gpu.vram_pct
        self["vram"].set(used_of(gpu.vram_used_gb, gpu.vram_total_gb),
                         "—" if pct is None else f"{pct:.0f}% in use")


class ResourceCharts(QWidget):
    """CPU, memory, GPU and GPU-memory use over the last two minutes: small multiples."""

    METRICS = (("cpu", "CPU"), ("ram", "Memory"), ("gpu", "GPU"), ("vram", "GPU memory"))

    def __init__(self, service: HardwareService, *, columns: int = 2,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.charts: dict[str, TimeSeriesChart] = {}
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        for index, (metric, title) in enumerate(self.METRICS):
            row, column = divmod(index, columns)
            chart = TimeSeriesChart()
            chart.setAccessibleName(f"{title} use, last two minutes")
            self.charts[metric] = chart
            grid.addWidget(label(title, "CardCaption"), row * 2, column)
            grid.addWidget(chart, row * 2 + 1, column)
        service.sampled.connect(lambda _sample: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        history: SampleHistory = self.service.history
        latest = history.latest
        for metric, chart in self.charts.items():
            chart.set_points(history.series(metric))
            if metric in ("gpu", "vram") and latest is not None and latest.gpu is None:
                chart.set_empty_text(no_gpu_reason(self.service))
