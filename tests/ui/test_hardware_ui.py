"""Hardware Monitor, Home dashboard, charts and the title-row meters."""

from __future__ import annotations

import time
from dataclasses import replace

from app.services.hardware import HardwareService
from app.ui.pages.hardware import HardwarePage
from app.ui.pages.home import HomePage
from app.ui.widgets import MeterBar, TimeSeriesChart
from core.common.paths import CODE_ROOT
from core.hardware.info import GpuInfo, HardwareInfo
from core.hardware.sampler import GpuSample, ResourceSample


class FakeSampler:
    def __init__(self, gpu: bool = True) -> None:
        self.gpu = gpu
        self.ticks = 0
        self.nvml_error = None if gpu else "no NVIDIA driver (NVMLError_LibraryNotFound)"

    @property
    def gpu_monitoring(self) -> bool:
        return self.gpu

    def sample(self) -> ResourceSample:
        self.ticks += 1
        gpus = (GpuSample(0, "RTX 3050", 50.0 + self.ticks, 1.0, 4.0),) if self.gpu else ()
        return ResourceSample(time.time(), 30.0, 4.0, 16.0, gpus=gpus)

    def close(self) -> None:
        pass


INFO = HardwareInfo("Windows 11", "AMD64", "Ryzen 7", 8, 16, 16.0, "3.11.9", r"C:\py\python.exe",
                    "2.5.1+cu121", "12.1", True, "90100", False, "551.86",
                    (GpuInfo(0, "NVIDIA GeForce RTX 3050 Laptop GPU", 4.0, "nvml"),), ())


def service(qapp, gpu: bool = True) -> HardwareService:
    hardware = HardwareService(python="python", code_root=CODE_ROOT, probe=False,
                               sampler=FakeSampler(gpu), interval_ms=50)
    hardware.start()
    return hardware


def test_service_samples_on_a_timer(qtbot, qapp) -> None:
    hardware = service(qapp)
    with qtbot.waitSignal(hardware.sampled, timeout=2_000):
        pass
    assert len(hardware.history) >= 2 and hardware.gpu_monitoring
    hardware.stop()


def test_hardware_page_shows_a_gpu_machine(ctx, qtbot, qapp) -> None:
    hardware = service(qapp)
    page = HardwarePage(replace(ctx, hardware=hardware))
    qtbot.addWidget(page)
    hardware.info = INFO
    hardware.info_changed.emit(INFO)
    assert page.accelerator_pill.text() == "CUDA ready"
    assert page.accelerators.value_text("PyTorch") == "2.5.1+cu121"
    assert page.system.value_text("Cores") == "8 physical · 16 logical"
    assert page.gpu_table.rowCount() == 1 and not page.notes.isVisibleTo(page)
    hardware.sample_now()
    assert page.stats["gpu"].value.text().endswith("%")
    assert page.stats["vram"].value.text() == "1.0 / 4.0 GB"
    assert page.charts.charts["gpu"].points()
    hardware.stop()


def test_hardware_page_on_a_cpu_only_machine(ctx, qtbot, qapp) -> None:
    hardware = service(qapp, gpu=False)
    page = HardwarePage(replace(ctx, hardware=hardware))
    qtbot.addWidget(page)
    cpu_only = replace(INFO, cuda_available=False, torch_cuda_build=None, nvidia_driver=None, gpus=(),
                       notes=("No NVIDIA driver found.", "PyTorch is a CPU-only build."))
    hardware.info = cpu_only
    hardware.info_changed.emit(cpu_only)
    hardware.sample_now()
    assert page.accelerator_pill.text() == "CPU only"
    assert page.accelerators.value_text("CUDA build") == "CPU-only build"
    assert page.stats["gpu"].value.text() == "—"
    assert page.stats["gpu"].note.full_text == "No NVIDIA GPU detected"
    assert page.notes.isVisibleTo(page) and "CPU-only build" in page.notes_column.text()
    hardware.stop()


def test_probe_failure_is_reported(ctx, qtbot, qapp) -> None:
    hardware = service(qapp)
    page = HardwarePage(replace(ctx, hardware=hardware))
    qtbot.addWidget(page)
    hardware.info_error = "Hardware probe failed: probe exited with 1"
    hardware.info_changed.emit(None)
    assert page.accelerator_pill.text() == "Probe failed" and page.notes.isVisibleTo(page)
    hardware.stop()


def test_real_probe_process_reports(qtbot, qapp) -> None:
    import sys
    hardware = HardwareService(python=sys.executable, code_root=CODE_ROOT, sampler=FakeSampler())
    with qtbot.waitSignal(hardware.info_changed, timeout=280_000) as reported:
        hardware.start()
    assert isinstance(reported.args[0], HardwareInfo), hardware.info_error
    assert hardware.info.python
    hardware.stop()


def test_chart_hover_gaps_and_empty_state(qtbot) -> None:
    chart = TimeSeriesChart()
    qtbot.addWidget(chart)
    chart.resize(400, 140)
    now = time.time()
    chart.set_points([(now - 10, 20.0), (now - 5, None), (now, 60.0)])
    chart.show()
    chart.grab()  # paints without error, including the gap
    qtbot.mouseMove(chart, chart.rect().topRight() - chart.rect().topLeft())
    assert chart.hover_text() == "60% · now"
    chart.set_points([])
    chart.set_empty_text("No NVIDIA GPU detected")
    chart.grab()


def test_title_row_meters(window) -> None:
    meters = window.top_bar.meters
    window.top_bar.show_sample(ResourceSample(time.time(), 28.0, 5.1, 7.7,
                                              gpus=(GpuSample(0, "RTX", 71.0, 3.2, 4.0),)))
    assert meters["cpu"].text() == "28%" and meters["gpu"].text() == "71%"
    assert meters["vram"].text() == "80%" and "3.2 of 4.0 GB" in meters["vram"].toolTip()
    window.top_bar.show_sample(ResourceSample(time.time(), 5.0, 1.0, 8.0))
    assert meters["gpu"].text() == "—" and meters["gpu"].toolTip() == "No NVIDIA GPU detected"
    assert isinstance(meters["ram"], MeterBar)


def test_home_dashboard_follows_jobs(ctx, qtbot) -> None:
    home = HomePage(ctx)
    qtbot.addWidget(home)
    assert home.jobs_tile.value.text() == "0"
    assert home.active_job.pill.text() == "Idle"
    assert home.experiments.empty.isVisibleTo(home)

    def work(token, report):
        token.wait(0.2)
        return "done"

    with qtbot.waitSignal(ctx.jobs.job_finished, timeout=10_000):
        ctx.jobs.submit_task(work, title="Warm-up task")
        assert home.active_job.title.text() == "Warm-up task"
        assert home.jobs_tile.value.text() == "1"
    assert home.active_job.pill.text() == "Completed"
    assert home.jobs_tile.value.text() == "0"
    assert home.history.table.rowCount() == 1


def test_home_counts_experiments(ctx, qtbot) -> None:
    ctx.store.create_experiment("mnist-cnn", "image_classification", "", "h")
    home = HomePage(ctx)
    qtbot.addWidget(home)
    assert home.experiments_tile.value.text() == "1"
    assert home.experiments.table.rowCount() == 1 and not home.experiments.empty.isVisibleTo(home)
