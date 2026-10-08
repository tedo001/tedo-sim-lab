"""Hardware probe and sampler: complete on a CPU-only machine, correct with a GPU."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from core.common.paths import CODE_ROOT
from core.hardware import info as info_module
from core.hardware import sampler as sampler_module
from core.hardware.devices import DeviceUnavailable, resolve_device
from core.hardware.info import GpuInfo, HardwareInfo, probe_hardware
from core.hardware.sampler import GpuSample, ResourceSample, ResourceSampler, SampleHistory


def test_probe_never_raises_and_explains_gaps() -> None:
    info = probe_hardware()
    assert info.cpu_cores_logical and info.ram_total_gb and info.python
    if not info.gpus:
        assert any("GPU" in note or "driver" in note or "CUDA" in note for note in info.notes)


def test_probe_json_round_trip() -> None:
    info = HardwareInfo("Linux 6", "x86_64", "CPU", 4, 8, 16.0, "3.11", "/py", "2.5", "12.4", True,
                        "90100", False, "550.1", (GpuInfo(0, "RTX 3050", 4.0, "nvml"),), ("note",))
    again = HardwareInfo.from_json(info.to_json())
    assert again == info and again.has_gpu


def test_probe_runs_as_its_own_process() -> None:
    result = subprocess.run([sys.executable, "-m", "core.hardware.info"], cwd=CODE_ROOT,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["python"] and "gpus" in data


def test_probe_without_psutil_or_torch(monkeypatch) -> None:
    monkeypatch.setattr(info_module, "optional_import", lambda *a, **k: None)
    info = probe_hardware()
    assert info.torch_version is None and info.ram_total_gb is None and not info.cuda_available
    assert any("PyTorch is not installed" in note for note in info.notes)


class FakeNvml:
    """Just enough of pynvml for one 4 GB GPU; temperature is unsupported."""

    NVML_TEMPERATURE_GPU = 0

    def __init__(self, *, fail_init: bool = False) -> None:
        self.fail_init = fail_init
        self.shut = False

    def nvmlInit(self):
        if self.fail_init:
            raise RuntimeError("NVML Shared Library Not Found")

    def nvmlShutdown(self):
        self.shut = True

    def nvmlSystemGetDriverVersion(self):
        return "550.54"

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetName(self, handle):
        return b"NVIDIA GeForce RTX 3050 Laptop GPU"

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(used=1 * 1024 ** 3, total=4 * 1024 ** 3)

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=71)

    def nvmlDeviceGetTemperature(self, handle, kind):
        raise RuntimeError("not supported")

    def nvmlDeviceGetPowerUsage(self, handle):
        return 35_000


def fake_imports(nvml):
    real = sampler_module.optional_import

    def optional_import(module, distribution=None):
        return nvml if module == "pynvml" else real(module, distribution)
    return optional_import


def test_sampler_reads_an_nvidia_gpu(monkeypatch) -> None:
    nvml = FakeNvml()
    monkeypatch.setattr(sampler_module, "optional_import", fake_imports(nvml))
    sampler = ResourceSampler()
    sample = sampler.sample()
    gpu = sample.gpu
    assert sampler.gpu_monitoring
    assert gpu.name == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert gpu.util_pct == 71 and gpu.vram_total_gb == pytest.approx(4.0)
    assert gpu.vram_pct == pytest.approx(25.0)
    assert gpu.temp_c is None and gpu.power_w == pytest.approx(35.0)  # unsupported field → None
    sampler.close()
    assert nvml.shut


def test_sampler_without_a_driver_has_no_gpu(monkeypatch) -> None:
    monkeypatch.setattr(sampler_module, "optional_import", fake_imports(FakeNvml(fail_init=True)))
    sampler = ResourceSampler()
    sample = sampler.sample()
    assert sample.gpus == () and not sampler.gpu_monitoring
    assert "no NVIDIA driver" in sampler.nvml_error
    assert 0 <= sample.cpu_pct <= 100 and 0 < sample.ram_used_gb <= sample.ram_total_gb


def test_probe_lists_nvml_gpus(monkeypatch) -> None:
    real = info_module.optional_import
    monkeypatch.setattr(info_module, "optional_import",
                        lambda m, d=None: FakeNvml() if m == "pynvml" else real(m, d))
    info = probe_hardware()
    assert info.nvidia_driver == "550.54"
    assert info.gpus == (GpuInfo(0, "NVIDIA GeForce RTX 3050 Laptop GPU", 4.0, "nvml"),)


def test_history_keeps_the_last_samples() -> None:
    history = SampleHistory(size=3)
    for second in range(5):
        gpu = GpuSample(0, "g", 10.0 * second, 1.0, 4.0) if second % 2 == 0 else None
        history.add(ResourceSample(float(second), 5.0 * second, 2.0, 8.0,
                                   gpus=(gpu,) if gpu else ()))
    assert len(history) == 3 and history.latest.ts == 4.0
    assert history.series("cpu") == [(2.0, 10.0), (3.0, 15.0), (4.0, 20.0)]
    assert history.series("gpu") == [(2.0, 20.0), (3.0, None), (4.0, 40.0)]  # gap, not zero
    assert history.series("ram")[0][1] == pytest.approx(25.0)


def test_devices_resolve_on_a_cpu_machine() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda:0", "mps")
    if resolve_device("auto") == "cpu":
        with pytest.raises(DeviceUnavailable, match="CUDA was requested"):
            resolve_device("cuda:0")
    with pytest.raises(DeviceUnavailable, match="unknown device"):
        resolve_device("tpu")
