"""Live resource use: CPU, memory and (NVIDIA) GPU, sampled cheaply and often.

Uses psutil and NVML only. It never touches PyTorch's CUDA API, which would
create a CUDA context and hold GPU memory inside the UI process. On a machine
without an NVIDIA driver the GPU part is simply empty.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from core.common.optional import optional_import

__all__ = ["GpuSample", "ResourceSample", "ResourceSampler", "SampleHistory"]

GIB = 1024 ** 3


@dataclass(frozen=True)
class GpuSample:
    index: int
    name: str
    util_pct: float | None
    vram_used_gb: float | None
    vram_total_gb: float | None
    temp_c: float | None = None
    power_w: float | None = None

    @property
    def vram_pct(self) -> float | None:
        if self.vram_used_gb is None or not self.vram_total_gb:
            return None
        return 100.0 * self.vram_used_gb / self.vram_total_gb


@dataclass(frozen=True)
class ResourceSample:
    ts: float
    cpu_pct: float
    ram_used_gb: float
    ram_total_gb: float
    cpu_per_core: tuple[float, ...] = ()
    gpus: tuple[GpuSample, ...] = field(default_factory=tuple)

    @property
    def ram_pct(self) -> float:
        return 100.0 * self.ram_used_gb / self.ram_total_gb if self.ram_total_gb else 0.0

    @property
    def gpu(self) -> GpuSample | None:
        """The first GPU (the one experiments use by default)."""
        return self.gpus[0] if self.gpus else None


class ResourceSampler:
    """Call :meth:`sample` on a timer. Keeps NVML initialised between samples."""

    def __init__(self, *, use_nvml: bool = True) -> None:
        self._psutil = optional_import("psutil")
        self._nvml = None
        self.nvml_error: str | None = None
        if self._psutil is not None:
            self._psutil.cpu_percent(interval=None)  # prime: the first reading is meaningless
            self._psutil.cpu_percent(interval=None, percpu=True)
        if use_nvml:
            self._start_nvml()

    def _start_nvml(self) -> None:
        pynvml = optional_import("pynvml", "nvidia-ml-py")
        if pynvml is None:
            self.nvml_error = "nvidia-ml-py is not installed"
            return
        try:
            pynvml.nvmlInit()
        except Exception as exc:
            self.nvml_error = f"no NVIDIA driver ({type(exc).__name__})"
            return
        self._nvml = pynvml

    @property
    def gpu_monitoring(self) -> bool:
        return self._nvml is not None

    def _gpu_samples(self) -> tuple[GpuSample, ...]:
        nvml = self._nvml
        if nvml is None:
            return ()
        try:
            count = nvml.nvmlDeviceGetCount()
        except Exception:
            return ()
        return tuple(self._gpu_sample(nvml, index) for index in range(count))

    @staticmethod
    def _gpu_sample(nvml, index: int) -> GpuSample:
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        memory = _call(lambda: nvml.nvmlDeviceGetMemoryInfo(handle))
        return GpuSample(
            index, _call(lambda: _text(nvml.nvmlDeviceGetName(handle))) or f"GPU {index}",
            _call(lambda: float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)),
            memory.used / GIB if memory else None, memory.total / GIB if memory else None,
            _call(lambda: float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))),
            _call(lambda: nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0))

    def sample(self) -> ResourceSample:
        psutil = self._psutil
        if psutil is None:
            return ResourceSample(time.time(), 0.0, 0.0, 0.0, gpus=self._gpu_samples())
        memory = psutil.virtual_memory()
        return ResourceSample(
            ts=time.time(), cpu_pct=float(psutil.cpu_percent(interval=None)),
            ram_used_gb=(memory.total - memory.available) / GIB, ram_total_gb=memory.total / GIB,
            cpu_per_core=tuple(float(v) for v in psutil.cpu_percent(interval=None, percpu=True)),
            gpus=self._gpu_samples())

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None


def _call(read):
    """One NVML field; a field the GPU does not support is ``None``, not an error."""
    try:
        return read()
    except Exception:
        return None


def _text(value: str | bytes) -> str:
    return value if isinstance(value, str) else value.decode()


class SampleHistory:
    """The last ``size`` samples, oldest first."""

    def __init__(self, size: int = 120) -> None:
        self._samples: deque[ResourceSample] = deque(maxlen=size)

    def add(self, sample: ResourceSample) -> None:
        self._samples.append(sample)

    def extend(self, samples: Iterable[ResourceSample]) -> None:
        self._samples.extend(samples)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def latest(self) -> ResourceSample | None:
        return self._samples[-1] if self._samples else None

    def series(self, metric: str) -> list[tuple[float, float | None]]:
        """``[(ts, value), ...]`` for ``cpu``, ``ram``, ``gpu`` or ``vram`` (percentages)."""
        pick = {"cpu": lambda s: s.cpu_pct, "ram": lambda s: s.ram_pct,
                "gpu": lambda s: s.gpu.util_pct if s.gpu else None,
                "vram": lambda s: s.gpu.vram_pct if s.gpu else None}[metric]
        return [(sample.ts, pick(sample)) for sample in self._samples]
