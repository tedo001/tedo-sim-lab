"""What this machine has: CPU, memory, GPUs, CUDA, MPS, Python and PyTorch.

:func:`probe_hardware` never raises. Anything it cannot find is ``None`` with a
note saying why ("no NVIDIA driver", "PyTorch is a CPU build"), so a CPU-only
laptop gets a complete, honest report instead of an error.

Run it in its own process (``python -m core.hardware.info``) from the app:
asking PyTorch about CUDA devices creates a CUDA context, which would hold
hundreds of MB of GPU memory inside the UI process for its whole life.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.common.optional import optional_import

__all__ = ["GpuInfo", "HardwareInfo", "probe_hardware"]

GIB = 1024 ** 3


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    vram_total_gb: float | None
    #: Where the figures came from: "nvml" (NVIDIA driver) or "torch".
    source: str


@dataclass(frozen=True)
class HardwareInfo:
    os: str
    machine: str
    cpu_model: str
    cpu_cores_physical: int | None
    cpu_cores_logical: int | None
    ram_total_gb: float | None
    python: str
    python_executable: str
    torch_version: str | None
    torch_cuda_build: str | None
    cuda_available: bool
    cudnn_version: str | None
    mps_available: bool
    nvidia_driver: str | None
    gpus: tuple[GpuInfo, ...] = ()
    #: Why something is missing, in words.
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus) or self.mps_available

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> HardwareInfo:
        data = json.loads(text)
        data["gpus"] = tuple(GpuInfo(**gpu) for gpu in data.get("gpus", ()))
        data["notes"] = tuple(data.get("notes", ()))
        return cls(**data)


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def _nvml_gpus(notes: list[str]) -> tuple[list[GpuInfo], str | None]:
    pynvml = optional_import("pynvml", "nvidia-ml-py")
    if pynvml is None:
        notes.append("nvidia-ml-py is not installed: NVIDIA GPUs are not monitored.")
        return [], None
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # no driver, no GPU, or a non-NVIDIA machine
        notes.append(f"No NVIDIA driver found ({type(exc).__name__}).")
        return [], None
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        gpus = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(GpuInfo(index, name if isinstance(name, str) else name.decode(),
                                round(memory.total / GIB, 2), "nvml"))
        return gpus, driver if isinstance(driver, str) else driver.decode()
    except Exception as exc:
        notes.append(f"NVIDIA driver found but could not be queried ({type(exc).__name__}).")
        return [], None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _torch_facts(notes: list[str], have_nvml_gpus: bool) -> dict:
    facts = {"torch_version": None, "torch_cuda_build": None, "cuda_available": False,
             "cudnn_version": None, "mps_available": False, "gpus": []}
    torch = optional_import("torch")
    if torch is None:
        notes.append("PyTorch is not installed: training is unavailable.")
        return facts
    facts["torch_version"] = torch.__version__
    facts["torch_cuda_build"] = torch.version.cuda
    try:
        facts["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        notes.append(f"CUDA check failed ({type(exc).__name__}).")
    if torch.version.cuda is None:
        notes.append("PyTorch is a CPU-only build: install the CUDA build to train on an NVIDIA GPU.")
    elif not facts["cuda_available"]:
        notes.append("PyTorch has CUDA support, but no usable GPU or driver was found.")
    if facts["cuda_available"]:
        cudnn = torch.backends.cudnn.version()
        facts["cudnn_version"] = str(cudnn) if cudnn else None
        if not have_nvml_gpus:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                facts["gpus"].append(GpuInfo(index, props.name,
                                             round(props.total_memory / GIB, 2), "torch"))
    mps = getattr(torch.backends, "mps", None)
    facts["mps_available"] = bool(mps and mps.is_available())
    return facts


def probe_hardware() -> HardwareInfo:
    notes: list[str] = []
    psutil = optional_import("psutil")
    if psutil is not None:
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        ram = round(psutil.virtual_memory().total / GIB, 2)
    else:
        physical, logical, ram = None, os.cpu_count(), None
        notes.append("psutil is not installed: memory is not reported.")
    nvml_gpus, driver = _nvml_gpus(notes)
    torch = _torch_facts(notes, bool(nvml_gpus))
    return HardwareInfo(
        os=f"{platform.system()} {platform.release()}", machine=platform.machine(),
        cpu_model=_cpu_model(), cpu_cores_physical=physical, cpu_cores_logical=logical,
        ram_total_gb=ram, python=platform.python_version(), python_executable=sys.executable,
        torch_version=torch["torch_version"], torch_cuda_build=torch["torch_cuda_build"],
        cuda_available=torch["cuda_available"], cudnn_version=torch["cudnn_version"],
        mps_available=torch["mps_available"], nvidia_driver=driver,
        gpus=tuple(nvml_gpus or torch["gpus"]), notes=tuple(notes))


def main() -> int:
    """Print :class:`HardwareInfo` as JSON (one line) for the app to read."""
    print(probe_hardware().to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
