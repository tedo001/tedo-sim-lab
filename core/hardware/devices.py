"""Turn a requested device ("auto", "cpu", "cuda:1", "mps") into one that exists."""

from __future__ import annotations

from core.common.optional import optional_import

__all__ = ["DeviceUnavailable", "resolve_device"]


class DeviceUnavailable(RuntimeError):
    """The experiment asked for a device this machine does not have."""


def resolve_device(requested: str) -> str:
    """``auto`` → first CUDA GPU, else Apple MPS, else CPU. Explicit requests are checked."""
    if requested == "cpu":
        return "cpu"
    torch = optional_import("torch")
    cuda = bool(torch and torch.cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None) if torch else None
    mps = bool(mps_backend and mps_backend.is_available())
    if requested == "auto":
        return "cuda:0" if cuda else "mps" if mps else "cpu"
    if requested.startswith("cuda"):
        if not cuda:
            raise DeviceUnavailable("CUDA was requested but is not available: no NVIDIA GPU, no "
                                    "CUDA build of PyTorch, or a driver problem. Use 'auto' or 'cpu'.")
        index = int(requested.partition(":")[2] or 0)
        count = torch.cuda.device_count()
        if index >= count:
            raise DeviceUnavailable(f"cuda:{index} was requested but only {count} GPU(s) exist")
        return f"cuda:{index}"
    if requested == "mps":
        if not mps:
            raise DeviceUnavailable("Apple MPS was requested but is not available on this machine")
        return "mps"
    raise DeviceUnavailable(f"unknown device {requested!r}")
