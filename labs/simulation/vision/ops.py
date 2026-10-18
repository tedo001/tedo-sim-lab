"""The Image lab's operations as data: each has a title, a group and typed parameters (with ranges),
so the interface builds its controls from them and a pipeline — a list of steps — is plain JSON
that replays exactly."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import filters as f

__all__ = ["OPERATIONS", "Operation", "Param", "Step", "run_pipeline"]


@dataclass(frozen=True)
class Param:
    name: str
    title: str
    kind: str  # float, int, choice, bool, kernel
    default: object
    low: float = 0.0
    high: float = 1.0
    step: float = 0.1
    choices: tuple[str, ...] = ()

    def clean(self, value: object) -> object:
        """``value`` made valid: clamped to the range, a known choice, or the default."""
        try:
            if self.kind == "float":
                return float(min(max(float(value), self.low), self.high))
            if self.kind == "int":
                return int(min(max(int(round(float(value))), self.low), self.high))
            if self.kind == "bool":
                return bool(value)
            if self.kind == "choice":
                return value if value in self.choices else self.default
            if self.kind == "kernel":
                values = tuple(float(v) for v in value)
                return values if len(values) == 9 else self.default
        except (TypeError, ValueError):
            return self.default
        return self.default


@dataclass(frozen=True)
class Operation:
    id: str
    title: str
    group: str
    fn: Callable[..., np.ndarray]
    params: tuple[Param, ...] = ()
    help: str = ""

    def defaults(self) -> dict[str, object]:
        return {p.name: p.default for p in self.params}

    def run(self, image: np.ndarray, values: dict[str, object]) -> np.ndarray:
        kwargs = {p.name: p.clean(values.get(p.name, p.default)) for p in self.params}
        return self.fn(image, **kwargs)


def _odd(name: str, title: str, default: int, high: int = 31) -> Param:
    return Param(name, title, "int", default, 1, high, 2)


_SEED = Param("seed", "Seed", "int", 0, 0, 9999, 1)

OPERATIONS: dict[str, Operation] = {op.id: op for op in (
    Operation("grey", "Greyscale", "Colour", f.grey, help="Luminance (ITU-R BT.601 weights)"),
    Operation("invert", "Invert", "Colour", f.invert),
    Operation("equalize", "Histogram equalisation", "Colour", f.equalize, help="Spreads the tones evenly"),
    Operation("gamma", "Gamma", "Colour", f.gamma, (Param("value", "Gamma", "float", 0.7, 0.1, 4.0, 0.05),)),
    Operation("brightness", "Brightness and contrast", "Colour", f.brightness_contrast,
              (Param("brightness", "Brightness", "float", 0.0, -1.0, 1.0, 0.05),
               Param("contrast", "Contrast", "float", 1.2, 0.0, 4.0, 0.05))),
    Operation("kmeans", "Colour quantisation (k-means)", "Colour", f.kmeans_colours,
              (Param("k", "Colours", "int", 4, 2, 16, 1), _SEED)),
    Operation("gaussian", "Gaussian blur", "Smooth", f.gaussian_blur,
              (Param("sigma", "Sigma (px)", "float", 1.5, 0.1, 15.0, 0.1),)),
    Operation("box", "Box blur", "Smooth", f.box_blur, (_odd("size", "Size (px)", 5),)),
    Operation("median", "Median filter", "Smooth", f.median, (_odd("size", "Size (px)", 3, 15),),
              help="Removes salt-and-pepper noise, keeps edges"),
    Operation("sharpen", "Sharpen (unsharp mask)", "Smooth", f.sharpen,
              (Param("amount", "Amount", "float", 1.0, 0.0, 5.0, 0.1),
               Param("sigma", "Radius (px)", "float", 1.0, 0.2, 10.0, 0.1))),
    Operation("sobel", "Sobel gradient", "Edges", f.sobel),
    Operation("laplacian", "Laplacian of Gaussian", "Edges", f.laplacian),
    Operation("canny", "Canny edges", "Edges", f.canny,
              (Param("sigma", "Smoothing", "float", 1.4, 0.3, 6.0, 0.1),
               Param("low", "Low threshold", "float", 0.1, 0.0, 1.0, 0.01),
               Param("high", "High threshold", "float", 0.25, 0.0, 1.0, 0.01))),
    Operation("harris", "Harris corners", "Edges", f.harris_corners,
              (Param("sigma", "Window", "float", 1.5, 0.5, 6.0, 0.1),
               Param("k", "k", "float", 0.05, 0.01, 0.2, 0.01),
               Param("threshold", "Threshold", "float", 0.05, 0.001, 0.5, 0.005))),
    Operation("threshold", "Threshold", "Segment", f.threshold,
              (Param("level", "Level", "float", 0.5, 0.0, 1.0, 0.01),
               Param("method", "Method", "choice", "fixed", choices=("fixed", "otsu")))),
    Operation("adaptive", "Adaptive threshold", "Segment", f.adaptive_threshold,
              (_odd("size", "Block (px)", 25, 151),
               Param("offset", "Offset", "float", 0.02, -0.5, 0.5, 0.01))),
    Operation("morphology", "Morphology", "Segment", f.morphology,
              (Param("operation", "Operation", "choice", "open",
                     choices=("erode", "dilate", "open", "close")),
               _odd("size", "Size (px)", 3, 21))),
    Operation("gauss_noise", "Gaussian noise", "Noise", f.noise_gaussian,
              (Param("sigma", "Sigma", "float", 0.08, 0.0, 0.5, 0.01), _SEED)),
    Operation("salt_pepper", "Salt-and-pepper noise", "Noise", f.noise_salt_pepper,
              (Param("amount", "Amount", "float", 0.05, 0.0, 0.5, 0.01), _SEED)),
    Operation("rotate", "Rotate", "Geometry", f.rotate,
              (Param("angle", "Angle (°)", "float", 15.0, -180, 180, 1),)),
    Operation("scale", "Scale", "Geometry", f.scale,
              (Param("factor", "Factor", "float", 0.5, 0.1, 4.0, 0.05),)),
    Operation("flip", "Flip", "Geometry", f.flip,
              (Param("direction", "Direction", "choice", "horizontal", choices=("horizontal", "vertical")),)),
    Operation("convolve", "Convolution (3×3 kernel)", "Frequency", f.convolve,
              (Param("kernel", "Kernel", "kernel", (0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0)),)),
    Operation("fft", "Fourier magnitude", "Frequency", f.fft_magnitude, help="log |F(u, v)|, centred"),
    Operation("frequency", "Frequency filter", "Frequency", f.frequency_filter,
              (Param("kind", "Kind", "choice", "low", choices=("low", "high")),
               Param("cutoff", "Cutoff", "float", 0.08, 0.005, 0.5, 0.005))),
)}


@dataclass
class Step:
    op: str
    params: dict[str, object] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict:
        values = {k: list(v) if isinstance(v, tuple) else v for k, v in self.params.items()}
        return {"op": self.op, "params": values, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, data: dict) -> Step:
        if data.get("op") not in OPERATIONS:
            raise ValueError(f"unknown operation {data.get('op')!r}")
        return cls(data["op"], dict(data.get("params", {})), bool(data.get("enabled", True)))


def run_pipeline(image: np.ndarray, steps: list[Step]) -> tuple[np.ndarray, list[float]]:
    """The image after every enabled step, and each step's time in milliseconds."""
    timings = []
    for step in steps:
        started = time.perf_counter()
        if step.enabled:
            image = OPERATIONS[step.op].run(image, step.params)
        timings.append((time.perf_counter() - started) * 1000)
    return image, timings
