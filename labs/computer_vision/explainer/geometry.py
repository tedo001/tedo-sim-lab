"""Convolution geometry for the hyperparameter playground: how input size, padding,
kernel size and stride decide the output size and where the kernel lands.

CNN Explainer refuses a stride that does not fit evenly; PyTorch accepts it and
drops the rows and columns the kernel cannot reach. The playground shows what
PyTorch does and says which cells are left out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ConvGeometry", "LIMITS"]

#: Control → (minimum, maximum) before the cross-constraints below.
LIMITS = {"input": (3, 9), "padding": (0, 3), "kernel": (1, 9), "stride": (1, 9)}


@dataclass(frozen=True)
class ConvGeometry:
    input: int
    kernel: int
    padding: int = 0
    stride: int = 1

    @classmethod
    def fitted(cls, input: int, kernel: int, padding: int, stride: int) -> ConvGeometry:
        """The nearest valid combination: padding < kernel, kernel ≤ padded input, and a stride
        no larger than the kernel's room to move (or 1)."""
        input = _clamp(input, *LIMITS["input"])
        widest = min(LIMITS["kernel"][1], input + 2 * LIMITS["padding"][1])
        kernel = _clamp(kernel, LIMITS["kernel"][0], widest)
        padding = _clamp(padding, LIMITS["padding"][0], min(LIMITS["padding"][1], kernel - 1))
        kernel = min(kernel, input + 2 * padding)
        stride = _clamp(stride, 1, max(input + 2 * padding - kernel + 1, 1))
        return cls(input, kernel, padding, stride)

    @property
    def padded(self) -> int:
        return self.input + 2 * self.padding

    @property
    def output(self) -> int:
        return (self.padded - self.kernel) // self.stride + 1

    @property
    def leftover(self) -> int:
        """Rows (and columns) at the bottom (and right) the kernel never reaches."""
        return (self.padded - self.kernel) % self.stride

    @property
    def max_padding(self) -> int:
        return min(LIMITS["padding"][1], self.kernel - 1)

    @property
    def max_kernel(self) -> int:
        return self.padded

    @property
    def max_stride(self) -> int:
        return max(self.padded - self.kernel + 1, 1)

    def formula(self) -> str:
        return (f"⌊({self.input} + 2·{self.padding} − {self.kernel}) / {self.stride}⌋ + 1 "
                f"= {self.output}")

    def window(self, y: int, x: int) -> tuple[int, int]:
        """Top-left cell, in padded coordinates, of the kernel for output cell (y, x)."""
        return y * self.stride, x * self.stride

    def coverage(self) -> np.ndarray:
        """(padded, padded) booleans: cells some kernel position touches."""
        covered = np.zeros((self.padded, self.padded), dtype=bool)
        reach = (self.output - 1) * self.stride + self.kernel
        covered[:reach, :reach] = True
        return covered

    def is_padding(self, row: int, column: int) -> bool:
        inner = range(self.padding, self.padding + self.input)
        return row not in inner or column not in inner


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(int(value), high))
