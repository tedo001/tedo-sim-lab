"""Electrons drawn as points that move: samples of the Kohn–Sham density.

Each walker takes Metropolis-adjusted Langevin steps: a drift up the gradient of ln n, a random
step, and the Metropolis–Hastings test, so in the long run the walkers are spread exactly as
n(r)/N. They are not classical electrons — where they spend their time *is* the averaged density —
but they show how it comes about: they rattle around an ion and now and then hop to the next one
through the density between them. A walker belongs to an ion until it comes close to another (a
third of the way between the closest ions), so rattling across a border is not a hop.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["DensityWalkers", "Hop"]


class Hop(NamedTuple):
    walker: int
    #: Ion indices; the walk is periodic, so the new ion can be an image across the edge.
    source: int
    target: int


class DensityWalkers:
    """``count`` walkers on a periodic density (ny × nx over the box)."""

    def __init__(self, density: np.ndarray, box: tuple[float, float], ions: np.ndarray, count: int, *,
                 step: float = 0.02, seed: int = 0) -> None:
        self.box = np.asarray(box, dtype=np.float64)
        self.ions = np.asarray(ions, dtype=np.float64)
        self.step_size = step  # τ (bohr²): each step's random part has variance τ per direction
        self.rng = np.random.default_rng(seed)
        ny, nx = density.shape
        self.cell = self.box / np.array([nx, ny])
        floor = max(float(density.max()) * 1e-9, 1e-300)
        self.log = np.log(np.maximum(density, floor))
        self.grad_x = (np.roll(self.log, -1, axis=1) - np.roll(self.log, 1, axis=1)) / (2 * self.cell[0])
        self.grad_y = (np.roll(self.log, -1, axis=0) - np.roll(self.log, 1, axis=0)) / (2 * self.cell[1])
        weights = np.maximum(density, 0).ravel()
        picks = self.rng.choice(weights.size, size=count, p=weights / weights.sum())
        rows, columns = np.divmod(picks, nx)
        jitter = self.rng.uniform(-0.5, 0.5, (count, 2))
        self.positions = (np.stack([columns, rows], axis=1) + jitter) * self.cell % self.box
        if len(self.ions) > 1:
            apart = self._offsets(self.ions)
            np.fill_diagonal(apart, np.inf)
            self.core = float(apart.min()) / 3
        else:
            self.core = float(self.box.min()) / 3
        self.home = self.nearest_ions(self.positions)[0]
        self.accepted = 0
        self.proposed = 0

    def _interpolate(self, field: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Bilinear, periodic."""
        ny, nx = field.shape
        u, v = points[:, 0] / self.cell[0], points[:, 1] / self.cell[1]
        i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u - i0, v - j0
        i0, j0 = i0 % nx, j0 % ny
        i1, j1 = (i0 + 1) % nx, (j0 + 1) % ny
        return ((1 - fu) * (1 - fv) * field[j0, i0] + fu * (1 - fv) * field[j0, i1]
                + (1 - fu) * fv * field[j1, i0] + fu * fv * field[j1, i1])

    def _drift(self, points: np.ndarray) -> np.ndarray:
        return np.stack([self._interpolate(self.grad_x, points), self._interpolate(self.grad_y, points)],
                        axis=1)

    def _offsets(self, points: np.ndarray) -> np.ndarray:
        """Distances (points × ions) to each ion's nearest periodic image."""
        offset = points[:, None, :] - self.ions[None, :, :]
        offset -= np.round(offset / self.box) * self.box
        return np.hypot(offset[..., 0], offset[..., 1])

    def nearest_ions(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(the nearest ion of each point, its distance)."""
        distances = self._offsets(points)
        nearest = np.argmin(distances, axis=1)
        return nearest, distances[np.arange(len(points)), nearest]

    def advance(self, steps: int = 1) -> list[Hop]:
        """Move every walker ``steps`` times; the hops made on the way."""
        tau = self.step_size
        hops: list[Hop] = []
        for _ in range(steps):
            here = self.positions
            drift_here = self._drift(here)
            moved = here + 0.5 * tau * drift_here + np.sqrt(tau) * self.rng.standard_normal(here.shape)
            there = moved % self.box
            drift_there = self._drift(there)
            forward = np.sum((moved - here - 0.5 * tau * drift_here) ** 2, axis=1)
            backward = np.sum((here - moved - 0.5 * tau * drift_there) ** 2, axis=1)
            log_ratio = (self._interpolate(self.log, there) - self._interpolate(self.log, here)
                         - (backward - forward) / (2 * tau))
            keep = np.log(self.rng.random(len(here))) < log_ratio
            self.positions = np.where(keep[:, None], there, here)
            self.proposed += len(here)
            self.accepted += int(keep.sum())
            nearest, distance = self.nearest_ions(self.positions)
            for walker in np.flatnonzero((nearest != self.home) & (distance < self.core)):
                hops.append(Hop(int(walker), int(self.home[walker]), int(nearest[walker])))
                self.home[walker] = nearest[walker]
        return hops

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 1.0
