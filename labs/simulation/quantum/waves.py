"""The time-dependent Schrödinger equation in one dimension (units with ħ = m = 1).

A Gaussian wave packet moves through a potential — free space, a barrier (tunnelling), a step,
a finite or infinite well, a harmonic or double well — advanced by the split-step Fourier
method: half a potential step, a kinetic step in momentum space, half a potential step. The
grid's edges absorb what reaches them (otherwise the periodic FFT would wrap it around).
Stationary states come from a finite-difference Hamiltonian.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["POTENTIALS", "USES", "WaveSetup", "WaveSimulation", "eigenstates", "potential"]

POTENTIALS = {
    "free": "Free particle",
    "barrier": "Rectangular barrier (tunnelling)",
    "step": "Potential step",
    "well": "Finite square well",
    "box": "Infinite square well (particle in a box)",
    "harmonic": "Harmonic oscillator",
    "double_well": "Double well",
}
#: The set-up values each potential depends on (the packet's are always used).
USES = {"free": (), "barrier": ("height", "width"), "step": ("height",), "well": ("height", "width"),
        "box": ("width",), "harmonic": ("omega",), "double_well": ("height", "width")}
WALL = 1e4  # an "infinite" wall, in the units above


@dataclass
class WaveSetup:
    potential: str = "barrier"
    #: Barrier / step height, well depth, or the double well's hump.
    height: float = 1.0
    #: Barrier / well width.
    width: float = 1.0
    #: The harmonic oscillator's angular frequency.
    omega: float = 0.5
    #: The packet: centre, width and mean wave number (momentum).
    x0: float = -15.0
    sigma: float = 2.0
    k0: float = 1.2
    length: float = 80.0
    points: int = 1024
    dt: float = 0.02
    absorbing: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> WaveSetup:
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        return cls(**known)

    def check(self) -> None:
        if self.potential not in POTENTIALS:
            raise ValueError(f"unknown potential {self.potential!r}")
        if not 128 <= self.points <= 16384 or self.points & (self.points - 1):
            raise ValueError("points must be a power of two from 128 to 16384")
        if self.length <= 0 or self.sigma <= 0 or self.dt <= 0 or self.width <= 0:
            raise ValueError("length, sigma, dt and width must be positive")


def potential(setup: WaveSetup, x: np.ndarray) -> np.ndarray:
    kind, height, half = setup.potential, setup.height, setup.width / 2
    if kind == "free":
        return np.zeros_like(x)
    if kind == "barrier":
        return np.where(np.abs(x) <= half, height, 0.0)
    if kind == "step":
        return np.where(x >= 0, height, 0.0)
    if kind == "well":
        return np.where(np.abs(x) <= half, -abs(height), 0.0)
    if kind == "box":
        return np.where(np.abs(x) <= half, 0.0, WALL)
    if kind == "harmonic":
        return 0.5 * setup.omega ** 2 * x ** 2
    if kind == "double_well":  # a quartic with minima at ±width/2 and a hump of `height` between
        return height * ((x / half) ** 2 - 1.0) ** 2
    raise ValueError(f"unknown potential {kind!r}")


class WaveSimulation:
    def __init__(self, setup: WaveSetup) -> None:
        setup.check()
        self.setup = setup
        n, length = setup.points, setup.length
        self.x = np.linspace(-length / 2, length / 2, n, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        self.k = 2 * np.pi * np.fft.fftfreq(n, d=self.dx)
        self.v = potential(setup, self.x)
        self.time = 0.0
        self.psi = self.packet()
        self._half_v = np.exp(-0.5j * self.v * setup.dt)
        self._kinetic = np.exp(-0.5j * self.k ** 2 * setup.dt)
        edge = 0.1 * length
        distance = np.minimum(self.x - self.x[0], self.x[-1] - self.x)
        ramp = np.clip(distance / edge, 0.0, 1.0)
        self._mask = (np.sin(0.5 * np.pi * ramp) ** 0.08) if setup.absorbing else None

    def packet(self) -> np.ndarray:
        s = self.setup
        psi = np.exp(-((self.x - s.x0) ** 2) / (4 * s.sigma ** 2) + 1j * s.k0 * self.x)
        return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * self.dx)

    def step(self, count: int = 1) -> None:
        psi = self.psi
        for _ in range(max(1, count)):
            psi = self._half_v * psi
            psi = np.fft.ifft(self._kinetic * np.fft.fft(psi))
            psi = self._half_v * psi
            if self._mask is not None:
                psi = psi * self._mask
        self.psi = psi
        self.time += self.setup.dt * max(1, count)

    # Observables ------------------------------------------------------------------------------------
    def density(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

    def norm(self) -> float:
        return float(np.sum(self.density()) * self.dx)

    def mean_x(self) -> float:
        return float(np.sum(self.x * self.density()) * self.dx / max(self.norm(), 1e-300))

    def mean_p(self) -> float:
        phi = np.fft.fft(self.psi)
        weight = np.abs(phi) ** 2
        return float(np.sum(self.k * weight) / max(np.sum(weight), 1e-300))

    def energy(self) -> float:
        """⟨H⟩ = ⟨T⟩ + ⟨V⟩ of what is still on the grid."""
        phi = np.fft.fft(self.psi)
        weight = np.abs(phi) ** 2
        kinetic = np.sum(0.5 * self.k ** 2 * weight) / max(np.sum(weight), 1e-300)
        potential_energy = np.sum(self.v * self.density()) * self.dx / max(self.norm(), 1e-300)
        return float(kinetic + potential_energy)

    def beyond(self, position: float) -> float:
        """Probability to the right of ``position`` (transmission past a barrier)."""
        return float(np.sum(self.density()[self.x > position]) * self.dx)

    def before(self, position: float) -> float:
        return float(np.sum(self.density()[self.x < position]) * self.dx)


def eigenstates(setup: WaveSetup, count: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lowest ``count`` stationary states: (energies, x, wavefunctions as columns), from
    H = −½ d²/dx² + V on the grid with the wavefunction zero at both ends."""
    from scipy.linalg import eigh_tridiagonal

    n = min(setup.points, 2048)
    x = np.linspace(-setup.length / 2, setup.length / 2, n)
    dx = x[1] - x[0]
    v = potential(setup, x)
    diagonal = 1.0 / dx ** 2 + v
    off = np.full(n - 1, -0.5 / dx ** 2)
    count = max(1, min(count, n - 1))
    energies, vectors = eigh_tridiagonal(diagonal, off, select="i", select_range=(0, count - 1))
    vectors = vectors / np.sqrt(dx)  # normalised: ∫|ψ|² dx = 1
    for column in range(vectors.shape[1]):  # a sign convention, so plots do not flip
        if vectors[np.argmax(np.abs(vectors[:, column])), column] < 0:
            vectors[:, column] *= -1
    return energies, x, vectors
