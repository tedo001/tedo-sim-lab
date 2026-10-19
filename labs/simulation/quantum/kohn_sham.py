"""The Kohn–Sham self-consistent field for electrons in a lattice (see :mod:`.lattice`).

Guess a density; build V_eff = V_ion + V_H + V_xc; solve [−½∇² + V_eff] ψᵢ = εᵢ ψᵢ in the
plane-wave basis; fill the states, two electrons each, with Fermi smearing; the new density is
n = Σ fᵢ |ψᵢ|². Mix it with the old one (Pulay) and repeat until the density stops changing:
then the electrons are consistent with the potential they make. Every iteration's density is
kept, to watch the calculation settle.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit

from .lattice import (
    LatticeSetup,
    PlaneWaves,
    hartree_potential,
    ion_ion_energy,
    ion_positions,
    ion_potential,
    xc_energy_potential,
)

__all__ = ["ENERGY_TERMS", "Iteration", "KohnShamResult", "occupations", "solve"]

#: The parts of the total energy, in the order they are listed.
ENERGY_TERMS = {"kinetic": "Kinetic T_s", "ion": "Electron-ion", "hartree": "Hartree",
                "xc": "Exchange-correlation", "ion_ion": "Ion-ion", "total": "Total"}
_HISTORY = 6  # densities Pulay mixing remembers


@dataclass(frozen=True)
class Iteration:
    number: int
    #: ∫|n_out − n_in| dA / N: how far this iteration's density moved.
    residual: float
    #: The total energy with this iteration's output density (hartree).
    energy: float
    fermi: float


@dataclass
class KohnShamResult:
    setup: LatticeSetup
    ions: np.ndarray
    box: tuple[float, float]
    x: np.ndarray
    y: np.ndarray
    #: Fields on the grid, rows along y: the density (electrons per bohr²) and the potentials (Ha).
    density: np.ndarray
    v_ion: np.ndarray
    v_hartree: np.ndarray
    v_xc: np.ndarray
    eigenvalues: np.ndarray
    occupations: np.ndarray
    fermi: float
    #: |ψᵢ|² of every state computed (states × ny × nx).
    state_densities: np.ndarray
    energies: dict[str, float]
    iterations: list[Iteration]
    #: The density each iteration started from; the last entry is the final density.
    densities: list[np.ndarray] = field(repr=False)
    converged: bool = False
    plane_waves: int = 0
    seconds: float = 0.0

    @property
    def electrons(self) -> float:
        return float(self.occupations.sum())

    @property
    def v_eff(self) -> np.ndarray:
        return self.v_ion + self.v_hartree + self.v_xc

    def potentials_of(self, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(V_H, V_xc) that a density makes: what the electrons felt in an earlier iteration."""
        return (hartree_potential(_grid_g(self.box, density.shape), density),
                xc_energy_potential(density, self.setup.functional)[1])

    def gap(self) -> float | None:
        """The gap between the highest full and the lowest empty level; None when a level at the
        Fermi level is only partly filled (a metal)."""
        partly = (self.occupations > 0.02) & (self.occupations < 1.98)
        if partly.any():
            return None
        full = self.eigenvalues[self.occupations >= 1.98]
        empty = self.eigenvalues[self.occupations <= 0.02]
        if not len(full) or not len(empty):
            return None
        return float(empty.min() - full.max())

    def ion_y(self) -> float:
        """The height y of the first row of ions."""
        return float(self.ions[:, 1].min())

    def cut(self, field: np.ndarray, y: float | None = None) -> np.ndarray:
        """A field along the line at ``y`` (default: through the first row of ions), exactly: the
        fields are sums of plane waves, so they are known between the grid's rows too."""
        rows = field.shape[0]
        g = 2 * math.pi * np.fft.fftfreq(rows, self.box[1] / rows)
        coefficients = np.fft.fft(field, axis=0) / rows
        return np.real(np.exp(1j * g * (self.ion_y() if y is None else y)) @ coefficients)


def _grid_g(box: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
    ny, nx = shape
    gx = 2 * math.pi * np.fft.fftfreq(nx, box[0] / nx)
    gy = 2 * math.pi * np.fft.fftfreq(ny, box[1] / ny)
    return np.hypot(*np.meshgrid(gx, gy))


def occupations(eigenvalues: np.ndarray, electrons: float, smearing: float) -> tuple[np.ndarray, float]:
    """Fermi–Dirac fillings, two electrons per level, and the Fermi level μ that holds them all."""
    low, high = float(eigenvalues.min()) - 1.0, float(eigenvalues.max()) + 1.0
    for _ in range(100):
        mu = (low + high) / 2
        if 2 * expit((mu - eigenvalues) / smearing).sum() > electrons:
            high = mu
        else:
            low = mu
    mu = (low + high) / 2
    return 2 * expit((mu - eigenvalues) / smearing), mu


def _guess(basis: PlaneWaves, ions: np.ndarray, electrons: float, width: float) -> np.ndarray:
    """A start: a Gaussian blob of charge on every ion."""
    x, y = np.meshgrid(basis.x, basis.y)
    density = np.zeros_like(x)
    for x0, y0 in ions:
        dx = (x - x0 + basis.lx / 2) % basis.lx - basis.lx / 2
        dy = (y - y0 + basis.ly / 2) % basis.ly - basis.ly / 2
        density += np.exp(-(dx ** 2 + dy ** 2) / (2 * width ** 2))
    return density * electrons / (density.sum() * basis.da)


def _pulay(inputs: list[np.ndarray], residuals: list[np.ndarray], mixing: float) -> np.ndarray:
    """The next input density: the mix of earlier ones whose residual is smallest, plus a step."""
    count = len(residuals)
    flat = np.array([r.ravel() for r in residuals])
    system = np.zeros((count + 1, count + 1))
    system[:count, :count] = flat @ flat.T
    system[count, :count] = system[:count, count] = -1.0
    target = np.zeros(count + 1)
    target[count] = -1.0
    weights = np.linalg.lstsq(system, target, rcond=None)[0][:count]
    mixed = sum(w * n for w, n in zip(weights, inputs, strict=True))
    return mixed + mixing * sum(w * r for w, r in zip(weights, residuals, strict=True))


def solve(setup: LatticeSetup, progress: Callable[[float, str], None] | None = None,
          cancel=None) -> KohnShamResult:
    """Iterate to self-consistency; ``cancel`` (a CancelToken) stops between iterations."""
    setup.check()
    start = time.perf_counter()
    ions, box, cells = ion_positions(setup)
    basis = PlaneWaves(box, setup.cutoff, cells)
    electrons = float(setup.charge * len(ions))
    v_ion = ion_potential(basis, ions, setup.charge, setup.height)
    e_ii = ion_ion_energy(ions, box, setup.charge, setup.height, electrons)
    density = _guess(basis, ions, electrons, 0.35 * setup.spacing)
    states = min(basis.size, math.ceil(electrons / 2) + max(8, int(electrons) // 2))
    inputs: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    history: list[Iteration] = []
    densities: list[np.ndarray] = []
    converged = False
    for number in range(setup.max_iterations):
        if cancel is not None:
            cancel.raise_if_cancelled()
        densities.append(density)
        v_hartree = hartree_potential(basis.g, density)
        v_xc = xc_energy_potential(density, setup.functional)[1]
        v_eff = v_ion + v_hartree + v_xc
        matrix = basis.hamiltonian(v_eff)
        while True:  # enough states that the last one is empty
            eigenvalues, vectors = eigh(matrix, subset_by_index=[0, states - 1], driver="evr")
            fill, fermi = occupations(eigenvalues, electrons, setup.smearing)
            if fill[-1] < 1e-8 or states == basis.size:
                break
            states = min(basis.size, states + max(8, states // 2))
        state_densities = np.abs(basis.wavefunctions(vectors)) ** 2
        output = np.einsum("k,kij->ij", fill, state_densities)
        residual = output - density
        moved = float(np.abs(residual).sum() * basis.da / electrons)
        energies = _energies(basis, output, v_eff, v_ion, eigenvalues, fill, setup.functional, e_ii)
        history.append(Iteration(number, moved, energies["total"], fermi))
        if progress is not None:
            progress(min(0.99, (number + 1) / setup.max_iterations),
                     f"Iteration {number + 1}: density moved {moved:.1e}, E = {energies['total']:.6f} Ha")
        if moved < setup.tolerance:
            converged = True
            break
        inputs, residuals = [*inputs, density][-_HISTORY:], [*residuals, residual][-_HISTORY:]
        density = np.maximum(_pulay(inputs, residuals, setup.mixing), 0.0)
        density *= electrons / (density.sum() * basis.da)
    densities.append(output)
    return KohnShamResult(setup, ions, box, basis.x, basis.y, output, v_ion, v_hartree, v_xc, eigenvalues,
                          fill, fermi, state_densities, energies, history, densities, converged, basis.size,
                          time.perf_counter() - start)


def _energies(basis: PlaneWaves, density: np.ndarray, v_eff: np.ndarray, v_ion: np.ndarray,
              eigenvalues: np.ndarray, fill: np.ndarray, functional: str, e_ii: float) -> dict[str, float]:
    """The Kohn–Sham total energy and its parts; T_s from the band energy Σ fᵢ εᵢ − ∫ V_eff n."""
    da = basis.da
    kinetic = float((fill * eigenvalues).sum() - (v_eff * density).sum() * da)
    ion = float((v_ion * density).sum() * da)
    hartree = float(0.5 * (hartree_potential(basis.g, density) * density).sum() * da)
    xc = float((xc_energy_potential(density, functional)[0] * density).sum() * da)
    total = kinetic + ion + hartree + xc + e_ii
    return {"kinetic": kinetic, "ion": ion, "hartree": hartree, "xc": xc, "ion_ion": e_ii, "total": total}
