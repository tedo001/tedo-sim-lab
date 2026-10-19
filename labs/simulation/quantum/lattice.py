"""Electrons in a crystal lattice, the set-up: where the ions are, the plane-wave basis and its
grid, and the three parts of the Kohn–Sham effective potential V_eff = V_ion + V_H + V_xc.

Hartree atomic units (ħ = mₑ = e = 1): lengths in bohr, energies in hartree. The electrons move in
a plane that repeats in x and y (periodic boundaries). Each ion has charge Z and sits a height d
above the plane, so an electron a distance r along the plane feels −Z/√(r² + d²); electrons repel
each other with the Coulomb 1/r. Exchange-correlation is the local density approximation of the
two-dimensional electron gas: exchange exactly, correlation from the quantum Monte Carlo fit of
Attaccalite, Moroni, Gori-Giorgi and Bachelet (Phys. Rev. Lett. 88, 256601, 2002). The uniform
(G = 0) parts of the potentials are left out, as in any plane-wave code: the cell is neutral, and
their energy is in the ion-ion term.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import erfc

__all__ = ["FUNCTIONALS", "LATTICES", "MAX_PLANE_WAVES", "LatticeSetup", "PlaneWaves", "hartree_potential",
           "ion_ion_energy", "ion_positions", "ion_potential", "xc_energy_potential"]

LATTICES = {"square": "Square", "triangular": "Triangular (hexagonal close-packed)",
            "honeycomb": "Honeycomb (graphene-like)"}
FUNCTIONALS = {"lda": "LDA: exchange and correlation", "exchange": "LDA exchange only",
               "hartree": "None (Hartree only)"}
#: The largest basis solved densely (about a second per iteration on a laptop).
MAX_PLANE_WAVES = 1600
TWO_PI = 2 * math.pi
#: Correlation of the unpolarized 2-D electron gas, α₀(r_s) of Attaccalite et al. (hartree).
_AMGB = {"A": -0.1925, "B": 0.0863136, "C": 0.0572384, "E": 1.0022, "F": -0.02069, "G": 0.33997,
         "H": 0.01747}


@dataclass
class LatticeSetup:
    lattice: str = "square"
    #: Unit cells along x; along y as many as keep the region close to square.
    cells: int = 4
    #: The lattice constant a (bohr): the distance between equivalent ions.
    spacing: float = 5.0
    #: Ion charge Z: the electrons each ion brings.
    charge: int = 1
    #: The ions' height d above the plane (bohr): smaller makes deeper wells.
    height: float = 1.0
    functional: str = "lda"
    #: Leave out the ion nearest the middle (a vacancy).
    vacancy: bool = False
    #: Plane waves up to this kinetic energy (hartree).
    cutoff: float = 8.0
    #: Fermi smearing kT (hartree): levels near the Fermi level share their electrons.
    smearing: float = 0.01
    mixing: float = 0.5
    max_iterations: int = 60
    #: Converged when ∫|n_out − n_in| dA is below this fraction of the electrons.
    tolerance: float = 1e-6

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LatticeSetup:
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})

    def check(self) -> None:
        if self.lattice not in LATTICES:
            raise ValueError(f"unknown lattice {self.lattice!r}")
        if self.functional not in FUNCTIONALS:
            raise ValueError(f"unknown functional {self.functional!r}")
        limits = (("cells", 1, 6), ("spacing", 2.0, 12.0), ("charge", 1, 4), ("height", 0.2, 5.0),
                  ("cutoff", 2.0, 40.0), ("smearing", 1e-4, 0.2), ("mixing", 0.05, 1.0),
                  ("max_iterations", 1, 500), ("tolerance", 1e-10, 0.1))
        for name, low, high in limits:
            if not low <= getattr(self, name) <= high:
                raise ValueError(f"{name} must be from {low} to {high}")
        if self.vacancy and len(ion_positions(self)[0]) < 2:
            raise ValueError("a vacancy needs at least two ions: use more cells")
        waves = len(_basis(ion_positions(self)[1], self.cutoff)[2])
        if waves > MAX_PLANE_WAVES:
            raise ValueError(f"this needs {waves} plane waves (at most {MAX_PLANE_WAVES}): lower the cutoff, "
                             "or use fewer or smaller cells")

    def counts(self) -> tuple[int, int, int]:
        """(ions, electrons, plane waves) of this set-up."""
        ions, box, _cells = ion_positions(self)
        return len(ions), self.charge * len(ions), len(_basis(box, self.cutoff)[2])


def _cell(setup: LatticeSetup) -> tuple[tuple[float, float], list[tuple[float, float]], int]:
    """The rectangular cell (width, height), the ions in it (centred), and how many cells along y."""
    a = setup.spacing
    if setup.lattice == "square":
        return (a, a), [(a / 2, a / 2)], setup.cells
    tall, rows = math.sqrt(3) * a, max(1, round(setup.cells / math.sqrt(3)))
    if setup.lattice == "triangular":
        return (a, tall), [(a / 4, tall / 4), (3 * a / 4, 3 * tall / 4)], rows
    half_bond = a / math.sqrt(3) / 2  # honeycomb: two triangular sublattices one bond apart
    return (a, tall), [(a / 4, tall / 4 - half_bond), (a / 4, tall / 4 + half_bond),
                       (3 * a / 4, 3 * tall / 4 - half_bond), (3 * a / 4, 3 * tall / 4 + half_bond)], rows


def ion_positions(setup: LatticeSetup) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """(N × 2 positions in bohr, the periodic box (Lx, Ly), the cells along x and y)."""
    (width, tall), basis, rows = _cell(setup)
    ions = np.array([(i * width + bx, j * tall + by)
                     for j in range(rows) for i in range(setup.cells) for bx, by in basis])
    box = (setup.cells * width, rows * tall)
    if setup.vacancy and len(ions) > 1:
        middle = np.array(box) / 2
        ions = np.delete(ions, int(np.argmin(np.hypot(*(ions - middle).T))), axis=0)
    return ions, box, (setup.cells, rows)


def _smooth_size(n: int, multiple: int) -> int:
    """The smallest even size ≥ n that is a multiple of ``multiple`` with only 2, 3, 5 as factors."""
    n = max(n, 2)
    while True:
        if n % multiple == 0 and n % 2 == 0:
            rest = n
            for p in (2, 3, 5):
                while rest % p == 0:
                    rest //= p
            if rest == 1:
                return n
        n += 1


def _basis(box: tuple[float, float], cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """The plane waves' integer indices (i, j), their kinetic energies, and the largest |i|, |j|."""
    lx, ly = box
    reach = math.sqrt(2 * cutoff)
    imax, jmax = int(reach * lx / TWO_PI), int(reach * ly / TWO_PI)
    i, j = np.meshgrid(np.arange(-imax, imax + 1), np.arange(-jmax, jmax + 1))
    kinetic = 0.5 * ((TWO_PI * i / lx) ** 2 + (TWO_PI * j / ly) ** 2)
    keep = kinetic <= cutoff
    return i[keep], j[keep], kinetic[keep], imax, jmax


class PlaneWaves:
    """The basis: plane waves e^{iG·r} with ½|G|² ≤ cutoff, G on the region's reciprocal lattice
    (its Γ point, equivalent to a k-point grid of one cell per cell in the region). The FFT grid
    holds every G − G′, so densities and potentials are free of aliasing, and its size is a
    multiple of the cells so every ion sits on the same spot of the grid."""

    def __init__(self, box: tuple[float, float], cutoff: float, cells: tuple[int, int] = (1, 1)) -> None:
        self.lx, self.ly = box
        self.area = self.lx * self.ly
        self.i, self.j, self.kinetic, imax, jmax = _basis(box, cutoff)
        if len(self.kinetic) > MAX_PLANE_WAVES:
            raise ValueError(f"this needs {len(self.kinetic)} plane waves (at most {MAX_PLANE_WAVES}): "
                             "lower the cutoff, or use fewer or smaller cells")
        self.nx = _smooth_size(4 * imax + 2, cells[0])
        self.ny = _smooth_size(4 * jmax + 2, cells[1])
        self.x = np.arange(self.nx) * self.lx / self.nx
        self.y = np.arange(self.ny) * self.ly / self.ny
        self.da = self.area / (self.nx * self.ny)
        gx = TWO_PI * np.fft.fftfreq(self.nx, self.lx / self.nx)
        gy = TWO_PI * np.fft.fftfreq(self.ny, self.ly / self.ny)
        self.gx, self.gy = np.meshgrid(gx, gy)
        self.g = np.hypot(self.gx, self.gy)
        self._slots = (self.j % self.ny, self.i % self.nx)
        self._pairs = (((self.j[:, None] - self.j[None, :]) % self.ny) * self.nx
                       + (self.i[:, None] - self.i[None, :]) % self.nx).astype(np.int32)

    @property
    def size(self) -> int:
        return len(self.kinetic)

    def hamiltonian(self, v_eff: np.ndarray) -> np.ndarray:
        """⟨G|−½∇² + V_eff|G′⟩ = ½|G|² δ + V_eff(G − G′)."""
        coefficients = np.fft.fft2(v_eff).ravel() / v_eff.size
        matrix = coefficients[self._pairs]
        matrix[np.diag_indices_from(matrix)] += self.kinetic
        return matrix

    def wavefunctions(self, vectors: np.ndarray) -> np.ndarray:
        """Basis coefficients (size × k, unit norm) → ψ on the grid (k × ny × nx), ∫|ψ|² dA = 1."""
        grid = np.zeros((vectors.shape[1], self.ny, self.nx), dtype=complex)
        grid[:, self._slots[0], self._slots[1]] = vectors.T
        return np.fft.ifft2(grid) * (self.nx * self.ny / math.sqrt(self.area))


def _coulomb(g: np.ndarray, height: float = 0.0) -> np.ndarray:
    """The 2-D Fourier transform of 1/√(r² + h²), 2π e^{−h|G|}/|G|, without G = 0."""
    kernel = np.zeros_like(g)
    nonzero = g > 0
    kernel[nonzero] = TWO_PI * np.exp(-height * g[nonzero]) / g[nonzero]
    return kernel


def ion_potential(basis: PlaneWaves, ions: np.ndarray, charge: float, height: float) -> np.ndarray:
    """V_ion(r) = −Σ Z/√(|r − R|² + d²) over the ions and their periodic images."""
    phase = np.zeros(basis.g.shape, dtype=complex)
    for x, y in ions:
        phase += np.exp(-1j * (basis.gx * x + basis.gy * y))
    coefficients = -charge * _coulomb(basis.g, height) * phase / basis.area
    return np.real(np.fft.ifft2(coefficients)) * basis.g.size


def hartree_potential(g: np.ndarray, density: np.ndarray) -> np.ndarray:
    """V_H(r) = ∫ n(r′)/|r − r′| dA′, the electrons' own electrostatic potential (``g``: |G| on the
    density's grid, :attr:`PlaneWaves.g`)."""
    return np.real(np.fft.ifft2(_coulomb(g) * np.fft.fft2(density)))


def xc_energy_potential(density: np.ndarray, functional: str = "lda") -> tuple[np.ndarray, np.ndarray]:
    """(ε_xc per electron, v_xc = d(n ε_xc)/dn) of the uniform 2-D gas at each point's density."""
    n = np.maximum(np.asarray(density, dtype=np.float64), 1e-14)
    if functional == "hartree":
        return np.zeros_like(n), np.zeros_like(n)
    eps = -(4 / 3) * math.sqrt(2 / math.pi) * np.sqrt(n)  # exchange: −4√2/(3π r_s)
    v = 1.5 * eps
    if functional == "exchange":
        return eps, v
    a, b, c, e, f, g, h = (_AMGB[k] for k in "ABCEFGH")
    d = -a * h
    rs = 1 / np.sqrt(math.pi * n)
    p, q = b * rs + c * rs ** 2 + d * rs ** 3, e * rs + f * rs ** 1.5 + g * rs ** 2 + h * rs ** 3
    log = np.log1p(1 / q)
    correlation = a + p * log
    slope = (b + 2 * c * rs + 3 * d * rs ** 2) * log - p * (e + 1.5 * f * np.sqrt(rs) + 2 * g * rs
                                                             + 3 * h * rs ** 2) / (q * q + q)
    return eps + correlation, v + correlation - rs / 2 * slope


def ion_ion_energy(ions: np.ndarray, box: tuple[float, float], charge: float, height: float,
                   electrons: float, eta: float | None = None) -> float:
    """The ions' Coulomb energy (Ewald sum, with the uniform background the left-out G = 0 terms
    imply) plus the energy of the charge sheets: ions a height d above the electrons' plane hold
    2π d N²/A, like a parallel-plate capacitor."""
    lx, ly = box
    area, count = lx * ly, len(ions)
    eta = eta or 2.0 / math.sqrt(area / count)
    reach = 7.0 / eta
    diff = ions[:, None, :] - ions[None, :, :]
    real = 0.0
    for i in range(-int(reach / lx) - 1, int(reach / lx) + 2):
        for j in range(-int(reach / ly) - 1, int(reach / ly) + 2):
            r = np.hypot(diff[..., 0] + i * lx, diff[..., 1] + j * ly)
            r = r[(r > 1e-12) & (r < reach)]
            real += 0.5 * np.sum(erfc(eta * r) / r)
    limit = 14.0 * eta
    i, j = np.meshgrid(np.arange(-int(limit * lx / TWO_PI) - 1, int(limit * lx / TWO_PI) + 2),
                       np.arange(-int(limit * ly / TWO_PI) - 1, int(limit * ly / TWO_PI) + 2))
    gx, gy = (TWO_PI * i / lx).ravel(), (TWO_PI * j / ly).ravel()
    g = np.hypot(gx, gy)
    use = (g > 0) & (g <= limit)
    structure = np.exp(-1j * (np.outer(gx[use], ions[:, 0]) + np.outer(gy[use], ions[:, 1]))).sum(axis=1)
    reciprocal = math.pi / area * np.sum(np.abs(structure) ** 2 * erfc(g[use] / (2 * eta)) / g[use])
    self_energy = eta / math.sqrt(math.pi) * count
    background = math.sqrt(math.pi) * count ** 2 / (eta * area)
    ewald = charge ** 2 * (real + reciprocal - self_energy - background)
    return float(ewald + TWO_PI * height * electrons ** 2 / area)
