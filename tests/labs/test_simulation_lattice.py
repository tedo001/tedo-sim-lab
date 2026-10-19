"""Electrons in a lattice: the Kohn–Sham engine against exact results (lattice sums, the electron
gas, a free particle, exact diagonalization), self-consistency itself, and the density walkers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import eigh

from labs.simulation.quantum.density_walk import DensityWalkers
from labs.simulation.quantum.kohn_sham import occupations, solve
from labs.simulation.quantum.lattice import (
    LatticeSetup,
    PlaneWaves,
    ion_ion_energy,
    ion_positions,
    xc_energy_potential,
)

SMALL = {"cells": 2, "spacing": 4.0, "cutoff": 6.0}


@pytest.fixture(scope="module")
def ground() -> object:
    return solve(LatticeSetup(**SMALL))


@pytest.mark.parametrize(("lattice", "madelung"), [("square", -1.100244), ("triangular", -1.106103)])
def test_the_ion_sum_gives_the_wigner_crystal_madelung_energies(lattice: str, madelung: float) -> None:
    """Point charges on a 2-D lattice in a neutralizing background: E/N = M / r_s (Bonsall and
    Maradudin), whatever splitting parameter the Ewald sum uses."""
    setup = LatticeSetup(lattice=lattice, cells=3, spacing=1.0)
    ions, box, _ = ion_positions(setup)
    r_s = math.sqrt(box[0] * box[1] / len(ions) / math.pi)
    for eta in (0.8, 1.5, 3.0):
        energy = ion_ion_energy(ions, box, 1.0, 0.0, 0.0, eta=eta)
        assert energy / len(ions) == pytest.approx(madelung / r_s, abs=2e-6)


def test_the_capacitor_term_is_the_charge_sheets_energy() -> None:
    ions, box, _ = ion_positions(LatticeSetup(cells=2, spacing=3.0))
    apart = ion_ion_energy(ions, box, 1.0, 0.7, len(ions)) - ion_ion_energy(ions, box, 1.0, 0.0, len(ions))
    assert apart == pytest.approx(2 * math.pi * 0.7 * len(ions) ** 2 / (box[0] * box[1]))


def test_exchange_correlation_of_the_two_dimensional_electron_gas() -> None:
    r_s = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    n = 1 / (math.pi * r_s ** 2)
    eps_x, _ = xc_energy_potential(n, "exchange")
    assert np.allclose(eps_x, -4 * math.sqrt(2) / (3 * math.pi * r_s))  # exact 2-D exchange
    eps, v = xc_energy_potential(n, "lda")
    correlation = eps - eps_x
    # The high-density limit of 2-D correlation (−0.1925 Ha) and an independent QMC fit
    # (Tanatar and Ceperley 1989, in rydberg) agree with Attaccalite et al. within 2 %.
    x = np.sqrt(r_s)
    tanatar = -0.3568 * (1 + 1.13 * x) / (1 + 1.13 * x + 0.9052 * x ** 2 + 0.4165 * x ** 3) / 2
    assert np.allclose(correlation[1:], tanatar[1:], rtol=0.02)
    assert xc_energy_potential(np.array([1e8]), "lda")[0][0] - xc_energy_potential(
        np.array([1e8]), "exchange")[0][0] == pytest.approx(-0.1925, abs=2e-3)
    step = n * 1e-6  # v_xc = d(n ε_xc)/dn
    above, below = xc_energy_potential(n + step)[0], xc_energy_potential(n - step)[0]
    numeric = ((n + step) * above - (n - step) * below) / (2 * step)
    assert np.allclose(v, numeric, rtol=1e-6)
    assert np.all(xc_energy_potential(n, "hartree")[1] == 0)


def test_a_free_particle_has_the_plane_wave_energies() -> None:
    basis = PlaneWaves((7.0, 5.0), 3.0)
    energies = np.linalg.eigvalsh(basis.hamiltonian(np.zeros((basis.ny, basis.nx))))
    assert np.allclose(energies, np.sort(basis.kinetic))
    assert energies[1] == pytest.approx(0.5 * (2 * math.pi / 7.0) ** 2)  # the longest wave fits once


def test_plane_waves_match_exact_diagonalization_on_the_grid() -> None:
    """A smooth potential: the cut basis gives the low levels of the full grid Hamiltonian, from
    above (a smaller basis is variational)."""
    basis = PlaneWaves((6.0, 6.0), 12.0)
    x, y = np.meshgrid(basis.x, basis.y)
    wave = 2 * math.pi / 6.0
    v = -0.8 * np.cos(wave * x) * np.cos(wave * y) + 0.3 * np.sin(2 * wave * y)
    low = eigh(basis.hamiltonian(v), eigvals_only=True, subset_by_index=[0, 5])
    ny, nx = v.shape  # reference: −½∇² spectral on every grid point, V diagonal
    k2 = (2 * math.pi * np.fft.fftfreq(nx, 6.0 / nx))[None, :] ** 2 + (2 * math.pi * np.fft.fftfreq(
        ny, 6.0 / ny))[:, None] ** 2
    eye = np.eye(nx * ny).reshape(nx * ny, ny, nx)
    kinetic = np.real(np.fft.ifft2(0.5 * k2 * np.fft.fft2(eye))).reshape(nx * ny, nx * ny)
    exact = np.linalg.eigvalsh(kinetic + np.diag(v.ravel()))[:6]
    assert np.all(low >= exact - 1e-10) and np.allclose(low, exact, atol=1e-5)


def test_wavefunctions_are_normalized_on_the_grid() -> None:
    basis = PlaneWaves((5.0, 4.0), 4.0)
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((basis.size, 3)) + 1j * rng.standard_normal((basis.size, 3))
    vectors /= np.linalg.norm(vectors, axis=0)
    psi = basis.wavefunctions(vectors)
    assert np.allclose((np.abs(psi) ** 2).sum(axis=(1, 2)) * basis.da, 1.0)


def test_occupations_hold_every_electron() -> None:
    levels = np.array([-1.0, -0.5, -0.5, 0.2, 0.9])
    fill, mu = occupations(levels, 4.0, 0.01)
    assert fill.sum() == pytest.approx(4.0) and np.allclose(fill[:3], [2.0, 1.0, 1.0], atol=1e-6)
    assert -0.5 < mu < 0.2 or mu == pytest.approx(-0.5, abs=0.05)


def test_the_ground_state_is_self_consistent(ground) -> None:
    result = ground
    assert result.converged and result.iterations[-1].residual < result.setup.tolerance
    da = result.box[0] * result.box[1] / result.density.size
    assert result.density.min() >= 0 and result.density.sum() * da == pytest.approx(result.electrons)
    assert result.electrons == pytest.approx(4.0)  # 2 × 2 ions, Z = 1
    v_h, v_xc = result.potentials_of(result.density)  # the density makes the potential it came from
    assert np.abs(v_h - result.v_hartree).max() < 1e-3 and np.abs(v_xc - result.v_xc).max() < 1e-3
    assert np.allclose(result.v_eff, result.v_ion + result.v_hartree + result.v_xc)
    parts = sum(value for key, value in result.energies.items() if key != "total")
    assert result.energies["total"] == pytest.approx(parts)
    assert result.energies["kinetic"] > 0 and result.energies["hartree"] > 0 > result.energies["xc"]
    assert result.energies["total"] <= result.iterations[0].energy + 1e-9
    assert len(result.densities) == len(result.iterations) + 1


def test_cuts_between_grid_rows_are_exact(ground) -> None:
    y = ground.y[3]
    assert np.allclose(ground.cut(ground.v_ion, y), ground.v_ion[3])  # on a row: the row itself
    through = ground.cut(ground.v_ion)  # through the ions, which sit between rows
    ions_x = ground.ions[np.isclose(ground.ions[:, 1], ground.ion_y()), 0]
    column = np.argmin(np.abs(ground.x[:, None] - ions_x[None, :]), axis=0)
    assert through.min() <= ground.v_ion.min() + 0.02 and np.all(through[column] < through.mean())


def test_the_density_has_the_lattice_symmetry(ground) -> None:
    density, ions = ground.density, ground.ions
    ny, nx = density.shape
    assert np.allclose(np.roll(density, nx // 2, axis=1), density, atol=1e-7)  # one cell along x
    assert np.allclose(np.roll(density, ny // 2, axis=0), density, atol=1e-7)
    assert np.allclose(density, density.T, atol=1e-7)  # square: x ↔ y
    row, column = np.unravel_index(np.argmax(density), density.shape)
    spot = np.array([ground.x[column], ground.y[row]])
    assert np.min(np.hypot(*(ions - spot).T)) < 0.5  # electrons gather on the ions
    lowest = np.unravel_index(np.argmin(ground.v_ion), density.shape)
    assert np.min(np.hypot(*(ions - [ground.x[lowest[1]], ground.y[lowest[0]]]).T)) < 0.5


def test_insulators_have_a_gap_and_half_filled_bands_are_metals() -> None:
    insulator = solve(LatticeSetup(cells=2, spacing=5.0, height=0.6, charge=2, cutoff=6.0))
    assert insulator.converged and insulator.gap() is not None and insulator.gap() > 0.1
    metal = solve(LatticeSetup(cells=2, spacing=4.0, cutoff=6.0))
    assert metal.gap() is None  # Z = 1: one electron per ion half fills the lowest band


@pytest.mark.parametrize("lattice", ["triangular", "honeycomb"])
def test_other_lattices_and_a_vacancy_converge(lattice: str) -> None:
    result = solve(LatticeSetup(lattice=lattice, cells=2, spacing=4.0, cutoff=5.0))
    assert result.converged
    honeycomb_bond = 4.0 / math.sqrt(3)
    if lattice == "honeycomb":
        ions = result.ions
        gaps = np.hypot(*(ions[:, None, :] - ions[None, :, :]).transpose(2, 0, 1))
        assert np.min(gaps[gaps > 0]) == pytest.approx(honeycomb_bond)
    vacancy = solve(LatticeSetup(lattice=lattice, cells=2, spacing=4.0, cutoff=5.0, vacancy=True))
    assert vacancy.converged and len(vacancy.ions) == len(result.ions) - 1
    assert vacancy.electrons == pytest.approx(len(vacancy.ions))


def test_set_ups_are_checked_and_round_trip() -> None:
    setup = LatticeSetup(lattice="honeycomb", charge=2, vacancy=True)
    assert LatticeSetup.from_dict({**setup.to_dict(), "unknown": 1}) == setup
    for wrong in ({"lattice": "cubic"}, {"cells": 9}, {"height": 0.0}, {"functional": "gga"},
                  {"cells": 1, "vacancy": True}, {"cells": 6, "spacing": 8.0}):
        with pytest.raises(ValueError):
            LatticeSetup(**wrong).check()
    assert LatticeSetup().counts() == (16, 16, 509)


def test_a_cancelled_solve_stops() -> None:
    class Stop:
        def raise_if_cancelled(self) -> None:
            raise RuntimeError("stopped")

    with pytest.raises(RuntimeError, match="stopped"):
        solve(LatticeSetup(**SMALL), cancel=Stop())


def test_walkers_sample_the_density_and_hop(ground) -> None:
    result = ground
    walkers = DensityWalkers(result.density, result.box, result.ions, 48, step=0.08, seed=3)
    ny, nx = result.density.shape
    cells, blocks = 2, 3  # 9 grid points per cell, in 3 × 3 blocks
    cell = result.box[0] / cells
    half_step = result.box[0] / nx / 2  # a grid value stands for the square around its point
    counts = np.zeros((blocks, blocks))
    hops = []
    for _ in range(2500):
        hops += walkers.advance(1)
        local = ((walkers.positions + half_step) % cell) / (cell / blocks)
        index = local.astype(int) % blocks
        np.add.at(counts, (index[:, 1], index[:, 0]), 1)
    folded = result.density.reshape(cells, ny // cells, cells, nx // cells).sum(axis=(0, 2))
    rows, columns = folded.shape
    expected = folded.reshape(blocks, rows // blocks, blocks, columns // blocks).sum(axis=(1, 3))
    distance = 0.5 * np.abs(counts / counts.sum() - expected / expected.sum()).sum()
    assert distance < 0.04
    assert hops and all(hop.source != hop.target for hop in hops)
    assert 0.5 < walkers.acceptance <= 1.0
    assert np.all((walkers.positions >= 0) & (walkers.positions < np.array(result.box)))
