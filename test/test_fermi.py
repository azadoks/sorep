"""Tests for the `sorep.fermi` module."""

import jax
import numpy as np
import pytest  # pylint: disable=unused-import

import sorep


def test_compute_occupations(bandstructure):
    """Test `sorep.fermi.compute_occupations."""
    bandstructure, smearing_type, smearing_width = bandstructure
    occupations = sorep.fermi.compute_occupations(
        bands=bandstructure.bands,
        fermi_energy=bandstructure.fermi_energy,
        smearing_type=smearing_type,
        smearing_width=smearing_width,
    )

    assert np.allclose(occupations, bandstructure.occupations, atol=1e-3)


def test_compute_n_electrons(bandstructure):
    """Test `sorep.fermi.compute_n_electrons`."""
    bandstructure, smearing_type, smearing_width = bandstructure
    n_electrons = sorep.fermi.compute_n_electrons(
        bands=bandstructure.bands,
        weights=bandstructure.weights,
        fermi_energy=bandstructure.fermi_energy,
        smearing_type=smearing_type,
        smearing_width=smearing_width,
    )

    assert n_electrons == pytest.approx(bandstructure.n_electrons, abs=1e-2)


def test_compute_n_electrons_derivative(bandstructure):
    """Test `sorep.fermi.compute_n_electrons_derivative`."""
    bandstructure, smearing_type, smearing_width = bandstructure
    n_electrons_derivative = sorep.fermi.compute_n_electrons_derivative(
        bands=bandstructure.bands,
        weights=bandstructure.weights,
        fermi_energy=bandstructure.fermi_energy,
        smearing_type=smearing_type,
        smearing_width=smearing_width,
    )

    expected = jax.grad(
        lambda eF: sorep.fermi.compute_n_electrons(
            bands=bandstructure.bands,
            weights=bandstructure.weights,
            fermi_energy=eF,
            smearing_type=smearing_type,
            smearing_width=smearing_width,
        ),
        argnums=0,
    )(bandstructure.fermi_energy)

    assert n_electrons_derivative == pytest.approx(expected, abs=1e-5)


def test_compute_n_electrons_2nd_derivative(bandstructure):
    """Test `sorep.fermi.compute_n_electrons_derivative`."""
    bandstructure, smearing_type, smearing_width = bandstructure
    n_electrons_2nd_derivative = sorep.fermi.compute_n_electrons_2nd_derivative(
        bands=bandstructure.bands,
        weights=bandstructure.weights,
        fermi_energy=bandstructure.fermi_energy,
        smearing_type=smearing_type,
        smearing_width=smearing_width,
    )

    expected = -jax.hessian(  # pylint: disable=invalid-unary-operand-type
        lambda eF: sorep.fermi.compute_n_electrons(
            bands=bandstructure.bands,
            weights=bandstructure.weights,
            fermi_energy=eF,
            smearing_type=smearing_type,
            smearing_width=smearing_width,
        ),
        argnums=0,
    )(bandstructure.fermi_energy)

    assert n_electrons_2nd_derivative == pytest.approx(expected, abs=1e-1)


def test_find_fermi_energy(bandstructure):
    """Test `sorep.fermi.find_fermi_energy`."""
    bandstructure, smearing_type, smearing_width = bandstructure
    bands = sorep.BandStructure(
        bands=bandstructure.bands,
        kpoints=bandstructure.kpoints,
        weights=bandstructure.weights,
        cell=bandstructure.cell,
        occupations=bandstructure.occupations,
        fermi_energy=bandstructure.fermi_energy,
        n_electrons=bandstructure.n_electrons,
    )
    fermi_energy = sorep.fermi.find_fermi_energy(
        bands=bandstructure.bands,
        weights=bandstructure.weights,
        n_electrons=bandstructure.n_electrons,
        smearing_type=smearing_type,
        smearing_width=smearing_width,
    )

    assert sorep.fermi.compute_n_electrons(
        bands=bandstructure.bands,
        weights=bandstructure.weights,
        fermi_energy=fermi_energy,
        smearing_type=smearing_type,
        smearing_width=smearing_width,
    ) == pytest.approx(bandstructure.n_electrons, abs=1e-4)

    if bands.is_metallic():
        assert fermi_energy == pytest.approx(bandstructure.fermi_energy, abs=1e-3)
    else:
        assert bands.vbm < fermi_energy < bands.cbm
