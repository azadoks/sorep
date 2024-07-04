"""Tests for the `sorep.fermi` module."""

import jax
import numpy as np
import pytest  # pylint: disable=unused-import

import sorep


def test_compute_occupations(bands_data):
    """Test `sorep.fermi.compute_occupations."""
    occupations = sorep.fermi.compute_occupations(
        bands=bands_data["bands"],
        fermi_energy=bands_data["fermi_energy"],
        smearing_type=bands_data["smearing_type"],
        smearing_width=bands_data["smearing_width"],
    )

    assert np.allclose(occupations, bands_data["occupations"], atol=1e-3)


def test_compute_n_electrons(bands_data):
    """Test `sorep.fermi.compute_n_electrons`."""
    n_electrons = sorep.fermi.compute_n_electrons(
        bands=bands_data["bands"],
        weights=bands_data["weights"],
        fermi_energy=bands_data["fermi_energy"],
        smearing_type=bands_data["smearing_type"],
        smearing_width=bands_data["smearing_width"],
    )

    assert n_electrons == pytest.approx(bands_data["n_electrons"], abs=1e-2)


def test_compute_n_electrons_derivative(bands_data):
    """Test `sorep.fermi.compute_n_electrons_derivative`."""
    n_electrons_derivative = sorep.fermi.compute_n_electrons_derivative(
        bands=bands_data["bands"],
        weights=bands_data["weights"],
        fermi_energy=bands_data["fermi_energy"],
        smearing_type=bands_data["smearing_type"],
        smearing_width=bands_data["smearing_width"],
    )

    expected = jax.grad(
        lambda eF: sorep.fermi.compute_n_electrons(
            bands=bands_data["bands"],
            weights=bands_data["weights"],
            fermi_energy=eF,
            smearing_type=bands_data["smearing_type"],
            smearing_width=bands_data["smearing_width"],
        ),
        argnums=0,
    )(bands_data["fermi_energy"])

    assert n_electrons_derivative == pytest.approx(expected, abs=1e-1)


def test_compute_n_electrons_2nd_derivative(bands_data):
    """Test `sorep.fermi.compute_n_electrons_derivative`."""
    n_electrons_2nd_derivative = sorep.fermi.compute_n_electrons_2nd_derivative(
        bands=bands_data["bands"],
        weights=bands_data["weights"],
        fermi_energy=bands_data["fermi_energy"],
        smearing_type=bands_data["smearing_type"],
        smearing_width=bands_data["smearing_width"],
    )

    expected = jax.jacfwd(
        lambda eF: sorep.fermi.compute_n_electrons(
            bands=bands_data["bands"],
            weights=bands_data["weights"],
            fermi_energy=eF,
            smearing_type=bands_data["smearing_type"],
            smearing_width=bands_data["smearing_width"],
        ),
        argnums=0,
    )(bands_data["fermi_energy"])

    assert n_electrons_2nd_derivative == pytest.approx(expected, abs=1e-1)
