"""Compute occupations, number of electrons, and their derivatives in Python."""

import typing as ty

import numpy as np
import numpy.typing as npt

from ..smearing import smearing_from_name


def get_max_occupation(n_spins: int) -> float:
    """Get the maximum occupation for a given number of spin channels.

    Args:
        n_spins (int): number of spin channels.

    Raises:
        ValueError: if the maximum occupation is unknown for the given number of spin channels.

    Returns:
        float: maximum occupation.
    """
    if n_spins == 1:
        max_occupation = 2.0
    elif n_spins == 2:
        max_occupation = 1.0
    else:
        raise ValueError(f"Unknown maximum occupation for n_spins={n_spins}")
    return max_occupation


def compute_occupations(
    eigenvalues: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.ArrayLike:
    """Compute the occupations given a Fermi energy and smearing.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        fermi_energy (float): fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.

    Raises:
        ValueError: if maximum occupation is unknown.

    Returns:
        npt.ArrayLike: (n_spins, n_kpoints, n_bands) occupations array.
    """
    max_occs = get_max_occupation(eigenvalues.shape[0])
    smearing = smearing_from_name(smearing_type)(center=fermi_energy, width=smearing_width)
    return max_occs * smearing.occupation(eigenvalues)


def compute_occupations_derivative(
    eigenvalues: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.ArrayLike:
    """Compute the derivative of the occupations with respect to the Fermi energy.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        fermi_energy (float): fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.

    Returns:
        npt.ArrayLike: (n_spins, n_kpoints, n_bands) occupations derivative array.
    """
    max_occs = get_max_occupation(eigenvalues.shape[0])
    smearing = smearing_from_name(smearing_type)(center=fermi_energy, width=smearing_width)
    return max_occs / smearing_width * smearing.occupation_derivative(eigenvalues)


def compute_occupations_2nd_derivative(
    eigenvalues: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.ArrayLike:
    """Compute the second derivative of the occupations with respect to the Fermi energy.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        fermi_energy (float): fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.

    Returns:
        npt.ArrayLike: (n_spins, n_kpoints, n_bands) occupations derivative array.
    """
    max_occs = get_max_occupation(eigenvalues.shape[0])
    smearing = smearing_from_name(smearing_type)(center=fermi_energy, width=smearing_width)
    return max_occs / smearing_width**2 * smearing.occupation_2nd_derivative(eigenvalues)


def compute_n_electrons(
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> float:
    r"""Compute the number of electrons (total occupation) given a Fermi energy and smearing.

    .. math::
        N_{\mathrm{el.}} = \sum_{\sigma,\mathbf{k},\nu}{\theta_{\sigma,\mathbf{k},\nu} w_{\mathbf{k}}}

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        fermi_energy (float): Fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.

    Returns:
        float: number of electrons.
    """
    occupations = compute_occupations(eigenvalues, fermi_energy, smearing_type, smearing_width)
    return np.einsum("skn,k->skn", occupations, weights).sum()


def compute_n_electrons_derivative(
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
):
    """Compute the derivative of the number of electrons (total occupation) with respect to the Fermi energy.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        fermi_energy (float): Fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.

    Returns:
        float: d[Nelec]/d[eF].
    """
    occupations_derivative = compute_occupations_derivative(eigenvalues, fermi_energy, smearing_type, smearing_width)
    return np.einsum("skn,k->skn", occupations_derivative, weights).sum()


def compute_n_electrons_2nd_derivative(
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
):
    """Compute the second derivative of the number of electrons (total occupation) with respect to the Fermi energy.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        fermi_energy (float): Fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.

    Returns:
        float: d2[Nelec]/d[eF]2.
    """
    occupations_curvature = compute_occupations_2nd_derivative(eigenvalues, fermi_energy, smearing_type, smearing_width)
    return np.einsum("skn,k->skn", occupations_curvature, weights).sum()
