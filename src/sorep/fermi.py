"""Fermi level solver."""
import typing as ty
import numpy as np
import numpy.typing as npt
import scipy as sp

from .smearing import smearing_from_name

__all__ = ('find_fermi_energy_bisection', 'find_fermi_energy_two_stage')


def _compute_n_electrons(
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
):
    occupations = compute_occupations(bands, fermi_energy, smearing_type,
                                      smearing_width)
    return np.einsum('skn,k->skn', occupations, weights).sum()


def _compute_n_electrons_derivative(
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
):
    occupations_derivative = compute_occupations_derivative(
        bands, fermi_energy, smearing_type, smearing_width)
    return np.einsum('skn,k->skn', occupations_derivative, weights).sum()


def _compute_n_electrons_curvature(
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
):
    occupations_curvature = compute_occupations_curvature(
        bands, fermi_energy, smearing_type, smearing_width)
    return np.einsum('skn,k->skn', occupations_curvature, weights).sum()


def compute_occupations(
    bands: npt.NDArray[np.float64],
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.NDArray[np.float64]:
    """Compute the occupations given a Fermi energy and smearing.

    Args:
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        fermi_energy (float): fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.

    Raises:
        ValueError: if maximum occupation is unknown.

    Returns:
        npt.NDArray[np.float64]: (n_spins, n_kpoints, n_bands) occupations array.
    """
    smearing = smearing_from_name(smearing_type)(center=fermi_energy,
                                                 width=smearing_width)
    return smearing.occupation(bands)


# TODO: check correctness!
def compute_occupations_derivative(
    bands: npt.NDArray[np.float64],
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.NDArray[np.float64]:
    smearing = smearing_from_name(smearing_type)(center=fermi_energy,
                                                 width=smearing_width)
    return 1 / smearing_width * smearing.occupation_derivative(bands)


# TODO: check correctness!
def compute_occupations_curvature(
    bands: npt.NDArray[np.float64],
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.NDArray[np.float64]:
    smearing = smearing_from_name(smearing_type)(center=fermi_energy,
                                                 width=smearing_width)
    return 1 / smearing_width**2 * smearing.occupation_curvature(bands)


def find_fermi_energy_bisection(
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    smearing_type: ty.Union[str, int],
    smearing_width: float,
    n_electrons: int,
):
    """Find the Fermi level by bisection.

    Adapated from DFTK.jl/src/occupation.jl.

    Args:
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.NDArray[np.float64]): (n_kpoints, ) k-point weights array.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width
        n_electrons (int): target number of electrons
    """

    def objective(ef):
        ne = _compute_n_electrons(bands, weights, ef, smearing_type, smearing_width)
        return ne - n_electrons

    # Get rough bounds for the Fermi level
    e_min = bands.min() - 10
    e_max = bands.max() + 10
    assert objective(e_min) < 0 < objective(e_max)

    fermi_energy = sp.optimize.bisect(objective, e_min, e_max)
    return fermi_energy


def find_fermi_energy_two_stage(
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    smearing_type: ty.Union[str, int],
    smearing_width: float,
    n_electrons: int,
) -> float:
    """Find the Fermi level using a two-stage algorithm which starts from a bisection with Gaussian
    smearing and follows up with a Newton refinement with the requested smearing.

    Adapated from DFTK.jl/src/occupation.jl.

    Args:
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.NDArray[np.float64]): (n_kpoints, ) k-point weights array.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width
        n_electrons (int): target number of electrons
    """
    # Start with bisection and Gaussian smearing
    fermi_energy = find_fermi_energy_bisection(bands, weights, 'gauss',
                                               smearing_width, n_electrons)

    # Refine with Newton and the requested smearing (probably cold)
    def objective(ef):
        return _compute_n_electrons_error(bands, weights, ef, smearing_type,
                                          smearing_width, n_electrons)

    fermi_energy = sp.optimize.newton(objective, fermi_energy)
    return fermi_energy
