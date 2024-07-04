"""Fermi level solvers."""

import typing as ty

import numpy as np
import numpy.typing as npt
import scipy as sp

from .smearing import Cold, Delta, FermiDirac, Gaussian, smearing_from_name

__all__ = (
    "find_fermi_energy",
    "find_fermi_energy_zero_temp",
    "find_fermi_energy_bisection",
    "find_fermi_energy_two_stage",
    "compute_occupations",
    "compute_occupations_derivative",
    "compute_occupations_2nd_derivative",
    "compute_n_electrons",
    "compute_n_electrons_derivative",
    "compute_n_electrons_curvature",
)


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
    smearing = smearing_from_name(smearing_type)(center=fermi_energy, width=smearing_width)
    return smearing.occupation(bands)


#! This may not be correct, should be checked
def compute_occupations_derivative(
    bands: npt.NDArray[np.float64],
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.NDArray[np.float64]:
    """Compute the derivative of the occupations with respect to the Fermi energy.

    Args:
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        fermi_energy (float): fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.

    Returns:
        npt.NDArray[np.float64]: (n_spins, n_kpoints, n_bands) occupations derivative array.
    """
    smearing = smearing_from_name(smearing_type)(center=fermi_energy, width=smearing_width)
    return 1 / smearing_width * smearing.occupation_derivative(bands)


#! This may not be correct, should be checked
def compute_occupations_2nd_derivative(
    bands: npt.NDArray[np.float64],
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.NDArray[np.float64]:
    """Compute the second derivative of the occupations with respect to the Fermi energy.

    Args:
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        fermi_energy (float): fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.

    Returns:
        npt.NDArray[np.float64]: (n_spins, n_kpoints, n_bands) occupations derivative array.
    """
    smearing = smearing_from_name(smearing_type)(center=fermi_energy, width=smearing_width)
    return 1 / smearing_width**2 * smearing.occupation_2nd_derivative(bands)


def compute_n_electrons(
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> float:
    r"""Compute the number of electrons (total occupation) given a Fermi energy and smearing.

    .. math::
        N_{\mathrm{el.}} = \sum_{\sigma,\mathbf{k},\nu}{\theta_{\sigma,\mathbf{k},\nu} w_{\mathbf{k}}}

    Args:
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.NDArray[np.float64]): (n_kpoints, ) k-point weights array.
        fermi_energy (float): Fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.

    Returns:
        float: number of electrons.
    """
    occupations = compute_occupations(bands, fermi_energy, smearing_type, smearing_width)
    return np.einsum("skn,k->skn", occupations, weights).sum()


def compute_n_electrons_derivative(
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
):
    """Compute the derivative of the number of electrons (total occupation) with respect to the Fermi energy.

    Args:
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.NDArray[np.float64]): (n_kpoints, ) k-point weights array.
        fermi_energy (float): Fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.

    Returns:
        float: d[Nelec]/d[eF].
    """
    occupations_derivative = compute_occupations_derivative(bands, fermi_energy, smearing_type, smearing_width)
    return np.einsum("skn,k->skn", occupations_derivative, weights).sum()


def compute_n_electrons_curvature(
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    fermi_energy: float,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
):
    """Compute the second derivative of the number of electrons (total occupation) with respect to the Fermi energy.

    Args:
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.NDArray[np.float64]): (n_kpoints, ) k-point weights array.
        fermi_energy (float): Fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.

    Returns:
        float: d2[Nelec]/d[eF]2.
    """
    occupations_curvature = compute_occupations_2nd_derivative(bands, fermi_energy, smearing_type, smearing_width)
    return np.einsum("skn,k->skn", occupations_curvature, weights).sum()


def find_fermi_energy(
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    smearing_type: ty.Union[str, int],
    smearing_width: float,
    n_electrons: int,
) -> float:
    """Find the Fermi level by bisection, two-stage algorithm, or at zero temperature depending on the smearing type.

    Args:
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.NDArray[np.float64]): (n_kpoints, ) k-point weights array.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.
        n_electrons (int): target number of electrons.

    Raises:
        ValueError: if the smearing class returned by `smearing_from_name` is unknown.

    Returns:
        float: Fermi energy.
    """
    smearing_cls = smearing_from_name(smearing_type)
    if smearing_cls is Delta:
        fermi_energy = find_fermi_energy_zero_temp(bands, weights, n_electrons)
    elif smearing_cls in (Gaussian, FermiDirac):
        fermi_energy = find_fermi_energy_bisection(bands, weights, smearing_type, smearing_width, n_electrons)
    elif smearing_cls is Cold:
        fermi_energy = find_fermi_energy_two_stage(bands, weights, smearing_type, smearing_width, n_electrons)
    else:
        raise ValueError(f"Unknown smearing class: {smearing_cls}")
    return fermi_energy


def find_fermi_energy_zero_temp(
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    n_electrons: int,
) -> float:
    """Find the Fermi level at zero electronic temperature (no smearing) by fully occupying bands.
    If the number of electrons is obtained by fully occupying every band, the Fermi level is the maximum band energy.
    If unfilled conduction bands exist, the Fermi level is the midpoint between the valence band maximum and conduction
    band minimum.

    Args:
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.NDArray[np.float64]): (n_kpoints, ) k-point weights array.
        n_electrons (int): target number of electrons.

    Raises:
        ValueError: if the number of electrons cannot be obtained without partial occupations.

    Returns:
        float: Fermi energy.
    """
    total_weight = weights.sum()

    if n_electrons % (bands.shape[0] * bands.shape[2] * total_weight) != 0:
        raise ValueError(f"{n_electrons} electrons cannot be obtained with no partial occupations.")

    return NotImplementedError("Zero temperature Fermi level not implemented.")


def find_fermi_energy_bisection(  # pylint: disable=too-many-arguments
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    smearing_type: ty.Union[str, int],
    smearing_width: float,
    n_electrons: int,
    n_electrons_tol: float = 1e-6,
) -> float:
    """Find the Fermi level by bisection.

    Adapated from DFTK.jl/src/occupation.jl.

    Args:
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.NDArray[np.float64]): (n_kpoints, ) k-point weights array.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.
        n_electrons (int): target number of electrons.

    Returns:
        float: Fermi energy.
    """

    def objective(ef):
        ne = compute_n_electrons(bands, weights, ef, smearing_type, smearing_width)
        return ne - n_electrons

    # Get rough bounds for the Fermi level
    e_min = bands.min() - 10 * smearing_width
    e_max = bands.max() + 10 * smearing_width

    fermi_energy = sp.optimize.bisect(objective, e_min, e_max)
    if np.abs(objective(fermi_energy) - n_electrons) > n_electrons_tol:
        raise RuntimeError(f"Failed to find Fermi energy with bisection: {fermi_energy}")

    return fermi_energy


def find_fermi_energy_two_stage(  # pylint: disable=too-many-arguments
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    smearing_type: ty.Union[str, int],
    smearing_width: float,
    n_electrons: int,
    return_all: bool = False,
    n_electrons_tol: float = 1e-6,
    newton_kwargs: ty.Optional[ty.Dict[str, ty.Any]] = None,
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

    Returns:
        float: Fermi energy.
    """
    # Start with bisection and Gaussian smearing
    bisection_fermi = find_fermi_energy_bisection(bands, weights, "gauss", smearing_width, n_electrons, n_electrons_tol)

    # Refine with Newton and the requested smearing (probably cold)
    newton_fermi = bisection_fermi
    newton_refined = True

    # Objective function: f(eF) = (total_occupation(eF) - n_electrons)^2
    def objective(ef):
        ne = compute_n_electrons(bands, weights, ef, smearing_type, smearing_width)
        return (ne - n_electrons) ** 2

    # Derivative of the objective function: f'(eF) = 2 * (f(eF) - n_electrons) * f'(eF)
    def objective_deriv(ef):
        ne = compute_n_electrons(bands, weights, ef, smearing_type, smearing_width)
        dne = compute_n_electrons_derivative(bands, weights, ef, smearing_type, smearing_width)
        return 2 * (ne - n_electrons) * dne

    # Second derivative of the objective function: f''(eF) = 2 * ((f(eF) - n_electrons) * f''(eF) + f'(eF)^2)
    def objective_2nd_deriv(ef):
        ne = compute_n_electrons(bands, weights, ef, smearing_type, smearing_width)
        dne = compute_n_electrons_derivative(bands, weights, ef, smearing_type, smearing_width)
        ddne = compute_n_electrons_curvature(bands, weights, ef, smearing_type, smearing_width)
        return 2 * ((ne - n_electrons) * ddne + dne**2)

    try:
        if newton_kwargs is None:
            newton_kwargs = {}
        newton_fermi = sp.optimize.newton(
            func=objective, fprime=objective_deriv, fprime2=objective_2nd_deriv, x0=bisection_fermi, **newton_kwargs
        )
    except RuntimeError:
        newton_refined = False

    if np.abs(objective(newton_fermi) - n_electrons) > n_electrons_tol:
        newton_refined = False
        newton_fermi = bisection_fermi

    if return_all:
        return newton_fermi, newton_refined, bisection_fermi
    return newton_fermi
