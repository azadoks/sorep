"""Density of states."""

import typing as ty

# import numpy as np
import jax.numpy as np
import numpy.typing as npt

from .smearing import smearing_from_name

__all__ = ("smeared_dos",)


def smeared_dos(
    energies: npt.NDArray[np.float64],
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.NDArray[np.float64]:
    """Compute a smeared density of states.

    Args:
        energies (npt.NDArray[np.float64]): energies at which to sample the DOS
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) array of eigenvalues
        weights (npt.NDArray[np.float64]): (n_kpoints,) array of k-point weights
        smearing_type (ty.Union[str,int]): type of smearing (see `smearing_from_name`)
        smearing_width (float): smearing width

    Returns:
        npt.NDArray: (n_spins, n_energies) array containing the DOS for each spin channel
    """
    smearing_cls = smearing_from_name(smearing_type)
    dos = np.zeros((energies.shape[0], bands.shape[0]))
    for i, energy in enumerate(energies):
        smearing = smearing_cls(center=energy, width=smearing_width)
        occ_deriv = smearing.occupation_derivative(bands)
        dos[i] = np.einsum("skn,k->s", occ_deriv, weights)

    return (dos / smearing_width).T
