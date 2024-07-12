"""Featurization methods."""

import typing as ty

import numpy as np
import numpy.typing as npt

from .band_structure import BandStructure
from .material import MaterialData

__all__ = ("fermi_centered", "vbm_centered", "cbm_centered", "vbm_cbm_concatenated", "vbm_fermi_cbm_concatenated")


def _dos_featurize(  # pylint: disable=too-many-arguments
    bands: BandStructure,
    center: float,
    e_min: float,
    e_max: float,
    n_energies: int,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.ArrayLike:
    """Compute DOS features for a given band structure.

    Energies are sampled linearly between `center + e_min` and `center + e_max`.

    Args:
        bands (BandStructure): Band structure object.
        center (float): Center of the energies.
        e_min (float): Minimum of the energy range around the center.
        e_max (float): Maximum of the energy range around the center.
        n_energies (int): Number of energies.
        smearing_type (ty.Union[str, int]): Smearing type.
        smearing_width (float): Smearing width.

    Returns:
        npt.ArrayLike: DOS features.
    """
    energies = np.linspace(center + e_min, center + e_max, n_energies)
    dos = bands.compute_smeared_dos(energies=energies, smearing_type=smearing_type, smearing_width=smearing_width)
    return dos.sum(axis=0)  # Total dos


def fermi_centered(  # pylint: disable=too-many-arguments
    material: MaterialData,
    e_min: float,
    e_max: float,
    n_energies: int,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.ArrayLike:
    """Compute DOS features centered around the Fermi energy.

    Args:
        material (MaterialData,): Material data object.
        center (float): Center of the energies.
        e_min (float): Minimum of the energy range around the center.
        e_max (float): Maximum of the energy range around the center.
        n_energies (int): Number of energies.
        smearing_type (ty.Union[str, int]): Smearing type.
        smearing_width (float): Smearing width.

    Returns:
        npt.ArrayLike: DOS features.
    """
    return _dos_featurize(
        material.bands, material.bands.fermi_energy, e_min, e_max, n_energies, smearing_type, smearing_width
    )


def vbm_centered(  # pylint: disable=too-many-arguments
    material: MaterialData,
    e_min: float,
    e_max: float,
    n_energies: int,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.ArrayLike:
    """Compute DOS features centered around the valence band maximum (VBM).
    For metals, the Fermi energy is used as the center.

    Args:
        material (MaterialData,): Material data object.
        center (float): Center of the energies.
        e_min (float): Minimum of the energy range around the center.
        e_max (float): Maximum of the energy range around the center.
        n_energies (int): Number of energies.
        smearing_type (ty.Union[str, int]): Smearing type.
        smearing_width (float): Smearing width.

    Returns:
        npt.ArrayLike: DOS features.
    """
    return _dos_featurize(material.bands, material.bands.vbm, e_min, e_max, n_energies, smearing_type, smearing_width)


def cbm_centered(  # pylint: disable=too-many-arguments
    material: MaterialData,
    e_min: float,
    e_max: float,
    n_energies: int,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> npt.ArrayLike:
    """Compute DOS features centered around the conduction band minimum (CBM).
    For metals, the Fermi energy is used as the center.

    Args:
        material (MaterialData,): Material data object.
        center (float): Center of the energies.
        e_min (float): Minimum of the energy range around the center.
        e_max (float): Maximum of the energy range around the center.
        n_energies (int): Number of energies.

    Returns:
        npt.ArrayLike: DOS features.
    """
    return _dos_featurize(material.bands, material.bands.cbm, e_min, e_max, n_energies, smearing_type, smearing_width)


def vbm_cbm_concatenated(
    material: MaterialData,
    vbm_params: dict[str, ty.Union[float, int]],
    cbm_params: dict[str, ty.Union[float, int]],
) -> npt.ArrayLike:
    """Compute the concatenation of DOS features centered around the VBM and CBM.

    Args:
        material (MaterialData,): Material data object.
        vbm_params (dict[str, ty.Union[float, int]]): kwargs passed to `vbm_centered`.
        cbm_params (dict[str, ty.Union[float, int]]): kwargs passed to `cbm_centered`.

    Returns:
        npt.ArrayLike: DOS features.
    """
    bands = material.bands
    vbm = bands.vbm
    cbm = bands.cbm
    vbm_dos = _dos_featurize(bands, vbm, **vbm_params)
    cbm_dos = _dos_featurize(bands, cbm, **cbm_params)
    dos = np.concatenate([vbm_dos, cbm_dos])
    return dos


def vbm_fermi_cbm_concatenated(
    material: MaterialData,
    vbm_params: dict[str, ty.Union[float, int]],
    fermi_params: dict[str, ty.Union[float, int]],
    cbm_params: dict[str, ty.Union[float, int]],
) -> npt.ArrayLike:
    """Compute the concatenation of DOS features centered around the VBM, Fermi energy, and CBM.

    Args:
        material (MaterialData,): Material data object.
        vbm_params (dict[str, ty.Union[float, int]]): kwargs passed to `vbm_centered`.
        fermi_params (dict[str, ty.Union[float, int]]): kwargs passed to `fermi_centered`.
        cbm_params (dict[str, ty.Union[float, int]]): kwargs passed to `cbm_centered`.

    Returns:
        npt.ArrayLike: DOS features.
    """
    bands = material.bands
    fermi_energy = bands.fermi_energy
    vbm = bands.vbm
    cbm = bands.cbm
    vbm_dos = _dos_featurize(bands, vbm, **vbm_params)
    fermi_dos = _dos_featurize(bands, fermi_energy, **fermi_params)
    cbm_dos = _dos_featurize(bands, cbm, **cbm_params)
    dos = np.concatenate([vbm_dos, fermi_dos, cbm_dos])
    return dos
