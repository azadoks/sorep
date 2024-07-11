# %%
from multiprocessing import Pool
import os
import pathlib as pl
import typing as ty

import h5py
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

import sorep


# %%
def featurize(
    bands: sorep.BandStructure,
    center: float,
    e_min: float,
    e_max: float,
    n_energies: int,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Compute DOS features for a given band structure.

    Energies are sampled linearly between `center + e_min` and `center + e_max`.

    Args:
        bands (sorep.BandStructure): Band structure object.
        center (float): Center of the energies.
        e_min (float): Minimum of the energy range around the center.
        e_max (float): Maximum of the energy range around the center.
        n_energies (int): Number of energies.
        smearing_type (ty.Union[str, int]): Smearing type.
        smearing_width (float): Smearing width.

    Returns:
        tuple[npt.ArrayLike, npt.ArrayLike]: Energies and DOS.
    """
    energies = np.linspace(center + e_min, center + e_max, n_energies)
    dos = bands.compute_smeared_dos(energies=energies, smearing_type=smearing_type, smearing_width=smearing_width)
    return energies, dos


def fermi_centered(
    material: sorep.MaterialData,
    e_min: float,
    e_max: float,
    n_energies: int,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Compute DOS features centered around the Fermi energy.

    Args:
        material (sorep.MaterialData): Material data object.
        center (float): Center of the energies.
        e_min (float): Minimum of the energy range around the center.
        e_max (float): Maximum of the energy range around the center.
        n_energies (int): Number of energies.
        smearing_type (ty.Union[str, int]): Smearing type.
        smearing_width (float): Smearing width.

    Returns:
        tuple[npt.ArrayLike, npt.ArrayLike]: Energies and DOS.
    """
    return featurize(
        material.bands, material.bands.fermi_energy, e_min, e_max, n_energies, smearing_type, smearing_width
    )


def vbm_centered(
    material: sorep.MaterialData,
    e_min: float,
    e_max: float,
    n_energies: int,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Compute DOS features centered around the valence band maximum (VBM).
    For metals, the Fermi energy is used as the center.

    Args:
        material (sorep.MaterialData): Material data object.
        center (float): Center of the energies.
        e_min (float): Minimum of the energy range around the center.
        e_max (float): Maximum of the energy range around the center.
        n_energies (int): Number of energies.
        smearing_type (ty.Union[str, int]): Smearing type.
        smearing_width (float): Smearing width.

    Returns:
        tuple[npt.ArrayLike, npt.ArrayLike]: Energies and DOS.
    """
    return featurize(material.bands, material.bands.vbm, e_min, e_max, n_energies, smearing_type, smearing_width)


def cbm_centered(
    material: sorep.MaterialData,
    e_min: float,
    e_max: float,
    n_energies: int,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Compute DOS features centered around the conduction band minimum (CBM).
    For metals, the Fermi energy is used as the center.

    Args:
        material (sorep.MaterialData): Material data object.
        center (float): Center of the energies.
        e_min (float): Minimum of the energy range around the center.
        e_max (float): Maximum of the energy range around the center.
        n_energies (int): Number of energies.

    Returns:
        tuple[npt.ArrayLike, npt.ArrayLike]: Energies and DOS.
    """
    return featurize(material.bands, material.bands.cbm, e_min, e_max, n_energies, smearing_type, smearing_width)


def vbm_cbm_concatenated(
    material: sorep.MaterialData,
    vbm_params: dict[str, ty.Union[float, int]],
    cbm_params: dict[str, ty.Union[float, int]],
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Compute the concatenation of DOS features centered around the VBM and CBM.

    Args:
        material (sorep.MaterialData): Material data object.
        vbm_params (dict[str, ty.Union[float, int]]): kwargs passed to `vbm_centered`.
        cbm_params (dict[str, ty.Union[float, int]]): kwargs passed to `cbm_centered`.

    Returns:
        tuple[npt.ArrayLike, npt.ArrayLike]: Energies and DOS.
    """
    bands = material.bands
    vbm = bands.vbm
    cbm = bands.cbm
    vbm_energies, vbm_dos = featurize(bands, vbm, **vbm_params)
    cbm_energies, cbm_dos = featurize(bands, cbm, **cbm_params)
    energies = np.concatenate([vbm_energies, cbm_energies])
    dos = np.concatenate([vbm_dos, cbm_dos], axis=1)
    return energies, dos


def vbm_fermi_cbm_concatenated(
    material: sorep.MaterialData,
    vbm_params: dict[str, ty.Union[float, int]],
    fermi_params: dict[str, ty.Union[float, int]],
    cbm_params: dict[str, ty.Union[float, int]],
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Compute the concatenation of DOS features centered around the VBM, Fermi energy, and CBM.

    Args:
        material (sorep.MaterialData): Material data object.
        vbm_params (dict[str, ty.Union[float, int]]): kwargs passed to `vbm_centered`.
        fermi_params (dict[str, ty.Union[float, int]]): kwargs passed to `fermi_centered`.
        cbm_params (dict[str, ty.Union[float, int]]): kwargs passed to `cbm_centered`.

    Returns:
        tuple[npt.ArrayLike, npt.ArrayLike]: Energies and DOS.
    """
    bands = material.bands
    fermi_energy = bands.fermi_energy
    vbm = bands.vbm
    cbm = bands.cbm
    vbm_energies, vbm_dos = featurize(bands, vbm, **vbm_params)
    fermi_energies, fermi_dos = featurize(bands, fermi_energy, **fermi_params)
    cbm_energies, cbm_dos = featurize(bands, cbm, **cbm_params)
    energies = np.concatenate([vbm_energies, fermi_energies, cbm_energies])
    dos = np.concatenate([vbm_dos, fermi_dos, cbm_dos], axis=1)
    return energies, dos


def get_feature_id(featurization: dict) -> str:
    def _get_feature_id(e_min, e_max, n_energies, smearing_type, smearing_width):
        return f"{e_min:.2f}_{e_max:.2f}_{n_energies}_{smearing_type}_{smearing_width:.2f}"

    name = featurization["function"].__name__
    if name in ("vbm_centered", "fermi_centered", "cbm_centered"):
        return _get_feature_id(**featurization["params"])
    else:
        sub_ids = []
        for key in ("vbm_params", "fermi_params", "cbm_params"):
            params = featurization["params"].get(key)
            if params:
                sub_ids.append(_get_feature_id(**params))
        return "_".join(sub_ids)


def featurize_h5(dir_):
    material = sorep.MaterialData.from_dir(dir_)
    material_id = dir_.parent.name
    calculation_type = dir_.name
    mode = "w" if OVERWRITE_ALL else "a"
    with h5py.File(str(dir_ / "features.h5"), mode) as fp:
        for featurization in FEATURIZATIONS:
            featurization_key = f"{material_id}/{calculation_type}/{featurization['function'].__name__}"
            featurization_group = (
                fp[featurization_key] if featurization_key in fp else fp.create_group(featurization_key)
            )
            feature_id = get_feature_id(featurization)
            if feature_id in featurization_group and not OVERWRITE_NEW:
                continue
            energies, dos = featurization["function"](material, **featurization["params"])
            feature_group = (
                featurization_group[feature_id]
                if feature_id in featurization_group
                else featurization_group.create_group(feature_id)
            )
            if "energies" not in feature_group and "dos" not in feature_group:
                feature_group["energies"] = energies
                feature_group["dos"] = dos
            else:
                feature_group["energies"][...] = energies
                feature_group["dos"][...] = dos


# %%
OVERWRITE_NEW = False
OVERWRITE_ALL = False
SMEARING_TYPE = "gauss"
SMEARING_WIDTH = 0.05  # eV

FEATURIZATIONS = [
    {
        "function": fermi_centered,
        "params": {
            "e_min": -5,
            "e_max": +5,
            "n_energies": 513,
            "smearing_type": SMEARING_TYPE,
            "smearing_width": SMEARING_WIDTH,
        },
    },
    {
        "function": vbm_centered,
        "params": {
            "e_min": -2,
            "e_max": +6,
            "n_energies": 513,
            "smearing_type": SMEARING_TYPE,
            "smearing_width": SMEARING_WIDTH,
        },
    },
    {
        "function": cbm_centered,
        "params": {
            "e_min": -6,
            "e_max": +2,
            "n_energies": 513,
            "smearing_type": SMEARING_TYPE,
            "smearing_width": SMEARING_WIDTH,
        },
    },
    {
        "function": vbm_cbm_concatenated,
        "params": {
            "vbm_params": {
                "e_min": -2,
                "e_max": +3 * SMEARING_WIDTH,
                "n_energies": 257,
                "smearing_type": SMEARING_TYPE,
                "smearing_width": SMEARING_WIDTH,
            },
            "cbm_params": {
                "e_min": -3 * SMEARING_WIDTH,
                "e_max": +2,
                "n_energies": 257,
                "smearing_type": SMEARING_TYPE,
                "smearing_width": SMEARING_WIDTH,
            },
        },
    },
    {
        "function": vbm_cbm_concatenated,
        "params": {
            "vbm_params": {
                "e_min": -2,
                "e_max": +2,
                "n_energies": 257,
                "smearing_type": SMEARING_TYPE,
                "smearing_width": SMEARING_WIDTH,
            },
            "cbm_params": {
                "e_min": -2,
                "e_max": +2,
                "n_energies": 257,
                "smearing_type": SMEARING_TYPE,
                "smearing_width": SMEARING_WIDTH,
            },
        },
    },
    {
        "function": vbm_fermi_cbm_concatenated,
        "params": {
            "vbm_params": {
                "e_min": -1,
                "e_max": +1,
                "n_energies": 171,
                "smearing_type": SMEARING_TYPE,
                "smearing_width": SMEARING_WIDTH,
            },
            "fermi_params": {
                "e_min": -1,
                "e_max": +1,
                "n_energies": 171,
                "smearing_type": SMEARING_TYPE,
                "smearing_width": SMEARING_WIDTH,
            },
            "cbm_params": {
                "e_min": -1,
                "e_max": +1,
                "n_energies": 171,
                "smearing_type": SMEARING_TYPE,
                "smearing_width": SMEARING_WIDTH,
            },
        },
    },
]


def main():
    dirs = list(pl.Path("../data/mc3d/").glob("*/single_shot")) + list(pl.Path("../data/mc3d/").glob("*/scf"))
    pbar = tqdm(dirs, desc="Compute SOREP features", ncols=80)
    with Pool(processes=12, maxtasksperchild=1) as p:
        p.map(featurize_h5, pbar)
    # for dir_ in pbar:
    #     featurize_h5(dir_)


# %%

if __name__ == "__main__":
    main()
