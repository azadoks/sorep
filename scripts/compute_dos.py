# %%
from collections.abc import MutableMapping
from functools import partial
from multiprocessing import Pool
import os
import pathlib as pl

import h5py
import numpy as np
from tqdm import tqdm

import sorep
from sorep.features import cbm_centered, fermi_centered, vbm_cbm_concatenated, vbm_centered, vbm_fermi_cbm_concatenated

# %%
MAT_HDF = pl.Path("../data/mc3d/materials.h5")
FEAT_HDF = pl.Path("../data/mc3d/features.h5")
OVERWRITE_NEW = False
OVERWRITE_ALL = True
SMEARING_TYPE = "gauss"
SMEARING_WIDTH = 0.05  # eV
FEATURE_DTYPE = "float32"

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


def _flatten(dictionary, parent_key="", separator="__"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(_flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def load_materials(hdf, ids, calc_type):
    for id_ in ids:
        yield sorep.MaterialData.from_hdf(hdf[id_][calc_type])


def featurize(hdf, calc_type: str, featurization: dict) -> dict:
    """Apply the featurization described in `featurization` to all materials in `dirs`.

    Args:
        dirs (list[os.PathLike]): Directories containing material files.
        featurization (dict): Featurization method and parameters.

    Returns:
        dict: Dictionary containing material ids and features.
    """
    result = {}
    ids = [id_ for id_ in hdf.keys() if f"{id_}/{calc_type}" in hdf]
    result["id"] = ids
    # This is a generator so we don't need to load all the materials at once
    materials = load_materials(hdf, ids, calc_type)
    # pmap requires that the function is pickleable and has one argument; partial gets us both
    _featurize = partial(featurization["function"], **featurization["params"])
    pbar = tqdm(materials, desc="Compute SOREP features", ncols=80, total=len(ids))
    with Pool(processes=12, maxtasksperchild=1) as p:
        result["features"] = np.array(p.map(_featurize, pbar))
    return result


def _equal_params_existing(group, params):
    for subgroup_key in group:
        subgroup_params = dict(group[subgroup_key].attrs)
        if all(subgroup_params.get(param_key) == param_value for (param_key, param_value) in _flatten(params).items()):
            return subgroup_key
    return None


def main():
    mode = "w" if OVERWRITE_ALL else "a"
    with h5py.File(FEAT_HDF, mode) as f:
        for calculation_type in ["single_shot", "scf"]:
            print(f"Processing {calculation_type} calculations")
            for featurization in FEATURIZATIONS:
                feature_method = featurization["function"].__name__
                attrs = _flatten(featurization["params"])
                key = f"{calculation_type}/{feature_method}"
                method_group = f[key] if key in f else f.create_group(key)
                if existing_key := _equal_params_existing(method_group, attrs):
                    if OVERWRITE_NEW:
                        del method_group[existing_key]
                    else:
                        print(f"Skipping existing instance of {key}")
                        continue

                print(f"Featurizing with {feature_method}")
                with h5py.File(MAT_HDF, "r") as materials_file:
                    features = featurize(materials_file, calculation_type, featurization)

                instance_key = str(max([int(key) for key in method_group.keys()], default=-1) + 1)
                instance_group = method_group.create_group(instance_key)
                for key, val in attrs.items():
                    instance_group.attrs[key] = val

                # Chunking is important for efficient access
                # We aim for chunks that are roughly 512 KB in size, completely capturing at least one feature
                # vector.
                chunk_rows = max(512_000 // features["features"][0].nbytes, 1)
                chunk_cols = features["features"].shape[1]
                instance_group.create_dataset(
                    "features",
                    data=features["features"],
                    compression="gzip",
                    chunks=(chunk_rows, chunk_cols),
                    shuffle=True,
                    dtype=FEATURE_DTYPE,
                )
                instance_group.create_dataset("id", data=features["id"], compression="gzip")


# %%

if __name__ == "__main__":
    main()
