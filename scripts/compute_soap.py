# %%
from collections.abc import MutableMapping
import os
import pathlib as pl
import typing as ty

from ase import Atoms
from ase.io import read
from dscribe.descriptors import SOAP
import h5py
import numpy as np
from tqdm import tqdm

MAT_HDF = pl.Path("../data/mc3d/materials.h5")
FEAT_HDF = pl.Path("../data/mc3d/features.h5")


# %%
def _flatten(dictionary, parent_key="", separator="__"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(_flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _equal_params_existing(group, params):
    for subgroup_key in group:
        match = True
        subgroup_params = dict(group[subgroup_key].attrs)
        for param_key, param_value in _flatten(params).items():
            if param_key == "species":
                # Check if the sets of species are different
                if len(set(subgroup_params.get(param_key)) ^ set(param_value)) != 0:
                    match = False
            elif subgroup_params.get(param_key) != param_value:
                match = False
        if match:
            return subgroup_key
    return None


# def compute_soap_all(calculation_type: str, params: dict):
#     images = read(f"../data/mc3d_structures_{calculation_type}.xyz", ":")
#     material_ids = [image.info["material_id"] for image in images]
#     for image in images:
#         image.set_atomic_numbers(np.ones(image.get_global_number_of_atoms(), dtype=int))
#     soap_dscribe = SOAP(**params)
#     features = soap_dscribe.create(images, n_jobs=12)
#     return material_ids, features


def load_images(hdf, train_test: str, calculation_type: str):
    images = []
    for material in hdf[train_test].values():
        if calculation_type in material:
            image = Atoms(**{k: v[()] for k, v in material[calculation_type]["atoms"].items()})
            image.info = dict(material[calculation_type]["atoms"].attrs)
            images.append(image)
    return images


def main():
    mode = "w" if OVERWRITE_ALL else "a"
    with h5py.File(FEAT_HDF, mode) as f_features:
        for train_test in ("test", "train"):
            g_train_test = f_features[train_test] if train_test in f_features else f_features.create_group(train_test)
            for calculation_type in ["single_shot", "scf"]:
                type_key = f"{calculation_type}/soap"
                if type_key in g_train_test.keys():
                    g_type = g_train_test[type_key]
                else:
                    g_type = g_train_test.create_group(type_key)
                for params in SOAP_PARAMS:
                    params = params.copy()
                    # Read in the images for each featurization; they may be modified below to remove species information
                    with h5py.File(MAT_HDF, "r") as f_materials:
                        images = load_images(f_materials, train_test, calculation_type)
                    # Construct the species list
                    if not params["species"]:
                        for image in images:
                            image.set_atomic_numbers(np.ones(image.get_global_number_of_atoms(), dtype=int))
                        params["species"] = [1]
                    else:
                        params["species"] = np.unique(np.concatenate([image.numbers for image in images]))
                    if existing_key := _equal_params_existing(g_type, params):
                        if OVERWRITE_NEW:
                            del g_type[existing_key]
                        else:
                            continue
                    # Get the group for this feature instance
                    instance_key = str(max([int(key) for key in g_type.keys()], default=-1) + 1)
                    g_instance = g_type.create_group(instance_key)
                    # Set the instance group attributes
                    for k, v in _flatten(params).items():
                        g_instance.attrs[k] = v
                    # Construct the featurizer
                    soap_dscribe = SOAP(**params)
                    # Initialize the target arrays in the HDF5 file
                    n_features = soap_dscribe.get_number_of_features()
                    material_ids = g_instance.create_dataset(
                        "id", (len(images),), dtype=h5py.string_dtype(encoding="utf-8", length=None)
                    )
                    chunk_rows = max(512_000 // np.zeros(n_features).nbytes, 1)
                    chunk_cols = n_features
                    features = g_instance.create_dataset(
                        name="features",
                        shape=(len(images), n_features),
                        dtype=FEATURES_DTYPE,
                        compression="gzip",
                        chunks=(chunk_rows, chunk_cols),
                        shuffle=True,
                    )
                    # Compute the SOAP features and store them in the HDF5 file
                    feature_chunk = chunk_rows * N_JOBS
                    for i in tqdm(range(0, len(images), feature_chunk), desc="Featurizing chunks: "):
                        image_chunk = images[i : i + feature_chunk]
                        material_ids[i : i + feature_chunk] = [str(image.info["id"]) for image in image_chunk]
                        if params.get("sparse", False):
                            features[i : i + feature_chunk] = soap_dscribe.create(image_chunk, n_jobs=N_JOBS).todense()
                        else:
                            features[i : i + feature_chunk] = soap_dscribe.create(image_chunk, n_jobs=N_JOBS)
                    f_features.flush()


# %%
N_JOBS = 12
FEATURES_DTYPE = "float32"
OVERWRITE_ALL = False
OVERWRITE_NEW = False
SOAP_PARAMS = [
    {
        "r_cut": 6.0,
        "n_max": 10,
        "l_max": 9,
        "sigma": 0.3,
        "rbf": "gto",
        "weighting": {
            "function": "poly",
            "r0": 5.0,
            "c": 1.0,
            "m": 1.0,
        },
        "average": "inner",
        "periodic": True,
        "sparse": False,
        "dtype": "float32",
        "species": False,
    },
    # {
    #     "r_cut": 6.0,
    #     "n_max": 6,
    #     "l_max": 4,
    #     "sigma": 0.3,
    #     "rbf": "gto",
    #     "weighting": {
    #         "function": "poly",
    #         "r0": 5.0,
    #         "c": 1.0,
    #         "m": 1.0,
    #     },
    #     "average": "inner",
    #     "periodic": True,
    #     "sparse": True,
    #     "dtype": "float32",
    #     "species": True,
    # },
]
# %%
if __name__ == "__main__":
    main()
# %%
# %%
