# %%
import json
import os
import pathlib as pl

import numpy as np


# %%
def collate_targets(base_dir: os.PathLike, path: os.PathLike) -> None:
    """Create a single npz file containing the following arrays:
    - material_id
    - band_gap
    - electron_effective_mass
    - hole_effective_mass
    - meets_band_gap
    - meets_electron_effective_mass
    - meets_hole_effective_mass
    - meets_tcm_criteria

    Args:
        base_dir (os.PathLike): Base directory containing the materials directories.
        path (os.PathLike): Path at which to save the collated npz file.
    """
    targets = {
        "material_id": [],
        "band_gap": [],
        "electron_effective_mass": [],
        "hole_effective_mass": [],
        "meets_band_gap": [],
        "meets_electron_effective_mass": [],
        "meets_hole_effective_mass": [],
        "meets_tcm_criteria": [],
    }
    for dir_ in pl.Path(base_dir).glob("*/bands/"):
        with open(dir_ / "tcm_criteria.json", "r", encoding="utf-8") as fp:
            criteria = json.load(fp)
        targets["material_id"].append(dir_.parent.name)
        targets["band_gap"].append(criteria["band_gap"])
        targets["electron_effective_mass"].append(
            criteria["electron_effective_mass"] if criteria["electron_effective_mass"] is not None else np.nan
        )
        targets["hole_effective_mass"].append(
            criteria["hole_effective_mass"] if criteria["hole_effective_mass"] is not None else np.nan
        )
        targets["meets_band_gap"] = criteria["meets_band_gap"]
        targets["meets_electron_effective_mass"] = criteria["meets_electron_effective_mass"]
        targets["meets_hole_effective_mass"] = criteria["meets_hole_effective_mass"]
        targets["meets_tcm_criteria"].append(
            criteria["meets_band_gap"]
            and criteria["meets_electron_effective_mass"]
            and criteria["meets_hole_effective_mass"]
        )
    np.savez_compressed(path, **targets)


def collate_features(
    base_dir: os.PathLike, calculation_type: str, feature_type: str, feature_id: str, path: os.PathLike, dtype=None
) -> None:
    """Create a single npz file containing the following arrays:
    - material_id
    - features

    Args:
        base_dir (os.PathLike): Base directory containing the materials directories.
        calculation (str): Type of calculation to consider (e.g. single_shot, scf, etc.). Must be an existsing
        subdirectory of each material directory.
        feature_type (str): Type of feature to consider (e.g. fermi_centered, vbm_centered, etc.).
        feature_id (str): ID of the 'dos_{feature_id}.npz' file to load (e.g. gauss_0.05, etc.). Should be within
        the calculation subdirectory of each material directory.
        path (os.PathLike): Path at which to save the collated npz file.
        dtype (npt.DTypeLike): Data type for storing the features.
    """
    features = {"material_id": [], "features": []}
    for dir_ in pl.Path(base_dir).glob(f"*/{calculation_type}/"):
        with open(dir_ / f"dos_{feature_id}.npz", "rb") as fp:
            npz = np.load(fp)
            x = npz[feature_type]
        if x.ndim == 2:  # Sum over spin channels
            x = x.sum(axis=0)
        features["material_id"].append(dir_.parent.name)
        features["features"].append(x)
    if dtype is not None:
        features["features"] = np.array(features["features"], dtype=dtype)
    np.savez_compressed(path, **features)


# %%
BASE_DIR = "../data/mc3d/"
CALCULATION_TYPES = ["single_shot", "scf"]
FEATURE_TYPES = ["dos_fermi_centered", "dos_vbm_centered", "dos_fermi_scissor"]
FEATURE_IDS = ["gauss_0.05"]
DTYPE = np.float32


def main():
    base_dir = pl.Path(BASE_DIR)
    database_name = base_dir.name

    collate_targets(
        base_dir=BASE_DIR,
        path=base_dir.parent / f"{database_name}_targets.npz",
    )

    for calculation_type in CALCULATION_TYPES:
        for feature_id in FEATURE_IDS:
            for feature_type in FEATURE_TYPES:
                collate_features(
                    base_dir=base_dir,
                    calculation_type=calculation_type,
                    feature_type=feature_type,
                    feature_id=feature_id,
                    path=base_dir.parent
                    / f"{database_name}_features_{calculation_type}_{feature_type}_{feature_id}.npz",
                    dtype=DTYPE,
                )


# %%
if __name__ == "__main__":
    main()
