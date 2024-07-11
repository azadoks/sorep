# %%
import json
import os
import pathlib as pl

from ase.io import read
from ase.io.extxyz import write_extxyz
import h5py
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
    - meets_ntcm_criteria
    - meets_ptcm_criteria
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
        "meets_ntcm_criteria": [],
        "meets_ptcm_criteria": [],
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
        targets["meets_ntcm_criteria"].append(criteria["meets_band_gap"] and criteria["meets_electron_effective_mass"])
        targets["meets_ptcm_criteria"].append(criteria["meets_band_gap"] and criteria["meets_hole_effective_mass"])
        targets["meets_tcm_criteria"].append(
            criteria["meets_band_gap"]
            and (criteria["meets_electron_effective_mass"] or criteria["meets_hole_effective_mass"])
        )
    np.savez_compressed(path, **targets)


def collate_structures(base_dir: os.PathLike, calculation_type: str, path: os.PathLike) -> None:
    """Create a single xyz file containing the structures of all materials.

    Args:
        base_dir (os.PathLike): Base directory containing the materials directories.
        calculation_type (str): Type of calculation to consider (e.g. single_shot, scf, etc.). Must be an existsing
        subdirectory of each material directory.
        path (os.PathLike): Path at which to save the collated structures.
    """
    images = []
    for dir_ in pl.Path(base_dir).glob(f"*/{calculation_type}/"):
        atoms = read(dir_ / "structure.xyz")
        atoms.info["material_id"] = dir_.parent.name
        images.append(atoms)
    write_extxyz(path, images)


def collate_features(
    base_dir: os.PathLike,
    calculation_type: str,
    path: os.PathLike,
    dtype=None,
) -> None:
    with h5py.File(path, "w") as collated_file:
        for dir_ in pl.Path(base_dir).glob(f"*/{calculation_type}/"):
            if not (dir_ / "features.h5").exists():
                continue
            with h5py.File(dir_ / "features.h5", "r") as features_file:
                for material_id, material_group in features_file.items():
                    for calculation_type, calculation_group in material_group.items():
                        for feature_type, feature_group in calculation_group.items():
                            for feature_id, feature_id_group in feature_group.items():
                                collated_file.create_group(
                                    f"{material_id}/{calculation_type}/{feature_type}/{feature_id}"
                                )
                                for key, dataset in feature_id_group.items():
                                    collated_file[f"{material_id}/{calculation_type}/{feature_type}/{feature_id}"][
                                        key
                                    ] = dataset[()]


# %%
BASE_DIR = "../data/mc3d/"


def main():
    base_dir = pl.Path(BASE_DIR)
    database_name = base_dir.name

    collate_targets(
        base_dir=BASE_DIR,
        path=base_dir.parent / f"{database_name}_targets.npz",
    )

    for calculation_type in ["scf", "single_shot", "bands"]:
        collate_structures(
            base_dir=BASE_DIR,
            calculation_type=calculation_type,
            path=base_dir.parent / f"{database_name}_structures_{calculation_type}.xyz",
        )

    for calculation_type in ["scf", "single_shot"]:
        collate_features(
            base_dir=BASE_DIR,
            calculation_type=calculation_type,
            path=base_dir.parent / f"{database_name}_features_{calculation_type}.h5",
        )


# %%
if __name__ == "__main__":
    main()
