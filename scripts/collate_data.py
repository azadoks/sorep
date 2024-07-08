# %%
import json
import pathlib as pl

import numpy as np

# %%
# TODO:
# - Collate targets into one npz
# - Collate features into one npz per feature type
# - Collate metadata (criteria, fermi energy, etc.) into another npz

def load_data(
    calculation: str = "zero_shot",
    features: list[str] = ["fermi_centered", "vbm_centered", "fermi_scissor"],
    smearing_type: str = "gauss",
    smearing_width: float = 0.05,
):
    data = {
        "sorep_id": [],
        "meets_tcm_criteria": [],
        "band_gap": [],
        "electron_effective_mass": [],
        "hole_effective_mass": [],
        **{feature: [] for feature in features}
    }
    dirs = list(pl.Path("../data/mc3d/").glob("*"))
    for dir_ in dirs:
        if (dir_ / calculation).exists() and (dir_ / "bands").exists():
            with open(dir_ / "bands" / "tcm_criteria.json", "r", encoding="utf-8") as fp:
                criteria = json.load(fp)
            with open(dir_ / calculation / f"dos_{smearing_type}_{smearing_width}.npz", "rb") as fp:
                npz = np.load(fp)
                for feature in features:
                    feature_array = npz[f'dos_{feature}']
                    if feature_array.ndim == 2:
                        feature_array = feature_array.sum(axis=0)  # Sum over spin channels
                    assert feature_array.ndim == 1
                    data[feature].append(feature_array)

            data["meets_tcm_criteria"].append(
                criteria["meets_band_gap"]
                and criteria["meets_electron_effective_mass"]
                and criteria["meets_hole_effective_mass"]
            )
            data["band_gap"].append(criteria["band_gap"])
            data["electron_effective_mass"].append(
                criteria["electron_effective_mass"] if criteria["electron_effective_mass"] is not None else np.nan
            )
            data["hole_effective_mass"].append(
                criteria["hole_effective_mass"] if criteria["hole_effective_mass"] is not None else np.nan
            )
            data["sorep_id"].append(dir_.name)
    return {k: np.array(v) for k, v in data.items()}


# %%
CALCULATION = "zero_shot"
FEATURES = ["fermi_centered", "vbm_centered", "fermi_scissor"]
SMEARING_TYPE = "gauss"
SMEARING_WIDTH = 0.05
data = load_data(CALCULATION, FEATURES, SMEARING_TYPE, SMEARING_WIDTH)
np.savez_compressed(f"../data/mc3d_{CALCULATION}_{SMEARING_TYPE}_{SMEARING_WIDTH}.npz", **load_data())

# %%
