# %%
import json
import pathlib as pl

import numpy as np


# %%
def load_data(features: str = "dos_fermi_centered", smearing_type: str = "gauss", smearing_width: float = 0.05):
    data = {
        "sorep_id": [],
        "features": [],
        "meets_tcm_criteria": [],
        "band_gap": [],
        "electron_effective_mass": [],
        "hole_effective_mass": [],
    }
    dirs = list(pl.Path("../data/mc3d/").glob("*"))
    for dir_ in dirs:
        if (dir_ / "zero_shot").exists() and (dir_ / "bands").exists():
            with open(dir_ / "bands" / "tcm_criteria.json", "r", encoding="utf-8") as fp:
                criteria = json.load(fp)
            with open(dir_ / "zero_shot" / f"dos_{smearing_type}_{smearing_width}.npz", "rb") as fp:
                data["features"].append(np.load(fp)[features].sum(axis=0))  # Sum over spin channels
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
for features in ("dos_fermi_centered", "dos_vbm_centered", "dos_fermi_scissor"):
    smearing_type = "gauss"
    smearing_width = 0.05
    np.savez_compressed(f"../data/mc3d_{features}_{smearing_type}_{smearing_width}.npz", **load_data())

# %%
