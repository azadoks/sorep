"""
Compute the bands screening quantities for TCMs:
- Band gap
- Electron effective mass
- Hole effective mass
"""

# %%
from multiprocessing import Pool
import pathlib as pl
import warnings

import h5py
import numpy as np
from tqdm import tqdm

import sorep

# %%
SMEARING_TYPE = "fd"
SMEARING_WIDTH = 300 * sorep.constants.KELVIN_TO_EV  # eV
FINITE_DIFFERENCES_ACCURACY = 2
MAT_HDF = pl.Path("../data/mc3d/materials.h5")
TARGET_HDF = pl.Path("../data/mc3d/targets.h5")
EFF_MASS_HDF = pl.Path("../data/mc3d/effective_masses.h5")


def _compute_criteria(id_: str) -> dict:
    with h5py.File(MAT_HDF, "r") as f:
        if id_ in f["test"]:
            material_bands = sorep.MaterialData.from_hdf(f["test"][id_]["bands"])
            material_scf = sorep.MaterialData.from_hdf(f["test"][id_]["scf"])
            material_single_shot = sorep.MaterialData.from_hdf(f["test"][id_]["single_shot"])
        else:
            material_bands = sorep.MaterialData.from_hdf(f["train"][id_]["bands"])
            material_scf = sorep.MaterialData.from_hdf(f["train"][id_]["scf"])
            material_single_shot = sorep.MaterialData.from_hdf(f["train"][id_]["single_shot"])

    scf_is_metallic = material_scf.bands.is_metallic()
    bands_is_metallic = material_bands.bands.is_metallic()
    if scf_is_metallic or bands_is_metallic:
        criteria = {
            "id": id_,
            "scf_band_gap": 0.0,
            "bands_band_gap": 0.0,
            "single_shot_band_gap": 0.0,
            "electron_effective_mass": np.nan,
            "hole_effective_mass": np.nan,
            "meets_band_gap": False,
            "meets_electron_effective_mass": False,
            "meets_hole_effective_mass": False,
            "meets_n_tcm_criteria": False,
            "meets_p_tcm_criteria": False,
            "meets_np_tcm_criteria": False,
            "meets_tcm_criteria": False,
            "cbm": np.nan,
            "vbm": np.nan,
            "effective_masses": [],
        }
    else:
        segments = material_bands.bands.path_segments
        cbm = material_bands.bands.cbm
        vbm = material_bands.bands.vbm
        effective_masses = []
        for i, segment in enumerate(segments):
            integral_effective_masses = segment.compute_integral_line_effective_mass(
                cbm=cbm,
                vbm=vbm,
                smearing_type=SMEARING_TYPE,
                smearing_width=SMEARING_WIDTH,
                acc=FINITE_DIFFERENCES_ACCURACY,
            )
            finite_differences_effective_masses = segment.compute_finite_diff_effective_mass(
                cbm=cbm, vbm=vbm, acc=FINITE_DIFFERENCES_ACCURACY
            )
            finite_differences_effective_masses = {
                key: [val if np.isfinite(val) else None for val in vals]
                for key, vals in finite_differences_effective_masses.items()
            }
            effective_masses.append(
                {
                    "segment": i,
                    "segment_length": len(segment.linear_k),
                    "segment_start_label": str(segment.start_label),
                    "segment_stop_label": str(segment.stop_label),
                    "contains_cbm": bool(np.isclose(segment.eigenvalues, cbm).any()),
                    "contains_vbm": bool(np.isclose(segment.eigenvalues, vbm).any()),
                    "integral_electron": integral_effective_masses["electron"],
                    "integral_hole": integral_effective_masses["hole"],
                    "finite_differences_electron": finite_differences_effective_masses["electron"],
                    "finite_differences_hole": finite_differences_effective_masses["hole"],
                }
            )

        electron_effective_mass = np.nanmax(
            [val["integral_electron"] for val in effective_masses if val["contains_cbm"]], initial=-np.inf
        )
        hole_effective_mass = np.nanmin(
            [val["integral_hole"] for val in effective_masses if val["contains_vbm"]], initial=+np.inf
        )

        criteria = {
            "id": id_,
            "scf_band_gap": material_scf.bands.band_gap,
            "bands_band_gap": material_bands.bands.band_gap,
            "single_shot_band_gap": material_single_shot.bands.band_gap,
            "electron_effective_mass": electron_effective_mass,
            "hole_effective_mass": hole_effective_mass,
            "meets_band_gap": bool(material_scf.bands.band_gap >= 0.5),
            "meets_electron_effective_mass": bool(0.0 < electron_effective_mass <= 0.5),
            "meets_hole_effective_mass": bool(-1.0 <= hole_effective_mass < 0.0),
            "cbm": cbm,
            "vbm": vbm,
            "effective_masses": effective_masses,
        }
        criteria["meets_n_tcm_criteria"] = criteria["meets_band_gap"] and criteria["meets_electron_effective_mass"]
        criteria["meets_p_tcm_criteria"] = criteria["meets_band_gap"] and criteria["meets_hole_effective_mass"]
        criteria["meets_np_tcm_criteria"] = criteria["meets_n_tcm_criteria"] and criteria["meets_p_tcm_criteria"]
        criteria["meets_tcm_criteria"] = criteria["meets_n_tcm_criteria"] or criteria["meets_p_tcm_criteria"]

    return criteria


# %%
def main():
    for train_test in ("test", "train"):
        with h5py.File(MAT_HDF, "r") as f:
            g = f[train_test]
            ids = [id_ for id_ in g.keys() if f"{id_}/bands" in g]

        pbar = tqdm(ids, desc="Compute TCM criteria", ncols=80)
        with Pool(processes=12, maxtasksperchild=1) as p:
            criteria = p.map(_compute_criteria, pbar)

        arrays = {
            "id": [],
            "cbm": [],
            "vbm": [],
            "scf_band_gap": [],
            "bands_band_gap": [],
            "single_shot_band_gap": [],
            "electron_effective_mass": [],
            "hole_effective_mass": [],
            "meets_band_gap": [],
            "meets_electron_effective_mass": [],
            "meets_hole_effective_mass": [],
            "meets_n_tcm_criteria": [],
            "meets_p_tcm_criteria": [],
            "meets_np_tcm_criteria": [],
            "meets_tcm_criteria": [],
        }
        for mat_criteria in criteria:
            for key, value in arrays.items():
                value.append(mat_criteria[key])

        with h5py.File(TARGET_HDF, "a") as f:
            if train_test in f:
                del f[train_test]
            g = f.create_group(train_test)
            g.create_dataset("id", data=arrays["id"])
            g.create_dataset("cbm", data=arrays["cbm"])
            g["cbm"].attrs["units"] = "eV"
            g.create_dataset("vbm", data=arrays["vbm"])
            g["vbm"].attrs["units"] = "eV"
            g.create_dataset("scf_band_gap", data=arrays["scf_band_gap"])
            g["scf_band_gap"].attrs["units"] = "eV"
            g.create_dataset("bands_band_gap", data=arrays["bands_band_gap"])
            g["bands_band_gap"].attrs["units"] = "eV"
            g.create_dataset("single_shot_band_gap", data=arrays["single_shot_band_gap"])
            g["single_shot_band_gap"].attrs["units"] = "eV"
            g.create_dataset("electron_effective_mass", data=arrays["electron_effective_mass"])
            g["electron_effective_mass"].attrs["units"] = "m_e"
            g.create_dataset("hole_effective_mass", data=arrays["hole_effective_mass"])
            g["hole_effective_mass"].attrs["units"] = "m_e"
            g.create_dataset("meets_band_gap", data=arrays["meets_band_gap"])
            g.create_dataset("meets_electron_effective_mass", data=arrays["meets_electron_effective_mass"])
            g.create_dataset("meets_hole_effective_mass", data=arrays["meets_hole_effective_mass"])
            g.create_dataset("meets_n_tcm_criteria", data=arrays["meets_n_tcm_criteria"])
            g.create_dataset("meets_p_tcm_criteria", data=arrays["meets_p_tcm_criteria"])
            g.create_dataset("meets_np_tcm_criteria", data=arrays["meets_np_tcm_criteria"])
            g.create_dataset("meets_tcm_criteria", data=arrays["meets_tcm_criteria"])

        with h5py.File(EFF_MASS_HDF, "a") as f:
            if train_test in f:
                del f[train_test]
            g_train_test = f.create_group(train_test)
            for id_, mat_criteria in zip(ids, criteria):
                g = g_train_test.create_group(id_)
                g.create_dataset("electron_effective_mass", data=mat_criteria["electron_effective_mass"])
                g["electron_effective_mass"].attrs["units"] = "m_e"
                g.create_dataset("hole_effective_mass", data=mat_criteria["hole_effective_mass"])
                g["hole_effective_mass"].attrs["units"] = "m_e"
                for semgment_eff_mass in mat_criteria["effective_masses"]:
                    segment_group = g.create_group(f"segment_{semgment_eff_mass['segment']}")
                    for key, value in semgment_eff_mass.items():
                        segment_group.create_dataset(key, data=value)
                        if "electron" in key or "hole" in key:
                            segment_group[key].attrs["units"] = "m_e"


# %%
if __name__ == "__main__":
    main()
