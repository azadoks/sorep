"""
Compute the bands screening quantities for TCMs:
- Band gap
- Electron effective mass
- Hole effective mass
"""

import json

# %%
from multiprocessing import Pool
import pathlib as pl

import numpy as np
from tqdm import tqdm

import sorep

# %%
SMEARING_TYPE = "fd"
SMEARING_WIDTH = 300 * sorep.constants.KELVIN_TO_EV  # eV
FINITE_DIFFERENCES_ACCURACY = 2


def _compute_criteria(dir_):
    material = sorep.MaterialData.from_dir(dir_)
    if material.bands.is_metallic():
        criteria = {
            "band_gap": 0.0,
            "electron_effective_mass": None,
            "hole_effective_mass": None,
            "meets_band_gap": False,
            "meets_electron_effective_mass": False,
            "meets_hole_effective_mass": False,
            "cbm": None,
            "vbm": None,
            "effective_masses": [],
        }
    else:
        band_gap = material.bands.band_gap
        segments = material.bands.path_segments
        cbm = material.bands.cbm
        vbm = material.bands.vbm
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
                    "segment_start_label": segment.start_label,
                    "segment_stop_label": segment.stop_label,
                    "contains_cbm": bool(np.isclose(segment.eigenvalues, cbm).any()),
                    "contains_vbm": bool(np.isclose(segment.eigenvalues, vbm).any()),
                    "integral_electron": integral_effective_masses["electron"],
                    "integral_hole": integral_effective_masses["hole"],
                    "finite_differences_electron": finite_differences_effective_masses["electron"],
                    "finite_differences_hole": finite_differences_effective_masses["hole"],
                }
            )

        electron_effective_mass = np.nanmax(
            [val["integral_electron"] for val in effective_masses if val["contains_cbm"]]
        )
        hole_effective_mass = np.nanmin([val["integral_hole"] for val in effective_masses if val["contains_vbm"]])

        criteria = {
            "band_gap": band_gap,
            "electron_effective_mass": electron_effective_mass,
            "hole_effective_mass": hole_effective_mass,
            "meets_band_gap": bool(band_gap >= 0.5),
            "meets_electron_effective_mass": bool(0.0 <= electron_effective_mass <= 0.5),
            "meets_hole_effective_mass": bool(-1.0 <= hole_effective_mass <= 0.0),
            "cbm": cbm,
            "vbm": vbm,
            "effective_masses": effective_masses,
        }

    with open(dir_ / "tcm_criteria.json", "w", encoding="utf-8") as fp:
        json.dump(criteria, fp)

    return criteria


# %%
def main():
    dirs = list(pl.Path("../data/mc3d/").glob("*/bands/"))
    pbar = tqdm(dirs, desc="Compute TCM criteria", ncols=80)
    with Pool(processes=12, maxtasksperchild=1) as p:
        p.map(_compute_criteria, pbar)


# %%
if __name__ == "__main__":
    main()
