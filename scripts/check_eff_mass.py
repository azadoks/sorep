# %%
import json
import pathlib as pl

import numpy as np
import pandas as pd
import plot_segments

import sorep

# %%
all_criteria = []
for tcm_criteria_file in pl.Path("../data/mc3d/").rglob("tcm_criteria.json"):
    with open(tcm_criteria_file, "r", encoding="utf-8") as fp:
        criteria = json.load(fp)
    with open(tcm_criteria_file.parents[1] / "scf" / "metadata.json", "r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    criteria["material_id"] = tcm_criteria_file.parents[1].name
    criteria["matproj_duplicate"] = metadata["matproj_duplicate"]
    criteria["formula_hill"] = metadata["formula_hill"]
    all_criteria.append(criteria)

# %%
insulators = [criteria for criteria in all_criteria if criteria["band_gap"] > 0.0]
insulators_df = pd.DataFrame(insulators)
insulators_df["meets"] = insulators_df["meets_band_gap"] & (
    insulators_df["meets_electron_effective_mass"] | insulators_df["meets_hole_effective_mass"]
)
# %%
insulators_df["abs_electron_effective_mass"] = insulators_df["electron_effective_mass"].abs()
ins_sorted = insulators_df.sort_values(by="abs_electron_effective_mass", ascending=True)[
    [
        "electron_effective_mass",
        "hole_effective_mass",
        "band_gap",
        "meets",
        "material_id",
        "matproj_duplicate",
        "formula_hill",
    ]
].dropna(axis="index")
# %%
i = 0
material_id = ins_sorted.iloc[i]["material_id"]

material = sorep.MaterialData.from_dir(pl.Path(f"../data/mc3d/{material_id}/bands/"))
fig, ax = plot_segments.plot_segments(material)
ax.set_ylim(material.bands.cbm - 1, material.bands.cbm + 1)
ax.set_title(
    r"$m^{*}_{e}=$"
    + f"{ins_sorted.iloc[i]['electron_effective_mass']:.4f}"
    + " | "
    + r"$m^{*}_{h}$="
    + f"{ins_sorted.iloc[i]['hole_effective_mass']:.4f}"
    + f" | {ins_sorted.iloc[i]['matproj_duplicate']}",
    fontsize=9,
)
# %%
mpids = [
    "mp-10064",
    "mp-10486",
    "mp-10913",
    "mp-1243",
    "mp-13003",
    "mp-13134",
    "mp-13803",
    "mp-13820",
    "mp-14113",
    "mp-16281",
    "mp-16293",
    "mp-1705",
    "mp-19006",
    "mp-19079",
    "mp-19148",
    "mp-19321",
    "mp-19784",
    "mp-20011",
    "mp-20546",
    "mp-22189",
    "mp-2229",
    "mp-22323",
    "mp-22606",
    "mp-22937",
    "mp-22979",
    "mp-23080",
    "mp-23358",
    "mp-25014",
    "mp-25043",
    "mp-25178",
    "mp-27175",
    "mp-27843",
    "mp-28931",
    "mp-29213",
    "mp-29297",
    "mp-29455",
    "mp-2951",
    "mp-29910",
    "mp-30284",
    "mp-3056",
    "mp-3188",
    "mp-3653",
    "mp-3810",
    "mp-3917",
    "mp-4590",
    "mp-505501",
    "mp-510309",
    "mp-5280",
    "mp-540688",
    "mp-540728",
    "mp-550008",
    "mp-560842",
    "mp-566788",
    "mp-572688",
    "mp-5909",
    "mp-697",
    "mp-7534",
    "mp-7762",
    "mp-8086",
    "mp-8275",
    "mp-8285",
    "mp-13334",
    "mp-14243",
    "mp-19803",
    "mp-2133",
    "mp-22598",
    "mp-3163",
    "mp-3443",
    "mp-504908",
    "mp-5794",
    "mp-5966",
    "mp-7831",
    "mp-856",
    "mp-886",
    "mp-8922",
]

# %%
