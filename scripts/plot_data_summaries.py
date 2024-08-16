# %%
from collections import Counter
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatviz as pmv
import seaborn as sns


# %%
def iter_mats():
    with h5py.File("../data/mc3d/materials.h5", "r") as f:
        for train_test, train_test_group in f.items():
            for material_id, material_group in train_test_group.items():
                yield {
                    "train_test": train_test,
                    "material_id": material_id,
                    "group": material_group,
                }


def iter_calcs():
    with h5py.File("../data/mc3d/materials.h5", "r") as f:
        for train_test, train_test_group in f.items():
            for material_id, material_group in train_test_group.items():
                for calc_type, calc_type_group in material_group.items():
                    yield {
                        "train_test": train_test,
                        "material_id": material_id,
                        "calc_type": calc_type,
                        "group": calc_type_group,
                    }


with h5py.File("../data/mc3d/targets.h5", "r") as f:
    target_df = pd.concat(
        [pd.DataFrame({k: v[()] for k, v in f[train_test].items()}) for train_test in ("test", "train")]
    )
    target_df["id"] = target_df["id"].apply(lambda x: x.decode("utf-8"))
# %%  Number of atoms
number_of_atoms = [mat["group"]["scf/atoms/numbers"].shape[0] for mat in iter_mats()]
with plt.style.context("../sorep.mplstyle"):
    fig, ax = plt.subplots(dpi=150)
    ax.hist(number_of_atoms, bins=np.arange(1, 52) - 0.5)
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Number of materials")
fig.savefig("../plots/mc3d_n_atoms.pdf", bbox_inches="tight")

# %%  Atomic species
atomic_species = sum([np.unique(mat["group"]["scf/atoms/numbers"][()]).tolist() for mat in iter_mats()], start=[])
atomic_species_counts = Counter(atomic_species)
ax = pmv.ptable_heatmap(atomic_species_counts, cbar_title="Number of materials containing element")
fig = ax.get_figure()
fig.savefig("../plots/mc3d_elements.pdf", bbox_inches="tight")
# %% Band Gaps
band_gaps = target_df[target_df.scf_band_gap > 0].scf_band_gap
with plt.style.context("../sorep.mplstyle"):
    fig, ax = plt.subplots(dpi=150)
    ax.hist(band_gaps, bins=int(np.sqrt(band_gaps.shape[0])))
    ax.set_xlabel("Band Gap (eV)")
    ax.set_ylabel("Number of materials")
fig.savefig("../plots/mc3d_band_gaps.pdf", bbox_inches="tight")
# %% Effective masses
hole_effective_masses = target_df[
    (target_df.scf_band_gap > 0) & (target_df.hole_effective_mass < 0) & (target_df.hole_effective_mass > -50)
].hole_effective_mass
n_left = target_df[(target_df.scf_band_gap > 0) & (target_df.hole_effective_mass <= -50)].shape[0]
electron_effective_masses = target_df[
    (target_df.scf_band_gap > 0) & (target_df.electron_effective_mass > 0) & (target_df.electron_effective_mass < 50)
].electron_effective_mass
n_right = target_df[(target_df.scf_band_gap > 0) & (target_df.electron_effective_mass >= 50)].shape[0]
with plt.style.context("../sorep.mplstyle"):
    fig, ax = plt.subplots(dpi=150)
    ax.hist(
        hole_effective_masses,
        bins=int(np.sqrt(hole_effective_masses.shape[0])),
        label="hole",
        color="tab:red",
    )
    ax.hist(
        electron_effective_masses,
        bins=int(np.sqrt(electron_effective_masses.shape[0])),
        label="electron",
        color="tab:blue",
    )
    ax.text(-50, 500, f"$\leftarrow$ {n_left}", fontdict={"va": "center", "ha": "left"})
    ax.text(50, 500, f"$\\rightarrow$ {n_right}", fontdict={"va": "center", "ha": "right"})

    ax.legend()
    ax.set_xlabel("Line effective mass (m$_e$)")
    ax.set_ylabel("Number of materials")
fig.savefig("../plots/mc3d_effective_masses.pdf", bbox_inches="tight")
# %%
