# %%
import json
import pathlib as pl

from matplotlib import colormaps
import matplotlib.pyplot as plt
import pandas as pd

# e = hc/λ
# λ = hc/e
PLANCK_CONSTANT_EVS = 6.582119569e-16  # eV s
PLANCK_CONSTANT_EV_HZ = 4.135667696e-15  # eV Hz^-1
SPEED_OF_LIGHT_M_S = 299792458  # m / s


def ev_to_nm(energy):
    return (PLANCK_CONSTANT_EVS * SPEED_OF_LIGHT_M_S / energy) * 1e9


def nm_to_ev(wavelength):
    return PLANCK_CONSTANT_EVS * SPEED_OF_LIGHT_M_S / (wavelength * 1e-9)


# %%
criteria = []
eff_masses = []
for dir_ in pl.Path("../data/mc3d/").glob("*/bands/"):
    with open(dir_ / "tcm_criteria.json", "r", encoding="utf-8") as fp:
        tmp = json.load(fp)
    eff_masses.append(tmp.pop("effective_masses"))
    criteria.append({"sorep_id": dir_.parent.name, **tmp})

criteria = pd.DataFrame(criteria).dropna(axis="index")
eff_masses = pd.DataFrame(eff_masses)

tcms = criteria[criteria.meets_band_gap & criteria.meets_electron_effective_mass & criteria.meets_hole_effective_mass]
# %%
tcms
# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

axes[0].scatter(tcms.band_gap, -tcms.hole_effective_mass, marker=".")
axes[0].axhline(1.0, color="black", linestyle="--")
axes[0].set_xlabel("Band gap (eV)")
axes[0].set_ylabel("Hole effective mass")

axes[1].scatter(tcms.band_gap, tcms.electron_effective_mass, marker=".")
axes[1].axhline(0.5, color="black", linestyle="--")
axes[1].set_xlabel("Band gap (eV)")
axes[1].set_ylabel("Electron effective mass")


for ax in axes:
    ax.axvline(0.5, color="black", linestyle="--")

# %%
