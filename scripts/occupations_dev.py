# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import sorep

# %%
results = []
i = 0
with h5py.File("../data/mc3d/materials_train_test.h5", "a") as f:
    for train_test, train_test_group in f.items():
        n = len(train_test_group)
        for mc3d_id, material in train_test_group.items():
            for calculation_type, calculation in material.items():
                bands = sorep.BandStructure.from_hdf(calculation["bands"])
                smearing = calculation["bands"].attrs["smearing"]
                degauss = calculation["bands"].attrs["degauss"]
                qe_ef = calculation["bands"].attrs["qe_fermi_energy"]

                ef = bands.find_fermi_energy(
                    smearing,
                    degauss,
                    n_electrons_tol=1e-6,
                    n_electrons_kwargs={"max_exponent": np.inf},
                    dn_electrons_kwargs={"max_exponent": np.inf},
                    ddn_electrons_kwargs={"max_exponent": np.inf},
                    newton_kwargs={"maxiter": 500},
                )
                flag = ef["flag"]
                ef = ef["fermi_energy"]

                bands.fermi_energy = ef
                if bands.is_insulating():
                    ef_midgap = bands.vbm + (bands.cbm - bands.vbm) / 2
                    ne_midgap = bands.compute_n_electrons("delta", 0.0, ef_midgap)
                    if np.abs(ne_midgap - bands.n_electrons) < 1e-6:
                        ef = ef_midgap
                        is_ins = True
                        smearing = "delta"
                        degauss = 0.0
                else:
                    is_ins = False

                ne = sorep.occupation.compute_n_electrons(
                    bands.eigenvalues,
                    bands.weights,
                    ef,
                    smearing,
                    degauss,
                    max_exponent=np.inf,
                )

                occupations = sorep.occupation.compute_occupations(bands.eigenvalues, ef, smearing, degauss)

                calculation["bands"]["fermi_energy"][()] = ef
                calculation["bands"]["occupations"][:, :, :] = occupations

                print(
                    f"{i},{n},{mc3d_id},{calculation_type},{int(bands.n_electrons)},{ne},{np.abs(ne - bands.n_electrons)},{qe_ef},{ef},{np.abs(qe_ef - ef)},{is_ins},{flag}",
                    flush=True,
                )
                results.append(
                    {
                        "mc3d_id": mc3d_id,
                        "calculation_type": calculation_type,
                        "n_electrons": bands.n_electrons,
                        "n_electrons_sorep": ne,
                        "fermi_energy_qe": qe_ef,
                        "fermi_energy_sorep": ef,
                        "flag": flag,
                    }
                )
            i += 1


# # %%
results_df = pd.DataFrame(results)
results_df.to_json("../data/mc3d/fermi_results.json")
##############################################################################
# %%
results_df = pd.read_csv(
    "./occs_log.tsv",
    names=[
        "i",
        "n",
        "mc3d_id",
        "calculation_type",
        "n_electrons",
        "n_electrons_sorep",
        "n_electrons_diff",
        "fermi_energy_qe",
        "fermi_energy_sorep",
        "fermi_energy_diff",
        "is_ins",
        "flag",
    ],
)
results_df
results_df[results_df.n_electrons_diff > 1e-6]

# %%
# def plot_n_elec(train_test, mc3d_id, calculation_type, lim=5):
#     with h5py.File("../data/mc3d/materials_train_test.h5", "r") as f:
#         bands = sorep.BandStructure.from_hdf(f[train_test][mc3d_id][calculation_type]["bands"])
#         smearing_type = f[train_test][mc3d_id][calculation_type]["bands"].attrs["smearing"]
#         smearing_width = f[train_test][mc3d_id][calculation_type]["bands"].attrs["degauss"]

#     x = np.linspace(bands.fermi_energy - lim, bands.fermi_energy + lim, 500)
#     y = [bands.compute_n_electrons(smearing_type, smearing_width, ei) for ei in x]

#     ef = bands.find_fermi_energy(
#         smearing_type,
#         smearing_width,
#         n_electrons_tol=1e-8,
#         n_electrons_kwargs={"max_exponent": np.inf},
#         dn_electrons_kwargs={"max_exponent": np.inf},
#         ddn_electrons_kwargs={"max_exponent": np.inf},
#         newton_kwargs={"maxiter": 50000},
#     )
#     flag = ef["flag"]
#     ef = ef["fermi_energy"]

#     ef_gauss = sorep.fermi.find_fermi_energy_bisection(
#         bands.eigenvalues,
#         bands.weights,
#         "gauss",
#         smearing_width,
#         bands.n_electrons,
#         n_electrons_tol=1e-8,
#     )
#     ne_gauss = bands.compute_n_electrons("gauss", smearing_width, ef_gauss["fermi_energy"])

#     ne = bands.compute_n_electrons(smearing_type, smearing_width, ef)
#     dne = bands.compute_n_electrons_derivative(smearing_type, smearing_width, ef)

#     bands.fermi_energy = ef
#     is_met = bands.is_metallic()

#     fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True, width_ratios=[1, 4])
#     axes[0].plot(y, x, color="k")

#     axes[0].scatter(ne, ef, color="tab:red", marker="x", label=smearing_type)
#     axes[0].scatter(ne_gauss, ef_gauss["fermi_energy"], color="tab:green", marker="^", label="gauss")
#     if not is_met:
#         ef_mid = bands.vbm + (bands.cbm - bands.vbm) / 2
#         ne_mid = bands.compute_n_electrons("delta", 0.0, ef_mid)
#         axes[0].scatter(ne_mid, ef_mid, color="tab:orange", marker="v", label="midgap")

#     axes[0].axvline(bands.n_electrons, linestyle="--", color="tab:blue")
#     axes[0].legend()
#     axes[0].set_ylim(ef - lim, ef + lim)
#     axes[0].set_title(f"({'metal' if is_met else 'insulator'})")
#     axes[0].set_xlim(bands.n_electrons - 2, bands.n_electrons + 2)

#     axes[1].plot(bands.eigenvalues[0], c="k")
#     if bands.n_spins == 2:
#         axes[1].plot(bands.eigenvalues[1], c="tab:blue")
#     axes[1].axhline(ef, linestyle="--", color="tab:red")
#     axes[1].set_ylim(ef - lim, ef + lim)
#     axes[1].set_title(flag)

#     return fig, ax


# bad_results = results_df[results_df.n_electrons_diff > 1e-6]
# row = bad_results.iloc[0]
# train_test = "test" if row.n == 2170 else "test"
# fig, ax = plot_n_elec(train_test, row.mc3d_id, row.calculation_type, lim=1)

# %%
