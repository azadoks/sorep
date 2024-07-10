# %%
import os
import pathlib as pl
import typing as ty

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
DATABASE = "mc3d"
CALCULATION_TYPE = "single_shot"


# %%
def plot_yield(database, calculation_type):
    target_df = pd.DataFrame({k: v.tolist() for (k, v) in np.load(pl.Path(f"../data/{database}_targets.npz")).items()})

    feature_kwargs = {
        "dos_vbm_centered_gauss_0.05_-2.00_6.00_512": {
            "label": r"$E_{\mathrm{VBM}}(-2,+6) \mathrm{eV}$",
            "c": "#aa0000",
            "marker": ".",
        },
        "dos_fermi_scissor_gauss_0.05_-2.00_0.15_-0.15_2.00_512": {
            "label": r"$E_{F}^{\mathrm{scissor}} \pm 2 \mathrm{eV}$",
            "c": "#000080",
            "marker": ".",
        },
        "dos_fermi_centered_gauss_0.05_-5.00_5.00_512": {
            "label": r"$E_{F} \pm 5 \mathrm{eV}$",
            "c": "#008000",
            "marker": ".",
        },
        "dos_cbm_centered_gauss_0.05_-6.00_2.00_512": {
            "label": r"$E_{\mathrm{CBM}} (-6,+2) \mathrm{eV}$",
            "c": "k",
            "marker": ".",
        },
        # "dos_fermi_scissor_gauss_0.05_-2.00_2.00_-2.00_2.00_512": {
        #     "label": r"$E_{F}^{\mathrm{scissor}} \pm 2 \mathrm{eV}$ v2",
        #     "c": "tab:orange",
        #     "marker": ".",
        # },
    }

    with plt.style.context("../sorep.mplstyle"):
        fig, ax = plt.subplots(dpi=300)

        handles = []
        for feature_id, kwargs in feature_kwargs.items():
            metrics = pd.read_json(pl.Path(f"../data/{DATABASE}_metrics_{calculation_type}_{feature_id}.json"))
            gb = metrics.groupby(by="train_size")
            yield_mean = gb["yield"].mean()
            yield_std = gb["yield"].std()
            # dummy_yield_mean = gb["dummy_yield"].mean()
            # dummy_yield_std = gb["dummy_yield"].std()
            errorbar = ax.errorbar(yield_mean.index, yield_mean, yerr=yield_std, **kwargs, capsize=3)
            handles.append(errorbar)

        hline = ax.axhline(target_df.meets_tcm_criteria.mean(), c="grey", ls="--", label=r"$f_{\mathrm{TCM}}$")
        handles.append(hline)

        top_sec_ax = ax.secondary_xaxis(
            "top", functions=(lambda x: x * 0.9 * target_df.shape[0], lambda x: x / 0.9 / target_df.shape[0])
        )
        top_sec_ax.set_xlabel(r"$N_{\mathrm{train}}$")

        ax.tick_params(axis="x", which="both", top=False)

        ax.set_xscale("log")
        ax.set_xlabel(r"$f_{\mathrm{train}}$")
        ax.set_ylabel("Yield (TCM / calculation)")
        ax.legend(handles=handles, loc=[0.1, 0.1])

        # fig.tight_layout()
    return fig


plot_yield(DATABASE, CALCULATION_TYPE)


# %%
def plot_balanced_accuracy(database, calculation_type):
    target_df = pd.DataFrame({k: v.tolist() for (k, v) in np.load(pl.Path(f"../data/{database}_targets.npz")).items()})

    feature_kwargs = {
        "dos_vbm_centered_gauss_0.05_-2.00_6.00_512": {
            "label": r"$E_{\mathrm{VBM}}(-2,+6) \mathrm{eV}$",
            "c": "#aa0000",
            "marker": ".",
        },
        "dos_fermi_scissor_gauss_0.05_-2.00_0.15_-0.15_2.00_512": {
            "label": r"$E_{F}^{\mathrm{scissor}} \pm 2 \mathrm{eV}$",
            "c": "#000080",
            "marker": ".",
        },
        "dos_fermi_centered_gauss_0.05_-5.00_5.00_512": {
            "label": r"$E_{F} \pm 5 \mathrm{eV}$",
            "c": "#008000",
            "marker": ".",
        },
        "dos_cbm_centered_gauss_0.05_-6.00_2.00_512": {
            "label": r"$E_{\mathrm{CBM}} (-6,+2) \mathrm{eV}$",
            "c": "k",
            "marker": ".",
        },
        # "dos_fermi_scissor_gauss_0.05_-2.00_2.00_-2.00_2.00_512": {
        #     "label": r"$E_{F}^{\mathrm{scissor}} \pm 2 \mathrm{eV}$ v2",
        #     "c": "grey",
        #     "marker": ".",
        # },
    }

    with plt.style.context("../sorep.mplstyle"):
        fig, ax = plt.subplots(dpi=300)

        handles = []
        for feature_id, kwargs in feature_kwargs.items():
            metrics = pd.read_json(pl.Path(f"../data/{DATABASE}_metrics_{calculation_type}_{feature_id}.json"))
            gb = metrics.groupby(by="train_size")
            balanced_accuracy_mean = gb["balanced_accuracy"].mean()
            balanced_accuracy_std = gb["balanced_accuracy"].std()
            errorbar = ax.errorbar(
                balanced_accuracy_mean.index, balanced_accuracy_mean, yerr=balanced_accuracy_std, **kwargs, capsize=3
            )
            handles.append(errorbar)

        # hline = ax.axhline(target_df.meets_tcm_criteria.mean(), c="grey", ls="--", label=r"$f_{\mathrm{TCM}}$")
        # handles.append(hline)

        top_sec_ax = ax.secondary_xaxis(
            "top", functions=(lambda x: x * 0.9 * target_df.shape[0], lambda x: x / 0.9 / target_df.shape[0])
        )
        top_sec_ax.set_xlabel(r"$N_{\mathrm{train}}$")

        ax.tick_params(axis="x", which="both", top=False)

        ax.set_xscale("log")
        ax.set_xlabel(r"$f_{\mathrm{train}}$")
        ax.set_ylabel("Balanced accuracy")
        ax.legend(handles=handles, loc=[0.5, 0.1])

        # fig.tight_layout()
    return fig


plot_balanced_accuracy(DATABASE, CALCULATION_TYPE)


# %%
def plot_feature_importances(database, calculation_type, train_size=0.5, axes=None):
    feature_kwargs = {
        "dos_vbm_centered_gauss_0.05_-2.00_6.00_512": {
            "label": r"$E_{\mathrm{VBM}}(-2,+6) \mathrm{eV}$",
            "c": "#aa0000",
            "marker": ".",
        },
        "dos_fermi_scissor_gauss_0.05_-2.00_0.15_-0.15_2.00_512": {
            "label": r"$E_{F}^{\mathrm{scissor}} \pm 2 \mathrm{eV}$",
            "c": "#000080",
            "marker": ".",
        },
        "dos_fermi_centered_gauss_0.05_-5.00_5.00_512": {
            "label": r"$E_{F} \pm 5 \mathrm{eV}$",
            "c": "#008000",
            "marker": ".",
        },
        "dos_cbm_centered_gauss_0.05_-6.00_2.00_512": {
            "label": r"$E_{\mathrm{CBM}} (-6,+2) \mathrm{eV}$",
            "c": "k",
            "marker": ".",
        },
        # "dos_fermi_scissor_gauss_0.05_-2.00_2.00_-2.00_2.00_512": {
        #     "label": r"$E_{F}^{\mathrm{scissor}} \pm 2 \mathrm{eV}$ v2",
        #     "c": "grey",
        #     "marker": ".",
        # },
    }

    x = {
        "dos_vbm_centered_gauss_0.05_-2.00_6.00_512": np.linspace(-2, 6, 512),
        "dos_fermi_scissor_gauss_0.05_-2.00_0.15_-0.15_2.00_512": np.arange(512) - 256,
        "dos_fermi_centered_gauss_0.05_-5.00_5.00_512": np.linspace(-5, 5, 512),
        "dos_cbm_centered_gauss_0.05_-6.00_2.00_512": np.linspace(-6, 2, 512),
    }

    xticks = {
        "dos_vbm_centered_gauss_0.05_-2.00_6.00_512": {
            "xticks": [-2, 0, 0.35, 0.5, 0.65, +6],
            "xtick_labels": [
                r"$E_{\mathrm{VBM}} - 2 \mathrm{eV}$",
                r"$E_{\mathrm{VBM}}$",
                "",
                r"$E_{g}=0.5$",
                "",
                r"$E_{\mathrm{VBM}} + 6 \mathrm{eV}$",
            ],
        },
        "dos_fermi_scissor_gauss_0.05_-2.00_0.15_-0.15_2.00_512": {
            "xticks": [-256, -19, 0, 19, 255],
            "xtick_labels": [
                r"$E_{F} - 2 \mathrm{eV}$",
                r"$E_{\mathrm{VBM}}$",
                "",
                r"$E_{\mathrm{CBM}}$",
                r"$E_{F} + 2 \mathrm{eV}$",
            ],
        },
        "dos_fermi_centered_gauss_0.05_-5.00_5.00_512": {
            "xticks": [-5, -0.25, 0, +0.25, +5],
            "xtick_labels": [
                r"$E_{F} - 5 \mathrm{eV}$",
                r"",
                r"$E_{F}$",
                r"",
                r"$E_{F} + 5 \mathrm{eV}$",
            ],
        },
        "dos_cbm_centered_gauss_0.05_-6.00_2.00_512": {
            "xticks": [-6, -0.65, -0.5, -0.35, 0, +2],
            "xtick_labels": [
                r"$E_{\mathrm{CBM}} - 6 \mathrm{eV}$",
                "",
                r"$E_{g}=0.5$",
                "",
                r"$E_{\mathrm{CBM}}$",
                r"$E_{\mathrm{CBM}} + 2 \mathrm{eV}$",
            ],
        },
    }

    with plt.style.context("../sorep.mplstyle"):
        if axes is None:
            fig, axes = plt.subplots(2, 2, dpi=300, figsize=(9, 6), sharey=True)
            axes = axes.flatten()
        else:
            fig = plt.gcf()

        handles = []
        for ax, (feature_id, kwargs) in zip(axes, feature_kwargs.items()):
            metrics = pd.read_json(pl.Path(f"../data/{DATABASE}_metrics_{calculation_type}_{feature_id}.json"))
            gb = metrics[metrics.train_size == train_size]
            feature_importances_mean = np.array(gb["feature_importances"].to_list()).mean(axis=0)
            kwargs.pop("marker")
            errorbar = ax.plot(x[feature_id], feature_importances_mean, **kwargs)[0]
            handles.append(errorbar)

            ax.grid(True)
            ax.tick_params(axis="x", which="both", top=False)
            ax.tick_params(axis="x", which="minor", bottom=False)
            ax.set_xlabel("Feature")
            ax.set_xticks(xticks[feature_id]["xticks"], xticks[feature_id]["xtick_labels"], rotation=45)
            ax.set_ylabel("Importance")
            ax.set_yticklabels([])
            # ax.legend(handles=handles, loc=[0.05, 0.5])

    return fig


plot_feature_importances(DATABASE, CALCULATION_TYPE, train_size=0.5)
# %%
with open("../data/mc3d_features_single_shot_dos_vbm_centered_gauss_0.05_-2.00_6.00_512.npz", "rb") as fp:
    vbm_features = dict(np.load(fp))
    vbm_x = np.linspace(-2, 6, 512)

with open("../data/mc3d_features_single_shot_dos_cbm_centered_gauss_0.05_-6.00_2.00_512.npz", "rb") as fp:
    cbm_features = dict(np.load(fp))
    cbm_x = np.linspace(-6, 2, 512)

with open("../data/mc3d_features_single_shot_dos_fermi_centered_gauss_0.05_-5.00_5.00_512.npz", "rb") as fp:
    fermi_features = dict(np.load(fp))
    fermi_x = np.linspace(-5, 5, 512)
# %%
i = 0
fig, ax = plt.subplots()
ax.plot(vbm_x, vbm_features["features"][i], label="VBM")
# ax.plot(cbm_x, cbm_features["features"][i], label="CBM")
ax.plot(fermi_x, fermi_features["features"][i], label="FERMI")
ax.legend()
fig
# %%
